# GRAM training script for RoboCasa/GR00T.
#
# This leaves train_robocasa.py untouched. Run with:
#   python -m rscl.train_robocasa_GRAM --dataset_path <task dirs...>

import math
import os
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import List

import cv2
import torch
import torch.nn.functional as F
import tyro
from transformers import TrainingArguments

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "Isaac-GR00T"))

from gr00t.data.dataset import LeRobotSingleDataset
from gr00t.experiment.data_config import load_data_config
from gr00t.experiment.trainer import DualBrainTrainer
from gr00t.model.gr00t_n1 import GR00T_N1_5

from .action_head_GRAM import FlowmatchingWithRSCLGRAM, RSCLGRAMConfig


@dataclass
class ArgsConfig:
    # dataset
    dataset_path: List[str] = None
    output_dir: str = "./outputs/robocasa_gram"
    data_config: str = "fourier_gr1_arms_waist"

    # model
    base_model_path: str = "nvidia/GR00T-N1.5-3B"
    tune_llm: bool = False
    tune_visual: bool = True
    tune_projector: bool = True
    tune_diffusion_model: bool = True

    # training
    batch_size: int = 8
    max_steps: int = 30000
    learning_rate: float = 1e-4
    weight_decay: float = 1e-5
    warmup_ratio: float = 0.05
    save_steps: int = 5000
    save_total_limit: int = 2
    gradient_accumulation_steps: int = 8
    dataloader_num_workers: int = 12
    report_to: str = "wandb"

    # rs-cl + gram
    contrastive_loss: str = "rscl_gram"
    tau: float = 0.2
    beta: float = 0.3
    w_q: float = 1.0      # proprio distance weight (0.0 = off)
    w_depth: float = 1.0  # depth distance weight   (0.0 = off)
    w_vel: float = 1.0    # state velocity distance weight (0.0 = off)
    lambda_init: float = 1.0
    proj_hidden: int = 2048
    proj_dim: int = 128
    modality_hidden: int = 256

    # data
    embodiment_tag: str = "gr1"
    video_backend: str = "torchvision_av"
    max_demos_per_task: int = None
    seed: int = 0
    resume_from_checkpoint: str = None


class GRAMTrainer(DualBrainTrainer):
    # updates cosine-decayed lambda on the action head each step

    def __init__(self, gram_config: RSCLGRAMConfig, total_steps: int, **kwargs):
        self.gram_config = gram_config
        self.total_steps = total_steps
        super().__init__(**kwargs)

    def compute_loss(
        self, model, inputs, return_outputs=False, num_items_in_batch=None
    ):
        step = self.state.global_step
        lam = (
            self.gram_config.lambda_init
            * 0.5
            * (1 + math.cos(math.pi * step / self.total_steps))
        )
        base_model = model.module if hasattr(model, "module") else model
        base_model.action_head._current_lambda = lam

        outputs = model(inputs)
        loss = outputs["loss"]

        # log contrastive metrics to wandb
        logs = {
            "fm_loss": outputs.get("fm_loss", torch.tensor(0.0)).item(),
            "gram_loss": outputs.get("gram_loss", torch.tensor(0.0)).item(),
            "lambda": lam,
            "h_proprio_corr": outputs.get("h_proprio_corr", torch.tensor(0.0)).item(),
            "gram_pos_volume": outputs.get("gram_pos_volume", torch.tensor(0.0)).item(),
            "gram_neg_volume": outputs.get("gram_neg_volume", torch.tensor(0.0)).item(),
            "gram_pos_neg_gap": outputs.get(
                "gram_pos_neg_gap", torch.tensor(0.0)
            ).item(),
            "rscl_target_entropy": outputs.get(
                "rscl_target_entropy", torch.tensor(0.0)
            ).item(),
        }
        self.log(logs)

        if step % 500 == 0:
            print(
                f"[step {step}] fm={logs['fm_loss']:.4f} gram={logs['gram_loss']:.4f} "
                f"lam={lam:.4f} pos_vol={logs['gram_pos_volume']:.4f} "
                f"neg_vol={logs['gram_neg_volume']:.4f} gap={logs['gram_pos_neg_gap']:.4f}",
                flush=True,
            )

        return (loss, outputs) if return_outputs else loss


class DepthAugmentedDataset(torch.utils.data.Dataset):
    """wraps a LeRobotSingleDataset and adds depth_map (64-d pooled float32) to each item.
    depth PNGs are loaded from {dataset_path}/depth_maps/episode_XXXXXX_NNNNNN.png"""

    def __init__(self, dataset, dataset_path: str, add_depth: bool):
        self._dataset = dataset
        self._depth_root = Path(dataset_path) / "depth_maps"
        self._add_depth = add_depth

    def __len__(self):
        return len(self._dataset)

    def __getitem__(self, idx):
        item = self._dataset[idx]
        if self._add_depth:
            trajectory_id, base_index = self._dataset.all_steps[idx]
            depth_path = (
                self._depth_root / f"episode_{trajectory_id:06d}_{base_index:06d}.png"
            )
            if depth_path.exists():
                gray = cv2.imread(str(depth_path), cv2.IMREAD_GRAYSCALE)
                depth_t = torch.from_numpy(gray).float() / 255.0
                depth_t = F.adaptive_avg_pool2d(
                    depth_t.unsqueeze(0).unsqueeze(0), (8, 8)
                )
                item["depth_map"] = depth_t.squeeze().flatten()
            else:
                item["depth_map"] = torch.zeros(64)
        return item

    def __getattr__(self, name):
        return getattr(self._dataset, name)


class EpisodeLimitedDataset(torch.utils.data.Dataset):
    """takes first n_episodes from a LeRobotSingleDataset, exposing correct trajectory_lengths."""

    def __init__(self, dataset, n_episodes: int):
        self._dataset = dataset
        n = min(n_episodes, len(dataset.trajectory_lengths))
        self.trajectory_lengths = dataset.trajectory_lengths[:n]
        self.trajectory_ids = dataset.trajectory_ids[:n]
        self._total_steps = int(self.trajectory_lengths.sum())

    def __len__(self):
        return self._total_steps

    def __getitem__(self, idx):
        return self._dataset[idx]

    def __getattr__(self, name):
        return getattr(self._dataset, name)


class RobustDataset(torch.utils.data.Dataset):
    # wrapper that catches corrupt video errors and retries with a random sample
    skip_count = 0
    total_count = 0

    def __init__(self, dataset):
        self.dataset = dataset

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        import random

        RobustDataset.total_count += 1
        for _ in range(10):
            try:
                return self.dataset[idx]
            except Exception:
                RobustDataset.skip_count += 1
                if RobustDataset.skip_count % 100 == 1:
                    pct = (
                        100
                        * RobustDataset.skip_count
                        / max(RobustDataset.total_count, 1)
                    )
                    print(
                        f"[warning] skipped {RobustDataset.skip_count} corrupt samples ({pct:.1f}% of total)"
                    )
                idx = random.randint(0, len(self.dataset) - 1)
        return self.dataset[0]

    def __getattr__(self, name):
        return getattr(self.dataset, name)


def _collect_metadata(dataset):
    if isinstance(dataset, LeRobotSingleDataset):
        return {dataset.tag: dataset.metadata.model_dump(mode="json")}
    if hasattr(dataset, "datasets"):
        metadata = {}
        for child in dataset.datasets:
            metadata.update(_collect_metadata(child))
        return metadata
    if hasattr(dataset, "dataset"):
        return _collect_metadata(dataset.dataset)
    if hasattr(dataset, "_dataset"):
        return _collect_metadata(dataset._dataset)
    return {}


def main():
    config = tyro.cli(ArgsConfig)
    torch.manual_seed(config.seed)
    os.environ.setdefault("WANDB_PROJECT", "rscl-gram")

    gram_config = RSCLGRAMConfig(
        contrastive_loss=config.contrastive_loss,
        tau=config.tau,
        beta=config.beta,
        w_q=config.w_q,
        w_depth=config.w_depth,
        w_vel=config.w_vel,
        lambda_init=config.lambda_init,
        proj_hidden=config.proj_hidden,
        proj_dim=config.proj_dim,
        modality_hidden=config.modality_hidden,
    )

    # load data config (robocasa-specific transforms and normalization)
    data_config = load_data_config(config.data_config)

    # create datasets — for multiple paths, build a concatenation
    datasets = []
    for path in config.dataset_path:
        ds = LeRobotSingleDataset(
            dataset_path=path,
            modality_configs=data_config.modality_config(),
            transforms=data_config.transform(),
            embodiment_tag=config.embodiment_tag,
            video_backend=config.video_backend,
        )
        ds = DepthAugmentedDataset(ds, path, add_depth=(config.w_depth > 0.0))
        if config.max_demos_per_task is not None:
            ds = EpisodeLimitedDataset(ds, config.max_demos_per_task)
        datasets.append(ds)

    # subsample to match paper's demo budget — paper reports 30/100/300 demos *per task*
    if config.max_demos_per_task is not None:
        actual = sum(len(ds.trajectory_lengths) for ds in datasets)
        print(
            f"[demo limit] {config.max_demos_per_task} episodes/task, {actual} total episodes"
        )

    if len(datasets) == 1:
        dataset = RobustDataset(datasets[0])
    else:
        dataset = RobustDataset(torch.utils.data.ConcatDataset(datasets))

    # load pretrained groot n1.5
    model = GR00T_N1_5.from_pretrained(
        config.base_model_path,
        tune_visual=config.tune_visual,
        tune_llm=config.tune_llm,
        tune_projector=config.tune_projector,
        tune_diffusion_model=config.tune_diffusion_model,
    )

    # swap action head with gram version
    if config.contrastive_loss != "none":
        original_head = model.action_head
        gram_head = FlowmatchingWithRSCLGRAM(original_head.config, gram_config)

        # copy pretrained weights
        gram_head.load_state_dict(original_head.state_dict(), strict=False)
        model.action_head = gram_head

        for name, param in model.action_head.named_parameters():
            if "summary_token" in name or "projector" in name:
                param.requires_grad = True

        n_added = sum(
            p.numel()
            for name, p in gram_head.named_parameters()
            if "summary_token" in name or "projector" in name
        )
        print(f"GRAM added {n_added:,} trainable params (summary token + projectors)")

    training_args = TrainingArguments(
        output_dir=config.output_dir,
        per_device_train_batch_size=config.batch_size,
        max_steps=config.max_steps,
        learning_rate=config.learning_rate,
        weight_decay=config.weight_decay,
        warmup_ratio=config.warmup_ratio,
        lr_scheduler_type="cosine",
        save_steps=config.save_steps,
        save_total_limit=config.save_total_limit,
        logging_steps=50,
        gradient_accumulation_steps=config.gradient_accumulation_steps,
        dataloader_num_workers=config.dataloader_num_workers,
        bf16=True,
        report_to=config.report_to,
        run_name=f"rscl-gram-{config.data_config}-s{config.seed}",
        seed=config.seed,
        remove_unused_columns=False,
    )

    from gr00t.model.transforms import DefaultDataCollator

    class GRAMDataCollator(DefaultDataCollator):
        # DefaultDataCollator uses torch.from_numpy(np.stack(values)) for all non-special keys.
        # that works for numpy arrays (state, action) but is fragile for torch.Tensor values
        # like depth_map. extract tensor keys first, stack them cleanly, then let the parent
        # handle the remaining numpy-array keys.
        def __call__(self, features):
            tensor_keys = {
                k: torch.stack([f[k] for f in features])
                for k in features[0]
                if isinstance(features[0][k], torch.Tensor)
            }
            stripped = [
                {k: v for k, v in f.items() if k not in tensor_keys} for f in features
            ]
            batch = super().__call__(stripped)
            batch.update(tensor_keys)
            return batch

    compute_dtype = torch.bfloat16 if training_args.bf16 else torch.float32
    trainer = GRAMTrainer(
        gram_config=gram_config,
        total_steps=config.max_steps,
        model=model,
        args=training_args,
        train_dataset=dataset,
        data_collator=GRAMDataCollator(),
        compute_dtype=compute_dtype,
    )

    # save experiment_cfg/metadata.json for inference service
    import json as _json

    exp_cfg_dir = Path(config.output_dir) / "experiment_cfg"
    exp_cfg_dir.mkdir(parents=True, exist_ok=True)
    with open(exp_cfg_dir / "metadata.json", "w") as f:
        _json.dump(_collect_metadata(dataset), f, indent=4)

    trainer.train(resume_from_checkpoint=config.resume_from_checkpoint)
    trainer.save_model(config.output_dir, _internal_call=True)


if __name__ == "__main__":
    main()
