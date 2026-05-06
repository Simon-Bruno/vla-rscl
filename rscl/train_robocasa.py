# training script: groot n1.5 fine-tuning with rs-cl contrastive loss
#
# wraps isaac-groot's fine-tuning pipeline and swaps in our
# extended action head. usage:
#   python -m rscl.train --dataset_path <path> --contrastive_loss rscl

import math
import os
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import List, Literal

import cv2
import numpy as np
import torch
import torch.nn.functional as F
import tyro
from transformers import TrainingArguments

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "Isaac-GR00T"))

from gr00t.data.dataset import LeRobotSingleDataset
from gr00t.experiment.data_config import load_data_config
from gr00t.experiment.runner import TrainRunner
from gr00t.experiment.trainer import DualBrainTrainer
from gr00t.model.gr00t_n1 import GR00T_N1_5
from gr00t.model.transforms import EMBODIMENT_TAG_MAPPING

from .action_head import FlowmatchingWithRSCL, RSCLConfig


@dataclass
class ArgsConfig:
    # dataset
    dataset_path: List[str] = None
    output_dir: str = "./outputs"
    data_config: str = "examples.Libero.custom_data_config:LiberoDataConfig"

    # model
    base_model_path: str = "nvidia/GR00T-N1.5-3B"
    tune_llm: bool = False
    tune_visual: bool = False
    tune_projector: bool = True
    tune_diffusion_model: bool = True

    # training
    batch_size: int = 32
    max_steps: int = 60000
    learning_rate: float = 1e-4
    weight_decay: float = 1e-5
    warmup_ratio: float = 0.05
    save_steps: int = 5000
    save_total_limit: int = 2
    gradient_accumulation_steps: int = 1
    dataloader_num_workers: int = 12
    report_to: str = "wandb"

    # rs-cl
    contrastive_loss: str = "rscl"  # "none", "vanilla_infonce", "rscl"
    tau: float = 0.2
    beta: float = 1.0
    alpha: float = 0.5   # depth blend: alpha*d_q + (1-alpha)*d_depth  (1.0 = depth off)
    gamma: float = 0.0   # action weight  (0.0 = action off)
    lambda_init: float = 1.0
    proj_hidden: int = 2048
    proj_dim: int = 128

    # data
    embodiment_tag: str = "new_embodiment"
    video_backend: str = "torchcodec"
    max_demos_total: int = None  # limit total demos across all tasks (e.g. 300 to match paper)

    seed: int = 0
    resume_from_checkpoint: str = None  # path to checkpoint dir to resume from


class RSCLTrainer(DualBrainTrainer):
    # updates cosine-decayed lambda on the action head each step

    def __init__(self, rscl_config: RSCLConfig, total_steps: int, **kwargs):
        self.rscl_config = rscl_config
        self.total_steps = total_steps
        super().__init__(**kwargs)

    def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):
        step = self.state.global_step
        lam = self.rscl_config.lambda_init * 0.5 * (1 + math.cos(math.pi * step / self.total_steps))
        
        # Handle DataParallel/DistributedDataParallel wrapping
        base_model = model.module if hasattr(model, "module") else model
        base_model.action_head._current_lambda = lam

        outputs = model(inputs)
        loss = outputs["loss"]

        # log contrastive metrics to wandb
        fm = outputs.get("fm_loss", torch.tensor(0.0))
        cl = outputs.get("cl_loss", torch.tensor(0.0))
        self.log({"fm_loss": fm.item(), "cl_loss": cl.item(), "lambda": lam})

        if step % 500 == 0:
            print(f"[step {step}] fm={fm.item():.4f}  cl={cl.item():.4f}  lam={lam:.4f}")

        return (loss, outputs) if return_outputs else loss


def main():
    config = tyro.cli(ArgsConfig)
    torch.manual_seed(config.seed)
    os.environ.setdefault("WANDB_PROJECT", "rscl")

    rscl_config = RSCLConfig(
        contrastive_loss=config.contrastive_loss,
        tau=config.tau,
        beta=config.beta,
        alpha=config.alpha,
        gamma=config.gamma,
        lambda_init=config.lambda_init,
        proj_hidden=config.proj_hidden,
        proj_dim=config.proj_dim,
    )

    # load data config (libero-specific transforms and normalization)
    data_config = load_data_config(config.data_config)

    # wrapper that catches corrupt video errors and retries with a random sample
    class RobustDataset(torch.utils.data.Dataset):
        skip_count = 0
        total_count = 0
        def __init__(self, dataset):
            self.dataset = dataset
        def __len__(self):
            return len(self.dataset)
        def __getitem__(self, idx):
            import random
            RobustDataset.total_count += 1
            for attempt in range(10):
                try:
                    return self.dataset[idx]
                except Exception as e:
                    RobustDataset.skip_count += 1
                    if RobustDataset.skip_count % 100 == 1:
                        pct = 100 * RobustDataset.skip_count / max(RobustDataset.total_count, 1)
                        print(f"[warning] skipped {RobustDataset.skip_count} corrupt samples ({pct:.1f}% of total)")
                    idx = random.randint(0, len(self.dataset) - 1)
            return self.dataset[0]
        def __getattr__(self, name):
            return getattr(self.dataset, name)

    # create datasets — for multiple paths, build a mixture
    from gr00t.data.dataset import LeRobotMixtureDataset

    class DepthAugmentedDataset(torch.utils.data.Dataset):
        """wraps a LeRobotSingleDataset and adds depth_map (64-d pooled float32) to each item.
        depth PNGs are loaded from {dataset_path}/depth_maps/episode_XXXXXX_NNNNNN.png"""
        def __init__(self, dataset, dataset_path: Path):
            self._dataset = dataset
            self._depth_root = Path(dataset_path) / "depth_maps"

        def __len__(self):
            return len(self._dataset)

        def __getitem__(self, idx):
            item = self._dataset[idx]
            trajectory_id, base_index = self._dataset.all_steps[idx]
            depth_path = self._depth_root / f"episode_{trajectory_id:06d}_{base_index:06d}.png"
            if depth_path.exists():
                gray = cv2.imread(str(depth_path), cv2.IMREAD_GRAYSCALE)  # (H, W) uint8
                depth_t = torch.from_numpy(gray).float() / 255.0          # (H, W)
                depth_t = F.adaptive_avg_pool2d(depth_t.unsqueeze(0).unsqueeze(0), (8, 8))
                item["depth_map"] = depth_t.squeeze().flatten()            # (64,)
            else:
                item["depth_map"] = torch.zeros(64)
            return item

        def __getattr__(self, name):
            return getattr(self._dataset, name)

    datasets = []
    for path in config.dataset_path:
        ds = LeRobotSingleDataset(
            dataset_path=path,
            modality_configs=data_config.modality_config(),
            transforms=data_config.transform(),
            embodiment_tag=config.embodiment_tag,
            video_backend=config.video_backend,
        )
        ds = DepthAugmentedDataset(ds, path)
        datasets.append(ds)

    # subsample to match paper's demo budget (e.g. 300 total across all tasks)
    # uses an episode-level wrapper that keeps trajectory_lengths correct so
    # LeRobotMixtureDataset can access it (plain Subset doesn't proxy attributes)
    if config.max_demos_total is not None:
        import numpy as np

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

        num_tasks = len(datasets)
        demos_per_task = config.max_demos_total // num_tasks
        datasets = [EpisodeLimitedDataset(ds, demos_per_task) for ds in datasets]
        actual = sum(len(ds.trajectory_lengths) for ds in datasets)
        print(f"[demo limit] {demos_per_task} episodes/task, {actual} total episodes")

    if len(datasets) == 1:
        dataset = RobustDataset(datasets[0])
    else:
        data_mixture = [(ds, 1.0) for ds in datasets]
        dataset = RobustDataset(LeRobotMixtureDataset(data_mixture=data_mixture, mode="train"))

    # load pretrained groot n1.5
    model = GR00T_N1_5.from_pretrained(
        config.base_model_path,
        tune_visual=config.tune_visual,
        tune_llm=config.tune_llm,
        tune_projector=config.tune_projector,
        tune_diffusion_model=config.tune_diffusion_model,
    )

    # swap action head with rs-cl version
    if config.contrastive_loss != "none":
        original_head = model.action_head
        rscl_head = FlowmatchingWithRSCL(original_head.config, rscl_config)

        # copy pretrained weights
        rscl_head.load_state_dict(original_head.state_dict(), strict=False)
        model.action_head = rscl_head

        for name, p in model.action_head.named_parameters():
            if "summary_token" in name or "projector" in name or "view_cutoff" in name:
                p.requires_grad = True

        n_rscl = sum(p.numel() for p in [rscl_head.summary_token] + list(rscl_head.projector.parameters()))
        print(f"rs-cl added {n_rscl:,} trainable params (summary token + projector)")

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
        run_name=f"{config.contrastive_loss}-{config.data_config.split(':')[-1] if ':' in config.data_config else config.data_config}-s{config.seed}",
        seed=config.seed,
        remove_unused_columns=False,
    )

    from gr00t.model.transforms import DefaultDataCollator
    compute_dtype = torch.bfloat16 if training_args.bf16 else torch.float32
    trainer = RSCLTrainer(
        rscl_config=rscl_config,
        total_steps=config.max_steps,
        model=model,
        args=training_args,
        train_dataset=dataset,
        data_collator=DefaultDataCollator(),
        compute_dtype=compute_dtype,
    )

    # save experiment_cfg/metadata.json for inference service
    import json as _json
    exp_cfg_dir = Path(config.output_dir) / "experiment_cfg"
    exp_cfg_dir.mkdir(parents=True, exist_ok=True)
    metadata_json = {}
    inner_ds = dataset.dataset if hasattr(dataset, "dataset") else dataset
    if isinstance(inner_ds, LeRobotSingleDataset):
        metadata_json[inner_ds.tag] = inner_ds.metadata.model_dump(mode="json")
    elif isinstance(inner_ds, LeRobotMixtureDataset):
        metadata_json.update(
            {tag: meta.model_dump(mode="json") for tag, meta in inner_ds.merged_metadata.items()}
        )
    with open(exp_cfg_dir / "metadata.json", "w") as f:
        _json.dump(metadata_json, f, indent=4)

    trainer.train(resume_from_checkpoint=config.resume_from_checkpoint)
    trainer.save_model(config.output_dir, _internal_call=True)


if __name__ == "__main__":
    main()
