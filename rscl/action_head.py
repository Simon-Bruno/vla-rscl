import math
import random
from dataclasses import dataclass, field

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import PretrainedConfig
from transformers.feature_extraction_utils import BatchFeature

from gr00t.model.action_head.flow_matching_action_head import FlowmatchingActionHead

from .losses import rs_cl_loss, vanilla_infonce_loss


@dataclass
class RSCLConfig:
    # rs-cl hyperparameters
    contrastive_loss: str = "rscl"  # "none", "vanilla_infonce", "rscl"
    tau: float = 0.2
    beta: float = 1.0
    alpha: float = 0.5   # blend: alpha * d_proprio + (1-alpha) * d_depth  (set 1.0 to disable depth)
    gamma: float = 0.0   # action weight: (1-gamma)*[alpha*d_q+(1-alpha)*d_dep] + gamma*d_action (0=off)
    lambda_init: float = 1.0  # cosine decayed to 0
    proj_hidden: int = 2048
    proj_dim: int = 128
    n_views: int = 2
    tokens_per_view: int = 64  # groot n1.5 uses pixel shuffle -> 64 tokens/view


class ViewCutoff(nn.Module):
    # masks out one camera view's token slice as augmentation
    def __init__(self, n_views: int = 2, tokens_per_view: int = 64):
        super().__init__()
        self.n_views = n_views
        self.tokens_per_view = tokens_per_view

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        if not self.training:
            return features
        h = features.clone()
        view_idx = random.randint(0, self.n_views - 1)
        start = view_idx * self.tokens_per_view
        end = start + self.tokens_per_view
        h[:, start:end, :] = 0.0
        return h


class Projector(nn.Module):
    # 2-layer mlp: summary token -> contrastive space
    def __init__(self, d_model: int, hidden_dim: int = 2048, proj_dim: int = 128):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(d_model, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, proj_dim),
        )

    def forward(self, w: torch.Tensor) -> torch.Tensor:
        return self.net(w)


class FlowmatchingWithRSCL(FlowmatchingActionHead):
    # extends groot's action head with rs-cl contrastive regularization
    #
    # adds a contrastive path alongside flow matching:
    # 1. append learnable summary token to adapter (vl_self_attention) input
    # 2. extract summary token output -> projector -> z
    # 3. apply view cutoff on backbone features -> re-run adapter -> z_aug
    # 4. compute rs-cl loss on (z, z_aug, proprio)

    def __init__(self, config, rscl_config: RSCLConfig = None):
        super().__init__(config)

        if rscl_config is None:
            rscl_config = RSCLConfig()
        self.rscl_config = rscl_config

        d_model = config.backbone_embedding_dim  # 1536 for groot n1.5

        self.summary_token = nn.Parameter(torch.randn(1, 1, d_model) * 0.02)
        self.projector = Projector(d_model, rscl_config.proj_hidden, rscl_config.proj_dim)
        self.view_cutoff = ViewCutoff(rscl_config.n_views, rscl_config.tokens_per_view)

    def _adapt_with_summary(self, backbone_features: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        # run features + summary token through the adapter (vlln + vl_self_attention)
        # returns h (adapted features for dit) and w (summary token output for projection)
        B = backbone_features.shape[0]
        summary = self.summary_token.expand(B, -1, -1)

        features_with_summary = torch.cat([backbone_features, summary], dim=1)
        features_with_summary = self.vlln(features_with_summary)
        adapted = self.vl_self_attention(features_with_summary)

        h = adapted[:, :-1, :]
        w = adapted[:, -1, :]
        return h, w

    def process_backbone_output(self, backbone_output: BatchFeature) -> BatchFeature:
        # override to route through summary token and compute contrastive embeddings
        raw_features = backbone_output["backbone_features"]

        if self.rscl_config.contrastive_loss == "none":
            return super().process_backbone_output(backbone_output)

        # main path
        h, w = self._adapt_with_summary(raw_features)
        z = self.projector(w)

        # augmented path (view cutoff -> re-run adapter, no grad to save memory)
        augmented = self.view_cutoff(raw_features.detach())
        with torch.no_grad():
            _, w_aug = self._adapt_with_summary(augmented)
        z_aug = self.projector(w_aug)

        backbone_output["backbone_features"] = h
        backbone_output["_rscl_z"] = z
        backbone_output["_rscl_z_aug"] = z_aug

        return backbone_output

    def forward(self, backbone_output: BatchFeature, action_input: BatchFeature) -> BatchFeature:
        # flow matching + rs-cl loss
        self.set_frozen_modules_to_eval_mode()
        backbone_output = self.process_backbone_output(backbone_output)

        if self.config.expand_batch is not None:
            for k, v in backbone_output.items():
                if isinstance(v, torch.Tensor):
                    ndim = len(v.shape)
                    factors = tuple([self.config.expand_batch] + [1] * (ndim - 1))
                    backbone_output[k] = v.repeat(*factors)
            for k, v in action_input.items():
                if isinstance(v, torch.Tensor):
                    ndim = len(v.shape)
                    factors = tuple([self.config.expand_batch] + [1] * (ndim - 1))
                    action_input[k] = v.repeat(*factors)

        vl_embs = backbone_output.backbone_features
        device = vl_embs.device
        embodiment_id = action_input.embodiment_id

        state_features = self.state_encoder(action_input.state, embodiment_id)

        actions = action_input.action
        noise = torch.randn(actions.shape, device=actions.device, dtype=actions.dtype)
        t = self.sample_time(actions.shape[0], device=actions.device, dtype=actions.dtype)
        t = t[:, None, None]

        noisy_trajectory = (1 - t) * noise + t * actions
        velocity = actions - noise

        t_discretized = (t[:, 0, 0] * self.num_timestep_buckets).long()
        action_features = self.action_encoder(noisy_trajectory, t_discretized, embodiment_id)

        if self.config.add_pos_embed:
            pos_ids = torch.arange(action_features.shape[1], dtype=torch.long, device=device)
            pos_embs = self.position_embedding(pos_ids).unsqueeze(0)
            action_features = action_features + pos_embs

        future_tokens = self.future_tokens.weight.unsqueeze(0).expand(vl_embs.shape[0], -1, -1)
        sa_embs = torch.cat((state_features, future_tokens, action_features), dim=1)

        vl_attn_mask = backbone_output.backbone_attention_mask

        model_output = self.model(
            hidden_states=sa_embs,
            encoder_hidden_states=vl_embs,
            encoder_attention_mask=vl_attn_mask,
            timestep=t_discretized,
            return_all_hidden_states=False,
        )
        pred = self.action_decoder(model_output, embodiment_id)
        pred_actions = pred[:, -actions.shape[1]:]

        action_mask = action_input.action_mask
        fm_loss = F.mse_loss(pred_actions, velocity, reduction="none") * action_mask
        fm_loss = fm_loss.sum() / action_mask.sum()

        # contrastive loss
        total_loss = fm_loss
        cl_loss_val = torch.tensor(0.0, device=device)

        if self.rscl_config.contrastive_loss != "none" and "_rscl_z" in backbone_output:
            z = backbone_output["_rscl_z"]
            z_aug = backbone_output["_rscl_z_aug"]
            proprio = action_input.state[:, 0, :]  # first timestep, (B, 64)
            depth = action_input.get("depth_map", None)   # (B, 64) or None, disabled by alpha=1.0
            # mean over 16-step horizon -> (B, 32); disabled when gamma=0.0
            action_mean = actions.mean(dim=1) if self.rscl_config.gamma > 0.0 else None

            if self.rscl_config.contrastive_loss == "vanilla_infonce":
                cl_loss_val = vanilla_infonce_loss(z, z_aug, tau=self.rscl_config.tau)
            else:
                cl_loss_val = rs_cl_loss(
                    z, z_aug, proprio,
                    tau=self.rscl_config.tau,
                    beta=self.rscl_config.beta,
                    depth=depth,
                    alpha=self.rscl_config.alpha,
                    action=action_mean,
                    gamma=self.rscl_config.gamma,
                )

            lam = getattr(self, "_current_lambda", self.rscl_config.lambda_init)
            total_loss = fm_loss + lam * cl_loss_val

        output_dict = {
            "loss": total_loss,
            "fm_loss": fm_loss.detach(),
            "cl_loss": cl_loss_val.detach(),
        }
        return BatchFeature(data=output_dict)

    @property
    def device(self):
        try:
            return next(iter(self.parameters())).device
        except StopIteration:
            return torch.device("cuda" if torch.cuda.is_available() else "cpu")

    @property
    def dtype(self):
        try:
            return next(iter(self.parameters())).dtype
        except StopIteration:
            dtype_str = getattr(self.config, "model_dtype", "float32")
            return getattr(torch, dtype_str, torch.float32)

