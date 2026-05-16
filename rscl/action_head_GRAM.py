import math
from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers.feature_extraction_utils import BatchFeature

from gr00t.model.action_head.flow_matching_action_head import FlowmatchingActionHead

from .losses_GRAM import rscl_gram_loss


@dataclass
class RSCLGRAMConfig:
    contrastive_loss: str = "rscl_gram"  # "none", "rscl_gram"
    tau: float = 0.2
    beta: float = 1.0
    w_q: float = 1.0
    w_depth: float = 1.0
    w_vel: float = 1.0
    lambda_init: float = 1.0
    proj_hidden: int = 2048
    proj_dim: int = 128
    modality_hidden: int = 256


class Projector(nn.Module):
    def __init__(self, d_model: int, hidden_dim: int = 2048, proj_dim: int = 128):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(d_model, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, proj_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class FlowmatchingWithRSCLGRAM(FlowmatchingActionHead):
    """GR00T flow-matching head with RSCL-soft GRAM contrastive alignment.

    Unlike the original RSCL action head, this path does not build z_aug with
    view cutoff. GRAM compares visual-language z_i against candidate robot
    modalities from sample j and uses negative volume as the contrastive logit.
    """

    def __init__(self, config, gram_config: RSCLGRAMConfig | None = None):
        super().__init__(config)
        self.gram_config = gram_config or RSCLGRAMConfig()

        d_model = config.backbone_embedding_dim
        self.summary_token = nn.Parameter(torch.randn(1, 1, d_model) * 0.02)
        self.projector = Projector(d_model, self.gram_config.proj_hidden, self.gram_config.proj_dim)

        self.proprio_projector = Projector(64, self.gram_config.modality_hidden, self.gram_config.proj_dim)
        if self.gram_config.w_depth > 0.0:
            self.depth_projector = Projector(64, self.gram_config.modality_hidden, self.gram_config.proj_dim)
        if self.gram_config.w_vel > 0.0:
            self.velocity_projector = Projector(64, self.gram_config.modality_hidden, self.gram_config.proj_dim)

    def _adapt_with_summary(self, backbone_features: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        bsz = backbone_features.shape[0]
        summary = self.summary_token.expand(bsz, -1, -1)
        features = torch.cat([backbone_features, summary], dim=1)
        features = self.vlln(features)
        adapted = self.vl_self_attention(features)
        return adapted[:, :-1, :], adapted[:, -1, :]

    def process_backbone_output(self, backbone_output: BatchFeature) -> BatchFeature:
        if self.gram_config.contrastive_loss == "none":
            return super().process_backbone_output(backbone_output)

        h, w = self._adapt_with_summary(backbone_output["backbone_features"])
        backbone_output["backbone_features"] = h
        backbone_output["_gram_z"] = self.projector(w)
        backbone_output["_gram_w"] = w
        return backbone_output

    @staticmethod
    def _state_velocity(state: torch.Tensor) -> torch.Tensor:
        if state.ndim >= 3 and state.shape[1] > 1:
            return state[:, 1, :] - state[:, 0, :]
        return torch.zeros_like(state[:, 0, :] if state.ndim >= 3 else state)

    def forward(self, backbone_output: BatchFeature, action_input: BatchFeature) -> BatchFeature:
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
        velocity_target = actions - noise

        t_discretized = (t[:, 0, 0] * self.num_timestep_buckets).long()
        action_features = self.action_encoder(noisy_trajectory, t_discretized, embodiment_id)

        if self.config.add_pos_embed:
            pos_ids = torch.arange(action_features.shape[1], dtype=torch.long, device=device)
            action_features = action_features + self.position_embedding(pos_ids).unsqueeze(0)

        future_tokens = self.future_tokens.weight.unsqueeze(0).expand(vl_embs.shape[0], -1, -1)
        sa_embs = torch.cat((state_features, future_tokens, action_features), dim=1)

        model_output = self.model(
            hidden_states=sa_embs,
            encoder_hidden_states=vl_embs,
            encoder_attention_mask=backbone_output.backbone_attention_mask,
            timestep=t_discretized,
            return_all_hidden_states=False,
        )
        pred = self.action_decoder(model_output, embodiment_id)
        pred_actions = pred[:, -actions.shape[1]:]

        action_mask = action_input.action_mask
        fm_loss = F.mse_loss(pred_actions, velocity_target, reduction="none") * action_mask
        fm_loss = fm_loss.sum() / action_mask.sum()

        total_loss = fm_loss
        gram_loss = torch.zeros((), device=device)
        gram_metrics = {
            "gram_pos_volume": torch.zeros((), device=device),
            "gram_neg_volume": torch.zeros((), device=device),
            "gram_pos_neg_gap": torch.zeros((), device=device),
            "rscl_target_entropy": torch.zeros((), device=device),
        }

        if self.gram_config.contrastive_loss != "none" and "_gram_z" in backbone_output:
            z = backbone_output["_gram_z"]
            proprio = action_input.state[:, 0, :]
            velocity = self._state_velocity(action_input.state)
            depth = action_input.get("depth_map", None)

            proprio_embed = self.proprio_projector(proprio)
            depth_embed = None
            if depth is not None and hasattr(self, "depth_projector"):
                depth_embed = self.depth_projector(depth.to(dtype=proprio.dtype, device=proprio.device))

            velocity_embed = None
            if hasattr(self, "velocity_projector"):
                velocity_embed = self.velocity_projector(velocity)

            gram_loss, gram_metrics = rscl_gram_loss(
                z,
                proprio_embed,
                proprio,
                depth_embed=depth_embed,
                depth=depth,
                velocity_embed=velocity_embed,
                velocity=velocity,
                tau=self.gram_config.tau,
                beta=self.gram_config.beta,
                w_q=self.gram_config.w_q,
                w_depth=self.gram_config.w_depth if depth is not None else 0.0,
                w_vel=self.gram_config.w_vel,
            )
            lam = getattr(self, "_current_lambda", self.gram_config.lambda_init)
            total_loss = fm_loss + lam * gram_loss

        h_proprio_corr = torch.zeros((), device=device)
        if "_gram_w" in backbone_output:
            with torch.no_grad():
                w_rep = backbone_output["_gram_w"]
                q = action_input.state[:, 0, :]
                if w_rep.shape[0] > 4:
                    dist_w = torch.cdist(w_rep.float(), w_rep.float(), p=2).flatten()
                    dist_q = torch.cdist(q.float(), q.float(), p=2).flatten()
                    dist_w = dist_w - dist_w.mean()
                    dist_q = dist_q - dist_q.mean()
                    denom = dist_w.norm() * dist_q.norm()
                    h_proprio_corr = (dist_w * dist_q).sum() / denom.clamp(min=1.0e-8)

        output_dict = {
            "loss": total_loss,
            "fm_loss": fm_loss.detach(),
            "gram_loss": gram_loss.detach(),
            "h_proprio_corr": h_proprio_corr.detach(),
        }
        output_dict.update({k: v.detach() for k, v in gram_metrics.items()})
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
