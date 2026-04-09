import torch
import torch.nn.functional as F


def rs_cl_weights(proprio: torch.Tensor, beta: float = 1.0) -> torch.Tensor:
    # soft weights from proprioceptive state distances
    dists = torch.cdist(proprio, proprio, p=2)
    return torch.softmax(-dists / beta, dim=1)


def rs_cl_loss(
    z: torch.Tensor,
    z_aug: torch.Tensor,
    proprio: torch.Tensor,
    tau: float = 0.2,
    beta: float = 1.0,
) -> torch.Tensor:
    # infonce with proprioceptive soft weights (eq. 3 in paper)
    z = F.normalize(z, dim=-1)
    z_aug = F.normalize(z_aug, dim=-1)

    logits = z @ z_aug.T / tau
    log_probs = logits - torch.logsumexp(logits, dim=1, keepdim=True)

    w = rs_cl_weights(proprio, beta=beta)
    return -(w * log_probs).sum(dim=1).mean()


def vanilla_infonce_loss(
    z: torch.Tensor,
    z_aug: torch.Tensor,
    tau: float = 0.2,
) -> torch.Tensor:
    # standard infonce with uniform weights (no proprio weighting)
    z = F.normalize(z, dim=-1)
    z_aug = F.normalize(z_aug, dim=-1)

    logits = z @ z_aug.T / tau
    labels = torch.arange(z.shape[0], device=z.device)
    return F.cross_entropy(logits, labels)
