import torch
import torch.nn.functional as F


def rs_cl_weights(
    proprio: torch.Tensor,
    beta: float = 1.0,
    depth: torch.Tensor = None,
    alpha: float = 0.5,
) -> torch.Tensor:
    # soft weights from state distances
    # if depth is provided: d = alpha * ||q_i - q_j|| + (1-alpha) * ||f^d_i - f^d_j||
    q = F.normalize(proprio, dim=-1)
    dists = torch.cdist(q, q, p=2)

    if depth is not None:
        d = F.normalize(depth, dim=-1)
        dists_depth = torch.cdist(d, d, p=2)
        dists = alpha * dists + (1 - alpha) * dists_depth

    return torch.softmax(-dists / beta, dim=1)


def rs_cl_loss(
    z: torch.Tensor,
    z_aug: torch.Tensor,
    proprio: torch.Tensor,
    tau: float = 0.2,
    beta: float = 1.0,
    depth: torch.Tensor = None,
    alpha: float = 0.5,
) -> torch.Tensor:
    # infonce with proprioceptive (+ optional depth) soft weights (eq. 3 in paper)
    z = F.normalize(z, dim=-1)
    z_aug = F.normalize(z_aug, dim=-1)

    logits = z @ z_aug.T / tau
    log_probs = logits - torch.logsumexp(logits, dim=1, keepdim=True)

    w = rs_cl_weights(proprio, beta=beta, depth=depth, alpha=alpha)
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
