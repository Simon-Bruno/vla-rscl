import torch
import torch.nn.functional as F


def rs_cl_weights(
    proprio: torch.Tensor,
    beta: float = 1.0,
    depth: torch.Tensor = None,
    alpha: float = 0.5,
    action: torch.Tensor = None,
    gamma: float = 0.0,
) -> torch.Tensor:
    # soft weights from multi-modal state distances
    # d(i,j) = (1-gamma) * [alpha * d_q + (1-alpha) * d_depth] + gamma * d_action
    # set alpha=1.0 to disable depth, gamma=0.0 to disable action
    q = F.normalize(proprio, dim=-1)
    dists = alpha * torch.cdist(q, q, p=2)

    if depth is not None and alpha < 1.0:
        d = F.normalize(depth, dim=-1)
        dists = dists + (1.0 - alpha) * torch.cdist(d, d, p=2)

    dists = (1.0 - gamma) * dists

    if action is not None and gamma > 0.0:
        a = F.normalize(action, dim=-1)
        dists = dists + gamma * torch.cdist(a, a, p=2)

    return torch.softmax(-dists / beta, dim=1)


def rs_cl_loss(
    z: torch.Tensor,
    z_aug: torch.Tensor,
    proprio: torch.Tensor,
    tau: float = 0.2,
    beta: float = 1.0,
    depth: torch.Tensor = None,
    alpha: float = 0.5,
    action: torch.Tensor = None,
    gamma: float = 0.0,
) -> torch.Tensor:
    # infonce with multi-modal soft weights (eq. 3 in paper, extended)
    z = F.normalize(z, dim=-1)
    z_aug = F.normalize(z_aug, dim=-1)

    logits = z @ z_aug.T / tau
    log_probs = logits - torch.logsumexp(logits, dim=1, keepdim=True)

    w = rs_cl_weights(proprio, beta=beta, depth=depth, alpha=alpha, action=action, gamma=gamma)
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
