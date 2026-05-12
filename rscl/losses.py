import torch
import torch.nn.functional as F


def rs_cl_weights(
    proprio: torch.Tensor,
    beta: float = 1.0,
    depth: torch.Tensor = None,
    action: torch.Tensor = None,
    ee_vel: torch.Tensor = None,
    w_q: float = 1.0,
    w_depth: float = 0.0,
    w_action: float = 0.0,
    w_vel: float = 0.0,
) -> torch.Tensor:
    # soft weights from multi-modal state distances
    # set a weight to 0.0 to disable that modality
    dists = torch.zeros(proprio.shape[0], proprio.shape[0], device=proprio.device)

    total = w_q + (w_depth if depth is not None else 0.0) + (w_action if action is not None else 0.0) + (w_vel if ee_vel is not None else 0.0)
    if total == 0.0:
        total = 1.0  # fallback: uniform weights

    if w_q > 0.0:
        q = F.normalize(proprio, dim=-1)
        dists = dists + (w_q / total) * torch.cdist(q, q, p=2)

    if depth is not None and w_depth > 0.0:
        # mask out samples where depth is all-zero (missing depth PNG -> fallback zeros)
        # so they don't spuriously attract each other in the distance matrix
        has_depth = (depth.norm(dim=-1) > 1e-6)  # (B,)
        if has_depth.any():
            d = F.normalize(depth, dim=-1)
            dist_d = torch.cdist(d, d, p=2)
            mask = has_depth.float().unsqueeze(1) * has_depth.float().unsqueeze(0)
            dists = dists + (w_depth / total) * dist_d * mask

    if action is not None and w_action > 0.0:
        a = F.normalize(action, dim=-1)
        dists = dists + (w_action / total) * torch.cdist(a, a, p=2)

    if ee_vel is not None and w_vel > 0.0:
        v = F.normalize(ee_vel, dim=-1)
        dists = dists + (w_vel / total) * torch.cdist(v, v, p=2)

    return torch.softmax(-dists / beta, dim=1)


def rs_cl_loss(
    z: torch.Tensor,
    z_aug: torch.Tensor,
    proprio: torch.Tensor,
    tau: float = 0.2,
    beta: float = 1.0,
    depth: torch.Tensor = None,
    action: torch.Tensor = None,
    ee_vel: torch.Tensor = None,
    w_q: float = 1.0,
    w_depth: float = 0.0,
    w_action: float = 0.0,
    w_vel: float = 0.0,
) -> torch.Tensor:
    # infonce with multi-modal soft weights
    z = F.normalize(z, dim=-1)
    z_aug = F.normalize(z_aug, dim=-1)

    logits = z @ z_aug.T / tau
    log_probs = logits - torch.logsumexp(logits, dim=1, keepdim=True)

    w = rs_cl_weights(proprio, beta=beta, depth=depth, action=action, ee_vel=ee_vel,
                      w_q=w_q, w_depth=w_depth, w_action=w_action, w_vel=w_vel)
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
