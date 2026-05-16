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


def gram_volume_loss(embeddings: list[torch.Tensor]) -> torch.Tensor:
    """GRAM volume loss for cross-modal alignment (Cicchetti et al., ICLR 2025).

    Computes the volume of the parallelotope spanned by k modality embeddings.
    Minimizing this volume encourages all modalities to be geometrically aligned
    in the shared embedding space.

    Args:
        embeddings: list of k tensors, each (B, d), already L2-normalized.
                    Requires k >= 2.

    Returns:
        Scalar loss = mean over batch of sqrt(det(G)),
        where G is the k x k Gram matrix.
    """
    assert len(embeddings) >= 2, f"gram_volume_loss requires >= 2 modalities, got {len(embeddings)}"
    A = torch.stack(embeddings, dim=1)                # (B, k, d)
    G = torch.bmm(A, A.transpose(1, 2))              # (B, k, k)
    # det + clamp + sqrt is simpler and more stable than slogdet for small k
    det_G = torch.linalg.det(G.float())               # (B,)
    vol = det_G.clamp(min=1e-8).sqrt()                 # (B,)
    return vol.mean()
