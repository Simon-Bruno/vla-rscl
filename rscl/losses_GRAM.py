import torch
import torch.nn.functional as F


def rscl_soft_weights(
    proprio: torch.Tensor,
    beta: float = 1.0,
    depth: torch.Tensor | None = None,
    velocity: torch.Tensor | None = None,
    w_q: float = 1.0,
    w_depth: float = 0.0,
    w_vel: float = 0.0,
) -> torch.Tensor:
    dists = torch.zeros(proprio.shape[0], proprio.shape[0], device=proprio.device)

    total = w_q
    total += w_depth if depth is not None else 0.0
    total += w_vel if velocity is not None else 0.0
    total = max(total, 1.0e-8)

    if w_q > 0.0:
        q = F.normalize(proprio, dim=-1)
        dists = dists + (w_q / total) * torch.cdist(q, q, p=2)

    if depth is not None and w_depth > 0.0:
        has_depth = depth.norm(dim=-1) > 1.0e-6
        if has_depth.any():
            d = F.normalize(depth, dim=-1)
            depth_dists = torch.cdist(d, d, p=2)
            mask = has_depth.float().unsqueeze(1) * has_depth.float().unsqueeze(0)
            dists = dists + (w_depth / total) * depth_dists * mask

    if velocity is not None and w_vel > 0.0:
        v = F.normalize(velocity, dim=-1)
        dists = dists + (w_vel / total) * torch.cdist(v, v, p=2)

    return torch.softmax(-dists / beta, dim=1)


def gram_volume_matrix(
    anchor: torch.Tensor, *modalities: torch.Tensor, eps: float = 1.0e-8
) -> torch.Tensor:
    """Pairwise GRAM volume matrix.

    Entry (i, j) is the volume spanned by anchor[i] and all modality[j]
    vectors. Smaller volume means stronger multimodal alignment.
    """
    if len(modalities) < 1:
        raise ValueError("gram_volume_matrix requires at least one candidate modality")

    anchor = F.normalize(anchor, dim=-1)
    modalities = tuple(F.normalize(m, dim=-1) for m in modalities)

    b_anchor = anchor.shape[0]
    b_mod = modalities[0].shape[0]
    k = 1 + len(modalities)

    rows = []
    aa = torch.ones(b_anchor, b_mod, device=anchor.device, dtype=anchor.dtype)
    anchor_to_mod = [anchor @ m.T for m in modalities]
    rows.append(torch.stack([aa] + anchor_to_mod, dim=-1))

    for m_i in modalities:
        row = [anchor @ m_i.T]
        for m_j in modalities:
            dot = torch.einsum("bd,bd->b", m_i, m_j)
            row.append(dot.unsqueeze(0).expand(b_anchor, -1))
        rows.append(torch.stack(row, dim=-1))

    gram = torch.stack(rows, dim=-2)
    if gram.shape[-2:] != (k, k):
        raise RuntimeError(
            f"unexpected Gram shape {gram.shape}; expected (..., {k}, {k})"
        )

    det = torch.linalg.det(gram.float())
    volume = det.clamp(min=eps).sqrt()
    return volume.to(anchor.dtype)


def rscl_gram_loss(
    z: torch.Tensor,
    proprio_embed: torch.Tensor,
    proprio: torch.Tensor,
    depth_embed: torch.Tensor | None = None,
    depth: torch.Tensor | None = None,
    velocity_embed: torch.Tensor | None = None,
    velocity: torch.Tensor | None = None,
    tau: float = 0.2,
    beta: float = 1.0,
    w_q: float = 1.0,
    w_depth: float = 0.0,
    w_vel: float = 0.0,
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    modalities = [proprio_embed]
    if depth_embed is not None:
        modalities.append(depth_embed)
    if velocity_embed is not None:
        modalities.append(velocity_embed)

    volume = gram_volume_matrix(z, *modalities)
    logits = -volume / tau
    log_probs = logits - torch.logsumexp(logits, dim=1, keepdim=True)

    weights = rscl_soft_weights(
        proprio,
        beta=beta,
        depth=depth,
        velocity=velocity,
        w_q=w_q,
        w_depth=w_depth,
        w_vel=w_vel,
    )
    loss = -(weights * log_probs).sum(dim=1).mean()

    with torch.no_grad():
        diag = volume.diag()
        off_diag = volume[
            ~torch.eye(volume.shape[0], dtype=torch.bool, device=volume.device)
        ]
        metrics = {
            "gram_pos_volume": diag.mean(),
            "gram_neg_volume": (
                off_diag.mean()
                if off_diag.numel() > 0
                else torch.zeros((), device=volume.device)
            ),
            "gram_pos_neg_gap": (
                (off_diag.mean() - diag.mean())
                if off_diag.numel() > 0
                else torch.zeros((), device=volume.device)
            ),
            "rscl_target_entropy": -(weights * weights.clamp_min(1.0e-8).log())
            .sum(dim=1)
            .mean(),
        }

    return loss, metrics
