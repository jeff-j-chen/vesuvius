"""surface pseudo-targets and losses for depth-localized papyrus geometry."""
from __future__ import annotations

import torch
import torch.nn.functional as F


def make_surface_targets(
    volume: torch.Tensor,
    threshold_frac: float = 0.35,
    min_contrast: float = 0.08,
    min_peak: float = 0.08,
    min_margin: float = 0.02,
    target_sigma: float = 0.75,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """derive a soft single-surface target from papyrus-to-air depth transitions.

    volume is normalized (B, 1, D, H, W). the returned target has the same
    shape and sums to one over depth. valid is (B, 1, H, W), while depth_index
    is the selected zero-based depth class for diagnostics.
    """
    if volume.ndim != 5 or volume.shape[1] != 1:
        raise ValueError(f"expected (B, 1, D, H, W), got {tuple(volume.shape)}")
    if volume.shape[2] < 2:
        raise ValueError("surface targets require at least two depth slices")

    with torch.no_grad():
        x = volume.detach().float()
        depth_smoothed = F.pad(x, (0, 0, 0, 0, 1, 1), mode="replicate")
        depth_smoothed = (
            depth_smoothed[:, :, :-2]
            + 2.0 * depth_smoothed[:, :, 1:-1]
            + depth_smoothed[:, :, 2:]
        ) * 0.25

        low = torch.quantile(depth_smoothed, 0.10, dim=2, keepdim=True)
        high = torch.quantile(depth_smoothed, 0.90, dim=2, keepdim=True)
        contrast = high - low
        threshold = low + float(threshold_frac) * contrast
        tau = (0.08 * contrast).clamp(min=0.01)
        occupancy = torch.sigmoid((depth_smoothed - threshold) / tau)

        transition = (occupancy[:, :, :-1] - occupancy[:, :, 1:]).clamp(min=0.0)
        spatial_size = min(transition.shape[-2:])
        fine_kernel = min(5, spatial_size if spatial_size % 2 else spatial_size - 1)
        coarse_kernel = min(17, spatial_size if spatial_size % 2 else spatial_size - 1)
        fine = F.avg_pool3d(
            transition,
            kernel_size=(1, fine_kernel, fine_kernel),
            stride=1,
            padding=(0, fine_kernel // 2, fine_kernel // 2),
        )
        coarse = F.avg_pool3d(
            transition,
            kernel_size=(1, coarse_kernel, coarse_kernel),
            stride=1,
            padding=(0, coarse_kernel // 2, coarse_kernel // 2),
        )
        transition = 0.35 * fine + 0.65 * coarse
        top2 = transition.topk(k=2, dim=2).values
        peak = top2[:, :, 0]
        margin = top2[:, :, 0] - top2[:, :, 1]
        depth_index = transition.argmax(dim=2)

        valid = (
            (contrast.squeeze(2) >= float(min_contrast))
            & (peak >= float(min_peak))
            & (margin >= float(min_margin))
        )
        depth_axis = torch.arange(x.shape[2], device=x.device, dtype=x.dtype).view(1, 1, -1, 1, 1)
        target = torch.exp(
            -0.5
            * ((depth_axis - depth_index.unsqueeze(2).to(x.dtype)) / max(float(target_sigma), 1e-3)) ** 2
        )
        target = target / target.sum(dim=2, keepdim=True).clamp(min=1e-8)
        return target, valid.float(), depth_index


def surface_supervision_loss(
    logits: torch.Tensor,
    volume: torch.Tensor,
    smooth_weight: float = 0.02,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """soft depth cross-entropy plus weak robust spatial smoothness."""
    target, valid, _ = make_surface_targets(volume)
    log_probs = F.log_softmax(logits.float(), dim=2)
    ce_map = -(target * log_probs).sum(dim=2)
    denom = valid.sum().clamp(min=1.0)
    ce_loss = (ce_map * valid).sum() / denom

    probs = F.softmax(logits.float(), dim=2)
    depth_axis = torch.arange(logits.shape[2], device=logits.device, dtype=probs.dtype).view(1, 1, -1, 1, 1)
    expected = (probs * depth_axis).sum(dim=2)
    eps = 1e-3

    dx = expected[:, :, :, 1:] - expected[:, :, :, :-1]
    dy = expected[:, :, 1:, :] - expected[:, :, :-1, :]
    vx = valid[:, :, :, 1:] * valid[:, :, :, :-1]
    vy = valid[:, :, 1:, :] * valid[:, :, :-1, :]
    sx = ((torch.sqrt(dx.square() + eps * eps) - eps) * vx).sum() / vx.sum().clamp(min=1.0)
    sy = ((torch.sqrt(dy.square() + eps * eps) - eps) * vy).sum() / vy.sum().clamp(min=1.0)
    smooth_loss = 0.5 * (sx + sy)

    total = ce_loss + float(smooth_weight) * smooth_loss
    return total, ce_loss.detach(), smooth_loss.detach(), valid.mean().detach()
