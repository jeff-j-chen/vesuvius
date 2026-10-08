"""hallucination route (crossres/PLAN.md section 0.5): the frozen plan-D generator in front of the ink model.

the generator (models/c39_mae_crossres_depth.generator.pth) predicts depth_factor sub-slices per native slice
(8 -> 32) from the normalised crop. the 8 real slices and the 32 predicted ones are stacked and a learned
per-voxel depth mix maps the 40 to out_depth (16) network slices. the mix starts as the predicted sub-slices
average-pooled in pairs, with zero weight on the real slices, so training can shift trust back to the real data.
"""
from __future__ import annotations

import sys
from pathlib import Path
from types import SimpleNamespace

import torch
import torch.nn as nn
import torch.nn.functional as F


class GeneratedDepthInput(nn.Module):
    def __init__(self, path: str, out_depth: int = 16):
        super().__init__()
        checkpoint = torch.load(path, map_location="cpu", weights_only=False)
        self.depth, self.factor = int(checkpoint["depth"]), int(checkpoint["depth_factor"])
        if int(checkpoint["xy_scale"]) != 1 or checkpoint.get("sr_head") != "deep":
            raise ValueError(f"{path}: only depth-only (xy_scale 1) deep-head generators are supported")
        predicted = self.depth * self.factor
        if predicted % out_depth:
            raise ValueError(f"out_depth {out_depth} must divide the {predicted} predicted sub-slices")
        self.out_depth = int(out_depth)

        sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "crossres"))
        from mae_pretrain_crossres import DeepSRHead, build_config
        from utils.model import NnUnet3dLcndz
        config = build_config(SimpleNamespace(
            fiber_coordinate_branch=True, data_norm_mode="surface_anchor", depth=self.depth,
            d_start=8, d_end=20, ctx=int(checkpoint["ctx"]),
        ))
        self.backbone = NnUnet3dLcndz(config)
        channels = int(self.backbone.early2d_head.in_channels)
        self.sr_head = DeepSRHead(channels, self.depth, predicted, 1)
        state = checkpoint["state_dict"]
        self.backbone.load_state_dict({k[9:]: v for k, v in state.items() if k.startswith("backbone.")})
        self.sr_head.load_state_dict({k[8:]: v for k, v in state.items() if k.startswith("sr_head.")})
        self.backbone.requires_grad_(False)
        self.sr_head.requires_grad_(False)

        pool = predicted // out_depth
        weight = torch.zeros(out_depth, self.depth + predicted)
        for k in range(out_depth):
            weight[k, self.depth + k * pool:self.depth + (k + 1) * pool] = 1.0 / pool
        self.mix_weight = nn.Parameter(weight)
        self.mix_bias = nn.Parameter(torch.zeros(out_depth))
        self.eval()

    def train(self, mode: bool = True):
        super().train(mode)
        # frozen generator: keep its normalisation in inference mode
        self.backbone.eval()
        self.sr_head.eval()
        return self

    @torch.no_grad()
    def predict(self, x: torch.Tensor) -> torch.Tensor:
        """(B, 1, D, H, W) normalised crop -> (B, 1, D * factor, H, W) predicted sub-slices."""
        _, dec1 = self.backbone._encode_decode_early_2d(x, None, None)
        residual = self.sr_head(dec1, x).unsqueeze(1).float()
        return residual + F.interpolate(x.float(), scale_factor=(self.factor, 1, 1), mode="trilinear",
                                        align_corners=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.shape[2] != self.depth:
            raise ValueError(f"generator expects {self.depth} input slices, got {x.shape[2]}")
        stacked = torch.cat((x.float(), self.predict(x)), dim=2)[:, 0]
        mixed = torch.einsum("od,bdhw->bohw", self.mix_weight, stacked) + self.mix_bias.view(1, -1, 1, 1)
        return mixed.unsqueeze(1)
