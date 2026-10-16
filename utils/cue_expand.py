"""grow a 3-cue (raw, lcn, dz) checkpoint to the 4-cue stem; the new cue starts with no effect on enc1."""
from __future__ import annotations

import torch


def expand_cue_state(state: dict, model_state: dict) -> dict:
    """copy `state` and pad every stem tensor that is one cue short: zero enc1 input channel, zero gate columns/row
    for the new cue, and the freshly initialized branch (taken from model_state) for the new cue."""
    out = dict(state)
    key = "enc1.net.0.weight"
    if key in state and key in model_state and state[key].shape[1] + 1 == model_state[key].shape[1]:
        out[key] = torch.cat([state[key], torch.zeros_like(state[key][:, :1])], dim=1)
    old = state.get("gated_cue_stem.gate.0.weight")
    new = model_state.get("gated_cue_stem.gate.0.weight")
    if old is not None and new is not None and old.shape[1] == 6 and new.shape[1] == 8:
        pad = torch.zeros(old.shape[0], 1, dtype=old.dtype, device=old.device)
        # statistics are (3 means, 3 stds) -> (4 means, 4 stds)
        out["gated_cue_stem.gate.0.weight"] = torch.cat([old[:, :3], pad, old[:, 3:], pad], dim=1)
        last_w, last_b = state["gated_cue_stem.gate.2.weight"], state["gated_cue_stem.gate.2.bias"]
        out["gated_cue_stem.gate.2.weight"] = torch.cat([last_w, torch.zeros_like(last_w[:1])], dim=0)
        out["gated_cue_stem.gate.2.bias"] = torch.cat([last_b, torch.zeros_like(last_b[:1])], dim=0)
        for suffix in ("0.weight", "1.weight", "1.bias", "3.weight", "3.bias"):
            name = f"gated_cue_stem.branches.3.{suffix}"
            if name in model_state:
                out[name] = model_state[name].detach().clone().to(old.device)
    return out
