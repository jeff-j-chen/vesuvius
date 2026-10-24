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


def build_mix6_init(path: str | None = None, seed: int | None = None) -> str:
    """the mix6 run's init weights: the 192 MAE grown to four input heads; the new head's enc1 weights start at
    0.5x the other heads' rms. the fresh gated-stem branch and the head's enc1 weights are drawn from `seed`."""
    import os

    from .config import MIX6_INIT, MIX6_MAE192, MIX6_SEED, Config, apply_mix6_run
    from .model import create_model

    path = path or MIX6_INIT
    seed = MIX6_SEED if seed is None else seed
    config = Config()
    config.device = "cpu"
    apply_mix6_run(config)
    config.model.compile_model = False
    with torch.random.fork_rng():
        torch.manual_seed(seed)
        model, _ = create_model(config)
    model_state = model.state_dict()
    state = expand_cue_state(torch.load(MIX6_MAE192, map_location="cpu", weights_only=True), model_state)
    bad = [k for k, v in state.items() if k in model_state and v.shape != model_state[k].shape]
    if bad:
        raise RuntimeError(f"expanded checkpoint does not fit the 4-head model: {bad[:5]}")
    weight, head = state["enc1.net.0.weight"], 3
    rms = [float(weight[:, i].pow(2).mean().sqrt()) for i in range(head)]
    fresh = torch.randn(weight[:, head].shape, generator=torch.Generator().manual_seed(seed))
    weight[:, head] = fresh * (0.5 * sum(rms) / head) / fresh.pow(2).mean().sqrt()
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    torch.save(state, path)
    print(f"[init] 4-head init {path} (seed {seed}): head enc1 rms {0.5 * sum(rms) / head:.4f}", flush=True)
    return path
