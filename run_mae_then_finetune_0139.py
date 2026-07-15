"""run_mae_then_finetune_0139.py — PHerc0139 w044 9.362um/113keV bridge experiment.

the FIRST attempt at ink detection on the new high-energy (>110keV) modality, via the
w044 segment that exists in BOTH 2.4um (ink already detected -> labels) and 9.362um.

chain (launch once, runs in sequence; each stage gated on the previous exiting rc==0):
  STAGE 0: precompute_norm for the rendered 9.3um zarr (if not cached)
  STAGE 1: MAE self-supervised pretrain on the 9.3um w044 surface volume
           - DEPTH 1: dense_unet's per-slice stem is depth-independent and its depth
             aggregation is a PARAMETER-FREE max(dim=2), so depth-1 pretraining trains
             exactly the transferable weights (stem + 2D U-Net) and transfers 1:1 to the
             depth-4 finetune -- while giving 28x more independent samples (one per slice).
           - saves models/mae_p0139_dense_unet.pth
  STAGE 2: ink finetune (dense_unet), warm-started from the MAE checkpoint, using the
           winning a1_blur config (cmp_scroll4_79um_v6_2026_07_12_a1_blur) EXCEPT:
           - DEPTH 4 (halved from a1_blur's 8, per the thinner 9.3um sheet)
           - full native depth window d[0:28]
           - horizontal split: top 80.5% (y 0-4850 of 6021) train, bottom 19.5% valid
           - probe_int / test_int OFF (9999): no probes/test figures set up for this scroll
           - eval_int 20: the single full-region eval figure fires once at the very end
  STAGE 3: v1 (InkDetector) TILE classifier baseline, from scratch (v1's conv3d+CBAM+MLP
           arch is incompatible with the dense_unet MAE, so no warm-start). identical data,
           split, ring, epochs -- but TILE-level single-value output (NO --dense-labels),
           the original 1-logit-per-tile training path. this is the head-to-head: per-pixel
           upscaled (dense_unet) vs single-value (v1) on the same 9.3um bridge data.

DATA (built by render_9um_surface.py, no VC3D):
  ves_zarrs2/20260115000000.zarr   (28, 6021, 8141) uint16  <- 9.3um flattened w044 surface
  eroded_inklabels/20260115000000.png   hand-written, resized to 9.3um frame (user)
  soft_inklabels/20260115000000.png     dilate1px + gaussian sigma=15 (a1_blur recipe)
  masks/20260115000000.png              valid-vertex footprint (renderer)
"""
from __future__ import annotations
import os, subprocess, sys, time

PY = sys.executable
REPO = os.path.dirname(os.path.abspath(__file__))

SID = "20260115000000"
MAE_NAME = "mae_p0139_dense_unet"
MAE_CKPT = f"models/{MAE_NAME}.pth"

# y 0->4850 of the 6021-tall render = upper 80.5% train, rest valid
SPLIT_FRAC = "0.8055"

NORM_CMD = [PY, "precompute_norm.py", "--scroll-id", SID]

MAE_CMD = [
    PY, "mae_pretrain.py",
    "-n", MAE_NAME,
    "--arch", "dense_unet_mae",
    "--scroll-ids", SID,
    "--tile-size", "64",
    "--depth", "4",                 # match finetune depth: exercises the depth-max
                                    # bottleneck + recon-all-slices pretext (no-op at depth 1)
    "--train-d-start", "0", "--train-d-end", "28",
    "--steps", "6000",
    "--batch-size", "64",
    "--mask-frac", "0.50",
    "--step-cooldown-ms", "120",
    "--cooldown-secs", "90",
    "--cooldown-int", "100",
]

FINETUNE_CMD = [
    PY, "train.py",
    "-n", "ink_p0139_9um_from_mae",
    "--arch", "dense_unet",
    "--init-weights", MAE_CKPT,
    "--scroll-id", SID,
    "--dense-labels", "--dense-soft-labels",
    "--tile-size", "32", "--depth", "4",             # depth halved 8 -> 4
    "--train-d-start", "0", "--train-d-end", "28",   # full native depth
    "--d-start", "0", "--d-end", "28",
    "--epochs", "20",
    "--eval-int", "20",           # full-region eval once, at the end
    "--probe-int", "9999",        # probes OFF (none set up)
    "--test-int", "9999",         # test figures OFF
    "--ring-negatives", "--ring-label-source", "eroded",
    "--crop-x-frac", "0.0,1.0", "--crop-y-frac", "0.0,1.0",
    "--split-axis", "y", "--train-split-frac", SPLIT_FRAC,
    "--data-aug", "0", "--l1-lambda", "0.0",
    "--batch-size", "1024", "--lr", "2e-4",
    "--log-dir", "runs_p0139_9um",
    "--num-workers", "2", "--mask-memmap",
    "--no-hard-mining", "--no-probe-rois",
    "--eval-cooldown", "240", "--val-cooldown", "90", "--fig-chunk-cooldown", "200",
]

# STAGE 3: v1 (InkDetector) TILE classifier -- single-value output, from scratch.
# NO --dense-labels => the dataloader emits scalar tile labels and v1 outputs one logit
# per tile (the original training path). same data/split/ring/epochs as the dense run.
V1_CMD = [
    PY, "train.py",
    "-n", "ink_p0139_9um_v1_tile",
    "--arch", "v1",
    "--scroll-id", SID,
    "--tile-size", "32", "--depth", "4",
    "--train-d-start", "0", "--train-d-end", "28",
    "--d-start", "0", "--d-end", "28",
    "--epochs", "20",
    "--eval-int", "20",
    "--probe-int", "9999",
    "--test-int", "9999",
    "--ring-negatives", "--ring-label-source", "eroded",
    "--crop-x-frac", "0.0,1.0", "--crop-y-frac", "0.0,1.0",
    "--split-axis", "y", "--train-split-frac", SPLIT_FRAC,
    "--data-aug", "0", "--l1-lambda", "0.0",
    "--batch-size", "1024", "--lr", "2e-4",
    "--log-dir", "runs_p0139_9um",
    "--num-workers", "2", "--mask-memmap",
    "--no-hard-mining", "--no-probe-rois",
    "--eval-cooldown", "240", "--val-cooldown", "90", "--fig-chunk-cooldown", "200",
]


def run(cmd, label):
    print(f"\n{'='*78}\n[chain] START {label}\n  {' '.join(cmd)}\n{'='*78}", flush=True)
    t0 = time.time()
    rc = subprocess.call(cmd, cwd=REPO)
    print(f"[chain] {label} exited rc={rc} after {(time.time()-t0)/60:.1f} min", flush=True)
    return rc


def _norm_cached():
    import json
    p = os.path.join(REPO, "norm_cache.json")
    if not os.path.exists(p):
        return False
    try:
        return SID in json.load(open(p))
    except Exception:
        return False


def main():
    if _norm_cached():
        print(f"[chain] norm already cached for {SID}; skipping STAGE 0")
    else:
        if run(NORM_CMD, "STAGE 0: precompute norm") != 0:
            print("[chain] ABORT — norm precompute failed"); return

    if run(MAE_CMD, "STAGE 1: MAE pretrain (depth 1, 6000 steps, 50% mask)") != 0:
        print("[chain] ABORT — MAE stage failed; not starting finetune."); return
    if not os.path.exists(os.path.join(REPO, MAE_CKPT)):
        print(f"[chain] ABORT — {MAE_CKPT} not found after MAE stage."); return

    run(FINETUNE_CMD, "STAGE 2: ink finetune from MAE (dense_unet, depth 4, a1_blur config)")
    run(V1_CMD, "STAGE 3: v1 tile-classifier baseline (single-value output, from scratch)")
    print("\n[chain] ALL DONE."
          "\n  dense_unet eval -> runs_p0139_9um/dense_figs/ (ink_p0139_9um_from_mae)"
          "\n  v1 tile eval   -> runs_p0139_9um/ (ink_p0139_9um_v1_tile)")


if __name__ == "__main__":
    main()
