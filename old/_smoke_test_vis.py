"""_smoke_test_vis.py — run all visualizer outputs against a saved checkpoint.

loads the best_model_loss.pth (dense_unet_depth, depth=64) checkpoint and fires:
  - add_dense_probe_figure   (all named probe ROIs)
  - add_dense_evaluation_figure (full region, all depths)
  - add_test_figures         (scroll2 + scroll3 via test-scroll2-only logic)

figures are written to runs_scroll4_79um/smoke_test/ so they can be inspected
without touching any live tensorboard run folder.
"""
import os, sys, torch

os.environ.setdefault("TF_ENABLE_ONEDNN_OPTS", "0")
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")

# ── config ──────────────────────────────────────────────────────────────────
SCROLL4_79_ID = 20240304161941
MODEL_PATH    = "models/best_model_loss.pth"
OUT_DIR       = "runs_scroll4_79um/smoke_test"
EPOCH         = 0      # epoch index passed to all figure methods (step for tensorboard)

# ── build config identical to the completed t02 run ──────────────────────────
from utils.config import Config
c = Config()
c.data.tra_scroll_id        = SCROLL4_79_ID
c.data.tra_scroll_ids       = [SCROLL4_79_ID]
c.data.tile_size            = 32
c.data.depth                = 8
c.data.d_start              = 0
c.data.d_end                = 64
c.data.train_d_start        = 0
c.data.train_d_end          = 64
c.data.ring_negatives       = True
c.data.ring_label_source    = "eroded"
c.data.split_axis           = "y"
c.data.train_split_frac     = 0.75
c.data.crop_x_frac          = (0.6, 1.0)
c.data.crop_y_frac          = (0.0, 0.75)
c.data.dense_labels         = True
c.data.dense_soft_labels    = False
c.data.mask_memmap          = False
c.data.test_scroll2_only    = True   # skip expensive training-scroll test figure
c.tra.log_dir               = OUT_DIR
c.tra.eval_int              = 1
c.tra.probe_int             = 1
c.tra.test_int              = 1
c.exp_name                  = "smoke"

os.makedirs(OUT_DIR, exist_ok=True)

# ── load model ────────────────────────────────────────────────────────────────
from utils.model import create_model
c.model.arch = "dense_unet"   # t01 architecture — best performer (valid AUC 0.5548)
model, _ = create_model(c)
state = torch.load(MODEL_PATH, map_location=c.device, weights_only=False)
# best_model_loss.pth was overwritten by later runs; load what's available.
# the weights won't match dense_unet exactly, but the smoke test validates
# all code paths regardless of weight quality.
try:
    model.load_state_dict(state)
    print(f"[smoke] loaded {MODEL_PATH} (dense_unet weights)")
except RuntimeError:
    # shape mismatch from a different arch — load with strict=False so we
    # can still exercise all figure generation code paths
    model.load_state_dict(state, strict=False)
    print(f"[smoke] loaded {MODEL_PATH} with strict=False (arch mismatch expected)")
model.eval()
print(f"[smoke] device={c.device}")

# ── init visualizer ───────────────────────────────────────────────────────────
from utils.visualizer import TensorboardVisualizer
print("[smoke] initializing visualizer (this loads all volumes + norms)...")
vis = TensorboardVisualizer(c)

# ── probe ROIs ────────────────────────────────────────────────────────────────
print("\n[smoke] ── probe ROIs ──")
try:
    vis.add_dense_probe_figure(EPOCH, model)
    print("[smoke] probe figures done")
except Exception as e:
    import traceback; traceback.print_exc()
    print(f"[smoke] probe FAILED: {e}")

# ── eval figure (full region, all depths) ────────────────────────────────────
print("\n[smoke] ── dense eval figure ──")
try:
    vis.add_dense_evaluation_figure(EPOCH, model)
    print("[smoke] eval figure done")
except Exception as e:
    import traceback; traceback.print_exc()
    print(f"[smoke] eval FAILED: {e}")

# ── test figures (scroll2 + scroll3) ─────────────────────────────────────────
print("\n[smoke] ── test figures (scroll2 + scroll3) ──")
try:
    vis.add_test_figures(EPOCH, model)
    print("[smoke] test figures done")
except Exception as e:
    import traceback; traceback.print_exc()
    print(f"[smoke] test FAILED: {e}")

vis.close()
print(f"\n[smoke] ALL DONE. figures in {OUT_DIR}/dense_figs/ and tensorboard run '{vis.log_path}'")
