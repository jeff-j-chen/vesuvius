import os, sys, torch
os.environ.setdefault("TF_ENABLE_ONEDNN_OPTS", "0")
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")

from utils.config import Config
c = Config()
c.data.tra_scroll_id      = 20240304161941
c.data.tra_scroll_ids     = [20240304161941]
c.data.tile_size          = 32
c.data.depth              = 8
c.data.d_start            = 0;  c.data.d_end = 64
c.data.train_d_start      = 0;  c.data.train_d_end = 64
c.data.ring_negatives     = True
c.data.ring_label_source  = "eroded"
c.data.split_axis         = "y"
c.data.train_split_frac   = 0.75
c.data.crop_x_frac        = (0.6, 1.0)
c.data.crop_y_frac        = (0.0, 0.75)
c.data.dense_labels       = True
c.data.mask_memmap        = False
c.tra.log_dir             = "runs_scroll4_79um/smoke_probe"
c.tra.eval_int            = 999; c.tra.probe_int = 1; c.tra.test_int = 999
c.exp_name                = "probe_test"
os.makedirs(c.tra.log_dir, exist_ok=True)

from utils.model import create_model
c.model.arch = "dense_unet"
model, _ = create_model(c)
state = torch.load("models/best_model_loss.pth", map_location=c.device, weights_only=False)
model.load_state_dict(state, strict=False)
model.eval()
print(f"[probe] model ready  device={c.device}")

from utils.visualizer import TensorboardVisualizer
vis = TensorboardVisualizer(c)
print("[probe] firing combined probe figure...")
vis.add_dense_probe_figure(0, model)
vis.close()
print("[probe] done -> runs_scroll4_79um/smoke_probe/dense_figs/probe_combined_ep01.png")
