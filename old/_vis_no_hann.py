"""_vis_no_hann.py — visualize the current best model using NON-overlapping chunks.

shows the eval composite without Hann blending so we can compare with vs without.
output saved to runs_scroll4_79um/smoke_probe/no_hann_composite.png
"""
import os, torch, numpy as np
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"

from utils.config import Config
from utils.model import create_model
import zarr, cv2
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from torch.amp.autocast_mode import autocast

SCROLL4_79_ID = 20240304161941
MODEL_PATH    = "models/best_model_loss.pth"
OUT_DIR       = "runs_scroll4_79um/smoke_probe"
os.makedirs(OUT_DIR, exist_ok=True)

c = Config()
c.data.tra_scroll_id   = SCROLL4_79_ID
c.data.tile_size       = 32
c.data.depth           = 8
c.data.d_start         = 0;  c.data.d_end = 64
c.model.arch           = "dense_unet_res_attn"
dev = c.device

# load model (InstanceNorm weights from v4)
model, _ = create_model(c)
model.load_state_dict(torch.load(MODEL_PATH, map_location=dev, weights_only=False))
model.eval()
print(f"model loaded  device={dev}")

# open zarr + norm
import json
vol = zarr.open(f"C:/Users/ChenJeff/Documents/ves_zarrs2/{SCROLL4_79_ID}.zarr", mode='r')
with open("norm_cache.json") as f:
    nc = json.load(f)[str(SCROLL4_79_ID)]
g_mean, g_std, g_min, g_max = nc["mean"], nc["std"], nc["min"], nc["max"]

# region: same crop as training (right 40% x, top 75% y)
T = 32; D = 8
H, W = int(vol.shape[1]), int(vol.shape[2])
x0 = (int(W*0.6)//T)*T; x1 = (int(W*1.0)//T)*T
y0 = 0;                  y1 = (int(H*0.75)//T)*T
print(f"region y[{y0},{y1}] x[{x0},{x1}]")

Hreg, Wreg = y1-y0, x1-x0
DS = max(1, int(np.ceil(max(Hreg,Wreg)/3000.0)))
Hc, Wc = Hreg//DS, Wreg//DS

# chunk size — NO OVERLAP, plain stride=CH
CH = (760//8)*8  # same size as Hann version, but no overlap

def _norm(blk):
    blk = (blk - g_mean) / g_std
    return np.clip((blk - g_min)/(g_max - g_min + 1e-12), 0, 1)

def _pad8(a):
    _, h, w = a.shape
    ph = (-h)%8; pw = (-w)%8
    if ph or pw: a = np.pad(a, ((0,0),(0,ph),(0,pw)), mode="reflect")
    return a, h, w

ys = list(range(y0, y1, CH))
xs = list(range(x0, x1, CH))
print(f"{len(ys)} y-chunks × {len(xs)} x-chunks = {len(ys)*len(xs)} per depth block")

def _predict_no_hann(z0):
    canvas = np.zeros((Hc, Wc), np.float32)
    import cv2 as _cv
    with torch.no_grad():
        for yy in ys:
            ch = min(CH, y1-yy)
            for xx in xs:
                cw = min(CH, x1-xx)
                blk = np.asarray(vol[z0:z0+D, yy:yy+ch, xx:xx+cw]).astype(np.float32)
                if blk.shape[0] != D: continue
                blk = _norm(blk)
                blk, oh, ow = _pad8(blk)
                bt = torch.from_numpy(blk).unsqueeze(0).unsqueeze(0).float().to(dev)
                with autocast(dev):
                    p = torch.sigmoid(model(bt))[0,0,:oh,:ow].float().cpu().numpy()
                cyd, cxd = (yy-y0)//DS, (xx-x0)//DS
                chd, cwd = oh//DS, ow//DS
                if chd<1 or cwd<1: continue
                pd = _cv.resize(p, (cwd,chd), interpolation=_cv.INTER_AREA)
                canvas[cyd:cyd+chd, cxd:cxd+cwd] = pd
    return canvas

# just do z=0-8 and the composite for speed
preds = []
for z0 in [0, 16, 32, 48]:
    print(f"  running z{z0}-{z0+D}...")
    preds.append((z0, _predict_no_hann(z0)))

composite = np.max(np.stack([p for _,p in preds], axis=0), axis=0)

# mask
mask = cv2.imread(f"masks/{SCROLL4_79_ID}.png", cv2.IMREAD_GRAYSCALE)/255.0
import cv2 as _cv
mask_ds = _cv.resize((mask[y0:y1,x0:x1]>0).astype(np.float32),(Wc,Hc),interpolation=_cv.INTER_AREA)
mask_ds = (mask_ds>0.5).astype(np.float32)
composite = composite * mask_ds

fig, ax = plt.subplots(1,1, figsize=(14, 10))
ax.imshow(composite, cmap="magma", vmin=0, vmax=1)
ax.set_title(f"depth-MAX composite — NO Hann overlap (v4 InstanceNorm model)\n"
             f"z-blocks z0,16,32,48  region y[{y0},{y1}] x[{x0},{x1}]")
ax.axis("off")
plt.tight_layout()
out = os.path.join(OUT_DIR, "no_hann_composite.png")
fig.savefig(out, dpi=100)
plt.close(fig)
print(f"saved -> {out}")
print(f"composite stats: min={composite.min():.4f}  max={composite.max():.4f}  mean={composite[mask_ds>0].mean():.4f}")
