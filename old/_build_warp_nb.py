"""builds warp_exploration.ipynb (w023 sheet, horizontal-flip, TPS peak-anchored warp).
overwrites in place via json.dump so it works while VS Code holds the file open.
run:  python _build_warp_nb.py   then reload the notebook."""
import json

def md(s): return {"cell_type": "markdown", "metadata": {}, "source": s.splitlines(keepends=True)}
def co(s): return {"cell_type": "code", "metadata": {}, "execution_count": None, "outputs": [], "source": s.splitlines(keepends=True)}

cells = []

cells.append(md(
"""# Scroll4 w023: 2.4um <-> 7.91um warp via scallop PEAK anchors (v4)

**Sheet swap.** w018 was one of the more warped sheets; we moved to **w023**, confirmed the same
physical region in both scans by aligning scan artifacts.

**Horizontal flip.** The 7.91um flatboi unrolling is **mirrored** relative to the 2.4um surface
(that is why earlier humps angled the wrong way). We flip the 7.91 frame `[:, ::-1]` up front;
after that, top AND bottom scallop peaks line up between the two frames.

**Warp method.** Tops are unreliable as full curves (ML surface-picking differs + torn edges), but
the scallop **peak points** on both edges DO correspond once flipped. So we use matched top+bottom
peak tips (+ a mid-height point per peak + 4 corners) as control points for a **thin-plate spline**
(scipy `RBFInterpolator`; this OpenCV build lacks TPS). This pins the sine-wave peaks and lets the
sheet deform smoothly between them.

**Final test.** Warp the 2.4 ink labels through the same field and overlay. Letters must stay
**clear and not mangled** (TPS is smooth, so shapes should be preserved).
"""))

cells.append(co(
"""import os
import numpy as np
import cv2
import tifffile
import matplotlib.pyplot as plt
from scipy.signal import find_peaks
from scipy.ndimage import minimum_filter1d
from scipy.interpolate import RBFInterpolator
from PIL import Image
Image.MAX_IMAGE_PIXELS = None

TMP = r'C:\\\\Users\\\\ChenJeff\\\\Documents\\\\_ves_tmp'
TEX24  = os.path.join(TMP, 'w023_24_tex.npy')        # 2.4 surface texture (level5 mid-depth)
MASK24 = os.path.join(TMP, 'w023_24_mask.png')       # 2.4 footprint mask
INK24  = os.path.join(TMP, 'w023_ink_full.tif')      # 2.4 ink prediction (full res)
TEX79F = os.path.join(TMP, 'w023_79_l32_flip.tif')   # 7.91 layer32, ALREADY horizontally flipped
MASK79F= os.path.join(TMP, 'w023_79_mask_flip.png')  # 7.91 mask, flipped

W = 1400   # detection/working width (peak detection is tuned for this; keep it)

def norm8(a, lp=2, hp=98):
    a = a.astype(np.float32); m = a > 0
    lo, hi = (np.percentile(a[m], [lp, hp]) if m.any() else (0, 1))
    return (np.clip((a - lo) / max(hi - lo, 1e-6), 0, 1) * 255).astype(np.uint8)

def tow(a, w=W, interp=cv2.INTER_AREA):
    s = w / a.shape[1]
    return cv2.resize(a, (w, int(round(a.shape[0]*s))), interpolation=interp)

tex79 = tow(norm8(tifffile.imread(TEX79F)))
tex24 = tow(norm8(np.load(TEX24)))
mask24 = tow(np.array(Image.open(MASK24).convert('L')), interp=cv2.INTER_NEAREST)
ink24 = tow((tifffile.imread(INK24)), interp=cv2.INTER_AREA)
H79, H24 = tex79.shape[0], tex24.shape[0]
print('7.91(flipped)', tex79.shape, ' 2.4', tex24.shape, ' ink', ink24.shape)

fig, ax = plt.subplots(2,1, figsize=(20,8))
ax[0].imshow(tex79, cmap='gray'); ax[0].set_title('7.91 texture (HORIZONTALLY FLIPPED) = TARGET')
ax[1].imshow(tex24, cmap='gray'); ax[1].set_title('2.4 texture = SOURCE (has ink labels)')
for a in ax: a.axis('off')
plt.tight_layout(); plt.show()
"""))

cells.append(md(
"""## 1. Sheet masks + boundary curves

7.91 sheet from texture (bright threshold + open + largest CC). 2.4 from its mask. Then per-column
top/bottom boundary curves. TOP = upper-envelope (rolling-min) to reject downward spikes where dark
cracks punch through the torn top edge; BOTTOM (clean scallops) only lightly smoothed."""))

cells.append(co(
"""def largest_cc(b):
    n, l, st, _ = cv2.connectedComponentsWithStats((b>0).astype(np.uint8), 8)
    return (l==(1+int(np.argmax(st[1:, cv2.CC_STAT_AREA])))).astype(np.uint8)*255 if n>1 else (b>0).astype(np.uint8)*255

def bright_sheet(tex, thr=60):
    b = cv2.morphologyEx((tex>thr).astype(np.uint8), cv2.MORPH_OPEN,
                         cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(5,5)))
    return largest_cc(b)

def boundaries(sheet, smooth=5, top_env=25):
    Wd = sheet.shape[1]
    top = np.full(Wd, np.nan); bot = np.full(Wd, np.nan)
    for x in range(Wd):
        ys = np.where(sheet[:, x] > 0)[0]
        if ys.size: top[x] = ys[0]; bot[x] = ys[-1]
    def fn(v):
        m = ~np.isnan(v); return np.interp(np.arange(len(v)), np.where(m)[0], v[m])
    top, bot = fn(top), fn(bot)
    top = cv2.GaussianBlur(minimum_filter1d(top, top_env).reshape(1,-1), (0,0), smooth).ravel()
    bot = cv2.GaussianBlur(bot.reshape(1,-1), (0,0), smooth).ravel()
    return top, bot

sheet79, sheet24 = bright_sheet(tex79), largest_cc(mask24)
top79, bot79 = boundaries(sheet79)
top24, bot24 = boundaries(sheet24)
print('sheets ok; 7.91 frac', round(float((sheet79>0).mean()),3), ' 2.4 frac', round(float((sheet24>0).mean()),3))
"""))

cells.append(md(
"""## 2. Detect + match scallop peaks (top AND bottom)

Peak tips of the boundary curves. After the horizontal flip these correspond on both edges. Match
by sorted x (common count per edge)."""))

cells.append(co(
"""def peaks(curve, kind, n=5):
    sig = curve if kind == 'down' else -curve   # 'down' = larger y (bottom scallops)
    pk, _ = find_peaks(sig, distance=len(curve)//(n*2), prominence=5)
    return pk

def match(a, b):
    k = min(len(a), len(b)); return a[:k], b[:k]

pb79, pb24 = match(peaks(bot79,'down'), peaks(bot24,'down'))
pt79, pt24 = match(peaks(top79,'up'),   peaks(top24,'up'))
print('bottom peaks matched:', len(pb79), ' 7.91', list(pb79), ' 2.4', list(pb24))
print('top peaks matched:   ', len(pt79), ' 7.91', list(pt79), ' 2.4', list(pt24))

vis = cv2.cvtColor(tex79, cv2.COLOR_GRAY2BGR)
for x in range(W):
    vis[int(np.clip(top79[x],0,H79-1)), x] = (0,255,0)      # green = top boundary
    vis[int(np.clip(bot79[x],0,H79-1)), x] = (0,0,255)      # red = bottom scallop line
for x in pb79: cv2.circle(vis,(int(x),int(bot79[x])),7,(255,0,255),-1)   # bottom peaks
for x in pt79: cv2.circle(vis,(int(x),int(top79[x])),7,(0,255,255),-1)   # top peaks
vis24 = cv2.cvtColor(tex24, cv2.COLOR_GRAY2BGR)
for x in range(W):
    vis24[int(np.clip(top24[x],0,H24-1)), x] = (0,255,0)
    vis24[int(np.clip(bot24[x],0,H24-1)), x] = (0,0,255)
for x in pb24: cv2.circle(vis24,(int(x),int(bot24[x])),7,(255,0,255),-1)
for x in pt24: cv2.circle(vis24,(int(x),int(top24[x])),7,(0,255,255),-1)
fig, ax = plt.subplots(2,1, figsize=(20,8))
ax[0].imshow(vis[:,:,::-1]);   ax[0].set_title('7.91 boundaries (green=top, RED=bottom scallops) + peaks')
ax[1].imshow(vis24[:,:,::-1]); ax[1].set_title('2.4 boundaries (green=top, RED=bottom scallops) + peaks')
for a in ax: a.axis('off')
plt.tight_layout(); plt.show()
"""))

cells.append(md(
"""## 2b. MANUAL feature anchors (optional, high value)

Auto scallop-peaks fix the boundaries, but interior letters are tiny so even small residual drift
matters. Add hand-read correspondences for interior visual features (dark artifacts, the top-left
wave, right corners) to force a cleaner match. Use the GRID below to read pixel coords in the
W=1400 preview frame, then fill `MANUAL_ANCHORS` as (src_x, src_y, dst_x, dst_y) where
src = 2.4 (SOURCE) and dst = 7.91-flipped (TARGET). These are added to the TPS control points."""))

cells.append(co(
"""# read coords off these grids (ticks every 100 px). SOURCE=2.4 (top), TARGET=7.91 (bottom).
for name, tx, Hh in [('2.4 SOURCE (read src_x,src_y)', tex24, H24), ('7.91 TARGET (read dst_x,dst_y)', tex79, H79)]:
    plt.figure(figsize=(22, max(3, 22*Hh/W)))
    plt.imshow(tx, cmap='gray')
    plt.xticks(range(0, W, 100)); plt.yticks(range(0, Hh, 100))
    plt.grid(True, color='cyan', alpha=0.45, linewidth=0.5)
    plt.title(name); plt.tight_layout(); plt.show()

# (src_x, src_y, dst_x, dst_y) in the W=1400 preview frame. seeded with the auto-matched
# middle-left dark artifact; ADD the features you can see (verify/adjust the seed too):
MANUAL_ANCHORS = [
    (208, 220, 197, 234),   # middle-left dark artifact (auto-matched; verify)
    # (src_x, src_y, dst_x, dst_y),   # second mid-left artifact
    # (src_x, src_y, dst_x, dst_y),   # top-left 'wave'
    # (src_x, src_y, dst_x, dst_y),   # right top corner
    # (src_x, src_y, dst_x, dst_y),   # right bottom corner
]
print(len(MANUAL_ANCHORS), 'manual anchors')
"""))

cells.append(md(
"""## 3. TPS warp (scipy RBF) at high resolution

Detection ran at W=1400 (reliable). SCALE the peak anchors to a higher working width `WW` so warped
letters are crisp (re-detecting at high res is unstable). Anchors: top peaks + bottom peaks + a
mid-height point per bottom-peak column + 4 corners. Fit dst(7.91)->src(2.4) TPS for x and y,
evaluate on the 7.91 grid -> mapx/mapy."""))

cells.append(co(
"""WW = 3600                       # hi-res warp width; raise for crisper letters (slower RBF)
sc = WW / W
H79h, H24h = int(round(H79*sc)), int(round(H24*sc))

src, dst = [], []
for x2, x7 in zip(pb24, pb79): src.append([x2*sc, bot24[x2]*sc]); dst.append([x7*sc, bot79[x7]*sc])
for x2, x7 in zip(pt24, pt79): src.append([x2*sc, top24[x2]*sc]); dst.append([x7*sc, top79[x7]*sc])
for x2, x7 in zip(pb24, pb79):
    src.append([x2*sc, (top24[x2]+bot24[x2])/2*sc]); dst.append([x7*sc, (top79[x7]+bot79[x7])/2*sc])
for cx in (0, WW-1):
    src += [[cx,0],[cx,H24h-1]]; dst += [[cx,0],[cx,H79h-1]]
for sx, sy, dx, dy in MANUAL_ANCHORS:      # hand-read interior feature correspondences
    src.append([sx*sc, sy*sc]); dst.append([dx*sc, dy*sc])
src, dst = np.array(src,float), np.array(dst,float)
print('TPS landmarks:', len(src), '(incl', len(MANUAL_ANCHORS), 'manual)')

fx = RBFInterpolator(dst, src[:,0], kernel='thin_plate_spline', smoothing=1.0)
fy = RBFInterpolator(dst, src[:,1], kernel='thin_plate_spline', smoothing=1.0)
gy, gx = np.mgrid[0:H79h, 0:WW]
pts = np.column_stack([gx.ravel(), gy.ravel()]).astype(float)
mapx = fx(pts).reshape(H79h, WW).astype(np.float32)
mapy = fy(pts).reshape(H79h, WW).astype(np.float32)

tex24h = cv2.resize(tex24, (WW, H24h))
tex79h = cv2.resize(tex79, (WW, H79h))
tex24_w = cv2.remap(tex24h, mapx, mapy, cv2.INTER_LINEAR, borderValue=0)
plt.figure(figsize=(22,7))
plt.imshow(np.dstack([tex79h, tex24_w, np.zeros_like(tex79h)]))
plt.title('overlay G=7.91(flip) R=2.4-warped  (scallops + interior should agree)'); plt.axis('off'); plt.show()
"""))

cells.append(md(
"""## 4. Warp the ink labels + letter-clarity test

Apply the SAME map to the 2.4 ink labels. Overlay on the 7.91 sheet (where they will be used) and
show a zoomed crop so letter shapes are inspectable. TPS is smooth, so coherent Greek letters
should survive. Compare the crop against the 2.4 reference (which has visible letters)."""))

cells.append(co(
"""inkb = cv2.resize((ink24>60).astype(np.uint8)*255, (WW, H24h), interpolation=cv2.INTER_NEAREST)
ink_w = cv2.remap(inkb, mapx, mapy, cv2.INTER_NEAREST, borderValue=0)
inside = float(((ink_w>0) & (cv2.resize(sheet79,(WW,H79h))>0)).sum()) / max((ink_w>0).sum(),1)
print('warped ink inside 7.91 sheet:', round(inside,3))

over79 = np.dstack([tex79h,(tex79h*0.35).astype(np.uint8),np.maximum((tex79h*0.35).astype(np.uint8),ink_w)])
plt.figure(figsize=(22,7)); plt.imshow(over79); plt.title('warped 2.4 ink (magenta) on flipped 7.91'); plt.axis('off'); plt.show()

CX0, CX1 = WW//3, WW//3 + 1200
crop_w = over79[:, CX0:CX1]
ref24  = np.dstack([tex24h,(tex24h*0.35).astype(np.uint8),np.maximum((tex24h*0.35).astype(np.uint8),inkb)])[:, CX0:CX1]
fig, ax = plt.subplots(2,1, figsize=(20,9))
ax[0].imshow(crop_w); ax[0].set_title('WARPED ink on 7.91 (letters should be clear, not mangled)')
ax[1].imshow(ref24);  ax[1].set_title('REFERENCE: 2.4 ink on 2.4 (baseline clear letters)')
for a in ax: a.axis('off')
plt.tight_layout(); plt.show()
"""))

cells.append(md(
"""## 5. Read-out & bake

- peak counts (cell 2): top+bottom should match between frames after the flip.
- overlay (cell 3): scallops and interior fibres should broadly agree.
- letter crop (cell 4): warped Greek letters must look like the 2.4 reference, not torn/folded.

If good: raise `WW` toward full res, recompute mapx/mapy, apply to the full-res 2.4 ink ->
`inklabels/<7.91_id>.png` -> erode. The 7.91 volume is then the training target for the diagnostic.
**No training until the user confirms the letters are clean.**
"""))

nb = {"cells": cells,
      "metadata": {"kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
                   "language_info": {"name": "python", "version": "3.12"}},
      "nbformat": 4, "nbformat_minor": 5}
json.dump(nb, open(r'warp_exploration.ipynb', 'w', encoding='utf-8'), indent=1)
print('wrote warp_exploration.ipynb with', len(cells), 'cells')
