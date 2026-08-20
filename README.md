# PHerc0139 Ink Detector

Binary tile-level ink detection on 9.362 µm / 113 keV CT scans of Herculaneum papyrus scrolls.
Current production path: **nnunet3d_lcndz** — a 3D nnU-Net-style encoder/decoder with a raw + LCN + depth-gradient stem, optional learned surface attention, and optional attention-MIL / spatial SupCon integrations.

---

## Quick start

```powershell
# activate venv
.venv\Scripts\Activate.ps1

# run training (all config is driven by utils/config.py)
python train.py -n "experiment_name"

# run the current integration sweep
python campaign_archs_9.py --dry-run
python campaign_archs_9.py

# compute/cache normalisation stats (needed once per new zarr)
python precompute_norm.py --scroll-id 20260206000001

# annotate readability probe windows
python roi.py
```

---

## Training data

Most fragments come from **PHerc0139** (Herculaneum scroll, 9.362 µm voxels, 113 keV, 1.2 m detector distance, raw volume ID `20250728140407`). The training set is **18 fragments**: 15 PHerc0139 fragments, 1 PHerc0814 segment (seg46527), 1 PHerc0500P2 segment (500P2_front), 1 PHerc1667 segment (w013).

| ID | Fragment | Zarr shape (D,H,W) | Mask valid frac | Split |
|---|---|---|---|---|
| `20250223000000` | **w059** | (28, 7220, 10020) | 0.295 (1.1 µm overlap band) | vertical (left 75% train) |
| `20260115000001` | **w056** | (28, 7161, 9721) | 0.866 | horizontal (top 50% train) |
| `20260206000001` | **w047** | (28, 5821, 8421) | 0.402 (1.1 µm overlap band) | vertical (left 75% train) |
| `20260115000000` | **w044** | (28, 6021, 8141) | 0.882 | horizontal (top 80.55% train) |
| `20260210000000` | **w058** | (28, 7500, 9880) | 0.841 | (left 75% train / right 25% valid) |
| `20260227000000` | **w052** | (28, 7700, 9760) | 0.880 | (left 75% train / right 25% valid) |
| `20260318000000` | **w049** | (28, 5660, 9400) | 0.879 | (left 75% train / right 25% valid) |
| `20260325000000` | **w046** | (28, 5980, 8260) | 0.872 | (left 75% train / right 25% valid) |
| `20260108000000` | **w041** | (28, 6200, 8020) | 0.863 | (left 75% train / right 25% valid) |
| `20250831000000` | **w040** | (28, 6400, 7980) | 0.851 | (left 75% train / right 25% valid) |
| `20260302000000` | **w039** | (28, 8560, 7720) | 0.622 | (left 75% train / right 25% valid) |
| `20260306000000` | **w038** | (28, 6200, 7440) | 0.844 | (left 75% train / right 25% valid) |
| `20260310000000` | **w037** | (28, 6140, 7200) | 0.838 | (left 75% train / right 25% valid) |
| `20260303000000` | **w034** | (28, 7040, 7720) | 0.85 | (left 75% train / right 25% valid) |
| `20260317000000` | **w035** (2026-08-12) | (28, 5820, 5240) | TBD | (left 75% train / right 25% valid) |

**PHerc0814 (2026-07-22)** — different scroll, horizontal split (top 75% train / bottom 25% valid):

| ID | Fragment | Zarr shape (D,H,W) | Mask valid frac | Eroded ink frac |
|---|---|---|---|---|
| `20260226000000` | **seg46527** (PHerc0814) | (28, 2180, 3560) | 0.565 (content bbox 2110×3480) | 0.032 (in-mask) |

**PHerc0500P2 (2026-08-07)** — different scroll, same 9.362 µm / 113 keV / 1.2 m scan parameters as PHerc0139. Horizontal split (left 60% train / right 40% valid):

| ID | Fragment | Zarr shape (D,H,W) | Mask valid frac | Eroded ink frac |
|---|---|---|---|---|
| `20250628074500` | **500P2_front** (PHerc0500P2) | (28, 6280, 3580) | 0.559 | 0.014 (in-mask) |

**PHerc1667 (2026-08-13)** — different scroll and scan physics; a pre-rendered 2.399 µm / 78 keV surface volume was converted to an isotropic ~9.5 µm training zarr. Vertical split (left 75% train / right 25% valid):

| ID | Fragment | Zarr shape (D,H,W) | Mask valid frac | Eroded ink frac |
|---|---|---|---|---|
| `20240304141531` | **w013** (PHerc1667) | (28, 10400, 4975) | 0.880 | 0.028 (in-mask) |

**w035** labels are downloaded separately: `python download_w035_labels.py` (1.129 µm / 59 keV source, same as all other PHerc0139 fragments). Assemble zarr via `python assemble_training_segments.py --only w035` (mask generation requires the zarr; re-run label script afterwards to apply it). Edit `inklabels/20260317000000.png` and regenerate eroded labels with `python download_w035_labels.py --erode-only`.

The PHerc0500P2 fragment is notable for its **crystal-clear inklabels** derived from a high-resolution 2.215 µm / 111 keV scan. The 2.215 µm ink detection TIF (shape 26440 × 15060) was resized to the 9.362 µm zarr frame at a 4.21× scale ratio, thresholded at 0.55 (140/255), and eroded with a 3×3 kernel (12 iterations) to produce the training labels. Split changed from horizontal to vertical (2026-08-11) for campaign_archs_7 single-scroll isolation testing. Edit `inklabels/20250628074500.png` then regenerate the eroded version with `python download_p500p2_labels.py --erode-only`. Assemble via `python assemble_training_segments.py --only 500P2_front`.

All **18** are wired into `DEFAULT_SCROLLS` in `utils/config.py`. Ink footprint = fraction of the frame with ink label > 0.

The masks for **w059** and **w047** are intersected with the 1.1 µm ink-detection footprint (ROI2). The **new 10 use the full 9.4 µm papyrus footprint** (not intersected), so ring negatives near the labeled band could in principle fall on un-scanned surface; in practice the ring hugs the ink so this is minor. The full-surface footprint is recoverable directly from the zarr (`z[mid] > 0`); no separate `_full9um.png` is stored.

Ink labels (1.129 µm source, 59 keV) live in `inklabels/` (continuous 0–255 ink probability) and `eroded_inklabels/` (binary, conservative — what training uses for ring negatives; new-fragment eroded fraction ≈ 0.02–0.04).

**seg46527 (PHerc0814) caveat:** only `eroded_inklabels/20260226000000.png` and `masks/20260226000000.png` are present — there is no non-eroded `inklabels/20260226000000.png`. Training with `ring_label_source='original'` (the twostage default) will log a warning and fall back to the eroded map for the ring boundary; `ring_label_source='eroded'` (isolation campaign) uses it directly. Norm stats are cached in `norm_cache.json`.

### Holdout sanity fragment — w055 (NOT trained on)

**w055** (`20251226000000`, PHerc0139, 9.362 µm) is assembled exactly like the training fragments (zarr + mask + 1.1 µm inklabel) but is **deliberately excluded from `DEFAULT_SCROLLS`**. It is a pure hallucination check: the model never sees it during training, so if inference on w055 does **not** reproduce its known 1.1 µm text, we know the model is hallucinating rather than genuinely detecting ink. Assemble it with `python assemble_training_segments.py --only w055`.

---

## Test segments


Four VC3D-grown patches are configured as default test targets (`test_scroll_ids` in `utils/config.py`): PHerc0813 ×1, PHerc0211 ×1, PHerc1203 ×1, PHerc1447 ×1. Test figures are generated when `test_int` fires (currently set to 9999 — disabled until a sufficiently good model is found). The visualizer loads each segment sequentially with CUDA cache cleared between renders to keep VRAM bounded for the larger segments.

### Segment 1 — PHerc0813 updated patch (2026-08-18)

| | |
|---|---|
| Segment name | `auto_grown_20260814140748456` |
| Scroll Source | PHerc0813 (9.362 µm / 113 keV / 1.2 m, raw volume `20250821151723`)
| Zarr ID | `20260814140748` |
| Zarr shape | (28, 5081, 5701) |
| Area | **33.31 cm²** |
| max_gen | 1359 (VC3D growth iterations) |
| Mask valid frac | TBD |
| tifxyz grid | 495 × 495 vertices (cropped to 255×286 valid region) |
| tifxyz bbox | x 29,468–57,955 µm, y 35,457–58,178 µm, z 45,994–84,581 µm (raw-volume voxel coords at 9.362 µm/vox) |
| tifxyz scale | 0.05 cm per grid step |
| tifxyz location | `~/.VC3D/remote_cache/open_data/projects/paths/auto_grown_20260814140748456/` |
| Notes | Replaces `auto_grown_20260716083545968`; much larger area (33 cm² vs 3 cm²), max_gen 1359 vs 175 | 

### Segment 2 — PHerc0211 large merged patch (2026-08-05)

| | |
|---|---|
| Segment name | `auto_grown_20260717193517520_0_1_2_3_4_merged` |
| Scroll Source | PHerc0211 (9.362 µm / 113 keV / 1.2 m, raw volume `20250821151803`)
| Zarr ID | `20260717193517` |
| Zarr shape | (28, 7181, 6501) |
| tifxyz grid | 360 × 326 vertices |
| Mask valid frac | ~0.72 |
| Notes | **Replaces** previous segments `auto_grown_20260717193517520` and `auto_grown_20260719202304218`. Combines 5 patches (0,1,2,3,4) into a single large surface. Re-rendered 2026-08-08 from updated merged tifxyz. |

### Segment 3 — PHerc1203 patch (2026-07-20)

| | |
|---|---|
| Segment name | `auto_grown_20260720090842117` |
| Scroll Source | PHerc1203 (9.362 µm / 113 keV / 1.2 m, raw volume `20250820131727`) |
| Zarr ID | `20260720090842` |
| Zarr shape | (28, 15921, 15921) |
| BBox | (4035×4455) |
| Area | **7.90 cm²** |
| max_gen | 345 (VC3D growth iterations) |
| Mask valid frac | 0.047 (sparse strip within large bounding box; content bbox 4035×4455) |
| tifxyz grid | 661 × 661 vertices |
| tifxyz location | `~/.VC3D/remote_cache/open_data/projects/paths/auto_grown_20260720090842117/` |
| Notes | First PHerc1203 segment; different scroll entirely from training data |

### Segment 4 — PHerc1447 large patch (2026-07-22)

| | |
|---|---|
| Segment name | `20250703034159` (editable mesh under `20250521151220_editable`) |
| Scroll Source | PHerc1447 (**8.640 µm** / 116 keV / 1.2 m, raw volume `20250521151220`, shape 24297×8343×8343) |
| Zarr ID | `20250703034159` |
| Zarr shape | (28, 6592, 8630) |
| BBox | (6264×8318) |
| Area | **51.27 cm²** (largest test segment) |
| max_gen | 638 (VC3D growth iterations) |
| Mask valid frac | 0.700 (dense; content bbox 6264×8318) |
| tifxyz grid | 360 × 471 vertices (cropped to valid from a 6203×6203 grid) |
| Render | 8.640 µm source upsampled ×18.36 to the 9.362 µm training frame; 28 layers, normal-step 1.0, crop-valid margin 8 |
| tifxyz location | `~/.VC3D/remote_cache/open_data/segments/PHerc1447/20250521151220_editable/20250703034159/` |
| Notes | Large strip, first PHerc1447 segment. Source scan is a coarser 8.64 µm volume, hence the upsample. |

### Segment 5 — PHerc0826 merged patch (2026-08-08)

| | |
|---|---|
| Segment name | `auto_grown_20260723112922652_merged` |
| Scroll Source | PHerc0826 (9.362 µm / 113 keV / 1.2 m, raw volume `20250821151701`, shape 16920×8169×8169) |
| Zarr ID | `20260723112922` |
| Zarr shape | (28, 9481, 4521) |
| tifxyz grid | 475 × 227 vertices (valid fraction 0.546) |
| Notes | New scroll entirely. Same scan parameters as PHerc0139 (9.362 µm, 113 keV). Assemble via `python assemble_test_segments.py`. |

All five are rendered from their VC3D tifxyz mesh against their respective raw CT volume. The tifxyz format stores a 2D grid of 3D raw-volume voxel coordinates — the actual CT intensities are fetched at render time via `assemble_test_segments.py`.

---

## Model

**nnunet3d_lcndz** (`utils/model.py`) is the only active architecture kept in the repo.

1. **Stem inputs** — the network ingests `[raw, lcn, dI/dz]` per tile window. LCN removes the slow papyrus baseline; the depth gradient makes interface structure explicit.
2. **3D encoder/decoder** — a compact nnU-Net-style 3-level encoder/decoder preserves dense spatial detail while still mixing information across depth.
3. **Optional learned surface attention** — `DepthSurfaceAttn` can amplify slices that look surface-proximal before the deeper encoder blocks.
4. **Per-voxel logits** — the decoder emits a `(B, 1, D, H, W)` voxel map.
5. **Bag aggregation** — default aggregation is learnable log-sum-exp; current sweeps also test gated attention-MIL with entropy regularization.
6. **Auxiliary representation** — when spatial SupCon is enabled, the bottleneck is pooled to a 256-d embedding and passed through a small projection head.

The current campaign (`campaign_archs_9.py`) keeps the backbone fixed and only varies integrations that still fit this path directly: soft augmentation, flip-consistency regularization, GCE + asymmetric label smoothing, spatial SupCon, and learned surface attention.

---

## Files

| File | Purpose |
|---|---|
| `train.py` | Current training loop for the cleaned nnUNet path. Only accepts `-n experiment_name`; all other config comes from `utils/config.py` or a campaign file. |
| `utils/config.py` | Current config surface for the nnUNet path: scroll list, tile/depth/context settings, augmentation, SupCon, TTA consistency, and visualization cadence. |
| `utils/model.py` | Current model definition: `nnunet3d_lcndz`, optional learned surface attention, attention-MIL, and spatial SupCon head. |
| `utils/dataloader.py` | Sparse-label tile dataset, multi-scroll merge, ring-negative mask building, and cached normalization hookup. |
| `utils/visualizer.py` | TensorBoard figures for eval/test/probe rendering on the current sparse-label path. |
| `utils/norm.py` | Shared chunk-aligned normalization cache/compute utility. Used by both the dataloader and visualizer. |
| `utils/training_utils.py` | Optimizer/scheduler factory, BCE/GCE loss builders, checkpoint save helpers, and scalar metrics. |
| `precompute_norm.py` | CLI: `python precompute_norm.py --scroll-id <id>`. Writes to `norm_cache.json`. |
| `roi.py` | Interactive probe ROI picker that writes `probe_rois.json`, now the single source for probe windows. |
| `assemble_training_zarrs.sh` | Downloads and assembles the three training zarrs from S3. See [Assembling zarrs](#assembling-training-zarrs). |
| `campaign_archs_9.py` | Current all-scroll integration sweep for the nnUNet backbone. |
| `old/` | Archived experiments, older campaigns, and retired architecture families. The radical archs6 cluster now lives here. |
| `old/download_surface_zarr.py` | Downloads a pre-rendered OME-Zarr surface volume from S3 (volume or midslice mode). |
| `old/render_9um_surface.py` | Renders a tifxyz mesh against the raw zarr via surface-normal sampling. Used for w047 and test segment. |
| `overlay_2p4_9um.py` | Alignment sanity: hi-res (red, half opacity) over lo-res (green), yellow = overlap. Reports NCC. |
| `test_inference.ipynb` | Standalone inference notebook. Set `MODEL_PATH` + `SCROLL_ID`, Run All → depth panels + MAX figure. |
| `campaign_archs_8.py` | Immediate predecessor sweep used as the comparison point for the current nnUNet family. |

---

## Assembling training zarrs

```bash
bash assemble_training_zarrs.sh [--workers 24]
```

Downloads w044 and w059 (pre-rendered OME-Zarr on S3 via `old/download_surface_zarr.py`) and renders w047 from its tifxyz mesh (`old/render_9um_surface.py`). After running, restrict the w047 mask to the 1.1 µm overlap band (see the snippet in the script comments). Then cache normalization stats:

```bash
python precompute_norm.py --scroll-id 20260115000000 --scroll-id 20250223000000 --scroll-id 20260206000001 --scroll-id 20260115000001
```

**Adding a new segment:**
1. Grow the segment in VC3D. The tifxyz lives at `~/.VC3D/remote_cache/open_data/projects/paths/<uuid>/`.
2. Render: `python old/render_9um_surface.py --mesh-dir <tifxyz-dir> --vol-base <S3-zarr-url/0> --vol-shape Z,Y,X --layers 28 --out-zarr ves_zarrs2/<id>.zarr --out-id <id>`.
3. Compute norm: `python precompute_norm.py --scroll-id <id>`.
4. Download ink labels from S3 (`<segment>/ink-detection/*.tif`), resize to the training frame, save to `inklabels/<id>.png`. Manually create `eroded_inklabels/<id>.png`.
5. (Optional) restrict the mask to the labeled footprint using `overlay_2p4_9um.py` + the morph-close snippet in `assemble_training_zarrs.sh`.
6. Add a `ScrollConfig` entry to `utils/config.py`'s `DEFAULT_SCROLLS`.

---

## Test inference notebook

`test_inference.ipynb` — set `MODEL_PATH`, `SCROLL_ID`, `ARCH`, `TILE_SIZE`, `DEPTH` in the CONFIG cell and Run All.

Loads the checkpoint, opens `ves_zarrs2/<SCROLL_ID>.zarr`, uses cached normalization, runs the real `predict_tiles` pipeline, and renders:
- One row per depth window (prediction only, YlGnBu colormap)
- A **MAX across all depths** collapsed panel + gold inklabel overlay
- Optional PNG save (`SAVE_PNG` variable)

Default: set `MODEL_PATH` to your best checkpoint (latest: `runs_lcn/` from the LCN sweep) on the PHerc0813 test segment (`20260716083545`).

---

## Historical notes

See `old/KNOWLEDGE.md` for the full research log: pre-2026 campaigns (arch10/18/28, scroll1/4 at 7.91 µm), 2.4 µm vs 9.4 µm investigation, scroll4 teacher-zarr work, and the 2026-07 PHerc0139 9.362 µm campaign series (triple-scroll sweep, LCN win, w056 addition).

## CATCH UP TO SPEED:
The latest results:
campaign 5 was our original cnn baseline. capable of overfitting on small amounts of data, but fell apart with larger scrolls.
campaign 7 8 and 9 proved that our nnunet is king; it can overfit no matter the amount of data we throw at it.
campaigns 11 and 12 proved that context is king (up to 192ds2, any higher is useless) and tested a bunch of further refinements (e.g. ibn, fda, etc.)

The *crux of the issue*: the model can overfit but cannot generalize. 

Note that per patch, we label ~10 letters (out of ~50 actual letters). The model can overfit very easily onto all 10 letters across all 18 patches, but it has NEVER found a SINGLE LETTER not in it's immediate training corpus, even those right next to labelled letters. It predicts what basically amounts to illegible noise in every location except those perfectly labelled. 

Our labels are uncertain, hence the ring setup (we strictly want to avoid accidentally labelling ink as papyrus). Previous agents have contended that this is a geometry issue: this is not true. Each tile only knows the immedaite information of it's context; it doesn't see that it's 'surrounded' by a ring of papyrus or anything like that.

Incredibly, no amount of regularization we throw at the model hinders it's progress. Campaign 11 and 12 are evidence of this, but we have done testing outside of that. l1, l2, adamw, and photometric are all useless. tta consistency helped a little. the spill reduction helped a little (it forces the model to focus on a handful of depths, as we intend). the ones that actually have an effect: depth mixing (catastrophic failure, model absolutely requires depth preservation) and cutout (model eventually overfits again, but the problem space is fundamentally changed and it simply performs much worse).

Campaign 13 is out attempt at faster iteration plus a smaller dataset. Note that scroll 20240304141531 hold by far the largest amount of labels, and we are slightly more confident than usual about it's label quality. For this smaller dataset (notably not regularized by the larger training corpus and cross scroll information): every test stil overfits, hard. 

the baseline (13_baseline_18_21-12-15) quickly climbs to 0.8 train pr auc after just 10 epochs, if i trained it more it would likely reach >0.9. The valid pr auc climbs, then stalls at 0.7: not catastrophic overfitting, but looking out our inference, again the model is failing to predict any letter not directly in its training corpus. 

in fact, a pr_auc of 0.8 means that the model fails on quite a number of the training corpus as well, but this is not the issue at hand. it's expected that the model cannot learn the representation on this smaller subset, this is an observed and assured phenomenon.. I know that if I add more data, the model will actually fit better to the training subset, but I just want faster iteration right now.

jitter and jitter large were uesless. skip drop and skip drop hard (0.6!!) were both useless. depth profile was useless. strong spill was useless. no dz was useless. 'useless' means that it basically followed the baseline curve for as long as it trained, witth minimal deviation.

The only two things with any punch: dropout took the train pr_auc from 0.8 to 0.7; the valid_pr auc matched the old 0.7. depth jitter took the train pr_auc to 0.6, and the valid pr_auc matched the old 0.7. These are actually very good signs and I feel silly not trying the most basic regularization earlier. I'm considering slamming these together and training the whole training corpus. 

the ring negative does not (in the vast vast majority of cases) ever eat into a positive letter, I've made sure of it. The close and gap were chosen to allow for the model to have wiggle room for it's ink label, while not eating into other letters. Note that the model does perform better if we label the boundary perfectly, but given the uncertainty of our labels, doing so is currently actively detrimental.

adding pure papyrus data does not help in the slightest. the model most needs to learn the differentiation point between ink and papyrus; closer labels help but again we cannot dot hat rightnow.

The ring is how we generate positive and negative labels; the pr_auc metric is over each 16x16 prediction at the center of the chosen 192x192 tiles. 

the papyrus actually 'undulates' up and down throughout the mesh (a necessary evil of segmentation), hence why we have tried and added depth aware components (learned surface, dl/dz, etc.). I sincerely doubt the model is predicting based on the same depth every time. Not to mention, the previous iteration of our model had even more depth information but failed to capitalize on it.

Let's also talk about the metrics. I don't understand your logic - how can it be blind to the failure mode? there are labels in both the train and validation section. if the model cannot learn the letters on the held-out region (learning letters = scoring well, because it must label the letter correctly and isolate it), then it will score badly on validation. Furthermore, note that this is actually almost a *meaningless* distinction: because of the way we've set up our model, *everything that is in the training corpus* is functionally a held-out section. That's why I have the eval figure at the end: I'm literally looking at the model output and seeing that it can learn the vast majority of the training letters, yet nothing *between* the training letters. Think of it this way: we have word spelled 'h e l l o'. We are unsure of the 2nd character, so we label 'h . l l o'. And we want a validation character, so then we take the o. We can look at both the performance on the 'e' and the 'o' (not trained on, we only trained on h l l) to see how the model does.

# multitile prediction head (nnunet3d_lcndz)

behind flag `config.model.multitile` (default False). single-tile path stays byte-identical.

## what it does
predicts the 32px center as a 4x4 grid of 8px sub-tiles (16 targets/window) instead of one
16px score. denser gradient without going fully dense.
- WINDOW GATING (train only, shuffle=True): keep a window only if its 32px center overlaps the
  RING/training mask (`_mt_window_touches_ring` in _gen_tile_coords). the dataset's `self.mask`
  IS the ring mask when ring_negatives is on. negatives = papyrus sub-tiles inside those windows.
- LABELS (`_fetch_label_mt`): each 8x8 sub-tile = 1 if .any() ink else 0 (ring ignored).
- MASK (`_fetch_mask_mt`): sub-tile valid ONLY if fully inside the SCROLL mask (papyrus bounds),
  which is threaded separately as `scroll_mask` (self.scroll_mask; falls back to self.mask when
  ring off). ring mask != scroll mask. train.py loss divides by mask.sum(); metrics drop masked
  sub-tiles (guarded by labels.shape[1]>1 so single-tile stays byte-identical).

## aggregators (multitile)
- default: per-cell log-sum-exp (`_multitile_aggregate`), param-light (just lse_r).
- attn_mil=True: per-sub-tile gated attention-MIL (`_multitile_attn_aggregate`) folds the 4x4
  grid into the batch dim, one attn_mil call pools each 8px bag; sets last_attn_entropy_loss.
  campaign arm `multi16_attn` uses this (attn_mil=True, attn_entropy_weight=0.03).

## config knobs
- model.multitile (bool), model.multitile_subtile=8, model.multitile_grid=4
- data.multitile_train_step (dataloader window stride px, default 16)

## CRITICAL invariant: cell ordering must match across 3 files
flat index = iy*grid + ix  (iy = y-block/row, ix = x-block/col, row-major)
- model.py `_multitile_aggregate`: reshape (B,D,ny,suby,nx,subx)->permute->(B, ny*n+nx, ...)
- dataloader.py `_fetch_label_mt`: out[iy*n+ix]
- visualizer.py `_process_chunk_mt._scatter`: cell (base_y+iy, base_x+ix)

## centering
32px label/pred region centered on the 16px tile: top-left offset = (tile-32)//2 = -8.
model crops 16 feat px center (=32 input px at ds2). all three agree.

## eval/visualizer
multitile output map is at sub-tile (8px) resolution, not tile(16px). predict_tiles builds
sum/count accumulators (windows overlap 4x at stride 16) and averages. TTA flips/rot must be
UNDONE on the (B,n,n) grid before averaging (grid dims 1=y,2=x). `_compute_tile_maps` takes a
`tile` arg (pass out_tile=8); add_evaluation_figures uses `_out_tile` for label maps + split idx.

## campaign
campaign_archs_14.py: single vs multi16 vs multi32 on w013 only. multitile arms set attn_mil=False
(the mt aggregator replaces the bag scorer; last_attn_entropy_loss stays None so no stray loss).
