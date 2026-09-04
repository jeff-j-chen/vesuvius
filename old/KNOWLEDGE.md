# Vesuvius Ink Tile Detector

This repository is the working folder for a Vesuvius Challenge entry focused on extracting text from carbonized Herculaneum scrolls with machine learning.

Core idea in this repo: do binary tile-level ink detection on 3D volume chunks instead of dense pixel reconstruction.

- Input unit: 3D tile (depth x 32 x 32)
- Label rule: if any ink exists in that 32 x 32 tile, label tile as ink
- Main model: compact 3D CNN + CBAM attention (not a U-Net)
- Motivation: simpler architecture, faster iteration, easier interpretability, lower compute barrier

This README is intentionally detailed so future-you and future agents can recover context quickly.

## 1) Current Project Status

- Active training script: train.py
- Primary model code: utils/model.py
- Data pipeline: utils/dataloader.py
- Metrics + eval figure generation + hard-mining file generation: utils/visualizer.py
- Hard mining injector/manager: utils/hard_mining.py
- Scroll4 standalone visualizer script: scroll4_vis.py
- Notebook for manual sanity checks/comparisons: comparer.ipynb
- Notebook for hard example overlay inspection: visualize_hard_examples.ipynb
- vis.ipynb is currently broken (details in Known Issues)

Current repository state observed during audit:

- README.md was empty
- get_data.sh is modified in working tree
- model checkpoints and hard mining files already exist

## 2) Data Inventory and Segments

### 2.1 Active local zarrs

Training zarr root (SSD, fast I/O):

- /media/jeff/SSD_2/ves_zarrs2/

Additional zarr storage (Seagate, larger capacity):

- /media/jeff/Seagate/ves_zarrs2/

Present zarrs:

- 20230702185753.zarr  (SSD)
- 20230827161847.zarr  (SSD)
- 20231210132040.zarr  (SSD)
- 20230709155141.zarr  (Seagate)

Observed zarr metadata:

- 20230702185753.zarr
	- shape: (64, 13513, 17381)
	- chunks: (8, 32, 32)
	- dtype: uint16
- 20230827161847.zarr
	- shape: (64, 9163, 5048)
	- chunks: (8, 32, 32)
	- dtype: uint16
- 20231210132040.zarr
	- shape: (64, 8790, 12122)
	- chunks: (8, 32, 32)
	- dtype: uint16
- 20230709155141.zarr  (Scroll 2 segment)
	- shape: (64, 2806, 8499)
	- chunks: (8, 32, 32)
	- dtype: uint16
	- source: paths/20230709155141/layers/ (surface-extracted segment, 65 TIF layers)

### 2.2 Segment notes (human strategy context)

- 20230827161847
	- small sample from scroll 1, 7.91um, 54keV, 20.8294 cm^2
	- roughly half annotated
	- strategic interest in lower region where text may be latent/unannotated
- 20230702185753
	- very large scroll 1 section, 7.91um, 54keV, 97.9346 cm^2
	- main training workhorse
- 20231210132040
	- small scroll 4 section, 7.91um, 53keV, 8.98639 cm^2
	- high-value target region (bottom-left focus, possible pi character hypothesis)
- 20230709155141
	- Scroll 2 (PHercParis3) surface-extracted segment, 7.91um, 54keV
	- 64 depth layers, spatial area 2806 × 8499
	- no ink labels yet; use as inference/exploration target

### 2.3 Labels and masks

Directories:

- inklabels/ (original labels)
- eroded_inklabels/ (eroded labels used by training)
- masks/ (scroll-vs-air masking)

Observed IDs present in all three folders include:

- 20230702185753
- 20230827161847
- 20230929220926
- 20231005123336
- 20231007101619
- 20231012184420
- 20231016151002
- 20231022170901
- 20231031143852
- 20231106155351
- 20231210121321
- 20231210132040
- 20231221180251

Normalization cache currently includes:

- 20230827161847
- 20231210132040
- 20231007101619
- 20230702185753

File: norm_cache.json

## 3) Data Acquisition Pipeline

This describes how to go from a remote Vesuvius scroll URL to a locally usable zarr.

### 3.1 Find the right source endpoint

Each volpkg on dl.ash2txt.org exposes two kinds of data:

- `volumes/` — raw CT scan slices, thousands of layers spanning the entire scan depth
  (NOT what the training pipeline uses — these are the vertical cross-sections of the scroll)
- `paths/` — surface-extracted segments (~65 layers, the ~0.5mm depth window of unwrapped papyrus)
  (THIS is the correct source for training and inference)

Browse available segments:

```bash
curl https://dl.ash2txt.org/full-scrolls/Scroll2/PHercParis3.volpkg/paths/
```

Check layer count for a specific segment before downloading:

```bash
curl https://dl.ash2txt.org/.../paths/<segment_id>/layers/ | grep -c '\.tif'
```

Expect ~65 TIF files (00.tif through 64.tif).

### 3.2 Download the layer TIFs

```bash
mkdir -p /path/to/raw_<segment_id>/layers
cd /path/to/raw_<segment_id>/layers
wget -r -np -nd -A "*.tif" \
    "https://dl.ash2txt.org/.../paths/<segment_id>/layers/"
```

### 3.3 Convert TIFs to zarr

Use the 3dstreamer converter with the same chunk sizes as all existing zarrs:

```bash
cd /media/jeff/Seagate/vesuvius-3dstreamer
python tools/converter.py \
    /path/to/raw_<segment_id>/layers \
    /path/to/ves_zarrs2/<segment_id> \
    --z_chunksize 8 --y_chunksize 32 --x_chunksize 32 --max_workers 4 --verify
```

This produces `<segment_id>.zarr` with:
- shape `(64, H, W)` — 64 depth slices, full spatial extent of the segment
- chunks `(8, 32, 32)` — matches tile_size=32, depth=8 in config
- dtype `uint16`

Delete the raw TIF folder after successful conversion.

### 3.3b Unified builder (recommended): build_scroll_zarr.py

`build_scroll_zarr.py` is the single entry point that replaces the older one-off
`reconstruct_scroll{3,4}_7um.py` / `reconstruct_scroll4_patch.py` scripts. It streams a scroll
fragment straight into our zarr format (`(64,H,W)`, chunks `(8,32,32)`, uint16, `zarr_format=2`)
without ever holding the whole volume in RAM, and writes `masks/<id>.png` alongside it.

It handles two source types:

- **`volpkg`** — a dl.ash2txt surface segment (`paths/<seg>/layers/{00..64}.tif`). Geometry
  (width, height, data offset) and the layer count are AUTO-DETECTED from the layer-0 TIFF
  header, so any segment works without editing code. Downloads are hardened (curl
  `--max-time`/`--retry`), fetch the 65 layers per y-block in parallel, and are RESUMABLE
  (a `.recon_progress` sidecar + mask checkpoint let a stalled run continue instead of
  restarting). `--flip` horizontally mirrors the frame when it must match flipped labels.
- **`s3patch`** — an S3 open-data surface-volume zarr (2.399um) plus its ink-detection
  prediction tif. Downloads only the chunk files intersecting a bbox, depth-resamples, and
  (when `--ink-key` is given) bakes `inklabels/` + `eroded_inklabels/` via otsu → close →
  de-speckle → erode.

```bash
# named presets (the three fragments already in use)
python build_scroll_zarr.py preset scroll3            # 7.91um goal scroll (PHerc332)
python build_scroll_zarr.py preset scroll4-79         # 7.91um scroll4 w023 (flipped)
python build_scroll_zarr.py preset scroll4-24-patch   # 2.4um scroll4 patch + labels

# ANY volpkg segment — geometry auto-detected, no code edits
python build_scroll_zarr.py volpkg \
    --base-url https://dl.ash2txt.org/full-scrolls/Scroll2/PHercParis3.volpkg/paths/<seg>/layers/ \
    --out-id <seg> [--flip] [--y0 0 --y1 4000] [--workers 8]

# a bbox patch from an S3 surface volume (+ optional ink labels)
python build_scroll_zarr.py s3patch \
    --seg PHerc1667/segments/<...>_flatboi \
    --vol-subpath surface-volumes/<vol>.zarr/0 \
    --ink-key <...>/ink-detection/<...>.tif \
    --out-id <id> --y0 0 --y1 9600 --x0 6144 --x1 16384
```

IMPORTANT — after building a large frame, precompute normalization so the first training run
does not hit the slow in-pipeline norm loop (it reads full z-slices against `(8,32,32)` chunks,
re-reading the whole volume ~8x with millions of tiny I/Os — slow enough to look like a hang):

```bash
python precompute_norm.py --scroll-id <id>   # one chunk-aligned pass -> norm_cache.json[<id>]
```

Note the visualizer eagerly computes norm for EVERY test region (scroll2/scroll3/scroll4) at
init, so precompute those ids too before training with them wired in.

### 3.4 Register the new segment

After creating the zarr:

1. update `utils/config.py` to add the new scroll ID (e.g. `scroll2_id`)
2. if labels and masks exist, add PNGs to `inklabels/`, `eroded_inklabels/`, `masks/`
3. if no labels exist, the segment is inference-only; wire it into a visualizer path
4. normalization is computed automatically on first run and cached in `norm_cache.json`

### 3.5 Cross-resolution label transfer (2.4um -> 7.91um dot-warp)

Some scroll4 sheets were scanned at BOTH 2.4um (78keV, where ink is visible -> our label
source) and 7.91um (53keV, the modality scroll2/scroll3 share, and what we train on). The two
scans were flattened DIFFERENTLY, so their frames do not line up pixel-for-pixel. We bridge them
with a hand-anchored **thin-plate-spline (TPS) warp**: mark matching features with colored dots
on both frames, fit a TPS, and carry the 2.4um ink labels into the 7.91um frame. This has been
the most reliable alignment method we've found.

The whole flow runs on small downscaled slices, so it's cheap. Steps:

**(a) Grab the two comparison slices efficiently.** You do NOT need the full volumes — just one
mid-depth surface slice from each scan, downscaled to a common width (we use 6000).

- 2.4um source texture: pull only the SMALL multiscale level (e.g. level 5 = 32x downsampled)
  of the S3 surface-volume OME-zarr, then take the mid-depth slice. Level 5 is a few hundred MB
  vs hundreds of GB for level 0.

  ```powershell
  # download just level 5 of the 2.4um surface volume
  aws s3 cp `
    "s3://vesuvius-challenge-open-data/PHerc1667/segments/<seg>_flatboi/surface-volumes/<vol>.zarr/5/" `
    "$env:USERPROFILE\Documents\_ves_tmp\<id>_24_l5" --recursive --no-sign-request
  ```
  Then in Python: `z = zarr.open(<dir>); sl = z[z.shape[0]//2]` -> plain `cv2.resize` to width
  6000 (INTER_AREA, NO normalization) -> save `warp_MARK_<id>_2p4_source.png`.

- 7.91um target texture: download the single MIDDLE layer TIFF (layer 32 of `paths/<seg>/layers/`),
  **flip it horizontally** (the 7.91um flattening is mirrored vs the 2.4um one), plain-resize to
  width 6000, plain 16->8 bit (`arr // 256`, NO contrast stretch) -> `warp_MARK_<id>_7p9_target.png`.

  ```powershell
  curl.exe -s --fail --max-time 900 --retry 5 --retry-all-errors `
    "https://dl.ash2txt.org/.../paths/<seg>/layers/32.tif" -o "$env:USERPROFILE\Documents\_ves_tmp\<id>_79_l32.tif"
  ```

  IMPORTANT: use **plain downscaling only** — no percentile/contrast normalization. Keeping the
  raw intensities makes the two frames easier to eyeball-match and avoids introducing artifacts.

**(b) Manually add the dots.** Open both `warp_MARK_*` PNGs in any image editor. For each feature
you can identify in BOTH frames (scallop humps, tears, distinctive fibers), place a dot of the
**same saturated color** on that feature in both images. Use a DIFFERENT palette color for each
correspondence. Supported palette (12 colors): red, green, blue, yellow, magenta, cyan, orange,
purple, pink, teal, brown, violet. A few pixels wide is plenty; the underlying image is grayscale
so any saturated pixel is detected as a dot. Save as `warp_MARK_<id>_2p4_source_dots.png` and
`warp_MARK_<id>_7p9_target_dots.png`. More dots -> tighter alignment (aim for 8-12, spread out).

**(c) Fit the warp** (writes `<tag>_dotwarp_map{x,y}.npy` + a QA overlay):

```powershell
python warp_from_dots.py --id <tag> `
  --src-dots warp_MARK_<id>_2p4_source_dots.png --dst-dots warp_MARK_<id>_7p9_target_dots.png `
  --src-tex  warp_MARK_<id>_2p4_source.png      --dst-tex  warp_MARK_<id>_7p9_target.png `
  --ink-tif  <path to 2.4um ink prediction tif>
```

Check `warp_dots_overlay_<tag>.png`: red (warped 2.4) and green (7.91) should sit on top of each
other (yellow = aligned). If text lines drift, add/adjust dots and rerun.

**(d) Bake + clean up the labels** (two stages, with a manual-correction pause between):

```powershell
# stage A: threshold the 2.4 ink, morph-clean, warp into the 7.91 frame -> editable PNG
python bake_scroll4_79_labels.py --out-id <id> --tag <tag> --ink-tif <2.4 ink tif> `
  --src-mark warp_MARK_<id>_2p4_source.png --ink-thr 99

# -> hand-correct <tmp>/<tag>_warp_edit.png  (paint WHITE=add ink, BLACK=remove)

# stage B: upscale the corrected PNG to the full frame (auto-read from the zarr) + erode
python bake_scroll4_79_labels.py --shrink --out-id <id> --tag <tag>
```

Stage B writes `inklabels/<id>.png` and `eroded_inklabels/<id>.png` (the trainer consumes the
eroded one). The full 7.91um volume zarr + mask come from `build_scroll_zarr.py volpkg ... --flip`
(§3.3b) — the `--flip` matches the horizontally-flipped target frame used here.

## 4) End-to-End Workflow in This Repo

### 4.1 Prepare labels (optional but standard here)

Script: ink_shrinker.py

- Applies morphological erosion to grayscale ink labels
- Default params in script:
	- kernel size: 3
	- iterations: 12

Why erosion is used here:

- tile-level labels are coarse (any-ink tile = positive)
- erosion reduces border spill and over-positive supervision pressure
- helps align objective with your detection strategy

### 4.2 Train tile classifier

Script: train.py

Main flow:

1. Build Config
2. Load datasets via DataManager
3. Compute class weighting by sampling dataset labels
4. Create InkDetector model
5. Train with mixed precision and gradient clipping
6. Validate every epoch
7. Log extensive TensorBoard diagnostics
8. Save best/loss checkpoints and periodic checkpoints
9. Generate hard-mining files + overlays at eval interval

Typical command:

```bash
python train.py -n my_experiment_name
```

TensorBoard:

```bash
tensorboard --logdir=./runs
```

### 4.3 Evaluate and visualize predictions

Two modes are embedded in visualizer behavior:

- Evaluation figures on train/valid region every eval_int epochs
- Test figures every test_int epochs for:
	- full test region of scroll1 segment (currently full spatial region)
	- sliced scroll4 region (y >= 6500, x <= 5000)
- Fixed readability probe figures every 5 epochs for:
	- easy ROI on small scroll 1
	- hard ROI on small scroll 1
	- target pi ROI on scroll 4

Readability probes are intentionally cheaper than full test inference and are meant to provide earlier qualitative feedback without changing the end-of-training role of `test_int`.

Standalone scroll4 visualization script:

```bash
python scroll4_vis.py -m models/best_model_f1.pth
```

### 4.4 Hard mining loop

Hard mining flow is split across visualizer + trainer + hard_mining manager/injector:

- On eval epochs, visualizer writes hard_negs/hard_mining_epoch_<epoch>.jsonl
- Hard negatives: label 0 tiles predicted above hn_cutoff
- Hard positives: label 1 tiles predicted below hp_cutoff
- Later training epochs sample/inject these hard examples into batches

With current defaults:

- eval interval: 20 epochs
- update condition in trainer: epoch % eval_int == 0 and epoch > 5
- this explains existing files:
	- hard_mining_epoch_19.jsonl
	- hard_mining_epoch_39.jsonl

Observed counts:

- hard_mining_epoch_19.jsonl
	- lines: 7994
	- meta: hard_negatives=2641, hard_positives=5352
- hard_mining_epoch_39.jsonl
	- lines: 25295
	- meta: hard_negatives=24533, hard_positives=761

Interpretation:

- by epoch 39, model appears much more over-confident on negatives (lots of false positive confidence), while hard positives shrink

## 4.5) Ring Negatives — Why and How

### What the problem was

Training on the full papyrus mask yields a heavily imbalanced dataset: typically ~10% ink tiles
and ~90% blank tiles. Blank tiles from regions far from any ink stroke are "easy negatives" —
the model trivially learns to suppress them early and then stops learning anything useful.
Worse, blank tiles from unmapped regions may contain actual ink the labelers missed; training
on those as negatives directly contradicts the ink signal.

### What ring negatives do

Instead of sampling from the full blank papyrus area, ring negatives restrict the negative set
to tiles that are ADJACENT TO confirmed ink tiles — the "ring" around the labeled region. This:

1. Keeps the decision boundary tight: the model must distinguish ink from near-ink papyrus,
   which is the hard and meaningful case.
2. Eliminates unlabeled-ink contamination: tiles far from any label are excluded entirely.
3. Dramatically reduces training set size (and therefore epoch time) while concentrating
   gradient signal where it matters most.
4. Consistently beats full-frame training across every campaign search that tested both.

### Implementation

The ring radius is computed by binary search: find the smallest dilation of the ink tile map
such that ring_tiles >= ink_tiles. This guarantees at minimum a 1:1 positive:negative ratio.
In practice the ratio lands at 1.6–1.9:1 for typical datasets:

- 2.4um teacher (T=106, y-split): ink=3,124, ring=5,818, ratio=1.86:1, total=8,942 tiles
- 7.9um w023 (T=32, x-split): ink=44,695, ring=74,757, ratio=1.67:1, total=119,452 tiles

### Ring label source

The `--ring-label-source` flag controls which labels define the ring BOUNDARY (not the
positive training labels, which always come from eroded_inklabels):

- `original`: ring touches any tile that intersects original (unmodified) inklabels.
  Safe: no original ink pixel can enter the ring as a negative.
  decent performer
- `closed`: closes letter holes before ringing, then adds an explicit air gap. Ensures
  tile interiors of large letters don't contaminate the ring with ink-containing tiles.
  worst performer
- `eroded` (legacy): ring built from eroded labels. Previously caused ~20.9% contamination.
  best performer, despite the contamination

**Always use `--ring-label-source eroded` for new runs.**

### Usage

```bash
# add to any train.py invocation:
--ring-negatives --ring-label-source closed

# what to expect in the log at startup:
[ring_negatives] source='closed' tile_radius=2  ink_tiles=3124  ring_tiles=5818  ratio=1.86
```

The ring mask is recomputed at each training run from the current labels and mask, so it
automatically adapts to any changes in inklabels without code changes.

### IMPORTANT: ring applies to training loop AND validation loop, NOT the eval figure

Three separate code paths use masks:

- **Training DataLoader** (every epoch): uses ring mask when ring_negatives=True.
  Gradient only flows through ink + ring-boundary tiles.
- **Validation DataLoader** (every epoch, metrics): also uses ring mask when ring_negatives=True.
  Metrics (F1, AUC, loss) are computed over ring tiles only — matches training distribution,
  avoids thousands of easy blank tiles dominating the average and hiding real signal.
- **Evaluation figure** (add_evaluation_figures, runs at eval_int): ALWAYS uses the full
  papyrus mask regardless of ring_negatives. This is intentional — the figure is a visual
  diagnostic showing model predictions across the ENTIRE cropped scroll region, not just the
  ring. Restricting it to ring tiles would destroy the scroll-level readability visualization.

This distinction is enforced in utils/visualizer.py (eval_mask = self.mask always) and
utils/dataloader.py (valid_mask = ring_mask when ring_negatives=True).

## 5) Model and Training Details


### 5.1 Core config defaults

Source: utils/config.py

- Data
	- zarr_path: /media/jeff/SSD_2/ves_zarrs2/
	- scroll1_id: 20230702185753
	- scroll4_id: 20231210132040
	- tile_size: 32
	- depth: 8
	- d_start: 28
	- d_end: 48
- Dataloader
	- batch_size: 96
	- num_workers: 2
	- data_aug: True
- Training
	- n_epochs: 50
	- lr: 1e-4
	- l1_lambda: 7e-6
	- grad_norm: 0.5
	- patience: 5
	- lr_decay: 0.5
	- save_int: 10
	- eval_int: 20
	- test_int: 50
- Hard mining
	- hn_cutoff: 0.8
	- hp_cutoff: 0.45
	- hm_frac: 0.1

### 5.2 Architecture

Source: utils/model.py

- Input shape per sample: (1, depth, 32, 32)
- 3D conv backbone with CBAM3D attention blocks
- global pooling + MLP classifier down to single logit
- final objective: BCEWithLogitsLoss (with optional pos_weight)

### 5.3 Data augmentation

Source: utils/dataloader.py Transform

- random channel mixing
- random 90/180/270 rotations
- random flips
- gaussian noise
- brightness and contrast perturbation

Augmentation activates after epoch 5 in trainer when config.dl.data_aug is true.

### 5.4 Labeling objective mismatch caveat

By design, supervision is tile-level binary (any ink in tile), not pixel-level segmentation.

This means:

- better scalar metrics may not create sharper character shapes
- model can improve confidence calibration while still looking visually noisy
- high scores can look like global brightening if contrast separation does not improve

This is consistent with your observed pain point.

## 6) TensorBoard Logging and What It Means

Logged classes of signals include:

- scalar train/valid loss + raw masked loss
- accuracy, precision, recall, F1, specificity
- ROC-AUC and PR-AUC
- readability-oriented scalars under `R_M/*` including:
	- local contrast
	- local ranking accuracy
	- recall and partial AUC in the 1% FPR regime
	- top-k precision at ink budget
	- tile ink-fraction correlation
	- spill ratio and readability composite
- confusion matrix figures
- output score histograms
- radar/bar metric comparisons
- readability summary figure across depth blocks
- fixed probe-region figures under `ProbeROIs/*`
- parameter and gradient histograms
- hard-mining overlays and mining-file evaluations

Important interpretation detail:

- these are tile-level classification metrics
- they are not direct character readability metrics
- for your goal, visual readability and high-confidence regional precision should be tracked separately

## 7) Known Issues (Important)

### 7.1 vis.ipynb is broken for multiple reasons

1. Notebook JSON is malformed
	 - raw file contains stray tokens like d_start and tra_id embedded in JSON
	 - this prevents notebook parsing/loading

2. Even if JSON is repaired, notebook code targets stale APIs
	 - imports functions not present in current utils/dataloader.py:
		 - load_tv_data
		 - get_or_compute_normalization
		 - get_tile_coords_for_split
	 - references stale config fields:
		 - config.data.train_segment_id (current code uses scroll1_id)

3. There is no vis.py file in repo root
	 - current script is scroll4_vis.py

### 7.2 finetune path is stale/broken

Files: finetune.py, utils/finetune_dataloader.py

Breakages observed:

- finetune.py imports missing symbols from utils/dataloader.py:
	- load_scroll4_data
	- get_or_compute_normalization
- finetune.py imports missing train-level symbols:
	- train_epoch
	- validate_epoch
- utils/finetune_dataloader.py uses config.dataloader.*
	- current config namespace is config.dl.*

Conclusion:

- fine-tuning code appears from an older API generation and is not runnable without refactor

### 7.3 get_data.sh has command typo

File content ends with a stray backtick after tar extraction command.

### 7.4 Seed config contradiction

In train.py set_seed sets:

- torch.backends.cudnn.deterministic = True
- torch.backends.cudnn.benchmark = True

These settings push in opposite directions for strict reproducibility/performance behavior.

## 8) File-by-File Purpose Map

- train.py
	- primary training entrypoint
	- owns epoch loop and checkpoint policy
- scroll4_vis.py
	- standalone full-scroll4 prediction map visualization
- ink_shrinker.py
	- erodes labels to reduce tile-spill supervision effects
- get_data.sh
	- one-line artifact download/extract helper (currently typoed)
- vesuvius.sh
	- environment bootstrap commands (pip install, terms acceptance, tmux)
- utils/config.py
	- all hyperparameters and paths
- utils/dataloader.py
	- iterable dataset over zarr + mask filtering + normalization
- utils/model.py
	- CBAM-based 3D classifier model
- utils/training_utils.py
	- optimizer/scheduler/loss/metrics/save
- utils/visualizer.py
	- TensorBoard scalars/figures + mining generation + mining re-eval
- utils/hard_mining.py
	- reservoir sampling and injection logic
- utils/finetune_dataloader.py
	- legacy fine-tune dataset logic (currently API-drifted)
- comparer.ipynb
	- sanity checks and normalization experiments
- visualize_hard_examples.ipynb
	- mined hard example overlays by depth

## 9) Existing Experiment Artifacts

### 9.1 Checkpoints in models/

- best_model_f1.pth
- best_model_loss.pth
- model_epoch_10.pth through model_epoch_100.pth

Note:

- checkpoints up to epoch 100 exist even though default n_epochs in config is 50
- at least one run used different epoch settings

### 9.2 TensorBoard run folders

- runs/20230702185753/
- runs/20230827161847/
- runs/20230827161847_recurring/
- runs/full_vis_09_182125/

Each contains event files for retrospective metric inspection.

## 10) External Helper Repos (Quick Audit)

You referenced these helper repos:

- /media/jeff/Seagate/vesuvius-docker
- /media/jeff/Seagate/vesuvius-3dstreamer
- /media/jeff/Seagate/vesuvius-zarrs

### 10.1 vesuvius-docker

- Dockerfile uses runpod/pytorch:2.1.0 CUDA 11.8 base
- clones this repo to /vesuvius
- installs requirements, accepts terms, sets git identity
- command currently keeps container alive

### 10.2 vesuvius-3dstreamer

- includes VesuviusStream iterable dataset for sampling 3D chunks
- supports multi-zarr / ndarray sources
- includes multithreaded TIFF->zarr converter with memory controls
- contains conversion command examples in all.sh

### 10.3 vesuvius-zarrs

- includes PNG/TIF stack download/build helpers
- contains multiple prebuilt .npz stacks (large)
- output/ currently has 00.tif..64.tif plus stack.npz
- frag/ contains 00.png..65.png
- file named full is a grayscale PNG image (not a directory)

## 11) Practical Notes for Future Iteration

The major pain point described (metrics move, readability does not) fits this setup's objective/aggregation behavior.

In this codebase, readability depends heavily on:

- tile-level score separation quality, not only global F1/AUC
- depth-window selection and aggregation policy
- region selection for inference targets (especially scroll4)
- hard-mining balance (false-positive suppression vs missed positives)

Given your stated goals (especially text discovery in unannotated regions), treat visual separability metrics as first-class outputs in addition to standard classification metrics.

## 12) Minimal Runbook

### Environment

```bash
pip3 install -r requirements.txt
vesuvius.accept_terms --yes
```

### Train

```bash
python train.py -n 20230702185753_experiment
```

### TensorBoard

```bash
tensorboard --logdir=./runs
```

### Scroll4 map visualization

```bash
python scroll4_vis.py -m models/best_model_f1.pth
```

### Erode labels

```bash
python ink_shrinker.py
```

## 13) Agent Handoff Notes

If an agent opens this repo cold, top priorities are:

1. Verify active data IDs and zarr path in utils/config.py
2. Confirm labels/masks exist for target IDs
3. Treat finetune.py and vis.ipynb as stale until repaired
4. Use train.py + scroll4_vis.py as current canonical executable paths
5. Interpret metric gains against visual readability, not alone
6. Inspect hard_negs/*.jsonl trend to understand failure mode shifts
7. For adding new scroll/fragment data, follow section 3 (Data Acquisition Pipeline)

New zarr locations to be aware of:

- /media/jeff/Seagate/ves_zarrs2/20230709155141.zarr (Scroll 2 segment, no labels)
  source: paths/20230709155141/layers/, 64 depth slices, 2806 × 8499 spatial

This README is intended to be updated as the process evolves, especially when strategy changes around scroll4 targeting and tile objective calibration.

## 14.0) Scroll4 w023 Patch Reconstruction Reference

Two zarrs were built for the scroll4 w023 fragment. Both require the warp dots and each other.

### Scroll4 w023 7.9um zarr (20240304161941)

The 7.9um scan is a standard volpkg segment. Build via:

```bash
python build_scroll_zarr.py preset scroll4-79
```

This downloads 65 layer TIFFs from dl.ash2txt.org, assembles into
`ves_zarrs2/20240304161941.zarr` (64, 13303, 31674), and writes `masks/20240304161941.png`.
The `--flip` flag is applied (the 7.9um flattening is horizontally mirrored vs the 2.4um one).

Labels for this zarr come from the 2.4um scan via dot-warp (section 3.5):
- Dots: `warp_MARK_2p4_source_dots.png` and `warp_MARK_7p9_target_dots.png`
- Warp: run `python warp_from_dots.py` with `--id w023` — writes
  `_ves_tmp/w023_dotwarp_map{x,y}.npy`
- Bake labels: `python bake_scroll4_79_labels.py --out-id 20240304161941 --tag w023`
- Labels live at: `inklabels/20240304161941.png`, `eroded_inklabels/20240304161941.png`

CRITICAL warp recipe (must match bake exactly):
- `ww=3600`, `smoothing=1.0`, 4 frame-corner anchors — see `fit_warp()` in build_teacher_zarr.py
- Using a different recipe (e.g. normalized/smoothing=1e-3) causes ~24px label/data offset

### Scroll4 w023 2.4um teacher zarr (20251217075048)

The 2.4um zarr covers the right 30% / top 40% of the w023 frame, warped into the 7.9um
coordinate system and scaled 3.3125× to maintain native 2.4um fidelity:

```bash
# 1. download the ink prediction tif (386MB, used for QA only — labels come from the 7.9 ones)
aws s3 cp --no-sign-request \
  "s3://vesuvius-challenge-open-data/PHerc1667/segments/20240304161941-w023_20240304161941_flatboi/ink-detection/PHerc1667-20240304161941-2.399um-0.22m-78keV-volume-20251217075048-20260417190342-new_canon_autoresearch_recipe-tile256-stride128.tif" \
  _ves_tmp/w023_ink_full.tif

# 2. build the teacher zarr (downloads ~56GB of S3 chunks, warps, writes zarr + labels + mask)
python build_teacher_zarr.py \
  --out-id 20240304161941_t24 \
  --x0f 0.70 --x1f 1.0 --y0f 0.0 --y1f 0.40 \
  --block 512 --workers 16

# 3. rename to numeric id
NID=20251217075048
mv ves_zarrs2/20240304161941_t24.zarr ves_zarrs2/${NID}.zarr
mv inklabels/20240304161941_t24.png inklabels/${NID}.png
mv eroded_inklabels/20240304161941_t24.png eroded_inklabels/${NID}.png
mv masks/20240304161941_t24.png masks/${NID}.png

# 4. precompute normalization
python precompute_norm.py --scroll-id ${NID}
```

Key facts:
- Source: S3 zarr `PHerc1667/segments/20240304161941-w023_20240304161941_flatboi/
  surface-volumes/2.399um-0.22m-78keV-volume-20251217075048.zarr/0/`
- Native 2.4um shape: (109, 41860, 102360) chunks (109, 128, 128) uint8
- Teacher output: (64, 17626, 31479), chunks (8,32,32) uint16 (109→64 depth resample)
- Labels are scaled-up version of 7.9um eroded_inklabels (user hand-cleaned, NOT the raw ink tif)
- Warp: same exact recipe as the 7.9um label bake (ww=3600, smoothing=1.0, 4 corners)
- QA: mask IoU=0.93, label/data IoU=0.71 at zero shift (validated)
- Training split: --split-axis y (top 75% train / bottom 25% valid) — x-axis causes label
  concentration confound due to ink being spatially clustered in the right portion of the frame
- Ring negatives: r=2 tile radius → ratio 1.86:1 at T=106, confirmed sufficient

## 14) Scroll4 Dual-Resolution Investigation (2026-07-08)

This section records findings from an extended investigation into whether the 7.9um/54keV scan
of scroll4 w023 can be made to yield ink signal, using the available 2.4um/78keV scan (which
is legible) as a reference and label source.

### 14.1 Background and Hypothesis

Scroll1 ink is detectable at 7.9um/54keV by our tile-level classifier. Scrolls 2, 3, and 4
at 7.9um are not — including by the researchers' heavy U-Net. The working hypothesis: scroll1
ink contains a heavy-metal component (likely lead-bearing) which creates a large absorption
contrast at 54keV. Scrolls 2/3/4 use purely carbon-based ink, which is elementally identical
to carbonized papyrus and therefore yields no absorption contrast — only a weak morphological
texture signal from pen pressure and fiber disruption. That morphological signal requires fine
spatial resolution to resolve; hence legibility at 2.4um but not 7.9um.

### 14.2 Data Assembled

- Teacher zarr: `ves_zarrs2/20251217075048.zarr` (64, 17626, 31479), uint16
  - The right 30% / top 40% of the 7.9um frame, resampled from 2.4um native chunks
  - Warp: hand-anchored thin-plate-spline from warp_MARK_2p4_source_dots.png / _7p9_target_dots.png
  - Labels: scaled-up eroded_inklabels/20240304161941.png (user hand-cleaned from 2.4um output)
  - QA: mask overlap 93.7% (crop-edge artifact), label/data IoU 0.71 at zero shift
- Ink prediction tif: `_ves_tmp/w023_ink_full.tif` (41860, 102360) uint8, 0-246
  - Source: researchers' canonical U-Net run on the 2.4um scan
- Warp maps validated: same recipe as bake_scroll4_79_labels.py (ww=3600, smoothing=1.0)
- Builder: `build_teacher_zarr.py` (new script; no-download flag for iterative reruns)

### 14.3 Teacher Training Attempt

Two runs were launched on the teacher zarr:
- Run 1: default depth window z32-40 (inherited from 7.9um config) — flat AUC=0.500, F1=0
- Run 2: corrected depth window z48-56 (strongest ink band in 2.4um depth profile) — same result

Root cause (3 compounding factors):
1. 78keV energy = near-zero absorption contrast between carbon ink and carbonized papyrus.
   The ink signal at 2.4um is morphological texture, not intensity.
2. Both v1 and v12_asym_attn_pool architectures use AdaptiveAvgPool3d (global spatial average),
   which is exactly wrong for a local texture signal: it collapses the spatial detail to zero.
3. The 106px tile with "any ink" binary label contains mostly blank area even when labeled
   positive, diluting the already-weak signal further.

Our global-pool tile classifier is the right tool for scroll1's absorption regime and the wrong
tool for the morphological texture regime of scrolls 2/3/4.

### 14.4 Correlation Analysis (key diagnostic)

Method: Spearman r between tile-level 7.9um features and the 2.4um ink labels/predictions.
Decision rule: r > 0.15 consistent across all spatial quadrants = real signal. Spatial confound
= consistent r overall but sign/magnitude reverses between quadrants.

Results (eroded inklabels vs 7.9um features, 43k tiles, right-30/top-40 crop):

| feature       | global r | top-L | top-R | bot-L | bot-R |
|---------------|----------|-------|-------|-------|-------|
| mean z28-40   | +0.061   | +0.13 | +0.17 | -0.01 | -0.01 |
| std z28-40    | +0.045   | +0.10 | +0.17 | -0.03 | -0.02 |
| mean z48-58   | +0.078   | +0.14 | +0.12 | +0.08 | -0.01 |

Verdict: SPATIAL CONFOUND. Global r is positive but collapses or inverts in the bottom half.
The top half of the sheet has slightly different papyrus exposure/texture that coincidentally
overlaps the labeled region — not a detectable ink signal.

Per-depth sweep (all 64 layers of 7.9um, same eroded labels):
- Best single layer: z=30, r=+0.084
- Mean |r| across all 64 layers: 0.040
- Same spatial confound pattern at every depth

Native 2.4um sanity check (raw 2.4um chunks vs ink tif, native coords, no warp):
- Best single layer: z=102 (of 109), r=+0.157
- Mean |r|: 0.102

## 15) Two-stage MIL era + receptive-field / dataset findings (2026-07-24)

This section captures the current architecture and the hard-won lessons from the twostage
context-window campaign. It supersedes the older 32x32-tile framing above (we now use 16px
label tiles at 9.362um/px, depth 24 ingested 4->28).

### 15.1 Current production architecture

`v15_twostage_wide_zgrad` (and its context variant `..._ctx`), ~1.18M params:

- Stage 1 (shared backbone, weights TIED across 3 depth windows 4-12 / 12-20 / 20-28):
  stem ingests `[raw, lcn, dI/dz]` -> two per-slice (1x3x3) convs -> learnable absolute
  depth positional encoding -> `depth_mix` (3D convs + CBAM + a single H,W maxpool) ->
  per-voxel logit map. Each window gets its correct absolute depth-PE offset so depth bands
  are genuinely distinguishable.
- Stage 2: WIDE fusion CNN `3 -> 32 -> 32 -> 16 -> 1` over the 3 stacked window maps. The
  original tiny fusion (3->16->8->1, ~4.8k params) UNDERFIT (train PR-AUC plateaued ~0.66);
  widening fixed that.
- Output: MIL log-sum-exp over voxels -> ONE scalar per 16px tile. Learnable LSE `r`
  interpolates mean(r->0) <-> max(r->inf). The tile-scalar MIL is what makes sparse thin-stroke
  ink survive aggregation (global spatial pooling dilutes it ~1000x).

### 15.2 The receptive-field / context window

`InkDetectorTwoStageWideZGradCtx`:
- Reads a `context_size` crop (e.g. 32px) centered on each 16px label tile.
- Runs the FULL backbone + stage-2 fusion on the whole crop, then CENTER-CROPS the fused
  voxel map to `tile_size//2` (pooled coords) before MIL-LSE.
- So supervision is UNCHANGED (still the central tile's ring label); context enters purely
  through the conv receptive field. Degrades gracefully to the plain model at ctx==tile.
- Param count is IDENTICAL to the plain model (the center-crop is parameter-free).

Cost/benefit observed:
- ctx32 = 4x the voxels through the backbone (~3x wall-clock), 4x the unsupervised surround
  the model can memorize (overfit amplifier), and inference OOM/crashes on the big fragments
  (e.g. 15921^2 PHerc1203) even with predict_tiles' 0.25x batch auto-scaling.
- Large RF clearly RAISES accuracy (context disambiguates a stroke continuing past the tile,
  since a 16px tile ~150um is narrower than a stroke) but at a steep overfit + compute price.

### 15.3 The shape-resolution problem (blob vs letters)

- With the CLOSED ring dataset the model localizes ink well (WHERE) but resolves the letter
  as a blob, not shapes. Root cause is shared with the overfit: the model memorizes
  scroll-specific texture at TILE granularity rather than a transferable ink-vs-papyrus
  boundary. The output is one scalar per 16px tile, so shape is inherently coarse.
- eroded vs closed is a LABEL-level lever, not a resolution lever:
  - closed (base=eroded, close=3, gap=3, shell=2) dilates positive tiles -> better balance +
    coverage -> higher WHERE-metric but blobbier (positives spread past the letter).
  - eroded keeps positives tight on the letter -> traces shape better at tile res but sparse
    positives -> class imbalance / collapse risk.
- IMPORTANT context correction: the historical "eroded gives sharper shapes" result was from
  SCROLL1 ONLY (easy letters, 100% concrete labels) and PLAIN (tile-RF) models. We are now on
  HARD letters with UNCERTAIN labels, so that tradeoff must be re-measured. eroded + large RF
  (ctx) had never been run -> this is the tsJe experiment.

### 15.4 DENSE SUPERVISION IS BANNED (do not revisit)

- Every dense (per-pixel BCE / U-Net decoder) experiment TANKED performance, including the
  researcher's exact dense-unet, which returned essentially nothing.
- The tile-scalar / single-value MIL return is precisely what lets THIS model learn. Keep the
  MIL-LSE tile output. `v15_twostage_dense` and `dense_labels` exist in the code but MUST NOT
  be used.

### 15.5 Regularization notes (for later)

- AdamW WEIGHT DECAY (`tra.weight_decay`, currently 0.0) is the cheap, principled overfit
  lever and is generally more effective here than L1 (`tra.l1_lambda`). Decoupled weight decay
  shrinks every weight toward 0 each step (multiplicative), penalizing the large-magnitude
  weights that memorization relies on, without adding a term to the loss/gradient. Nearly free.
- TTA-CONSISTENCY loss (candidate, not yet added): during TRAINING, forward two augmented
  views (e.g. a flip) of the same tile and penalize disagreement of their predictions, in
  ADDITION to the supervised loss. Differs from plain flip/rotation augmentation: aug shows
  each transformed view with its label independently (the model may still give inconsistent
  answers across views -> the hallucinations we see on holdout); the consistency term directly
  requires f(aug(x)) == f(x), regularizing the function to be invariant. Cost is ~2x forward
  passes per step (one extra view), NOT the 4-6x of inference-time TTA, and no gradient through
  a stop-gradient target if implemented that way. Deferred for now.

### 15.6 Campaign bookkeeping

- Two-stage runs log to `./runs_ts_mae`; finals saved to `models/twostage/{tid}_{tag}_final.pth`
  only on completion. Periodic checkpoints (every save_int=2 epochs) + best_model_* dump to the
  flat `models/` dir with GENERIC, non-namespaced names (model_epoch_N.pth) -> later runs
  clobber them. Warm-start from MAE via `models/mae_twostage.pth` (stage1.* keys, strict=False).
- Every finished twostage final uses `ring_label_source="closed"`; eroded+ctx was a genuine
  testing gap. `campaign_runner_twostage._base_config` is the single source of truth; only keys
  present in a TESTS dict override it (via `_OVERRIDES`).
- Next campaign (tsJe, tsJf): single-variable changes off tsJd -- (1) tsJe = ctx32 + eroded
  labels; (2) tsJf = closed + COARSE context (context_size stays 32 but context_downsample=2:
  avg-pool the input 2x at the stem, keeping the full 32px extent at half resolution). Other
  regularizers deferred.

### 15.7 Coarse-context knob (context_downsample)

Instead of shrinking the context window to reduce the receptive-field cost, keep the full extent
and coarsen it. `config.data.context_downsample` (int, default 1 = off): when >1 the ctx model
`InkDetectorTwoStageWideZGradCtx` avg-pools the input crop by that factor at the stem
(`F.avg_pool3d` kernel/stride `(1,ds,ds)`, depth preserved), then adjusts the MIL center-crop to
`tile_size // (2*ds)` (depth_mix's one H,W maxpool times the input downsample). Effect at ds=2 on
ctx32: same 32px context EXTENT but half resolution -> ~1/4 the activations (near-plain compute),
smaller overfit surface, and it removes the big-fragment inference CUDA OOM. Tradeoff: the CENTER
label tile is coarsened too (unlike a foveated full-res-center design). Param-count is unchanged
(avg-pool is parameter-free) so MAE warm-start and existing ctx checkpoints load cleanly. Verified
forward shapes: ctx32/ds2 -> center 4, ds1 -> center 8 (backward compatible).

### 15.8 Foveated context (arch v15_twostage_wide_zgrad_fovea)

Coarsening everything (15.7) also blurs the CENTER tile, which is costly at ~10um where ink is
already near the resolution limit (prior models resolved ink at 1-2um). The foveated arch keeps
the middle full-res and coarsens only the surround. `InkDetectorTwoStageWideZGradFovea` runs TWO
tied-backbone streams: (a) the full-resolution central tile_size crop, and (b) the whole
context_size crop avg-pooled to tile res (coarse, full extent). The surround's center-tile region
is upsampled and fused with the center stream via a tiny 1x1x1 `fovea_fuse` head (+41 params),
then MIL-LSE aggregates. Cost ~2x a plain tile pass (two tile-sized backbone passes) vs ~4x for
full-res ctx32. Requires context_size a multiple of tile_size (surround pools exactly to tile
res, ds = ctx//tile). MAE warm-start still transfers stage1 (strict=False); stage2 + fovea_fuse
init fresh. Verified: forward -> (B,1), params 1,183,117, warm-start loads 38 stage1 keys.
Campaign test tsJg = tsJd (closed) with this arch. Caveat: stage1 BatchNorm sees two input scales
per step (full-res tile + coarse full-extent) -> slightly noisier BN stats; acceptable, watch it.
- Quadrant check: top-right r=+0.35, bot-left r=+0.01 — still partly confounded
- Self-check (ink tif vs ink tif): r=1.000 — methodology sound
- Signal is ~2.5x stronger at 2.4um than 7.9um, but even at 2.4um it is a local texture
  signal that tile-mean intensity cannot reliably detect

### 14.5 Conclusion

The 7.9um/54keV scan of scroll4 does not appear to contain ink-detectable information under
the current tile-level approach. The U-Net null result (researchers could not read it either)
is the strongest prior. Our correlation analysis confirms this: no 7.9um voxel feature
correlates with known ink location in a spatially consistent way.

The 2.4um/78keV scan contains a real but weak morphological signal (~r=0.16 at best, spatially
concentrated). The researchers' U-Net can read it; our global-pool tile classifier cannot.

### 14.6 Recommended Next Steps (if pursuing 7.9um further)

In priority order:

1. LOCAL CONTRAST NORMALIZATION (preprocessing, cheapest)
   Subtract a Gaussian-blurred version and divide by local std before feeding to the model.
   This collapses the spatial brightness confound while preserving fine texture from ink
   strokes. Apply to both training input and the correlation test to see if r improves.

2. DENSE / FULLY-CONVOLUTIONAL PREDICTION (architecture, highest leverage)
   Drop AdaptiveAvgPool3d. Use a small FCN head that produces a per-pixel heatmap at the
   tile level. Train against per-pixel labels (not tile-level "any ink"). This preserves
   spatial structure the global pool discards. This is the structural difference between
   our approach and the researchers' U-Net.

3. TEXTURE-BASED SCROLL1 PRETRAINING → SCROLL4 TRANSFER
   Train a scroll1 model with local contrast normalization (forcing it to learn texture
   rather than absorption). Use that texture-trained model as initialization for scroll4.
   Scroll1 at 7.9um presumably has the same papyrus fiber texture patterns; a scroll4
   ink signal may be learnable by transfer if it exists at all.

4. QUADRANT SPLIT AS CONFOUND STRESS TEST (diagnostic)
   Train on the top half of the scroll4 7.9um crop (where the spurious correlation is
   strongest) and test on the bottom half (r≈0). If any ink signal transfers, it is real.
   If not, the data is conclusively a dead end for this approach.

5. WAIT FOR HIGHER-RESOLUTION SCANS
   The moment a 2.4um or better scan of scrolls 2/3 is released, the existing pipeline
   (build_teacher_zarr.py, warp infrastructure, train.py with --tile-size) is ready to
   run immediately. Pipeline preparation is complete. The ink signal at 2.4um is strong
   enough that even our lightweight model should learn, as validated on the scroll4 case.

> UPDATE (2026-07-09, see section 15): recommendation #1 (local contrast normalization) was
> DISPROVEN by direct probe on the 2.4um data — background subtraction HURTS separability.
> Recommendation #2 (dense per-pixel prediction) is now the active hypothesis under test.
> Read section 15 before acting on this list.

## 15) Scroll4 2.4um Learning Investigation (2026-07-09)

Continuation of section 14, focused specifically on the 2.4um/78keV scroll4 w023 teacher
zarr (`ves_zarrs2/20251217075048.zarr`, shape (64, 17626, 31479)). Goal: get a model to
TRULY LEARN the ink here (2.4um is legible to humans and the researchers' U-Net), so the
signal/insight can later be lifted onto the lower-fidelity 7.9um scans of scrolls 2/3/4.
This section is deliberately blunt about what was DISPROVEN so no one re-runs dead ends.

### 15.1 The problem space (why this is hard)

- Scroll1 @ 7.9um/54keV: ink readable by a simple 3D CNN — the ink carries heavy metals
  (likely lead), giving ~30x X-ray absorption contrast. The signal is a scalar INTENSITY
  offset, which global-average pooling reads directly.
- Scrolls 2/3/4 @ 7.9um/54keV: ink undetectable by ANYONE (including the researchers' heavy
  U-Net). The ink is carbon-based, elementally identical to carbonized papyrus → no
  absorption contrast at this energy/resolution.
- Scroll4 @ 2.4um/78keV: ink becomes readable again. At this resolution the discriminative
  cue is a weak, LOCAL, in-plane morphological/textural pattern, NOT a mean-intensity shift.
- Ink strokes are LARGE: ~400+ px wide, thousands of px long at 2.4um. Consequently the vast
  majority of tiles are pure interior (all-ink or all-papyrus); only ~45% of ink-containing
  106px tiles straddle a letter boundary (measured, see 15.3).

### 15.2 Architectural constraints (hard rules for this problem)

- Tile scale is fixed by the competition + evidence, NOT a free knob. 32x32 @ 7.9um (≈106 @
  2.4um) already worked for scroll1; researchers succeeded at 128; 256 is BEYOND the
  competition tile limit and hallucination-prone. Do NOT scale tiles up to "see more."
- Binary "is there ink anywhere in this tile" is a legitimate, well-posed framing BECAUSE
  voxels are ~binary ink/blank and most tiles are pure interior. Segmentation is not required
  as a product — but see 15.4 for why dense LABELS still help as a training signal.
- The three campaign architectures (v14_mil_deep, v17_2p1d_maxattn, v18_2p1d_lv) are all
  PEAK-POOLED: v14 uses LSE (logsumexp), v17/v18 use AdaptiveMaxPool3d (hard spatial max).
  A tile only has to fire at ONE location to satisfy its label — an "exists somewhere"
  shortcut, trivially met by a spurious peak. This is a prime suspect for why they don't
  learn a real feature.

### 15.3 What was MEASURED (no-training probes, `_ves_tmp/*_probe.py`)

All AUC values are ink-tile vs ring-negative-tile, on the geographic y-split validation region.

- Hand-crafted per-tile feature separability (depth-mean, z=24-40):
  intensity-mean 0.557, std 0.544, local-variance 0.519, high-pass 0.530, gradient 0.527,
  fiber-band FFT 0.567. Multivariate logistic-regression ceiling ~0.58.
  → The LOCAL-VARIANCE hypothesis behind v18 is REFUTED (0.52, sign unstable across regions).
- Depth structure: sheet undulates (material COM-z mean 30.2, std 6.8, spans z≈12-50), but
  SURFACE-ALIGNING the depth profile did not improve ink vs papyrus (0.54 aligned vs 0.55
  raw). → Depth-window misalignment is NOT what hides the signal. Fixing depth alignment is
  not the lever.
- Spatial aggregation (the sqrt-N idea): averaging the per-tile intensity over K×K tile
  neighborhoods (K=1..13) stayed FLAT at ~0.55. → The weak intensity signal is not a noisy
  estimate of a coherent per-location truth; aggregation does NOT rescue it. REFUTED.
- Local-contrast normalization (tile mean minus large-window background): 0.539, WORSE than
  raw 0.550 at every neighborhood size. → Background/difference subtraction removes the
  little low-frequency signal that exists. This matches the earlier scroll1 result that
  difference-BANDS failed. Do NOT propose subtraction/difference inputs again without
  evidence — disproven multiple times.
- Label geometry (eroded labels, tile=106): of ink-containing tiles, 39.7% are ~pure ink,
  7.7% near-pure, 21.9% mostly-ink, 22.6% mixed, 8.0% trace. Genuine BOUNDARY tiles
  (10-90% ink) = 44.6% at tile=106 and 68.9% at tile=256.

Consolidated reading: EVERY scalar statistic and every post-hoc combination sits at
0.52-0.58, yet the best campaign CNN (t01/v14) visibly renders faint letter shapes. The
learnable signal is therefore a genuine LEARNED in-plane spatial pattern that hand features,
averaging, subtraction, and depth realignment cannot capture. Feature engineering is
exhausted; the remaining lever is the LEARNING SIGNAL itself.

### 15.4 The validation-metric bug (do not trust the campaign `Valid` scalars)

The campaign `AUC/ROC_AUC/Valid` reads EXACTLY 0.5000, `PR_AUC/Valid` EXACTLY the positive
prevalence (0.4337), `F1/Valid` = 0, `Specificity/Valid` = 1.0 — byte-identical across all
three architectures and every epoch. That is the exact algebraic signature of a CONSTANT
score vector (all tiles predicted one class) fed to `calculate_metrics`
(`utils/training_utils.py`): `roc_auc_score` → 0.5000 and `average_precision_score` →
positive_ratio precisely when `y_scores` is constant.

Verified (loading the saved checkpoint, `_ves_tmp/val_metric_lean.py`): reproducing the
figure inference path gives NON-constant scores and roc≈0.51 — so it is NOT an autocast/fp16
issue and NOT the model literally being dead. The constant appears specifically inside the
in-loop `validate_epoch`. Exact trigger still unconfirmed (suspects: BatchNorm eval-state
timing, degenerate ring-valid batch). The TRUSTWORTHY signals are the `R_M/*` readability
scalars and the eval FIGURE (both from the figure path), which correctly rank t01 best.
Bottom line: rank experiments visually / by the figure path, not by `AUC/*/Valid`.

### 15.5 Active hypothesis under test: dense per-pixel supervision

Rationale (NOT "more gradients" — that was an overstatement given 40% pure tiles):
1. It removes the peak-pooling "fire somewhere" shortcut (15.2): every interior pixel must be
   classified from its LOCAL receptive field ("be right everywhere"), a much stronger
   constraint on the learned feature — biting even on pure all-ink tiles.
2. The ~45-69% boundary tiles supply letter-SHAPE supervision a tile classifier never sees.
3. Must pair with a LARGE receptive field (U-Net decoder) so each pixel decides WITH context,
   at the accepted 106-256 tile scale — i.e. the researchers' proven recipe minus the scale
   inflation.

Test harness: `_ves_tmp/dense_train.py` — a self-contained 2.5D U-Net (per-slice Conv3d
texture stem → depth-max → 2D U-Net decoder → per-pixel logits), per-pixel masked
BCEWithLogits on eroded labels, y-split 75/25, 128px tiles, z=24-40, 15 epochs. It reads the
zarr directly (bypassing the DataManager setup) and renders a stitched raw|prediction|ground-
truth figure over an ink-rich validation window every few epochs → `_ves_tmp/dense_pred_epNN.png`.
Verdict is VISUAL: does it produce sharper letters than t01's blobs?

INTEGRATED into train.py (config.data.dense_labels / --dense-labels):
  - dataloader emits the (1,T,T) eroded ink-label MAP per tile (before any augmentation);
  - trainer uses per-pixel masked BCE (_train_batch_dense / _validate_epoch_dense) with a
    PIXEL-level pos_weight (calc_dense_pos_weight);
  - archs: `dense_unet` (per-slice stem → hard depth-max → 2D U-Net) and `dense_unet_depth`
    (per-slice stem → 3D depth-MIXING convs → LEARNED depth-attention softmax-over-depth pool →
    2D U-Net) — the latter for putting weight on depth (campaign-15 finding);
  - eval figure `add_dense_evaluation_figure`: seamless overlap-blended (Hann-window) inference,
    and — per this README's convention — it ALWAYS sweeps the FULL inference depth d_start..d_end
    (e.g. 0→64) in blocks of `depth`, rendering one prediction panel per depth block plus a
    depth-MAX composite and the ground truth, EVEN THOUGH training uses only the narrower
    train_d_start..train_d_end window (e.g. 24-40). the composite drives the logged
    Dense/Valid_PixelAUC_window.

  Example (train on the known ink band 24-40, always visualize the whole sheet depth):
  ```bash
  python train.py -n dense_s4_24um_depth --log-dir runs_dense_pipeline \
    --scroll-id 20251217075048 --arch dense_unet_depth --dense-labels \
    --tile-size 128 --depth 16 --train-d-start 24 --train-d-end 40 --d-start 0 --d-end 64 \
    --epochs 15 --eval-int 5 --batch-size 12 --lr 2e-4 --data-aug 0 \
    --ring-negatives --ring-label-source eroded --split-axis y --train-split-frac 0.75 \
    --no-hard-mining --mask-memmap
  ```

### 15.6 Directions explicitly ABANDONED (evidence-backed)

- Architecture roulette on fixed-window, global/peak-pooled, one-label-per-tile classifiers.
- Local-variance / texture-energy input channels (v18 idea) — disproven, 0.52.
- Difference/contrast/background-subtraction inputs — disproven twice (scroll1 bands + 15.3).
- sqrt-N spatial aggregation of the weak scalar signal — disproven, flat ~0.55.
- Depth-window realignment as the primary lever — disproven, aligned≈raw.
- Readability-scalar tuning as an objective — legibility is judged visually; not a lever.
- Over-indexing on 2.4um-only FINE texture — transfer dead-end: fiber-scale texture is
  physically unresolved at 7.9um, so only LOW-frequency density/morphology can ever transfer.

---

## 16) PHerc0139 9.362 µm Campaign Series (2026-07 onwards)

This section documents the shift to the **PHerc0139** scroll (9.362 µm / 113 keV / 1.2 m detector),
away from the old PHerc0332/PHerc1667 7.91 µm work in sections 1-15. The ink here is
carbon-based (same as scrolls 2/3/4) but at 113 keV there is measurable morphological contrast
from pen pressure — enough for MIL-based tile classifiers to learn from.

The old sections (1-15) remain as a reference for the 7.91 µm / 2.4 µm investigation history.
All active training code, zarr paths, and config have been updated to the new data.

### 16.1 Data and Infrastructure

**Training zarrs** (all PHerc0139, 9.362 µm, chunks (8,32,32), uint16, zarr_format=2):
- `20260115000000` w044 — shape (28, 6021, 8141), mask valid 0.882
- `20250223000000` w059 — shape (28, 7220, 10020), mask valid 0.295 (1.1 µm overlap band)
- `20260206000001` w047 — shape (28, 5821, 8421), mask valid 0.402 (1.1 µm overlap band)
- `20260115000001` w056 — shape (28, 7180, 9740), mask valid ~0.313 (1.1 µm overlap band)
  added 2026-07-19; labeled band y≈1837-4472; horizontal split top 50% train

**Zarr root:** `C:\Users\ChenJeff\Documents\ves_zarrs2\`

**Test zarr:** `20260716083545` — PHerc0813, (28, 4421, 4421), VC3D-grown segment
`auto_grown_20260716083545968`, 2.98 cm², max_gen=175 (restored snap 9, 2026-07-17).
External backup: `Documents/vc3d_recovered_0211/RESTORED_auto_grown_20260716083545968_snap9_latest`

**Config system:** `utils/config.py` — dataclass-based, no CLI args to `train.py` (only `-n`).
Campaign runners instantiate `Config()`, mutate fields, pass to `Trainer(c).run()`.

**Depth convention:** d=0 is the topmost layer. The ink signal for PHerc0139 concentrates
most strongly in the middle stack (d=8-16); d=0-4 and d=24-28 tend to be noise or empty
air above/below the papyrus.

**VC3D segment recovery procedure:** if the active paths folder gets zeroed by a crash,
autosave backups survive at `~/.VC3D/remote_cache/open_data/projects/backups/<uuid>/<N>/`.
Restore by copying the highest-numbered good snapshot back into `paths/<uuid>/`. Check with
`($b = [System.IO.File]::ReadAllBytes("$snap\x.tif"); ($b | Where-Object {$_ -ne 0}).Count)`.

### 16.2 Architecture Family (v14)

All active models share the v14 MIL backbone, registered in `utils/model.py`:

**v14_mil_deep** (base, 1,136,210 params at tile=16, depth=8):
1. Per-slice stem: two Conv3d with depth kernel=1. Learns 2D texture per depth independently.
2. Depth-mix: two full Conv3d(3,3,3) + CBAM3D attention + MaxPool3d(1,2,2) spatial only.
3. Per-voxel logit head: Conv1×1×1 → scalar per voxel.
4. LSE aggregation: tile_logit = (1/r) × (logsumexp(r·v) − log N), r learnable [0.5, 10].
5. BCE loss against eroded-inklabel tile label.

Key insight: global-average-pool (all arch10/18/28 models) dilutes the sparse ink signal
~1000×. LSE lets a handful of high-confidence voxels drive the tile prediction.

**v14b_mil_zgrad:** feeds [raw, dI/dz] to the per-slice stem. The z-derivative peaks at
ink-layer interfaces and is invariant to the slowly-varying papyrus bulk-density baseline.

**v14c_mil_lcn:** feeds [raw, LCN(raw)] to the per-slice stem plus a learnable depth
positional encoding. LCN removes the global bulk baseline while preserving local texture.
The depth PE lets the model key on the absolute depth band where ink sits, exploiting the
fact that the papyrus surface lies at a consistent depth range across tiles.

### 16.3 Triple-Scroll Sweep: campaign_p0139_triple_v2 (2026-07-17)

12 tests (t02-t13) covering:
- tile: 16, 24
- depth: 8, 4
- range: 0-28, 8-16
- arch: v14_mil_deep, v14b_mil_zgrad, v14c_mil_lcn
- one aug test (t13)

Training scrolls: w044 + w059 + w047 (3-scroll). Logs in `runs_p0139_triple/`.

WINNER: **t06 — v14c_mil_lcn, tile=24, depth=8, range=0-28**
`cmp_p0139_triple_v2_2026_07_17_t06_v14c_mil_lcn_t24_d8_r0to28`

Extreme overfitting but clearly the best learner — it finds ink where others don't. The
overfitting just means regularization needs to be tightened, not that the architecture is wrong.

Key findings from the sweep:
- d8 > d4 overall (more depth context helps)
- t16 > t24 overall (smaller tiles = tighter locality = harder decision boundary = better generalization)
- range 8-16 > 0-28 overall (concentrating on the ink-rich middle stack reduces noise)
- EXCEPTION: d4 performed better than d8 when combined with range 8-16 (shallower depth
  window is more tolerable when the spatial range is already tight)
- v14c_mil_lcn dominated across every tile/depth combination it appeared in

WHY LCN WON: at 113 keV the papyrus bulk density creates a slowly-varying spatial
brightness gradient that is different in every tile (beam hardening, scan geometry,
papyrus thickness variation). Raw intensities force the network to spend capacity learning
invariance to this gradient. LCN removes it by subtracting the local neighborhood mean and
dividing by local std — the network sees only LOCAL CONTRAST which is what ink actually is:
a fine-scale density perturbation relative to the surrounding papyrus. The depth PE is a
bonus: ink consistently sits at a specific depth range within the papyrus stack, and the PE
gives the model a soft prior on this without hardcoding it.

### 16.4 LCN Refinement Campaign: campaign_lcn (2026-07-19)

3 tests, all v14c_mil_lcn, range 8-16, 4-scroll (w044+w059+w047+w056):
- t01: tile=16, depth=8
- t02: tile=16, depth=4
- t03: tile=8, depth=4

Key changes vs triple_v2:
- l1_lambda raised 3e-7 → 7e-6 (to fight the overfitting the LCN win exposed)
- w056 added as 4th training scroll (horizontal split, top 50% train)
- range fixed to 8-16 (best range from triple_v2)
- only LCN arch tested (winner confirmed, refining not re-searching)

Logs in `runs_lcn/`. Runner: `campaign_runner_lcn.py`.

---

## 17) Current nnU-Net / Multitile Era (2026-09-03)

This section supersedes earlier statements about the active architecture, objective, and campaign.
Sections 1-16 remain historical evidence and should not be interpreted as the current runbook.

### 17.1 Current scientific objective

- Ink is demonstrably detectable in the approximately 9.3-9.6 micron isotropic volumes.
- The papyrus sheet undulates through the 24-slice input box. Ink is expected at one or a few
  decisive depths near one physical papyrus-air surface.
- Ink labels are transferred from much higher-resolution scans. They are sparse and uncertain;
  a wrong ink/papyrus boundary is more damaging than leaving a target unsupervised.
- The immediate iteration target is w013 (`20240304141531`), which has the largest and most
  trustworthy label set. It has different acquisition physics from the eventual 9.362 micron /
  113 keV deployment domain, so success here is necessary but not sufficient for final transfer.
- No current model has recovered a convincing character outside its immediate labeled corpus.
  Do not use model predictions as ink pseudo-labels until that changes.

### 17.2 Why sparse multitile supervision is mandatory

Dense supervision repeatedly failed with imperfect labels. Single-score MIL at the other extreme
did not provide enough localized gradient. Multitile is the successful middle ground:

- read a 192x192 context window
- average-pool XY by 2 before the backbone
- predict only the central 16x16 area
- divide that center into a 2x2 grid of four 8x8 targets
- aggregate each target's depth/spatial voxel bag independently

The four targets per window narrowly beat the tested 4px/8px/16px subtile and 16px-64px center
alternatives. The margin is not large, but the direction is consistent: more useful localized
gradient than single-tile MIL without imposing dense-label correctness.

### 17.3 Label safety invariants

The active dataset uses `ring_label_source="closed"`:

- close radius 3 tiles
- gap radius 3 tiles
- negative shell radius 2 tiles
- positive labels from `eroded_inklabels`

`multitile_pos_only=True` is a WINDOW-LEVEL invariant. If a window contains any supervised
positive, only its positive subtiles contribute to loss; every non-positive cell in that center
is masked, including otherwise valid ring negatives. Ink-free ring windows provide negative
supervision. Positive and negative labels must never touch inside one supervised center.

Campaign 16 replaced the geographic axis split with `train_masks/20240304141531.png`. The hand
mask is expanded to complete 8px target cells. Training and validation target masks are disjoint:
training contains concrete/easy letters plus a small hard subset, while validation contains the
difficult held-out letters. This validation distribution is intentionally harder than training
and directly measures the desired letter-to-letter transfer.

### 17.4 Current model: `nnunet3d_lcndz`

1. Input `(B,1,24,192,192)` is average-pooled to `(B,1,24,96,96)`.
2. Stem concatenates raw, 5x5 per-slice LCN, and forward depth difference.
3. Three-level 3D nnU-Net uses channels 32/64/128 with a 256-channel bottleneck.
4. IBN-a is used on the first normalization in `enc1` and `enc2`; remaining norms are IN.
5. Decoder skip connections restore high-resolution detail; weak conv/head dropout is active.
6. Legacy `DepthSurfaceAttn` applies an unsupervised depth-local multiplicative feature gate.
7. Campaign 17 adds `NewDepthSurfaceHead`: a spatial/dilated 3D CNN predicts one softmax depth
  distribution per spatial position and injects it into `enc1` through a zero-initialized
  residual 1x1 projection.
8. Decoder emits one voxel logit channel. The 16px center is divided into four 8px bags.
9. Gated attention-MIL pools each bag. Attention entropy regularization is essential empirically.
10. The global bottleneck embedding feeds SupCon. DANN uses the same embedding only when more
   than one domain exists.

The legacy attention-MIL path attends over scalar voxel logits (`feat_dim=1`). Campaign 17 adds an
optional feature-level path over 32-channel decoder vectors. Multitile SupCon now pools one decoder
embedding per target cell and uses that cell's label and validity mask.

### 17.5 Campaign findings, 9 through 17

- nnU-Net is the strongest tested family and can fit all supplied labels.
- Context improves results through 192px/ds2. 256px provides no useful gain for much more cost.
- MAE initialization is dramatically better than random initialization.
- Legacy learned surface attention is beneficial.
- Variance-based spill is beneficial but rapidly reaches its hinge threshold; stronger spill
  does not keep improving the model.
- Weak convolution/head dropout helps. Heavy dropout harms learning.
- Skip-drop 0.2 versus 0.6 showed little difference. This does not prove skip features are absent:
  the implementation drops each whole skip for the entire batch and restores all skips at eval.
- Attention-MIL helps only with entropy regularization.
- Spatial SupCon helps, although its current global-label use is not multitile-aligned.
- DANN damaged low-contrast regions when genuinely active and needs weaker, cleaner testing.
  It is a no-op in current one-scroll runs because `dann_n_domains=1`.
- FDA, brightness/contrast/noise, and elastic augmentation have shown negligible improvements.
- Depth jitter and weak dropout were the only campaign-13 regularizers that materially reduced
  training fit; validation remained approximately flat.
- Multitile beat single-tile MIL. Campaign 15 selected the 16px center with four 8px targets by
  a narrow margin.
- Campaign 16 introduced the manual target split. Campaign 17 tests the new supervised surface.

### 17.6 Effective campaign-17 configuration

- six arms: corrected baseline, surface feature, surface-guided aggregation, feature attention,
  and two alternate center/subtile geometries
- w013 only, manual split, 15 epochs, batch 32, learning rate 1e-4
- context 192, ds2, depth 24, z=4:28
- center 16, subtile 8, grid 2x2, train step 16, `pos_only=True`
- BCE with automatic positive weight
- no ink-label smoothing and no soft ink labels; the auxiliary depth target is soft
- `gce_q=0.7` exists in config but is inactive because `loss_type="bce"`
- attention-MIL with entropy 0.03
- legacy surface enabled; new surface enabled with loss 0.1 and smoothness 0.02
- variance spill lambda 0.5, minimum depth variance 0.8
- SupCon curriculum 0.05 -> 0.5 over 8 epochs
- IBN enabled; dropout 0.05/0.05/head 0.10; skip-drop 0.20
- depth jitter 4; TTA consistency enabled
- flip/rotation 0.6, noise 0.3, brightness/contrast 0.6, FDA 0.5; elastic disabled
- DANN configured at 0.25 but inactive with one domain
- current warm start: `models/mae_nnunet_96.pth` (96px/ds2, no IBN)
- planned warm start: 192px/ds2 MAE with IBN
- full-scroll surface arrays are QA artifacts; campaign 17 derives surface targets online per crop

### 17.7 Geometric augmentation correction

Before campaign 17, flips, rotations, and elastic deformation transformed the input without
transforming the multitile target and validity grids. Earlier multitile geometric-augmentation
results are therefore confounded. Campaign 17 applies identical flips/rotations to all three.
Elastic is fail-safe disabled for multitile until the dense source label/mask can be warped with
the exact displacement field. Context jitter is disabled because the fixed output crop otherwise
ceases to coincide with its target. Photometric/FDA/noise transforms remain spatially safe.

### 17.8 Highest-priority roadmap

1. Rerun the manual baseline and surface tests with corrected joint flip/rotation augmentation.
2. If campaign 17 validates the surface head, use its probability distribution directly in the
  depth aggregator. Blend a surface-band branch with the existing global branch according to
  surface entropy so uncertain surface predictions cannot erase the ink signal.
3. Replace scalar-logit attention-MIL with feature-level attention over decoder vectors inside
  each subtile. Keep a low-capacity shared scorer and entropy regularization to limit overfit.
4. Pretrain at 192px/ds2 with IBN. Then test frozen-encoder/head-only warmup followed by gradual
  unfreezing and a lower encoder learning rate to prevent immediate destruction of MAE features.
5. Make SupCon target-aligned: pool one decoder embedding per 8px cell and apply contrastive loss
  only to supervised cells. The current global context embedding inherits one window label.
6. Sample uniformly by connected ink component/letter rather than by overlapping windows. A large
  letter currently contributes many highly correlated examples and can dominate the objective.
7. Add character/component-level validation metrics so large letters and abundant papyrus cells
  cannot dominate PR-AUC. Retain target-level PR-AUC as a secondary metric.
8. Do not prioritize GCE yet. Current campaign uses BCE and the manual/closed/pos-only pipeline
  deliberately removes ambiguous labels. GCE suppresses confident-wrong gradients and can also
  suppress genuinely hard ink. If retested, sweep q=0.3/0.5/0.7 against BCE; do not return first
  to q=0.9, which previously killed MAE fine-tuning.
9. Once the w013 methodology transfers between held-out letters, add fragments back gradually.
  Test weak delayed DANN only after domain-balanced batches exist; it cannot be evaluated on one
  scroll and should not be bundled with the data-scale change.

---

## 18) Character-Aware Sampling and Metrics (2026-09-03)

Campaign 18 copies the corrected campaign-17 baseline. Character metrics are instrumentation and
therefore enabled in every arm; only `character_balanced_sampling` changes the training stream in
the principal A/B.

Character construction:

- connected components are computed from the full-resolution eroded ink label
- components smaller than 8 pixels are discarded as noise
- each 8px target cell receives its dominant positive component id
- each valid ring-negative cell is assigned to its nearest character
- components crossing the fixed manual split are excluded from character sampling and metrics

Balanced training cycles uniformly over characters. For each character it samples one positive
target's containing window and then one associated ink-free closed-ring window. The epoch length remains equal to the
ordinary dataset length, so the comparison changes representation rather than update count.

Character metrics are computed independently per component against its associated ring:

- macro positive recall
- macro local-ring FPR
- macro F1
- macro average precision
- success fraction: recall >= 0.5 and ring FPR <= 0.1

`Character/SuccessFraction/Valid` is the primary new selection metric and writes a separate
`*_best_character.pth` checkpoint. It does not replace target-level PR-AUC in the logs.

Campaign-18 tests:

- `baseline`: ordinary windows, character metrics only
- `character_balanced`: uniform character positive/ring pairs
- `depth_warp`: smooth +/-2-slice spatial depth displacement
- `surface_attenuation`: local contrast attenuation near the estimated surface
- `acquisition_blur`: mild in-plane Gaussian point-spread blur
- `correlated_noise`: low-frequency 3D reconstruction-like noise
- `context_cutout`: two small cutouts restricted to the context outside the 16px target center
- `context_jitter`: same global target displaced up to +/-32px inside the 192px context; all
  prediction-local crops consume the transformed offset

All new augmentations default off. Surface attenuation, blur, and correlated noise are explicitly
exploratory: they are implemented as isolated tests, not asserted to match an observed w013 defect.

---

## 19) Standalone Improved Baseline (2026-09-04)

Campaign-17 results:

- feature attention was a marginal positive and is promoted
- surface-guided aggregation was marginal; surface feature injection alone looked qualitatively
  stronger and is promoted
- c32_t8 was materially stronger than c16_t8; c16_t4 was worse

The outer context remained 192px in these runs. `c16`/`c32` refers to prediction-center size, not
input context. A later audit found a sample-count confound: ring-overlap gating yields approximately
11k c16, 20.9k c32, and 27.8k c64 training windows at the same 16px stride. Campaign 19 caps every
arm at 20,000 windows/epoch so c64 does not receive more optimizer steps.

Campaign-18 results through correlated noise:

- character balancing changed the representation dramatically: train macro AP 0.690 -> 0.802 and
  validation macro F1 0.534 -> 0.595
- validation macro AP was approximately flat (0.647 -> 0.643), so ranking separability did not
  improve equivalently
- the fixed-threshold character success fraction fell, demonstrating calibration sensitivity
- depth warp and correlated noise had small favorable final scalar differences, but no augmentation
  separated decisively enough to promote based on one seed

Best-character checkpointing now defaults to threshold-free character macro AP. Success fraction
remains a useful operating-point diagnostic.

Campaign 19 is fully standalone and hardcodes its complete operating point. Baseline:

- w013 manual split, c32_t8, 192/ds2, 20,000 windows/epoch
- feature attention, surface softmax injection, character-balanced sampling
- weak conv/head dropout, IBN, spill, attention entropy, cell SupCon, TTA consistency
- BCE with automatic positive weight, MAE96 warm start
- synchronized flip/rotation only; noise, photometric, FDA, and elastic disabled

Tests:

- c64_t8: same 8px target resolution, 64 targets
- c64_t16: same c64 center but 16 coarser targets
- surface_weak: surface auxiliary lambda 0.03 instead of 0.10
- context_cutout: center-protected cutout
- context_jitter: target-aware +/-32px context displacement

The c64_t8 versus c64_t16 comparison holds center size and optimizer steps fixed while changing
target resolution/density. Baseline versus c64_t16 keeps 16 targets but changes covered center and
target size. It is not a perfect factorial decomposition, but it is substantially cleaner than the
earlier unequal-epoch comparisons.

Multi-scroll character balancing is implemented for future use. It round-robins scroll datasets so
each contributes equally to batches, samples characters uniformly within each scroll, cycles small
scrolls, and retains the original total epoch length. Character IDs are namespaced by domain. This
mode requires compatible per-scroll split configuration; campaign 19 remains w013-only to isolate
the architecture and geometry questions.
