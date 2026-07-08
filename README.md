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
