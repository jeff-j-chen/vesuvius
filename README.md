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

Configured zarr root in code:

- /media/jeff/SSD_2/ves_zarrs2/

Present zarrs:

- 20230702185753.zarr
- 20230827161847.zarr
- 20231210132040.zarr

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

## 3) End-to-End Workflow in This Repo

### 3.1 Prepare labels (optional but standard here)

Script: ink_shrinker.py

- Applies morphological erosion to grayscale ink labels
- Default params in script:
	- kernel size: 3
	- iterations: 12

Why erosion is used here:

- tile-level labels are coarse (any-ink tile = positive)
- erosion reduces border spill and over-positive supervision pressure
- helps align objective with your detection strategy

### 3.2 Train tile classifier

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

### 3.3 Evaluate and visualize predictions

Two modes are embedded in visualizer behavior:

- Evaluation figures on train/valid region every eval_int epochs
- Test figures every test_int epochs for:
	- full test region of scroll1 segment (currently full spatial region)
	- sliced scroll4 region (y >= 6500, x <= 5000)

Standalone scroll4 visualization script:

```bash
python scroll4_vis.py -m models/best_model_f1.pth
```

### 3.4 Hard mining loop

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

## 4) Model and Training Details

### 4.1 Core config defaults

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
	- batch_size: 64
	- num_workers: 8
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

### 4.2 Architecture

Source: utils/model.py

- Input shape per sample: (1, depth, 32, 32)
- 3D conv backbone with CBAM3D attention blocks
- global pooling + MLP classifier down to single logit
- final objective: BCEWithLogitsLoss (with optional pos_weight)

### 4.3 Data augmentation

Source: utils/dataloader.py Transform

- random channel mixing
- random 90/180/270 rotations
- random flips
- gaussian noise
- brightness and contrast perturbation

Augmentation activates after epoch 5 in trainer when config.dl.data_aug is true.

### 4.4 Labeling objective mismatch caveat

By design, supervision is tile-level binary (any ink in tile), not pixel-level segmentation.

This means:

- better scalar metrics may not create sharper character shapes
- model can improve confidence calibration while still looking visually noisy
- high scores can look like global brightening if contrast separation does not improve

This is consistent with your observed pain point.

## 5) TensorBoard Logging and What It Means

Logged classes of signals include:

- scalar train/valid loss + raw masked loss
- accuracy, precision, recall, F1, specificity
- ROC-AUC and PR-AUC
- confusion matrix figures
- output score histograms
- radar/bar metric comparisons
- parameter and gradient histograms
- hard-mining overlays and mining-file evaluations

Important interpretation detail:

- these are tile-level classification metrics
- they are not direct character readability metrics
- for your goal, visual readability and high-confidence regional precision should be tracked separately

## 6) Known Issues (Important)

### 6.1 vis.ipynb is broken for multiple reasons

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

### 6.2 finetune path is stale/broken

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

### 6.3 get_data.sh has command typo

File content ends with a stray backtick after tar extraction command.

### 6.4 Seed config contradiction

In train.py set_seed sets:

- torch.backends.cudnn.deterministic = True
- torch.backends.cudnn.benchmark = True

These settings push in opposite directions for strict reproducibility/performance behavior.

## 7) File-by-File Purpose Map

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

## 8) Existing Experiment Artifacts

### 8.1 Checkpoints in models/

- best_model_f1.pth
- best_model_loss.pth
- model_epoch_10.pth through model_epoch_100.pth

Note:

- checkpoints up to epoch 100 exist even though default n_epochs in config is 50
- at least one run used different epoch settings

### 8.2 TensorBoard run folders

- runs/20230702185753/
- runs/20230827161847/
- runs/20230827161847_recurring/
- runs/full_vis_09_182125/

Each contains event files for retrospective metric inspection.

## 9) External Helper Repos (Quick Audit)

You referenced these helper repos:

- /media/jeff/Seagate/vesuvius-docker
- /media/jeff/Seagate/vesuvius-3dstreamer
- /media/jeff/Seagate/vesuvius-zarrs

### 9.1 vesuvius-docker

- Dockerfile uses runpod/pytorch:2.1.0 CUDA 11.8 base
- clones this repo to /vesuvius
- installs requirements, accepts terms, sets git identity
- command currently keeps container alive

### 9.2 vesuvius-3dstreamer

- includes VesuviusStream iterable dataset for sampling 3D chunks
- supports multi-zarr / ndarray sources
- includes multithreaded TIFF->zarr converter with memory controls
- contains conversion command examples in all.sh

### 9.3 vesuvius-zarrs

- includes PNG/TIF stack download/build helpers
- contains multiple prebuilt .npz stacks (large)
- output/ currently has 00.tif..64.tif plus stack.npz
- frag/ contains 00.png..65.png
- file named full is a grayscale PNG image (not a directory)

## 10) Practical Notes for Future Iteration

The major pain point described (metrics move, readability does not) fits this setup's objective/aggregation behavior.

In this codebase, readability depends heavily on:

- tile-level score separation quality, not only global F1/AUC
- depth-window selection and aggregation policy
- region selection for inference targets (especially scroll4)
- hard-mining balance (false-positive suppression vs missed positives)

Given your stated goals (especially text discovery in unannotated regions), treat visual separability metrics as first-class outputs in addition to standard classification metrics.

## 11) Minimal Runbook

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

## 12) Agent Handoff Notes

If an agent opens this repo cold, top priorities are:

1. Verify active data IDs and zarr path in utils/config.py
2. Confirm labels/masks exist for target IDs
3. Treat finetune.py and vis.ipynb as stale until repaired
4. Use train.py + scroll4_vis.py as current canonical executable paths
5. Interpret metric gains against visual readability, not alone
6. Inspect hard_negs/*.jsonl trend to understand failure mode shifts

This README is intended to be updated as the process evolves, especially when strategy changes around scroll4 targeting and tile objective calibration.
