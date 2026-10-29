# Vesuvius Ink Detector

nnunet3d architecture baseline with some changes. Notably, it receives a 192x192ds2 input to guess at it's 64x64 center. The center is split into 256 targets (4x4) to provide gradients. The model receives 8 slices of depth at a time, centered around the predicted surface (guessed via elastics).

Experiment history and campaign-specific details are kept in `old/KNOWLEDGE.md`.

## Current setup: mix6 + PHerc0846A, straightened2 held out

Everything we train today uses the **mix6 corpus** (`utils/mix6.py`): 7 hand-labeled PHerc0846A sheets plus 8 fragments from six other
scrolls. The PHerc0846A sheet **straightened2** (`auto_grown_20261012065803387_straightened2_abf`, render `20261012065814`, alias `20261012065815`)
is our **prize sheet**. It is rendered and scored over the whole patch, but no model is ever trained or validated on it.

| Alias (label / cue / norm id) | Render (zarr / mask id) | Fragment | Physical scroll | Role |
|---|---|---|---|---|
| `20260927134931` | `20260927134928` | ABF patch (`auto_grown_20260927134928530_abf`) | PHerc0846A | trained |
| `20261012073246` | `20261012073244` | 073244 patch (`auto_grown_20261012073244544_abf`) | PHerc0846A | trained |
| `20261012065813` | `20261012065812` | straight sheet -28 voxels (`..._straight_abf_m28`) | PHerc0846A | trained |
| `20261012065817` | `20261012065816` | straightened2 in_1 | PHerc0846A | trained |
| `20261012065819` | `20261012065818` | straightened2 out_1 | PHerc0846A | trained |
| `20261012065821` | `20261012065820` | straightened2 out_2 | PHerc0846A | trained |
| `20261012065823` | `20261012065822` | straightened2 out_3 | PHerc0846A | trained |
| `20261007000000` | same | w047-w048 first letters (NOT 343P) | PHerc0343 | trained |
| `20250919125754` | same | patch 487 | PHerc0009B | trained |
| `20260930144758` | same | w058 (flat surface, layer 14) | PHerc1447 | trained |
| `20260930144760` | same | w060 (flat surface, layer 15) | PHerc1447 | trained |
| `20260221022814` | same | auto-grown 405 | PHerc0841 | trained |
| `20260226000000` | same | seg46527 | PHerc0814 | trained |
| `20251111010954` | same | w068 | PHerc0172 | trained |
| `20251112000002` | same | w087 | PHerc0172 | trained |
| `20261012065815` | `20261012065814` | **straightened2** | PHerc0846A | **held out (prize sheet)** |

The PHerc0846A sheets are read through flat depth-13 aliases (see [PHerc0846A](#pherc0846a)). The other scrolls use their stored surface labels with window offset 0.

## Quick start

The project normally runs in a Docker image with system Python. No project virtual environment is required for the cue_dropout model
(`train.py`); the reader-v3 and v8-in fine-tunes build their own environments (`reader-v3/.venv`, `nader-fork/.venv`) on first run.

```bash
# everything the mix6 runs read, in one command. It fetches the mix6 scrolls (7 trained PHerc0846A sheets + 343 first letters + 0009B
# + 1447 + 0841 + 0814 + 0172) and the straightened2 holdout, and builds the auxiliary data: masks, norms, surface detections
# (PHerc1447: flat; PHerc0846A: flat depth-13 aliases), surface anchors and 3 depths of asf cues. --check only reports what is missing
python3 assemble_training_segments.py --mix6

# the three mix6 models (each prints where its model is, see the table below)
python3 train.py --dry-run && python3 train.py        # cue_dropout_labelsv6_mix6, 5 epochs (--download runs the data command first)
python3 finetune_readerv3.py                          # reader-v3, 2000 steps
python3 finetune_v8in.py                              # v8-in, 5 epochs
tensorboard --logdir ./runs --bind-all
```

| model | script | output |
|---|---|---|
| cue_dropout_labelsv6_mix6 | `python3 train.py` | `models/runs/cue_dropout_labelsv6_mix6/best_model_loss.pth` (+ `final.pth`, `model_epoch_5.pth`, ...) |
| reader-v3 fine-tune | `python3 finetune_readerv3.py` (clones reader-v3 and the koine_machines training code, builds `reader-v3/.venv`, stages the corpus) | `reader-v3/runs/mix6/ckpt_002000.pth` |
| v8-in fine-tune | `python3 finetune_v8in.py` (clones nader-fork, builds `nader-fork/.venv`, stages the corpus) | `nader-fork/runs/v8in_mix6_5ep/model/` |

All three models train on the same corpus and none of them sees straightened2. `--smoke` on any of the three training scripts runs a tiny version into
separate output folders.

`python3 train.py` trains `cue_dropout_labelsv6_mix6` (`apply_mix6_run` in `utils/config.py`, see [The mix6 run](#the-mix6-run-cue_dropout_labelsv6_mix6)).
Checkpoints go to `models/runs/<run>/` and TensorBoard logs to `runs/<run>/`. An existing run is never overwritten (`_2`, `_3`, ...).
Experiment runners call `train.main(configure)` to edit the plain `Config()` instead. Earlier campaign scripts live in `old/`.

Other runners:

- `python3 k_folds.py`: cross-validation and leave-patches-out arms of the earlier best run `iter2_asf_cue` (see [Cross-validation](#cross-validation))
- `python3 train_iter2.py --arm <arm>`: iteration 2, the 192 px baseline on the PHerc0846A set and its single-change arms (logs in `runs_iter2/`). Its `asf_cue` arm became the base of the mix6 run
- `python3 train_0175A.py`: rotation-augmented arms that try to read the unlabeled PHerc0175A quick patches more cleanly, starting from the 0175A MAE (`mae_pretrain_0175a.py`)
- `python3 assemble_training_segments.py` without `--mix6`: every training fragment, including the 8 mix6 PHerc0846A sheets (7 trained + straightened2)
- `python3 assemble_test_segments.py`: the test surfaces and the PHerc0846A renders outside mix6 (REDO set, out_4/out_5, ...)

## Evaluation notebooks

| Notebook | Role |
|---|---|
| `ensemble_eval.ipynb` | Ensembles the three mix6 models on straightened2 (and other sheets in `SHEETS`). Saves a 3x2 grid: released reader-v3 (comparison only) / reader-v3 fine-tune / cue_dropout / v8-in / z-score composite / mid-boost composite. Output goes to `output/ensemble/mix6/<sheet>/` |
| `0846A_eval.ipynb` | Runs one of the three mix6 models (the kind is detected from `CHECKPOINT`) on straightened2, the 7 trained sheets, the 8 other mix6 scrolls, the held-out out_4/out_5 and the PHerc0846A quick patches. See [0846A notebook](#0846a-notebook) |
| `eval_scrolls.ipynb` | Runs a cue_dropout checkpoint on every labeled training fragment. Useful for iterative relabeling and sanity checks |
| `test_inference.ipynb` | Runs a cue_dropout checkpoint on the test surfaces |

`eval_scrolls` and `test_inference` read `CHECKPOINT` (`VESUVIUS_CHECKPOINT` overrides it) and replay the checkpoint's saved run config.
The mix6 model files are listed in `utils/mix6.py` `MODEL_PATHS`.

The cue_dropout notebooks start with two depth-ensemble options:

- `depth_ensemble = []` predicts at one depth; `[low, high]` predicts at every offset `low..high` around the notebook's usual depth (`[-3, 3]` = 7 depths)
- `ensemble_method` combines them: `'z-score'` (top-2, scaled to 0..1 by its p99.5), `'max'`, `'mid-boost'` (corroborated, k=3) or `'mean'` (top-2); empty means `'mid-boost'`

The combined map replaces the single-depth prediction in the usual figure (no extra panels or files) and the figure
title names the method. The methods live in `utils/depth_ensemble.py`. Cells that already sweep or combine depths
(the ±2 shift panels of `test_inference`) ignore these options.

`test_inference.ipynb` covers 13 test patches from 10 test scrolls (PHerc1447, PHerc0800 and PHerc0211 have two surfaces each), all generated by me. For each fragment it renders `raw inference | inference | fiber composite | composite with 1 cm scale bar` into `output/test_visualizations/<run_id>/`. Missing zarrs, masks, norms and surface labels are generated on demand. The figure needs some small edits before it can be submitted for a prize.

## Fragments

`assemble_training_segments.py` brings every fragment onto the same training grid: 28 layers at 9.362µm. **Scan** is the CT volume the surface is sampled from (voxel size / beam energy). **Rendering** says how it got onto the grid. The tables below list the full corpus of the plain `Config()`: the 24 fragments used by Campaign 40, the PHerc1447 fine-tune windings and the first PHerc0846A patches. The mix6 run trains only the subset in [Current setup](#current-setup-mix6--pherc0846a-straightened2-held-out); see [PHerc0846A](#pherc0846a) for all PHerc0846A renders. The same fragments, in the same groups, appear in the scroll visual guide (`python3 generate_scroll_visual_guide.py` → `output/scroll_patch_guide.png`). Splits come from `train_masks/` (see [Splits and character sampling](#splits-and-character-sampling)).

| ID | Fragment | Physical scroll | Scan | Rendering | Notes |
|---|---|---|---|---|---|
| `20260115000000` | w044 | PHerc0139 | 9.362µm / 113keV | Native surface volume | Established evaluation fragment |
| `20260317000000` | w035 | PHerc0139 | 9.362µm / 113keV | Native surface volume | Separately downloaded 1.1µm labels |
| `20250223000000` | w059 | PHerc0139 | 9.362µm / 113keV | Native surface volume | Mask restricted to labeled footprint |
| `20250108000005` | w030 | PHerc0139 | 9.362µm / 113keV | Native surface volume | Official researcher labels (2026-09-18 release) |
| `20260112000000` | w043 | PHerc0139 | 9.362µm / 113keV | Native surface volume | Official researcher labels (2026-09-18 release) |
| `20260126000000` | w045 | PHerc0139 | 9.362µm / 113keV | Native surface volume | Official researcher labels (2026-09-18 release) |
| `20250831000000` | w040 | PHerc0139 | 9.362µm / 113keV | Native surface volume | Full papyrus footprint |
| `20260108000000` | w041 | PHerc0139 | 9.362µm / 113keV | Native surface volume | Full papyrus footprint |
| `20260302000000` | w039 | PHerc0139 | 9.362µm / 113keV | Native surface volume | Full papyrus footprint |
| `20251111010954` | w068 | PHerc0172 | 7.91µm / 53keV | Isotropic XYZ resample to 9.362µm | |
| `20251112000002` | w087 | PHerc0172 | 7.91µm / 53keV | Isotropic XYZ resample to 9.362µm | |
| `20240304141531` | w013 | PHerc1667 | 2.399µm / 78keV | Level-2 XY (~9.6µm); 109 depths pooled to 28; left 25% crop | Pooled high-resolution surface |
| `20240304144031` | w018 | PHerc1667 | 2.399µm / 78keV | Level-2 XY (~9.6µm); 109 depths pooled to 28 | Full-size pooled surface |
| `20231201215900` | Cr1 Fr3 | PHerc1667 | 3.24µm / 70keV | dl.ash2txt flattened TIFF stack resampled in XYZ | |
| `20250919125754` | patch 487 | PHerc0009B | 8.64µm / 116keV | Isotropic XYZ resample to 9.362µm | |
| `20231210121321` | Paris4 | PHercParis4 | 2.4µm / 78keV | Level-2 XY (~9.6µm); 109 depths pooled to 28 | |
| `20250628074500` | 500P2 front | PHerc0500P2 | 9.362µm / 113keV | Native surface volume (mesh traced on the 2.215µm / 111keV scan) | Clear labels transferred from 2.215µm |
| `20260226000000` | seg46527 | PHerc0814 | 9.362µm / 113keV | Native surface volume | Eroded label only; no continuous label |
| `20230301213755` | Fr143 | PHercParis2 | 3.24µm / 88keV | **High-energy rescan** through the 54keV exposed-surface PPM (fitted affine + deformation field) | Fragment deformed between scans |
| `20230205142449` | Fr47 | PHercParis2 | 3.24µm / 88keV | **High-energy rescan** through the 54keV exposed-surface PPM (fitted affine) | Researcher inklabel copied unchanged |
| `20231205222200` | Cr4 Fr8 | PHerc51 | 3.24µm / 88keV | **High-energy rescan** through the 53keV surface PPM (fitted affine) | |
| `20230301213423` | Fr34 | PHercParis1 | 3.24µm / 88keV | **High-energy rescan** through the 54keV exposed-surface PPM (fitted affine) | |
| `20250511003658` | tifxyz segment | PHerc0343P | 8.64µm / 116keV | 31-layer surface volume resampled in XYZ to 28 layers | |
| `20261007000000` | w047-w048 letters | PHerc0343 | 8.64µm / 116keV | Community tifxyz mesh (`concat_w047-w048_R5B2_z9500-11000`) rendered and resampled in XYZ to 9.362µm | Hand-drawn letters |
| `20260221022814` | auto-grown 405 | PHerc0841 | 9.366µm / 113keV | Native surface volume, upstream re-crop (2026-09-22) to 3760×4900; copied without resampling (0.04% off the 9.362µm grid) | |
| `20260930144758` | w058 | PHerc1447 | 8.64µm / 116keV | Hugging Face 24-layer render; depth reversed, resampled in XYZ to 28 layers | Released `labels/inklabels.png`; fine-tune only |
| `20260930144760` | w060 | PHerc1447 | 8.64µm / 116keV | Hugging Face 24-layer render; depth reversed, resampled in XYZ to 28 layers | Released `labels/inklabels.png`; fine-tune only |
| `20260930144762` | w062 | PHerc1447 | 8.64µm / 116keV | Hugging Face 24-layer render; depth reversed, resampled in XYZ to 28 layers | Pseudo-label from held-out `loo_w062.png`; no train mask |
| `20260927134931` | ABF patch | PHerc0846A | 9.362µm / 113keV | tifxyz mesh render (`20260927134928`), flat depth-13 alias | Hand-drawn labels; train-only |
| `20261012073246` | 073244 patch | PHerc0846A | 9.362µm / 113keV | tifxyz mesh render (`20261012073244`), flat depth-13 alias | Hand-drawn labels; train-only |
| `20261012065813` | straight -28 (m28) | PHerc0846A | 9.362µm / 113keV | tifxyz mesh render (`20261012065812`), flat depth-13 alias | Hand-drawn labels; train-only |
| `20261012065817` | straightened2 in_1 | PHerc0846A | 9.362µm / 113keV | tifxyz mesh render (`20261012065816`), flat depth-13 alias | Hand-drawn labels; train-only |
| `20261012065819` | straightened2 out_1 | PHerc0846A | 9.362µm / 113keV | tifxyz mesh render (`20261012065818`), flat depth-13 alias | Hand-drawn labels; train-only |
| `20261012065821` | straightened2 out_2 | PHerc0846A | 9.362µm / 113keV | tifxyz mesh render (`20261012065820`), flat depth-13 alias | Hand-drawn labels; train-only |
| `20261012065823` | straightened2 out_3 | PHerc0846A | 9.362µm / 113keV | tifxyz mesh render (`20261012065822`), flat depth-13 alias | Hand-drawn labels; train-only |
| `20250825181859` | 500P2 -1 | PHerc0500P2 | 9.362µm / 113keV | Native surface volume, 28×5880×3300 (mesh traced on the 2.215µm / 111keV scan) | NEW (2026-10-28), not in `Config()`; starter label from the official 2.215µm ink-labels zarr; mask 60.1% valid, label 2.8% ink, reg. NCC 0.87 |
| `20250825181917` | 500P2 -2 | PHerc0500P2 | 9.362µm / 113keV | Native surface volume, 28×5520×3240 (mesh traced on the 2.215µm / 111keV scan) | NEW (2026-10-28), not in `Config()`; starter label = thresholded 2.215µm ink detection; mask 59.5%, label 8.9%, NCC 0.87 |
| `20250902170441` | -4_b2 | PHerc0343P | 8.64µm / 116keV | 31×3300×2520 surface volume resampled in XYZ to 28×3046×2326 | NEW (2026-10-28), not in `Config()`; starter label = thresholded 2.215µm ink detection; mask 62.7%, label 3.2%, NCC 0.80 |
| `20250919132115` | patch 592 | PHerc0009B | 8.64µm / 116keV | 31×7240×6560 surface volume resampled in XYZ to 28×6682×6054 | NEW (2026-10-28), not in `Config()`; starter label = thresholded 2.401µm ink detection; mask 74.0%, label 9.2%, NCC 0.62 |
| `20260220213127` | w00 | PHerc0841 | 9.366µm / 113keV | Native surface volume, 28×4220×4760, copied without resampling | NEW (2026-10-28), not in `Config()`; starter label from the official 2.403µm ink-labels zarr; mask 71.8%, label 2.0%, NCC 0.77 |
| `20250925222237` | auto-grown 459 | PHerc0814 | 9.362µm / 113keV | Native surface volume, 28×3920×11900 | NEW (2026-10-28), not in `Config()`; starter label = thresholded 2.399µm ink detection; mask 40.0%, label 5.7%, NCC 0.87 |
| `20260422000000` | title | PHerc0139 | 9.362µm / 113keV | Native surface volume, 28×1780×5360 | NEW (2026-10-28), not in `Config()`; starter label = thresholded 2.399µm ink detection; mask 84.9%, label 10.6%, NCC 0.63 |

**PHerc0343 and PHerc0343P are NOT the same scroll.** They are physically unrelated scrolls that happen to have similar names. Never group, merge or treat them as one domain: the w047-w048 first letters (`20261007000000`) are PHerc0343; the tifxyz segment (`20250511003658`) and -4_b2 (`20250902170441`) are PHerc0343P.

### Special rendering

- **Native 1.2m surface volumes (9.362µm, 9.366µm, 8.64µm).** The published surface volumes are 2× bin-mean renders of 4.681µm / 4.683µm (113keV) and 4.32µm (116keV) ESRF scans (`source_zarr` in each volume's `.zattrs`).
- **High-energy rescans (Fr143, Fr47, Cr4 Fr8, Fr34).** The labels were drawn on the low-energy (53/54keV) exposed-surface renders. We sample the 88keV scan instead, so these fragments sit closer to the high-energy test scans. See [High-energy fragments](#high-energy-fragments). Chunks missing from the published 88keV zarr are rebuilt from the raw slice TIFFs.
- **Pooled 2.4µm surfaces (w013, w018, Paris4).** Level 2 of the pyramid gives 9.596µm (w013, w018) or 9.6µm (Paris4) in XY, kept as is. 109 native depth samples are pooled to 28 layers (~9.34µm in Z). The meshes were traced on older scans (3.24µm for PHerc1667, 7.91µm for Paris4).
- **Resampled scans (PHerc0172, PHerc0009B, PHerc0343P, PHerc1447, Cr1 Fr3).** Resampled from the native voxel size to 9.362µm: linear interpolation in depth, area averaging in XY (linear when upsampling).
- **PHerc1447 windings.** These come from the Hugging Face dataset `YoussefMoNader/ink-8um-pherc1447-surfaces` and get synthetic IDs. Their mesh normal points the opposite way, so the 24 layers are written in reversed order (`depth_reversed.json`, `depth_flipped=true`). The labels are copied into both `inklabels/` and `dilated_inklabels/` after XY resampling. `campaign_finetune_96.py` uses w058 and w060 with manual splits from `train_masks/`.
- **2026-10-28 additions (500P2 -1/-2, 343P -4_b2, 0009B 592, 0841 w00, 0814 459, 0139 title).** Assembled by `assemble_training_segments.py` (default download path; 343P and 0009B use the 8.64µm XYZ resample). Starter labels come from `old/new_segment_labels_20261028.py`: the official `ink-labels/` zarr (500P2 -1, 0841 w00) or the `ink-detection/` tif, registered from the fine-scan render onto our frame (SIFT affine + displacement field, tile NCC 0.62-0.87, registration stats in `output/new_segments_20261028/labels_manifest.json`), thresholded at 0.5 into `inklabels/` + `dilated_inklabels/`; the soft detection map is in `inklabels/2_4um/`. `train_masks/` are all zero. Not used by any training config or the correlation pipeline yet.
- **Offset fragment surfaces (`offset_fragment_surfaces.py`).** The separated fragments (500P2 front/-1/-2, Fr143, Fr47, Cr4 Fr8, Fr34, Cr1 Fr3) re-rendered 29 layers (271µm) deeper, towards the denser side picked from a normal intensity profile. IDs `<zid>_in29`; reader-v3 maps in `output/offset29/<id>/`. Readability check only, not training segments.

### Legacy PHerc0139 windings

These are still assembled (9.362µm / 113keV, native surface volumes). They have no `dilated_inklabels/` entry, so current campaigns and the visual guide skip them.

| ID | Fragment | Notes |
|---|---|---|
| `20260206000001` | w047 | Mask restricted to labeled footprint |
| `20260115000001` | w056 | Mesh-rendered from the raw volume (`old/render_9um_surface.py`) |
| `20260210000000` | w058 | Full papyrus footprint |
| `20260227000000` | w052 | Full papyrus footprint |
| `20260318000000` | w049 | Full papyrus footprint |
| `20260325000000` | w046 | Full papyrus footprint |
| `20260306000000` | w038 | Full papyrus footprint |
| `20260310000000` | w037 | Full papyrus footprint |
| `20260303000000` | w034 | Full papyrus footprint |

### Holdout

| ID | Fragment | Physical scroll | Scan | Purpose |
|---|---|---|---|---|
| `20251226000000` | w055 | PHerc0139 | 9.362µm / 113keV | Deliberately excluded from supervised training; checks whether a model reproduces known text rather than familiar texture |

## PHerc0846A

PHerc0846A is the submission scroll, and its sheet straightened2 is the prize sheet. Every patch is rendered from its tifxyz mesh and the
9.362µm / 113keV volume `20250728152254`. It is read through a **flat depth-13 alias**:
`<alias>.zarr` and the mask / label / train-mask PNGs are symlinks to the source render, and
`surface_labels/<alias>/` is a constant depth 13 over the mask (no detected surface). With
`surface_window_offset = -1` the model reads slices 9-16. Aliases get their own norm and surface-anchor entries.
Hand-drawn labels are in `dilated_inklabels/`, named after the alias.

Two scripts assemble these renders:

- **the 8 mix6 sheets** (7 trained + straightened2): `assemble_training_segments.py` (`MESHES_0846A`). A plain run includes them and so does `--mix6`. The script renders each one with `assemble_test_segments.render_fragment` and builds its alias with `make_flat_alias`. These sheets skip label checks, surface detection and degradation
- **every other PHerc0846A render** (out_4/out_5, 081237, the straight sheets 065810 / 084610, the REDO set): `assemble_test_segments.py` (`FRAGMENTS` + `FLAT_ALIASES`). The script refuses to start if a `FLAT_ALIASES` source is missing from `FRAGMENTS`

| Render | Alias | Mesh (`tifxyz/`) | Assembler | mix6 |
|---|---|---|---|---|
| `20260927134928` | `20260927134931` | `quick_patches_from_D/PHerc0846A/auto_grown_20260927134928530_abf` (ABF) | training | trained |
| `20261012073244` | `20261012073246` | `3_new/auto_grown_20261012073244544_abf` | training | trained |
| `20261012065812` | `20261012065813` | `auto_grown_20261012065803387_straight_abf_m28` (the straight mesh -28 voxels along its normal, `old/offset_mesh.py`) | training | trained |
| `20261012065816` | `20261012065817` | `auto_grown_20261012065803387_straightened2_abf_in_1` (one sheet inward of straightened2) | training | trained |
| `20261012065818` | `20261012065819` | `auto_grown_20261012065803387_straightened2_abf_out_1` | training | trained |
| `20261012065820` | `20261012065821` | `auto_grown_20261012065803387_straightened2_abf_out_2` | training | trained |
| `20261012065822` | `20261012065823` | `auto_grown_20261012065803387_straightened2_abf_out_3` | training | trained |
| `20261012065814` | `20261012065815` | `auto_grown_20261012065803387_straightened2_abf` (**straightened2, the prize sheet**) | training | **held out**, rendered and scored over the whole patch |
| `20261012065824` | `20261012065825` | `auto_grown_20261012065803387_straightened2_abf_out_4` | test | not used (whole-patch validation entry only) |
| `20261012065826` | `20261012065827` | `auto_grown_20261012065803387_straightened2_abf_out_5` | test | not used (whole-patch validation entry only) |

- the straightened2 sheets and `20261014084610` have upside-down meshes: their coordinate grids are rotated 180° in-plane before sampling (`ROTATE_180`, zarr attribute `rotated_180`), so every stored array is upright and letters are not mirrored
- display only: the 073244 render/alias is shown rotated 90° CCW and the ABF patch 180° (`utils/fast_figure.display_rotation`); arrays and metrics stay in the native frame

### The mix6 run (`cue_dropout_labelsv6_mix6`)

`apply_mix6_run` (`utils/config.py`) turns the plain `Config()` into what `python3 train.py` trains:

- corpus: the mix6 corpus ([Current setup](#current-setup-mix6--pherc0846a-straightened2-held-out)). The seven PHerc0846A sheets form domain `pherc0846a` (weight 2, draws per epoch fixed by `equalize_domain`). PHerc0009B, PHerc1447, PHerc0841, PHerc0814, PHerc0172 and PHerc0343 are one domain each (weight 1). The 0846A sheets are train-only (no validation split; a train mask only adds its painted explicit labels) and read at flat depth 13 with window offset -1. The other scrolls train on their `train_masks/` region with their stored surface labels
- straightened2 is rendered in the end-of-training figures (with the seven trained sheets) and scored over its whole patch. It is never trained or validated on
- half-brightness strokes in the PHerc0846A labels become hard negatives (`extra_hard_negative_scroll_ids`)
- 192 px field (batch 32, lr 1e-4), dual scale (mix 0.5), strong nnPU (`pu_lambda` 3, with 50% of the negative draws taken at least 96 px from any label), text-region gate, ring close 2 / gap 2 / shell 4, the 192 px native-noise bank, seed 17, 5 epochs (2 warmup)
- a fourth gated input head, next to raw / lcn / dz, reads the precomputed asf_morph map (`model.input_cue = "pre_clahe"`, `data.cue_dir = ./cue_asf_morph`) and trains at 5× the learning rate. In 30% of the samples the cue is replaced by a flat map (cue dropout). Depth jitter reads the shifted maps `cue_asf_morph_dm1` / `_dp1`. `assemble_training_segments.py --mix6` builds all three depths (the holdout only gets depth 0)
- starts from `models/init_mae192_cue4_seed17.pth`, which is the 192 px campaign-40 MAE grown to four input heads (`utils/cue_expand.py`). `train.py` builds it when it is missing

The recipe is the earlier best run `iter2_asf_cue` (`apply_best_run`: the 0846A set plus PHerc0009B, 10 epochs) with the cue dropout added, on the larger corpus.

### Cross-validation

`k_folds.py` runs arms of `iter2_asf_cue` (`apply_best_run`), 10 epochs each. PHerc0009B always trains.

- `cv_fold_1` .. `cv_fold_3`: every PHerc0846A patch trains, but in each fold a third of its letters (8-connected ink components) and the papyrus nearest to them become validation; every letter is held out exactly once
- `loo_*`: PHerc0009B plus two named 0846A patches train, and the other five are scored over the whole patch: `loo_A_out2_out3`, `loo_B_in1_out1`, `loo_C_073246_abf`, `loo_D_m28_out3`

Every arm renders the same figures as `iter2_asf_cue`. TensorBoard, figures and `config.json` go to
`runs_kfolds/<arm>/`, console logs to `runs_kfolds/logs/`, checkpoints to `models/runs_kfolds/<arm>/`, fold masks and
previews to `runs_kfolds/splits/`. `--only cv_fold_1,...` selects arms; `--dry-run` checks inputs.

### 0846A notebook

`0846A_eval.ipynb` runs one of the three mix6 models on straightened2, the mix6 scrolls and the PHerc0846A quick patches (`assemble_test_segments.py --include-extras`):

- settings cell: `CHECKPOINT` (any of the three; the kind is detected from the file by `utils/predict_models.detect_kind`), the depth-ensemble options and `DEPTH_OFFSET = -1` (cue_dropout only: slices 9-16 of the flat depth-13 surface). For cue_dropout the run config is replayed, including its input heads; a long-lived kernel must be restarted if the 4-head check fails
- each model reads a scroll its native way: cue_dropout reads 8 slices and predicts one value per 16 px cell, reader-v3 reads the 17 central layers and v8-in the 24 central layers, both predicting per pixel. The two per-pixel models run in their own virtualenvs (`utils/predict_models.py`)
- one cell per patch, in order: straightened2 (held out), the seven trained 0846A sheets, the eight other mix6 scrolls (stored surface labels, window offset 0), then the held-out out_4, out_5 and three quick patches. Trained patches get a 2×2 figure (prediction, composite, prediction with the inklabel in red, asf_morph cue); all others get prediction | composite. Figures go to `output/0846A/<model>/`
- the last cell renders the remaining quick patches into one `quick_patches/` subfolder

## Test surfaces

The thirteen VC3D-grown test surfaces are the final discovery targets. They come from physical scrolls outside the supervised corpus. The same thirteen IDs are registered in `DEFAULT_TEST_SCROLL_IDS` (`utils/config.py`), `FRAGMENTS` in `assemble_test_segments.py`, and `FRAGMENTS` in `test_inference.ipynb` (notebook keys are the tifxyz names, values are the zarr IDs).

Shape `(H,W)` and mask coverage are measured from the current files in `masks/`; every zarr has 28 layers. Surface area is geometric mesh area from `meta.json`; `~` values are approximate because merged surfaces lack a reliable `area_cm2`.

| # | ID | Physical scroll | Shape `(H,W)` | Mask coverage | Surface area | Scan | tifxyz mesh |
|---|---|---|---:|---:|---:|---|---|
| 1 | `20260814140748` | PHerc0813 | `(5081, 5701)` | 43.4% | 33.31 cm² | 9.362µm / 113keV | `auto_grown_20260814140748456_flatboi` (updated patch) |
| 2 | `20260717193517` | PHerc0211 | `(7181, 6501)` | 70.5% | ~28.60 cm² | 9.362µm / 113keV | `auto_grown_20260717193517520_0_1_2_3_4_merged_flatboi` (five patches) |
| 3 | `20260720090842` | PHerc1203 | `(3421, 3821)` | 65.1% | 7.90 cm² | 9.362µm / 113keV | `auto_grown_20260720090842117_flatboi` |
| 4 | `20250703034159` | PHerc1447 | `(7181, 9401)` | 70.0% | 51.27 cm² | 8.640µm / 116keV | `20250703034159_flatboi` |
| 5 | `20260723112922` | PHerc0826 | `(5981, 3201)` | 60.0% | ~18.93 cm² | 9.362µm / 113keV | `auto_grown_20260723112922652_merged_flatboi` |
| 6 | `20260921094413` | PHerc0846A | `(5021, 7721)` | 63.3% | 22.88 cm² | 9.362µm / 113keV | `auto_grown_20260921094413486_flatboi` |
| 7 | `20260918132724` | PHerc0175A | `(4501, 5141)` | 61.0% | 11.95 cm² | 8.640µm / 116keV | `auto_grown_20260918132724424_flatboi` |
| 8 | `20260922073234` | PHerc0306B | `(5121, 6181)` | 56.9% | 14.13 cm² | 8.640µm / 116keV | `auto_grown_20260922073234974_flatboi` |
| 9 | `20260922161631` | PHerc0800 | `(6921, 8081)` | 55.9% | ~23.36 cm² | 8.640µm / 116keV | `auto_grown_20260922161631422_merged_flatboi` |
| 10 | `20260925085345` | PHerc1447 | `(4901, 4621)` | 64.3% | 10.90 cm² | 8.640µm / 116keV | `auto_grown_20260925085345806_abf` (second surface) |
| 11 | `20260928000001` | PHerc0800 | `(2321, 4141)` | 77.4% | 5.50 cm² | 8.640µm / 116keV | `PHerc0800_z9264_w040_abf` (expanded auto-winder segment) |
| 12 | `20260928000002` | PHerc0125 | `(1161, 3141)` | 77.3% | 2.50 cm² | 9.362µm / 113keV | `PHerc0125_z15344_w040_abf` (expanded auto-winder segment) |
| 13 | `20260928000003` | PHerc0211 | `(1941, 3641)` | 53.5% | 26.229 cm² | 9.362µm / 113keV | `PHerc0211_z7312_w080_abf` (expanded auto-winder segment) |

`assemble_test_segments.py` renders each surface from its tifxyz coordinates and the listed raw CT volume; the scan column is the volume the mesh was built on. Every render lands on the 9.362µm training grid (in-plane pixels and depth steps): 9.362µm volumes are nearest-sampled, 8.640µm volumes are trilinearly resampled, like the 8.64µm training fragments. Zarrs record `voxel_um`/`source_voxel_um`; shapes in the table above for 8.640µm surfaces predate this (~8% larger), and stale surface labels, norms and composites are dropped on re-render.

### Flattened meshes

`tifxyz/` holds only the SLIM-flattened `_flatboi` meshes (`x/y/z.tif` + `meta.json`, no `mask.tif`); the unflattened originals were removed. They were produced headlessly with the patched VC3D build in `~/Downloads/squashfs-root/usr/bin`, reproducing the GUI's SLIM-flatten action (keep 1.5%, 50 iterations, `symmetric_dirichlet`, tol 1e-5):

```bash
vc_tifxyz2obj <seg> <seg>_coarse.obj --keep=1.5
flatboi <seg>_coarse.obj 50 symmetric_dirichlet --tol=1e-05
vc_tifxyz2obj <seg> <seg>_fine.obj
vc_obj_uv_lift <seg>_coarse.obj <seg>_coarse_flatboi.obj <seg>_fine.obj <seg>_lifted.obj
vc_obj2tifxyz <seg>_lifted.obj <seg>_flatboi --tifxyz-source=<seg>
```

Notes:

- `vc_obj2tifxyz` copies the source `meta.json`, so `area_cm2` in `_flatboi` folders is inherited, not recomputed
- PHerc0800 (`auto_grown_20260922161631422_merged_flatboi`) was flattened in the VC3D GUI; the other eight headlessly
- the second PHerc1447 surface (`auto_grown_20260925085345806_abf`) is used as supplied, without a `_flatboi` pass, and keeps its `mask.tif`
- test surfaces 11-13 are expanded versions of the atlas extras `PHerc0800_z9264_w040`, `PHerc0125_z15344_w040` and `PHerc0211_z7312_w080`, used as supplied; their numeric IDs are synthetic, and their areas are computed from the mesh because `meta.json` has no `area_cm2` or a stale one
- `assemble_test_segments.py` reads the `_flatboi` meshes, but zarrs in `ves_zarrs2/` rendered before 2026-09-23 came from the unflattened meshes; re-render with `--force`

### Extra segment sets

Besides the numbered test surfaces, `assemble_test_segments.py --include-extras` renders two unlabeled extra sets for broader test-time review. They are keyed by their folder name and stored as uint8. Their masks go in `masks/extras/`, surface labels in `surface_labels/extras/` and composites in `output/composite_cache/extras/`. None of these are committed: each set's surface labels and composites are synced through R2 as `<set>_surface_labels.zip` and `<set>_composites.zip`. Extras are not in `DEFAULT_SCROLLS` or `DEFAULT_TEST_SCROLL_IDS`, so MAE pretraining never samples them.

- `atlas`: auto-winder meshes, downloaded from R2 as `atlas.zip` into `tifxyz/atlas/`
- `quick_patches_from_D`: meshes tracked in git under `tifxyz/quick_patches_from_D/<scroll>/<segment>/`. Any segment folder added there is assembled automatically, provided its scroll has a raw volume in `EXTRA_VOLUMES`.

| Scroll | quick_patches_from_D segments | Scan |
|---|---:|---|
| PHerc0846A | 37 | 9.362µm / 113keV |
| PHerc0175A | 25 | 8.640µm / 116keV |

A few PHerc0846A meshes reach up to ~4% past the top of the scan; those points are clamped to the last slice.

```bash
# render missing extras, then pull or generate their surface labels + composites
python3 assemble_test_segments.py --include-extras
# re-zip and upload every set's surface labels + composites (needs R2_PUBLIC_URL + ACCESS_KEY/SECRET_KEY)
python3 assemble_test_segments.py --include-extras --upload-extras
```

Do not combine `--only` with an R2-configured `--include-extras` run: the automatic re-upload would replace a set's R2 zips with only the selected segments.

## Data layout

### High-energy fragments

Paris1 Fr34, Paris2 Fr143, Paris2 Fr47 and Cr4 Fr8 are assembled from their 88keV scans by default
(`RESCAN_88KEV` in `assemble_training_segments.py`, `utils/fragment_rescan.py`). The low-energy
exposed-surface `result.ppm` gives per-pixel positions and normals; a fitted affine (no registration is
published) maps them into the 88keV volume, which is sampled at 28 layers on the old surface-TIFF frame,
so labels, masks and splits carry over. Chunks missing from a published 88keV zarr are rebuilt from the
raw slice TIFFs. Fr143 deformed between scans, so it also applies a smooth residual field
(`rescan_fields/20230301213755.npy`, from local 3D matches between the scans). A 48 px band of extrapolated
surface is rendered past the mesh edge as context (as the old TIFF volumes had); it is outside the mask.
An existing zarr without the `rescan_88kev` attribute is superseded on the next run.

By default these four are not re-rendered: the assembler downloads pre-rendered `<id>.tar.gz` archives
(zarr + mask, checked against `SHA256SUMS`) from `$R2_PUBLIC_URL/fragments_88kev/`, which takes seconds.
`R2_PUBLIC_URL` must be set (`export R2_PUBLIC_URL=https://pub-<hash>.r2.dev`). Pass `--no-r2` to render
from dl.ash2txt instead (~1 h; the server serves ~20 MB/s); re-upload the archive afterwards so the bucket
matches.
The replaced 54keV surface supervision, masks and norms are kept in `backup_54kev_fragments/`.

Fr47 (`20230205142449`) is named after the 54keV volume its surface was traced on. Its researcher inklabel
is scaled into `inklabels/` and copied unchanged into `dilated_inklabels/`.

### Artifacts
Each labeled fragment has four main artifacts:

- `ves_zarrs2/<id>.zarr` — 28-layer surface volume
- `masks/<id>.png` — valid papyrus footprint
- `inklabels/<id>.png` — mask-clipped radius-12 dilation of the conservative target
- `eroded_inklabels/<id>.png` — conservative binary training target

Generate regular labels from the conservative targets with a radius-12 dilation:

```bash
python3 old/ink_shrinker.py dilate --input-folder eroded_inklabels \
	--output-folder inklabels --radius 12 --mask-folder masks
```

After creating the Paris1 Fr34 conservative label, generate only its regular label with:

```bash
python3 old/ink_shrinker.py dilate --input-folder eroded_inklabels \
	--output-folder inklabels --radius 12 --mask-folder masks --only 20230301213423
```

Archive the supplied researcher references for the three newest fragments separately from
the volume downloader:

```bash
python3 archive_research_inklabels.py p343 cr1fr3 p841
```

Manual train masks use exact grayscale values: `255` for the normal training region,
`185` for explicit positives, `119` for explicit negatives, and `0` for unassigned pixels.

Literal surface maps are stored under `surface_labels/<id>/` as `depth.npy`, `confidence.npy`, and `metadata.json`.

Published whole-scroll 3D ink predictions can be sampled onto a patch's 28-layer grid. Only the
ink chunks intersected by the patch are downloaded, and they are deleted after resampling:

```bash
python3 extract_ink3d_patch.py            # Paris4 defaults -> ink3d_labels/20231210121321.zarr
python3 generate_surface_supervision.py --scroll-id 20231210121321 --overlays-only \
	--ink3d-zarr ink3d_labels/20231210121321.zarr --view-all-layers
```

By default the script writes only `output/surface_review/<id>/surface_depth_overview.jpg`; per-depth layer views require `--view-all-layers` (implied by `--review`). The layer views show the guessed surface in red and 3D ink in white. The ink-to-surface offset
histogram is written to `output/surface_review/<id>/ink_surface_alignment.json`.

## Labels and sampling

### Eroded labels

Files in `eroded_inklabels/` are the authoritative training labels. These are hand drawn by me, based on outputs of models (researchers' and my own). Not to be generated by agents.

Files in `inklabels/` are retained for alignment checks and figures. They are not the default training target.

### Ring negatives

The dataset does not treat all unlabeled papyrus as negative. Negative samples come from a narrow closed ring around known ink (radii in 16 px tiles):

- close radius: 2 (letter holes closed)
- gap radius: 2 (unsupervised band between letter and ring)
- shell radius: 4 (negative ring width)

This concentrates training near the ink boundary and reduces the chance of teaching the model that unknown ink is blank papyrus. Ink-free ring windows provide negative supervision.

**Note**: The ring dataset is necessary to allow for bad/uncertain labels to not punish the model. There is a buffer zone between letter and papyrus where the model can guess anything; this allows letter shapes to adapt, while not labelling ink as papyrus. When labels are better, we will move to tighter gradients (proven better).

### Splits and character sampling

Experiments may use axis-aligned splits from `utils/config.py` or explicit masks in `train_masks/`. Explicit masks are better. Training is marked in full white, forced negatives are gray, anything else is validation.

Connected components in an eroded label are treated as characters. Character-balanced sampling rotates through characters and pairs positive targets with nearby ring-negative windows. Components crossing a split are excluded to prevent leakage.

Multi-scroll experiments can group fragments by physical scroll. The sampler rotates through fragments within each group before applying the configured physical-scroll weights.

## Normalization

Normalization is computed once per fragment over valid papyrus pixels only. `utils/norm.py` reads each zarr in chunk-aligned bands and records:

- mean
- standard deviation
- normalized minimum
- normalized maximum

The results are stored in `norm_cache.json`, keyed by fragment ID. Chunk-aligned reads avoid repeatedly decompressing the same data. Missing entries can be computed on demand, but precomputing them is recommended.

## Main files

| Path | Purpose |
|---|---|
| `train.py` | Training loop, losses, metrics, checkpoints, and logging |
| `k_folds.py` | Letter k-folds and leave-patches-out arms of `iter2_asf_cue` on the PHerc0846A patches |
| `train_iter2.py` | Iteration-2 arms on the PHerc0846A set (`asf_cue` is the base of the mix6 run) |
| `train_0175A.py` | Rotation-augmented arms for the unlabeled PHerc0175A quick patches |
| `mae_pretrain_0175a.py` | MAE continuation on the PHerc0175A and PHerc0846A quick patches |
| `finetune_readerv3.py` | Clones reader-v3 + koine_machines, stages the mix6 corpus and fine-tunes reader-v3 for 2000 steps |
| `finetune_v8in.py` | Clones nader-fork, stages the mix6 corpus and fine-tunes v8-in for 5 epochs |
| `campaign_archs_*.py` | Complete experiment definitions and campaign runners |
| `campaign_finetune.py` | Fine-tunes the archs40 native-96 nnPU checkpoint with PHerc1447; PHerc0211 and PHerc0813 are label-free, end-of-training visualizations only |
| `campaign_finetune_96.py` | Fine-tunes the archs40 native-96 combined-surface checkpoint on PHerc1447 + selected scrolls |
| `generate_scroll_visual_guide.py` | Contact sheet of the training corpus and test surfaces (`output/scroll_patch_guide.png`) |
| `utils/config.py` | Shared configuration and registered train, holdout, and test IDs; `apply_mix6_run` is what `train.py` trains |
| `utils/mix6.py` | The mix6 corpus, its data checks (`python3 -m utils.mix6`), flat PHerc1447 surfaces, 3-depth asf cues and the model locations |
| `utils/predict_models.py` | Inference of reader-v3 and v8-in in their own environments; detects which of the three models a checkpoint is |
| `utils/model.py` | 3D nnU-Net model and experimental architecture branches |
| `utils/dataloader.py` | Labels, rings, splits, sampling, augmentation, and zarr loading |
| `utils/norm.py` | Chunk-aligned masked normalization and cache access |
| `utils/surface.py` | Surface-relative depth targets and losses |
| `utils/training_utils.py` | Optimizers, schedulers, losses, metrics, and checkpoint helpers |
| `utils/visualizer.py` | TensorBoard metrics and training-time figures |
| `utils/precomputed_asf.py` | Builds the whole-patch asf_morph cue maps in `cue_asf_morph/` |
| `build_domain_cues.py` | Builds the shift 0/-1/+1 cue maps for the add-one domain runs of `old/train_sequence.py` |
| `build_pair_cues.py` | Builds the +-1/+-2 shifted cue maps (`cue_asf_morph_dm1/dp1/dm2/dp2`) read by the depth-MIL runs of `old/train_iter2.py` |
| `utils/depth_ensemble.py` | Depth-ensemble combination methods shared by the evaluation notebooks |
| `assemble_training_segments.py` | Builds the training zarrs, masks and norms, including the 8 mix6 PHerc0846A sheets and their flat aliases; `--mix6` sets up exactly the mix6 corpus + straightened2 |
| `assemble_test_segments.py` | Renders the unlabeled test surfaces, the PHerc0846A renders outside mix6 and the extra sets |
| `generate_surface_supervision.py` | Builds literal depth and confidence maps |
| `extract_ink3d_patch.py` | Samples whole-scroll 3D ink predictions onto one patch's 28-layer grid |
| `precompute_norm.py` | Populates `norm_cache.json` without starting training |
| `mae_pretrain_nnunet.py` | Architecture-matched masked-autoencoder pretraining |
| `jepa_pretrain_nnunet.py` | Optional feature-predictive pretraining |
| `0846A_eval.ipynb` | Runs any of the three mix6 models on the PHerc0846A renders and quick patches |
| `ensemble_eval.ipynb` | Ensembles cue_dropout, reader-v3 and v8-in on one sheet (2x2 figure) |
| `eval_scrolls.ipynb` | Evaluates every labeled training fragment and the holdout |
| `test_inference.ipynb` | Searches the thirteen prize-target test surfaces for readable letters |
| `manual_split_visualization.ipynb` | Reviews explicit train/validation masks |
| `roi.py` | Edits probe windows stored in `probe_rois.json` |
| `norm_cache.json` | Persistent per-fragment normalization statistics |
| `probe_rois.json` | Saved qualitative probe locations |
| `requirements.txt` | Python dependencies, including the CUDA 12.8 PyTorch stack |
| `old/KNOWLEDGE.md` | Campaign history, detailed findings, and abandoned approaches |

## Important directories

| Directory | Contents |
|---|---|
| `ves_zarrs2/` | Local 28-layer surface volumes |
| `masks/` | Valid papyrus footprints |
| `inklabels/` | Continuous aligned labels and archived source resolutions |
| `dilated_inklabels/` | Training-time labels used by the newer campaigns and the PHerc1447 fine-tune |
| `eroded_inklabels/` | Conservative binary training labels |
| `train_masks/` | Explicit train, validation, and guaranteed-negative assignments |
| `surface_labels/` | Literal surface depth and confidence maps |
| `cue_asf_morph/` | Precomputed asf_morph cue maps read by the mix6 run's fourth input head (git-ignored; rebuilt by `--mix6`) |
| `models/` | Pretraining and supervised checkpoints |
| `runs_*` | TensorBoard logs and saved experiment configurations |
| `output/` | Evaluation figures and generated reviews |
| `_ves_tmp/` | Regenerable scratch data and fixtures |
