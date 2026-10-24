# Agent quickstart: the three mix6 models

Run everything from `/vesuvius`. Smoke-test (`--smoke`) before any long run, run long jobs in `tmux`, and never `pkill -f` a pattern that also appears in a tmux
command line (kill the session by name instead). The corpus (`utils/mix6.py`) is the 7 trained PHerc0846A sheets + PHerc0343 first letters (NOT 343P) +
PHerc0009B + PHerc1447 w058/w060 + PHerc0841 + PHerc0814 + PHerc0172 w068/w087; straightened2 is the held-out sheet.

## 0. Data (one command)

```bash
python3 assemble_training_segments.py --mix6          # downloads + sets up everything; add --check to only report what is missing
python3 -m utils.mix6                                 # the same read-only report (python3 -m utils.mix6 --build makes the derived files)
```

Per scroll: volume + mask (`ves_zarrs2/<id>.zarr`, `masks/<id>.png`), norm, surface detection (`surface_labels/<id>/`; PHerc1447 is a flat surface at its
measured centre layer), surface anchors, and 3 depths of asf cues (`cue_asf_morph`, `_dm1`, `_dp1`; the holdout only the first). The 0846A part is
`assemble_test_segments.py --mix6`, run by the command above. Labels (`dilated_inklabels/`, `train_masks/`) and meshes (`tifxyz/`) are tracked in git.
Aliases like `20261012065815` are symlinks to the render `20261012065814`: the label/cue name is the alias, the zarr/mask name is the render id.
Downloads stop when free disk falls below `--min-free-gb` (default 10). `--rebuild-cues` rebuilds existing cue maps (they are kept otherwise).

## 1. cue_dropout_labelsv6_mix6: `train.py`

```bash
python3 train.py --smoke      # one tiny epoch -> models/runs_smoke, runs_smoke
python3 train.py              # 5 epochs, seed 17 (builds models/init_mae192_cue4_seed17.pth first if it is missing)
```

Output `models/runs/cue_dropout_labelsv6_mix6/` (`best_model_loss.pth` is what the notebooks read, also `final.pth`, `model_epoch_5.pth`); TensorBoard in `runs/<run>/`.
An existing run is never overwritten (`_2`, `_3`). `utils/config.py` `apply_mix6_run` is the whole recipe.

## 2. reader-v3 fine-tune: `finetune_readerv3.py`

```bash
python3 finetune_readerv3.py --smoke    # 20 steps -> reader-v3/runs/mix6_smoke
python3 finetune_readerv3.py            # setup -> prep -> train, 2000 optimizer steps
```

`setup` clones `DomRusso2/reader-v3` and the koine_machines training code (ScrollPrize/villa, sparse) into `reader-v3/` and `villa-ink/` at pinned commits, builds
`reader-v3/.venv` with uv from `requirements_readerv3.txt` and downloads `reader-v3/reader-v3.pth` from Hugging Face. `prep` stages 21-layer crops around each scroll's
surface layer (`reader-v3/data/mix6`; a segment is re-staged when its label, train mask or mask changes). Output `reader-v3/runs/mix6/ckpt_002000.pth` (step 1000 is `ckpt_001000.pth`).
uv's cache goes to `_ves_tmp/uv_cache` (the container's home disk is small; set `UV_CACHE_DIR` to change it).

## 3. v8-in fine-tune: `finetune_v8in.py`

```bash
python3 finetune_v8in.py --smoke        # 1 epoch of 6 batches -> nader-fork/runs/v8in_mix6_smoke
python3 finetune_v8in.py                # setup -> stage -> train -> export, 5 epochs
```

`setup` clones `jeff-j-chen/nader-fork` (git lfs pulls the v8-in weights), builds `nader-fork/.venv` from `requirements_v8in.txt`. `stage` writes the central 24 layers as tiffs and the label/mask PNGs
to `nader-fork/runs/v8in_mix6_data/data`. Output `nader-fork/runs/v8in_mix6_5ep/model/` (exported epoch 5; the Lightning checkpoints are under `runs/*/checkpoints/`, 0-based:
`epochepoch=4.ckpt` is the last).

## Labels differ per model

- cue_dropout (train.py): half-brightness strokes of `dilated_inklabels` are hard negatives in 0846A and unused letters elsewhere; ring negatives and nnPU far background around the ink.
- reader-v3 and v8-in: half-brightness strokes are not ink and stay supervised background in 0846A; every other scroll trains on its `train_masks` region only (255 trains,
  119 = hard negative inside the papyrus mask, black = unused, half-brightness inklabel strokes unused). reader-v3 sees a 21-layer crop centred on the surface layer (17-layer window jittered +-2),
  v8-in the central 24 layers of the render.

## Evaluate

- `ensemble_eval.ipynb`: the three models on one sheet, caches each prediction under `output/ensemble/mix6/<sheet>/cache/` (delete a file to recompute), saves the 2x2 grid
  (cue_dropout | reader-v3 / v8-in | z-score ensemble). Headless: `jupyter nbconvert --to notebook --execute --ExecutePreprocessor.timeout=-1 ensemble_eval.ipynb`.
- `0846A_eval.ipynb`: set `CHECKPOINT` to any of the three models (cue_dropout `.pth`, reader-v3 `.pth`, v8-in model folder or `.ckpt`); the kind is detected, the figures are the same.
  The depth-ensemble / ablation / denoising cells run for cue_dropout only.
- Locations of the three models: `utils/mix6.py` `MODEL_PATHS`.

## hecate 9.6 um (`hecate/`, runs in the reader venv; not part of the mix6 ensemble)

```bash
reader-v3/.venv/bin/python - <<'EOF'      # hecate needs an unnormalised uint8 (Z, Y, X) array
import numpy as np, zarr
a = np.asarray(zarr.open("ves_zarrs2/20261012065814.zarr", mode="r")[:])
np.save("_ves_tmp/render_u8.npy", np.clip(a, 0, 255).astype(np.uint8))
EOF
TMPDIR=/vesuvius/_ves_tmp reader-v3/.venv/bin/python hecate/hecate.py --checkpoint hecate/hecate_9.6um.pth \
  --input _ves_tmp/render_u8.npy --spacing-um 9.6 --precision bf16 --batch-size 16 --device cuda --output output/hecate/out.png
```

Output: a uint8 PNG (value = 255 x ink probability), same H x W as the render; it reads the central 16 of the 28 planes. Forward depth looked better than `--reverse`.
