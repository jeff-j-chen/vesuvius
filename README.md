# PHerc0139 Ink Detector

Binary tile-level ink detection on 9.362 µm / 113 keV CT scans of Herculaneum papyrus scrolls.
Winning architecture: **v14_mil_deep** — Multiple Instance Learning with per-voxel logits and a learnable-hardness log-sum-exp bag aggregation (see [Model](#model)).

---

## Quick start

```powershell
# activate venv
.venv\Scripts\Activate.ps1

# run training (all config is driven by utils/config.py)
python train.py -n "experiment_name"

# run a named campaign (overrides config fields per test)
python campaign_runner_p0139_triple.py

# compute/cache normalisation stats (needed once per new zarr)
python precompute_norm.py --scroll-id 20260206000001
```

---

## Training data

All three fragments come from **PHerc0139** (Herculaneum scroll, 9.362 µm voxels, 113 keV, 1.2 m detector distance). Raw volume ID: `20250728140407`.

| ID | Fragment | Scroll | Zarr shape (D,H,W) | Mask valid frac | Split |
|---|---|---|---|---|---|
| `20260115000000` | **w044** | PHerc0139 | (28, 6021, 8141) | 0.882 | horizontal (top 80.55% train) |
| `20250223000000` | **w059** | PHerc0139 | (28, 7220, 10020) | 0.295 (1.1 µm overlap band) | vertical (left 75% train) |
| `20260206000001` | **w047** | PHerc0139 | (28, 5821, 8421) | 0.402 (1.1 µm overlap band) | vertical (left 75% train) |

The masks for **w059** and **w047** are intersected with the 1.1 µm ink-detection footprint (ROI2). Only the portion of the 9.4 µm surface that was also scanned at high resolution carries reliable ink labels. Full 9.4 µm masks are preserved as `masks/<id>_full9um.png`.

Ink labels (1.129 µm source, 59 keV) live in `inklabels/` (continuous 0–255 ink probability) and `eroded_inklabels/` (binary, conservative — what training uses for ring negatives).

---

## Test segment
(segment from 0211 also looks great)
The test/inference segment is from a different scroll — **PHerc0813** (also 9.362 µm / 113 keV / 1.2 m, raw volume `20250821151723`).

| | |
|---|---|
| Segment name | `auto_grown_20260716083545968` |
| Zarr ID | `20260716083545` |
| Zarr shape | (28, 4421, 4421) |
| Area | **2.84 cm²** |
| max_gen | 179 (VC3D growth iterations) |
| tifxyz grid | 222 × 222 vertices, 8002 valid (16.2% of grid) |
| tifxyz bbox | x 4176–5730 µm, y 3261–4570 µm, z 9208–11370 µm (in 9.362 µm raw-volume voxel coords) |
| tifxyz scale | 0.05 cm per grid step (i.e. 0.5 mm per grid unit) |
| tifxyz format | Each (u,v) cell stores the raw-volume (x,y,z) voxel coordinate of the corresponding surface point. Invalid cells = –1. `uuid` = segment creation timestamp = zarr ID. |
| tifxyz location | `~/.VC3D/remote_cache/open_data/projects/paths/auto_grown_20260716083545968/` — files `x.tif`, `y.tif`, `z.tif`, `generations.tif`, `meta.json` |

The tifxyz `uuid` (`auto_grown_20260716083545968`) and zarr ID (`20260716083545`) together satisfy the competition traceability requirement: "name each image after the tifxyz mesh used to generate it."

---

## Model

**v14_mil_deep** (`utils/model.py`), parameters: **1,136,210** (tile 16, depth 8):

1. **Per-slice stem** — two `Conv3d` with depth kernel=1: learns 2D texture per depth layer independently, no depth mixing yet. → `(B, 64, D, H, W)`.
2. **Depth-mix** — two full `Conv3d(3,3,3)` + CBAM attention, one `MaxPool3d(1,2,2)` (spatial only): learns which depth layers carry the ink signal. → `(B, 256, D, H/2, W/2)`.
3. **Per-voxel logit head** — `Conv1×1×1` → one scalar logit per voxel. → `(B, 1, D, H/2, W/2)`.
4. **LSE aggregation** — `tile_logit = (1/r) × (logsumexp(r·v) − log N)`. Temperature `r` is learnable (init 2.0, clamped [0.5, 10]). Interpolates from mean (r→0) to max (r→∞). Backprop concentrates on the highest-confidence voxels.
5. **Output** — one scalar tile logit → binary cross-entropy (BCE) against the eroded-inklabel tile label.

**Physics rationale:** at 113 keV carbon ink is a sparse, through-depth morphological feature at the sheet interface — not an in-plane brightness change. Global-average-pool (all prior architectures) dilutes that sparse signal ~1000×. MIL's LSE lets a handful of high-confidence voxels drive the tile prediction regardless of surrounding background.

**Physics variants** (same training protocol, same BCE):
- `v14b_mil_zgrad` — feeds `[raw, dI/dz]` to the per-slice stem. The z-derivative peaks at ink-layer interfaces and is invariant to the slowly-varying papyrus bulk-density baseline dominant at 113 keV.
- `v14c_mil_lcn` — feeds `[raw, LCN]` (local contrast normalization removes the bulk baseline) plus a learnable depth positional encoding so the model can key on the absolute depth band where ink sits (depth is the dominant variable at 9.4 µm).

---

## Files

| File | Purpose |
|---|---|
| `train.py` | Training loop. Only accepts `-n experiment_name`; all config from `utils/config.py`. |
| `utils/config.py` | **Single source of truth.** All hyperparameters, scroll list, per-scroll splits. Campaign runners override fields by mutating a `Config()` instance before passing to `Trainer`. |
| `utils/model.py` | Three architectures: `v14_mil_deep`, `v14b_mil_zgrad`, `v14c_mil_lcn` + `create_model()`. |
| `utils/dataloader.py` | `InkVolumeDataset`, `MultiScrollIterableDataset`, `DataManager`, ring-negative mask building. Reads per-scroll split config from `Config.split_overrides()`. |
| `utils/visualizer.py` | TensorBoard figure generation: eval figures (per-depth + MAX-collapse row + gold overlay), test figures. Uses YlGnBu (`ylgnbu_nan`) colormap throughout. |
| `utils/norm.py` | Fast chunk-aligned zarr normalization. Called automatically by DataManager; standalone via `precompute_norm.py`. |
| `utils/training_utils.py` | Optimizer/scheduler factory, loss function, metrics (F1, AUC, balanced accuracy). |
| `precompute_norm.py` | CLI: `python precompute_norm.py --scroll-id <id>`. Writes to `norm_cache.json`. |
| `assemble_training_zarrs.sh` | Downloads and assembles the three training zarrs from S3. See [Assembling zarrs](#assembling-training-zarrs). |
| `old/download_surface_zarr.py` | Downloads a pre-rendered OME-Zarr surface volume from S3 (volume or midslice mode). |
| `old/render_9um_surface.py` | Renders a tifxyz mesh against the raw zarr via surface-normal sampling. Used for w047 and test segment. |
| `overlay_2p4_9um.py` | Alignment sanity: hi-res (red, half opacity) over lo-res (green), yellow = overlap. Reports NCC. |
| `test_inference.ipynb` | Standalone inference notebook. Set `MODEL_PATH` + `SCROLL_ID`, Run All → depth panels + MAX figure. |
| `campaign_runner_p0139_triple.py` | 13-test sweep: tile 16/24 × base/zgrad/lcn × depth-8/4 × range-0-28/8-16 + augmentation. |

---

## Assembling training zarrs

```bash
bash assemble_training_zarrs.sh [--workers 24]
```

Downloads w044 and w059 (pre-rendered OME-Zarr on S3 via `old/download_surface_zarr.py`) and renders w047 from its tifxyz mesh (`old/render_9um_surface.py`). After running, restrict the w047 mask to the 1.1 µm overlap band (see the snippet in the script comments). Then cache normalization stats:

```bash
python precompute_norm.py --scroll-id 20260115000000 --scroll-id 20250223000000 --scroll-id 20260206000001
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

Default: winning model (`models/arch28/s07_v14_mil_deep_d8_p2.pth`) on the PHerc0813 test segment (`20260716083545`).

---

## Historical notes

See `old/KNOWLEDGE.md` for the full research log: architecture search campaigns (arch10/18/28/triple), failure modes (saturation, hard-max depth collapse, BatchNorm vs InstanceNorm), physics analysis (113 keV ink model), PHerc0211/0813 segment work, 2.4 µm vs 9.4 µm comparison, overlay analysis, and hardware crash diagnosis.
