# PHerc0139 Ink Detector

Binary tile-level ink detection on 9.362 µm / 113 keV CT scans of Herculaneum papyrus scrolls.
Current leading architecture: **v14c_mil_lcn** — MIL with LCN preprocessing + learnable depth positional encoding, winner of the triple-scroll sweep (see [Model](#model)). Base variant **v14_mil_deep** is the stable reference.

---

## Quick start

```powershell
# activate venv
.venv\Scripts\Activate.ps1

# run training (all config is driven by utils/config.py)
python train.py -n "experiment_name"

# run a named campaign (overrides config fields per test)
python campaign_runner_p0139_triple_v2.py   # 12-test arch/tile/depth sweep (completed)
python campaign_runner_lcn.py                # LCN refinement sweep (3 tests, active)

# compute/cache normalisation stats (needed once per new zarr)
python precompute_norm.py --scroll-id 20260206000001
```

---

## Training data

All four fragments come from **PHerc0139** (Herculaneum scroll, 9.362 µm voxels, 113 keV, 1.2 m detector distance). Raw volume ID: `20250728140407`.

| ID | Fragment | Scroll | Zarr shape (D,H,W) | Mask valid frac | Split |
|---|---|---|---|---|---|
| `20260115000000` | **w044** | PHerc0139 | (28, 6021, 8141) | 0.882 | horizontal (top 80.55% train) |
| `20250223000000` | **w059** | PHerc0139 | (28, 7220, 10020) | 0.295 (1.1 µm overlap band) | vertical (left 75% train) |
| `20260206000001` | **w047** | PHerc0139 | (28, 5821, 8421) | 0.402 (1.1 µm overlap band) | vertical (left 75% train) |
| `20260115000001` | **w056** | PHerc0139 | (28, 7161, 9721) | 0.866 (rendered from tifxyz; label coverage ~full footprint) | horizontal (top 50% train) |

The masks for **w059** and **w047** are intersected with the 1.1 µm ink-detection footprint (ROI2). Only the portion of the 9.4 µm surface that was also scanned at high resolution carries reliable ink labels. Full 9.4 µm masks are preserved as `masks/<id>_full9um.png`.

Ink labels (1.129 µm source, 59 keV) live in `inklabels/` (continuous 0–255 ink probability) and `eroded_inklabels/` (binary, conservative — what training uses for ring negatives).

---

## Test segments

Three VC3D-grown patches are configured as default test targets (`test_scroll_ids` in `utils/config.py`). Test figures are generated when `test_int` fires (currently set to 9999 — disabled until a sufficiently good model is found). The visualizer loads each segment sequentially with CUDA cache cleared between renders to keep VRAM bounded for the larger segments.

### Segment 1 — original reference patch

| | |
|---|---|
| Segment name | `auto_grown_20260716083545968` |
| Scroll Source | Pherc0813 (9.362 µm / 113 keV / 1.2 m, raw volume `20250821151723`)
| Zarr ID | `20260716083545` |
| Zarr shape | (28, 4421, 4421) |
| Area | **2.98 cm²** |
| max_gen | 175 (VC3D growth iterations, restored from autosave snap 9) |
| Mask valid frac | 0.87 (compact rectangular patch) |
| tifxyz grid | 222 × 222 vertices |
| tifxyz bbox | x 4176–5730 µm, y 3261–4570 µm, z 9208–11370 µm (raw-volume voxel coords) |
| tifxyz scale | 0.05 cm per grid step |
| tifxyz location | `~/.VC3D/remote_cache/open_data/projects/paths/auto_grown_20260716083545968/` |
| Notes | First segment I unrolled; results are shoddy | 

### Segment 2 — large elongated strip (2026-07-17)

| | |
|---|---|
| Segment name | `auto_grown_20260717193517520` |
| Scroll Source | Pherc0211 (9.362 µm / 113 keV / 1.2 m, raw volume `20250821151803`)
| Zarr ID | `20260717193517` |
| Zarr shape | (28, 10821, 10821) |
| Area | **11.49 cm²** |
| max_gen | 740 (VC3D growth iterations) |
| Mask valid frac | 0.055 (re-rendered from updated .VC3D mesh; mesh coverage unchanged) |
| tifxyz grid | 542 × 542 vertices |
| tifxyz location | `~/.VC3D/remote_cache/open_data/projects/paths/auto_grown_20260717193517520/` |
| Notes | Stretches the entire length of the scroll, page is connected to segment xx218)

### Segment 3 — wide patch (2026-07-19)

| | |
|---|---|
| Segment name | `auto_grown_20260719202304218` |
| Scroll Source | Pherc0211 (9.362 µm / 113 keV / 1.2 m, raw volume `20250821151803`)
| Zarr ID | `20260719202304` |
| Zarr shape | (28, 6741, 6741) |
| Area | **10.74 cm²** |
| max_gen | 392 (VC3D growth iterations) |
| Mask valid frac | 0.120 (re-rendered from updated .VC3D mesh; mesh coverage unchanged) |
| tifxyz grid | 338 × 338 vertices |
| tifxyz location | `~/.VC3D/remote_cache/open_data/projects/paths/auto_grown_20260719202304218/` |
| Notes | Stretches the majority of the scroll (too much artifacting at top and bottom) | 

### Segment 4 — PHerc1203 patch (2026-07-20)

| | |
|---|---|
| Segment name | `auto_grown_20260720090842117` |
| Scroll Source | PHerc1203 (9.362 µm / 113 keV / 1.2 m, raw volume `20250820131727`) |
| Zarr ID | `20260720090842` |
| Zarr shape | (28, 13201, 13201) |
| Area | **7.90 cm²** |
| max_gen | 345 (VC3D growth iterations) |
| Mask valid frac | 0.049 (sparse strip within large bounding box) |
| tifxyz grid | 661 × 661 vertices |
| tifxyz location | `~/.VC3D/remote_cache/open_data/projects/paths/auto_grown_20260720090842117/` |
| Notes | First PHerc1203 segment; different scroll entirely from training data |

All four are rendered from their VC3D tifxyz mesh against their respective raw CT volume. The tifxyz format stores a 2D grid of 3D raw-volume voxel coordinates — the actual CT intensities are fetched at render time via `old/render_9um_surface.py`. See `assemble_test_zarr.sh` for the reproduction command for segment 1.

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
| `campaign_runner_p0139_triple_v2.py` | 12-test sweep: tile 16/24 × base/zgrad/lcn × depth-8/4 × range 0-28/8-16 + aug. **v14c_mil_lcn t24 d8 r8-16 won.** |
| `campaign_runner_lcn.py` | LCN refinement sweep: 3 tests (tile 16/8 × depth 8/4, range 8-16, l1=7e-6, 4 scrolls). |

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
