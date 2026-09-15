# PHerc0139 Ink Detector

Binary tile-level ink detection on 9.362 µm / 113 keV CT scans of Herculaneum papyrus scrolls.
Current production path: **nnunet3d_lcndz** — a 3D nnU-Net-style encoder/decoder with a raw + LCN + depth-gradient stem, IBN, learned surface features, attention-MIL, spatial SupCon, and a sparse multitile objective.

---

## Quick start

```bash
# inspect the active campaign without training
python campaign_archs_23.py --dry-run

# run the active combined baseline and matched-context test
python campaign_archs_23.py

# compute/cache normalisation stats (needed once per new zarr)
python precompute_norm.py --scroll-id 20260206000001

# annotate readability probe windows
python roi.py
```

The active environment is a Docker image using system Python; no project venv is required.

---

## Architecture operating point — campaign 17

Campaign 17 is a fast-iteration experiment on **w013 only** (`20240304141531`). It inherits
campaign 16's hand-authored train/validation mask rather than using an axis split. Its six arms
test corrected augmentation, the supervised surface feature, entropy-gated surface aggregation,
feature-level attention-MIL, and three center/subtile geometries.

Effective configuration:

- input: 24 depth slices, z=4:28
- context: 192x192, spatially averaged by 2 before the backbone
- prediction center: 16x16, divided into a 2x2 grid of four 8x8 targets
- labels: binary `eroded_inklabels`, closed-ring negatives, `multitile_pos_only=True`
- split: `train_masks/20240304141531.png`; train and validation target cells are disjoint
- manual train-mask pixels near half intensity are cell-aligned guaranteed negatives; full-value
  pixels retain their normal training-assignment meaning
- model: 32/64/128/256-channel 3D nnU-Net, IBN in the shallow blocks
- aggregation: per-subtile gated attention-MIL with entropy weight 0.03
- regularization: conv dropout 0.05/0.05, head dropout 0.10, skip-drop 0.20
- auxiliary objectives: variance spill, spatial SupCon curriculum, TTA consistency, surface loss
- initialization: `models/mae_nnunet_96.pth`
- optimizer objective: auto-positive-weighted **BCE**

Important corrections:

- Campaign 17 does **not** currently use GCE. `gce_q=0.7` is configured but inactive because
  `loss_type="bce"`.
- Campaign 17 does **not** currently use soft ink targets or ink-label smoothing. Both positive
  and negative smoothing values are zero. Its auxiliary surface target is soft across depth.
- DANN is configured but inactive on this one-scroll run: one domain produces no adversarial
  classification signal.
- The intended future MAE geometry is 192x192/ds2 with IBN, not 196x196. The current checkpoint
  was pretrained at 96x96/ds2 without IBN.
- Active campaign-17 augmentation probabilities are flip 0.6, rotation 0.6, noise 0.3,
  brightness 0.6, contrast 0.6, and FDA 0.5. Flip and rotation now transform the multitile
  labels and masks identically. Elastic and context jitter are disabled until dense target
  warping is implemented.

### Campaign 18

Campaign 18 copies the corrected campaign-17 baseline and enables connected-component character
metrics in every arm. Its training A/B is ordinary window sampling versus uniform character
sampling. Six additional arms independently add depth warp, surface-local attenuation,
acquisition blur, correlated reconstruction noise, center-protected cutout, or target-aware
context jitter.

Character-balanced sampling repeatedly shuffles the training characters, selects one positive
target and emits its containing window plus one associated ink-free closed-ring window. The association is the
nearest connected character in target-grid space. A character crossing the fixed manual split is
excluded from character-aware sampling and metrics, preserving the baseline split without leakage.

The primary character metric is the fraction of held-out characters satisfying both:

- positive-cell recall >= 0.5
- associated-ring false-positive rate <= 0.1

`Character/APMacro/Valid` controls a separate best-character checkpoint. Macro recall, ring FPR,
F1, success fraction, and character count are also logged for train and validation.

The `context_jitter` arm reads the same global target at random even offsets up to +/-32px within
the 192px context and passes that offset to every model-side target crop. Prediction, spill,
feature attention, SupCon, and surface-guided aggregation therefore remain aligned. Flips,
rotations, and TTA consistency transform the offset with the image. Validation and inference use
zero offset. This costs one normal forward pass rather than adding a paired consistency forward.

### Campaign 19

Campaign 19 is standalone: it imports no earlier campaign and assigns every effective setting in
`base_config()`. Its baseline combines the strongest current ingredients:

- c32 center with sixteen 8px targets
- feature-level attention-MIL
- depth-softmax surface feature without guided aggregation
- character-balanced sampling and character metrics
- corrected flip/rotation targets
- no noise, brightness/contrast, or FDA after campaign 18 found no clear benefit

Every arm is capped at 20,000 training windows per epoch. This corrects a newly identified geometry
confound: at step 16, larger centers touch the ring from more origins (approximately 11k natural
windows for c16, 20.9k for c32, and 27.8k for c64). Earlier center comparisons therefore changed
both geometry and optimizer-step count.

Tests are baseline, context-size and center/target geometry variants, stronger surface supervision,
center-protected context cutout, target-aware context jitter, real-context replacement, and a
BCE/GCE/soft-label matrix.
`character_ap_macro` is the best-character checkpoint criterion; the fixed-threshold success
fraction remains logged but is calibration-sensitive.

Real-context replacement automatically preserves the prediction center plus a 16px margin on each
side (64px for c32, 96px for c64), replaces the outer context with a
same-scroll training-split donor, aligns donor depth columns to the recipient surface, and feathers
the transition over 16px. The complete 192px donor must contain no known ink and at least 80% valid
papyrus. This changes nuisance fibers while preserving all c32 target evidence.

Campaign-19 findings establish the campaign-20 hypothesis: matched 192px MAE improves the baseline;
c64_t16 is the best compromise between dense clean supervision and uncertain-label overfit;
ctx224 adds noise; jitter and especially real-context replacement can improve visual generalization;
hard-label GCE q=0.9 is the strongest isolated GCE arm; and surface loss 0.2 improves the result.
Cutout is retained as a weak visual positive despite no AP improvement by itself.

Later combined runs invalidated the initial visual impression that ctx128 was cleaner. Completed
ctx128 and strengthened-replacement runs produced lower specificity, more connected components,
and lower full-map precision than ctx192. The original cutout implementation also protected the
geometric rather than jittered target center, accidentally acting as target dropout; this is fixed.
With intact local c64 evidence, q0.9 can weakly correct confident fiber false positives, so isolated
denoising effects must not be assumed additive. Future baselines therefore return to ctx192 and
compare matched GCE q0.9 against BCE-soft before later objectives.

### Campaign 20

Campaign 20 combines the selected findings rather than testing them independently. `future_baseline`
is the corrected ctx192 control for every new arm. Future arms use
c64_t16, protected cutout, target-aware context jitter, hard-label GCE q=0.9, and surface loss 0.2.
The future operating point is 192px/ds2. Real-context replacement uses probability 0.25, a 24px
margin per side around the c64 prediction center, and a 24px feather, matching the completed c20
geometry while retaining the corrected jitter-aware cutout.

The baseline uses `models/mae_nnunet_192_ibn.pth`. Future single-scroll arms independently test:

- BCE with positive 0.90 / negative 0.05 soft targets at ctx192, directly after the matched
  GCE q=0.9 future baseline
- paired target-logit consistency under distant real-context intervention
- top-k character-bag versus assigned-ring ranking
- persistent capped character GroupDRO
- worst-quartile character CVaR
- physical surface-relative canonicalization retaining 24 slices
- an eight-slice surface-relative ink backbone whose locator still examines all 24 slices
- 3D JEPA feature-predictive initialization
- stronger context consistency (lambda 0.3)
- stronger character-bag ranking (lambda 0.4)
- faster GroupDRO adaptation (eta 0.1)
- worst-half character CVaR
- a twelve-slice surface-relative backbone

Surface supervision comes from pre-generated full-scroll depth and confidence maps in
`surface_labels`. Training fails immediately when either map is missing. The loader crops and
transforms the maps with each sample, converts absolute depth to crop-local depth, and uses
confidence to weight the soft depth loss. Canonicalization, when enabled, uses the learned
surface distribution rather than rerunning the transition heuristic.

Surface files use one directory per scroll: `surface_labels/<scroll_id>/depth.npy`,
`confidence.npy`, and `metadata.json`. Generation writes only the final half-resolution review
image to `output/surface_review/<scroll_id>/surface_depth_overview.jpg`; per-depth review images
are no longer emitted.

Campaign 23 uses literal surface-relative eight-slice input and plain LSE on w013, 500P2, and
w044. Its baseline retains ordinary SupCon but disables DANN, cross-fragment SupCon, context
replacement, cutout, and depth jitter. Focused arms independently test cross-fragment SupCon,
the proven replacement/cutout settings, jitter +/-1, and their selected combinations. Three
follow-up arms test feature-level attention-plus-max depth fusion, preservation of the MAE encoder
with a two-epoch freeze and 0.1x encoder learning rate, and literal per-column surface
canonicalization from a twelve-slice safety slab to an eight-slice backbone input. A paired
depth-view arm keeps the supervised view surface-centered and matches its prediction against a
second overlapping eight-slice view shifted by two slices, without changing inference geometry.

Campaign 24 runs an 18-way leave-one-fragment-out matrix. Every run trains on 17 fragments using
the campaign-23 full-strength combination, context-replacement margin 20, and fixed DANN lambda
0.03 for 20 epochs. Scroll/character round-robin sampling remains enabled. The excluded fragment
is eagerly loaded as the sole visualization scroll and receives a full evaluation figure at epoch
20. Before allocating training data, the campaign validates all 18 train masks, zarrs, labels, and
literal surface maps and reports full-intensity training pixels, positive/non-ink pixels, and
explicit half-intensity negatives.

Campaign 25 is one 12-fragment baseline spanning seven physical scrolls. It adds PHerc0172 w068
and w087, PHerc1667 w018, two PHerc0009B patches, and PHercParis4 to six established anchors.
Sampling is round-robin by physical scroll rather than segment: PHerc0139 has weight 2 and every
other scroll has weight 1. Segments within each physical scroll rotate uniformly. The run uses
the full-strength literal-surface configuration, fixed DANN 0.03, 12 epochs, fast evaluation at
epoch 12, and renders only PHerc0139 w044. Its isolated arms are:

- unchanged baseline
- fixed locally supported MIL using the top 2 or top 4 spatial responses
- WELDON-style top-4 positive plus bottom-4 negative evidence
- CLAM-lite top/bottom instance supervision (`k=4`, lambda 0.1)
- SAM with rho 0.01 and 0.05
- ELR beginning at epoch 5 (`beta=0.7`, lambda 0.1)
- a zero-initialized structure-tensor fiber-coordinate branch
- an early local-3D-stem → 2D U-Net with attention-plus-max depth fusion
- depth-only and divided depth/windowed-XY attention at encoder stage 3
- zero-initialized anisotropic MedNeXt adapters with spatial kernels 5 and 7
- a combined kernel-7 MedNeXt plus divided-attention arm

The early 3D→2D arm is intentionally distinct from late feature-depth fusion: it allocates almost
the entire encoder/decoder to sheet-tangent morphology after a short surface-normal stem. The
divided-attention variants test whether explicit depth and spatial token mixing adds useful
long-range structure without replacing the convolutional backbone. MedNeXt adapters retain the
pretrained nnU-Net exactly at initialization, then learn residual `(3,k,k)` large-kernel features;
kernel 5 is the conservative arm and kernel 7 tests broader fiber/stroke context. The combined
arm tests whether large-kernel local bias and nonlocal attention are complementary.

Campaign 25 preflight also enforces MAE initialization. The baseline, aggregation, and loss arms
share a campaign-specific 12-fragment continuation of the established 22-scroll MAE checkpoint.
Fiber coordinates, early 3D→2D, both
divided-attention variants, both MedNeXt kernels, and the combined MedNeXt/attention model each
use a matching architecture-specific checkpoint. Missing checkpoints are trained automatically
for 2,000 steps before a real campaign run; architecture variants warm-start from the completed
baseline MAE checkpoint and freeze its restored parameters while pretraining only the newly added
modules. This preserves a common backbone and avoids giving architecture arms extra backbone
optimization. Interrupted runs are not accepted as complete. A dry run reports missing pretraining
without launching it.

To continue the existing 18-scroll MAE warm start for 1,000 additional optimizer steps while
adding the five configured test scrolls, use `--init-weights` together with
`--include-test-scrolls`; the continuation writes a new checkpoint rather than overwriting its
source.

The continuation uses depth 8 with random safe starts over `[0, 28)`, matching the fine-tuning
backbone while exposing it to every reconstructed layer. Newly assembled zarrs default to chunks
of `(8, 64, 64)`: depth 8 matches one model window, while 64px spatial chunks balance arbitrary
192px context reads against over-read and small-file overhead.

The optional matched-128 MAE command remains available for a separate pretraining-scale study:

```bash
python mae_pretrain_nnunet.py --name mae_nnunet_128_ibn --ctx 128 --ds 2 \
  --depth 24 --d-start 4 --d-end 28 --steps 6000 --batch-size 32 \
  --require-all-scrolls
```

The A100 campaign profile uses batch 96, LR 1.5e-4, and eight workers. Figure inference uses batch
192 with two
prefetch workers, and a 1GB tile-buffer target: faster than the emergency campaign-19 reruns while
retaining bounded buffering and worker shutdown before full-scroll figures.

All A100 run ids and tags end in `_a100`. The baseline strengthens context regularization with
cutout probability 0.65 (three patches, max 16%), target-aware jitter 32px, and replacement
probability 0.35 with a 20px protected margin/feather. Positive weighting is fixed at 1.5 instead
of the previous automatic ~2.1 to counter the tendency to predict ink everywhere.

Five three-scroll tests occur between the main arms and stronger hyperparameter variants. They use
w013, w044, and 500P2_front—the three fragments with manual train masks. Scroll round-robin plus
uniform character sampling supplies 6,667 windows per scroll, keeping the total near 20,000.

```bash
python assemble_training_segments.py --only w044,500P2_front,w013
```

Tests are a no-DANN multiscroll control and annealed DANN lambdas 0.0025, 0.005, 0.01, and 0.02.
Every DANN run renders all three scrolls. Fast evaluation uses the tile-aligned bounding box of a
scroll's manual train mask; scrolls without one retain the previous probe/corner behavior.
The domain head always receives full cross-entropy gradients; lambda scales only the reversed
gradient entering the shared backbone. Domain accuracy and effective GRL scale are logged.

The A100 loop reduces synchronization overhead by bundling scalar diagnostics into one transfer,
removing unused surface scalar reads, making active spill reduction branchless, skipping zero-weight
L1 work, using `zero_grad(set_to_none=True)`, branchless SupCon, and fused CUDA AdamW. The guarded
future baseline is checked after epoch 3; the complete campaign aborts if character AP is below
0.55 or specificity below 0.25, preventing clearly divergent settings from consuming later runs.

Pretrain the feature-predictive 3D JEPA checkpoint across all 24 default fragments with:

```bash
python jepa_pretrain_nnunet.py --name jepa_nnunet_192_ibn --ctx 192 --ds 2 \
  --depth 24 --d-start 4 --d-end 28 --steps 6000 --batch-size 8 \
  --accum-steps 4 --require-all-scrolls
```

The fine-tune artifact is a plain nnU-Net state dict at
`models/jepa_nnunet_192_ibn.pth`; a separate `_resume.pth` stores the student, EMA teacher,
predictor, optimizer, scheduler, and scaler.

When a selected campaign includes `jepa192` and its checkpoint is absent, campaign 20 runs this
JEPA pretraining command as a preflight before starting any supervised arm.

The current RTX 5090 continuation uses BCE-soft as the inherited baseline (positive 0.90,
negative 0.05), disables positive weighting, and runs batch 48 / LR 1.2e-4 / eval batch 96.
Run ids and tags end in `_5090`, with logs under `runs_archs20_5090`. RTX 5090 requires a
Blackwell-capable CUDA 12.8 PyTorch wheel; the repository pins torch 2.11/cu128, torchvision
0.26/cu128, and torchaudio 2.11/cu128 in `requirements.txt`.

Multi-scroll character balancing is now available through `character_balance_scrolls=True`.
Training draws scrolls round-robin while drawing characters uniformly inside each scroll, cycles
smaller scrolls as needed, and preserves the original total epoch length. Component IDs are domain-
namespaced so character metrics cannot merge letters from different fragments. Dot-positive extras
are disabled in this mode because they have no connected-character identity.

List-based `data.scrolls` retains that segment-level round robin unchanged. An optional
`data.train_scroll_dict` groups segment IDs by physical scroll, while the corresponding
`data.train_scroll_weights` supplies one positive integer per dictionary key. The grouped sampler
rotates uniformly through segments inside each physical scroll and repeats each scroll in the
outer schedule according to its weight. Physical groups also become DANN/cross-fragment SupCon
domains, while character IDs keep a separate per-segment namespace.

---

## Training data

Most fragments come from **PHerc0139** (Herculaneum scroll, 9.362 µm voxels, 113 keV, 1.2 m detector distance, raw volume ID `20250728140407`). The default MAE/training corpus is now **24 fragments**: the previous 18 plus two PHerc0172 patches, PHerc1667 w018, two PHerc0009B patches, and one PHercParis4 segment.

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

### Added cross-scroll training fragments

| ID | Fragment | Physical scroll | Source | Training conversion |
|---|---|---|---|---|
| `20251112000002` | w087 | PHerc0172 | 7.91 µm / 53 keV | area-resampled XY plus linear depth resample to 28 layers at 9.362 µm |
| `20251111010954` | w068 | PHerc0172 | 7.91 µm / 53 keV | area-resampled XY plus linear depth resample to 28 layers at 9.362 µm |
| `20240304144031` | w018 | PHerc1667 | 2.399 µm / 78 keV | level-2 XY (9.596 µm), clean-text crop, depth pool 109→28 |
| `20250919125754` | auto-grown 487 | PHerc0009B | 8.64 µm / 116 keV | area-resampled XY plus linear depth resample to 28 layers at 9.362 µm |
| `20250919131352` | auto-grown 722 | PHerc0009B | 8.64 µm / 116 keV | area-resampled XY plus linear depth resample to 28 layers at 9.362 µm |
| `20231210121321` | Paris4 | PHercParis4 | 2.4 µm / 78 keV | level-2 XY (9.6 µm), full segment, depth pool 109→28 |

The assembler downloads each configured ink prediction, resizes it into the exact training frame,
writes the primary map to `inklabels/<id>.png`, archives aligned copies under
`inklabels/2_4um/` (and the w018 1.129 µm prediction under `inklabels/1_1um/`), and creates the
conservative binary target in `eroded_inklabels/`. Label generation aborts unless at least 95% of
thresholded ink pixels overlap the output zarr's midslice-derived papyrus mask.

The Paris4 137 keV volume cannot yet be substituted safely. The available tifxyz is expressed in
the 78 keV full-volume grid `(75784, 32693, 32693)`, while the 137 keV scan is a separately cropped
grid `(6625, 8431, 8431)` with no published crop origin or affine transform. Direct coordinate
reuse would sample the wrong anatomy; the 78 keV surface remains the reproducible source until a
cross-energy registration is established.

Assemble the six additions serially to bound temporary disk and RAM use:

```bash
python assemble_training_segments.py \
  --only w087,w068,w018,p9b_487,p9b_722,paris4 \
  --concurrent-fragments 1
```

After assembly and train-mask authoring, generate their literal surface inputs with:

```bash
for sid in 20251112000002 20251111010954 20240304144031 \
           20250919125754 20250919131352 20231210121321; do
  python generate_surface_supervision.py --scroll-id "$sid"
done
git add surface_labels/20251112000002 surface_labels/20251111010954 \
  surface_labels/20240304144031 surface_labels/20250919125754 \
  surface_labels/20250919131352 surface_labels/20231210121321
```

**w035** labels are downloaded separately: `python download_w035_labels.py` (1.129 µm / 59 keV source, same as all other PHerc0139 fragments). Assemble zarr via `python assemble_training_segments.py --only w035` (mask generation requires the zarr; re-run label script afterwards to apply it). Edit `inklabels/20260317000000.png` and regenerate eroded labels with `python download_w035_labels.py --erode-only`.

The PHerc0500P2 fragment is notable for its **crystal-clear inklabels** derived from a high-resolution 2.215 µm / 111 keV scan. The 2.215 µm ink detection TIF (shape 26440 × 15060) was resized to the 9.362 µm zarr frame at a 4.21× scale ratio, thresholded at 0.55 (140/255), and eroded with a 3×3 kernel (12 iterations) to produce the training labels. Split changed from horizontal to vertical (2026-08-11) for campaign_archs_7 single-scroll isolation testing. Edit `inklabels/20250628074500.png` then regenerate the eroded version with `python download_p500p2_labels.py --erode-only`. Assemble via `python assemble_training_segments.py --only 500P2_front`.

All **24** are wired into `DEFAULT_SCROLLS` in `utils/config.py`, so future default MAE runs include the six additions. Ink footprint = fraction of the frame with ink label > 0.

The masks for **w059** and **w047** are intersected with the 1.1 µm ink-detection footprint (ROI2). The **new 10 use the full 9.4 µm papyrus footprint** (not intersected), so ring negatives near the labeled band could in principle fall on un-scanned surface; in practice the ring hugs the ink so this is minor. The full-surface footprint is recoverable directly from the zarr (`z[mid] > 0`); no separate `_full9um.png` is stored.

Ink labels (1.129 µm source, 59 keV) live in `inklabels/` (continuous 0–255 ink probability) and `eroded_inklabels/` (binary, conservative — what training uses for ring negatives; new-fragment eroded fraction ≈ 0.02–0.04).

**seg46527 (PHerc0814) caveat:** only `eroded_inklabels/20260226000000.png` and `masks/20260226000000.png` are present — there is no non-eroded `inklabels/20260226000000.png`. Training with `ring_label_source='original'` (the twostage default) will log a warning and fall back to the eroded map for the ring boundary; `ring_label_source='eroded'` (isolation campaign) uses it directly. Norm stats are cached in `norm_cache.json`.

### Holdout sanity fragment — w055 (NOT trained on)

**w055** (`20251226000000`, PHerc0139, 9.362 µm) is assembled exactly like the training fragments (zarr + mask + 1.1 µm inklabel) but is **deliberately excluded from `DEFAULT_SCROLLS`**. It is a pure hallucination check: the model never sees it during training, so if inference on w055 does **not** reproduce its known 1.1 µm text, we know the model is hallucinating rather than genuinely detecting ink. Assemble it with `python assemble_training_segments.py --only w055`.

---

## Test segments


Five VC3D-grown patches are configured as default test targets (`test_scroll_ids` in `utils/config.py`): PHerc0813, PHerc0211, PHerc1203, PHerc1447, and PHerc0826. Test figures are generated when `test_int` fires (currently set to 9999 — disabled until a sufficiently good model is found). The visualizer loads each segment sequentially with CUDA cache cleared between renders to keep VRAM bounded for the larger segments.

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

1. **Input preparation** — 192px context is average-pooled to 96px; depth remains 24.
2. **Stem** — concatenate `[raw, LCN(raw), dI/dz]`, then map 3 channels to 32.
3. **Encoder** — three spatial/depth pooling levels with 32/64/128 channels and a 256-channel
  bottleneck. IBN-a is used in the first normalization of `enc1` and `enc2`; other norms are IN.
4. **Decoder** — three transposed-convolution stages with nnU-Net skip connections and a
  one-channel voxel-logit head.
5. **Legacy surface gate** — a tiny depth-only sigmoid branch amplifies early features by 1-2x.
6. **New surface branch** — a spatial/depth CNN predicts a softmax distribution over the 24
  depths at every downsampled spatial point. A zero-initialized 1x1 projection adds that volume
  residually to `enc1`; an auxiliary soft-target loss supervises the papyrus-air transition.
7. **Multitile output** — crop the decoded voxel map to the central 16px and divide it into four
  8px cells. Each cell is aggregated independently over its depth/spatial voxel bag.
8. **Attention-MIL** — the active aggregator applies gated attention to scalar voxel logits.
  Entropy regularization is required empirically to stop brittle attention collapse. Campaign 17
  also tests attention over the full 32-channel decoder vectors.
9. **Auxiliary embedding** — multitile SupCon now pools one decoder embedding per target cell and
  filters it with that cell's label/mask. DANN still uses the global bottleneck embedding.

The surface feature is generated internally. The model receives `(B,1,24,192,192)`, downsamples
to `(B,1,24,96,96)`, and predicts `(B,1,24,96,96)` surface probabilities. Pre-generated depth and
confidence maps are training teachers only, so inference still needs only the volume. Generate
each training scroll before startup with `python generate_surface_supervision.py --scroll-id ID
--z-start 4 --z-end 28`; `data.surface_label_dir` selects the map directory.

`model.better_surface` replaces the original learned head with a broad-context residual head whose
fixed input evidence follows the same relative-occupancy papyrus-to-air transition used by the
offline generator. `model.surface_teacher_input` is a separate oracle experiment: it injects the
cropped map's literal local depth, confidence, and signed distance from every input slice directly
into the first encoder stage. Unlike either learned head, that mode requires generated maps during
both training and validation.

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
| `campaign_archs_17.py` | Current w013 hand-mask experiment for the supervised depth-softmax surface feature. |
| `campaign_archs_18.py` | Character-balanced sampling, character-macro metrics, and isolated hard-augmentation tests. |
| `campaign_archs_19.py` | Standalone c32 feature-attention + surface + character-balanced baseline and c64 follow-ups. |
| `campaign_archs_20.py` | Combined c64_t16/GCE/context/surface baseline with matched 192px vs 128px MAE. |
| `campaign_archs_23.py` | Triple-scroll literal-surface refinements and depth-representation tests. |
| `campaign_archs_24.py` | Eighteen-way leave-one-fragment-out full-strength training with fixed DANN and held-out full-scroll visualization. |
| `campaign_archs_25.py` | Twelve-fragment tests with weighted physical-scroll sampling, forced architecture MAE, and w044-only fast evaluation. |
| `jepa_pretrain_nnunet.py` | 3D masked-block feature prediction with an EMA teacher and collapse guards. |
| `generate_surface_supervision.py` | Builds full-resolution papyrus-air pseudo-labels and review figures. |
| `utils/surface.py` | Offline-map soft surface targets and robust smoothness loss. |
| `old/` | Archived experiments, older campaigns, and retired architecture families. |
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

The notebook defaults to the campaign-23 baseline and renders all five configured test patches:
PHerc0813, PHerc0211, PHerc1203, PHerc1447, and PHerc0826.

---

## Historical notes

See `old/KNOWLEDGE.md` for the full research log: pre-2026 campaigns (arch10/18/28, scroll1/4 at 7.91 µm), 2.4 µm vs 9.4 µm investigation, scroll4 teacher-zarr work, and the 2026-07 PHerc0139 9.362 µm campaign series (triple-scroll sweep, LCN win, w056 addition).

## Current research status and constraints

### Problem definition

- Ink is detectable at approximately 9.3-9.6 microns. The signal may occupy only one or a few
  decisive depths within a 24-slice box around an imperfectly flattened, undulating sheet.
- Labels come from much higher-resolution scans and are sparse and uncertain after registration.
- The immediate objective is generalization between held-out letters on w013, not scale-up.
  Multi-fragment training improves representation capacity but slows iteration substantially.
- No model in this project has yet recovered a convincing character outside its immediate
  supervised corpus. Self-training or pseudo-ink labeling is therefore prohibited for now.

### Hard architectural constraints

- Dense segmentation has repeatedly failed under these imperfect labels. Villa succeeds with
  dense supervision because its labels and recipe are better suited to that objective; copying
  its dense path is not the objective of this project.
- Single-tile MIL supplies too little spatially resolved gradient. Multitile is the chosen middle
  ground between one scalar/window and dense pixel supervision.
- The current optimum is a 16px center split into four 8px targets. Its advantage over nearby
  geometries is real but narrow.
- `pos_only` is a window-level safety rule: if a center contains a supervised positive, every
  non-positive subtile in that center is ignored. Ink-free closed-ring windows supply negatives.
  Positive and negative labels must never touch within one supervised center.
- The closed ring is part of the label definition, not generic negative sampling: close radius 3,
  gap radius 3, shell radius 2. More distant blank papyrus has not helped.
- The manual split assigns easy/concrete letters plus a small hard subset to training and reserves
  difficult letters for validation. Validation is intentionally harder than training but directly
  measures the failure mode of interest.

### Campaign findings through 17

- Campaigns 7-9: 3D nnU-Net is the strongest architecture family and can fit the available labels.
- Campaigns 10-12: larger context helps through 192px/ds2. At 256px, quality is effectively flat
  while computation rises sharply.
- Legacy learned surface attention helps. Campaign 17 tests a physically supervised depth-softmax
  surface estimate with spatial context and robust smoothness.
- Variance spill is useful initially but saturates quickly after the voxel logits become selective
  across depth. Stronger spill does not continue improving the representation.
- Weak convolution/head dropout helps. Heavy dropout reduces learning too much.
- Skip-drop 0.2 versus 0.6 produced little observable difference; this does not yet prove that skip
  features are unused because the implementation drops whole batch-level branches during training
  and restores all skips during evaluation.
- Gated attention-MIL helps only with entropy regularization. The entropy term prevents attention
  collapse; without it the learned pooling is brittle.
- Spatial SupCon has been useful. It is now aligned to one decoder embedding, label, and validity
  value per multitile cell; campaign 17 is the first campaign using the corrected version.
- DANN was too destructive in multi-domain testing. In current w013-only campaigns it is a no-op.
- Photometric, FDA, and elastic augmentations produced negligible gains over broad strength ranges.
- Earlier geometric augmentation and context-jitter results were confounded by target misalignment.
  Campaign 17 synchronizes flips/rotations and disables elastic/context jitter.
- Campaign 14 established multitile as materially better than one-score-per-center MIL.
- Campaign 15 found the 16px center / 8px subtile / four-target geometry narrowly best.
- Campaign 16 replaced the geographic axis split with a hand-authored target partition.
- Campaign 17 is the active supervised-surface experiment on that manual split.

### Multitile invariants

- Flat target index is `iy * grid + ix` everywhere.
- The active geometry is grid=2 and subtile=8, producing four targets over a 16px center.
- Training windows advance by 16px while reading 192px context.
- The loss divides by the target validity-mask sum; masked cells produce no gradient.
- In a positive window, `pos_only` masks every non-positive cell. In an ink-free ring window,
  valid ring cells remain negative.
- Inference reverses TTA transforms on the 2x2 output grid before overlap averaging.

### Geometric augmentation status

Multitile flips and 90-degree rotations now transform the input, target grid, and validity grid
together. Elastic deformation is fail-safe disabled when multitile targets are present because
warping a 2x2 grid is not physically correct; it requires warping the dense source labels and
masks with the same displacement field before reducing them to target cells. Context jitter is
also disabled for multitile until the output crop follows the shifted target.
