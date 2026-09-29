# Cross-resolution MAE: plan and handover


Written 2026-09-29 for the agent who will run this on a machine with enough disk. Nothing here has been
run: the authoring machine has no disk to spare, so every script is **untested**. You are now this agent, and you have 500GB of available space with which to download. 


**Goal.** Raise recall on native 9.36 µm / 113 keV / 1.2 m scans (held-out pherc0841 and pherc0009b, then
unseen test scrolls) without losing pAUC@1%FPR or clean papyrus. The idea: several training segments
also exist as ~2.4 µm scans. Use those pairs to teach the backbone what the 9.36 µm data *hides*.

**Hard rules.**
- Never download a `1.129um-*` volume. `crossres/pairs.py:url()` raises if asked to.
- Download only what is needed:
  - the 2.4 µm render only inside tiles near ink labels (training scrolls);
  - random footprint tiles for held-out scrolls, **never** chosen from their labels.
- The 9.36 µm side is never re-downloaded. Inputs are cut from the local training zarrs (`ves_zarrs2`),
  so they are exactly what fine-tuning sees.

## 0. What to train: three pretrains, one shot

This section supersedes the single "option #5" in §3. Train all three and send back the three
checkpoints plus logs (§0.4). Each one is a drop-in `init_weights` for the unchanged fine-tuning model.

### 0.1 Answers to the open questions

**Does this change the model's input?** No.
- Fine-tuning and inference still see a 96 × 96 × 8 crop at 9.36 µm.
- The high-resolution data only ever feeds a throwaway prediction head during pretraining. Only the
  backbone weights are kept.

**Why would predicting detail the input doesn't contain help?**
- The detail isn't in the voxels, but it is statistically recoverable from their neighbourhood: fibre
  texture, the density profile through the sheet, and partial-volume signatures. A standard MAE
  likewise reconstructs masked voxels that aren't in its input.
- The key point: the 2.4 µm scan at the surface is where the researchers could see the ink. Predicting
  it from 9.36 µm is therefore a dense, label-free proxy for "what ink looks like after the 9.36 µm
  scan blurs it", over every pixel of every paired segment, not just our sparse labels.
- Features that do this well should separate faint ink better at low false-positive rates, which is
  what recall at 1% FPR measures.
- It is not guaranteed. The worst case is no gain, and the fine-tuning recipe is untouched either way.

**Can we unsquish the depth?** Yes, and it is the cheapest target.
- The ~2.4 µm renders keep all ~109 layers at every pyramid level, because the pyramids downsample only
  in x/y. Our 28 layers cover the same ~262 µm slab, so each of our slices corresponds to ~3.9 layers
  of 2.4 µm.
- Level 2 of the 2.4 µm render is ~9.6 µm in x/y, the same x/y grid as our input, with 4× the depth
  resolution.
- So the target "4 sub-slices per input slice, same x/y" asks the model exactly how the 9.36 µm column
  was squished. Carbon ink is a thin layer on the sheet, so the relations between layers are where it
  shows.
- It is also far less sensitive to x/y misregistration than an x/y super-resolution target, because a
  one-pixel error is 9.6 µm, not 4.8 µm.

**Can we download only around the labels?** Yes. `build_pairs.py` never downloads a full high-resolution
volume.
- It streams only the chunks under 512 px tiles inside each segment's ink bounding box grown by 256 px.
  That covers the papyrus between and around the text as densely as the ink. Ink is 0.3–3% of those
  pixels, so the model mostly learns to upscale papyrus.
- `--region near` keeps only tiles touching ink, if disk is short. 0841 gets random footprint tiles,
  and only with `--include-holdouts`.
- Surface-volume chunks are [all layers, 128, 128], so a level-L chunk covers 128·2^L high pixels:
  ~131 native px at level 2 and ~66 at level 1.
- Bytes streamed for an ink region of A native px²:
  - level 2 ≈ 3.7 × the same region of our 9.36 µm zarr (uint8);
  - level 1 ≈ 15 ×;
  - level 0 ≈ 60 × (not used).
- A 3000 × 3000 px text region therefore costs ≈ 1 GB at level 2 or ≈ 4 GB at level 1.
- Run `build_pairs.py --count-only --plan depth` (and `--plan xyz`) first: it prints tiles, streamed GB
  and disk GB per segment from the label PNGs alone.

### 0.2 The three pretrains

All three continue from the production MAE `mae_nnunet_192_campaign34_early_gated_native96_2k.pth`
for 2000 steps, batch 32, lr 1.5e-4. All keep the standard 9.36 µm MAE over all 37 volumes in the
batch.

All three also add **slab masking** to that standard branch. On half the samples a run of 1–3 whole
slices is hidden (every pixel), so the model must rebuild layers from the layers around them. This is
the depth-relation objective that needs no pairs.

| id | name | paired target | pairs / batch | streamed from S3 | what it isolates |
|---|---|---|---|---|---|
| **D** (primary) | `mae_crossres_depth` | level 2: 8 slices × 4 sub-slices = 32 layers at the input x/y (96 × 96) | 16 / 32 | ≈ 3.7 × the ink regions | unsquishing depth from real 2.4 µm data |
| **X** | `mae_crossres_xyz` | level 1: 32 layers at 2× x/y (192 × 192) | 16 / 32 | ≈ 15 × the ink regions | whether in-plane detail adds to D |
| **S** (control) | `mae_slab_native96` | none (`--plan none`) | 0 / 32 | nothing | how much of D/X comes from the pairs rather than from slab masking |

How to compare them:
- D − S is the effect of real high-resolution pairs.
- X − D is the effect of adding x/y detail.
- S against the production MAE is the effect of slab masking alone.

If the other machine can only afford two runs, run D and S.

Honest runs use `--exclude-holdout-pairs`:
- **0841** (the harder holdout, and the strict measure) stays in the standard branch unlabelled, as now,
  but its 2.4 µm pair is not used.
- **0009B**'s pair *is* used: `pairs.py` marks it `train`, so its tiles come from its ink bounding box
  like the training scrolls. It stays held out from fine-tuning, but its holdout numbers become
  optimistic.
- Judge the pretrains by **0841**.

**Depth sampled.** Around the surface, not all depths.
- The 28-layer renders are centred on the fitted surface.
- Tiles store slices 8–19, and the 8-slice window is placed at random within them (slices 8–15 up to
  12–19) in both branches. The production MAE fixed it at 10–17.
- The ±2-slice jitter matches fine-tuning better: its window follows the local surface depth map, which
  wanders around the render's centre.

### 0.3 Commands, in order (other machine, repo root)

```bash
python crossres/probe_pairs.py                                   # names and sizes, KB of metadata
python crossres/sanity_midslice.py --high-level 3                # cheap first registration pass
python crossres/sanity_midslice.py                               # level 2 = your ÷4 Dice check; fix any INVESTIGATE
python crossres/build_pairs.py --count-only --plan depth         # download / disk budget
python crossres/build_pairs.py --count-only --plan xyz
python crossres/build_pairs.py --zarr-dir ves_zarrs2 --plan depth
python crossres/build_pairs.py --zarr-dir ves_zarrs2 --plan xyz  # skip if disk is short (then run D and S only)
INIT=models/mae_nnunet_192_campaign34_early_gated_native96_2k.pth
for PLAN in depth xyz none; do
  NAME=$([ $PLAN = none ] && echo mae_slab_native96 || echo mae_crossres_$PLAN)
  python crossres/mae_pretrain_crossres.py --plan $PLAN --name $NAME --init-weights $INIT \
      --exclude-holdout-pairs --dry-run &&
  python crossres/mae_pretrain_crossres.py --plan $PLAN --name $NAME --init-weights $INIT \
      --exclude-holdout-pairs
done
```

`build_pairs.py` per tile:
- refines the depth offset in 1/8-slice steps against the input's layer profile;
- then re-estimates an x/y shift;
- drops tiles whose midslice NCC stays below 0.3.

Check each `crossres/pairs/<plan>/<zid>/meta.json` for `ncc_median`, `dropped` and
`tile_depth_shift_slices`. A wide spread of depth shifts means the renders disagree on the surface.

### 0.4 What to send back

- `models/mae_crossres_depth.pth`, `models/mae_crossres_xyz.pth`, `models/mae_slab_native96.pth`
- `models/mae_crossres_depth.generator.pth`, `models/mae_crossres_xyz.generator.pth` (§0.5)
- `runs_mae/<name>_*` (TensorBoard)
- `crossres/sanity/` (registration.json and overlays)
- every `crossres/pairs/*/*/meta.json`

What the logs must show:
- `CrossRes/monitor_sr` clearly below `CrossRes/monitor_sr_trilinear` for D and X; otherwise the head
  learned nothing and the checkpoint is not worth a campaign arm.
- `monitor_rec` near its starting value; a large rise means the standard objective was traded away.

Then register each checkpoint as in §2 step 6 and run one campaign arm per checkpoint, plus the
production MAE as the control, under an identical fine-tuning config.

Simply make sure all of these files will be committed to git, I (the user) will handle transfer back to the main training machine.

### 0.5 The hallucination route: keep the prediction

D and X also save `models/<name>.generator.pth`: the whole network including the super-resolution head,
plus its geometry. Send these back too.

A quarter of the paired samples are trained unmasked at full weight (`--unmasked-paired-frac 0.25`), so
the generator has practised the exact job it does at inference: 8 real slices in, a sharper volume out.

When the generators come back, the plan is one extra fine-tuning arm per generator (implemented on this
machine then):
1. **Frozen front end.** The generator runs frozen in front of the ink model, the same plumbing as
   `model.input_denoiser`. For every window it predicts the unsquished 32-layer column.
2. **Input.** The ink model sees the 8 real slices plus the 32 predicted sub-slices, stacked in depth.
   The real data is always there, so a bad prediction can be down-weighted rather than trusted.
   - The early-2D model already collapses depth early, so the change is a depth-40 input and a new MAE
     (or a warm start with the depth stem re-initialised).
3. **What it can add over plan D's weights alone.** The fine-tuned model can use the predicted fine
   layers directly as evidence. Plan D only shapes the features, and fine-tuning may drift away from them.
4. **The risk.** The generator invents plausible detail, which could show up as confident false ink.
   - It was trained without labels, so it has no reason to invent ink specifically.
   - The papyrus check in the renders is decisive here.

This is not the banned "second stage on heatmaps": the extra input is a predicted image, not a
prediction of ink.

It also leaves a cheap diagnostic: render the generator's output over 0841 next to the labels. If
strokes appear sharper in the predicted fine layers than in the 9.36 µm slices, the route is worth its
arm.

## 1. Pair inventory

Every name below was confirmed by an S3 listing on 2026-09-29. Two were truncated in the listing and
are marked *verify*; `probe_pairs.py` checks them.

URL pattern: `https://vesuvius-challenge-open-data.s3.amazonaws.com/<segment>/surface-volumes/<volume>/<level>`.
Surface-volume pyramids downsample only x/y: level L has pixel size 2^L × the level-0 size and keeps
every layer.

| name | zid | scroll | role | low (native) volume | high volume |
|---|---|---|---|---|---|
| w044, w059, w030, w043, w045, w040, w041, w039, w035 | see `pairs.py` | PHerc0139 | train | `9.362um-1.2m-113keV-volume-20250728140407.zarr` | `2.399um-0.22m-78keV-volume-20260102150214.zarr` |
| 500P2_front | 20250628074500 | PHerc0500P2 | train | `9.362um-1.2m-113keV-volume-20250820143440.zarr` | `2.215um-0.4m-111keV-volume-20250526151718.zarr` (also `4.317um-1.2m-111keV-volume-20250528085330.zarr`) |
| seg46527 | 20260226000000 | PHerc0814 | train | `9.362um-1.2m-113keV-volume-20250804134230.zarr` | `2.399um-0.22m-78keV-volume-20260309142202.zarr` |
| p9b_487 | 20250919125754 | PHerc0009B | **holdout** | `8.64um-1.2m-116keV-volume-20250521125136.zarr` (*verify*) | `2.401um-0.35m-77keV-volume-20250820154339.zarr` |
| p841 | 20260221022814 | PHerc0841 | **holdout** | `9.366um-1.2m-113keV-volume-20250821151531.zarr` (*verify*) | `2.403um-0.22m-77keV-volume-20260319124803.zarr` |

Segment folders:
- 0139: `PHerc0139/segments/<zid>-wNNN_...` (full names in `pairs.py`)
- 500P2: `PHerc0500P2/segments/20250628074500-500P2_front`
- 0814: `PHerc0814/segments/20260226000000-46527_2um_try2`
- 0009B: `PHerc0009B/segments/20250919125754-auto_grown_20250919055754487_inp_hr`
- 0841: `PHerc0841/segments/20260221022814-auto_grown_20260220174252405`

### Scalings

| pair | high µm | low µm (native) | high px per native px | high px per training-frame px | level-2 pixel | level 2 → training frame |
|---|---:|---:|---:|---:|---:|---:|
| 0139 (all 9) | 2.399 | 9.362 | 3.902 | 3.902 | 9.596 µm | ×1.025 |
| 0814 seg46527 | 2.399 | 9.362 | 3.902 | 3.902 | 9.596 µm | ×1.025 |
| 500P2 front | 2.215 | 9.362 | 4.227 | 4.227 | 8.860 µm | ×0.946 |
| 500P2 (alt 4.317) | 4.317 | 9.362 | 2.169 | 2.169 | 17.27 µm (level 0 is already 2.2× finer) | — |
| **0009B** | 2.401 | **8.64** | **3.599** | 3.899 | 9.604 µm | ×1.112 vs native, ×1.026 vs frame |
| 0841 | 2.403 | 9.366 | 3.898 | 3.898 | 9.612 µm | ×1.026 |

Notes on the table:
- **0009B is the odd one.** Its native scan is 8.64 µm at 116 keV, and its training zarr is that scan
  resampled in x/y/z onto the 9.362 µm / 28-layer frame (`assemble_training_segments._assemble_resampled_volume`).
  Register the 2.4 µm render against the native 8.64 µm render (factor 3.599). Then map native → training
  frame by 9.362/8.64. `build_pairs.py` does both.
- **Other training frames.** The frames of 0139, 0814 and 500P2 are the low volume's level 0 unchanged.
  0841's frame is the level 0 cropped to (0, 3760, 0, 4900), which is the whole current volume.
- **Depth.** 28 layers × 9.362 µm ≈ 262 µm ≈ 109 layers × 2.4 µm (w013 and paris4 have 109). Both renders
  should cover the same slab around the surface; `probe_pairs.py` prints both thicknesses.
- **Acquisition differs.** Most 2.4 µm scans use 0.22 m propagation (0009B 0.35 m, 500P2 0.4 m) versus
  1.2 m for the low scans, and 77–78 keV versus 113 keV (500P2 is 111 keV on both).
  - Contrast and phase-contrast edge fringes therefore differ, so targets need intensity matching (§4).
  - 500P2's `4.317um-1.2m-111keV` render matches the low scan's propagation and energy: a cleaner but only
    2.2× target for that one segment.

### Optional extra pairs (no labels needed)

The objective below does not use labels. So the unlabelled PHerc0139 segments assembled elsewhere could
add native-regime pairs from random footprint tiles:
- w047, w056, w058, w052, w049, w046, w038, w037 and w034 (segment names in `assemble_training_segments.SEGMENTS`)
- keep w055 out: it is the hallucination-check fragment

Their 2.4 µm volume names are not verified. Add them to `pairs.py` with `role: "unlabelled"` only after
`probe_pairs.py` confirms they exist. `_select_tiles` would then need the footprint branch for that role.

## 2. Pipeline (commands for the next agent)

Run from the repo root, with `norm_cache.json`, `masks/`, `dilated_inklabels/` and `ves_zarrs2/` present.

1. **Metadata:** `python crossres/probe_pairs.py`
   - It reads only `.zarray` files and writes `crossres/pairs_manifest.json`.
   - It flags any pair whose render sizes disagree with its voxel sizes by more than 3% (different crop,
     mesh version or pixel spacing).
2. **Midslice sanity and registration:** `python crossres/sanity_midslice.py`
   - This is the requested check: midslice of the low render at level 0; midslice of the high render at
     level 2 (the 2.4 µm render downscaled 4×); resample onto the low grid; binarise papyrus (Otsu inside
     the footprint); Dice and NCC.
   - Three hypotheses are scored: as rendered, phase-correlation shift (both signs), and SIFT + RANSAC
     similarity. The best is kept in `crossres/sanity/registration.json` as `low_to_high0`
     (native-low px → high level-0 px).
   - Also saved: a depth offset from the two layer-intensity profiles, and
     `crossres/sanity/<name>_overlay.jpg` (green = high, red = low) plus `_checker.jpg`.
   - Cost: surface-volume chunks span the full depth, so one slice streams the whole level (~0.1–1 GB low,
     ~1–4 GB level-2 high per segment). Only slices and reports reach disk. `--high-level 3` is 4× cheaper
     for a first pass.
   - Verdicts: Dice ≥ 0.85 good; 0.70–0.85 check the overlay; < 0.70 `INVESTIGATE` (`build_pairs.py`
     skips these until resolved). Checklist for a low Dice:
     - **Large constant offset.** The renders use different crops or UV origins; the phase shift or SIFT
       should fix it. If `as_rendered` is far below the others, that was it.
     - **Scale off** (`residual_scale` ≠ 1 by more than 2%). One render is not at native pixel spacing, or
       it is a different mesh version. Compare `probe_pairs.py` ratios.
     - **Mirrored or rotated.** SIFT with no inliers but a clean overlay after `cv2.flip`. Add a flip
       hypothesis for that pair.
     - **Non-rigid.** Good in one region, bad elsewhere: a different mesh or re-flattening.
       - Tiles still work, because `build_pairs.py` re-estimates a residual shift per 512 px tile and drops
         tiles with midslice NCC < 0.3.
       - For badly warped pairs, fit a dense flow at level 2 (the old 0841 label registration used SIFT +
         optical flow and reached NCC 0.77).
     - **Wrong layer.** Papyrus patterns look alike but fibres don't line up, and the depth offset is at
       the ±3-slice search limit. The two renders sit on different sheets.
     - **Contrast, not geometry.** The overlay lines up but Dice is mediocre. 0.22 m phase contrast brightens
       edges, so Otsu thresholds differ. Trust NCC and the overlay over Dice here.
3. **Disk budget:** `python crossres/build_pairs.py --count-only --plan depth [--include-holdouts]`
   - Prints tile counts, streamed GB and uncompressed disk GB per pair, from the label (or mask) PNGs.
   - Defaults: 512 px tiles inside the ink bounding box ± 256 px; slices 8–19 (the MAE window moves inside).
   - Plan `depth` streams level 2 and stores 32 × 512² target layers per tile; plan `xyz` streams level 1
     and stores 32 × 1024².
4. **Build tiles:** `python crossres/build_pairs.py --zarr-dir ves_zarrs2 --plan depth` (then `--plan xyz`)
   - Output: `crossres/pairs/<plan>/<zid>/pairs.zarr` + `meta.json`.
   - Check `meta.json` `ncc_median`, `dropped` and `tile_depth_shift_slices` per pair.
   - `--include-holdouts` adds 64 random footprint tiles each for 0841 and 0009B (optimistic runs only).
5. **Pretrain:** see §0.3 for the loop over the three plans.
6. **Register the checkpoint for campaigns** so `preflight_pretraining` accepts it without retraining:
   ```python
   KEY = "early_gated_native96_crossres"
   campaign34.PRETRAIN_SPECS[KEY] = campaign31._spec(
       *campaign34.EARLY_GATED_ARGS, required=campaign34.EARLY_GATED_REQUIRED, ctx=96, ds=1)
   # copy models/mae_crossres_native96.pth to campaign34._pretrain_path(KEY), then:
   campaign34._pretrain_path(KEY).with_suffix(".complete.json").write_text(
       json.dumps(campaign34._pretrain_metadata(KEY), indent=2) + "\n")
   ```
   Arms then use `arch={"pretrain_key": KEY}`.
   - The key names equal the production MAE's, so transfer coverage is 100%.
   - For the fiber variant, pass `--fiber-coordinate-branch`, init from
     `mae_nnunet_192_campaign34_early_gated_fiber_native96_2k.pth`, and add the fiber spec args.

## 3. MAE options considered

Superseded by §0: plans D and X extend option #5 with a 4× depth target, and S is its no-pair control.

"Target" means what the throwaway head reconstructs. The input is always a 9.36 µm crop.

| # | objective | what the backbone must learn | for | against | verdict |
|---|---|---|---|---|---|
| 1 | Same-grid clean target: 2.4 µm pooled onto the 9.36 grid ("noise2clean") | undo 113 keV / 1.2 m noise and blur | cheapest; exact alignment matters less | teaches smoothing; the frozen denoiser arm showed smoothing removes faint ink (worst c38 arm, 0841 R@5% 0.098) | no |
| 2 | 2× super-resolution, no masking | predict ~4.8 µm structure from local 9.36 µm evidence | directly the missing detail | solvable by the first layers alone, so deep features barely change | partial |
| 3 | 4× super-resolution (2.4 µm level 0) | predict 2.4 µm detail | maximal detail | most of it is unrecoverable from 9.36 µm, so MSE regresses to blur; 4× the disk and download; 4× more misregistration-sensitive | no |
| 4 | Masked input + 2× SR target (cross-resolution MAE) | infer hidden columns from context **and** render them finer than the input | deep context features plus sub-voxel texture; masking prevents a pure local deblur | only 11–13 segments; drifts the backbone towards those scrolls | strong |
| **5** | **#4 mixed 50/50 with the existing 9.36 µm MAE over all 37 volumes, warm-started from the production MAE** | #4, while keeping the exposure to every training, held-out and test volume | adds the paired signal without losing what the current MAE gives unseen scrolls; cheap (2k steps); a drop-in checkpoint | the SR head never touches fine-tuning, so the transfer is only as good as the features | **chosen** |
| 6 | Distil features or ink maps from a 2.4 µm ink model | copy a teacher's evidence | strongest signal if the teacher is good | needs a trained 2.4 µm teacher; ink maps are pseudo-labels (excluded); features are the only allowed form | later |
| 7 | Contrastive 9.36 ↔ 2.4 agreement | acquisition-invariant embedding | — | an invariance objective, which this project excludes | excluded |

### Why #5 should raise recall without costing papyrus

At 9.36 µm, faint ink is a sub-voxel change in texture and density at the surface. The model only finds
ink its features represent.
- **Rewards the right features.** Predicting 4.8 µm structure from 9.36 µm context rewards exactly the
  features that carry sub-voxel detail, so more of the faint strokes become separable at a fixed low FPR.
- **Not a smoothing prior.** The target contains more detail than the input, which is the opposite of the
  denoiser's objective (the worst arm in c38).
- **Keeps test-scroll exposure.** The standard branch keeps the current MAE's unlabelled exposure to every
  test and held-out volume. Dropping it would move the representation towards 11 segments and risk more
  false positives on unseen scrolls.
- **Leaves calibration alone.** Both heads are discarded, and fine-tuning starts from a backbone with the
  same keys and the same recipe.

### Spec of #5 (as implemented in `mae_pretrain_crossres.py`)

- **Architecture.** The production early-gated native-96 backbone (`--early-2d-unet --gated-stems
  --norm-mode ibn_full`), ctx 96, ds 1, depth 8, slices 10–17 (the MAE window). The 9.36 µm recon head is
  unchanged. SR head: conv3×3 → GELU → conv1×1 to 8×4 channels → PixelShuffle(2), zero-initialised, on the
  same decoder map.
- **Batch.** 16 standard crops (the physical round robin over all 37 volumes, as now) + 16 paired crops
  (round robin over physical scrolls, then segments).
- **Masking.** The existing 4×4 block mask at 0.65 on all 32. Loss:
  - `rec`: masked-voxel MSE on the 9.36 µm input, for all 32 samples;
  - `sr`: MSE to the 2× target on the paired 16, weight 1 on masked and 0.25 on visible positions, only
    where the target is valid;
  - total = `rec + sr`.
- **Normalisation.** Inputs use the scroll's `norm_cache.json` stats, exactly as in training. Targets get a
  per-segment robust affine (median/MAD) onto the normalised input's intensity distribution. This absorbs
  the 78 vs 113 keV contrast difference without a learned per-scroll parameter.
- **Target construction** (`build_pairs.py`):
  - Stream level 1 (~4.8 µm, the pyramid's 2× average of 2.4 µm, which also halves 2.4 µm noise).
  - Box-average the 2.4 µm layers inside each 9.362 µm slice, shifted by the measured depth offset.
  - Warp onto 2× the training grid.
  - Re-estimate a residual shift per tile and drop tiles with midslice NCC < 0.3.
- **Optimiser.** Warm start from `mae_nnunet_192_campaign34_early_gated_native96_2k.pth`, lr 1.5e-4
  (half the from-scratch lr), 5% warmup, cosine to 2%, weight decay 0 (the production MAE used 1e-4).
- **Health check.** `monitor_sr` (tiles never trained on) must fall clearly below `monitor_sr_trilinear`
  (plain upsampling of the input); otherwise the head learned nothing. `monitor_rec` should stay near the
  production MAE's value, or the standard objective is being traded away.
- **Knobs worth one ablation each.**
  - `--paired-frac 0.25` or `0.75`.
  - `--visible-weight 0` (masked positions only).
  - A higher weight on the surface slices (not implemented; ink sits on the middle slices of the window).
  - 500P2's 4.317 µm / 1.2 m target instead of its 2.215 µm one.

## 4. How this interacts with the current pretrain

**Option A: a clean replacement, from scratch.** Train #5's mixture from random initialisation. It
discards 2k steps of work that the mixture would largely redo, and it confounds "new objective" with
"new run". **Not recommended.**

**Option B: on top (continued pretraining). Recommended.**
- **Starting point.** Warm-start from the current MAE, which saw all 37 volumes including the test scrolls.
- **What continues.** The standard objective keeps running on those same 37 volumes for half of every
  batch, and the paired SR objective is added.
- **What changes.** The result is a strict extension of the current checkpoint, same keys and same
  architecture, so it replaces it only in the `init_weights` slot of an arm. The fine-tuning recipe does
  not change.
- **Clean comparison.** A campaign arm with the same fine-tune config and only the new key isolates the
  pretraining effect.

**Option C: a separate adapter or branch (the SR features as an extra input path).** More parameters and
fine-tune changes, and no advantage for a first test. **Not now.**

### Held-out scrolls

The current MAE already reconstructs 0841 and 0009B unlabelled, as it does every test scroll.
- **What the pairs add.** They give something the real test scrolls will not have: a 2.4 µm view of the
  same papyrus. Pretraining on the 0841 / 0009B pairs therefore flatters the holdout metrics.
- **Honest run (primary):** `--exclude-holdout-pairs`. The pairs come from the 11 training segments only;
  the standard branch still includes 0841, 0009B and the test scrolls unlabelled, as now.
- **Optimistic variant:** `build_pairs.py --include-holdouts` then pretrain without the flag. Holdout tiles
  are random footprint tiles, never chosen from their labels. Report it separately as an upper bound.

## 5. Evaluation

Arms, run under identical fine-tune configs:
1. The current base with the production MAE (control).
2. The same config with `early_gated_native96_crossres` built with `--exclude-holdout-pairs`.
3. The best combined recipe (`holdout_n96_combined` from campaign 38) with the crossres key.
4. Optionally, the optimistic run including holdout pairs.

Judge each arm on:
- **Metrics:** 0841 and 0009B `R_M/.../RecallAt1PctFPR`, `RecallAt5PctFPR` and `PartialAUCAt1PctFPR`
  (Train/Valid), plus the full-render visuals.
- **Noise:** replicate noise is ~0.03 recall, so one seed is not enough for a small gain; add a seed-42
  replicate of the winner.
- **Papyrus:** check the renders for inflated papyrus (the private-heads failure).
- **Not in-domain PR-AUC:** in-domain validation PR-AUC does not track the holdouts.

## 6. Risks

- **Misregistration.** A few-pixel offset makes the SR target teach blur. Mitigations:
  - Dice/NCC gate per pair;
  - residual shift per tile;
  - NCC tile filter;
  - target at 2× rather than 4× (half as sensitive).
  Inspect `meta.json` drops before training.
- **Phase-contrast fringes** (0.22 m) in the target could teach edge hallucination. Level-1 averaging
  softens them. If `monitor_sr` improves but downstream papyrus gets noisier, blur the target (σ 0.5 px)
  or use 500P2's 1.2 m render.
- **Too few paired scrolls** (3 physical scrolls for honest runs, 0139 dominant). The physical round robin
  equalises scrolls; the standard branch keeps the representation broad.
- **Disk.** Use `--count-only` first. The standard branch needs all 37 training zarrs locally, as fine-tuning does.

## 7. Files

| file | purpose |
|---|---|
| `crossres/pairs.py` | pair table, URL builder (refuses 1.129 µm), scale helpers; `python crossres/pairs.py` prints the scaling table |
| `crossres/probe_pairs.py` | metadata-only existence and size check → `pairs_manifest.json` |
| `crossres/sanity_midslice.py` | ÷4 midslice Dice / NCC, registration, depth offset, overlays → `sanity/registration.json` |
| `crossres/build_pairs.py` | tile selection (labels for training scrolls, random footprint for holdouts), 2× target warping, per-tile residual alignment → `pairs/<zid>/pairs.zarr`; `--count-only` for disk planning |
| `crossres/mae_pretrain_crossres.py` | mixed standard + cross-resolution MAE, warm-started; saves a drop-in backbone checkpoint |
