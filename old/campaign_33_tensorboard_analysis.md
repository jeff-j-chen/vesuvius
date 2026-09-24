# Campaign 33 TensorBoard analysis

Snapshot: 2026-09-24. 16 of 17 arms are complete. `early_deep_residual_patchdro_nods` crashed at start: its event file has no epochs.

Scripts:
- `/data/extra/tmp/c33_extract.py` extracts the raw scalars.
- `/data/extra/tmp/c33_stats.py` runs the statistics and writes the full output to `/data/extra/tmp/c33_stats.txt`.

## 0. Comparability and noise

Campaign 33 stands alone. The labels changed to dilated researcher labels, pherc0139 grew from 2 to 9 patches, and there are 23 validation patches with equal domain weights. No run from campaigns 29–32 can be compared on absolute values. AP is about 0.15 higher across the board than in Campaign 31.

**Campaign 33 has no replicate pairs and no plain mid control.** The references used below are:

| label | runs | notes |
|---|---|---|
| early reference (E42) | `early_patchdro_seed42` | full training corpus |
| early3 | E42 + both early LODO runs | only on the 10 domains and 20 patches that every run trains on ("shared") |
| midLODO | the two mid LODO runs | the only mid runs without GroupDRO; shared subset only |

**Noise.** Single-run σ is borrowed from the Campaign 31 replicate groups. To check that it still applies, I computed the spread among the three early runs and between the two mid LODO runs on shared domains. This spread includes any LODO spill-over, so it is an upper bound on noise:

| metric | Campaign 31 σ | Campaign 33 internal spread |
|---|---:|---:|
| Equal-domain PR-AUC mean (shared) | 0.0092 | 0.0048 |
| Train per-patch mean (shared) | 0.0062 | 0.0064 |
| Per-domain PR-AUC, median ratio | — | 1.01 |

The ratio is only a median. pherc1667, pherc0009b and pherc0814 spread about 2x more than their Campaign 31 σ. **Per-domain claims below use the larger of the two estimates.**

Per-patch spread is 0.048 median, with a maximum of 0.08. Patch `20260126000000` ranges from 0.14 to 0.59 across runs, so single-patch claims are not reportable.

95% detectable difference, one run vs one run:

| metric | detectable Δ |
|---|---:|
| Character AP, epochs 7–9 | 0.011 |
| Best-epoch AP | 0.0105 |
| Best-epoch F1 | 0.019 |
| Pixel PR-AUC, epochs 7–9 | 0.029 |
| Train character F1 | 0.029 |
| Train character AP | 0.020 |
| Train pixel PR-AUC | 0.009 |

Calibrated F1 has only 1 degree of freedom for its noise estimate and cannot be tested.

## 1. Run table

| arm | F1best | AP7-9 | APbest | calF1 7-9 | PR-AUC 7-9 | train F1 | train AP | s/epoch |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| `early_patchdro_native196` | **0.693** | **0.761** | **0.773** | **0.702** | **0.774** | 0.822 | 0.898 | 657 |
| `mid_groupdro_ema` | 0.652 | 0.746 | 0.747 | 0.694 | 0.763 | 0.869 | 0.932 | 241 |
| `early_patchdro_ema` | 0.651 | 0.743 | 0.744 | 0.682 | 0.761 | 0.860 | 0.926 | 203 |
| `mid_air_offset_m1` | 0.679 | 0.742 | 0.747 | 0.689 | 0.752 | 0.862 | 0.927 | 223 |
| `early_patchdro_ink_band` | 0.666 | 0.737 | 0.742 | 0.683 | 0.754 | 0.845 | 0.912 | 245 |
| `early_patchdro_depth_interp` | 0.672 | 0.735 | 0.743 | 0.683 | 0.749 | 0.858 | 0.926 | 258 |
| `early_deep_residual_patchdro` | 0.662 | 0.733 | 0.739 | 0.674 | 0.751 | 0.876 | 0.940 | 222 |
| `early_patchdro_seed42` (E42) | 0.667 | 0.732 | 0.735 | 0.676 | 0.751 | 0.863 | 0.928 | 215 |
| `early_residual_depth3_patchdro` | 0.668 | 0.724 | 0.726 | 0.675 | 0.742 | **0.886** | **0.943** | 212 |
| `mid_pcgrad_groups4` | 0.646 | 0.722 | 0.730 | 0.676 | 0.740 | 0.874 | 0.931 | 560 |
| `early_planar_patchdro` | 0.647 | 0.699 | 0.700 | 0.670 | 0.716 | 0.836 | 0.905 | 243 |
| `early_planar` | 0.639 | 0.674 | 0.679 | 0.660 | 0.698 | 0.838 | 0.902 | 210 |
| LODO runs | lower by construction; see §3 | | | | | | | |

## 2. Per-test analysis

### `early_patchdro_seed42` (replicate of the Campaign 31 winner)

It is the reference, and there is no same-label seed-41 twin. On shared domains the two early LODO runs are within 0.008 of it on equal-domain PR-AUC (not significant). The recipe behaves stably.

### `early_patchdro_ema`

- **Versus E42:** AP7-9 **+0.011 (p=0.049)**, best AP +0.009 (p=0.082), best F1 −0.016 (p=0.078). 11 of 12 domains up.
- **Versus early3, shared domains:** +0.024 (p=0.072), 10 of 10 domains up.
- **Verdict:** a small, borderline, very consistent ranking gain. It matches Campaign 31, where EMA helped only when combined with GroupDRO. Keep EMA on by default for GroupDRO arms.

### `mid_groupdro_ema` (EMA without anchor)

- **Versus the mid LODO mean, shared domains:** +0.029 (p=0.051), 8 of 10 domains up. GroupDRO+EMA helps the mid backbone again.
- **Versus E42:** AP7-9 **+0.014 (p=0.022)**.
- **Versus `early_patchdro_ema`:** everything not significant (AP +0.003).
- **Verdict:** mid with domain GroupDRO+EMA equals early with patch GroupDRO+EMA. Anchor is not needed; this replaces the Campaign 31 triple. The comparison confounds backbone with GroupDRO type, so neither is shown to be better.

### `early_planar` and `early_planar_patchdro`

- **Planar, given patch GroupDRO:** AP7-9 **−0.033 (p=0.001)**, best AP **−0.036 (p<0.001, Holm 0.052)**, PR-AUC −0.035 (p=0.028). Only 2 of 23 patches and 2 of 12 domains are up. Train is also lower (train AP −0.024).
- **Patch GroupDRO, given planar:** AP7-9 **+0.025 (p=0.002)**. Patch GroupDRO now shows a significant AP gain on three backbones (+0.020, +0.026 and +0.025).
- **Verdict:** preventing depth mixing before the collapse hurts badly. Reject planar convs. The early cross-slice convolutions carry real signal.

### `early_patchdro_depth_interp`

- **Versus E42:** AP7-9 +0.003, best AP +0.008 (p=0.11), best F1 +0.005, all not significant. 18 of 23 patches up, but only 3 of 12 domains up.
- **Verdict:** null. Slice-interpolation dropout neither helps nor hurts.

### `early_patchdro_ink_band`

- **Versus E42:** AP7-9 +0.005 and best AP +0.006, both not significant. Train pixel PR-AUC is −0.014 (p=0.009).
- **Verdict:** null on validation. Per-domain window centring does not help. Combined with the Campaign 31 offset results, the default surface-centred window is adequate.

### `early_patchdro_native196`

- **Versus E42:**
  - AP7-9 **+0.029 (p=0.001)**
  - best AP **+0.038 (p<0.001, survives Holm, 0.041)**
  - best F1 **+0.026 (p=0.016)**
  - PR-AUC +0.023 (p=0.095)
  - calibrated F1 +0.026 (untestable)
  - 18 of 23 patches up
- **Versus `early_patchdro_ema` (the next-best early arm):** best F1 **+0.043 (p=0.002)**, AP7-9 **+0.018 (p=0.008)**.
- **Against every other arm:** it beats all of them on AP7-9 (p≤0.016) and on best AP (p≤0.001).
- **Train is significantly *lower*:** train F1 **−0.041 (p=0.015)**, train AP −0.030. It generalises better while fitting the training set less, the opposite of depth3.
- **Caveats:**
  - It is a single seed with its own pretrain.
  - The change is resolution only: both arms see the same field (192 vs 196 raw px), but the reference average-pools it 2x inside the model to a 96-px grid, while native196 keeps the full 196-px grid.
  - It is 3.1x slower.
  - pherc0814 is lower (−0.085).
- **Verdict:** the clear winner of Campaign 33 and the largest single effect seen so far. It needs a replicate; the field of view was matched, so the gain is attributable to input resolution.

### `early_deep_residual_patchdro`

- **Versus E42:** everything on validation is not significant (AP +0.001, best F1 −0.005). Train pixel PR-AUC is +0.009 (p=0.041).
- **Verdict:** under patch GroupDRO the Campaign 31 result reproduces. EDR ties the shallow early model on validation and fits the training set slightly more. It adds nothing. The native-resolution EDR variant (`_nods`) crashed, so the most interesting EDR question is still untested.

### `early_residual_depth3_patchdro`

- **Versus E42:** AP7-9 −0.008 (p=0.11), best AP −0.010 (p=0.068), best F1 +0.001. Train F1 is +0.023 (p=0.097) and train pixel PR-AUC +0.012 (p=0.014).
- **Versus early3, shared subset:** validation −0.008 (not significant), train per-patch mean **+0.035 (p=0.004)**.
- **Versus EDR:** best AP −0.013 (p=0.024), matching Campaign 31 again.
- **Verdict:** it has the best train scores in the campaign (train F1 0.886, train AP 0.943) and middle-of-pack validation, ranking 9th of 12 non-LODO arms on AP7-9. The pattern is the same as in Campaign 31: more fitting without better generalisation. Its best-F1 tie with E42 is real, but that is not "great" on validation.

### `mid_pcgrad_groups4`

- **Versus `mid_groupdro_ema`:** AP7-9 **−0.024 (p=0.002)**, best AP **−0.017 (p=0.010)**. Only 2 of 23 patches up.
- **Versus E42:** best F1 **−0.021 (p=0.036)**, AP7-9 −0.010 (p=0.06).
- **Versus the mid LODO mean, shared domains:** +0.007, not significant. It is no better than mid without any GroupDRO.
- **The per-patch pattern the user noticed is on *train*.** On the 23 training patches, pcgrad is top-3 on 14 and bottom-3 on 1. On validation (epochs 7–9) it is top-3 on 1 and bottom-3 on 11. Its train pixel PR-AUC is 0.982, far above the GroupDRO arms (0.95), and train F1 is third best.
- **Verdict:** grouped PCGrad is a strong fitter and a weak generaliser. It costs 2.6x, with no validation gain over plain mid and a significant loss versus GroupDRO+EMA. Reject.

### `mid_air_offset_m1` (mid + patch GroupDRO + window −1)

- **Versus E42:** best AP +0.011 (p=0.040), AP7-9 +0.010 (p=0.071).
- **Versus `mid_groupdro_ema`:** best F1 +0.026 (p=0.016), AP not significant.
- **Versus the mid LODO mean, shared domains:** +0.020 (not significant), 7 of 10 domains up.
- **Verdict:** a top-4 arm, but **the offset cannot be isolated.** No mid + patch-GroupDRO run at offset 0 exists, so the offset effect is confounded with patch GroupDRO on mid. It is at least not harmful, unlike the +2/+3 air offsets in Campaign 31.

## 3. LODO

Design: two held-out domains, pherc0172 (2 patches) and phercparis4 (1 patch), each on mid (no GroupDRO) and early (patch GroupDRO). The held-out domain still has its validation region scored. Its "seen" score comes from the otherwise-identical LODO run that held out the other domain; for early, the E42 run is averaged in as well. MAE pretraining still saw the held-out volumes unlabeled, so this is an optimistic estimate of transfer to a truly new scroll.

### The held-out domain collapses

| family | held-out domain | domain PR-AUC 7-9, held → seen | Δ | patch F1 7-9, held → seen |
|---|---|---|---:|---|
| mid | pherc0172 | 0.593 → 0.812 | **−0.220 (p<0.001)** | 0.170 / 0.142 → 0.707 / 0.752 |
| mid | phercparis4 | 0.632 → 0.849 | **−0.217 (p<0.001)** | 0.516 → 0.757 |
| early + patch GroupDRO | pherc0172 | 0.498 → 0.775 | **−0.277 (p<0.001)** | 0.398 / 0.316 → 0.707 / 0.726 |
| early + patch GroupDRO | phercparis4 | 0.619 → 0.834 | **−0.215 (p<0.001)** | 0.426 → 0.734 |

- **Rank:** when seen, both domains rank in the top half of the 12 domains (0.77–0.85). When held out, they fall to last or second-to-last, below the historically hardest seen domains, pherc0841 and pherc0500p2 (0.60–0.70).
- **The two families lose about the same:** −0.22 versus −0.22/−0.28. Early collapse with patch GroupDRO gives no measurable advantage on an unseen domain, even though it wins on within-domain validation.
- **Calibration differs:** early keeps a higher fixed-threshold F1 on the held-out pherc0172 patches (0.40/0.32 versus 0.17/0.14 for mid) despite a lower PR-AUC.

### Other domains do not move

- **Mid, holding out pherc0172 versus holding out phercparis4:** three domains are nominally significant:
  - pherc1667 −0.033
  - pherc0009b −0.061
  - pherc0814 +0.055

  With the conservative σ, none is significant, and none survives Holm across 10 domains.
- **Early:** 0 of 10 domains differ between the two LODO runs. Against the full-data E42 run, 1 of 20 domain comparisons is nominally significant (pherc0814 −0.047), which is about what chance predicts.
- **Mean over the 10 shared domains:** −0.009 (mid), −0.002 (early), −0.008 and −0.006 (each early LODO versus E42). All are not significant.
- **Train on the remaining domains** changes by at most 0.008 PR-AUC.

**Conclusion:** removing a domain costs that domain about 0.22 PR-AUC. It has no detectable effect (< about 0.02–0.03 PR-AUC) on any other domain. Domains do not measurably support each other: each one's validation score comes from its own training data. Within-domain validation therefore overstates new-scroll performance by about 0.22 PR-AUC, and none of the Campaign 33 tricks has yet been tested against that gap.

## 4. Consistent best performers

- **Validation:** `early_patchdro_native196`, alone. It beats every other arm on AP7-9 (p≤0.016) and best AP (p≤0.001), and has the highest best F1, calibrated F1 and PR-AUC. The next cluster cannot be separated internally: `mid_groupdro_ema`, `early_patchdro_ema`, `mid_air_offset_m1`, then `ink_band`, `depth_interp`, EDR, E42.
- **Train:** `early_residual_depth3_patchdro` is top on train F1, train AP and per-patch train mean. The top four cannot be separated (p≥0.18): depth3, EDR, `mid_pcgrad_groups4`, `mid_groupdro_ema`. On train pixel PR-AUC, the mid runs without GroupDRO lead (0.982–0.984), because GroupDRO lowers pixel PR-AUC on train.
- **Train and validation do not agree.** Across the 12 non-LODO arms, the rank correlation of train F1 with validation AP is −0.11 (p=0.75). The validation winner has one of the lowest train scores.

## 5. Recommendations

1. **Replicate native196** (seed 42) and add **native196 + EMA**. Also re-run the crashed `early_deep_residual_patchdro_nods`: EDR at native resolution is still the open architecture question.
2. **Map resolution against field of view:** native resolution at a smaller field (128 px), and 2x downsampling over a larger field (384 px, same network grid as native196). Also try a **mid native196** with domain GroupDRO+EMA, to see whether the gain belongs to resolution or to the early backbone.
3. **Make LODO the headline metric for promoted recipes.** Run LODO pherc0172/phercparis4 for native196 and `mid_groupdro_ema`. A recipe is only better for new scrolls if it narrows the 0.22 gap. Add a third held-out domain (pherc1667 or pherc0814) for breadth.
4. **Isolate the −1 offset:** a mid + patch GroupDRO run at offset 0.
5. **Drop:** planar convs, depth-interp, ink-band windows, grouped PCGrad, and depth3/EDR at 2x downsampling.
6. **depth3 + PCGrad** has not been tested; PCGrad has only ever run on the mid backbone. Both components independently raise train fit without improving validation, so the combination is expected to fit train even better and not help validation. It is low priority. If it is run, score it with LODO, not within-domain validation.
