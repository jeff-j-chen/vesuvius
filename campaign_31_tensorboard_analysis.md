# Campaign 31 TensorBoard analysis

Snapshot: 2026-09-23. 30 of 34 Campaign 31 arms are complete. `mid_pcgrad_gram_sparse4` stopped after epoch 8. `early_raw_ibn` is at epoch 3. `researcher_ds2_full`, `researcher_full`, and `mid_pcgrad_gram_groupdro` have not started.

Scripts: `/data/extra/tmp/c31_extract.py` reads the raw scalars from `runs_archs29` and `runs_archs31` into `c31_raw.json`, using the same EventAccumulator logic as `old/analyze_tb.py`. `/data/extra/tmp/c31_stats.py` runs the statistics and writes the full contrast output to `/data/extra/tmp/c31_stats.txt`.

## 1. Noise floor

### Replicate groups

These configs differ only by seed or by recorded no-op defaults:

| group | runs | notes |
|---|---|---|
| MIDC (mid control) | c29 `baseline`, c30 `current_mid_control`, c31 `mid_control_seed42` | c29 compiled; seeds 41/41/42 |
| TRIPLE | c31 `mid_groupdro_anchor_ema`, `triple_seed42` | seeds 41/42 |
| EDR | c30 `early_deep_residual2d`, c31 `early_deep_residual_replica` | **same seed** |
| OV12 | c29 `overlap_depth12`, c31 `mid_overlap12` | **same seed** |

Two pairs used the same seed and still diverged, so the runs are non-deterministic regardless of seed. I pooled the within-group variance to get a single-run σ with 5 degrees of freedom. Every contrast below is a linear combination of runs. Its SE is σ·‖w‖, tested with t on 5 df. Groups are used as means wherever possible, e.g. the reference is the mean of 3 control runs, not one run.

| metric | single-run σ | 95% detectable: run vs MIDC mean | 95% detectable: run vs run |
|---|---:|---:|---:|
| Character F1, epoch 9 (0.5 threshold) | 0.0253 | ±0.075 | ±0.092 |
| Character F1, mean of epochs 7-9 | 0.0328 | ±0.097 | ±0.119 |
| **Character F1, best epoch** | **0.0052** | **±0.016** | **±0.019** |
| Character AP, epoch 9 | 0.0079 | ±0.023 | ±0.029 |
| **Character AP, mean of epochs 7-9** | **0.0030** | **±0.009** | **±0.011** |
| **Character AP, best epoch** | **0.0029** | **±0.009** | **±0.011** |
| Pixel PR-AUC, epoch 9 | 0.0109 | ±0.032 | ±0.040 |
| Equal-patch mean F1, epochs 7-9 | 0.0308 | ±0.092 | ±0.112 |
| Bottom-4 patch mean, epochs 7-9 | 0.0520 | ±0.154 | ±0.189 |
| Worst patch, epochs 7-9 | 0.1073 | ±0.318 | ±0.390 |
| Single-patch F1, epoch 9 | 0.016–0.153 | — | — |

### What the noise floor implies

1. **Fixed-threshold endpoint F1 cannot resolve anything smaller than about 0.08–0.09.** No Campaign 29, 30, or 31 arm differs from the control by that much in either direction, apart from the fragment-only runs. Every endpoint-F1 ranking in the previous analysis is inside the noise.
2. **The endpoint noise is mostly calibration drift, not ranking quality.** The calibrated threshold swings from 0.10 to 0.57 between neighboring epochs of the same run. The endpoint collapses all coincide with extreme calibrated thresholds:
   - `mid_groupdro_anchor` epoch 9: F1 0.505 at 0.5, 0.552 calibrated, threshold 0.17.
   - `mid_pcgrad_gram_exact` epoch 7: F1 0.472 at 0.5, 0.534 calibrated, threshold 0.10.
   - `early_raw_instance_groupdro` epoch 9: F1 0.475 at 0.5, 0.552 calibrated, threshold 0.12.

   AP and best-epoch F1 are 3–10x less noisy. **They are the primary metrics below.** Best-epoch F1 is optimistic because it takes the maximum over 10 validation epochs, but the bias is the same for every arm.
3. **There are no per-patch specialists.** The two control seeds differ by 0.138 on pherc0841 (`20260221022814`) at epoch 9: 0.389 versus 0.527. On `20231201215900` the three controls read 0.085, 0.523, and 0.218. The per-patch σ is larger than almost every patch-win margin in the previous analysis. The c29/c30 patch-winner table, the "specialist" labels, and the patch-correlation structure built from single endpoints should all be withdrawn.
4. **Lower-tail claims are not testable at n=1.** With worst-patch σ ≈ 0.107, the earlier "best worst-patch" claims for `early_raw_instance` and `mae_anchor` are not significant: +0.17 versus MIDC, p=0.22, and +0.19, p=0.18.
5. **Early training is reproducible; trajectories diverge later.** At epoch 3 the F1 σ is only 0.0045, but it reaches 0.025 by epoch 9.
6. **This σ understates noise for some arms.** The replicates reused the same MAE pretrain checkpoint. Arms that got a new Campaign 31 pretrain also carry pretrain-instance variance that σ does not capture:
   - `mid_depth12`, `mid_overlap8`, `early_overlap12`
   - `early_residual_depth3`, `early_residual_extra320`, `early_deep_nonresidual`
   - `early_gated_instance`, `early_raw_ibn`

   Treat their significant results as weaker.

### Notation and multiple comparisons

- Δ is the arm minus the reference.
- "AP7-9" is character AP averaged over epochs 7-9; "F1best" is character F1 at its best epoch.
- p is two-sided. `*` means p<0.05 and `**` means p<0.01.

About 70 contrasts were run. Under Holm correction, these survive on AP7-9:

- both air offsets;
- `topk_bag_positive`;
- TRIPLE versus control;
- `entropy_strong_topk`;
- `early_raw_instance` versus `early_gated`;
- `early_overlap12`;
- `deep_nonresidual` versus `early_gated`;
- c29 `physical_patch_groupdro` versus control;
- `early_gated_patch_groupdro` versus control.

On F1best, only `topk_bag_positive` and `air_offset3` survive Holm.

## 2. Per-test analysis

### 2.1 `mid_control_seed42` (noise measurement)

| | F1 ep9 | F1 7-9 | F1best | AP7-9 | pherc0841 ep9 |
|---|---:|---:|---:|---:|---:|
| c30 control, seed 41 | 0.5424 | 0.5439 | 0.5589 | 0.5862 | 0.389 |
| c31 control, seed 42 | 0.5460 | 0.5376 | 0.5460 | 0.5872 | 0.527 |

No aggregate differs significantly. Patch swings as large as 0.14 are ordinary. The calibrated F1 over epochs 7-9 is 0.5513, and the calibrated thresholds of 0.22–0.30 show that the 0.5 threshold is badly off for the control too.

### 2.2 `mid_groupdro_anchor`

- **Versus MIDC:** F1best +0.012 (p=0.098), AP7-9 +0.002 (ns), F1 7-9 +0.009 (ns).
- **Versus c29 `physical_groupdro`:** F1best +0.012 (ns), AP best −0.008 (p=0.11).
- **Versus c29 `mae_anchor`:** all ns.
- **Linkage:** the combination beats neither single component. On its own, physical GroupDRO beats the control on AP best (+0.013, p=0.011). MAE anchor does not (AP best −0.001).
- **Verdict:** no evidence that anchor adds to GroupDRO.
- **Aside:** the 387 s/epoch is twice the 207 s of the identical-plus-EMA arm. It is almost certainly host contention, not a property of the arm.

### 2.3 `mid_groupdro_anchor_ema` and `triple_seed42` (TRIPLE, n=2)

| | F1best | F1 7-9 | AP7-9 | pixel PR-AUC | calibrated F1 7-9 |
|---|---:|---:|---:|---:|---:|
| seed 41 | 0.5646 | 0.5606 | 0.6071 | 0.6226 | 0.5684 |
| seed 42 | 0.5567 | 0.5519 | 0.6091 | 0.6348 | 0.5627 |
| MIDC mean | 0.5508 | 0.5316 | 0.5855 | 0.5889 | 0.5513 (seed 42 only) |

- **Versus MIDC:** **AP7-9 +0.023 (t=8.2, p<0.001, survives Holm)**, AP best +0.023 (p<0.001), pixel PR-AUC +0.040 (p=0.010), F1best +0.010 (p=0.093), fixed-threshold F1 ns.
- **Versus `mid_groupdro_anchor`:** AP7-9 **+0.020 (p=0.003)**. EMA carries the gain given GroupDRO+anchor.
- **Versus c29 `model_ema`** (same decay, 0.995): AP7-9 **+0.021 (p=0.002)**. EMA alone does not produce it; `model_ema` minus MIDC is +0.001 AP7-9 (ns).
- **Linkage:** neither EMA alone nor GroupDRO+anchor alone raises AP7-9 by more than 0.005. Only EMA combined with GroupDRO(+anchor) does. That is an interaction: EMA appears to average out the oscillation GroupDRO's reweighting introduces. Anchor is probably not required (see section 3), but `groupdro_ema` without anchor has not been run.
- **Verdict:** **this is the only Campaign 31 mid-architecture result that has been replicated across seeds.** The two seeds agree within 0.002 AP. It improves ranking and pixel PR-AUC. The fixed-threshold F1 gain is not demonstrated.

### 2.4 `mid_domain_mean`

- **Versus MIDC:** all ns. F1best +0.009 (p=0.18), AP7-9 −0.001.
- **At epoch 3:** significantly *below* MIDC (F1 −0.017, p=0.024). This does not persist to the endpoint.
- **Verdict:** equal-domain loss weighting has no detectable endpoint effect.

### 2.5 `mid_pcgrad_gram_exact`

- **Versus MIDC:** all ns. F1best −0.007, AP7-9 −0.000.
- **Versus `mid_domain_mean`:** F1best −0.017 (p=0.076), AP ns.
- **Linkage:** a PCGrad benefit would require exact-gram > domain-mean. That is not observed; if anything it trends worse.
- **Versus full c29 PCGrad at epoch 3**, the only epoch full PCGrad reached:
  - Full PCGrad beats MIDC there: F1 over epochs 1-3 +0.011 (p=0.012), patch mean +0.027 (p=0.004), AP +0.027 (p=0.032).
  - Exact-gram is significantly *below* MIDC (F1 −0.025, p=0.005) and far below full PCGrad (F1 −0.037, AP −0.046, both p<0.01).
- **Verdict:** the gram re-weighting does not reproduce full PCGrad's early behavior, and it has no endpoint benefit. It costs 1.6x (311 s/epoch). Reject. Full PCGrad's epoch-3 advantage is real at early-epoch noise levels, but there is no evidence it would survive to epoch 9.

### 2.6 `mid_pcgrad_gram_sparse4` (incomplete, 9 epochs)

Compared at epoch 8, against MIDC, domain-mean, and exact: every metric ns (|t| < 0.6). No conclusion.

### 2.7 Depth and overlap decomposition: `mid_depth12`, `mid_overlap8`, `mid_overlap12`

| arm | slices | windows | F1best | AP7-9 | Δ F1best vs MIDC | Δ AP7-9 vs MIDC |
|---|---:|---|---:|---:|---:|---:|
| MIDC | 8 | no | 0.5508 | 0.5855 | — | — |
| `mid_depth12` | 12 | no | 0.5356 | 0.5666 | −0.015 (p=0.054) | **−0.019 (p=0.003)** |
| `mid_overlap8` | 8 | yes | 0.5338 | 0.5787 | **−0.017 (p=0.038)** | −0.007 (p=0.11) |
| OV12 (c29 + c31) | 12 | yes | 0.5347 | 0.5748 | **−0.016 (p=0.020)** | **−0.011 (p=0.011)** |

Linkage logic: the overlapping-window architecture would be proven beneficial only if OV12 beat the control *and* also beat `mid_depth12`, showing the gain came from the windows rather than the extra slices. `mid_overlap8` should also beat the control.

- OV12 is significantly **worse** than the control, so the first condition fails.
- Extra slices without windows (`depth12`) cost AP.
- Windows without extra slices (`overlap8`) cost F1best.
- OV12 versus `depth12`: F1best tied, AP best +0.013 (p=0.013). Windows recover part of the AP that the 12-slice input lost, but not up to the control level.
- OV12 versus `overlap8`: ns.
- Only 1/17 patches improve for OV12.

Replication: the c29 overlap run's late collapse (epochs 7-9 F1 0.426) was **not** reproduced by the same-seed c31 run (0.522). That instability was noise.

**Verdict:** reject both the extra slices and the overlapping windows. They are slower too: 286, 274, and 448 s/epoch.

### 2.8 `early_overlap12`

- **Versus c30 `early_gated`:** AP7-9 **−0.032 (p=0.001, survives Holm)**, F1best **−0.020 (p=0.043)**.
- **Verdict:** harmful, and 2.3x slower (398 s/epoch). This agrees with the mid result. Caveat: new pretrain.

### 2.9 `mid_air_offset2` and `mid_air_offset3`

| arm | Δ F1best | Δ AP7-9 | Δ pixel PR-AUC |
|---|---:|---:|---:|
| offset 2 vs MIDC | **−0.037 (p=0.002)** | **−0.041 (p<0.001)** | −0.022 (ns) |
| offset 3 vs MIDC | **−0.051 (p<0.001)** | **−0.088 (p<0.001)** | **−0.085 (p=0.001)** |
| offset 3 vs offset 2 | −0.014 (ns) | **−0.047 (p<0.001)** | **−0.062 (p=0.010)** |

- **Linkage:** a clean dose-response. Each step toward the air side monotonically degrades ranking. All three AP contrasts survive Holm.
- **pherc0841**, which motivated the change, did not improve either. Its PR-AUC over the last three epochs is 0.36–0.38 for both offsets, versus 0.38–0.50 across the three controls.
- **Verdict:** the surface-centred window is correct. Reject the offsets.

### 2.10 Depth-latching arms

| arm | F1best | AP7-9 | Δ AP7-9 vs MIDC | key linkage |
|---|---:|---:|---:|---|
| `mid_depth_entropy_weak` | 0.5453 | 0.5818 | −0.004 (ns) | — |
| `mid_depth_entropy_strong` | 0.5552 | 0.5793 | −0.006 (p=0.135); AP best −0.009 (p=0.044) | strong vs weak: ns |
| `mid_depth_entropy_strong_topk` | 0.5418 | 0.5572 | **−0.028 (p<0.001, Holm)** | vs strong: **−0.022 (p=0.004)** |
| `mid_lse_capped` | 0.5530 | 0.5727 | **−0.013 (p=0.015)** | — |
| `mid_depth_entropy_overstrong_lse` | 0.5459 | 0.5774 | −0.008 (p=0.067); AP best −0.011 (p=0.022) | vs lse_capped: ns; vs strong: ns |

- The entropy floor is null at the weak setting and slightly negative for AP at the strong setting. There is no F1 effect at any strength.
- The strong-vs-strong_topk contrast isolates top-k collapse, and top-k is clearly harmful.
- Capping LSE costs AP.
- The overstrong bracket's deficit matches the LSE cap alone and adds nothing beyond the strong floor. Its AP loss is therefore attributable to the cap.
- **Verdict:** no depth-latching intervention improves any metric. Reject all of them, top-k and the LSE cap most clearly. The latching hypothesis is not supported as a limiting factor.

### 2.11 `mid_topk_bag_positive`

- **Versus MIDC:** F1best **−0.056 (p<0.001, survives Holm)**, AP7-9 **−0.035 (p<0.001, Holm)**, 0/17 patches up.
- The calibrated threshold is pinned at the grid minimum of 0.10 for the last three epochs, so the predictions are systematically under-confident. AP is threshold-free, though, and it is also clearly worse.
- **Verdict:** reject.

### 2.12 `mid_trusted_negative_mining`

- **Versus MIDC:** AP7-9 −0.009 (p=0.049), AP best −0.007 (p=0.088), F1 ns.
- **Verdict:** no benefit and a marginal ranking cost. Reject at this setting.

### 2.13 `early_deep_residual_groupdro_anchor`

- **Versus EDR (n=2):** AP7-9 **+0.010 (p=0.047)**, AP best +0.010 (p=0.036), F1 ns.
- **Versus MIDC:** AP7-9 **+0.013 (p=0.013)**, pixel PR-AUC +0.033 (p=0.046), F1best −0.007 (ns).
- **Versus `mid_groupdro_anchor`:** F1best **−0.019 (p=0.049)**, AP7-9 +0.011 (p=0.052).
- **Linkage:** GroupDRO(+anchor) lifts AP on this backbone by about the same amount it does elsewhere. The early-deep backbone swaps F1best for AP relative to mid, but EDR itself does not beat the control (section 2.17).
- **Verdict:** no reason to prefer this over the mid or early-gated alternatives.

### 2.14 `early_raw_instance_groupdro`

- **Versus c30 `early_raw_instance`:** AP7-9 **+0.028 (p=0.001)**, AP best +0.030 (p=0.001), F1best **+0.022 (p=0.030)**.
- Fixed-threshold F1 at epoch 9 is −0.063 (ns). That is a calibration collapse: threshold 0.12, while calibrated F1 is 0.552.
- **Verdict:** GroupDRO significantly improves this backbone on both ranking and best-epoch F1. The backbone itself, however, is significantly worse than `early_gated` on AP (section 2.19).

### 2.15 `early_raw_instance_groupdro_anchor`

- **Versus `early_raw_instance_groupdro`:** AP7-9 −0.010 (p=0.077), F1best ns. The +0.082 epoch-9 F1 (p=0.072) is again threshold drift.
- **Versus MIDC:** nothing significant; AP7-9 +0.007 (p=0.086).
- **Verdict:** anchor adds nothing detectable and trends toward lower AP.

### 2.16 `mid_wide15_groupdro`

- **Versus c30 `mid_wide15_gated`:** AP7-9 **+0.018 (p=0.008)**, AP best +0.024 (p=0.002), F1 ns.
- **Versus c29 `physical_groupdro`:** F1best identical (0.5508), AP7-9 +0.010 (p=0.059).
- **Width alone** (c30 `mid_wide15` versus MIDC): every metric ns.
- **Linkage:** GroupDRO helps the wide model. Width shows no effect either alone or given GroupDRO.
- **Verdict:** width is unproven. Do not pay for it.

### 2.17 Early-2D decomposition

| arm | block depth | extra level | residual | F1best | AP7-9 |
|---|---:|---|---|---:|---:|
| c30 `early_gated` | 2 | no | no | 0.5526 | 0.6070 |
| c30 `early_residual2d` | 2 | no | yes | 0.5375 | 0.5839 |
| `early_residual_depth3` | 3 | no | yes | 0.5150 | 0.5671 |
| `early_residual_extra320` | 2 | 320 | yes | 0.5311 | 0.5695 |
| `early_deep_nonresidual` | 3 | 320 | no | 0.5371 | 0.5756 |
| EDR (n=2) | 3 | 320 | yes | 0.5421 | 0.5888 |

Contrasts and verdicts:

- **Residual on the shallow model:** `early_residual2d` versus `early_gated` gives AP7-9 **−0.023 (p=0.003)** and F1best −0.015 (p=0.096). Residual blocks hurt.
- **Block depth alone:** `depth3` versus `early_residual2d` gives F1best **−0.023 (p=0.029)** and AP7-9 **−0.017 (p=0.011)**. Harmful.
- **Extra level alone:** `extra320` versus `early_residual2d` gives AP7-9 **−0.014 (p=0.020)** and F1best ns. Slightly harmful.
- **Both together:** EDR versus `early_residual2d` is ns on everything (F1best +0.005, AP7-9 +0.005). **Campaign 30's claim that "adding depth to early residual improves every aggregate" does not survive replication.**
- **Interaction:** EDR beats `depth3` (F1best +0.027, p=0.008; AP7-9 +0.022, p=0.002). The two components are each harmful alone and neutral together. Because `depth3` and `extra320` used new pretrains while EDR and `early_residual2d` did not, pretrain variance could explain part of this.
- **Residual given depth plus extra level:** `deep_nonresidual` versus EDR gives AP7-9 **−0.013 (p=0.016)** and F1 ns. Residual helps only once capacity has been added.
- **Against the simplest early model:** EDR versus `early_gated` gives AP7-9 **−0.018 (p=0.004)**, and `deep_nonresidual` versus `early_gated` gives AP7-9 **−0.031 (p=0.001, Holm)**.

**Verdict:** every capacity or residual addition to the early head is neutral or harmful relative to plain `early_gated`. The block-depth-4 arm is not justified.

### 2.18 `early_gated_patch_groupdro`

| | F1best | F1 7-9 | AP7-9 | pixel PR-AUC | calibrated F1 7-9 |
|---|---:|---:|---:|---:|---:|
| `early_gated_patch_groupdro` | **0.5760** | 0.5576 | **0.6268** | **0.6409** | **0.5695** |
| c30 `early_gated` | 0.5526 | 0.5320 | 0.6070 | 0.6256 | — |
| c29 `physical_patch_groupdro` (mid) | 0.5579 | 0.5417 | 0.6112 | 0.6266 | — |
| TRIPLE mean | 0.5607 | 0.5563 | 0.6081 | 0.6287 | 0.5656 |
| MIDC mean | 0.5508 | 0.5316 | 0.5855 | 0.5889 | — |

- **Versus MIDC:** F1best **+0.025 (p=0.009)**, AP7-9 **+0.041 (t=11.9, p<0.001, Holm)**, pixel PR-AUC **+0.052 (p=0.009)**. F1 7-9 +0.026 (ns).
- **Versus `early_gated`:** F1best **+0.023 (p=0.025)**, AP7-9 **+0.020 (p=0.006)**, 12/17 patches up over epochs 7-9.
- **Versus mid patch-GroupDRO:** AP7-9 **+0.016 (p=0.014)**, F1best +0.018 (p=0.058).
- **Versus TRIPLE:** AP7-9 **+0.019 (p=0.004)**, F1best +0.015 (p=0.062).
- **Linkage, 2x2 of backbone by patch-DRO:**
  - Early collapse helps given no DRO: `early_gated` minus MIDC is +0.022 AP7-9 (p=0.002).
  - Early collapse helps given patch-DRO: +0.016 (p=0.014).
  - Patch-DRO helps on mid: c29 patch-DRO minus MIDC is +0.026 (p=0.001).
  - Patch-DRO helps on early: +0.020 (p=0.006).

  Both main effects are significant in both strata, and they are roughly additive.
- **Verdict:** **the best arm in all three campaigns on every low-noise metric.** It is a single seed, but the effect is 4–12 σ and the 2x2 structure explains it. It needs a seed replicate before it is adopted.

### 2.19 Stem and normalization controls: `early_gated_instance` and `early_raw_ibn`

2x2 at epoch 9, AP7-9:

| | IBN | instance |
|---|---:|---:|
| gated stem | 0.6070 (c30) | 0.6021 |
| raw stem | *running* | 0.5740 (c30) |

- **Normalization given the gated stem:** `early_gated_instance` versus `early_gated` is ns on every metric.
- **Stem given instance norm:** `early_raw_instance` versus `early_gated_instance` gives AP7-9 **−0.028 (p=0.001)** and pixel PR-AUC −0.043 (p=0.040).
- **`early_raw_ibn`, preliminary at epoch 3** (epoch-3 σ is small):
  - versus `early_gated`: F1 **−0.026 (p=0.010)**
  - versus `early_raw_instance`: F1 **−0.021 (p=0.023)**
  - versus MIDC: F1 **−0.037 (p=0.001)**
- **Verdict:** the raw-only stem, not instance normalization, causes the loss. This matches c30 `mid_raw_only`. Instance versus IBN has no detectable effect. The `early_raw_instance` "floor" advantage was never significant (section 1).

### 2.20 Not yet run

`researcher_ds2_full`, `researcher_full`, and `mid_pcgrad_gram_groupdro` have no event files. `early_raw_ibn` is not complete (see 2.19).

## 3. Pooled factor effects

Each factor effect is estimated from several independent contrasts with the run weights combined.

| factor | contrasts | Δ F1best | Δ AP best | Δ AP7-9 | Δ F1 7-9 |
|---|---|---:|---:|---:|---:|
| Physical-domain GroupDRO | mid/MIDC, early_raw_instance, mid_wide15 | +0.008 (p=0.11) | **+0.023 (p<0.001)** | **+0.017 (p=0.001)** | +0.002 (ns) |
| Patch GroupDRO | mid/MIDC, early_gated | — | — | +0.026 (p=0.001) and +0.020 (p=0.006) separately | ns |
| Patch vs domain GroupDRO (mid) | c29 pair | +0.007 (ns) | **+0.019 (p=0.006)** | **+0.020 (p=0.005)** | ns |
| MAE anchor | mid/MIDC, given GroupDRO, given early_raw GroupDRO | +0.004 (ns) | **−0.006 (p=0.045)** | −0.004 (p=0.12) | ns |
| EMA | alone, given GroupDRO+anchor | −0.004 (ns) | +0.008 (p=0.020) | +0.011 (p=0.008) | ns (gain comes only from the GroupDRO context) |
| Early collapse (gated, IBN) | vs MIDC | +0.002 (ns) | **+0.023 (p=0.001)** | **+0.022 (p=0.002)** | ns |

## 4. Previous conclusions that no longer hold

- Patch "specialists", patch-winner tables, and "indicator patch" claims: inside per-patch noise.
- Endpoint-F1 leaderboards and the claimed physical GroupDRO endpoint superiority: inside noise. GroupDRO's real effect is on ranking.
- The "best lower tail" for `early_raw_instance` and `mae_anchor`: not significant. The raw stem significantly hurts AP.
- MAE anchor as a robustness mechanism: no effect, or a small AP cost.
- "Early deep residual repairs early residual": not replicated.
- "Width is a secondary candidate": no effect.
- The c29 `overlap_depth12` late instability: not reproduced; it was noise.

## 5. Recommendations

1. **Replicate `early_gated_patch_groupdro` with seed 42.** Then run `early_gated_patch_groupdro + EMA`. EMA's AP gain appeared specifically when combined with GroupDRO, so the combination is the natural next arm. Drop the anchor.
2. **Isolate EMA from anchor with `mid_groupdro_ema`.** If it matches TRIPLE, the anchor can be removed everywhere.
3. **Make model selection and reporting use low-noise quantities:** character AP averaged over epochs 7-9, best-epoch F1, and calibrated F1. Store the calibrated threshold with the checkpoint. The fixed 0.5 threshold is off by 0.2–0.3 on most late epochs.
4. **Replicate any arm whose headline effect is smaller than the detectable-difference table.** For fixed-threshold F1 or per-patch claims, use at least 3 seeds per arm.
5. **Stop these directions:** air offsets, overlap/depth-12, top-k collapse, the LSE cap, entropy floors, top-k bag positives, trusted-negative mining, PCGrad-gram, width, extra early-2D depth or levels or residuals, the raw-only stem, and MAE anchor.
6. **Researcher arms:** they bundle the raw-only stem, which is significantly harmful here, with residual extra levels, which are also not helpful. Keep them only as a bundled replication of external results; there is little reason to expect them to beat `early_gated`.
