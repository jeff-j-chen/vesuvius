# Campaign 29 and 30 TensorBoard analysis

Snapshot: 2026-09-22. Campaign 30 is complete through `early_raw_instance`; `researcher_like` has not started.

## Scope and method

- Read every scalar event in `runs_archs29`; excluded the `f1_bounds_0_to_1` utility run.
- Primary comparison: epoch 9 for completed ten-epoch runs. Best epoch and mean over epochs 7-9 are secondary checks against endpoint noise.
- `Character/F1Macro/Valid` is macro-F1 over connected characters. `Character/Patch/<id>/F1/Valid` applies the same character metric independently to each acquisition patch. The equal-patch mean below weights all 17 patches equally; global character F1 weights all characters equally, so the two need not agree.
- `Per_Scroll/PR_AUC_Valid/<domain>` pools predictions by the 12 physical domains. `pherc0139` contains three patches, `pherc0172` two, and `pherc1667` three; the other domains contain one patch each.
- There is one training seed per arm. Patch comparisons are paired observations, not independent training replicates. Conclusions about observed results are firm; claims about causal generalization require replication.

### Interpretation boundary

These campaigns measure held-out regions within known patches, not wholly unseen scrolls. That distinction matters for the eventual generalization study, but it does not prevent the present results from selecting mechanisms and architecture directions for the next campaign. Here, global F1/AP, equal-patch mean, lower-tail performance, breadth of patch gains, and late-epoch stability are considered together.

## Run inventory

- Completed standard 17-patch runs: 28.
- Completed fragment-only runs: 3.
- Partial: full `pcgrad` has epochs 0-3.
- No usable metrics: `cue_dropout` contains only `Run/Initializing` and `Run/Initialized`.
- No event directory: Campaign 29 `coordinate_hash_split`; Campaign 30 `researcher_like`.
- Campaign 29 `baseline` and `domain_cvar` used `compile_model=True`; the other analyzed runs used `False`. The baseline averaged 135.0 s/epoch versus 193.0 s/epoch for Campaign 30's uncompiled control.

## Overall completed-run leaders

| rank | arm | global F1 | character AP | equal-patch mean | worst patch | epochs 7-9 mean |
|---:|---|---:|---:|---:|---:|---:|
| 1 | `physical_groupdro` | 0.5508 | 0.6011 | 0.5533 | 0.3275 | 0.5490 |
| 2 | `mid_wide15_gated` | 0.5493 | 0.5862 | 0.5471 | 0.2535 | 0.5378 |
| 3 | `mae_anchor` | 0.5463 | 0.5904 | 0.5531 | **0.3934** | 0.5486 |
| 4 | `pcgrad_lite_head4` | 0.5447 | 0.5984 | 0.5448 | 0.2679 | 0.5350 |
| 5 | `model_ema` | 0.5424 | 0.5899 | 0.5397 | 0.2746 | 0.5431 |
| 6 | `current_mid_control` | 0.5424 | 0.5941 | **0.5596** | 0.3785 | 0.5439 |
| 7 | `mid_residual2d` | 0.5423 | 0.6032 | 0.5510 | 0.1432 | 0.4957 |
| 8 | `mid_pure_instance` | 0.5412 | 0.5846 | 0.5451 | 0.2023 | 0.5372 |
| 9 | `early_deep_residual2d` | 0.5380 | 0.5991 | 0.5305 | 0.3309 | 0.5322 |
| 10 | `early_raw_instance` | 0.5379 | 0.5708 | 0.5335 | **0.4236** | 0.5278 |

Metric-specific leaders:

- Character AP and pixel PR-AUC: `physical_patch_groupdro`, 0.6233 and 0.6266.
- Pixel F1 and balanced accuracy: `model_ema`, 0.5881 and 0.6925.
- Equal-patch mean: `current_mid_control`, 0.5596.
- Worst-patch F1: `early_raw_instance`, 0.4236.
- Late-epoch stability and global F1: `physical_groupdro`, 0.5490 and 0.5508.
- Success fraction is not aligned with overall quality: `early_wide15_gated` leads at 0.1758 despite being last in global F1 at 0.4401.

The AP/F1 divergence is important. `physical_patch_groupdro` and `early_gated` rank first and second in character AP but only 12th and 13th in fixed-threshold character F1. They learned useful ranking functions but are poorly calibrated at the fixed 0.5 threshold or have an unfavorable precision/recall operating point.

For across-the-board robustness, the lower tail is more informative than a single global average:

| arm | bottom-four patch mean | 10th percentile | worst patch | patches below 0.30 |
|---|---:|---:|---:|---:|
| `early_raw_instance` | **0.4434** | **0.4311** | **0.4236** | 0 |
| `mae_anchor` | 0.4341 | 0.4258 | 0.3934 | 0 |
| `current_mid_control` | 0.4125 | 0.3991 | 0.3785 | 0 |
| `physical_groupdro` | 0.4052 | 0.4255 | 0.3275 | 0 |
| `mid_wide15_gated` | 0.3973 | 0.4444 | 0.2535 | 1 |
| `gradient_conflict_blend` | 0.3754 | 0.3885 | 0.2996 | 1 |
| `pcgrad_lite_head4` | 0.3693 | 0.3962 | 0.2679 | 1 |
| `early_deep_residual2d` | 0.3585 | 0.3509 | 0.3309 | 0 |

`early_raw_instance` is the strongest lower-tail architecture, MAE anchor is the strongest lower-tail training mechanism, and physical GroupDRO is the strongest global/late mechanism. These are complementary candidates rather than interchangeable winners.

## Patch winners and specialization

`G` is global-F1 rank and `O` is rank by mean of the other 16 patches.

| patch | epoch-9 winner | F1 | G | O | interpretation |
|---|---|---:|---:|---:|---|
| 20230301213423 | `gradient_conflict` | 0.7648 | 24 | 25 | strong specialist |
| 20230301213755 | `gradient_conflict_blend` | 0.7243 | 10 | 8 | moderate specialist |
| 20231201215900 | `fixed_depth_8_16` | 0.6040 | 19 | 25 | strong specialist |
| 20231205222200 | `mid_pure_instance` | 0.7698 | 8 | 7 | specialist shift |
| 20231210121321 | `physical_groupdro` | 0.6986 | 1 | 3 | broad winner |
| 20240304141531 | `early_deep_residual2d` | 0.5414 | 9 | 12 | specialist |
| 20240304144031 | `physical_groupdro` | 0.5452 | 1 | 3 | broad winner |
| 20250223000000 | `physical_groupdro` | 0.4439 | 1 | 3 | broad winner |
| 20250511003658 | `mid_deep2d` | 0.7736 | 21 | 13 | strong specialist |
| 20250628074500 | `overlap_depth12` | 0.4634 | 14 | 15 | strong specialist |
| 20250919125754 | `early_raw_instance` | 0.4986 | 10 | 13 | specialist |
| 20251111010954 | `early_residual2d` | 0.6591 | 18 | 18 | strong specialist |
| 20251112000002 | `early_gated` | 0.7030 | 13 | 16 | strong specialist |
| 20260115000000 | `mid_pure_instance` | 0.6416 | 8 | 8 | specialist shift |
| 20260221022814 | `mid_deep_residual2d` | 0.5343 | 15 | 18 | strong specialist |
| 20260226000000 | `early_gated` | 0.7193 | 13 | 18 | strong specialist |
| 20260317000000 | `pcgrad_lite_head4` | 0.5473 | 4 | 8 | partly specialized |

Breadth by podium count: `physical_groupdro` won 3 patches and placed top-three on 4; `early_gated` and `mid_pure_instance` each had 2/3; `mid_deep2d` and `pcgrad_lite_head4` each had 1/3; `early_raw_instance` had 1/2. `mid_residual2d` won none but placed top-three on 4, while `mid_wide15_gated` had 0/3. Most patch winners were not top general models. Physical GroupDRO is the clear exception.

## Patch correlation

- Within an individual completed run, the median pairwise Pearson correlation of patch trajectories had a median of 0.407 across runs and ranged from 0.149 (`mid_raw_only`) to 0.743 (`model_ema`). Depending on the run, only 57-77% of patch pairs were positively correlated. Patches therefore share a learning trend, but substantial patch-specific movement remains.
- Full PCGrad's four early epochs have median Pearson 0.725 and Spearman 0.800, but four monotonic warm-up points are too few to compare fairly with ten-epoch runs.
- Across interventions at epoch 9, the median patch-patch correlation is only 0.175; 71.3% are positive. Architecture/objective changes redistribute performance materially rather than moving every patch together.
- Strong clusters: `20240304141531` with `20240304144031` ($r=0.772$), `20240304141531` with `20260317000000` ($r=0.757$), and `20250628074500` with `20260221022814` ($r=0.720$).
- Strong conflicts: `20230301213423` versus `20231201215900` ($r=-0.648$), `20240304141531` versus `20250511003658` ($r=-0.445$), `20250223000000` versus `20251112000002` ($r=-0.439$), `20230301213423` versus `20240304141531` ($r=-0.413$), and `20251111010954` versus `20260226000000` ($r=-0.410$).
- Best general-purpose indicator patches are `20240304141531` ($r=0.887$ with global F1), `20240304144031` (0.831), `20260115000000` (0.826), and `20260317000000` (0.778).
- Poor model-selection proxies are `20250511003658` ($r=-0.250$), `20230301213423` (-0.132), `20251112000002` (-0.092), and `20251111010954` (-0.049). Winning one of these says little about overall quality.

## Campaign 29 comparisons

All rows below compare epoch 9 with the Campaign 29 baseline. `improved` counts per-patch F1 gains.

| arm | delta global F1 | delta AP | delta patch mean | delta worst patch | improved | epochs 7-9 mean |
|---|---:|---:|---:|---:|---:|---:|
| `physical_groupdro` | +0.0669 | +0.0306 | +0.0570 | +0.2423 | 12/17 | 0.5490 |
| `mae_anchor` | +0.0625 | +0.0200 | +0.0568 | +0.3082 | 11/17 | 0.5486 |
| `model_ema` | +0.0586 | +0.0194 | +0.0435 | +0.1895 | 13/17 | 0.5431 |
| `gradient_conflict_blend` | +0.0521 | +0.0175 | +0.0460 | +0.2145 | 13/17 | 0.5323 |
| `physical_patch_groupdro` | +0.0342 | +0.0528 | +0.0402 | +0.1768 | 13/17 | 0.5417 |
| `depth_antialias` | +0.0351 | +0.0212 | +0.0251 | +0.1666 | 11/17 | 0.4988 |
| `explicit_depth` | +0.0272 | -0.0009 | +0.0106 | +0.2195 | 9/17 | 0.5239 |
| `domain_cvar` | +0.0262 | +0.0152 | +0.0046 | -0.0732 | 9/17 | 0.5225 |
| `fixed_depth_8_16` | +0.0177 | -0.0764 | +0.0039 | +0.2360 | 9/17 | 0.4737 |
| `depth_shift_aux` | +0.0163 | +0.0190 | +0.0261 | +0.0135 | 12/17 | 0.5142 |
| `multitile_8px` | +0.0112 | -0.0214 | -0.0061 | -0.0198 | 6/17 | 0.4939 |
| `domain_vrex` | +0.0075 | +0.0075 | +0.0015 | +0.1107 | 10/17 | 0.5166 |
| `gradient_conflict` | +0.0004 | +0.0313 | -0.0065 | -0.0539 | 6/17 | 0.5207 |

The old baseline ended unusually poorly: 0.4839 at epoch 9 after peaking at 0.5474 at epoch 5. Campaign 30's nominally equivalent control, apart from compilation and newly recorded no-op defaults, ended at 0.5424 and peaked at 0.5589. Their epoch-9 patch vectors correlate only $r=0.775$. Against the later control, physical GroupDRO is only +0.0083 global F1 and +0.0051 late mean, MAE anchor is +0.0039 and +0.0047, and EMA ties on endpoint and is -0.0008 on late mean. Therefore the large table deltas establish superiority over that particular baseline run, not effect sizes that should be expected to reproduce.

Mechanism conclusions:

- Physical GroupDRO has the best complete evidence: top global F1, top late mean, strong AP, three broad patch wins, and 10/17 patches above the later control. Its equal-patch mean and worst patch are below the later control, so it does not dominate every fairness criterion.
- MAE anchoring is the best worst-patch mechanism and is stable late. It plausibly reduces destructive drift from the pretrained representation.
- EMA's apparent large gain is mostly the weak old baseline. It nevertheless leads pixel F1 and balanced accuracy and produces the most synchronized patch trajectories, consistent with smoothing noisy late updates.
- Physical-patch GroupDRO is the best ranking model by character AP and pixel PR-AUC, but fixed-threshold F1 is mediocre. It is a calibration/checkpointing candidate, not the best final 0.5-threshold model.
- Half-strength conflict weighting is much better than full-strength conflict weighting. Full weighting peaks at 0.5640 at epoch 8 but collapses to 0.4843, while the 0.5 blend ends at 0.5360 with 13/17 gains over the old baseline. The unblended objective is too aggressive.
- V-REx and domain CVaR do not show broad gains. CVaR's worst patch collapses to 0.0119. Neither should be promoted at these settings.
- Explicit depth channels improve the old baseline's floor but not AP. Depth antialiasing has a decent endpoint but worse late mean and best epoch than baseline. Depth-shift auxiliary is neutral. These are secondary ideas, not winners.
- Twelve-slice overlap is not worth 447 s/epoch (2.32x the uncompiled control) and is unstable late. Eight-pixel targets and fixed absolute depth both hurt AP/stability. Reject these settings.

## Campaign 30 controlled comparisons

All rows compare with `current_mid_control` unless stated otherwise.

| comparison | delta global F1 | delta AP | delta patch mean | delta worst patch | improved |
|---|---:|---:|---:|---:|---:|
| `mid_pure_instance` | -0.0012 | -0.0096 | -0.0146 | -0.1762 | 8/17 |
| `mid_raw_only` | -0.0490 | -0.0400 | -0.0691 | -0.1798 | 6/17 |
| `early_raw_instance` | -0.0046 | -0.0233 | -0.0261 | +0.0451 | 5/17 |
| `pcgrad_lite_head4` | +0.0023 | +0.0042 | -0.0148 | -0.1106 | 11/17 |
| `mid_wide15_gated` | +0.0069 | -0.0080 | -0.0125 | -0.1250 | 10/17 |
| `mid_residual2d` | -0.0002 | +0.0090 | -0.0086 | -0.2353 | 9/17 |
| `early_gated` | -0.0262 | +0.0203 | -0.0364 | -0.1421 | 5/17 |
| `early_residual2d` | -0.0347 | -0.0059 | -0.0470 | -0.0799 | 5/17 |
| `early_deep_residual2d` | -0.0044 | +0.0050 | -0.0292 | -0.0476 | 7/17 |
| `mid_deep2d` | -0.0444 | +0.0007 | -0.0278 | -0.1291 | 7/17 |
| `mid_deep_residual2d` | -0.0299 | -0.0019 | -0.0424 | -0.2329 | 6/17 |
| `early_wide15_gated` | -0.1024 | -0.0034 | -0.1206 | -0.1393 | 2/17 |

- Pure instance normalization did **not** beat `ibn_full`. It ties global F1 within noise but loses AP, equal-patch mean, late mean, and almost half the worst-patch F1. Its wins on `20231205222200` and `20260115000000` are specialization. It also reused the IBN pretrain with non-strict architecture loading, so this is not a clean end-to-end normalization pretraining comparison.
- Removing the gated cue stem in `mid_raw_only` is clearly harmful: global F1 is -0.0490, AP -0.0400, patch mean -0.0691, worst patch -0.1798, and only 6/17 patches improve. The gated raw/surface cue combination is carrying transferable signal within the known patches.
- Mid 1.5x width has the best architecture endpoint and five top-three patch finishes, but loses AP, equal-patch mean, worst patch, late mean, and best-observed F1. This is a promising endpoint fluctuation, not proof that width helps.
- Early collapse raises AP but lowers fixed-threshold and equal-patch F1. Widening early collapse is clearly harmful. Residual blocks do not rescue shallow early collapse.
- Adding depth to the early residual model is the one internally positive architecture comparison: versus `early_residual2d`, F1 is +0.0302, AP +0.0109, patch mean +0.0178, worst patch +0.0323, and 10/17 patches improve. It still does not beat the mid control.
- Mid residual is globally tied and AP-positive but has a severe worst-patch and late-epoch regression. Deeper mid models are worse. `mid_residual2d` reused the baseline pretrain with non-strict loading, so a matched pretrain is required before dismissing it conclusively.
- `early_raw_instance` finishes essentially tied in global F1 but below the control in AP and patch mean. Its worst patch is +0.0451 better, its bottom-four mean is the best of all completed runs, and no patch falls below 0.30. Because `mid_raw_only` and `mid_pure_instance` are individually worse, this appears to be an early-collapse-specific interaction rather than evidence that raw-only stems or instance normalization are generally better.

### Why the early family still merits investigation

Four early arms beat the weak Campaign 29 baseline endpoint, but no completed early arm beats the matched Campaign 30 control on global F1 or equal-patch mean. The family nevertheless contains three useful signals:

1. `early_gated` has the second-best character AP (0.6145). Collapsing depth earlier may suppress acquisition-specific depth profiles and force the network to rank ink using more transferable 2D morphology. Its lower F1 implies loss of depth discrimination and/or calibration error, not an absence of useful signal.
2. `early_deep_residual2d` repairs much of the shallow early model's deficit and improves every reported aggregate over `early_residual2d`. Extra 2D capacity appears necessary after early depth collapse.
3. `early_raw_instance` has the best lower tail despite weaker AP. The combination may suppress patch-specific cue/intensity shortcuts while early collapse preserves enough morphology. Since the two changes were bundled, the interaction must be decomposed before adopting it.

The current deep comparison changes both block depth and the number of 2D resolution levels, so it does not prove that arbitrary additional depth will help. Do not jump directly to a much deeper head. Run a controlled decomposition with matched MAE pretraining:

- early residual, block depth 3, no extra level;
- early residual, block depth 2, one 320-channel extra level;
- early deep, non-residual, to isolate whether residual blocks are necessary;
- the existing block-depth-3 plus extra-level design as a replicate;
- only if the depth-only arm improves the same validation metrics, add block depth 4.

Also run the missing 2x2 controls around the robust early arm: early gated + IBN (existing), early gated + instance norm, early raw + IBN, and early raw + instance norm (existing). This identifies whether its floor comes from raw input, instance normalization, or their interaction.

The best early architecture should then be combined with physical GroupDRO and MAE anchor. Early-wide should not be continued: width without the deeper hierarchy was decisively harmful.

## Fragment-only controls

| arm | global F1 | AP | patch mean | worst patch |
|---|---:|---:|---:|---:|
| `fragments_only` | 0.4690 | 0.4964 | 0.4852 | 0.2156 |
| `fragments_gap1_dilated2` | 0.3060 | 0.3042 | 0.3081 | 0.0523 |
| `fragments_gap1_dilated2_replica` | 0.3215 | 0.3107 | 0.3200 | 0.0731 |

The dilated-label/gap recipe loses on all five patches versus `fragments_only`. The two dilated runs have nearly identical patch ordering ($r=0.998$); the replica is +0.0155 F1 and +0.0066 AP, improving four of five patches. This is reproducible evidence that the combined dilation/ring-gap recipe is harmful, but it does not identify which component causes the damage.

## Full PCGrad and the efficient replacement

Full PCGrad has only four epochs, but its matched early comparison is consistently positive:

| epoch | baseline F1 | PCGrad F1 | baseline AP | PCGrad AP | patch mean delta | patches improved |
|---:|---:|---:|---:|---:|---:|---:|
| 0 | 0.4577 | 0.4644 | 0.4203 | 0.4310 | +0.0057 | 12/17 |
| 1 | 0.4933 | 0.5036 | 0.4847 | 0.5037 | +0.0160 | 11/17 |
| 2 | 0.5139 | 0.5241 | 0.5182 | 0.5351 | +0.0151 | 11/17 |
| 3 | 0.5257 | 0.5378 | 0.5331 | 0.5567 | +0.0280 | 13/17 |

At epoch 3, the largest gains are on `20230301213423` (+0.1513), `20231205222200` (+0.1162), `20260226000000` (+0.0747), `20250511003658` (+0.0506), and `20231201215900` (+0.0445). The largest losses are `20250919125754` (-0.0571) and `20250628074500` (-0.0466). The result is broad across 10/12 physical domains, but there is no completed endpoint.

Cost is decisive. Full PCGrad averaged 1,586 s/epoch versus 139 s for the baseline over the same first four epochs: 11.38x. Against the uncompiled control it is still 8.22x. The implementation takes gradients for every represented domain over every trainable parameter, then performs pairwise projections; with up to 12 domains this requires many retained-graph reverse passes and $O(D^2P)$ projection work.

`pcgrad_lite_head4` costs only 1.04x the uncompiled control. Its epoch-3 patch-delta vector strongly resembles full PCGrad's ($r=0.888$, Spearman 0.841), so it captures the direction of redistribution. It does not preserve the magnitude: at epoch 9 it is only +0.0023 global F1 and +0.0042 AP, while equal-patch mean is -0.0148, worst patch -0.1106, and late mean -0.0089. Applying surgery only to final decoder/head parameters is too narrow.

Best path to retain the PCGrad effect:

1. Use physical-domain GroupDRO as the production default. It attacks the same domain imbalance with one weighted backward pass, costs 1.04x the uncompiled control, and has stronger completed evidence than PCGrad.
2. If gradient surgery remains desirable, test `pcgrad_lite_scope="all"` with two sampled domains first. That existing option should cost much less than all-domain PCGrad while allowing conflict correction in shared features. Four-domain/all-parameter is the next rung if memory and time permit.
3. A better targeted implementation would add a shared-late scope covering the mid-depth fusion and 2D encoder/decoder, not only `mid2d_dec1` and `mid2d_head`. Select two domains using persistent loss or conflict EMA rather than uniform random sampling, and apply surgery every second or fourth batch. These cost estimates are hypotheses and must be timed.
4. The existing 0.5 conflict-weighted objective is another practical approximation: it estimates domain conflict on a tiny prediction-head probe and performs one ordinary backward. It costs about 1.08x the uncompiled control and is much more stable than full-strength conflict weighting, although it does not beat the later control.

## Why known patches fail

The results support a domain-shift diagnosis more strongly than a simple capacity diagnosis:

- Patch responses are weakly correlated across interventions and sometimes anti-correlated. Models are learning features useful for particular acquisition/material conditions rather than one invariant ink rule.
- Physical GroupDRO helps because ordinary empirical risk permits easy or heavily sampled domains to dominate. This is especially relevant because `pherc0139` receives sampling weight 4 while every other physical domain receives weight 1.
- MAE anchor helps the lower tail because fine-tuning otherwise overwrites broad self-supervised features with shortcuts specific to the labeled patches.
- EMA helps pixel metrics and synchronizes patch trajectories because late stochastic updates move toward different domain optima; averaging suppresses those excursions.
- Full PCGrad's broad early gain shows genuine gradient conflict between domains. Its largest gains and losses occur on different patches, which is exactly what conflicting domain objectives predict.
- Early collapse's AP gain suggests that some late 3D processing overfits patch-specific depth/intensity signatures. Its F1 loss shows that depth also contains real discriminative evidence; early collapse needs enough 2D capacity and calibration to compensate.
- Raw-only failure shows that removing the gated cue pathway discards useful invariants rather than merely removing a shortcut.
- The active augmentations are rotation, flip, context replacement, and cutout. Acquisition-shift augmentations already implemented in the loader (`fda`, depth warp, surface attenuation, acquisition blur, correlated noise, brightness, contrast) were all disabled. The training distribution therefore does little to simulate a new scanner/reconstruction/material appearance.

These explanations are mechanistic hypotheses consistent with the observed metrics. The next ablations should test the interactions directly.

## Immediate recommendation

### 1. Combine the broad mechanisms first

Use the current mid gated architecture and run:

1. physical GroupDRO + MAE anchor;
2. physical GroupDRO + EMA;
3. MAE anchor + EMA;
4. physical GroupDRO + MAE anchor + EMA.

This is the strongest path because the components address different failure modes:

- GroupDRO prevents persistently hard domains from disappearing inside the mean loss;
- MAE anchor preserves broad features instead of allowing labeled-domain shortcuts to overwrite them;
- EMA reduces late movement among competing domain optima at almost no compute cost.

The triple is the leading candidate, but the partial factorial is necessary: anchoring can impede the adaptation GroupDRO requests, and EMA can hide rather than fix unstable optimization. The existing single-factor runs provide the controls. Retain a combination only if it improves global F1/AP and the lower tail without creating a new catastrophic patch.

### 3. Architecture investigation

Continue the early family, but make it controlled. Decompose block depth from the extra resolution level and raw input from normalization as described above, using matched pretraining. Then test:

- best early-deep variant + physical GroupDRO;
- best early-deep variant + physical GroupDRO + MAE anchor;
- EMA on the winning combination.

An even deeper 2D head is worth one controlled test, but only in the early-collapse family. The existing result says “the combined deeper/hierarchical early head is better than shallow early residual,” not “more depth is monotonically better.” The deeper mid models regressed, so a generic depth increase is not supported.

`mid_wide15_gated` is a secondary architecture candidate. Its endpoint and 10/17 patch improvements are encouraging, but its lower AP, lower patch mean, lower worst patch, and lower late mean make it weaker than early-deep for a generalization-focused program.

### 3. Pursue the strongest specialist branches

- **AP/ranking:** `physical_patch_groupdro` and `early_gated` are the strongest candidates. Test each with EMA and evaluate an AP-selected checkpoint with one globally calibrated threshold. A combined `early_gated + physical_patch_groupdro` arm is reasonable after those controls; additivity is not guaranteed because both may emphasize hard/rankable examples.
- **Patch floor:** test `early_raw_instance + physical_groupdro`, then add MAE anchor if the floor is retained. This combines the best architecture floor with the best broad objective.
- **Capacity:** test `mid_wide15_gated + physical_groupdro`. Width alone has the second-best endpoint but weak lower-tail evidence; GroupDRO is the most plausible correction.
- **Gradient conflict:** full PCGrad remains scientifically strong but operationally unacceptable. Keep the cheaper broader-scope PCGrad variants as a separate branch, not in the first combination matrix.

### 4. Test acquisition-shift augmentation separately

After establishing the GroupDRO/anchor baseline, test one augmentation family at a time:

- mild acquisition blur plus correlated noise;
- brightness/contrast plus FDA-style low-frequency amplitude exchange;
- depth warp plus surface attenuation.

These perturb the scanner/reconstruction and surface-depth cues most likely to change on a new scroll. Tune them conservatively; combining all of them immediately would make a failure uninterpretable. Context replacement and cutout should remain as the baseline because they already discourage local-context memorization.

### 5. Preserve the PCGrad signal economically

Keep physical GroupDRO as the production objective. In parallel, test all-parameter PCGrad-lite with two selected domains and surgery every fourth batch, then a shared-late scope if needed. Select domains by persistent high loss or negative gradient cosine rather than uniformly. Do not combine GroupDRO and PCGrad initially: they optimize the same domain conflict through different mechanisms and are currently mutually exclusive.

### 6. Push every metric coherently

- Compare the best character-AP checkpoint as well as the final checkpoint; several arms reversed late.
- Calibrate one global validation threshold to determine whether AP gains from early models and patch GroupDRO convert into F1 gains.
- Track the Pareto frontier of global F1/AP, equal-patch mean, bottom-four mean, worst patch, and dispersion. Reject runs that improve one headline metric by creating a catastrophic patch.
- Preserve final and EMA checkpoints and report epochs 7-9 averages. Several arms showed large endpoint reversals.

## Final priority order

1. Run the four mid-architecture GroupDRO/anchor/EMA combinations.
2. Run `early_deep_residual2d + physical_groupdro + mae_anchor` and `early_raw_instance + physical_groupdro`.
3. Decompose the early-deep and early-raw interactions; include one early block-depth-4 arm only after the depth-3 isolated arm.
4. Run the AP/calibration branch with patch GroupDRO and early gated.
5. Test `mid_wide15_gated + physical_groupdro`.
6. Explore sparse, broader-scope PCGrad and acquisition-shift augmentation as parallel branches.

The strongest immediate combination candidate is physical GroupDRO + MAE anchor + EMA on the current mid gated architecture. The two strongest architecture candidates are early deep residual for balanced F1/AP and early raw instance for patch-floor robustness. A deeper 2D head is worth testing specifically after early collapse, but the experiment must isolate depth from the extra scale; the completed deeper-mid results argue against simply making every 2D head deeper.