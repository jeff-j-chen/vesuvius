# PHerc0846A Fine-Tune Diagnosis

Evidence collected on 2026-10-12 from the current code, real datasets, campaign checkpoints,
and TensorBoard events in `runs_finetune_letters/`. Historical claims in `KNOWLEDGE.md`,
`share.md`, and `README.md` were treated as experiment history, not as the current configuration.

## Findings

### Completed six-patch MAE continuation

`mae_pretrain_letters.py` completed 2,000 adaptation steps from the matched campaign-40 MAE,
not from an ink checkpoint and not from scratch. It uses the four main notebook patches
(ABF, 065803, 073244, 081237) plus 20260927143929299 and 20260927154033403. The offset sheet
073249 is not one of these six. All six are sampled equally, without ink labels.

Output: `models/mae_0846a_six_flat13_m1_2k.pth`, with a `.complete.json` provenance sidecar.
Architecture: full IBN, early-2D U-Net, gated cues, fiber-coordinate branch, native 96x96x8,
surface-anchor normalization at flat depth 13, fixed slices 9-16. Batch 24, LR 1e-4, 65%
spatial masking; all backbone parameters train. The literal-surface adapter is not trained
by the existing MAE reconstruction path, and is initialized in subsequent supervised training.
The reconstruction head is discarded; this backbone is not an ink-inference checkpoint.

Private aliases, masks and normalization caches are under `_ves_tmp/mae_0846a/`; originals
and shared caches were not overwritten. All 188 backbone keys warm-started with zero
mismatches; encoder, gated and fiber branches updated, all exported tensors are finite,
and the original MAE checksum is unchanged. GPU training took about 144 seconds. Final
masked reconstruction MSE: train 0.01397, monitor 0.01046. The existing block-based monitor
is reconstruction instrumentation, not independent ink validation or proof of generalization.

**Provenance correction:** the original MAE completion record already lists PHerc0846A
patch 20260921094413. These six patches are newly included, but this is not the first
exposure to the physical scroll. Missing MAE exposure is therefore not an established
explanation for the ink-transfer failure.

This is an all-six discovery initialization. Do not describe a later leave-one-out run using
it as completely unexposed to the held-out patch. If submission rules exclude any training
on the target, its MAE inputs and physically overlapping renderings must be excluded too.
No full supervised retraining was launched, and the inference notebook was not changed.

### Updated-label refinement and inference notebook

The user intentionally revised the ink labels after the focused comparison. Do not expect
exact metric reproduction across label versions. The new default queue starts every arm
from the same campaign-41 checkpoint and uses the updated labels:

- `refine_baseline_m1`: frozen encoder, no PU or far-background sampling, traditional
  close 2 / gap 2 / shell 4, positive-only windows, 5:1 domains, six epochs, slices 9-16.
- `refine_depth_m2`: identical, except slices 8-15 on the four PHerc0846A training patches.
- `refine_encoder_3e6`: identical to the baseline, but previously frozen encoder/cue-stem
  parameters train at 3e-6; existing adapters, decoder and heads stay at 1e-4.

Started in tmux `tra`, logging to `_ves_tmp/letters_refine.log`; new label/split hashes are in
`_ves_tmp/letters_refine_labels.sha256`. The baseline completed and wrote
`models/finetune_letters/refine_baseline_m1/final.pth`. The other controls were still running
when the notebook was validated. Only the two requested patches render during training.

`0846A_eval.ipynb` is now pure forward inference: settings at the top put `DEPTH_OFFSET = -1`
immediately below the checkpoint declaration. It defaults to the completed new baseline.
It uses flat depth 13, the selected eight input slices for its composite, saved model geometry,
and training-matched normalization. No direction variants, surface detection, or label edits.

After setup, four cells render the original main PHerc0846A entries (ABF, 065803, 073244,
081237), two cells render `20260927143929299` and `20260927154033403`, and the final cell
renders all remaining PHerc0846A quick patches into one `quick_patches` subfolder. There
are currently 32 remaining meshes after excluding the three quick-patch meshes represented
individually. The notebook streams zarr reads and caches extra flat-surface normalization
under `_ves_tmp/0846A_inference`, without changing shared surface maps or normalization files.

Validation: eleven campaign regression tests, three-arm preflight, notebook schema/syntax,
all 38 input paths, six-plus-32 unique render routing, strict checkpoint loading, exact
main-patch normalization reuse, and a real quick-patch crop through prediction and PNG output.
The full 38-segment notebook inference was not executed during training.

### Current recommendation after visual review

The user's visual review supersedes the initial metric-led recommendation below. Fully
unfrozen and long runs overfit visually; frozen/no-PU is the preferred family. Seven focused
six-epoch runs completed successfully, rendering only 20261012073246 and 20261012065806.
SHA256 checks confirmed all five training inklabel files and all five train masks were
unchanged through the comparison (`_ves_tmp/letters_focus_labels.sha256`).

| Focus arm | 0846A training AP | Recall@1% FPR, 073246 | Recall@1% FPR, 065806 | User visual assessment |
|---|---:|---:|---:|---|
| Control | 0.8375 | 0.0773 | 0.0432 | Reference |
| Loose ring | 0.8769 | 0.0702 | 0.0553 | No meaningful difference |
| Ratio 1:1 | 0.7731 | 0.0595 | 0.0553 | More noise and darker |
| Depth -1 | 0.8307 | 0.0559 | 0.0553 | Substantially better |
| Depth +1 | 0.8454 | 0.0927 | 0.0449 | Substantially worse |
| Encoder LR 1e-6 | 0.8581 | 0.0749 | 0.0501 | Almost unchanged |
| Encoder LR 1e-5 | 0.9625 | 0.1177 | 0.1054 | Strongly overfit |

The metrics do not reproduce the visual depth ranking. Better agreement with assumed labels
is not established evidence of better ink discovery. Ring changes additionally change the
scored training distribution. PHerc0009B training AP rises from 0.7827 to 0.8779 in the 1:1
arm while PHerc0846A AP falls: this supports a domain tradeoff, not a direct measurement of
visual noise. The 1:1 arm also gives PHerc0846A 40% fewer windows at equal total epoch length,
so it cannot separate harmful domain interference from insufficient adaptation by itself.

For future fine-tune experiments use the traditional **close 2 / gap 2 / shell 4 with
multitile_pos_only=True**, retaining soft edges, for consistency with the main campaign.
Keep completed arm definitions unchanged for reproducibility. The proposed next reference is
frozen/no-PU, 5:1, six epochs, and fixed slices **9-16** (offset -1). This combines the neutral
loose-ring change with the visually preferred depth, and needs one matched confirmation run.
Compare that reference only against fixed slices **8-15** (offset -2). Do not add random jitter,
longer training, further encoder LR sweeps, or more PHerc0009B in that comparison.

Keep labels fixed through that two-arm comparison. Afterwards, manually correct unambiguous
errors supported by the source scan and preserve the current label version; model disagreement
alone is not evidence that a correction is valid. Both rendered patches are training patches:
these comparisons are not independent held-out validation or confirmation of recovered text.

The slow-encoder runs used an explicit `encoder_prefixes=FREEZE_PREFIXES` optimizer group:
all 4,679,702 previously frozen parameters received 1e-6 or 1e-5, and the existing trainable
adapters/decoder/heads stayed at 1e-4. Defaults for other campaigns remain unchanged.

Reproducible comparison artifacts: `output/letters_focus/metrics.json` and
`output/letters_focus/{20261012073246,20261012065806}_comparison.png`, generated by
`python3 old/diagnose_finetune_letters.py --compare-focus`. Ten regression tests passed.
The seven completed `focus_*` experiments remain available for explicit selection. The newer
updated-label refinement queue above is now the default.

### Historical metric-led recommendation (superseded)

Three full six-epoch runs completed, each from the same original campaign-41 checkpoint.
The single-factor controls identify permanent encoder freezing as the principal fitting
bottleneck against the supplied labels at this training budget. Keeping PU improved the
label-based full-extent scores, but the subsequent visual assessment rejected this as the
preferred ink detector. These historical metrics are retained as evidence of the mismatch.

| Configuration | PHerc0846A Train AP | PHerc0009B Train AP |
|---|---:|---:|
| Frozen + PU, existing ring baseline | 0.7541 | 0.8319 |
| Unfrozen + PU, `diag_unfreeze` | 0.9772 | 0.8964 |
| Frozen, no PU, `diag_no_pu` | 0.8376 | 0.7827 |
| Unfrozen, no PU, `diag_unfreeze_no_pu` | 0.9995 | 0.9772 |

Full-extent recall at 1% FPR on the same patches and labels:

| Patch | Frozen + PU | Unfrozen + PU | Frozen, No PU | Unfrozen, No PU |
|---|---:|---:|---:|---:|
| ABF | 0.0229 | **0.4799** | 0.0268 | 0.2008 |
| 065803 | 0.0639 | **0.5492** | 0.0432 | 0.1744 |
| 073244 | 0.1570 | **0.4673** | 0.0773 | 0.3627 |
| Offset sheet | 0.0513 | **0.3864** | 0.0178 | 0.1969 |

Unfreezing alone preserves the supervised sampling, ring, depth, losses, LR, and checkpoint.
Its AP curve was 0.8121, 0.9234, 0.9489, 0.9641, 0.9731, 0.9772. The large full-extent gains
cannot be explained just by a monotonic brightness shift. This is still one seed, and the
full-extent metrics are relative to incomplete labels, not independently verified new text.
These results do not establish a recovered passage on the unlabeled held-out patch.

Historical metric-winning checkpoint: `models/finetune_letters/diag_unfreeze/final.pth`.
Its six final figures are in `runs_finetune_letters/ftl_diag_unfreeze_12_06-03-31/eval_figs/`.
The held-out figure is `eval_s20261012081238_ep06.png`. The no-PU-only control is recorded
under `runs_finetune_letters/ftl_diag_no_pu_12_06-11-15/`.

### Combined adaptation result

`diag_unfreeze_no_pu` completed from the original campaign-41 checkpoint with all other
ring-baseline settings retained. It did not start from the fixed-batch probe weights.
PHerc0846A training AP by epoch was **0.8735, 0.9749, 0.9944, 0.9982, 0.9992, 0.9995**.
PHerc0009B finished at 0.9772; raw supervised loss fell to 0.0872, versus 0.6194 in B.

Full-extent recall at 1% FPR, computed by the same figure path on the same labels:

| Patch | Ring Baseline B | Unfrozen, No PU |
|---|---:|---:|
| ABF | 0.0229 | 0.2008 |
| 065803 | 0.0639 | 0.1744 |
| 073244 | 0.1570 | 0.3627 |
| Offset sheet | 0.0513 | 0.1969 |
| PHerc0009B train region | 0.1158 | 0.1753 |

The inability to fit the labeled training cells is resolved in this experiment. Full-extent
selectivity improves too, but remains far below training-cell AP. The held-out patch still
contains many unverified detections; neither this result nor its image proves new readable text.
The single-factor controls above provide the attribution missing from this combination.

Artifacts: `models/finetune_letters/diag_unfreeze_no_pu/final.pth` and
`runs_finetune_letters/ftl_diag_unfreeze_no_pu_12_05-53-42/eval_figs/` (six final PNGs,
including `eval_s20261012081238_ep06.png` for the never-trained patch).

### The model can fit these inputs

`old/diagnose_finetune_letters.py --steps 100` selected 32 real PHerc0846A windows
(four ink-bearing and four ring-bearing windows per patch, 127 supervised positive cells).
Both trials started from the same campaign-41 checkpoint, used the actual normalized crops,
literal surface inputs, soft targets, masked BCE, AdamW LR 1e-4, and gradient clipping 0.5.
The probe used FP32 and omitted PU and bag ranking; it did not use PHerc0009B.

| Encoder | Step 0 train AP | Step 50 train AP | Step 100 train/eval AP | Step 100 BCE |
|---|---:|---:|---:|---:|
| Frozen, current recipe | 0.6905 | 0.9418 | 1.0000 / 1.0000 | 0.03246 |
| Unfrozen | 0.6905 | 1.0000 | 1.0000 / 1.0000 | 0.02722 |

This rules out a blanket inability to fit the supplied input/target pairs. It does **not**
prove that these features are ink, that all labels are correct, or that memorization transfers.
The initial eval AP was 0.6214 versus train AP 0.6905; the difference disappeared on this
repeated batch. BatchNorm running statistics can matter even with frozen parameters.

The checkpoint has 7,522,842 parameters; the old freeze removes 4,679,702 from optimization,
leaving 2,843,140 trainable. This is an early-3D-to-2D U-Net, not the older mid-collapse
architecture: raw/LCN/depth-gradient gated cues enter `enc1`, learned depth attention and
depth-max features collapse into 2D, then the 2D U-Net encodes and decodes. The existing
freeze includes the cue stem and the entire 2D encoder. Full unfreezing is still fine-tuning,
not a fresh MAE or a from-scratch training run.

### The ordinary campaign is still improving at six epochs

PHerc0846A online training PR-AUC from the recorded runs:

| Run | Epoch 1 | Epoch 6 |
|---|---:|---:|
| Flat forward, no augmentation, three patches | 0.7281 | 0.8549 |
| Flat backward, no augmentation, three patches | 0.7224 | 0.8495 |
| Detected surface, forward, three patches | 0.7002 | 0.8604 |
| Flat forward plus offset sheet | 0.6698 | 0.7961 |
| Same, close 0 / gap 1 / shell 4, mixed windows | 0.6373 | 0.7541 |

These AP values are not directly comparable when ring geometry, positive-only masking, or
the patch set changes. They change which examples are scored. Within each run, the continuing
rise is evidence against a demonstrated plateau. Adding the offset sheet changes both the
task and each patch's sampling share; the drop alone does not prove it is bad data.

In the latest ring run, full-extent training recall at 1% FPR was 0.0229 (ABF),
0.0639 (065803), 0.1570 (073244), and 0.0513 (offset sheet). Ring-local AP therefore
overstates full-patch selectivity. Compare final full-extent figures and these fixed-FPR
metrics, not just map brightness or aggregate training AP. Training AP is accumulated
while weights change during the epoch, not a final-checkpoint rescore.

### Augmentation off does not mean only fitting the labels

The latest ring run ended with reported raw supervised loss 0.6194, weighted bag loss
0.1020 (0.5099 times 0.2), and PU loss 0.1941: total 0.9156. PU is a substantial active
objective, although scalar loss magnitudes do not measure relative gradient norms.

Far-background windows are unlabeled, but they still receive gradient through
`softplus(U).mean() - 0.2 * softplus(P).mean()` and the negative-risk correction.
The inherited prior 0.2 was not measured on this scroll. Annotated letters and far-background
windows also need not have the same positive distribution. This motivated the no-PU controls;
their actual results show that keeping PU gives better full-extent selectivity after unfreezing.

**Disable both `tra.pu_lambda` and `data.far_negative_share` for the no-PU control.**
Turning off only the loss would make the far cells ordinary BCE negatives. No diagnostic
introduces dense background-negative supervision or generates additional ink labels.

### Five to one already strongly favors PHerc0846A

The effective domain weights are exactly `[5, 1]`. The sampler draws five PHerc0846A
windows per PHerc0009B window, rotating equally among the four PHerc0846A patches.
Each new patch gets approximately 20.83% of all windows; PHerc0009B gets 16.67%.
The greater number of PHerc0009B letters does not change that ratio.

On 128 sampled windows per patch, mean supervised cells/window were 8.69, 7.88,
8.70, 8.18 for PHerc0846A and 10.06 for PHerc0009B. Combining those with the schedule
gives about 81% of supervised cells to PHerc0846A. Loss is cell-averaged, so window
weights are not exact gradient shares. PU has its own independently averaged reduction.

Only 2, 5, 6, and 5 connected components were sampleable on the new patches, versus
24 on PHerc0009B. These are connected label components, not necessarily individual
Greek letters. One of six components on 065803 was not sampleable. Component sampling
balances these objects rather than every labeled cell or every visible character.

Actual child epoch lengths were 2,147 + 2,473 + 3,778 + 4,081 + 6,667 = 19,146 windows.
The configured cap of 6,667 is **per child**, not a total or a guarantee. Consequently,
removing PHerc0009B also shortens an epoch to 12,479 windows. The pure-scroll arm is a
negative-transfer control, not an exactly update-matched comparison. A gain despite fewer
updates is informative; a null result should be followed by matching optimizer steps.

### Loading and labels passed the sampled integrity checks

- Direct raw-zarr reads followed by the production normalizer matched the loader's crops.
- All sampled PHerc0846A windows used exactly slices 10-17. Surface maps were local depth 3.
- Mean clipped fractions were below 0.4% at either end for every patch. No sampled input
  was constant or non-finite. This does not exclude subtle normalization effects.
- Median positive targets were 1.0 on every patch; minimum sampled soft positives were
  approximately 0.80-0.84. Soft edges are not capping most labels at 0.55.
- PHerc0846A had zero validation windows. PHerc0009B retained 6,679 validation windows,
  and its validation loss drives ReduceLROnPlateau. Long controls set patience to 18 to
  preserve the intended LR for their 18 epochs. Existing six-epoch runs stayed at 1e-4.
- The active input is native 96x96x8. Its central 64x64 prediction area contains sixteen
  16x16 LSE bags, not 256 four-pixel targets. A cell-level fit does not require the model
  to reconstruct all labeled pixels. The finer sparse-target control keeps the same center.

These checks validate sample integrity, not the anatomical surface identity or semantic
truth of the bright strokes. Closely packed sheets can still place distinct evidence inside
one eight-slice window. No claim of a solved ink-detection problem follows from this audit.

## Diagnostic Matrix

All new arms start from the existing campaign-41 checkpoint by default, retain the current
ring baseline, soft positives, no augmentation, fixed surface, six epochs and 5:1 weights
unless listed otherwise. The vis-only 081237 patch is never trained on.

Use the existing `flat_10_18_0009b_noaug_forward_offsetextra_ring` as baseline B.
Use `diag_unfreeze` (PU retained) as adaptation control A. The remaining, unrun follow-ups
were based on A after the single-factor results; completed arm definitions were not changed.

| Arm | Compare With | Question |
|---|---|---|
| `diag_unfreeze` | B | Does a permanently frozen feature extractor limit fitting? |
| `diag_no_pu` | B | Does far-background PU interfere with fitting the new signature? |
| `diag_unfreeze_no_pu` | B and both above | Freeze-by-PU interaction; not a one-factor attribution |
| `diag_long` | B | Does 18-epoch constant-LR decoder fitting continue improving? |
| `diag_unfreeze_long` | A | Does full adaptation need more updates? |
| `diag_stem` | B | Is adapting just the cue stem and first 3D encoder sufficient? |
| `diag_no_bag` | A | Does the top-half character ranking objective interfere? |
| `diag_weight20` | A | Is residual PHerc0009B gradient competition important? |
| `diag_0846_only` | A | Remove PHerc0009B training entirely; still render it; shorter epochs |
| `diag_depth_m2` | A | Read slices 8-15 on PHerc0846A only, including vis-only patch |
| `diag_depth_p2` | A | Read slices 12-19 on PHerc0846A only, including vis-only patch |
| `diag_tiles8` | A | 64 sparse 8x8 targets in the same 64x64 center; no dense output; inherited PU prior stays 0.2 |
| `diag_loose_ring` | A | Restore close 2 / gap 2 / shell 4 and positive-only windows after adaptation |

Depth shifts preserve the literal flat surface at absolute depth 13; its input-local depth
changes with the crop. PHerc0009B's window is untouched. The common 10-17 raw composite
remains a fixed visual reference even in depth-shift arms; it is not their exact input stack.

The prior code did not forward configured surface-window offsets into the ordinary training
and test-fragment figure calls. Those calls now forward them. A synthetic depth ramp verifies
that training and prediction select starts 8, 10, and 12 and matching local surface depths.
This defect cannot explain the old zero-offset runs.

Missing scroll-specific MAE was not a prerequisite for these gains: supervised unfreezing
adapted the existing representation successfully. No new architecture,
pseudo-labeling, stronger photometric augmentation, dense supervision, or weight decay is
needed to run this diagnosis.

## Run and Interpret

```bash
python3 -m unittest old.test_campaign_finetune_letters -v
python3 old/diagnose_finetune_letters.py --steps 100
python3 campaign_finetune_letters.py --dry-run
python3 campaign_finetune_letters.py --only diag_unfreeze,diag_no_pu,diag_unfreeze_no_pu
python3 campaign_finetune_letters.py --from diag_long
python3 campaign_finetune_letters.py --only diag_unfreeze_long,diag_stem,diag_depth_m2,diag_depth_p2,diag_loose_ring
```

The campaign queues all active arms by default. `--only` and `--from` are mutually exclusive.
`--smoke` uses separate model/log directories and renders only the first trained patch.
Each arm writes `letters_experiment.json` beside its checkpoint, including dynamic freeze
and train-only fields omitted by the shared dataclass serializer. Keep dictionary order when
pairing `train_scroll_dict` and `train_scroll_weights`.

If training fit improves but the held-out patch does not, the next problem is transfer, not
capacity. If only longer training helps, the six-epoch stopping point was premature. If no-PU
helps, confirm with the frozen/no-PU control before blaming encoder freezing. If removing
PHerc0009B helps, the similar-looking scroll is not necessarily a compatible gradient source.
If a depth arm helps, inspect its actual input slices before inferring a surface direction.
Map brightness alone is insufficient; inspect new strokes at comparable false-positive levels.

## Verification

- Eight focused regression tests passed (matrix, sampler, optimizer, checkpoint shapes,
  effective manifest fields, train/inference depth agreement, actual figure callers,
  and empty validation).
- Real-data fixed-batch probe passed, with the measured results above.
- End-to-end `diag_unfreeze_no_pu --smoke` completed: all 189 checkpoint tensors loaded,
  a new smoke checkpoint was written, and the first patch's final PNG rendered successfully.
- Full six-epoch `diag_unfreeze_no_pu` completed and rendered all six patches; results above.
- Full six-epoch `diag_unfreeze` and `diag_no_pu` attribution controls also completed, with
  six final figures each. The other ten diagnostic arms are configured and tested, not trained.
- All 14 active arms passed the campaign's real preflight and dry-run path.