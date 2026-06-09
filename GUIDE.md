# Visualizer Metrics Guide (Beginner-Friendly)

Last updated: 2026-06-08

This guide explains every current metric family logged by `utils/visualizer.py`.

Goal of this guide:

- explain what each metric is and how it is computed
- explain why it matters for your actual goal (ink readability)
- rank importance for this specific project
- explain what ideal vs realistic values look like
- explain metric combinations and what they usually mean

## 1) First Principles (Very Important)

Your model is trained as tile-level binary classification, not pixel segmentation.

That means:

- many scalar metrics describe classification behavior, not direct readability
- a metric can improve while output maps still look mushy or bright
- readability metrics (`R_M/*`) and probe metrics (`R_M/Probe/*`) are your best bridge to human-visible quality

Use this rule of thumb:

- `P_M/*`, `AUC/*`, `G_M/Acc/*`: useful but indirect for readability
- `R_M/*` and `R_M/Probe/*`: most directly tied to readability

Importance labels used in this guide:

- `high`: directly useful for readable ink maps
- `medium`: useful context, but indirect
- `low`: mostly operational or debugging context

## 2) Tag Map (What Exists Right Now)

Current scalar tag families logged by visualizer:

- `G_M/Loss/Train`, `G_M/Loss/Train_Raw`, `G_M/Loss/Valid`
- `G_M/Acc/Train`, `G_M/Acc/Valid`
- `P_M/Precision/Train`, `P_M/Precision/Valid`
- `P_M/Recall/Train`, `P_M/Recall/Valid`
- `P_M/F1_Score/Train`, `P_M/F1_Score/Valid`
- `P_M/Specificity/Train`, `P_M/Specificity/Valid`
- `AUC/ROC_AUC/Train`, `AUC/ROC_AUC/Valid`
- `AUC/PR_AUC/Train`, `AUC/PR_AUC/Valid`
- `Learning_Rate`
- `Time_Elapsed`
- `HardMining/HardNegatives`, `HardMining/HardPositives`
- `G_M/Loss/HM_<epoch>`
- `G_M/Acc/HM_<epoch>`
- `P_M/Precision/HM_<epoch>`
- `P_M/Recall/HM_<epoch>`
- `P_M/F1_Score/HM_<epoch>`
- `P_M/Specificity/HM_<epoch>`
- `AUC/ROC_AUC/HM_<epoch>`
- `AUC/PR_AUC/HM_<epoch>`
- `R_M/LocalContrast`
- `R_M/LocalRanking`
- `R_M/RecallAt1PctFPR`
- `R_M/PartialAUCAt1PctFPR`
- `R_M/TopKPrecision`
- `R_M/InkFractionPearson`
- `R_M/InkFractionSpearman`
- `R_M/SpillRatio`
- `R_M/ComponentCount`
- `R_M/MeanComponentSize`
- `R_M/ReadabilityComposite`
- `R_M/Probe/Easy/LocalContrast`, `R_M/Probe/Hard/LocalContrast`, `R_M/Probe/Scroll4_Pi/LocalContrast`
- `R_M/Probe/Easy/TopKPrecision`, `R_M/Probe/Hard/TopKPrecision`, `R_M/Probe/Scroll4_Pi/TopKPrecision`
- `R_M/Probe/Easy/ReadabilityComposite`, `R_M/Probe/Hard/ReadabilityComposite`, `R_M/Probe/Scroll4_Pi/ReadabilityComposite`
- `Hyperparameters/*` (run context scalars, not performance outcomes)

Current figure families logged by visualizer:

- `Confusion_Matrix`
- `Output_Histogram`
- `Metrics_Comparison`
- `Evaluation/Depth_Block_<d0>-<d1>`
- `Readability/Summary`
- `Readability/Compass`
- `ProbeROIs/Easy`, `ProbeROIs/Hard`, `ProbeROIs/Scroll4_Pi`
- `ProbeROIs/AllPatches_ByDepth`
- `HardMined/Overlay`
- `Test/Test_All_Depth_Blocks`, `Test/Scroll4_All_Depth_Blocks`

## 3) Core Classification Metrics

These are computed from thresholded predictions (`score > 0.5`) and labels.

### 3.1 Loss (`G_M/Loss/*`)

Tags:

- `G_M/Loss/Train`
- `G_M/Loss/Train_Raw`
- `G_M/Loss/Valid`
- `G_M/Loss/HM_<epoch>`

What it is:

- BCEWithLogits-based objective value
- lower is better in pure optimization terms
- `Train_Raw` is the raw masked BCE before extra regularization terms

Why relevant:

- tells you if optimization is working
- does not directly tell you readability quality

Importance for readability: `medium`

Ideal vs realistic:

- ideal: train and valid both trend downward and stabilize
- realistic: they will not go to zero in this weak-label setup
- if train goes down while valid goes flat/up for many epochs, that is overfitting pressure

Combination patterns:

- low train loss + high valid loss: overfit
- both high and flat: underfit or optimization issue
- both slowly decreasing while `R_M/*` improves: healthy training

### 3.2 Accuracy (`G_M/Acc/*`)

Tags:

- `G_M/Acc/Train`
- `G_M/Acc/Valid`
- `G_M/Acc/HM_<epoch>`

How calculated:

- $$\text{accuracy} = \frac{TP + TN}{TP + TN + FP + FN}$$

Why relevant:

- fast broad sanity check
- can be misleading in class-imbalanced settings (many easy negatives)

Importance for readability: `low`

Ideal vs realistic:

- ideal: high train and high valid together
- realistic: can look "good" even when readability is bad
- do not use this as primary model-selection metric

Combination patterns:

- high train + high valid + poor `R_M/LocalContrast`: model may be classifying easy negatives but not producing readable separation
- high train + lower valid: classic overfit

### 3.3 Precision (`P_M/Precision/*`)

Tags:

- `P_M/Precision/Train`
- `P_M/Precision/Valid`
- `P_M/Precision/HM_<epoch>`

How calculated:

- $$\text{precision} = \frac{TP}{TP + FP}$$

Why relevant:

- measures false-positive control
- higher precision usually means less broad bright spill

Importance for readability: `high`

Ideal vs realistic:

- ideal: increase precision without collapsing recall
- realistic: pushing precision very high often lowers recall in weak-ink regions

Combination patterns:

- high precision + very low recall: map may be too sparse and miss letters
- improving precision + stable recall + lower `R_M/SpillRatio`: strong readability direction

### 3.4 Recall (`P_M/Recall/*`)

Tags:

- `P_M/Recall/Train`
- `P_M/Recall/Valid`
- `P_M/Recall/HM_<epoch>`

How calculated:

- $$\text{recall} = \frac{TP}{TP + FN}$$

Why relevant:

- measures how many positive tiles are found
- useful for not missing weak ink

Importance for readability: `high`

Ideal vs realistic:

- ideal: moderate-high recall with controlled spill
- realistic: very high recall alone can look like brightness haze

Combination patterns:

- high recall + low precision + high `R_M/SpillRatio`: likely spillout
- moderate recall + strong local contrast metrics: often better human readability

### 3.5 F1 (`P_M/F1_Score/*`)

Tags:

- `P_M/F1_Score/Train`
- `P_M/F1_Score/Valid`
- `P_M/F1_Score/HM_<epoch>`

How calculated:

- harmonic mean of precision and recall
- $$F1 = 2\cdot\frac{PR}{P+R}$$

Why relevant:

- balanced summary of precision and recall
- commonly useful for model checkpointing

Importance for readability: `medium`

Ideal vs realistic:

- ideal: valid F1 rises and stays stable
- realistic in this repo: F1 is usually moderate, not near 1.0
- from historical runs in `runs.md`, valid F1 has often sat in the rough ~0.27 to ~0.46 band

Combination patterns:

- F1 up while `R_M/ReadabilityComposite` flat/down: objective mismatch is still present
- F1 stable but readability metrics up: this can still be a practical win

### 3.6 Specificity (`P_M/Specificity/*`)

Tags:

- `P_M/Specificity/Train`
- `P_M/Specificity/Valid`
- `P_M/Specificity/HM_<epoch>`

How calculated:

- true-negative rate
- $$\text{specificity} = \frac{TN}{TN + FP}$$

Why relevant:

- captures background suppression
- especially important where false bright regions destroy readability

Importance for readability: `high`

Ideal vs realistic:

- ideal: high specificity while recall remains acceptable
- realistic: very high specificity can come at recall cost

Combination patterns:

- specificity up + recall down too far: oversuppression
- specificity up + stable recall + lower spill ratio: excellent direction

### 3.7 ROC-AUC (`AUC/ROC_AUC/*`)

Tags:

- `AUC/ROC_AUC/Train`
- `AUC/ROC_AUC/Valid`
- `AUC/ROC_AUC/HM_<epoch>`

How calculated:

- ranking quality across all thresholds using ROC curve area

Why relevant:

- threshold-independent ranking measure
- can hide low-FPR failures that matter for readability

Importance for readability: `medium`

Ideal vs realistic:

- ideal: upward trend and stable train-valid gap
- realistic: good ROC-AUC does not guarantee readable maps

Combination patterns:

- high ROC-AUC + poor `R_M/RecallAt1PctFPR`: ranking is okay globally, weak where strict false-positive control matters

### 3.8 PR-AUC (`AUC/PR_AUC/*`)

Tags:

- `AUC/PR_AUC/Train`
- `AUC/PR_AUC/Valid`
- `AUC/PR_AUC/HM_<epoch>`

How calculated:

- area under precision-recall curve (average precision)

Why relevant:

- usually more informative than ROC-AUC in class-imbalanced problems
- still global and not explicitly local-readability aware

Importance for readability: `medium`

Ideal vs realistic:

- ideal: steady increase on valid without train-valid blowup
- realistic: moderate values can still yield useful readability if local metrics are good

Combination patterns:

- PR-AUC up + local contrast down: better global ranking but blurrier local separation

## 4) Readability Metrics (`R_M/*`) - Most Important Section

These metrics were added to align evaluation with what humans care about.

### 4.1 Local Contrast (`R_M/LocalContrast`)

How calculated:

- for each positive tile, compute tile score minus mean score of nearby negative tiles in a local window
- average that difference

Range and direction:

- practical range is usually around negative to modest positive values
- higher is better

Importance for readability: `high`

Ideal vs realistic:

- ideal: clearly positive and trending upward
- realistic: small but stable positive gains are meaningful

Combination patterns:

- high local contrast + low spill ratio: letters should pop better
- low local contrast + high recall: likely broad bright haze

### 4.2 Local Ranking (`R_M/LocalRanking`)

How calculated:

- for each positive tile, compare score against nearby negative tile scores
- metric is fraction of local negatives that the positive tile outranks

Range and direction:

- 0 to 1
- higher is better

Importance for readability: `high`

Ideal vs realistic:

- ideal: approach 1.0
- realistic: strong practical progress can happen well below perfect ranking

Combination patterns:

- ranking up + top-k precision up: strong sign maps are becoming more usable
- ranking up but spill unchanged: local ordering better, but global suppression still needs work

### 4.3 Recall At 1 Percent FPR (`R_M/RecallAt1PctFPR`)

How calculated:

- from ROC curve, find maximum recall where false positive rate is at most 1%

Range and direction:

- 0 to 1
- higher is better

Importance for readability: `high`

Ideal vs realistic:

- ideal: high recall even under strict FP budget
- realistic: this is hard; gradual improvement matters

Combination patterns:

- high standard recall but low recall@1%FPR: model only finds positives if allowed to over-brighten

### 4.4 Partial AUC At 1 Percent FPR (`R_M/PartialAUCAt1PctFPR`)

How calculated:

- normalized area under ROC curve only in FPR range [0, 0.01]

Range and direction:

- 0 to 1
- higher is better

Importance for readability: `high`

Ideal vs realistic:

- ideal: increase over time, especially on valid/eval depth blocks
- realistic: absolute values may look modest; trend is important

Combination patterns:

- pAUC@1% up + spill ratio down: strong low-FP readability improvement

### 4.5 Top-K Precision (`R_M/TopKPrecision`)

How calculated:

- choose k equal to number of true positive tiles in eval set
- precision among top-k highest scored tiles

Range and direction:

- 0 to 1
- higher is better

Importance for readability: `high`

Ideal vs realistic:

- ideal: top-ranked tiles are mostly true positives
- realistic: this should improve before full-map readability looks perfect

Combination patterns:

- top-k precision up + local contrast up: very good signal for readable hotspots
- top-k precision up but recall down sharply: model may only keep easiest positives

### 4.6 Ink-Fraction Correlations (`R_M/InkFractionPearson`, `R_M/InkFractionSpearman`)

How calculated:

- per tile, compare prediction score with tile ink fraction from labels
- Pearson checks linear relation
- Spearman checks rank-order relation

Range and direction:

- -1 to 1
- higher is better

Importance for readability: `medium`

Ideal vs realistic:

- ideal: positive correlation that improves over time
- realistic: Spearman is usually more stable than Pearson in noisy settings

Combination patterns:

- Spearman up while F1 flat: scores may be becoming more "density-aware" even if threshold metrics lag

### 4.7 Spill Ratio (`R_M/SpillRatio`)

How calculated:

- compute predicted score mass outside a 1-tile dilated GT mask
- divide by total predicted score mass on valid tiles

Range and direction:

- lower is better
- near 0 is best
- can exceed 1.0 in poor cases

Importance for readability: `high`

Ideal vs realistic:

- ideal: low and decreasing
- realistic: this may fluctuate by depth block; watch moving trend

Combination patterns:

- recall up + spill ratio up: likely over-brightening
- recall stable + spill ratio down: cleaner map with preserved sensitivity

### 4.8 Component Count (`R_M/ComponentCount`)

How calculated:

- build top-k budget mask (k = number of positives)
- count connected components in that mask

Range and direction:

- integer >= 0
- there is no universal "best" number

Importance for readability: `medium`

Ideal vs realistic:

- ideal: not too fragmented, not one giant blob
- realistic: use trend and compare against visuals, not absolute target

Combination patterns:

- very high count + low mean component size: noisy speckles
- very low count + high spill: over-merged blobs

### 4.9 Mean Component Size (`R_M/MeanComponentSize`)

How calculated:

- mean size of connected components in top-k budget mask

Range and direction:

- positive value
- no universal ideal absolute number

Importance for readability: `medium`

Ideal vs realistic:

- ideal: component sizes reflect plausible stroke clusters, not random dots or giant floods
- realistic: compare relative changes with component count and probe figures

Combination patterns:

- mean size tiny + count huge: salt-and-pepper behavior
- mean size huge + count tiny + high spill: coarse flood behavior

### 4.10 Readability Composite (`R_M/ReadabilityComposite`)

How calculated:

- average of normalized readability sub-metrics:
  - normalized local contrast
  - local ranking
  - recall@1%FPR
  - partial AUC@1%FPR
  - top-k precision
  - normalized Spearman fraction correlation
  - spill suppression term (`1 - spill_ratio`, clipped)

Range and direction:

- 0 to 1
- higher is better

Importance for readability: `high`

Ideal vs realistic:

- ideal: steady upward trend over eval points
- realistic: expect noise; smooth trend matters more than single-point spikes

Combination patterns:

- composite up while F1 flat: likely true readability progress
- composite down while F1 up: likely classification/readability mismatch

## 5) Probe Metrics (`R_M/Probe/*`) - Your Fast Reality Check

Probe tags are computed on fixed ROIs:

- easy small-scroll1 ROI
- hard small-scroll1 ROI
- scroll4 pi ROI

Per-probe scalar tags:

- `R_M/Probe/<tag>/LocalContrast`
- `R_M/Probe/<tag>/TopKPrecision`
- `R_M/Probe/<tag>/ReadabilityComposite`

Why relevant:

- faster and more stable qualitative checkpoints than waiting only for full `test_int`
- directly tied to known regions you care about

Importance for readability: `high`

Ideal vs realistic:

- ideal: all three probes improve over time
- realistic: easy probe improves first, hard/scroll4 lag and move slower

Cadence:

- probe figures are logged every `tra.probe_int` epochs (default `5`)

Combination patterns:

- easy up + hard flat: model is learning obvious cases but not weak structure
- hard up + scroll4 flat: domain transfer gap remains
- scroll4 up without major spill growth: very promising for your goal

## 6) Hard-Mining Counters and HM Replay Metrics

### 6.1 Hard example counts (`HardMining/HardNegatives`, `HardMining/HardPositives`)

What they are:

- number of hard negatives and hard positives mined during eval pass

Importance for readability: `medium`

How to read:

- high hard negatives can indicate false-positive pressure
- high hard positives can indicate missed true positives
- trends matter more than one value

### 6.2 HM replay metrics (`*_HM_<epoch>`)

What they are:

- normal metrics computed by re-evaluating stored hard-mining files from earlier epochs

Importance for readability: `medium`

How to read:

- improving HM precision/specificity can show better control on difficult negatives
- improving HM recall can show fewer misses on difficult positives
- if HM gets worse while global metrics improve, model may be overfitting easy regions

## 7) Operational Scalars

### 7.1 Learning rate (`Learning_Rate`)

Importance for readability: `low`

Use:

- helps explain metric jumps/drops after scheduler changes

### 7.2 Time elapsed (`Time_Elapsed`)

Importance for readability: `low`

Use:

- runtime tracking only

### 7.3 Hyperparameter scalars (`Hyperparameters/*`)

Importance for readability: `low`

Use:

- run-context metadata for comparisons
- not a quality signal by itself

## 8) Figure Panels (Not Scalars, But Important)

### 8.1 `Confusion_Matrix`

Importance: `medium`

Use:

- quick class-error intuition
- dominated by easy negatives, so do not use alone

### 8.2 `Output_Histogram`

Importance: `medium`

Use:

- checks score distribution drift
- bimodality is not required for readability success

### 8.3 `Metrics_Comparison`

Importance: `low`

Use:

- visual summary of already-logged scalar metrics

### 8.4 `Evaluation/Depth_Block_*` and `Readability/Summary`

Importance: `high`

Use:

- direct depth-by-depth readability inspection
- helps detect specific depth blocks causing failure

### 8.5 `ProbeROIs/*`

Importance: `high`

Use:

- fixed-region qualitative truth check
- usually most actionable during long training

### 8.6 `HardMined/Overlay`

Importance: `medium`

Use:

- shows where hard examples cluster
- useful for understanding failure geography

### 8.7 `Test/*_All_Depth_Blocks`

Importance: `high`

Use:

- full qualitative check
- sparse cadence by design (`test_int`), so pair with probes

### 8.8 `Readability/Compass`

Importance: `high`

Use:

- radial summary of normalized readability dimensions
- fast check of aggregate readability balance versus best depth block

### 8.9 `ProbeROIs/AllPatches_ByDepth`

Importance: `high`

Use:

- easy, hard, and scroll4 patches shown side-by-side for each depth block
- quickest way to compare depth behavior across all probe regions in one panel

## 9) Most Useful Metric Combinations (Cheat Sheet)

### 9.1 Good general direction

- `R_M/ReadabilityComposite` up
- `R_M/LocalContrast` up
- `R_M/SpillRatio` down
- `R_M/RecallAt1PctFPR` up or stable
- probe composites (especially `Hard` and `Scroll4_Pi`) up

Interpretation:

- model is becoming more readable, not just more confident

### 9.2 Classic overfit pattern

- train `P_M/F1_Score` up
- valid `P_M/F1_Score` flat/down
- train loss down while valid loss up
- probe metrics flat/down

Interpretation:

- model memorizing training distribution, not improving useful readability

### 9.3 Brightness spill pattern

- recall up
- precision down
- specificity down
- `R_M/SpillRatio` up
- local contrast flat/down

Interpretation:

- model is finding positives by raising score floor too broadly

### 9.4 Over-suppressed pattern

- precision and specificity high
- recall very low
- top-k precision decent but probe hard region still weak

Interpretation:

- map is clean but missing weak strokes

### 9.5 Global metrics improve but readability does not

- `AUC/*` and maybe `P_M/F1_Score/*` up
- `R_M/ReadabilityComposite` flat/down
- probe metrics flat

Interpretation:

- objective mismatch still dominating

## 10) Practical Priority Order For Your Decisions

When choosing checkpoints or comparing experiments for this project, prioritize in this order:

1. `R_M/ReadabilityComposite`
2. `R_M/LocalContrast`, `R_M/TopKPrecision`, `R_M/SpillRatio`, `R_M/RecallAt1PctFPR`
3. `R_M/Probe/*` trends, especially `Hard` and `Scroll4_Pi`
4. `P_M/F1_Score/Valid`, `P_M/Precision/Valid`, `P_M/Recall/Valid`, `P_M/Specificity/Valid`
5. `AUC/PR_AUC/Valid` and `AUC/ROC_AUC/Valid`
6. `G_M/Acc/*`

If this order and the old order disagree, trust readability-first metrics for your stated goal.

## 11) Expected Good vs Bad Ranges (By Section)

These are practical targets for this project and dataset style, not universal ML constants.

### 11.1 Section 3 (Core Classification Metrics)

| Metric | Good (practical) | Bad / warning |
|---|---|---|
| `G_M/Loss/Valid` | roughly `<= 1.20` and not diverging from train by more than ~15% | persistently `>= 1.60`, or valid up while train down for many epochs |
| `G_M/Acc/Valid` | often `>= 0.85` | `<= 0.75` (or high accuracy with poor readability metrics) |
| `P_M/Precision/Valid` | roughly `0.45` to `0.75` | `<= 0.25` (too many false positives) |
| `P_M/Recall/Valid` | roughly `0.35` to `0.65` | `<= 0.20` (misses ink) or very high with rising spill |
| `P_M/F1_Score/Valid` | solid `>= 0.35`, strong `>= 0.42` | `<= 0.25` or degrading over evals |
| `P_M/Specificity/Valid` | usually `>= 0.92` | `<= 0.85` (weak background suppression) |
| `AUC/ROC_AUC/Valid` | `>= 0.88` is usually good | `<= 0.75` |
| `AUC/PR_AUC/Valid` | `>= 0.40` is usually strong here | `<= 0.20` |

### 11.2 Section 4 (Readability Metrics)

| Metric | Good (practical) | Bad / warning |
|---|---|---|
| `R_M/LocalContrast` | `> 0.10` and trending up | `<= 0.00` |
| `R_M/LocalRanking` | `>= 0.70` | `<= 0.55` |
| `R_M/RecallAt1PctFPR` | `>= 0.25` (hard metric; gradual gains count) | `<= 0.10` |
| `R_M/PartialAUCAt1PctFPR` | `>= 0.20` and rising | `<= 0.08` |
| `R_M/TopKPrecision` | `>= 0.45` | `<= 0.25` |
| `R_M/InkFractionPearson` | `>= 0.15` | near `0` or negative |
| `R_M/InkFractionSpearman` | `>= 0.30` | `<= 0.10` |
| `R_M/SpillRatio` | `< 0.45` | `>= 0.80` |
| `R_M/ComponentCount` | typically stable over evals; roughly ~`5%` to `40%` of top-k budget count | very high fragmentation (`> 80%` of top-k budget) or over-merged collapse (`< 3%`) |
| `R_M/MeanComponentSize` | often around `2` to `20` tiles | `< 1.5` (salt-and-pepper) or `> 60` (blob flood) |
| `R_M/ReadabilityComposite` | `>= 0.50` good, `>= 0.60` very good | `<= 0.35` |

### 11.3 Section 5 (Probe Metrics)

| Metric | Good (practical) | Bad / warning |
|---|---|---|
| `R_M/Probe/Easy/ReadabilityComposite` | `>= 0.55` | `<= 0.40` |
| `R_M/Probe/Hard/ReadabilityComposite` | `>= 0.40` | `<= 0.28` |
| `R_M/Probe/Scroll4_Pi/ReadabilityComposite` | `>= 0.35` | `<= 0.22` |
| probe `LocalContrast` | trending positive and up | flat near zero or negative |
| probe `TopKPrecision` | typically `>= 0.40` on easy, improving on hard/scroll4 | persistently `<= 0.25` |

### 11.4 Section 6 (Hard-Mining Counters and HM Replay)

| Metric family | Good (practical) | Bad / warning |
|---|---|---|
| `HardMining/HardNegatives` | downward trend over several evals (about `>= 20%` drop over ~3 evals is healthy) | persistent growth (`>= 20%` rise over ~3 evals) |
| `HardMining/HardPositives` | steady or down while recall stays stable | rising with falling recall |
| `*_HM_<epoch>` replay metrics | replay precision/specificity improving without replay recall collapse | replay metrics degrade while global metrics improve |

### 11.5 Section 7 (Operational Scalars)

| Metric | Good vs bad meaning |
|---|---|
| `Learning_Rate` | no direct good/bad value by itself; interpret only relative to scheduler events |
| `Time_Elapsed` | no quality signal; operational only |
| `Hyperparameters/*` | run metadata; not a score |

### 11.6 Section 8 (Figure Panels)

| Figure family | Good visual pattern | Bad visual pattern |
|---|---|---|
| `Evaluation/Depth_Block_*` | clear localized ink hotspots with limited haze | broad brightness, weak stroke separation |
| `Readability/Summary` | multiple readability bars rising with lower spill contribution | flat/declining readability bars, unstable heatmap |
| `Readability/Compass` | large, balanced polygon across ranking/top-k/low-fpr axes | collapsed shape on low-fpr and spill suppression axes |
| `ProbeROIs/*` | easy/hard/scroll4 all show progressively clearer strokes | easy only improves while hard/scroll4 stagnate |
| `ProbeROIs/AllPatches_ByDepth` | one or more depth bands look consistently strong across all probes | no depth block looks consistently usable |

## 12) Super-Quick Glance Card

Use this when scanning TensorBoard quickly.

| Metric | Should be doing | Good-zone target | Fast bad sign |
|---|---|---|---|
| `R_M/ReadabilityComposite` | up | `>= 0.50` | `<= 0.35` or flat |
| `R_M/LocalContrast` | up | `> 0.10` | `<= 0.00` |
| `R_M/LocalRanking` | up | `>= 0.70` | `<= 0.55` |
| `R_M/RecallAt1PctFPR` | up slowly | `>= 0.25` | `<= 0.10` |
| `R_M/PartialAUCAt1PctFPR` | up | `>= 0.20` | `<= 0.08` |
| `R_M/TopKPrecision` | up | `>= 0.45` | `<= 0.25` |
| `R_M/SpillRatio` | down | `< 0.45` | `>= 0.80` |
| `R_M/Probe/Hard/ReadabilityComposite` | up | `>= 0.40` | `<= 0.28` |
| `R_M/Probe/Scroll4_Pi/ReadabilityComposite` | up | `>= 0.35` | `<= 0.22` |
| `P_M/F1_Score/Valid` | up or stable | `>= 0.35` | `<= 0.25` |

## 13) Scope Clarifier (Full Scroll vs Patch)

- `R_M/*` metrics are computed on the full evaluation map (tile grid over the eval region, aggregated across depth blocks)
- `R_M/Probe/*` metrics are computed only on fixed probe patches (easy, hard, scroll4)
- So readability is tracked at both levels: global map quality and targeted ROI quality
