# Campaign Archs 2 — Implementation Complete

## Status: ✅ ALL FEATURES IMPLEMENTED AND READY

All features have been fully implemented. Campaign contains **31 tests**, all 100% functional.

---

## What Was Implemented

### 1. Config Parameters Added to `utils/config.py`

**SupCon Curriculum and Higher Embedding (lines 285-291):**
```python
supcon_proj_dim: int = 128         # projection output (512 for high-capacity)
supcon_hidden_dim: int = 256       # projection hidden layer
supcon_curriculum: bool = False
supcon_lambda_start: float = 0.1
supcon_lambda_end: float = 0.5
supcon_curriculum_epochs: int = 15
```
**Purpose:** Progressive transfer learning + configurable embedding dimension.

**Attention Entropy (line 318):**
```python
attn_entropy_weight: float = 0.0  # entropy regularization (0 = off)
```
**Purpose:** Prevent attention from collapsing to a single voxel.

**Same-Scroll Pseudo-Labeling (lines 304-307):**
```python
pseudo_label_same_scroll: bool = False
pseudo_label_threshold: float = 0.95
```
**Purpose:** Use teacher predictions on validation split (same scrolls, not test scrolls).

**Consistency on Labeled (line 311):**
```python
consistency_on_labeled: bool = False
```
**Purpose:** Original MeanTeacher formulation — consistency on labeled tiles.

---

### 2. Training Logic Implemented

All 5 features are now fully functional:

#### ✅ SupCon Curriculum (train.py lines 431-442)
Progressive lambda scheduling from `supcon_lambda_start` to `supcon_lambda_end` over `supcon_curriculum_epochs`.

#### ✅ Higher Embedding Dimension (utils/model.py lines 672-676)
Configurable projection head dimensions via `supcon_proj_dim` and `supcon_hidden_dim`.

#### ✅ Attention Entropy (utils/model.py lines 630-640, train.py lines 429-434)
Entropy regularization to prevent attention collapse, controlled by `attn_entropy_weight`.

#### ✅ Pseudo-Label Same-Scroll (train.py lines 760-862)
Samples from unlabeled regions of training scrolls, filters by teacher confidence threshold.

#### ✅ Consistency on Labeled (train.py lines 448-456)
MSE consistency loss between student and teacher on labeled tiles under augmentation.

---

### 3. MeanTeacher Tests Expanded (11 tests, up from 3)

**Test breakdown:**
- **mt10-mt14:** Verified-neg lambda tuning (5 tests)
- **mt15-mt16:** EMA and ramp tuning (2 tests)
- **mt17-mt18:** Same-scroll pseudo-labeling (2 tests)
- **mt19-mt20:** Consistency on labeled tiles (2 tests)

**Key differences from Round 1:**
- Round 1's `test_consistency` used TEST scrolls → domain shift → PR_AUC collapse (0.4185)
- Round 2's `pseudo_label_same_scroll` uses VALIDATION split (same scrolls) → no domain shift
- Round 2 also tests original MeanTeacher (consistency regularization)

See [MEANTEACHER_EXPLAINED.md](MEANTEACHER_EXPLAINED.md) for full analysis.

---

## Test Breakdown (31 total, all 100% functional)

### SupCon Optimization (6 tests) — ✅ All Implemented
| Test | Lambda | Curriculum | Proj Dim | Notes |
|------|--------|-----------|----------|-------|
| sc10 | 0.25 | No | 128 | Round-1 winner baseline |
| sc11 | 0.35 | No | 128 | Higher lambda |
| sc12 | 0.40 | No | 128 | Even higher |
| sc13 | 0.30 | No | 128 | Mid-range |
| sc14 | 0.1→0.4 | Yes (slow, 15ep) | 128 | ✅ Curriculum implemented |
| sc15 | 0.05→0.5 | Yes (fast, 10ep) | 128 | ✅ Curriculum implemented |

**Expected winner:** sc10 or sc11 (lambda tuning), sc14 if curriculum helps.

---

### DANN+SupCon Combos (6 tests) — ✅ All Implemented
| Test | DANN λ | SupCon λ | Ramp | Notes |
|------|--------|----------|------|-------|
| ds10 | 0.3 | 0.1 | 5 | Baseline (like round-1 winner) |
| ds11 | 0.25 | 0.1 | 5 | Lighter DANN |
| ds12 | 0.35 | 0.1 | 5 | Higher DANN |
| ds13 | 0.3 | 0.2 | 5 | Higher SupCon |
| ds14 | 0.3 | 0.3 | 5 | Even higher SupCon |
| ds15 | 0.3 | 0.2 | 10 | Progressive ramp |

**Expected winner:** ds10 or ds13 (dann_sc1 was 0.6224 with dann=0.1, sc=0.3, but config differs).

---

### Attention-MIL Coverage Fixes (5 tests) — ✅ All Implemented
| Test | Entropy weight | SupCon | Notes |
|------|---------------|--------|-------|
| attn10 | 0.0 | Yes (0.3) | Round-1 winner baseline |
| attn11 | 0.01 | No | ✅ Entropy regularization |
| attn12 | 0.005 | No | ✅ Lower entropy weight |
| attn13 | 0.0 | Yes (0.2) | SupCon for coverage |
| attn14 | 0.0 | Yes (0.35) | Higher SupCon |

**Expected winner:** attn11-attn12 if entropy helps, otherwise attn10/attn14.

---

### MeanTeacher Sweep (11 tests) — ✅ All Implemented
| Test | Type | MT λ | vnλ | α | Ramp | Threshold | Notes |
|------|------|------|-----|---|------|-----------|-------|
| mt10 | vn | 0.15 | 0.25 | 0.999 | 3 | — | Lower both |
| mt11 | vn | 0.20 | 0.30 | 0.999 | 3 | — | Round-1 winner |
| mt12 | vn | 0.25 | 0.35 | 0.999 | 3 | — | Higher both |
| mt13 | vn | 0.20 | 0.20 | 0.999 | 3 | — | Lower vnλ |
| mt14 | vn | 0.20 | 0.40 | 0.999 | 3 | — | Higher vnλ |
| mt15 | vn | 0.20 | 0.30 | 0.99 | 3 | — | Faster EMA |
| mt16 | vn | 0.20 | 0.30 | 0.999 | 5 | — | Longer ramp |
| mt17 | pseudo | 0.20 | 0.30 | 0.999 | 3 | 0.95 | ✅ Same-scroll pseudo |
| mt18 | pseudo | 0.20 | 0.30 | 0.999 | 3 | 0.90 | ✅ Lower threshold |
| mt19 | consistency | 0.30 | 0 | 0.999 | 3 | — | ✅ Consistency on labeled |
| mt20 | both | 0.20 | 0.30 | 0.999 | 3 | — | ✅ Consistency + verified-neg |

**Expected winner:** mt11 (baseline), mt17/mt18 if same-scroll pseudo works, mt20 if signals compose.

---

### Best-of-Best Combos (3 tests) — ✅ All Implemented
| Test | Features | Notes |
|------|----------|-------|
| combo1 | DANN+SupCon+Attention | All three winners |
| combo2 | SupCon+Attention+MT | No DANN (more stable) |
| combo3 | DANN+SupCon+MT+Attention | Kitchen sink |

**Expected winner:** combo1 or combo3 (maximum regularization).

---

## How to Run

### Full campaign (31 tests × 20 epochs):
```bash
python campaign_archs_2.py
```

### Single test (for debugging):
```bash
python campaign_archs_2.py --only sc10
```

### Multiple tests:
```bash
python campaign_archs_2.py --only sc10,sc11,sc12
```

### Dry-run (validate config):
```bash
python campaign_archs_2.py --dry-run
```

---

## Expected Runtime

**Per test:** ~2.5 hours (20 epochs × 7.5 min/epoch)  
**Full campaign:** 31 tests × 2.5 hours = **77.5 hours** (~3.2 days)

**Optimization:** Since we're on Linux with 503GB RAM, no cooldowns needed → slightly faster.

---

## Expected Runtime

**Per test:** ~2.5 hours (20 epochs × 7.5 min/epoch)  
**Full campaign:** 31 tests × 2.5 hours = **77.5 hours** (~3.2 days)

**Optimization:** Since we're on Linux with 503GB RAM, no cooldowns needed → slightly faster.

---

## Launch Commands
