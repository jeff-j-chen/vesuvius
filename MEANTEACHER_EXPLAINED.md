# MeanTeacher Explained — Why Round 2 is Different

## What Went Wrong in Round 1

### Test-Consistency Failure (mt_tc1): PR_AUC Collapsed to 0.4185

**What it tried to do:**
- Maintain EMA teacher model (slow-moving average of student weights)
- Use teacher predictions on **test-scroll tiles** (unlabeled) as pseudo-labels
- Train student to match teacher on those tiles

**Why it failed:**
```
Training scrolls:  PHerc0139 (15 segments), PHerc0814 (1 segment)
Test scrolls:      PHerc0813, PHerc0211, PHerc1203, PHerc1447
```

**Domain shift:** Test scrolls are **different physical objects** with different:
- Papyrus texture (fiber orientation, density variation)
- Ink morphology (PHerc1447 is 8.64µm upsampled, PHerc1203 is new material)
- Scan parameters (slight differences in keV/distance)

**Result:** Teacher's "confident" predictions reflected **scroll-specific biases**, not ground truth. Example failure mode:
- PHerc1203 has textured papyrus (rough fibers)
- Teacher (trained on PHerc0139's smooth papyrus) predicts "ink" on texture
- Student learns: "textured papyrus = ink" → systematic error → PR_AUC collapse

**The loss curve confirms this:**
```
mt_tc1: Loss spiked to 0.8217 at epoch 5 (when test-consistency ramped up)
        PR_AUC dropped to 0.4185 (vs 0.6+ baseline)
```

---

## What Works in Round 1: Verified-Negative Supervision (mt_vn1, mt_vn3)

### Why It Works: Conservative, Hi-Res Ground Truth

**Method:**
- Use 2.4µm inklabels (10x better SNR than 9.4µm) to identify **definite papyrus**
- Tiles where `2.4µm_label < threshold` (e.g., 31/255) are trusted negatives
- Add extra BCE supervision on these tiles (weight = `verified_neg_lambda`)

**Why it's safe:**
- 2.4µm labels are **ground truth** (human-verified at high resolution)
- Only asserts **papyrus** where SNR is unambiguous (conservative)
- Never asserts ink on unlabeled regions (no speculation)

**Results:**
```
mt_vn1 (λ=0.2, vnλ=0.3): PR_AUC=0.6146 (4th place)
mt_vn3 (λ=0.4, vnλ=0.4): PR_AUC=0.5892 (overshoots — too much weight on hard negs)
```

**Interpretation:** The 2.4µm labels identify **hard negatives** — high-density papyrus that looks ink-like at 9.4µm. Reinforcing them as negatives sharpens the decision boundary.

---

## Round 2 Fixes: Same-Scroll Pseudo-Labeling

### Strategy 1: Pseudo-Label on Validation Split (Same Scrolls)

**New approach (mt17, mt18):**
- Use teacher predictions on **unlabeled tiles from TRAINING scrolls** (the 20-25% validation split)
- Same domain → teacher's predictions reflect ink features, not scroll-specific texture
- Confidence threshold (e.g., p>0.95 or p<0.05) filters uncertain predictions

**Why this should work:**
- Teacher and student see the **same scroll distribution** (just different tiles)
- No domain shift → pseudo-labels are more reliable
- Standard in semi-supervised learning (Tarvainen 2017, original MeanTeacher)

**Parameters:**
```python
pseudo_label_same_scroll: bool = True
pseudo_label_threshold: float = 0.95  # use teacher pred only if confidence > 95%
```

**Expected outcome:** Should outperform verified-neg alone (more supervision), without the collapse of cross-scroll pseudo-labeling.

---

### Strategy 2: Consistency Regularization on Labeled Tiles (mt19, mt20)

**Original MeanTeacher formulation:**
- Don't pseudo-label unlabeled tiles
- Instead: enforce **student-teacher consistency on labeled tiles** under different augmentations
- Teacher sees original, student sees augmented → consistency loss

**Why this is different:**
- Not asserting any new labels (no pseudo-labeling)
- Just enforcing: "model should give consistent predictions under augmentation"
- This is a **regularizer**, not a data augmentation

**Expected outcome:** Smoother optimization, better generalization. Should compose well with verified-neg.

---

## Round 2 MeanTeacher Test Matrix (11 tests)

### Batch A: Verified-Neg Lambda Tuning (mt10-mt14)
Fine-tune the round-1 winner (mt_vn1).

| Test | λ (MT) | vnλ (verified-neg) | Notes |
|------|--------|-------------------|-------|
| mt10 | 0.15 | 0.25 | Lower both |
| mt11 | 0.20 | 0.30 | Round 1 winner |
| mt12 | 0.25 | 0.35 | Higher both |
| mt13 | 0.20 | 0.20 | Lower vnλ only |
| mt14 | 0.20 | 0.40 | Higher vnλ only |

**Goal:** Find optimal tradeoff between consistency (λ) and hard-negative reinforcement (vnλ).

---

### Batch B: EMA and Ramp Tuning (mt15, mt16)

| Test | α (EMA) | Ramp epochs | Notes |
|------|---------|-------------|-------|
| mt11 | 0.999 | 3 | Baseline (slow teacher) |
| mt15 | 0.99 | 3 | Fast teacher (updates 10x faster) |
| mt16 | 0.999 | 5 | Longer ramp (give model time to stabilize) |

**Goal:** Test whether teacher EMA speed or ramp duration matters.

---

### Batch C: Same-Scroll Pseudo-Labeling (mt17, mt18)

| Test | Threshold | Notes |
|------|-----------|-------|
| mt17 | 0.95 | High-confidence only (precision) |
| mt18 | 0.90 | Lower threshold (coverage) |

**Goal:** Validate that same-scroll pseudo-labeling works without collapse.

---

### Batch D: Consistency on Labeled (mt19, mt20)

| Test | Consistency | Verified-neg | Notes |
|------|-------------|--------------|-------|
| mt19 | Yes | No | Original MT (regularization only) |
| mt20 | Yes | Yes | Both signals |

**Goal:** Test whether consistency regularization adds value on top of verified-neg.

---

## Expected Outcomes

### Best Case:
- **mt17 or mt18** (same-scroll pseudo) hits PR_AUC > 0.63 → validates the fix
- **mt20** (consistency + verified-neg) hits PR_AUC > 0.64 → signals compose

### Worst Case:
- All mt tests plateau at 0.61-0.62 → MeanTeacher gains are marginal, focus elsewhere

### Learning:
- If mt17/mt18 work, we've proven that **domain shift was the problem**, not MeanTeacher itself
- If mt19/mt20 work, we've validated the **original MeanTeacher formulation** for this task
- Lambda tuning (mt10-mt14) tells us **how much weight to put on each signal**

---

## Key Differences from Round 1 (Summary)

| Aspect | Round 1 | Round 2 |
|--------|---------|---------|
| **Test-consistency target** | Test scrolls (different domain) | Validation split (same scrolls) |
| **Failure mode** | Domain shift → PR_AUC collapse | Fixed: same domain |
| **# of MT tests** | 3 | 11 (comprehensive sweep) |
| **Verified-neg tuning** | 2 values (λ=0.1, 0.3) | 5 values (λ=0.15-0.25, vnλ=0.2-0.4) |
| **Consistency formulation** | Pseudo-labeling only | Pseudo + regularization |
| **Expected gain** | ~0.01 (mt_vn1=0.6146) | 0.02-0.03 if same-scroll works |

---

## The Bottom Line

**Round 1's test-consistency failed because it tried to pseudo-label DIFFERENT scrolls.**  
**Round 2 fixes it by pseudo-labeling SAME scrolls (validation split).**

If this works, it proves:
1. MeanTeacher is sound for this task
2. Domain shift was the bottleneck, not the method
3. Semi-supervised learning can help when labels are expensive

If it doesn't work (all plateau at 0.62), then:
1. The signal in unlabeled validation tiles is too noisy
2. Verified-neg alone is the ceiling for MeanTeacher
3. Focus effort on SupCon + DANN + Attention instead
