# Campaign Archs 2 — Test Plan

## Overview

**28 tests total** (vs 15 in original campaign_archs)  
**Runtime:** ~3-4 days at 20 epochs each  
**Log directory:** `runs_archs2` (separate tensorboard)  
**Key optimization:** Only renders 1 scroll during eval (much faster)

## Test Batches

### Batch 1: SupCon Optimization (sc10-sc15) — 6 tests
**Rationale:** SupCon T=0.07 was the clear winner. Optimize lambda and add curriculum.

- **sc10-sc12:** Lambda interpolation (0.25, 0.35, 0.4) around winner (0.3)
- **sc13:** Winner λ=0.3 for 20 epochs (baseline)
- **sc14-sc15:** Lambda curriculum (progressive transfer)
  - Start low (ink detection), ramp up (cross-scroll transfer)
  - **Requires implementation:** `supcon_curriculum`, `supcon_lambda_start/end/epochs`

**Expected outcome:** If curriculum works, should see smooth improvement vs plateau.

---

### Batch 2: DANN+SupCon Combos (ds10-ds15) — 6 tests
**Rationale:** dann_sc1 was TOP SCORER (0.6224). Optimize the combo.

- **ds10:** Winner (DANN λ=0.3, SupCon λ=0.1) for 20 epochs
- **ds11-ds12:** Vary DANN lambda (0.25, 0.35)
- **ds13-ds14:** Vary SupCon lambda (0.2, 0.3)
- **ds15:** Progressive DANN ramp (15 epochs instead of 8)

**Expected outcome:** Confirm dann_sc1 is robust; find even better λ combo.

---

### Batch 3: Attention-MIL Coverage Fixes (attn10-attn14) — 5 tests
**Rationale:** Attention scored well (0.6098) but was sparse. Fix coverage.

- **attn10:** Pure attention for 20 epochs (baseline)
- **attn11-attn12:** Entropy regularizer (force spread)
  - **Requires implementation:** `attn_entropy_weight` in model
- **attn13-attn14:** Attention + SupCon (vary SupCon lambda)

**Expected outcome:** Probe scores >0.35 (vs 0.295 for sparse attention).

---

### Batch 4: MeanTeacher Same-Scroll Unlabeled (mt10-mt12) — 3 tests
**Rationale:** Verified-neg works (0.6146). Fix test-consistency by pseudo-labeling SAME scrolls.

- **mt10:** Verified-neg only for 20 epochs (baseline)
- **mt11:** Pseudo-label on same-scroll validation regions
  - **Requires implementation:** `pseudo_label_same_scroll`, `pseudo_label_threshold`
- **mt12:** Consistency regularization on labeled tiles (original MT)
  - **Requires implementation:** `consistency_on_labeled`

**Expected outcome:** mt11 should work (same domain). mt12 is standard MT.

---

### Batch 5: Grand Combos (combo1-combo3) — 3 tests
**Rationale:** Combine the winners to see if gains compound.

- **combo1:** DANN + SupCon + Attention
- **combo2:** SupCon + Attention + MeanTeacher
- **combo3:** Kitchen sink (all features)

**Expected outcome:** If features are orthogonal, should see additive gains.

---

## Implementation Status

### ✓ Already Implemented (works immediately):
- SupCon basic (lambda, temp)
- DANN (lambda, ramp_epochs)
- Attention-MIL (basic)
- MeanTeacher (verified-neg, test-consistency)

### ⚠ Requires Implementation (will skip with warning):
- `supcon_curriculum` + related params (sc14, sc15)
- `attn_entropy_weight` (attn11, attn12)
- `pseudo_label_same_scroll`, `pseudo_label_threshold` (mt11)
- `consistency_on_labeled` (mt12)

**Strategy:** Run the campaign now. Implemented features will run; unimplemented ones will skip with warnings. Implement the missing features based on which batch shows promise.

---

## Priority Order (if running selectively)

### High Priority (run first):
1. **ds10-ds14** (DANN+SupCon tuning) — top scorer, low-hanging fruit
2. **sc10-sc13** (SupCon lambda tuning) — most stable, proven winner
3. **attn13-attn14** (Attention+SupCon) — works without new implementation

### Medium Priority:
4. **mt10** (MeanTeacher baseline) — verify round-1 result at 20 epochs
5. **combo1** (DANN+SupCon+Attention) — test if features compose

### Low Priority (requires implementation):
6. **sc14-sc15** (curriculum)
7. **attn11-attn12** (entropy)
8. **mt11-mt12** (same-scroll pseudo-label)
9. **combo2-combo3** (full combos)

---

## Quick Start

```bash
# Dry run to verify all configs
python campaign_archs_2.py --dry-run

# Run high-priority DANN+SupCon batch
python campaign_archs_2.py --only ds10,ds11,ds12,ds13,ds14

# Run full campaign
log="_ves_tmp/campaign_archs2_$(date +%Y%m%d_%H%M%S).log" && \
stdbuf -oL -eL python campaign_archs_2.py 2>&1 | tee "$log"
```

## Expected Outcomes

**Target:** PR_AUC > 0.64 (vs 0.6224 baseline from round 1)

If we hit 0.65+, that's a **4% relative gain** → validates the optimization strategy and confirms that:
1. SupCon can be pushed further
2. DANN+SupCon combo is robust
3. Features compose (not just noise)

**Next iteration:** Implement the promising unimplemented features (curriculum, entropy, etc.) based on which batch shows the most potential.
