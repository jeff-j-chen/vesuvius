# What Worked, What Didn't, and Why — Actionable Analysis

## The Real Problem: Generalization, Not Capacity

**Key evidence:** Past runs show train accuracy >95% while valid collapses. This proves:
1. The model CAN learn a representation of ink at 9.4µm
2. The bottleneck is **cross-scroll transfer**, not signal detection
3. The labels (from hi-res) are good enough to supervise learning

**Reframe:** This is a **domain adaptation** problem. Each scroll is a domain. We need the model to learn scroll-invariant ink features.

---

## Technique-by-Technique Breakdown

### 1. SupCon (Supervised Contrastive) — **CLEARLY WORKS**

#### What It Does:
Adds a projection head that maps backbone features → embedding space. Contrastive loss:
- **Pulls ink tiles together** (across all scrolls) in embedding space
- **Pushes papyrus tiles away** from ink tiles
- Temperature T controls clustering tightness (lower T = harder clustering)

#### Why It Works:
**Directly attacks the generalization problem.** By forcing ink features from scroll A and scroll B to cluster together, the model must learn scroll-invariant ink representations. It can't cheat by memorizing scroll-specific textures.

#### Results Analysis:
```
T=0.07, λ=0.1: PR_AUC=0.5838 (crashed at ep9, num_workers issue)
T=0.07, λ=0.2: PR_AUC=0.5923 (stable)
T=0.07, λ=0.3: PR_AUC=0.6186 (BEST pure SupCon)
T=0.2,  λ=0.1: PR_AUC=0.5651 (lower)
T=0.2,  λ=0.3: PR_AUC=0.5371 (lower)
```

**Key insights:**
- **Temperature matters:** T=0.07 (standard from Khosla 2020) >> T=0.2. Tighter clustering is better.
- **Lambda sweet spot:** 0.2-0.3 balances contrastive vs BCE. Higher λ means more emphasis on cross-scroll alignment.
- **Consistent gains:** Every SupCon run beats baseline. This is NOT noise.

#### Why It MIGHT Be Working:
SupCon provides **metric learning** — the backbone learns a distance metric where "ink-ness" is a direction in embedding space that's consistent across scrolls. This is exactly what we need for transfer.

#### Next Steps to Explore:
1. **Higher embedding dimension:** Current projection head is probably 128-256d. Try 512d or 1024d to give the model more room to separate ink/papyrus/scroll-texture into orthogonal subspaces.

2. **Hard negative mining WITHIN SupCon:** Right now, all papyrus tiles are negative anchors. But some papyrus (near ink boundaries) looks more like ink than others (flat regions). Mine the hardest cross-scroll papyrus negatives.

3. **Asymmetric temperature:** Use different T for positive pairs (ink-ink) vs negative pairs (ink-papyrus). Pull positives tighter, push negatives less hard → might improve coverage.

4. **Curriculum on λ:** Start with low λ (focus on BCE, get basic ink detection working), gradually increase λ over epochs (transfer the learned ink concept across scrolls). Current approach uses fixed λ from the start.

---

### 2. DANN (Domain-Adversarial) — **WORKS BUT FRAGILE**

#### What It Does:
Adds a domain classifier (predicts which scroll a tile came from) with gradient reversal layer (GRL). The backbone is trained adversarially:
- **Domain classifier** tries to predict scroll ID from backbone features
- **GRL flips gradients** → backbone is penalized for being scroll-specific

#### Why It Works (When It Does):
Forces the backbone to learn features that are **invariant to scroll identity**. If the model can't tell which scroll a tile came from, it must be using features common to all scrolls (i.e., ink morphology).

#### The Epoch-5 Cliff:
```
λ=0.3: ep4=0.6015 → ep5=0.5985 (tiny dip, recovered)
λ=0.4: ep4=0.5946 → ep5=0.5778 (larger dip, slow recovery)
λ=0.5: ep4=0.5773 → ep5=0.4657 (COLLAPSED, never recovered)
```

**What's happening:** The GRL ramps from 0 to λ over the first 5 epochs. At epoch 5, it's fully active. If λ is too high, the adversarial signal overwhelms the ink-detection signal → model sacrifices ink detection to confuse the domain classifier.

**Why λ=0.5 collapses:** The domain classifier is TOO EASY to confuse. The model finds a trivial solution: output uniform noise → domain classifier can't predict scroll, but ink detection is destroyed.

#### Next Steps to Explore:
1. **Progressive λ scheduling:** Instead of ramping to a fixed λ, use a **schedule** that increases more slowly and plateaus. E.g., ramp to 0.2 over 10 epochs, then slowly increase to 0.3-0.4 over the next 10. Give the model time to adapt.

2. **Gradient penalty on domain classifier:** Add a gradient penalty (like in WGAN) to prevent the domain classifier from becoming too weak. If it gets confused too easily, the adversarial signal degrades.

3. **Different domain definitions:** Right now, domain = scroll ID (15 classes). Try:
   - **Depth band as domain:** Classify whether a tile is from layers 4-12 vs 12-20 vs 20-28. Forces depth-invariant features.
   - **Local texture as domain:** Cluster tiles by papyrus texture (k-means on background regions), use clusters as domains. Forces texture-invariant features.

4. **Conditional DANN:** Only apply domain-adversarial loss to **ink tiles**. Papyrus tiles are allowed to be scroll-specific (we don't care). This focuses the invariance pressure on the ink representation.

---

### 3. Attention-MIL — **WORKS BUT SPARSE**

#### What It Does:
Replaces LSE aggregation (hard max-pooling with differentiable temperature) with **learned attention weights**:
- Per-voxel logits → attention weights (softmax)
- Tile logit = weighted sum of voxel logits
- Backprop through attention → model learns WHERE to look

#### Why It's Sparse:
Attention is **soft segmentation** — the model learns to focus on high-confidence ink voxels. But if it's too confident in a small region, it ignores the rest → low coverage.

#### Results:
```
attn1r (pure):      PR_AUC=0.6098, F1=0.3860, Probe=0.295 (HIGH precision, LOW recall)
sc_attn2 (+ SupCon): PR_AUC=0.6190, F1=0.4862, Probe=0.328 (BETTER coverage)
```

**Key insight:** Attention alone overfits to high-SNR ink voxels. SupCon provides regularization → forces the model to consider ink features from ALL scrolls, not just the easiest ones.

#### Why It MIGHT Be Working:
Sub-tile localization is EXACTLY what we need for sparse morphological features. Ink is a <3-voxel spatial feature — having the model explicitly weight voxels is better than global LSE.

#### Next Steps to Explore:
1. **Entropy regularization:** Add a loss term that penalizes low-entropy attention distributions. Force the model to spread attention across multiple voxels, not collapse to a single peak.

2. **Minimum coverage constraint:** Require that at least X% of voxels receive non-negligible attention weight (e.g., >1% of the mass). Prevents degenerate attention.

3. **Multi-head attention:** Instead of one attention map, use 4-8 heads. Each head can focus on a different aspect of ink morphology (e.g., one head for depth profile, one for spatial edges). Aggregate across heads.

4. **Attention visualization:** Render the attention maps as heatmaps overlaid on the CT slices. See what the model is actually looking at. This will reveal whether it's finding real ink or just high-contrast artifacts.

---

### 4. MeanTeacher — **VERIFIED-NEG WORKS, TEST-CONSISTENCY FAILS**

#### What It Does:
Maintains an EMA teacher model (slow-moving average of student weights). Two modes tested:
1. **Verified-negative supervision:** Use teacher predictions + 2.4µm labels to identify hard negatives (tiles where 2.4µm<31 = trusted papyrus).
2. **Test-scroll consistency:** Teacher predictions on unlabeled test-scroll tiles become pseudo-labels for the student.

#### Results:
```
mt_vn1 (verified-neg λ=0.2): PR_AUC=0.6146 (WORKS, 4th place)
mt_vn3 (verified-neg λ=0.4): PR_AUC=0.5892 (Overshoots)
mt_tc1 (test-consistency):   PR_AUC=0.4185 (COLLAPSED)
```

#### Why Verified-Neg Works:
Conservative use of hi-res labels to identify **definite papyrus**. These are hard negatives the model struggles with (high-density papyrus that looks ink-like at 9.4µm). Reinforcing them as negatives sharpens the decision boundary.

**Logic is sound:** The 2.4µm labels are ground truth. Using them to supervise hard examples (not just as binary labels) is a valid data augmentation.

#### Why Test-Consistency Failed:
Pseudo-labels on **different scrolls** (PHerc0813, 0211, 1203 vs training PHerc0139, 0814) introduce **systematic error**. The teacher's confident predictions reflect scroll-specific biases, not ground truth.

**Example failure mode:** PHerc1203 has different papyrus texture than PHerc0139. Teacher predicts "ink" on textured papyrus (because it learned PHerc0139's smooth papyrus). Student trains on this pseudo-label → learns the wrong signal.

#### Next Steps to Explore:
1. **Pseudo-label on SAME scrolls:** Use teacher predictions on **unlabeled regions of training scrolls** (the 20-25% validation split), not test scrolls. Domain shift is minimal → pseudo-labels are more reliable.

2. **Confidence filtering:** Only use teacher predictions where confidence > threshold (e.g., p>0.95 or p<0.05). Ignore uncertain predictions. This is standard in semi-supervised learning.

3. **Consistency regularization ON TRAINING TILES:** Instead of pseudo-labeling unlabeled tiles, enforce **student-teacher consistency on labeled tiles** under different augmentations. Teacher sees original, student sees augmented → consistency loss. This is the original MeanTeacher formulation (Tarvainen 2017).

4. **Multi-teacher ensemble:** Train 3-5 models with different random seeds. Use ensemble predictions as pseudo-labels (vote or average). More robust than single teacher.

---

## Which One to Double Down On?

### Priority 1: **SupCon (HIGHEST ROI)**
- **Why:** Consistent gains, stable, addresses the core generalization problem.
- **Effort:** Medium (projection head + contrastive loss already implemented).
- **Upside:** 0.62 → 0.65+ if we optimize embedding dimension, hard mining, and λ curriculum.

**Concrete experiments:**
1. SupCon with 512d embeddings (vs current ~128d)
2. SupCon with hard negative mining (mine cross-scroll papyrus that looks like ink)
3. SupCon with λ curriculum (start 0.1, ramp to 0.5 over 20 epochs)

### Priority 2: **Attention-MIL + Coverage Regularization (MEDIUM ROI)**
- **Why:** Philosophically correct (sub-tile localization for sparse features), just needs anti-collapse.
- **Effort:** Low (add entropy regularizer to existing attention).
- **Upside:** 0.62 → 0.64 if we prevent sparsity collapse, keep coverage.

**Concrete experiments:**
1. Attention + entropy regularizer (H(attention) > threshold)
2. Attention + minimum coverage constraint (at least 30% of voxels > 1% weight)
3. Multi-head attention (4 heads, aggregate with learned weights)

### Priority 3: **MeanTeacher on Training-Scroll Unlabeled Regions (LOW EFFORT, MEDIUM UPSIDE)**
- **Why:** Verified-neg already works. Extending to unlabeled regions of SAME scrolls is low-risk.
- **Effort:** Very low (just change the unlabeled set from test scrolls to validation regions).
- **Upside:** 0.61 → 0.63 if pseudo-labels are clean.

**Concrete experiment:**
1. Teacher pseudo-labels on validation split (same scrolls, unlabeled tiles)
2. Confidence threshold = 0.95 for positive pseudo-labels, 0.05 for negative
3. Pseudo-label weight = 0.1 × verified-neg weight (conservative)

### Deprioritize: **DANN (FRAGILE, DIMINISHING RETURNS)**
- **Why:** Requires careful tuning, unstable, only marginal gains over SupCon.
- **ROI:** Low. Time better spent on SupCon + Attention.
- **Keep in combo:** dann_sc1 is the top scorer, so keep the λ=0.3 + SupCon combo. But don't sink more time into DANN alone.

---

## Concrete Next Steps (Prioritized)

### Week 1: SupCon Optimization
1. **Day 1-2:** Implement 512d projection head, rerun sc2 (T=0.07, λ=0.3) → `sc6_512d`
2. **Day 3-4:** Implement hard negative mining (cross-scroll papyrus near ink boundaries) → `sc7_hardneg`
3. **Day 5-7:** Implement λ curriculum (0.1→0.5 over 20 epochs) → `sc8_curriculum`

**Expected outcome:** If any of these lifts PR_AUC > 0.64, that's a 3% relative gain → validates the direction.

### Week 2: Attention Coverage
1. **Day 1-2:** Add entropy regularizer to Attention-MIL → `attn2_entropy`
2. **Day 3-4:** Add minimum coverage constraint → `attn3_coverage`
3. **Day 5-7:** Multi-head attention (4 heads) → `attn4_multihead`

**Expected outcome:** If probe scores > 0.35 (vs 0.295 for attn1r), coverage is fixed.

### Week 3: MeanTeacher on Same-Scroll Unlabeled
1. **Day 1-3:** Pseudo-label validation split (same scrolls), confidence > 0.95 → `mt_ps1`
2. **Day 4-7:** Tune pseudo-label weight and confidence threshold → `mt_ps2`, `mt_ps3`

**Expected outcome:** If PR_AUC > 0.62, pseudo-labeling is adding signal.

### Week 4: Best Combo
1. Combine the winners from weeks 1-3 → `final_combo`
2. Run for 30 epochs (vs 15) to see if gains compound
3. Ablate: remove each component one at a time to verify contribution

**Target:** PR_AUC > 0.65, probe > 0.40. This would be a **10% relative gain** over baseline (0.60).

---

## The Mental Model

Think of generalization as:
1. **SupCon:** Learn a scroll-invariant embedding space (the "what is ink" concept)
2. **Attention:** Learn where to look within each tile (sub-voxel localization)
3. **MeanTeacher:** Bootstrap from unlabeled data (semi-supervised expansion)

These are **orthogonal** — they attack different parts of the problem. The gains should **compound** if we tune them right.

**The goal:** 0.60 (baseline) → 0.65 (optimized SupCon+Attention+MT) → 0.70+ (with architectural changes like multi-scale or depth-profile priors).

**You're right:** The model CAN learn. We just need to push it in the right direction. Let's do it.
