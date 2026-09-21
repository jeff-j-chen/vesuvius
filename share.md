AGENT FAST INITIALIZATION

Contrary to the researchers, I've found it extremely detrimental to train with a fully dense output head (too much gradient on uncertain labels, I suspect). Additionally, due to the uncertainty of the labels, I've found it beneficial to use a 'closed ring setup' - i.e. I label the positive core of a letter, a buffer zone with no labels and no allowed gradient for the model, then a ring of negatives around it (that don't touch any other letters). I use a 64x64 center cut up into 256 4x4 tiles that each perform their own prediction. It provides a tiny fraction off the gradient but it still enough to learn. For the model itself, I use a modified nnunet3d at 192x192ds2 with an MAE pretrain for scroll texture.

**Core Issue**: I initially assumed that more ink data = better results, even across scrolls. That's the whole reason I set up my repo the way I did - it pulls from 12 different scroll sources to train on them all. However, this does not seem to hold true. For whatever reason, the model cannot generalize ink / payprus across scrolls - somehow, the differences are too big, even with an MAE pretrain performed on 40+ patches from 16 different scrolls. I'll talk about the cross-domain generalization I tried, but first, mky findings on a per-scroll basis:
*pherc0500p2* - very hard to train on, performs worse with more scrolls (aside from pherc0172, which seemed to better performance).
*pherc0172* - best to train on by far, even easier than paris1.
*pherc1667* - a very interesting case. individually can train excellently and improves results of 0500p2. However, seems an overall detriment to training, despite the excellent labels. Decreases performance across the board when it does better. When held out, models examining it predict basically entirely ink everywhere. The fragment is much of the same.
*pherc paris4* - again, stellar individual training ability.
*pherc0814* - hard to train on due to lack of data but gains from more scrolls.
*pherc0009b* - super easy to train on, but the upper section is very dubious.
*pherc0139* - near impossible to get any results in any form. 
*phercparis2_fr143* - hard to train on despite the excellent labels. seems to be a trend with the fragments.
*pherc51cr4_fr8* - an exception to the fragment rule, trains great.
*phercparis1_fr34* - another fragment hard to get results out of.
*pherc0343p* - letters are trainable but the model loves to rpedict ink everywhere.\
*pherc0841* - easy to train on. my model shows hints of ink near the 'stem', requires further investigation.

**Cross Domain Generalization Results**:
*Batch norm* - IBN > Instance norm + batchnorm >> batchnorm. Never use plain old batchnnorm.
*DANN* - basically useless in any implementation. Even with dann lambda 0.8 (or a curriculum that peaks there), offers no functional difference, because the model is unable to predict what scroll the data came from. Possibly due to the MAE pretrain, IBN, normalization - requires further investigation.
*Cross-fragment supcon* - similar case. No matter the temperature, lambda, gradient strength, curriculum, etc. it cannot force the model to generalize, and offers no performance benefit.
*GroupDRO / GroupCVAR* - Bad when performed on a per-patch basis; strong when done on a physical domain basis (i.e. patches from the same scroll get grouped).
*Character round robin sampling* - required due to the disparate amounts of data per scroll. Huge performance boost.
*GCE / Soft labeling* - Extremely detrimental to performance, do not use. (Perhaps due to my uncertain labels)
*MLDG* - Very inconsistent. Fixed holdouts were actively bad due to the scroll differences I described earlier.
*SAM* - Slightly useful but not worth the performacne cost.
*Gradient conflict weighting* - Very strong, requires further investigation.
*MixStyle / SagNet* - Neither solved the problem; style randomization is not enough when the useful ink cues themselves differ between scrolls.
*Prototype alignment / CORAL / CDAN* - All basically flat or negative.

**Optimal resolution** - 192x192. Anything more is not worth the tradeoff of computation time. You can downscale by 2 with basically no negative effects, just faster results. 

**Optimal depth slices** - 8. Again, not worth the tradeoff of running computation on 24. It's very beneficial for the model to select the 8 slices based on the surface - I use an precomputed elastic prediction of the surface and use that for both training and evaluation. Note: there is a lot still unknown here. I corroborated NAME HERE's result that a jitter of more than 10% (on depth 8, *a single layer*) is extremely damaging to both training and evaluation. This means that if training or prediction were run even on a single layer offset from the true surface, the results would be much much worse. I'm considering performing the final evaluation stage on every depth permutation and selecting the best result out of those. It's as of yet unclear why jitter is so damaging when my model is fed the literal surface prediction layer alongside the 8 layers surrounding it, and it knows it's absolute depth as well.

**Augmentations that work:**
*Dropout* - useful in very small quantities (0.05) for convolutional heads. Interestingly, functionless for unet skip drop (even 0.8 showed on change)
*Cutout* - useful in large quantities. 3-4 smaller cutouts is stronger than 1 huge cutout; the model can handle up to ~50% cutout.
*Context replace* - powerful augmentation that proves the model can handle it. Involves taking a negative 'donor' outer rectangle and pasting + feathering it on.  The model can handle ~50 of the non-core (64x64 center) being replaced.
*TTA Consistency* - useful for hallucination protection and used in the final eval figures, but the performance tradeoff is too steep.
*Jitter* - useful, but maximum is 15%. Requires further invesigation.
*Flip / rotation* - reliable because these are exact symmetries of the data. One important implementation trap: the image, multitile labels and validity mask must all receive the exact same transform.
*Context jitter* - useful when it moves the 64x64 prediction target inside the 192x192 context rather than moving only the image.


**Augmentations that don't work:**
*Anything photometric* - basically useless. The model can immediately train these out no matter how strong. 
*FDA / frequency mixing* - no meaningful improvement. It changes superficial style statistics without creating a more transferable ink cue.
*Elastic deformation* - no useful gain, hard to implement.
*Depth warp / surface attenuation* - neither beat the baseline. Smoothly bending the depth stack or weakening contrast around the surface sounds physically motivated, but did not improve generalization.
*Acquisition blur / correlated noise* - both were approximately flat. Simulating scanner defects did not reproduce the actual differences between scroll domains.
*Depth-view / context consistency* - expensive paired-forward regularizers with no durable improvement. The simpler one-slice depth jitter and context replacement were stronger.

**Architectural Ideas That Don't Work:**
*Attention-MIL* - bad on it's own. Okay with attention entropy. Outperformed by plain LSE.
*Dense U-Net output* - consistently disastrous with these uncertain transferred labels. Likely because I hand-draw my labels to go with my ring datasets, rather than using the researchers' perfect ones?
*Global average / hard max pooling* - average pooling dilutes a sparse stroke, while hard max pooling lets the network satisfy a positive label with one spurious voxel. LSE or sparse multitile pooling is much more stable.
*Early 3D-to-2D / factorized (2+1)D / depth-attention head* - none beat the matched 3D baseline. Collapsing to 2D in the middle was the exception; it repeatedly improved difficult domains and survived the expanded-data rerun.
*Residual U-Net / MedNeXt / divided attention* - extra capacity and larger kernels did not produce a broad transfer gain. Some arms raised one metric or one scroll, but none were reliable enough to keep.
*WELDON / minimum-support / CLAM* - occasionally promising, especially in combinations on Campaign 27, but the gains did not reproduce cleanly after adding more fragments. The larger stacks mostly added interactions rather than a universal improvement.
*Dual-scale by itself* - useful on particular scrolls, but not a general win. The local branch often moved performance between domains instead of lifting all of them.

**Current winner:** The raw character-F1 winner is gradient conflict at .5905. The model I would actually use is mid-3D/2D + gated stems: .5865 character F1, the best global PR-AUC (.6762), the best 12-domain mean (.7004), and the best new-five mean (.7230). Adding dual-scale + physical GroupDRO raises F1 slightly to .5882 but lowers global PR-AUC to .6725 and the domain mean to .6856, so it is not the best balanced model.

**Methodology:**
1. *Download and standardize the volumes* - the assembly scripts fetch flattened surface volumes from S3 or dl.ash2txt. Native 7.91um, 8.64um, 3.24um and 2.4um sources are resampled in XYZ to a common ~9.362um grid with 28 depth layers. The current supervised campaign contains 17 fragments grouped into 12 physical scroll domains; the wider download/pretraining pool contains 40+ fragments. Volumes are stored as chunked uint16 zarrs, with a matching papyrus mask for every fragment.
2. *Normalize each fragment* - I calculate one mean and standard deviation over all voxels inside that fragment's papyrus mask, cache them in `norm_cache.json`, and z-score every crop with those per-volume statistics. The implementation then maps the observed normalized min/max to [0,1]. This removes scanner scale differences without normalizing each patch independently and erasing local contrast.
3. *Predict the surface* - for every XY column I estimate the papyrus-to-air transition inside the 28-layer volume, spatially regularize it and save a depth map plus confidence map. Training reads eight contiguous slices centered on that literal local surface. I allow only +/-1 slice of depth jitter; larger offsets are extremely destructive.
4. *MAE pretrain* - before supervised training I pretrain the matching nnU-Net backbone on unlabeled papyrus crops. I mask 65% of 4x4 spatial blocks through the entire depth column, replace them with the visible mean, and reconstruct only the masked positions with MSE. Scrolls are sampled round-robin. Architecture-changing variants such as full IBN and mid-3D/2D receive their own matched MAE adaptation rather than pretending incompatible weights are equivalent.
5. *Prepare conservative supervision* - high-resolution ink predictions are aligned/resampled to the 9.362um surface and manually cleaned where possible. I train from the positive cores, close small holes, leave an unsupervised gap, then take a negative shell around each character. Current geometry is close radius 2, gap 2 and shell 4. In a positive multitile window only positive cells receive loss; negative supervision comes from separate ink-free ring windows.
6. *Sample characters and scrolls evenly* - connected ink components are treated as characters. Sampling round-robins characters within each physical scroll and physical scrolls across the corpus, rather than allowing a large fragment or large letter to dominate. PHerc0139 is deliberately repeated four times per cycle; the other physical domains receive one turn each.
7. *Build the input and target* - each example reads an 8x192x192 surface-relative crop, then downsamples XY by two. The stem receives raw intensity, local contrast and depth-gradient cues. Only the central 64x64 region is predicted, split into sixteen 16x16 targets. This gives much more localized gradient than one MIL score without trusting a dense pixel mask.
8. *Augment* - synchronized flips and rotations handle exact geometric symmetries. Three small cutouts, target-aware context jitter, feathered replacement of distant context with real negative papyrus, and +/-1 depth jitter attack the shortcuts that actually appeared in validation. Photometric noise, FDA and elastic warps are off.
9. *Train* - the current baseline is a full-IBN 3D nnU-Net initialized from MAE, using hard-label BCE, AdamW, mixed precision, gradient clipping and a warmup/plateau learning-rate schedule. The best balanced variant collapses depth midway through the network and uses learned gates over the raw/LCN/depth-gradient stems. DANN, SupCon, soft labels and GCE are disabled.


**Reproduce My (Shoddy) Results**:
1. On Runpod: Select template `jeffchen23/vesuvius`, or manually clone from dockerhub. Use all default values. `cd /vesuvius` (the working directory)
2. If on a newer gpu (e.g. 5090): run `pip install "torch>=2.7" --index-url https://download.pytorch.org/whl/cu128`. Ignore any errors.
3. Update repo via `git pull --depth=1` and `git reset --hard origin/main`
4. Get training frags: run `python3 assemble_training_segments.py`
5. Train: run ` python3 train.py -n "reproduction"`
6. Get results: `tensorboard --logdir ./runs --bind_all`
7. For further visualizations: Use `eval_scrolls.ipynb` to see how the model does on all training fragments