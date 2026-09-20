Hi all, 

Throwing my hat in the ring in case it helps anyone else. I have a repo with 1-click runpod setup via docker. The repo has scripts to download 40+ fragments scaled to the same resolution and begin training on them immediately, plus visualizations afterwards. Hooked up to tensorboard to track 100+ metrics for analysis. It is admittedly a mess right now because I'm still in the process of fast iteration, but I'm happy to answer any questions! I have no impressive results but a huge amount of findings over the past several months.

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
*MLDG* -
*SAM* - Slightly useful but not worth the performacne cost.
AGENT: YOUR TASK IS TO FINISH THIS SECTION.

**Optimal resolution** - 192x192. Anything more is not worth the tradeoff of computation time. You can downscale by 2 with basically no negative effects, just faster results. 

**Optimal depth slices** - 8. Again, not worth the tradeoff of running computation on 24. It's very beneficial for the model to select the 8 slices based on the surface - I use an precomputed elastic prediction of the surface and use that for both training and evaluation. Note: there is a lot still unknown here. I corroborated NAME HERE's result that a jitter of more than 10% (on depth 8, *a single layer*) is extremely damaging to both training and evaluation. This means that if training or prediction were run even on a single layer offset from the true surface, the results would be much much worse. I'm considering performing the final evaluation stage on every depth permutation and selecting the best result out of those. It's as of yet unclear why jitter is so damaging when my model is fed the literal surface prediction layer alongside the 8 layers surrounding it, and it knows it's absolute depth as well.

**Augmentations that work:**
*Dropout* - useful in very small quantities (0.05) for convolutional heads. Interestingly, functionless for unet skip drop (even 0.8 showed on change)
*Cutout* - useful in large quantities. 3-4 smaller cutouts is stronger than 1 huge cutout; the model can handle up to ~50% cutout.
*Context replace* - powerful augmentation that proves the model can handle it. Involves taking a negative 'donor' outer rectangle and pasting + feathering it on.  The model can handle ~50 of the non-core (64x64 center) being replaced.
*TTA Consistency* -
*Jitter* - useful, but maximum is 15%. Requires further invesigation.
AGENT: FINISH THIS SECTION


**Augmentations that don't work:**
*Anything photometric* - basically useless. The model can immediately train these out no matter how strong. 
AGENT: FINISH THIS SECTION

**Architectural Ideas That Don't Work:**
*Attention-MIL* - bad on it's own. Okay with attention entropy. Outperformed by plain LSE.
AGENT: FINISH THIS SECTION