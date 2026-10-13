Methodology
Architecture

A 3D nnU-Net encoder with an 8 slice depth, fitted to the estimated sheet surface. Has a triple gated stem for raw input, local contrast normalized, and dl/dZ (depth gradient). It collapses early after the first encoder and runs the rest of the network in 2D. 
The model additionally receives information through a fiber coordinate branch and surface teacher input at enc1. Sheets are normalized by z-score, batch inputs are normalized with full instance batchnorm. Output is multitile rather than dense; it outputs 16 4x4 tiles at the center of the 96x96 provided context.

Labels & Loss

Each label is given a ‘ring’: a morphological close + dilation gives the model room to breath. There is a positive core and a negative ring outside; the center does not contribute gradient in any way. BCE loss with no smoothing and no reweighting. Edges are blurred with sigma 8 and floored at 0.55. Bag ranking forces cells to score above negatives. Labels are derived from the researcher inklabels (I’ve relabelled some in very specific cases). 
Domain

Physical patch groupDRO reweights towards the worst patches. RSC with 33% forces the model to not rely on the most loud predictions. Domains are weighted by number of samples and difficulty: pherc0139 ×2, pherc0500p2 ×3, pherc0343p ×4, pherc0814 ×5, all others ×1. Character balanced sampling grabs from each scroll in a round-robin to ensure even training.

Regularization & Augmentation
Dropout: 0.2 on conv1 and conv2, 0.3 on the head
Skip drop: each whole skip connection is dropped with probability 0.2 (likely no effect)
Rotation 0.6, flip 0.6
Cutout 0.2 (the prediction centre protected)
Context replace 0.15: the surrounding context is swapped for another window's, aligned on the surface (margin 7, feather 13).
Context jitter: up to 10 px, target-aware.
Depth jitter: ±1 slice.
Training
Initial: 10 epochs, batch 96, lr1e-4, 5 warmup epochs. Gradient clipping at 0.5, no weight decay. Trained on the full domain shown.
Fine-tune: frozen enc1/enc2, 5e-5 decoder/head. 8 epochs, 5 warmup epochs.
Tuned on pherc1447 w058+w060, pherc0139 w035+w044, pherc500p2, pherc0009b, pherc0211.


Notes:
Future plans: 
(in-progress) Train Youssef’s model with my inklabels and see if it shows the same results.
Grow other patches from Pherc.0211 where letterforms are found. See if there are more nearby, and see if they improve detection when trained on.
Train the base model to predict less ink as a whole.



