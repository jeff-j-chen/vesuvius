# Run History and Timing Notes

Last updated: 2026-06-05

This file documents the currently available TensorBoard runs in this repository.
It is intentionally scoped to the runs present under `runs/` and is not a full experiment history.

## Data Source and Method

- Source: TensorBoard event files in `runs/*/events.out.tfevents.*`
- Metrics used here:
  - Best and final validation F1 from `P_M/F1_Score/Valid`
  - Wall-clock duration from first to last `G_M/Loss/Train` event
  - Average epoch time from wall duration divided by epoch intervals
  - Model size and selected hyperparameters from `Hyperparameters/*` tags

## Recorded Runs (Current Subset)

| Run | Epochs | Best Val F1 (epoch) | Final Val F1 | Wall Duration | Avg Epoch Time | Model Params | Pos Weight |
|---|---:|---:|---:|---:|---:|---:|---:|
| 20230702185753 | 50 | 0.340027 (17) | 0.331542 | 14.655 h | 17.945 min | 1,315,401 | 7.551881 |
| 20230827161847 | 50 | 0.330082 (17) | 0.283591 | 3.030 h | 3.710 min | 1,315,401 | 7.660508 |
| 20230827161847_recurring | 100 | 0.312111 (8) | 0.253980 | 6.176 h | 3.743 min | 1,315,401 | 7.660508 |
| full_vis_09_182125 | 50 | 0.457343 (37) | 0.426734 | 1.817 h | 2.225 min | 1,315,401 | not logged |

## Hyperparameter Snapshot (What Was Logged)

Common logged settings across this subset:

- Tile Size: 32
- Depth: 8
- Batch Size: 64
- Num Workers: 8
- Learning Rate (initial): 1e-4
- Weight Decay: 0
- L1 Lambda: 7e-6
- LR Scheduler Factor: 0.5
- Patience: 5
- Dropout: conv1=0.0, conv2=0.05, fc1=0.2, fc2=0.1
- Model Complexity: 1,315,401 parameters

Observed variation in this subset:

- Pos Weight differs by run (and is missing in `full_vis_09_182125` logs)
- Max Grad Norm is mostly 0.5, but `full_vis_09_182125` logged 1.0
- Recurring run extends to 100 epochs

Note: these runs do not include full TensorBoard HParams plugin session blobs.
Configuration visibility comes from scalar tags logged under `Hyperparameters/*`.

## Timing Guidance for Future Agents

Model size is effectively constant in these runs (about 1.3M parameters), so runtime variation is mostly data-pipeline and training-set size driven.

### Empirical timing bands from this subset

- Small/medium observed runs: about 2.2 to 3.7 min per epoch
- Large observed run: about 17.9 min per epoch

### Practical wait-time expectations

- If run speed is around 2 to 4 min/epoch:
  - 50 epochs: about 1.8 to 3.3 hours
  - 100 epochs: about 3.7 to 6.6 hours
- If run speed is around 18 min/epoch:
  - 50 epochs: about 15 hours
  - 100 epochs: about 30 hours

### Useful central estimates from this subset

- 50-epoch runs in this subset average about 6.6 hours (mean is inflated by one slow run)
- 50-epoch median-like expectation from the non-slow cluster is about 3.1 hours
- 100-epoch run observed here completed in about 6.2 hours

Interpretation:

- For quick sanity checks on smaller training slices, waiting a few hours is usually enough
- For full-size or slower I/O conditions, plan for overnight runs