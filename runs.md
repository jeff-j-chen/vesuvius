# Run History and Timing Notes

Last updated: 2026-06-08

## Windows Dataloader Benchmark Notes (2026-06-08)

Machine context:

- GPU: NVIDIA RTX PRO 5000 Blackwell Generation Laptop GPU (24,463 MiB VRAM)
- CPU: Intel Core Ultra 9 285HX (24 logical cores)

Root-cause notes for failures seen on Windows:

- `num_workers >= 4` repeatedly failed with multiprocessing spawn errors (`OSError: [Errno 22] Invalid argument` and `_pickle.UnpicklingError: pickle data was truncated`)
- `batch_size=160` with `num_workers=2` failed with CUDA illegal memory access during AMP optimizer step

### 1-epoch screening timings

| Config | Time (min) | Result |
|---|---:|---|
| bs=128, nw=0 | 12.05 | stable |
| bs=96, nw=2 | 4.58 | stable |
| bs=128, nw=2 | 4.62 | stable |
| bs=128, nw=4 | 0.82 | failed (worker spawn) |
| bs=160, nw=4 | 0.78 | failed (worker spawn) |
| bs=192, nw=4 | 0.77 | failed (worker spawn) |

### 3-epoch confirmation timings

| Config | Time (min) | Result |
|---|---:|---|
| bs=96, nw=2 | 13.24 | stable |
| bs=128, nw=2 | 13.72 | stable |
| bs=128, nw=3 | 13.90 | stable |
| bs=160, nw=2 | 1.34 | failed (cuda illegal memory access) |

Selected config for long sanity rerun:

- `batch_size=96`
- `num_workers=2`

## Active Run (Windows Sanity Fast50)

- Status: running
- Start date: 2026-06-08
- Run name prefix: sanity_20230827161847_win_fast50
- TensorBoard run dir: runs/sanity_20230827161847_win_fast50_08_13-28-39
- Command:
  - $env:MPLBACKEND='Agg'; c:/Users/ChenJeff/Documents/vesuvius/.venv/Scripts/python.exe train.py -n sanity_20230827161847_win_fast50 --epochs 50 --scroll-id 20230827161847 --scroll4-id 20230827161847 --zarr-path .\ves_zarrs2 --batch-size 96 --num-workers 2
- Purpose:
  - 50-epoch sanity rerun that will hit test/evaluation on epoch 50 (`test_int=50`)
- Current progress snapshot:
  - passed setup and entered epoch 1 with active training progress on Windows

This file documents the currently available TensorBoard runs in this repository.
It is intentionally scoped to the runs present under `runs/` and is not a full experiment history.

## Active Run (Windows Sanity)

- Status: completed
- Start date: 2026-06-05
- End date: 2026-06-06
- Run name prefix: sanity_20230827161847_win
- TensorBoard run dir: runs/sanity_20230827161847_win_05_20-39-43
- Command:
  - $env:MPLBACKEND='Agg'; c:/Users/ChenJeff/Documents/vesuvius/.venv/Scripts/python.exe train.py -n sanity_20230827161847_win --epochs 40 --scroll-id 20230827161847 --scroll4-id 20230827161847 --zarr-path .\\ves_zarrs2 --num-workers 0
- Purpose:
  - 40-epoch sanity training on small scroll segment on Windows after environment/data migration
- Outcome:
  - completed all 40 epochs (final step 39 in TensorBoard scalars)
  - best valid F1: 0.353205 at epoch 17
  - final valid F1: 0.271733 at epoch 40
  - final valid loss: 1.099923
  - final train loss: 1.154560
  - training wall time: 5.41 h
- Notes:
  - uses CUDA-enabled torch in local .venv (torch 2.11.0+cu128)
  - scroll4 id is temporarily set to the same small segment for this sanity pass so visualizer init does not depend on incomplete 202312 extraction
  - first Windows launch failed due invalid ':' in TensorBoard logdir timestamp; visualizer timestamp format was made Windows-safe
  - second launch failed during validation multiprocessing spawn on Windows; rerun uses num_workers=0 for stability
  - later rerun failed at epoch 8 with tkinter/tcl thread teardown (`Tcl_AsyncDelete`); visualizer now forces matplotlib backend `Agg` and relaunch sets `MPLBACKEND=Agg`
  - run produced `model_epoch_40.pth` and updated `hard_negs/hard_mining_epoch_39.jsonl`

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



## Automated Campaign readability_2026_06_08

This section is auto-updated by campaign_runner.py.
Each test entry includes what changed, why, expected result, observed result, and next planned run based on results.


### Test 01: cmp_readability_2026_06_08_t01_baseline_probe1
- started_at: 2026-06-09 05:39:56 UTC
- status: started
- changed: epochs=30, scroll-id=20230827161847, scroll4-id=20230827161847, batch-size=96, num-workers=2, probe-int=1, eval-int=10, test-int=30
- why: baseline with per-epoch probe metrics for fast readability monitoring
- expected: stable baseline for readability composite and probe trends
- next_planned_based_on_results: pending completion
- note: run logged before scroll4 override fix, so scroll4-id points to small scroll 1 in this run

- status: completed
- run_dir: C:\Users\ChenJeff\Documents\vesuvius\runs\cmp_readability_2026_06_08_t01_baseline_probe1_08_22-40-20
- results: valid_f1_last=0.2351994812488556, readability_last=0.3715526759624481, probe_easy_last=0.5491999983787537, probe_hard_last=0.35394230484962463
- next_planned_based_on_results: 02:t02_no_channel_mix


### Test 02: cmp_readability_2026_06_08_t02_no_channel_mix
- started_at: 2026-06-09 11:24:44 UTC
- status: started
- changed: epochs=30, scroll-id=20230827161847, scroll4-id=20230827161847, batch-size=96, num-workers=2, probe-int=1, eval-int=10, test-int=30, channel-mixing-prob=0.0
- why: remove depth permutation which can break physical depth cues
- expected: better local contrast and less spill compared with baseline
- next_planned_based_on_results: pending completion
- note: run logged before scroll4 override fix, so scroll4-id points to small scroll 1 in this run

- status: completed
- run_dir: C:\Users\ChenJeff\Documents\vesuvius\runs\cmp_readability_2026_06_08_t02_no_channel_mix_09_04-25-07
- results: valid_f1_last=0.2671343684196472, readability_last=0.39386186003685, probe_easy_last=0.5472447872161865, probe_hard_last=0.34306973218917847
- next_planned_based_on_results: 03:t03_low_channel_mix



### Test 03: cmp_readability_2026_06_08_t03_low_channel_mix
- started_at: 2026-06-09 19:40:55 UTC
- status: started
- changed: epochs=30, scroll-id=20230827161847, scroll4-id=20231210132040, batch-size=96, num-workers=2, probe-int=1, eval-int=10, test-int=30, channel-mixing-prob=0.1
- why: test partial channel mixing as lighter regularization
- expected: middle ground between baseline and no channel mixing
- next_planned_based_on_results: pending completion

- status: completed
- run_dir: C:\Users\ChenJeff\Documents\vesuvius\runs\cmp_readability_2026_06_08_t03_low_channel_mix_09_12-41-12
- results: valid_f1_last=0.24227124452590942, readability_last=0.3582670986652374, probe_easy_last=0.5383473038673401, probe_hard_last=0.3090948462486267
- next_planned_based_on_results: 04:t04_pool_max


### Test 04: cmp_readability_2026_06_08_t04_pool_max
- started_at: 2026-06-10 00:31:01 UTC
- status: started
- changed: epochs=30, scroll-id=20230827161847, scroll4-id=20231210132040, batch-size=96, num-workers=2, probe-int=1, eval-int=10, test-int=30, pooling=max
- why: test sparse-evidence pooling instead of averaging
- expected: sharper positives and stronger local ranking
- next_planned_based_on_results: pending completion
