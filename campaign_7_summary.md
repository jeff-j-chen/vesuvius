# Campaign 7 Summary

## Data Source

- Results were pulled directly from TensorBoard event files in `runs_archs7` using the project `.venv`.
- The raw extraction summaries are saved in `runs_archs7/archs7_tb_analysis_report.txt` and `runs_archs7/archs7_tb_family_summary.txt`.

## Baseline Caveat

- The named baseline for comparison was `cmp_archs7_w044_noaug_w044_no_augs_11_21-34-49`.
- Its saved config shows `n_epochs=10`, so the run ended at step 9 rather than the 12 epochs described in the current script.
- Final baseline metrics were:
  - train PR AUC: `0.90197`
  - train F1: `0.81553`
  - train loss: `2.73227`
  - valid PR AUC: `0.49908`
  - valid F1: `0.47874`
  - valid loss: `0.55441`
- It was still climbing hard on train at the end, so marginal wins over this baseline should be interpreted carefully.

## Best Single Runs

### Best raw learner

- `nnunet3d` was the strongest raw overfitter.
- Final metrics:
  - train PR AUC: `0.99905`
  - train F1: `0.99662`
  - train loss: `0.18081`
  - valid PR AUC: `0.53349`
  - valid F1: `0.26076`
  - valid loss: `0.53095`
- Interpretation:
  - It clearly exceeded the baseline on learnability and capacity.
  - It did not hold up as the best overall model because validation recall collapsed and dragged final validation F1 down.

### Best overall model

- `nonlocal` was the clearest balanced winner.
- Final metrics:
  - train PR AUC: `0.96081`
  - train F1: `0.90827`
  - train loss: `2.73045`
  - valid PR AUC: `0.55325`
  - valid F1: `0.61288`
  - valid loss: `0.55315`
- It beat the baseline on all four core score metrics at once:
  - train PR AUC
  - train F1
  - valid PR AUC
  - valid F1
- It was also still climbing on train late in the run.

### Best validation-oriented nnU-Net lifts

- `late_unet`
  - train PR AUC: `0.90560`
  - train F1: `0.83926`
  - valid PR AUC: `0.56719`
  - valid F1: `0.65042`
  - best final valid F1 in the whole sweep
- `latecollapse32`
  - train PR AUC: `0.91306`
  - train F1: `0.83944`
  - valid PR AUC: `0.58469`
  - valid F1: `0.63038`
  - best final valid PR AUC among the late binary-head variants
- Both beat the baseline on train and validation, and both were still improving late.

## Strong Baseline Beaters

These runs beat the baseline on all four core metrics: train PR AUC, train F1, valid PR AUC, and valid F1.

- `nonlocal`
- `fpn`
- `latecollapse32`
- `late_unet`
- `coord_attn`
- `depth_se`
- `convnext3d`

## What Worked

- Structured attention over preserved depth helped more than expected.
- The strongest evidence came from:
  - `nonlocal`
  - `coord_attn`
  - `depth_se`
  - `axial`
  - `deform`
- Rich dense spatial features were useful when paired with a binary late head.
- The best examples were:
  - `late_unet`
  - `latecollapse32`
- FPN-style cross-window refinement was also compatible with the binary setup.

## What Did Not Transfer Cleanly

- Simply going denser or more exotic was not enough.
- Pure transformer or object-centric radical models mostly underfit:
  - `vit3d`
  - `xcit3d`
  - `slot3d`
- The Villa-inspired depth-attention family and most normalization swaps were weak overall.
- Dual-stream variants memorized well but usually paid for it in validation PR AUC.

## Interpretation

- The useful lesson from the released dense model was not "ignore depth".
- The part that transferred was "keep richer spatial structure alive longer".
- The data still supports the older conclusion that early depth destruction is a bad idea.
- Dense spatial processing helps, but pure dense final prediction is unstable.
- The strongest direction is:
  - nnU-Net-like rich spatial features
  - a binary late decision head
  - structured depth-aware refinement such as non-local, coordinate attention, depth squeeze-excitation, or FPN-style cross-window fusion

## Best Future Direction

- Best model overall: `nonlocal`
- Best idea family overall: nnU-Net-style rich late features with a binary late head
- Best concrete follow-up targets:
  - `late_unet` plus structured refinement
  - `latecollapse32` plus structured refinement

## Extra Notes

- `efficient` and `inverted` used different configs but logged identical TensorBoard curves on the main tracked metrics, so they should be treated as one signal until their implementations are audited.
- `nnunet3d` looks more like a capacity and calibration signal than a final production winner in its current form.