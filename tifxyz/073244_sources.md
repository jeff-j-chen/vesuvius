# 073244 Mesh Sources

- `3_new/auto_grown_20261012073244544_abf/`: active, committed base mesh. Builds **20261012073244**, flat alias **20261012073246**. Label source: **20261012073244.png**. The top-level `auto_grown_20261012073244544_abf/` is an identical restored copy retained for inspection.
- `3_new_offset/auto_grown_20261012073244544_abf_p29/`: active, committed +29 offset mesh, restored byte-for-byte from its verified backup. Builds **20261012073249**, flat alias **20261012073250**. Independent label source: **20261012073249.png**.
- `auto_grown_20261012073244544_abf_offset_source/`: edited candidate supplied on 2026-10-12, retained for reference only. It is no longer used for either active volume after the offset regression.

Both active volumes are restored to `(28,3941,2861)`, with their matching masks, detected/flat surfaces and normalization. Each retains its own inklabels and training mask. Restoration only rotates labels to match their native frame; it does not redraw or share labels.

## Orientation Diagnosis

- The edited mesh grid was 90 degrees counterclockwise relative to the committed grid: `(144,198)` versus `(198,144)`. The renderer preserves that grid orientation; it does not add a rotation. Reverting the offset therefore turns its raw grid clockwise relative to the rebuilt version.
- The base display was accidentally changed from `np.rot90(..., 1)` to `np.rot90(..., 3)`, a 180-degree inversion. It is now restored to **90 degrees counterclockwise** from the native base grid.
- The offset uses **90 degrees clockwise** (`k=3`) from its restored native grid. Keeping that display mapping while restoring the raw grid gives the requested clockwise correction relative to the rebuilt offset figures.
- Prediction, input composite and label overlays share the same display transform. Training, labels, voxel arrays and metric computation remain in native coordinates.
- Notebook helper setup reloads the figure module so an existing kernel cannot retain an obsolete display map after setup is rerun. Existing PNGs are not rewritten automatically.

Previous artifacts are preserved under `_ves_tmp/rebuild_073244_pair/previous`, `_ves_tmp/restore_073244_original/replaced`, and `_ves_tmp/restore_073249_original/replaced`. Do not regenerate the active offset from the edited candidate.