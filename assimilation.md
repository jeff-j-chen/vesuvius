
We're going to exclusively focus on 20231210121321 from pherc paris4.

We have 3 forms of data:
Surface volume: https://vesuvius-challenge-open-data.s3.amazonaws.com/PHercParis4/segments/20231210121321/surface-volumes/2.4um-0.22m-78keV-volume-20260411134726.zarr/

Zarr of the surface volume. Only available in the high-res low-energy source.

Scroll zarrs:
https://vesuvius-challenge-open-data.s3.amazonaws.com/PHercParis4/volumes/20260411134726-2.400um-0.2m-78keV-masked.zarr/
https://vesuvius-challenge-open-data.s3.amazonaws.com/PHercParis4/volumes/20260323153942-2.400um-0.2m-137keV-masked.zarr/
available in high res, but low energy AND high energy.

Mesh tifxyz:
https://vesuvius-challenge-open-data.s3.amazonaws.com/PHercParis4/segments/20231210121321/mesh/20231210121321-on-20260411134726-2.4um.tifxyz/
Only available in the high-res low-energy.

64-tif-layer-stack:
https://dl.ash2txt.org/full-scrolls/Scroll1/PHercParis4.volpkg/paths/20231210121321/layers/
Unknown which fragment this comes from.

There should be a few methods to re-create our high-energy data. 
1: using the tifxyz, as is currently done for test segments. The tifxyz should be able to index the exact location. Use the assemble_test_fragments as an example - if we just use the tif to extract from the higher energy scan, that should do it!


Fetch the inputs:

the segment's mesh folder mesh/20231210121321-on-20260411134726-2.4um.tifxyz/ (x.tif, y.tif, z.tif, meta.json);
<volume>.transform.json and <volume>.metadata.json for both full-scroll scans, from PHercParis4/volumes/.
Register the two scans with no image matching. Every full-scroll scan ships a transform.json: a 3×4 affine that maps its voxels (x, y, z) into one common reference scan (here PHercParis4-20230205180739_masked). To take a point from the 78 keV scan (the mesh's scan) into the 137 keV scan, apply the 78 keV affine and then the inverse of the 137 keV affine. The combined transform is almost a pure rotation plus shift (scale 0.999–1.001).

Check overlap before downloading anything (the step I should have done first). Transform the corners of the target scan into the mesh's scan and compare with the mesh meta.json bounding box, then estimate the covered fraction of the segment mask at 16 px steps with a 256-voxel safety margin. That check reported 4.9% for this segment.

Evaluate the mesh on the existing zarr's pixel grid. The existing zarr is surface-volume level 2, so each pixel is 4 level-0 surface pixels. The mesh stores one point per 20 level-0 pixels (scale 0.05). For every zarr pixel, interpolate the mesh x/y/z and its normal from the x/y/z gradients. Points where the mesh is invalid are dropped.

Sample 28 layers along the normal. The existing zarr pooled 109 level-0 layers into 28, so each new layer sits at the centre of one of those pooling bins. Each sample point is mapped through the combined transform into the 137 keV scan and read from its level-2 pyramid (~9.6 µm) with trilinear interpolation. Level 2 matches the existing zarr's in-plane resolution, and its 4×4×4 averaging roughly matches the depth pooling.

Download only what's needed. 128³ uint8 chunks are fetched on demand into a disk cache, 404 = air, with retries on anything else.

Measure the residual misalignment:

Render 192 px test patches on a 384 px grid inside the overlap.
Depth offset: the layer shift (−2 to +2, sub-layer by parabolic fit) that maximises correlation with the existing zarr.
In-plane shift: phase correlation on the middle slice.
Result: 19 patches, correlation 0.82–0.92. Depth −0.22 ± 0.49 layers, dy +0.21 ± 0.57 px, dx +0.02 ± 0.42 px. So the published transforms alone are accurate to under a pixel, and this correction step is mostly a safety check.
Render. 256 px blocks go into ves_zarrs2/<new_id>.zarr.partial: uint16, chunks (8, 64, 64), uncompressed, same shape and pixel frame as the existing zarr. Each block applies the smoothed correction (weighted by patch correlation, 500 px smoothing radius, falling back to the global mean) and zeroes anything outside the overlap. The render was stopped at block 11 of 125.

Finish and verify (never reached). Rename the partial folder to <new_id>.zarr and write the mask from the middle slice. Then run verify_paris4_137.py, which checks:

shape and dtype;
that every depth slice is filled across the rendered footprint;
middle-slice mask overlap and Dice against the old mask (target > 99%);
per-layer correlation;
a red (new) on green (old) overlay.
Note the old-mask overlap is measured inside the rendered footprint.

For the next attempt
Pick the segment by overlap first. Take any segment's mesh bounding box (in the 78 keV scan) and keep those inside x 9.3k–17.8k, y 9.6k–18.1k, z 42.1k–48.8k (with the margin), or test the mask fraction as in step 3. Aim for over 90% before downloading anything.
The same pipeline works for any pair of scans that both have a transform.json, including coarse native scans. It gives true high-energy data at the target resolution, not an approximation.
Keep the safety rules: new ID only, write to .partial then rename, keep an eye on disk.