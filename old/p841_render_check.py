"""decisive render test on PHerc0841 (20260221022814): render its published mesh through the test pipeline
(nearest and trilinear), align to the published surface volume, and register the renders as scrollable ids.

    python3 old/p841_render_check.py render     # download chunks + render both modes
    python3 old/p841_render_check.py align      # depth/xy alignment report + aligned zarrs, masks
    python3 old/p841_render_check.py score <checkpoint> [--shifts -1,0,1]
"""
import json
import os
import shutil
import sys

import cv2
import numpy as np
import zarr

REPO = "/vesuvius"
sys.path.insert(0, REPO)
os.chdir(REPO)

PUBLISHED = "20260221022814"
MESH = "tifxyz/p841_check_20260221022814"
VOLUME = ("https://vesuvius-challenge-open-data.s3.amazonaws.com/PHerc0841/volumes/"
          "20250821151531-9.366um-1.2m-113keV-masked.zarr/0")
VOLUME_SHAPE = (19400, 7703, 7703)
WORK = "_ves_tmp/p841check"
# synthetic numeric ids so the test notebook's helpers treat them like test fragments
IDS = {"nearest": "20260221022891", "trilinear": "20260221022892"}
if os.getenv("P841_MODES"):
    IDS = {mode: IDS[mode] for mode in os.environ["P841_MODES"].split(",")}


def raw_path(mode):
    return os.path.join(WORK, f"{mode}_raw.zarr")


def render():
    import assemble_test_segments as ats

    os.makedirs(WORK, exist_ok=True)
    for mode in IDS:
        if os.path.isdir(raw_path(mode)):
            print(f"[render] {mode}: exists")
            continue
        ats.render_surface_volume(
            MESH, os.path.join("_ves_tmp", "render_p841check"), VOLUME, VOLUME_SHAPE, layers=28,
            normal_step=1.0, upsample_factor=20.0, workers=32, out_zarr=raw_path(mode), out_id=f"{mode}_raw",
            chunk_depth=8, chunk_y=64, chunk_x=64, crop_valid=False, mask_dir=WORK,
            source_um=9.362, trilinear=(mode == "trilinear"))


def _norm(image):
    image = image.astype(np.float32)
    valid = image > 0
    image = (image - image[valid].mean()) / (image[valid].std() + 1e-6)
    image[~valid] = 0
    return image


def xy_offset(pub, ours):
    """(dy, dx) such that pub[y, x] ~ ours[y - dy, x - dx], from template matching on 4x-downsampled slices."""
    a = cv2.resize(_norm(pub), None, fx=0.25, fy=0.25, interpolation=cv2.INTER_AREA)
    b = cv2.resize(_norm(ours), None, fx=0.25, fy=0.25, interpolation=cv2.INTER_AREA)
    h, w = b.shape
    template = b[h // 4: 3 * h // 4, w // 4: 3 * w // 4]
    score = cv2.matchTemplate(a, template, cv2.TM_CCOEFF_NORMED)
    _, best, _, (x, y) = cv2.minMaxLoc(score)
    coarse = (y * 4 - (h // 4) * 4, x * 4 - (w // 4) * 4)
    # refine at full resolution around the coarse estimate
    pub_n, ours_n = _norm(pub), _norm(ours)
    best_full, refined = -2.0, coarse
    for dy in range(coarse[0] - 6, coarse[0] + 7):
        for dx in range(coarse[1] - 6, coarse[1] + 7):
            a_win, b_win = overlap(pub_n, ours_n, dy, dx)
            r = float(np.corrcoef(a_win.ravel(), b_win.ravel())[0, 1])
            if r > best_full:
                best_full, refined = r, (dy, dx)
    return refined, best, best_full


def overlap(pub, ours, dy, dx):
    """matching windows of pub and ours under pub[y, x] = ours[y - dy, x - dx] (central 50%)."""
    H, W = pub.shape
    y0, y1, x0, x1 = H // 4, 3 * H // 4, W // 4, 3 * W // 4
    return pub[y0:y1, x0:x1], ours[y0 - dy:y1 - dy, x0 - dx:x1 - dx]


def depth_profile(pub_vol, ours_vol, dy, dx, k=14):
    """corr(pub layer k, ours layer j) for every j: locates the depth correspondence."""
    a = _norm(np.asarray(pub_vol[k]))
    out = []
    for j in range(ours_vol.shape[0]):
        b = _norm(np.asarray(ours_vol[j]))
        a_win, b_win = overlap(a, b, dy, dx)
        out.append(float(np.corrcoef(a_win.ravel(), b_win.ravel())[0, 1]))
    return np.array(out)


def subvoxel_peak(profile):
    j = int(np.argmax(profile))
    if 0 < j < len(profile) - 1:
        l, c, r = profile[j - 1:j + 2]
        return j + 0.5 * (l - r) / (l - 2 * c + r)
    return float(j)


def align():
    pub = zarr.open(f"ves_zarrs2/{PUBLISHED}.zarr", mode="r")
    pub_mask = cv2.imread(f"masks/{PUBLISHED}.png", cv2.IMREAD_GRAYSCALE) > 0
    report = {}
    for mode, zid in IDS.items():
        ours = zarr.open(raw_path(mode), mode="r")
        (dy, dx), coarse_r, fine_r = xy_offset(np.asarray(pub[14]), np.asarray(ours[14]))
        profile = depth_profile(pub, ours, dy, dx)
        peak = subvoxel_peak(profile)
        shift = int(round(peak)) - 14
        print(f"[align] {mode}: xy offset dy={dy} dx={dx} (r coarse {coarse_r:.3f}, full {fine_r:.3f}); "
              f"pub layer 14 <-> ours layer {peak:.2f}  profile {np.round(profile[10:19], 3).tolist()}")
        D, H, W = pub.shape
        aligned = np.zeros((D, H, W), dtype=np.uint16)
        for k in range(D):
            j = k + shift
            if not 0 <= j < ours.shape[0]:
                continue
            layer = np.asarray(ours[j])
            ys, xs = slice(max(0, dy), min(H, layer.shape[0] + dy)), slice(max(0, dx), min(W, layer.shape[1] + dx))
            aligned[k, ys, xs] = layer[ys.start - dy:ys.stop - dy, xs.start - dx:xs.stop - dx]
        per_layer = []
        for k in range(D):
            a, b = np.asarray(pub[k]).astype(np.float32), aligned[k].astype(np.float32)
            keep = pub_mask & (a > 0) & (b > 0)
            per_layer.append(float(np.corrcoef(a[keep], b[keep])[0, 1]) if keep.sum() > 1000 else float("nan"))
        print(f"[align] {mode}: per-layer corr with published {np.round(per_layer, 3).tolist()}")
        out = zarr.open(f"ves_zarrs2/{zid}.zarr", mode="w", shape=aligned.shape, chunks=(8, 64, 64),
                        dtype="<u2", compressor=None)
        out[:] = aligned
        out.attrs.update({"p841_render_check": mode, "dy": dy, "dx": dx, "depth_shift": shift,
                          "subvoxel_peak": peak})
        mask = pub_mask & (aligned[14] > 0)
        cv2.imwrite(f"masks/{zid}.png", mask.astype(np.uint8) * 255)
        for name in ("inklabels", "dilated_inklabels", "eroded_inklabels"):
            source = os.path.join(name, f"{PUBLISHED}.png")
            if os.path.isfile(source):
                shutil.copyfile(source, os.path.join(name, f"{zid}.png"))
        report[mode] = {"dy": dy, "dx": dx, "depth_shift": shift, "subvoxel_peak": peak,
                        "per_layer_corr": per_layer}
    with open(os.path.join(WORK, "align.json"), "w") as handle:
        json.dump(report, handle, indent=2)


def recall_at_fpr(scores, labels, fpr=0.01):
    negatives = scores[~labels]
    if negatives.size == 0 or labels.sum() == 0:
        return float("nan")
    threshold = np.quantile(negatives, 1.0 - fpr)
    return float((scores[labels] > threshold).mean())


def score(checkpoint, shifts):
    from sklearn.metrics import average_precision_score, roc_auc_score

    sys.path.insert(0, os.path.join(REPO, "old"))
    import test_eval

    ns = test_eval.notebook_namespace(checkpoint)
    label = cv2.imread(f"inklabels/{PUBLISHED}.png", cv2.IMREAD_GRAYSCALE) > 0
    results = {}
    maps = {}
    for name, zid in [("published", PUBLISHED), *IDS.items()]:
        if name != "published":
            ns["ensure_fragment_ready"](int(zid))
        mask = cv2.imread(ns["mask_path_for"](int(zid)), cv2.IMREAD_GRAYSCALE) > 0
        inputs = ns["load_inputs"](ns["C"], int(zid), (0, mask.shape[0], 0, mask.shape[1]))
        for shift in shifts:
            pred = np.asarray(ns["predict_map"](ns["MODEL"], ns["C"], inputs, surface_shift=shift), np.float32)
            maps[(name, shift)] = pred
        del inputs
    # common cell grid and the cells valid in every render
    mh, mw = maps[("published", shifts[0])].shape
    cell = label.shape[0] / mh
    common = np.ones((mh, mw), bool)
    for zid in [PUBLISHED, *IDS.values()]:
        mask = cv2.imread(ns["mask_path_for"](int(zid)), cv2.IMREAD_GRAYSCALE) > 0
        common &= cv2.resize(mask.astype(np.float32), (mw, mh), interpolation=cv2.INTER_AREA) >= 0.99
    for pred in maps.values():
        common &= np.isfinite(pred)
    ink = cv2.resize(label.astype(np.float32), (mw, mh), interpolation=cv2.INTER_AREA) > 0
    print(f"[score] cell {cell:.1f}px  common cells {common.sum()}  ink cells {(ink & common).sum()}")
    reference = maps[("published", 0 if 0 in shifts else shifts[0])][common]
    for (name, shift), pred in maps.items():
        s, y = pred[common], ink[common]
        row = {
            "recall@1%fpr": recall_at_fpr(s, y),
            "roc_auc": float(roc_auc_score(y, s)),
            "pr_auc": float(average_precision_score(y, s)),
            "mean_score": float(s.mean()),
            "corr_with_published": float(np.corrcoef(s, reference)[0, 1]),
        }
        results[f"{name}{shift:+d}"] = row
        print(f"[score] {name:9s} shift {shift:+d}: " + "  ".join(f"{k}={v:.4f}" for k, v in row.items()))
    with open(os.path.join(WORK, f"score_{os.path.basename(os.path.dirname(checkpoint))}.json"), "w") as handle:
        json.dump(results, handle, indent=2)
    np.savez_compressed(os.path.join(WORK, "maps.npz"), **{f"{n}{s:+d}": m for (n, s), m in maps.items()},
                        common=common, ink=ink)


if __name__ == "__main__":
    command = sys.argv[1]
    if command == "render":
        render()
    elif command == "align":
        align()
    elif command == "score":
        shifts = [0]
        if "--shifts" in sys.argv:
            shifts = [int(v) for v in sys.argv[sys.argv.index("--shifts") + 1].split(",")]
        score(sys.argv[2], shifts)
    else:
        raise SystemExit(__doc__)
