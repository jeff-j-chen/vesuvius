"""assemble_new_fragments.py -- download + assemble new PHerc0139 training fragments.

per fragment, in order (each step skips if its output already exists):
  1. download 9.362um surface volume (level 0)  -> ves_zarrs2/<id>.zarr + masks/<id>.png
  2. download the 1.129um-L1 ink-detection TIF  -> resize to 9.3um frame -> inklabels/<id>.png
  3. overlap check: IoU(mask_footprint, ink>0 footprint) -- both are flattenings of the same
     surface so they must co-register up to the ~4.14x L1/9.362 scale. reports the number.
  4. eroded_inklabels/<id>.png: threshold ink high + erode 3x3, AND with mask (conservative
     binary positives the trainer consumes for ring negatives + tile labels)
  5. precompute normalization stats

usage:
  python assemble_new_fragments.py --only w058          # one fragment (pilot)
  python assemble_new_fragments.py                       # all fragments
  python assemble_new_fragments.py --from w039           # resume from a fragment
  python assemble_new_fragments.py --skip-norm           # skip the (slow) norm precompute
"""
from __future__ import annotations
import argparse, os, subprocess, sys
import numpy as np
import cv2
from PIL import Image
Image.MAX_IMAGE_PIXELS = None

BUCKET = "https://vesuvius-challenge-open-data.s3.amazonaws.com"
# output zarr dir: honor $VESUVIUS_ZARR_PATH (same var config/precompute read); default is
# /vesuvius/ves_zarrs2 on linux, the local documents path on windows.
ZARR_DIR = os.getenv("VESUVIUS_ZARR_PATH",
                     "/vesuvius/ves_zarrs2" if os.name == "posix"
                     else r"C:\Users\ChenJeff\Documents\ves_zarrs2")
TMP = "_ves_tmp"

# constant across all PHerc0139 segments
VOL9_NAME = "9.362um-1.2m-113keV-volume-20250728140407.zarr"
VOL1_NAME = "1.129um-0.22m-59keV-volume-20260413113053-L1.zarr"

# (name, segment_prefix, zarr_id, ink_tif_basename)
INK = ("PHerc0139-{seg}-1.129um-0.22m-59keV-volume-20260413113053-L1-"
       "20260709123958-mrg20736-1um-s1z2-tile256-stride128.tif")
SEGMENTS = [
    # original 4 training fragments. download URLs for reproducibility; their eroded labels
    # + (ROI2-restricted) masks are already final, so ink/overlap steps are skipped for them
    # (see FRAG_OPTS). w056 was mesh-rendered (not a pre-rendered surface-volume) -- its zarr
    # already exists so step1 skips; a fresh repro of w056 needs old/render_9um_surface.py.
    ("w044", "PHerc0139/segments/20260115000000-w044_2026011522", "20260115000000"),
    ("w059", "PHerc0139/segments/20250223000000-w059_2025022312", "20250223000000"),
    ("w047", "PHerc0139/segments/20260206000001-w047_2026020613", "20260206000001"),
    ("w056", "PHerc0139/segments/20260115000001-w056_2026011514", "20260115000001"),
    # 10 new training fragments (2026-07-21)
    ("w058", "PHerc0139/segments/20260210000000-w058_2026021020", "20260210000000"),
    ("w052", "PHerc0139/segments/20260227000000-w052_2026022705", "20260227000000"),
    ("w049", "PHerc0139/segments/20260318000000-w049_20260318",   "20260318000000"),
    ("w046", "PHerc0139/segments/20260325000000-w046_20260325",   "20260325000000"),
    ("w041", "PHerc0139/segments/20260108000000-w041_2026010816", "20260108000000"),
    ("w040", "PHerc0139/segments/20250831000000-w040_2025083102", "20250831000000"),
    ("w039", "PHerc0139/segments/20260302000000-w039_2026030210", "20260302000000"),
    ("w038", "PHerc0139/segments/20260306000000-w038_2026030608", "20260306000000"),
    ("w037", "PHerc0139/segments/20260310000000-w037_2026031015", "20260310000000"),
    ("w034", "PHerc0139/segments/20260303000000-w034_2026030317", "20260303000000"),
    # HOLDOUT sanity fragment -- assembled but NOT added to DEFAULT_SCROLLS. exclusive
    # hallucination check: if inference on w055 doesn't match its 1.1um text, we hallucinated.
    ("w055", "PHerc0139/segments/20251226000000-w055_2025122611", "20251226000000"),
]

# per-fragment behaviour overrides. the original 4 already have final labels/masks
# (w059/w047/w056 masks are ROI2-restricted), so we skip ink download + overlap + eroded
# regeneration for them and only (re)ensure the surface volume + norm.
FRAG_OPTS = {
    "w044": {"skip_labels": True},
    "w059": {"skip_labels": True},
    "w047": {"skip_labels": True},
    "w056": {"skip_labels": True},
    "w055": {"holdout": True},
}

# eroded-label generation: ink prob threshold (0-255) then erosion
ERODE_THRESH = 140     # ink probability > ~0.55 = confident ink
ERODE_KSIZE  = 3
ERODE_ITERS  = 1


def run(cmd):
    print(f"  $ {' '.join(cmd)}", flush=True)
    r = subprocess.run(cmd)
    if r.returncode != 0:
        raise RuntimeError(f"command failed ({r.returncode}): {' '.join(cmd)}")


def step1_volume(name, seg, zid, workers):
    zpath = os.path.join(ZARR_DIR, f"{zid}.zarr")
    mpath = f"masks/{zid}.png"
    if os.path.exists(zpath) and os.path.exists(mpath):
        print(f"  [1/5] volume+mask exist -> skip")
        return
    url = f"{BUCKET}/{seg}/surface-volumes/{VOL9_NAME}"
    run([sys.executable, "old/download_surface_zarr.py",
         "--mode", "volume", "--level", "0",
         "--url", url, "--out-id", zid, "--out-zarr", zpath,
         "--cache-dir", os.path.join(TMP, f"dl_{zid}"), "--workers", str(workers)])


def step2_ink(name, seg, zid):
    outp = f"inklabels/{zid}.png"
    if os.path.exists(outp):
        print(f"  [2/5] inklabels exist -> skip")
        return
    import zarr
    z = zarr.open(os.path.join(ZARR_DIR, f"{zid}.zarr"), mode="r")
    H, W = z.shape[1], z.shape[2]

    ink_name = INK.format(seg=zid)
    url = f"{BUCKET}/{seg}/ink-detection/{ink_name}"
    tmp_tif = os.path.join(TMP, f"ink_{zid}.tif")
    os.makedirs(TMP, exist_ok=True)
    if not os.path.exists(tmp_tif):
        print(f"  [2/5] downloading ink TIF ...")
        run(["curl", "-fL", "--connect-timeout", "20", "--max-time", "1800",
             "--retry", "5", "--retry-delay", "3", "--retry-all-errors",
             url, "-o", tmp_tif])
    print(f"  [2/5] loading + resizing ink TIF to ({H},{W}) ...")
    ink = np.array(Image.open(tmp_tif).convert("L"))
    print(f"        ink TIF native shape={ink.shape}  scale={ink.shape[0]/H:.3f}x")
    ink9 = cv2.resize(ink, (W, H), interpolation=cv2.INTER_AREA)
    os.makedirs("inklabels", exist_ok=True)
    Image.fromarray(ink9).save(outp)
    print(f"        saved {outp}  valid(>0)={(ink9>0).mean():.3f}  mean={ink9.mean():.1f}")


def step3_overlap(name, seg, zid):
    mask = np.array(Image.open(f"masks/{zid}.png").convert("L")) > 0
    ink  = np.array(Image.open(f"inklabels/{zid}.png").convert("L"))
    inkf = ink > 0
    inter = (mask & inkf).sum()
    union = (mask | inkf).sum()
    iou = inter / max(union, 1)
    cov = inter / max(inkf.sum(), 1)   # fraction of ink footprint inside mask
    print(f"  [3/5] overlap: mask_valid={mask.mean():.3f} ink_valid={inkf.mean():.3f} "
          f"IoU={iou:.3f} ink-in-mask={cov:.3f}")
    if cov < 0.80:
        print(f"        !! WARNING low ink-in-mask coverage ({cov:.3f}) -- possible misalignment")


def step4_eroded(name, seg, zid):
    outp = f"eroded_inklabels/{zid}.png"
    if os.path.exists(outp):
        print(f"  [4/5] eroded exist -> skip")
        return
    mask = np.array(Image.open(f"masks/{zid}.png").convert("L")) > 0
    ink  = np.array(Image.open(f"inklabels/{zid}.png").convert("L"))
    binp = ((ink >= ERODE_THRESH) & mask).astype(np.uint8)
    k = np.ones((ERODE_KSIZE, ERODE_KSIZE), np.uint8)
    er = cv2.erode(binp, k, iterations=ERODE_ITERS)
    os.makedirs("eroded_inklabels", exist_ok=True)
    Image.fromarray((er * 255).astype(np.uint8)).save(outp)
    print(f"  [4/5] saved {outp}  thresh={ERODE_THRESH} valid={er.mean():.4f}")


def step5_norm(name, seg, zid, skip):
    if skip:
        print(f"  [5/5] --skip-norm -> skip")
        return
    import json
    if os.path.exists("norm_cache.json"):
        try:
            if zid in json.load(open("norm_cache.json")):
                print(f"  [5/5] norm cached -> skip")
                return
        except Exception:
            pass
    run([sys.executable, "precompute_norm.py", "--scroll-id", zid, "--zarr-path", ZARR_DIR])


def process_fragment(name, seg, zid, workers, skip_norm, prefix=""):
    """run the full assembly pipeline for one fragment. isolated in try/except so a
    single failure never kills a concurrent batch. respects FRAG_OPTS (skip_labels)."""
    opts = FRAG_OPTS.get(name, {})
    tag = f"{prefix}{name} ({zid})"
    try:
        print(f"\n{'='*70}\n{tag}  id={zid}\n{'='*70}", flush=True)
        step1_volume(name, seg, zid, workers)
        if opts.get("skip_labels"):
            print(f"  [labels] skip_labels -> keeping existing masks/inklabels/eroded")
        else:
            step2_ink(name, seg, zid)
            step3_overlap(name, seg, zid)
            step4_eroded(name, seg, zid)
        step5_norm(name, seg, zid, skip_norm)
        print(f"[done] {tag}")
        return (name, "OK")
    except Exception as e:
        import traceback
        print(f"[FAIL] {tag}: {e}")
        traceback.print_exc()
        return (name, f"FAIL: {e}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--only", type=str, default=None)
    ap.add_argument("--from", dest="from_name", type=str, default=None)
    ap.add_argument("--workers", type=int, default=16,
                    help="parallel S3 chunk-download workers PER fragment")
    ap.add_argument("--concurrent-fragments", type=int, default=1,
                    help="number of fragments to assemble in parallel (1 = sequential). "
                         "each uses --workers download threads, so total connections "
                         "= concurrent_fragments * workers; watch RAM (~2-5GB per fragment).")
    ap.add_argument("--skip-norm", action="store_true")
    args = ap.parse_args()

    segs = SEGMENTS
    if args.only:
        segs = [s for s in SEGMENTS if s[0] == args.only]
    elif args.from_name:
        names = [s[0] for s in SEGMENTS]
        segs = SEGMENTS[names.index(args.from_name):]

    cf = max(1, int(args.concurrent_fragments))
    print(f"[assemble] {len(segs)} fragment(s): {[s[0] for s in segs]}  "
          f"(concurrent_fragments={cf}, workers/frag={args.workers})")

    results = []
    if cf == 1:
        for name, seg, zid in segs:
            results.append(process_fragment(name, seg, zid, args.workers, args.skip_norm))
    else:
        from concurrent.futures import ThreadPoolExecutor, as_completed
        with ThreadPoolExecutor(max_workers=cf) as ex:
            futs = {ex.submit(process_fragment, name, seg, zid, args.workers,
                              args.skip_norm, prefix=f"[{name}] "): name
                    for name, seg, zid in segs}
            for fut in as_completed(futs):
                results.append(fut.result())

    print(f"\n{'='*70}\n[assemble] SUMMARY\n{'='*70}")
    for nm, status in results:
        print(f"  {nm}: {status}")


if __name__ == "__main__":
    main()
