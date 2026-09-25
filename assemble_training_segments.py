"""assemble_training_segments.py -- download + assemble training segments.

Most inputs are native 9.362um surfaces. Configured exceptions are resampled in XYZ:
7.91um and 8.64um isotropic volumes are interpolated to the target grid, while 2.4um
volumes use level-2 XY and pool 109 native depth samples to 28 output layers.

per fragment, in order (each step skips if its output already exists):
  1. download 9.362um surface volume (level 0)  -> ves_zarrs2/<id>.zarr + masks/<id>.png
    2. verify that the repository-provided eroded label is present; inklabels are never fetched
  3. precompute normalization stats

Label files are repository-owned inputs. This script never downloads or rewrites them.
Use old/ink_shrinker.py to derive top-level inklabels from conservative targets.

usage:
  python assemble_training_segments.py --only w058          # one fragment (pilot)
  python assemble_training_segments.py                       # all fragments
  python assemble_training_segments.py --from w039           # resume from a fragment
  python assemble_training_segments.py --skip-norm           # skip the (slow) norm precompute
"""
from __future__ import annotations
import argparse, json, os, shutil, subprocess, sys, time
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor
import cv2
import numpy as np
from PIL import Image
Image.MAX_IMAGE_PIXELS = None

BUCKET = "https://vesuvius-challenge-open-data.s3.amazonaws.com"
# output zarr dir: honor $VESUVIUS_ZARR_PATH (same var config/precompute read); default is
# /vesuvius/ves_zarrs2 on linux, the local documents path on windows.
def _default_zarr_dir():
    if os.name != "posix":
        return r"C:\Users\ChenJeff\Documents\ves_zarrs2"
    # distinguish desktop (external Seagate) from runpod
    if os.path.exists("/media/jeff/Seagate/"):
        return "/media/jeff/Seagate/ves_zarrs2"
    return "/vesuvius/ves_zarrs2"
ZARR_DIR = os.getenv("VESUVIUS_ZARR_PATH", _default_zarr_dir())
TMP = "_ves_tmp"

# constant across all PHerc0139 segments
VOL9_NAME = "9.362um-1.2m-113keV-volume-20250728140407.zarr"

# (name, segment_prefix, zarr_id)
SEGMENTS = [
    # original 4 training fragments. their eroded labels + masks are already final,
    # so skip_labels=True keeps those and only (re)fetches the surface volume + norm.
    # w056 was mesh-rendered (not a pre-rendered surface-volume) -- its zarr already
    # exists so step1 skips; a fresh repro of w056 needs old/render_9um_surface.py.
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
    # PHerc0139 w035 (2026-08-12). inklabels downloaded via download_w035_labels.py.
    ("w035", "PHerc0139/segments/20260317000000-w035_2026031718", "20260317000000"),
    # PHerc0139 segments with official researcher ink labels (2026-09-18 release)
    ("w030", "PHerc0139/segments/20250108000005-w030_2025010818", "20250108000005"),
    ("w043", "PHerc0139/segments/20260112000000-w043_2026011217", "20260112000000"),
    ("w045", "PHerc0139/segments/20260126000000-w045_2026012619", "20260126000000"),
    # HOLDOUT sanity fragment -- assembled but NOT added to DEFAULT_SCROLLS. exclusive
    # hallucination check: if inference on w055 doesn't match its 1.1um text, we hallucinated.
    ("w055", "PHerc0139/segments/20251226000000-w055_2025122611", "20251226000000"),
    # PHerc0814 segment (2026-07-22). DIFFERENT scroll -> different surface-volume name
    # (raw vol 20250804134230, see FRAG_OPTS['vol9_name']). only mask + eroded label exist
    # (no non-eroded 1um ink), so skip_labels keeps those and just fetches the zarr + norm.
    ("seg46527", "PHerc0814/segments/20260226000000-46527_2um_try2", "20260226000000"),
    # PHerc0500P2 front segment (2026-08-07). DIFFERENT scroll, same resolution/energy as
    # PHerc0139 (9.362um/113keV/1.2m). crystal-clear 2.215um inklabels available, making
    # this a high-quality training fragment from a new scroll domain.
    ("500P2_front", "PHerc0500P2/segments/20250628074500-500P2_front", "20250628074500"),
    # PHerc1667 w013 (2026-08-13). assembled by the special level-2 path below (surface zarr
    # at 2.399um, z-pooled 109->28 layers, left 25% crop). shape: (28, 10400, 4975) ~9.6um.
    ("w013", "PHerc1667/segments/20240304141531-w013_20240304141531_flatboi", "20240304141531"),
    # PHerc0172 / Scroll 5: native 7.91um volumes resampled in XYZ to the 28-layer 9.362um frame.
    ("w087", "PHerc0172/segments/20251112000002-w087_20251112000002214_flatboi", "20251112000002"),
    ("w068", "PHerc0172/segments/20251111010954-w068_20251111010954408_flatboi", "20251111010954"),
    # PHerc1667 w018: full 2.399um flattening, using level 2 at 9.596um in XY.
    ("w018", "PHerc1667/segments/20240304144031-w018_20240304144031_flatboi", "20240304144031"),
    # PHerc0009B: native 8.64um volumes resampled in XYZ to the training frame.
    ("p9b_487", "PHerc0009B/segments/20250919125754-auto_grown_20250919055754487_inp_hr", "20250919125754"),
    # PHercParis4: level-2 XY is 9.6um; pool 109 source depths to 28.
    ("paris4", "PHercParis4/segments/20231210121321", "20231210121321"),
    # dl.ash2txt fragments: native 3.24um surface TIFF stacks resampled in XYZ.
    ("paris2_fr143", "", "20230301213755"),
    ("paris2_fr47", "", "20230205142449"),
    ("scroll6_fr8", "", "20231205222200"),
    ("paris1_fr34", "", "20230301213423"),
    ("p343", "PHerc0343P/segments/20250511003658-tifxyz", "20250511003658"),
    ("cr1fr3", "", "20231201215900"),
    ("p841", "PHerc0841/segments/20260221022814-auto_grown_20260220174252405", "20260221022814"),
]

# per-fragment behaviour overrides. skip_labels=True skips the eroded label check
# (for fragments whose labels are already verified present and final).
FRAG_OPTS = {
    "w044": {"skip_labels": True},
    "w059": {"skip_labels": True},
    "w047": {"skip_labels": True},
    "w056": {"skip_labels": True},
    "w055": {"holdout": True},
    # PHerc0814: its surface volume is a different raw scan than the PHerc0139 constant,
    # and its labels are already final (eroded only), so keep them and only fetch the zarr.
    "seg46527": {"skip_labels": True,
                 "vol9_name": "9.362um-1.2m-113keV-volume-20250804134230.zarr"},
    # PHerc0500P2: different raw volume name from PHerc0139 constant.
    # inklabels will be generated separately (download_p500p2_labels.py) from the
    # 2.215um ink detection TIF and saved to inklabels/ and eroded_inklabels/.
    "500P2_front": {"vol9_name": "9.362um-1.2m-113keV-volume-20250820143440.zarr"},
    # PHerc1667 w013: assembled via a dedicated script (pre-rendered level-2 surface zarr,
    # z-pooled to 28 layers). skip_labels=True because inklabels are generated by that script.
    "w013": {"skip_labels": True, "w013_special": True},
    "w087": {
        "vol9_name": "7.91um-53keV-volume-20241024131838.zarr",
        "resample_um": 7.91,
        "force_norm": True,
    },
    "w068": {
        "vol9_name": "7.91um-53keV-volume-20241024131838.zarr",
        "resample_um": 7.91,
        "force_norm": True,
    },
    "w018": {
        "pooled_special": True,
        "surface_name": "2.399um-0.22m-78keV-volume-20251217075048.zarr",
        "surface_level": 2,
        "surface_expected_shape": (109, 10595, 24525),
        "output_dtype": "|u1",
        "force_norm": True,
    },
    "p9b_487": {
        "vol9_name": "8.64um-1.2m-116keV-volume-20250521125136.zarr",
        "resample_um": 8.64,
        "force_norm": True,
    },
    "paris4": {
        "pooled_special": True,
        "surface_name": "2.4um-0.22m-78keV-volume-20260411134726.zarr",
        "surface_level": 2,
        "surface_expected_shape": (109, 12750, 9995),
        "force_norm": True,
    },
    # the four fragments below are rendered from their 88 keV scan (see RESCAN_88KEV / utils/fragment_rescan.py)
    "paris2_fr143": {
        "dlash_surface_base": "https://dl.ash2txt.org/fragments/Frag2/PHercParis2Fr143.volpkg/working/54keV_exposed_surface",
        "surface_expected_shape": (65, 14830, 9506),
        "source_um": 3.24,
        "force_norm": True,
    },
    # id = uuid of the 54 keV volume the exposed surface was traced on
    "paris2_fr47": {
        "dlash_surface_base": "https://dl.ash2txt.org/fragments/Frag1/PHercParis2Fr47.volpkg/working/54keV_exposed_surface",
        "surface_expected_shape": (65, 8181, 6330),
        "source_um": 3.24,
        "force_norm": True,
    },
    "scroll6_fr8": {
        "dlash_surface_base": "https://dl.ash2txt.org/fragments/Frag6/PHerc51Cr4Fr8.volpkg/working/PHerc0051Cr04Fr08_53keV_3.24um/surface_processing",
        "surface_expected_shape": (65, 8853, 6205),
        "source_um": 3.24,
        "force_norm": True,
    },
    "paris1_fr34": {
        "dlash_surface_base": "https://dl.ash2txt.org/fragments/Frag3/PHercParis1Fr34.volpkg/working/54keV_exposed_surface",
        "surface_expected_shape": (65, 7606, 5249),
        "source_um": 3.24,
        "force_norm": True,
    },
    "p343": {
        "vol9_name": "8.64um-1.2m-116keV-volume-20250521134555.zarr",
        "resample_um": 8.64,
        "surface_expected_shape": (31, 3440, 2060),
        "force_norm": True,
    },
    "cr1fr3": {
        "dlash_surface_base": "https://dl.ash2txt.org/fragments/Frag5/PHerc1667Cr1Fr3.volpkg/working/PHerc1667Cr01Fr03_70keV_3.24um/surface_processing",
        "surface_expected_shape": (65, 7309, 4560),
        "source_um": 3.24,
        "force_norm": True,
    },
    "p841": {
        "cropped_surface_url": "https://vesuvius-challenge-open-data.s3.amazonaws.com/PHerc0841/segments/20260221022814-auto_grown_20260220174252405/surface-volumes/9.366um-1.2m-113keV-volume-20250821151531.zarr/0",
        "surface_expected_shape": (28, 21560, 12260),
        "surface_crop": (14656, 21560, 0, 5248),
        "force_norm": True,
    },
}

# default source for these dl.ash2txt fragments: the 88 keV scan, sampled through the low-energy exposed-surface
# PPM (<dlash_surface_base>/result.ppm) into the same frame as the old surface-tiff assembly. affine = fitted
# low-energy voxel xyz -> 88 keV voxel xyz (3x4; no published registration). tif_url rebuilds chunks that are
# missing from the published 88 keV zarr. an existing zarr without the rescan marker is superseded.
_DLASH = "https://dl.ash2txt.org/fragments"
RESCAN_88KEV = {
    "paris1_fr34": {
        "volume_url": f"{_DLASH}/Frag3/PHercParis1Fr34.volpkg/volumes_zarr/88keV_3.24um_.zarr/0",
        "tif_url": f"{_DLASH}/Frag3/PHercParis1Fr34.volpkg/volumes/20230212182547",
        "tif_digits": 4,
        "affine": [[1.00028, 0.00101, -0.00023, -72.77],
                   [0.00012, 1.00301, 0.00016, 7.97],
                   [0.00003, 0.00060, 0.99997, -4.65]],
    },
    "scroll6_fr8": {
        "volume_url": f"{_DLASH}/Frag6/PHerc51Cr4Fr8.volpkg/volumes_zarr/88keV_3.24um_.zarr/0",
        "tif_url": f"{_DLASH}/Frag6/PHerc51Cr4Fr8.volpkg/volumes/20231201112849",
        "tif_digits": 4,
        "affine": [[1.00056, 0.00038, -0.00008, -1.52646],
                   [-0.00008, 0.99994, 0.00144, -0.32086],
                   [-0.00001, -0.00145, 1.00063, -0.61886]],
    },
    "paris2_fr143": {
        "volume_url": f"{_DLASH}/Frag2/PHercParis2Fr143.volpkg/volumes_zarr/88keV_3.24um_.zarr/0",
        "tif_url": f"{_DLASH}/Frag2/PHercParis2Fr143.volpkg/volumes/20230226143835",
        "tif_digits": 5,
        "affine": [[1.00019, 0.00239, -0.00006, 15.74067],
                   [-0.00062, 0.99700, 0.00027, -211.31795],
                   [-0.00006, 0.00182, 0.99952, 4.91978]],
    },
    "paris2_fr47": {
        "volume_url": f"{_DLASH}/Frag1/PHercParis2Fr47.volpkg/volumes_zarr/88keV_3.24um_.zarr/0",
        "tif_url": f"{_DLASH}/Frag1/PHercParis2Fr47.volpkg/volumes/20230213100222",
        "tif_digits": 4,
        "affine": [[1.00032, -0.00001, 0.00011, 150.21086],
                   [-0.00067, 1.00025, 0.00029, 251.15753],
                   [0.00000, -0.00015, 0.99998, 6.11317]],
    },
}
RESCAN_MARKER = "rescan_88kev"

TARGET_VOXEL_UM = 9.362
TARGET_DEPTH = 28

# PHerc1667 w013 is a pre-rendered 2.399um volume. Level 2 is ~9.596um in XY;
# pooling 109 source depths to 28 gives ~9.34um in Z. Only the left 25% is useful.
W013_SURF_BASE = (
    f"{BUCKET}/PHerc1667/segments/20240304141531-w013_20240304141531_flatboi/"
    "surface-volumes/2.399um-0.22m-78keV-volume-20251217075048.zarr"
)
W013_SURF_LEVEL = 2
W013_SOURCE_SHAPE = (109, 10400, 19900)
W013_OUTPUT_SHAPE = (28, 10400, 4975)

CHUNK_XY = 128     # S3 surface volume source chunks are 128x128

# OUTPUT zarr chunks optimized for arbitrary 192px contexts and 8-slice windows
DEFAULT_CHUNK_DEPTH = 8   # matches 8-slice depth windows in triple mode
DEFAULT_CHUNK_Y = 64      # balances arbitrary 192px context reads and chunk over-read
DEFAULT_CHUNK_X = 64


def run(cmd):
    print(f"  $ {' '.join(cmd)}", flush=True)
    r = subprocess.run(cmd)
    if r.returncode != 0:
        raise RuntimeError(f"command failed ({r.returncode}): {' '.join(cmd)}")


def _pbar(total, desc):
    """progress reporter: a real tqdm bar if tqdm is importable, else periodic percent prints.
    returns (update, close) callables so the caller doesn't care which backend is used."""
    try:
        from tqdm import tqdm
        bar = tqdm(total=total, desc=f"[dl] {desc}", unit="it")
        return (lambda n=1: bar.update(n)), bar.close
    except Exception:
        st = {"n": 0}
        step = max(1, total // 10)
        def upd(n=1):
            st["n"] += n
            if st["n"] % step == 0 or st["n"] >= total:
                print(f"[dl] {desc} {st['n']}/{total}", flush=True)
        return upd, (lambda: None)


def _get_json(url):
    r = subprocess.run(["curl", "-s", "--fail", "--max-time", "60", url], capture_output=True)
    if r.returncode != 0:
        raise RuntimeError(f"failed to fetch {url}")
    return json.loads(r.stdout.decode("utf-8"))


def _curl_code(url, out, tries=3):
    """curl one url -> out, returning the HTTP status code as a string. retries TRANSIENT
    failures (5xx / timeout / connection reset) with backoff, but never retries a 404, and
    leaves NO file behind on any non-200 (removes 404 error-xml / partial bodies). this
    http-code-aware classification is what stops a transient failure from being silently
    treated as an 'air' (blank) chunk -- the bug that corrupted the aria2c/runpod volumes."""
    code = "000"
    for _i in range(max(1, tries)):
        r = subprocess.run(
            ["curl", "-s", "--connect-timeout", "20", "--max-time", "120",
             "-o", out, "-w", "%{http_code}", url], capture_output=True)
        code = (r.stdout.decode("utf-8", "ignore").strip() or "000")[-3:]
        if code == "200":
            return code
        if os.path.exists(out):
            try: os.remove(out)
            except OSError: pass
        if code == "404":
            return code
        time.sleep(1)                 # transient -> brief backoff, then retry
    return code


def _fetch_chunk(args):
    """download one chunk. 200 -> data file; 404 -> zero-byte air sentinel; any OTHER failure
    -> NO file left, so _verify_repair re-checks/aborts instead of silently blanking it."""
    base, level, yc, xc, cache_dir = args
    out = os.path.join(cache_dir, f"{level}_{yc}_{xc}.raw")
    if os.path.exists(out):
        return (yc, xc, "cached")
    code = _curl_code(f"{base}/{level}/0/{yc}/{xc}", out)
    if code == "200":
        return (yc, xc, "ok")
    if code == "404":
        open(out, "wb").close()       # confirmed air
        return (yc, xc, "air")
    return (yc, xc, "fail")           # transient -> leave missing for _verify_repair


def _load_chunk(cache_dir, level, yc, xc, D):
    p = os.path.join(cache_dir, f"{level}_{yc}_{xc}.raw")
    try:
        if os.path.getsize(p) == D * CHUNK_XY * CHUNK_XY:
            return np.frombuffer(open(p, "rb").read(), dtype=np.uint8).reshape(D, CHUNK_XY, CHUNK_XY)
    except Exception:
        pass
    return None


def _aria2_fetch(base, level, jobs, cache_dir, workers):
    """bulk-download all not-yet-cached chunks with ONE aria2c process (fast, connection reuse).
    aria2c CANNOT distinguish a 404 from a transient failure, so it writes NO air sentinels here
    -- _verify_repair afterwards http-code-classifies everything aria2c left missing. returns
    (success_bool, elapsed_seconds)."""
    if shutil.which("aria2c") is None:
        return False, 0
    todo = [(f"{base}/{level}/0/{yc}/{xc}", f"{level}_{yc}_{xc}.raw")
            for (_b, _lvl, yc, xc, _cd) in jobs
            if not os.path.exists(os.path.join(cache_dir, f"{level}_{yc}_{xc}.raw"))]
    if not todo:
        return True, 0
    listfile = os.path.join(cache_dir, "_aria2_urls.txt")
    with open(listfile, "w") as f:
        for url, name in todo:
            f.write(f"{url}\n  dir={cache_dir}\n  out={name}\n")
    j = max(1, min(int(workers), 64))
    print(f"[dl] aria2c: bulk-fetching {len(todo)} chunks with -j{j} (missing chunks verified below)...", flush=True)
    t0 = time.time()
    subprocess.run(
        ["aria2c", "-i", listfile, f"-j{j}", "-x1", "-s1",
         "--max-tries=2", "--retry-wait=1", "--connect-timeout=20", "--timeout=120",
         "--auto-file-renaming=false", "--allow-overwrite=true",
         "-q", "--download-result=hide"],
        check=False)
    elapsed = time.time() - t0
    try:
        os.remove(listfile)
    except OSError:
        pass
    return True, elapsed


def _verify_repair(base, level, jobs, cache_dir, D, workers):
    """CORRECTNESS GATE. verify every expected chunk is EITHER a correctly-sized data file OR a
    confirmed-404 air chunk (zero-byte sentinel). anything missing or wrong-sized is re-fetched
    with an http-code-aware curl and classified 200->data / 404->air / else->hard failure. raises
    RuntimeError if any chunk cannot be resolved -- we refuse to assemble a silently-blanked
    volume (better to fail loudly and retry than train on corrupt data). returns (data, air)."""
    expected = D * CHUNK_XY * CHUNK_XY
    need = []
    data_n = air_n = 0
    for (_b, _lvl, yc, xc, _cd) in jobs:
        p = os.path.join(cache_dir, f"{level}_{yc}_{xc}.raw")
        if os.path.exists(p):
            sz = os.path.getsize(p)
            if sz == expected:
                data_n += 1; continue          # valid data
            if sz == 0:
                air_n += 1; continue           # confirmed-404 air sentinel
            try: os.remove(p)                  # partial/corrupt -> re-fetch
            except OSError: pass
        need.append((yc, xc))
    if need:
        print(f"[dl] verify: re-checking {len(need)} missing/partial chunk(s) with http-code...", flush=True)
        def _chk(coord):
            yc, xc = coord
            p = os.path.join(cache_dir, f"{level}_{yc}_{xc}.raw")
            code = _curl_code(f"{base}/{level}/0/{yc}/{xc}", p, tries=4)
            if code == "200" and os.path.exists(p) and os.path.getsize(p) == expected:
                return (coord, "data")
            if code == "404":
                open(p, "wb").close()
                return (coord, "air")
            return (coord, f"ERR:{code}")
        results = {}
        with ThreadPoolExecutor(max_workers=max(1, min(int(workers), 32))) as ex:
            for coord, status in ex.map(_chk, need):
                results[coord] = status
        errs = [(c, s) for c, s in results.items() if s.startswith("ERR")]
        if errs:
            raise RuntimeError(
                f"[dl] {len(errs)} chunk(s) failed to download for a NON-404 reason, e.g. {errs[:5]}. "
                f"refusing to assemble a blanked volume -- re-run to retry (S3 may be rate-limiting).")
        data_n += sum(1 for s in results.values() if s == "data")
        air_n += sum(1 for s in results.values() if s == "air")
    return data_n, air_n


def _download_surface_zarr(base, level, out_zarr, out_id, workers, cache_dir, chunk_depth, chunk_y, chunk_x):
    """download a pre-rendered OME-Zarr surface volume from S3 -> local zarr + mask png.
    these segment surface-volumes are ALREADY flattened (no mesh sampling needed), stored as
    OME-Zarr: a group with pyramid levels 0..5, each an array with chunks [D,128,128], uint8,
    NO compressor, dimension_separator "/". a chunk lives at  {base}.zarr/{level}/{zc}/{yc}/{xc}.
    the chunk depth == full D, so zc is always 0; missing chunk (404) = all-air background."""
    base = base.rstrip("/")
    za = _get_json(f"{base}/{level}/.zarray")
    D, H, W = za["shape"]
    print(f"[dl] level {level} shape (D,H,W)=({D},{H},{W}) dtype={za['dtype']}")
    os.makedirs(cache_dir, exist_ok=True)

    n_yc = (H + CHUNK_XY - 1) // CHUNK_XY
    n_xc = (W + CHUNK_XY - 1) // CHUNK_XY
    jobs = [(base, level, yc, xc, cache_dir) for yc in range(n_yc) for xc in range(n_xc)]
    print(f"[dl] fetching {len(jobs)} chunks ({n_yc}x{n_xc}) with {workers} workers "
          f"(~{len(jobs) * D * CHUNK_XY * CHUNK_XY / 1e9:.1f} GB max)")
    
    # fast path: single aria2c process (connection reuse). falls back to per-chunk curl if absent.
    used_aria2, aria2_time = _aria2_fetch(base, level, jobs, cache_dir, workers)
    if used_aria2 and aria2_time > 0:
        print(f"[dl] aria2c completed in {aria2_time:.1f}s ({aria2_time/60:.1f}m)", flush=True)
    
    if not used_aria2:
        done = 0
        with ThreadPoolExecutor(max_workers=workers) as ex:
            for _ in ex.map(_fetch_chunk, jobs):
                done += 1
                if done % 500 == 0:
                    print(f"[dl] fetched {done}/{len(jobs)}", flush=True)
    
    # CORRECTNESS GATE: verify every chunk is valid data or a confirmed-404 air chunk, re-fetching
    # anything missing/partial and ABORTING on a non-404 failure. this is what stops a transient
    # download error from being silently baked into the volume as a blank region.
    data_n, air_n = _verify_repair(base, level, jobs, cache_dir, D, workers)
    print(f"[dl] all {len(jobs)} chunks verified: {data_n} data + {air_n} air (blank)", flush=True)

    # write zarr directly from cache chunks (avoids ~2GB intermediate array).
    # writing full-z columns means each zarr z-chunk gets all its data in one shot,
    # eliminating the read-modify-write that layer-by-layer writes cause (~7x redundant
    # reads per z-chunk group). columns are disjoint in zarr chunk space and mask space,
    # so parallel writes are safe with DirectoryStore (one file per chunk, no shared state).
    import zarr
    store = zarr.open(out_zarr, mode="w", shape=(D, H, W), 
                      chunks=(min(chunk_depth, D), chunk_y, chunk_x),
                      dtype="<u2", compressor=None, zarr_format=2)
    mask_buf = np.zeros((H, W), dtype=np.uint8)

    def _write_col(yx):
        yc, xc = yx
        ch = _load_chunk(cache_dir, level, yc, xc, D)
        y0, x0 = yc * CHUNK_XY, xc * CHUNK_XY
        y1, x1 = min(y0 + CHUNK_XY, H), min(x0 + CHUNK_XY, W)
        if ch is None:
            return
        data = ch[:, :y1-y0, :x1-x0]
        store[:, y0:y1, x0:x1] = data.astype(np.uint16)
        mask_buf[y0:y1, x0:x1] = (data.max(axis=0) > 0).astype(np.uint8) * 255

    pairs = [(yc, xc) for yc in range(n_yc) for xc in range(n_xc)]
    upd, close = _pbar(len(pairs), "writing zarr")
    # 32 threads is a practical sweet spot: more causes small-file write contention
    with ThreadPoolExecutor(max_workers=min(workers, 32)) as ex:
        for _ in ex.map(_write_col, pairs):
            upd()
    close()
    print(f"[dl] wrote zarr {out_zarr} ({D},{H},{W}) uint16")

    # mask = footprint where ANY layer is nonzero (valid rendered surface)
    os.makedirs("masks", exist_ok=True)
    Image.fromarray(mask_buf).save(f"masks/{out_id}.png")
    print(f"[dl] wrote masks/{out_id}.png  valid_frac={(mask_buf > 0).mean():.3f}")


def _verify_zarr_integrity(zarr_path, zid):
    """post-download integrity check: verify all depth slices have valid (non-zero) data.
    raises RuntimeError if the volume appears corrupted/incomplete (e.g., only first few
    slices downloaded, rest are zeros). this catches the bug that corrupted w044/w034."""
    import zarr
    print(f"[verify] checking zarr integrity: {zarr_path}", flush=True)
    try:
        z = zarr.open(zarr_path, mode='r')
        D, H, W = z.shape
        
        # check every depth slice for valid data (sample a stripe across middle)
        valid_depths = []
        for d in range(D):
            # sample every 100th pixel across middle row to avoid loading full slice
            sample = z[d, H//2, ::100]
            if sample.mean() > 0 and len(np.unique(sample)) > 1:
                valid_depths.append(d)
        
        valid_count = len(valid_depths)
        coverage = valid_count / D if D > 0 else 0
        
        print(f"[verify] {zid}: {valid_count}/{D} depths have valid data ({coverage*100:.1f}%)")
        
        # require at least 90% depth coverage (allow a few air slices at edges)
        if coverage < 0.90:
            missing = sorted(set(range(D)) - set(valid_depths))
            raise RuntimeError(
                f"INTEGRITY CHECK FAILED: {zid} has only {valid_count}/{D} valid depths "
                f"({coverage*100:.1f}% coverage). Missing/zero depths: {missing[:10]}{'...' if len(missing) > 10 else ''}. "
                f"This indicates an incomplete download. Delete {zarr_path} and re-run to retry.")
        
        # check spatial coverage on middle depth slice
        mid_d = D // 2
        if mid_d in valid_depths:
            # sample 10x10 regions at 5 locations
            regions = [
                z[mid_d, H//4:H//4+10, W//4:W//4+10],
                z[mid_d, H//4:H//4+10, 3*W//4:3*W//4+10],
                z[mid_d, H//2:H//2+10, W//2:W//2+10],
                z[mid_d, 3*H//4:3*H//4+10, W//4:W//4+10],
                z[mid_d, 3*H//4:3*H//4+10, 3*W//4:3*W//4+10],
            ]
            nonzero_regions = sum(1 for r in regions if r.mean() > 0)
            if nonzero_regions == 0:
                raise RuntimeError(
                    f"INTEGRITY CHECK FAILED: {zid} depth {mid_d} has no valid data in sampled regions. "
                    f"Volume may be corrupted. Delete {zarr_path} and re-run.")
        
        print(f"[verify] ✓ {zid} integrity check passed", flush=True)
        
    except Exception as e:
        if "INTEGRITY CHECK FAILED" in str(e):
            raise
        raise RuntimeError(f"Failed to verify zarr integrity for {zid}: {e}")


SWEEP_GRID = 16           # NxN probe grid across the frame
SWEEP_MIN_COVERAGE = 0.5  # min fraction of mask-valid probes that must carry data


def _zarr_integrity(zid, mask_dir="masks", grid=SWEEP_GRID, min_coverage=SWEEP_MIN_COVERAGE):
    """cheap spatial integrity check for one scroll's on-disk zarr. samples an NxN grid of
    depth-columns across the frame and verifies the zarr carries data wherever the mask says
    the surface is valid. a partial/interrupted write covers only a sliver of its mask (e.g.
    w044/w047 were ~7%); a middle-row-only check misses this. returns (ok: bool, msg: str)."""
    import zarr
    zpath = os.path.join(ZARR_DIR, f"{zid}.zarr")
    if not os.path.isdir(zpath):
        return False, "MISSING zarr"
    try:
        z = zarr.open(zpath, mode="r")
        D, H, W = map(int, z.shape)
    except Exception as e:
        return False, f"UNREADABLE ({type(e).__name__})"
    mpath = os.path.join(mask_dir, f"{zid}.png")
    if not os.path.exists(mpath):
        return False, "MISSING mask"
    m = np.array(Image.open(mpath).convert("L"))
    hh, ww = min(H, m.shape[0]), min(W, m.shape[1])
    ys = np.linspace(0, hh - 1, grid).astype(int)
    xs = np.linspace(0, ww - 1, grid).astype(int)
    valid = have = 0
    for y in ys:
        for x in xs:
            if m[y, x] > 0:                       # mask claims valid surface here
                valid += 1
                if bool((np.asarray(z[:, y, x]) > 0).any()):   # any depth carries data
                    have += 1
    if valid == 0:
        return False, "MASK_EMPTY"
    cov = have / valid
    mb_frac = float((m > 0).mean()) * 100
    if cov < min_coverage:
        return False, f"DATA_UNDERFILL cov={cov:.2f} (mask={mb_frac:.0f}% valid)"
    return True, f"cov={cov:.2f}"


def _pool_w013_depth(volume, layers_out):
    """area-average a w013 row strip from 109 source depths to 28 output depths."""
    depth = int(volume.shape[0])
    out = np.zeros((layers_out, volume.shape[1], volume.shape[2]), dtype=np.float32)
    for index in range(layers_out):
        start = int(round(index * depth / layers_out))
        end = int(round((index + 1) * depth / layers_out))
        out[index] = volume[start:end].mean(axis=0)
    return out


def _resample_depth_physical(volume, source_um, layers_out=TARGET_DEPTH):
    """linearly sample source depths on a centered TARGET_VOXEL_UM grid."""
    source_depth = int(volume.shape[0])
    source_center = (source_depth - 1) / 2.0
    output_center = (layers_out - 1) / 2.0
    positions = source_center + (
        np.arange(layers_out) - output_center
    ) * TARGET_VOXEL_UM / float(source_um)
    output = np.zeros((layers_out, volume.shape[1], volume.shape[2]), dtype=np.float32)
    for output_depth, position in enumerate(positions):
        if position < 0 or position > source_depth - 1:
            continue
        lower = int(np.floor(position))
        upper = min(lower + 1, source_depth - 1)
        fraction = float(position - lower)
        output[output_depth] = volume[lower] * (1.0 - fraction)
        if upper != lower:
            output[output_depth] += volume[upper] * fraction
    return output


def _write_mask_from_midslice(output, mask_path):
    """write the physical surface footprint from the output volume's middle slice."""
    middle = np.asarray(output[int(output.shape[0]) // 2])
    mask = (middle > 0).astype(np.uint8) * 255
    os.makedirs(os.path.dirname(mask_path), exist_ok=True)
    Image.fromarray(mask).save(mask_path)
    print(f"  [mask] wrote {mask_path} from midslice valid_frac={(mask > 0).mean():.3f}")


def _curl_range(url, start, end, output):
    command = [
        "curl", "-s", "--fail", "--connect-timeout", "20", "--max-time", "180",
        "--retry", "3", "--retry-delay", "2", "--show-error",
        "-r", f"{start}-{end}", "-o", output, url,
    ]
    result = subprocess.run(command, capture_output=True, text=True)
    if result.returncode != 0:
        if os.path.exists(output):
            os.remove(output)
        detail = result.stderr.strip() or "no curl error text"
        raise RuntimeError(
            f"range download failed ({result.returncode}) for {url} "
            f"bytes={start}-{end}: {detail}"
        )


def _fetch_dlash_band(args):
    layer, url, start, end, output, rows, width = args
    _curl_range(url, start, end, output)
    band = np.fromfile(output, dtype="<u2")
    if band.size != rows * width:
        raise RuntimeError(f"layer {layer}: downloaded {band.size} pixels, expected {rows * width}")
    return layer, band.reshape(rows, width)


def _dlash_depth_positions(source_depth, source_um):
    source_center = (source_depth - 1) / 2.0
    output_center = (TARGET_DEPTH - 1) / 2.0
    return source_center + (np.arange(TARGET_DEPTH) - output_center) * TARGET_VOXEL_UM / source_um


def _download_once(url, path):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    if os.path.isfile(path) and os.path.getsize(path) > 0:
        return
    run([
        "curl", "-L", "--fail", "--retry", "5", "--retry-delay", "2",
        "-o", path, url,
    ])


def _assemble_dlash_surface(name, zid, opts, workers, chunk_depth, chunk_y, chunk_x, force=False):
    """stream a dl.ash2txt uint16 TIFF stack and resample it to the training voxel grid."""
    import zarr

    source_depth, source_height, source_width = map(int, opts["surface_expected_shape"])
    source_um = float(opts["source_um"])
    output_height = int(round(source_height * source_um / TARGET_VOXEL_UM))
    output_width = int(round(source_width * source_um / TARGET_VOXEL_UM))
    output_shape = (TARGET_DEPTH, output_height, output_width)
    output_path = os.path.join(ZARR_DIR, f"{zid}.zarr")
    partial_path = output_path + ".partial"
    cache_dir = os.path.join(TMP, f"dlash_{zid}")
    progress_path = os.path.join(partial_path, ".assembly_progress.json")
    mask_path = os.path.join("masks", f"{zid}.png")
    if force:
        shutil.rmtree(output_path, ignore_errors=True)
        shutil.rmtree(partial_path, ignore_errors=True)
        if os.path.exists(mask_path):
            os.remove(mask_path)
    if os.path.isdir(output_path):
        print("  [1/3] dl.ash2txt volume exists -> skip")
        return

    os.makedirs(cache_dir, exist_ok=True)
    row_height = 512
    next_source_y = 0
    output = None
    if os.path.isdir(partial_path) and os.path.isfile(progress_path):
        with open(progress_path, encoding="utf-8") as handle:
            progress = json.load(handle)
        if tuple(progress.get("output_shape", ())) == output_shape:
            output = zarr.open(partial_path, mode="r+")
            next_source_y = int(progress.get("next_source_y", 0))
        else:
            shutil.rmtree(partial_path)
    if output is None:
        shutil.rmtree(partial_path, ignore_errors=True)
        output = zarr.open(
            partial_path,
            mode="w",
            shape=output_shape,
            chunks=(min(chunk_depth, TARGET_DEPTH), chunk_y, chunk_x),
            dtype="<u2",
            compressor=None,
            zarr_format=2,
        )

    positions = _dlash_depth_positions(source_depth, source_um)
    valid_depths = np.flatnonzero((positions >= 0) & (positions <= source_depth - 1))
    base = opts["dlash_surface_base"].rstrip("/")
    print(
        f"  [1/3] dl.ash2txt resample {(source_depth, source_height, source_width)} "
        f"@{source_um:.3f}um -> {output_shape} @{TARGET_VOXEL_UM:.3f}um; "
        f"valid target depths={valid_depths[0]}..{valid_depths[-1]}"
    )
    for source_y0 in range(next_source_y, source_height, row_height):
        source_y1 = min(source_y0 + row_height, source_height)
        output_y0 = int(round(source_y0 * output_height / source_height))
        output_y1 = int(round(source_y1 * output_height / source_height))
        rows = source_y1 - source_y0
        start = 8 + source_y0 * source_width * 2
        end = 8 + source_y1 * source_width * 2 - 1
        jobs = [
            (
                layer,
                f"{base}/surface_volume/{layer:02d}.tif",
                start,
                end,
                os.path.join(cache_dir, f"band_{layer:02d}.raw"),
                rows,
                source_width,
            )
            for layer in range(source_depth)
        ]
        source_band = np.empty((source_depth, rows, source_width), dtype=np.uint16)
        with ProcessPoolExecutor(max_workers=max(1, min(int(workers), source_depth))) as executor:
            for layer, band in executor.map(_fetch_dlash_band, jobs):
                source_band[layer] = band
        for output_z, position in enumerate(positions):
            if position < 0 or position > source_depth - 1:
                output[output_z, output_y0:output_y1, :] = 0
                continue
            lower = int(np.floor(position))
            upper = min(lower + 1, source_depth - 1)
            fraction = float(position - lower)
            plane = source_band[lower].astype(np.float32)
            if upper != lower:
                plane *= 1.0 - fraction
                plane += source_band[upper] * fraction
            resized = cv2.resize(
                plane,
                (output_width, output_y1 - output_y0),
                interpolation=cv2.INTER_AREA,
            )
            output[output_z, output_y0:output_y1, :] = np.clip(
                np.rint(resized), 0, 65535
            ).astype(np.uint16)
        with open(progress_path, "w", encoding="utf-8") as handle:
            json.dump({"output_shape": output_shape, "next_source_y": source_y1}, handle)
        print(f"  [dlash] rows {source_y1}/{source_height}", flush=True)
    del output
    os.remove(progress_path)
    os.replace(partial_path, output_path)
    print(f"  [dlash] wrote {output_path} shape={output_shape}")


def _is_rescan_zarr(path):
    try:
        with open(os.path.join(path, ".zattrs"), encoding="utf-8") as handle:
            return RESCAN_MARKER in json.load(handle)
    except (OSError, ValueError):
        return False


def _assemble_dlash_rescan(name, zid, opts, chunk_depth, chunk_y, chunk_x, force=False):
    """render the fragment from its 88 keV scan; atomically supersedes an older (low-energy) zarr + mask."""
    from utils import fragment_rescan

    rescan = RESCAN_88KEV[name]
    output_path = os.path.join(ZARR_DIR, f"{zid}.zarr")
    partial_path = output_path + ".partial"
    mask_path = os.path.join("masks", f"{zid}.png")
    if os.path.isdir(output_path) and _is_rescan_zarr(output_path) and not force:
        print("  [1/3] 88 keV rescan volume exists -> skip")
        return
    base = opts["dlash_surface_base"].rstrip("/")
    cache_dir = os.path.join(TMP, f"rescan_{zid}")
    source_depth, source_height, source_width = map(int, opts["surface_expected_shape"])
    source_mask_path = os.path.join(cache_dir, "mask.png")
    _download_once(f"{base}/mask.png", source_mask_path)
    source_mask = cv2.imread(source_mask_path, cv2.IMREAD_GRAYSCALE)
    if source_mask is None or source_mask.shape != (source_height, source_width):
        raise RuntimeError(f"{name}: bad source mask {None if source_mask is None else source_mask.shape}")
    frame = fragment_rescan.out_shape(source_height, source_width)
    footprint = cv2.resize(source_mask, frame[::-1], interpolation=cv2.INTER_NEAREST) > 0
    positions = _dlash_depth_positions(source_depth, float(opts["source_um"]))
    empty_layers = [int(k) for k in np.flatnonzero((positions < 0) | (positions > source_depth - 1))]
    print(f"  [1/3] 88 keV rescan -> {(TARGET_DEPTH,) + frame}; empty layers {empty_layers}", flush=True)
    shutil.rmtree(partial_path, ignore_errors=True)
    mask = fragment_rescan.render(
        partial_path, f"{base}/result.ppm", rescan["volume_url"], rescan["affine"], cache_dir,
        footprint=footprint, empty_layers=empty_layers, tif_url=rescan.get("tif_url"),
        tif_digits=int(rescan.get("tif_digits", 4)), chunks=(min(chunk_depth, TARGET_DEPTH), chunk_y, chunk_x),
    )
    with open(os.path.join(partial_path, ".zattrs"), "w", encoding="utf-8") as handle:
        json.dump({RESCAN_MARKER: {"volume_url": rescan["volume_url"], "affine": rescan["affine"],
                                   "ppm": f"{base}/result.ppm"}}, handle, indent=1)
    if os.path.isdir(output_path):
        print(f"  [1/3] superseding the existing zarr {output_path}")
        shutil.rmtree(output_path)
    os.replace(partial_path, output_path)
    os.makedirs("masks", exist_ok=True)
    cv2.imwrite(mask_path, mask.astype(np.uint8) * 255)
    print(f"  [1/3] wrote {output_path} + {mask_path} (footprint {mask.mean():.3f}, "
          f"{mask.sum() / max(footprint.sum(), 1):.4f} of the source mask)")
    shutil.rmtree(os.path.join(cache_dir, "chunks"), ignore_errors=True)


def _build_dlash_mask(name, zid, opts):
    base = opts["dlash_surface_base"].rstrip("/")
    cache_dir = os.path.join(TMP, f"dlash_{zid}")
    source_mask_path = os.path.join(cache_dir, "mask.png")
    _download_once(f"{base}/mask.png", source_mask_path)
    source_mask = cv2.imread(source_mask_path, cv2.IMREAD_GRAYSCALE)
    expected_shape = tuple(map(int, opts["surface_expected_shape"][1:]))
    if source_mask is None:
        raise RuntimeError(f"{name}: failed to load downloaded mask")
    if source_mask.shape != expected_shape:
        raise RuntimeError(
            f"{name}: source mask shape {source_mask.shape} != {expected_shape}"
        )

    output = __import__("zarr").open(os.path.join(ZARR_DIR, f"{zid}.zarr"), mode="r")
    target_shape = tuple(map(int, output.shape[1:]))
    mask = cv2.resize(source_mask, target_shape[::-1], interpolation=cv2.INTER_NEAREST)
    os.makedirs("masks", exist_ok=True)
    cv2.imwrite(f"masks/{zid}.png", mask)
    print(f"  [mask] wrote masks/{zid}.png shape={mask.shape}")


def _verify_dlash_outputs(name, zid, opts):
    import zarr

    source_depth, source_height, source_width = map(int, opts["surface_expected_shape"])
    source_um = float(opts["source_um"])
    expected_shape = (
        TARGET_DEPTH,
        int(round(source_height * source_um / TARGET_VOXEL_UM)),
        int(round(source_width * source_um / TARGET_VOXEL_UM)),
    )
    volume = zarr.open(os.path.join(ZARR_DIR, f"{zid}.zarr"), mode="r")
    if tuple(volume.shape) != expected_shape or np.dtype(volume.dtype) != np.dtype("<u2"):
        raise RuntimeError(f"{name}: zarr shape/dtype {volume.shape}/{volume.dtype} != {expected_shape}/uint16")
    positions = _dlash_depth_positions(source_depth, source_um)
    valid_depths = np.flatnonzero((positions >= 0) & (positions <= source_depth - 1))
    for depth in valid_depths:
        if not np.any(np.asarray(volume[int(depth), ::64, ::64])):
            raise RuntimeError(f"{name}: target depth {depth} has no sampled data")
    mask = np.asarray(Image.open(f"masks/{zid}.png").convert("L")) > 0
    if mask.shape != expected_shape[1:]:
        raise RuntimeError(f"{name}: output mask dimensions do not match zarr XY")
    midslice = np.asarray(volume[TARGET_DEPTH // 2]) > 0
    overlap = float((mask & midslice).sum() / max(int(mask.sum()), 1))
    if overlap <= 0.99:
        raise RuntimeError(f"{name}: mask/midslice overlap {overlap:.5f} is not >0.99")
    print(f"  [verify] {name}: shape={volume.shape} mask/midslice={overlap:.5f}")


def _assemble_resampled_volume(seg, zid, source_um, vol_name, chunk_depth, chunk_y, chunk_x,
                               expected_shape=None, force=False):
    """resample a small isotropic surface zarr in XYZ to the 28-layer training frame."""
    import cv2
    import zarr

    out_zarr = os.path.join(ZARR_DIR, f"{zid}.zarr")
    partial = out_zarr + ".partial"
    mask_path = os.path.join("masks", f"{zid}.png")
    if force:
        shutil.rmtree(out_zarr, ignore_errors=True)
        shutil.rmtree(partial, ignore_errors=True)
        if os.path.exists(mask_path):
            os.remove(mask_path)
    if os.path.isdir(out_zarr) and os.path.exists(mask_path):
        print("  [1/3] resampled volume+mask exist -> skip")
        return

    source_url = f"{BUCKET}/{seg}/surface-volumes/{vol_name}/0"
    source = zarr.open(source_url, mode="r")
    source_depth, source_height, source_width = map(int, source.shape)
    if expected_shape is not None and tuple(source.shape) != tuple(expected_shape):
        raise RuntimeError(
            f"{zid}: source surface shape {tuple(source.shape)} != expected "
            f"{tuple(expected_shape)}"
        )
    output_height = int(round(source_height * float(source_um) / TARGET_VOXEL_UM))
    output_width = int(round(source_width * float(source_um) / TARGET_VOXEL_UM))
    output_shape = (TARGET_DEPTH, output_height, output_width)
    print(
        f"  [1/3] isotropic resample {source.shape} @{source_um:.3f}um -> "
        f"{output_shape} @{TARGET_VOXEL_UM:.3f}um"
    )
    shutil.rmtree(partial, ignore_errors=True)
    output = zarr.open(
        partial,
        mode="w",
        shape=output_shape,
        chunks=(min(chunk_depth, TARGET_DEPTH), chunk_y, chunk_x),
        dtype="<u2",
        compressor=None,
        zarr_format=2,
    )
    interpolation = cv2.INTER_AREA if output_height < source_height else cv2.INTER_LINEAR
    source_row_height = 128
    for source_y0 in range(0, source_height, source_row_height):
        source_y1 = min(source_y0 + source_row_height, source_height)
        output_y0 = int(round(source_y0 * output_height / source_height))
        output_y1 = int(round(source_y1 * output_height / source_height))
        if output_y1 <= output_y0:
            continue
        source_strip = np.asarray(source[:, source_y0:source_y1, :], dtype=np.uint8)
        depth_resampled = _resample_depth_physical(
            source_strip,
            source_um,
            TARGET_DEPTH,
        )
        output_strip = np.empty(
            (TARGET_DEPTH, output_y1 - output_y0, output_width),
            dtype=np.uint16,
        )
        for output_z in range(TARGET_DEPTH):
            plane = cv2.resize(
                depth_resampled[output_z],
                (output_width, output_y1 - output_y0),
                interpolation=interpolation,
            )
            output_strip[output_z] = np.clip(np.rint(plane), 0, 255).astype(np.uint16)
        output[:, output_y0:output_y1, :] = output_strip
        print(f"  [resample] rows {output_y1}/{output_height}", flush=True)
    del output
    if os.path.isdir(out_zarr):
        shutil.rmtree(out_zarr)
    os.replace(partial, out_zarr)
    output = zarr.open(out_zarr, mode="r")
    _write_mask_from_midslice(output, mask_path)
    print(f"  [resample] wrote {out_zarr} shape={output_shape}")


def _assemble_cropped_surface(zid, opts, chunk_depth, chunk_y, chunk_x, force=False):
    """stream a native-target-resolution remote zarr into a tightly cropped local zarr."""
    import zarr

    out_zarr = os.path.join(ZARR_DIR, f"{zid}.zarr")
    partial = out_zarr + ".partial"
    mask_path = os.path.join("masks", f"{zid}.png")
    if force:
        shutil.rmtree(out_zarr, ignore_errors=True)
        shutil.rmtree(partial, ignore_errors=True)
        if os.path.exists(mask_path):
            os.remove(mask_path)
    if os.path.isdir(out_zarr) and os.path.exists(mask_path):
        print("  [1/3] cropped volume+mask exist -> skip")
        return

    source = zarr.open(str(opts["cropped_surface_url"]), mode="r")
    expected_shape = tuple(map(int, opts["surface_expected_shape"]))
    if tuple(source.shape) != expected_shape:
        raise RuntimeError(f"{zid}: source surface shape {tuple(source.shape)} != {expected_shape}")
    y0, y1, x0, x1 = map(int, opts["surface_crop"])
    if not (0 <= y0 < y1 <= source.shape[1] and 0 <= x0 < x1 <= source.shape[2]):
        raise ValueError(f"{zid}: invalid surface crop {(y0, y1, x0, x1)} for {source.shape}")
    output_shape = (int(source.shape[0]), y1 - y0, x1 - x0)
    shutil.rmtree(partial, ignore_errors=True)
    output = zarr.open(
        partial,
        mode="w",
        shape=output_shape,
        chunks=(min(chunk_depth, output_shape[0]), chunk_y, chunk_x),
        dtype="<u2",
        compressor=None,
        zarr_format=2,
    )
    for source_y0 in range(y0, y1, 128):
        source_y1 = min(source_y0 + 128, y1)
        output[:, source_y0 - y0:source_y1 - y0, :] = np.asarray(
            source[:, source_y0:source_y1, x0:x1],
            dtype=np.uint16,
        )
        print(f"  [crop] rows {source_y1 - y0}/{y1 - y0}", flush=True)
    del output
    if os.path.isdir(out_zarr):
        shutil.rmtree(out_zarr)
    os.replace(partial, out_zarr)
    output = zarr.open(out_zarr, mode="r")
    _write_mask_from_midslice(output, mask_path)
    print(f"  [crop] wrote {out_zarr} shape={output_shape} crop={(y0, y1, x0, x1)}")


def _assemble_pooled_surface(
    seg, zid, opts, chunk_depth, chunk_y, chunk_x, expected_shape=None, force=False
):
    """stream a high-resolution surface pyramid and pool its native depth to 28."""
    import zarr

    out_zarr = os.path.join(ZARR_DIR, f"{zid}.zarr")
    partial = out_zarr + ".partial"
    mask_path = os.path.join("masks", f"{zid}.png")
    if force:
        shutil.rmtree(out_zarr, ignore_errors=True)
        shutil.rmtree(partial, ignore_errors=True)
        if os.path.exists(mask_path):
            os.remove(mask_path)
    if os.path.isdir(out_zarr) and os.path.exists(mask_path):
        print("  [1/3] pooled volume+mask exist -> skip")
        return

    level = int(opts.get("surface_level", 2))
    source_url = f"{BUCKET}/{seg}/surface-volumes/{opts['surface_name']}/{level}"
    source = zarr.open(source_url, mode="r")
    if expected_shape is not None and tuple(source.shape) != tuple(expected_shape):
        raise RuntimeError(
            f"{zid}: source surface shape {tuple(source.shape)} != expected "
            f"{tuple(expected_shape)}"
        )
    source_depth, source_height, source_width = map(int, source.shape)
    y0, y1, x0, x1 = opts.get("surface_crop", (0, source_height, 0, source_width))
    y0, y1 = max(0, int(y0)), min(source_height, int(y1))
    x0, x1 = max(0, int(x0)), min(source_width, int(x1))
    output_shape = (TARGET_DEPTH, y1 - y0, x1 - x0)
    output_dtype = np.dtype(opts.get("output_dtype", "<u2"))
    print(
        f"  [1/3] level-{level} pool {source.shape} crop={(y0, y1, x0, x1)} "
        f"depth={source_depth}->{TARGET_DEPTH} output={output_shape}"
    )
    shutil.rmtree(partial, ignore_errors=True)
    output = zarr.open(
        partial,
        mode="w",
        shape=output_shape,
        chunks=(min(chunk_depth, TARGET_DEPTH), chunk_y, chunk_x),
        dtype=output_dtype,
        compressor=None,
        zarr_format=2,
    )
    row_height = 128
    for row_start in range(y0, y1, row_height):
        row_end = min(row_start + row_height, y1)
        source_row = np.asarray(source[:, row_start:row_end, x0:x1], dtype=np.float32)
        pooled = _pool_w013_depth(source_row, TARGET_DEPTH)
        output[:, row_start - y0:row_end - y0] = np.clip(
            np.rint(pooled), 0, 255
        ).astype(output_dtype)
        print(f"  [pool] rows {row_end - y0}/{y1 - y0}", flush=True)
    del output
    if os.path.isdir(out_zarr):
        shutil.rmtree(out_zarr)
    os.replace(partial, out_zarr)
    output = zarr.open(out_zarr, mode="r")
    _write_mask_from_midslice(output, mask_path)
    print(f"  [pool] wrote {out_zarr} shape={output_shape}")


def _assemble_w013_volume(zid, chunk_depth, chunk_y, chunk_x, force=False):
    """stream, pool, and crop PHerc1667 w013 without a standalone helper script."""
    import zarr

    out_zarr = os.path.join(ZARR_DIR, f"{zid}.zarr")
    mask_path = os.path.join("masks", f"{zid}.png")
    if force:
        if os.path.isdir(out_zarr):
            shutil.rmtree(out_zarr, ignore_errors=True)
        if os.path.exists(mask_path):
            os.remove(mask_path)
    if os.path.isdir(out_zarr) and os.path.exists(mask_path):
        print("  [1/3] w013 volume+mask exist -> skip")
        return

    source_depth, output_height, _source_width = W013_SOURCE_SHAPE
    output_depth, _, output_width = W013_OUTPUT_SHAPE
    source_url = f"{W013_SURF_BASE}/{W013_SURF_LEVEL}"
    print(
        f"  [1/3] w013 source={source_url} "
        f"pool={source_depth}->{output_depth} output={W013_OUTPUT_SHAPE}"
    )
    source = zarr.open(source_url, mode="r")
    os.makedirs(ZARR_DIR, exist_ok=True)
    output = zarr.open(
        out_zarr,
        mode="w",
        shape=W013_OUTPUT_SHAPE,
        chunks=(min(chunk_depth, output_depth), chunk_y, chunk_x),
        dtype="<u2",
        compressor=None,
        zarr_format=2,
    )
    mask = np.zeros((output_height, output_width), dtype=np.uint8)
    row_height = 128
    row_count = (output_height + row_height - 1) // row_height
    for row_index in range(row_count):
        y0 = row_index * row_height
        y1 = min(y0 + row_height, output_height)
        row = np.asarray(source[:, y0:y1, :output_width], dtype=np.float32)
        pooled = _pool_w013_depth(row, output_depth)
        output[:, y0:y1] = np.clip(pooled, 0, 255).astype(np.uint16)
        mask[y0:y1] = (pooled[output_depth // 2] > 0).astype(np.uint8) * 255
        if (row_index + 1) % 10 == 0 or row_index + 1 == row_count:
            print(f"  [w013] rows {row_index + 1}/{row_count}", flush=True)
    os.makedirs("masks", exist_ok=True)
    Image.fromarray(mask).save(mask_path)
    print(f"  [w013] wrote {out_zarr} and {mask_path}")


def step1_volume(name, seg, zid, workers, chunk_depth, chunk_y, chunk_x, force=False):
    opts = FRAG_OPTS.get(name, {})
    if name in RESCAN_88KEV:
        _assemble_dlash_rescan(name, zid, opts, chunk_depth, chunk_y, chunk_x, force=force)
        return
    if opts.get("dlash_surface_base"):
        _assemble_dlash_surface(
            name, zid, opts, workers, chunk_depth, chunk_y, chunk_x, force=force
        )
        return
    if opts.get("cropped_surface_url"):
        _assemble_cropped_surface(
            zid, opts, chunk_depth, chunk_y, chunk_x, force=force
        )
        return
    if opts.get("w013_special"):
        _assemble_w013_volume(zid, chunk_depth, chunk_y, chunk_x, force=force)
        return
    if opts.get("pooled_special"):
        _assemble_pooled_surface(
            seg,
            zid,
            opts,
            chunk_depth,
            chunk_y,
            chunk_x,
            expected_shape=opts.get("surface_expected_shape"),
            force=force,
        )
        return
    if opts.get("resample_um"):
        _assemble_resampled_volume(
            seg,
            zid,
            float(opts["resample_um"]),
            str(opts["vol9_name"]),
            chunk_depth,
            chunk_y,
            chunk_x,
            force=force,
        )
        return
    zpath = os.path.join(ZARR_DIR, f"{zid}.zarr")
    mpath = f"masks/{zid}.png"
    if os.path.exists(zpath) and os.path.exists(mpath) and not force:
        print(f"  [1/3] volume+mask exist -> skip")
        return
    if force:
        # clear stale outputs + the per-fragment chunk cache so the re-download is clean;
        # a reused cache can carry the corrupt/air chunks that blanked the volume
        cache = os.path.join(TMP, f"dl_{zid}")
        if os.path.isdir(zpath):
            shutil.rmtree(zpath, ignore_errors=True)
        if os.path.exists(mpath):
            try: os.remove(mpath)
            except OSError: pass
        if os.path.isdir(cache):
            shutil.rmtree(cache, ignore_errors=True)
        print(f"  [1/3] --force: cleared zarr/mask + chunk cache, re-downloading")
    # per-fragment volume name override (non-PHerc0139 scrolls have a different raw vol id)
    vol9 = FRAG_OPTS.get(name, {}).get("vol9_name", VOL9_NAME)
    url = f"{BUCKET}/{seg}/surface-volumes/{vol9}"
    cache = os.path.join(TMP, f"dl_{zid}")
    _download_surface_zarr(url, level=0, out_zarr=zpath, out_id=zid, 
                           workers=workers, cache_dir=cache,
                           chunk_depth=chunk_depth, chunk_y=chunk_y, chunk_x=chunk_x)
    
    # INTEGRITY CHECK: verify all depth slices have valid data (catches incomplete downloads)
    _verify_zarr_integrity(zpath, zid)


def step2_check_eroded_labels(name, seg, zid):
    """verify eroded_inklabels exist (the only labels we actually use for training).
    these are pre-generated conservative binary labels: high-confidence ink pixels,
    eroded and masked. the inklabels/ dir (raw 1um ink detection) is NOT used."""
    eroded_path = f"eroded_inklabels/{zid}.png"
    if os.path.exists(eroded_path):
        # check it's not empty/corrupt
        try:
            img = np.array(Image.open(eroded_path).convert("L"))
            valid_frac = (img > 0).mean()
            print(f"  [2/3] eroded labels present: {eroded_path}  valid={valid_frac:.4f}")
            return
        except Exception as e:
            print(f"  [2/3] \033[91m!! ERROR: eroded label exists but failed to load: {e}\033[0m")
            return
    
    # MISSING eroded labels -> big red warning
    print(f"  [2/3] \033[91m{'='*60}")
    print(f"  !! WARNING: MISSING ERODED LABEL FILE !!")
    print(f"  Expected: {eroded_path}")
    print(f"  This fragment CANNOT be used for training without labels.")
    print(f"  Eroded labels must be pre-generated (e.g., from 1um ink detection)")
    print(f"  and placed in eroded_inklabels/ before running training.")
    print(f"  {'='*60}\033[0m")


def step3_norm(name, seg, zid, skip, force=False):
    if skip:
        print(f"  [3/3] --skip-norm -> skip")
        return
    import json
    if not force and os.path.exists("norm_cache.json"):
        try:
            if zid in json.load(open("norm_cache.json")):
                print(f"  [3/3] norm cached -> skip")
                return
        except Exception:
            pass
    run([sys.executable, "precompute_norm.py", "--scroll-id", zid, "--zarr-path", ZARR_DIR])


def process_fragment(name, seg, zid, workers, skip_norm, chunk_depth, chunk_y, chunk_x, prefix="", force=False):
    """run the full assembly pipeline for one fragment. isolated in try/except so a
    single failure never kills a concurrent batch. respects FRAG_OPTS (skip_labels)."""
    opts = FRAG_OPTS.get(name, {})
    tag = f"{prefix}{name} ({zid})"
    try:
        print(f"\n{'='*70}\n{tag}  id={zid}\n{'='*70}", flush=True)
        step1_volume(name, seg, zid, workers, chunk_depth, chunk_y, chunk_x, force=force)
        if opts.get("dlash_surface_base"):
            if name not in RESCAN_88KEV:
                _build_dlash_mask(name, zid, opts)
            _verify_dlash_outputs(name, zid, opts)
        if not opts.get("skip_labels"):
            step2_check_eroded_labels(name, seg, zid)
        else:
            print(f"  [labels] skip_labels -> keeping existing eroded_inklabels")
        step3_norm(
            name,
            seg,
            zid,
            skip_norm,
            force=force or bool(opts.get("force_norm", False)),
        )
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
    ap.add_argument("--workers", type=int, default=32,
                    help="parallel S3 chunk-download workers PER fragment (default 32 for EPYC 7702)")
    ap.add_argument("--concurrent-fragments", type=int, default=5,
                    help="number of fragments to assemble in parallel (default 5). "
                         "each uses --workers download threads, so total connections "
                         "= concurrent_fragments * workers; watch RAM (~2-5GB per fragment).")
    ap.add_argument("--skip-norm", action="store_true")
    ap.add_argument("--force", action="store_true",
                    help="re-download/re-render + recompute norm even if outputs exist; also clears "
                         "the per-fragment chunk cache for a clean S3 re-fetch (use to fix a corrupted zarr)")
    ap.add_argument("--chunk-depth", type=int, default=DEFAULT_CHUNK_DEPTH,
                    help=f"zarr depth chunk size (default {DEFAULT_CHUNK_DEPTH}, optimized for 8-slice windows)")
    ap.add_argument("--chunk-y", type=int, default=DEFAULT_CHUNK_Y,
                    help=f"zarr Y chunk size (default {DEFAULT_CHUNK_Y})")
    ap.add_argument("--chunk-x", type=int, default=DEFAULT_CHUNK_X,
                    help=f"zarr X chunk size (default {DEFAULT_CHUNK_X})")
    args = ap.parse_args()

    segs = SEGMENTS
    if args.only:
        # comma-separated list so several fragments can be forced in one run
        want = {o.strip() for o in args.only.split(",") if o.strip()}
        segs = [s for s in SEGMENTS if s[0] in want]
    elif args.from_name:
        names = [s[0] for s in SEGMENTS]
        segs = SEGMENTS[names.index(args.from_name):]

    cf = max(1, int(args.concurrent_fragments))
    print(f"[assemble] {len(segs)} fragment(s): {[s[0] for s in segs]}  "
          f"(concurrent_fragments={cf}, workers/frag={args.workers}, "
          f"chunks=({args.chunk_depth},{args.chunk_y},{args.chunk_x}))")

    results = []
    if cf == 1:
        for name, seg, zid in segs:
            results.append(process_fragment(name, seg, zid, args.workers, args.skip_norm,
                                          args.chunk_depth, args.chunk_y, args.chunk_x,
                                          force=args.force))
    else:
        from concurrent.futures import ThreadPoolExecutor, as_completed
        with ThreadPoolExecutor(max_workers=cf) as ex:
            futs = {ex.submit(process_fragment, name, seg, zid, args.workers,
                              args.skip_norm, args.chunk_depth, args.chunk_y, args.chunk_x,
                              prefix=f"[{name}] ", force=args.force): name
                    for name, seg, zid in segs}
            for fut in as_completed(futs):
                results.append(fut.result())

    print(f"\n{'='*70}\n[assemble] SUMMARY\n{'='*70}")
    for nm, status in results:
        print(f"  {nm}: {status}")

    # HARDENING: integrity-sweep EVERY scroll (not just the ones touched this run) so a
    # partial/interrupted zarr that 'exists -> skip' silently kept gets caught and reported.
    print(f"\n{'='*70}\n[assemble] INTEGRITY SWEEP ({len(SEGMENTS)} scrolls)\n{'='*70}")
    broken = []
    for name, _seg, zid in SEGMENTS:
        ok, msg = _zarr_integrity(zid)
        if ok:
            print(f"  OK   {name:<12} {zid}  {msg}")
        else:
            print(f"  \033[91mFAIL {name:<12} {zid}  {msg}\033[0m")
            broken.append(name)
    if broken:
        ids = ",".join(broken)
        print(f"\n\033[91m[assemble] {len(broken)} scroll(s) FAILED integrity: {broken}")
        print(f"  fix with: python assemble_training_segments.py --force --only {ids}\033[0m")
    else:
        print(f"[assemble] integrity sweep: all {len(SEGMENTS)} scrolls OK")


if __name__ == "__main__":
    main()
