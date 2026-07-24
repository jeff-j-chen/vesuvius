"""download_surface_zarr.py — download a pre-rendered OME-Zarr surface-volume from S3.

these segment surface-volumes are ALREADY flattened (no mesh sampling needed), stored as
OME-Zarr: a group with pyramid levels 0..5, each an array with chunks [D,128,128], uint8,
NO compressor, dimension_separator "/". a chunk lives at  {base}.zarr/{level}/{zc}/{yc}/{xc}.
the chunk depth == full D, so zc is always 0; missing chunk (404) = all-air background.

two modes:
  volume   : download a full level -> local training zarr (D,H,W) uint16 (our train format,
             chunks (min(8,D),32,32), compressor=None, zarr_format=2) + masks/<id>.png footprint.
  midslice : download a level but keep only the middle z-plane -> png (+ mask png). cheap: reads
             only the mid plane out of each chunk, so RAM stays tiny even for big levels.

usage:
  # 9.4um w059 training volume (level 0), -> ves_zarrs2/<id>.zarr + masks/<id>.png
  python download_surface_zarr.py --mode volume --level 0 --out-id 20250223000000 \
     --url ".../9.362um-1.2m-113keV-volume-20250728140407.zarr"

  # 1.1um w059 midslice for overlay (use a downsampled level to stay cheap)
  python download_surface_zarr.py --mode midslice --level 3 \
     --out-png C:/Users/ChenJeff/Documents/_ves_tmp/w059_1p1um_midslice.png \
     --url ".../1.129um-0.22m-59keV-volume-20260413113053-L1.zarr"
"""
from __future__ import annotations
import argparse, json, os, shutil, subprocess, sys
from concurrent.futures import ThreadPoolExecutor
import numpy as np
from PIL import Image
Image.MAX_IMAGE_PIXELS = None

CHUNK_XY = 128


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


def _fetch_chunk(args):
    """download one chunk file to cache. 404 (air) -> empty sentinel. returns (yc,xc,status)."""
    base, level, yc, xc, cache_dir = args
    out = os.path.join(cache_dir, f"{level}_{yc}_{xc}.raw")
    if os.path.exists(out):
        return (yc, xc, "cached")
    url = f"{base}/{level}/0/{yc}/{xc}"
    # NO --retry-all-errors: a 404 here is an EXPECTED air chunk (sparse surface in a rect bbox);
    # retrying it burned ~4x2s each and dominated runtime. --retry still covers transient 5xx/timeouts.
    r = subprocess.run(
        ["curl", "-s", "--fail", "--connect-timeout", "20", "--max-time", "120",
         "--retry", "3", "--retry-delay", "1", url, "-o", out],
        capture_output=True)
    if r.returncode != 0:
        open(out, "wb").close()   # air sentinel
        return (yc, xc, "air")
    return (yc, xc, "ok")


def _load_chunk(cache_dir, level, yc, xc, D):
    p = os.path.join(cache_dir, f"{level}_{yc}_{xc}.raw")
    try:
        if os.path.getsize(p) == D * CHUNK_XY * CHUNK_XY:
            return np.frombuffer(open(p, "rb").read(), dtype=np.uint8).reshape(D, CHUNK_XY, CHUNK_XY)
    except Exception:
        pass
    return None


def _aria2_fetch(base, level, jobs, cache_dir, workers):
    """batch-download all chunks with ONE aria2c process (connection reuse + -j parallel files).
    hugely faster than spawning one curl per tiny chunk. returns False if aria2c is not on PATH
    (caller falls back to the curl loop). a 404 = expected air chunk -> gets a zero-byte sentinel
    so re-runs skip it and the assembler treats it as air."""
    if shutil.which("aria2c") is None:
        return False
    # queue only chunks not already cached (a real file OR an air sentinel both count as done)
    todo = [(f"{base}/{level}/0/{yc}/{xc}", f"{level}_{yc}_{xc}.raw")
            for (_b, _lvl, yc, xc, _cd) in jobs
            if not os.path.exists(os.path.join(cache_dir, f"{level}_{yc}_{xc}.raw"))]
    if todo:
        listfile = os.path.join(cache_dir, "_aria2_urls.txt")
        with open(listfile, "w") as f:
            for url, name in todo:
                f.write(f"{url}\n  dir={cache_dir}\n  out={name}\n")
        # -j parallel files; -x1/-s1 (chunks are tiny, no splitting). -q + --download-result=hide
        # SILENCE aria2c: most chunks 404 (= expected air), and its per-URL error lines + the final
        # results table would spam thousands of lines. we print a one-line summary below instead.
        j = max(1, min(int(workers), 64))
        print(f"[dl] aria2c: fetching {len(todo)} chunks with -j{j} (404s = air, output silenced)...", flush=True)
        subprocess.run(
            ["aria2c", "-i", listfile, f"-j{j}", "-x1", "-s1",
             "--max-tries=2", "--retry-wait=1", "--connect-timeout=20", "--timeout=120",
             "--auto-file-renaming=false", "--allow-overwrite=true",
             "-q", "--download-result=hide"],
            check=False)
        try:
            os.remove(listfile)
        except OSError:
            pass
    # zero-byte air sentinel for anything still missing; tally real vs air for a one-line summary
    real = air = 0
    for (_b, _lvl, yc, xc, _cd) in jobs:
        p = os.path.join(cache_dir, f"{level}_{yc}_{xc}.raw")
        if not os.path.exists(p):
            open(p, "wb").close()
            air += 1
        elif os.path.getsize(p) > 0:
            real += 1
        else:
            air += 1
    print(f"[dl] aria2c done: {real} data chunks + {air} air (blank) chunks", flush=True)
    return True


def download(base, level, mode, out_zarr, out_png, out_id, workers, cache_dir):
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
    if _aria2_fetch(base, level, jobs, cache_dir, workers):
        print("[dl] aria2c batch fetch complete", flush=True)
    else:
        done = 0
        with ThreadPoolExecutor(max_workers=workers) as ex:
            for _ in ex.map(_fetch_chunk, jobs):
                done += 1
                if done % 500 == 0:
                    print(f"[dl] fetched {done}/{len(jobs)}", flush=True)
    print(f"[dl] all {len(jobs)} chunks present", flush=True)

    if mode == "midslice":
        zc_plane = D // 2
        out2d = np.zeros((H, W), dtype=np.uint8)
        for yc in range(n_yc):
            for xc in range(n_xc):
                ch = _load_chunk(cache_dir, level, yc, xc, D)
                if ch is None:
                    continue
                y0, x0 = yc * CHUNK_XY, xc * CHUNK_XY
                y1, x1 = min(y0 + CHUNK_XY, H), min(x0 + CHUNK_XY, W)
                out2d[y0:y1, x0:x1] = ch[zc_plane, :y1 - y0, :x1 - x0]
        os.makedirs(os.path.dirname(out_png) or ".", exist_ok=True)
        Image.fromarray(out2d).save(out_png)
        mp = os.path.splitext(out_png)[0] + "_mask.png"
        Image.fromarray(((out2d > 0).astype(np.uint8) * 255)).save(mp)
        print(f"[dl] wrote midslice {out_png} ({H}x{W}) z={zc_plane} + mask {mp}")
        return

    # volume mode -> assemble full uint8 (D,H,W), write local training zarr uint16 + mask
    import zarr
    vol = np.zeros((D, H, W), dtype=np.uint8)
    upd, close = _pbar(n_yc * n_xc, "assembling chunks")
    for yc in range(n_yc):
        for xc in range(n_xc):
            ch = _load_chunk(cache_dir, level, yc, xc, D)
            upd()
            if ch is None:
                continue
            y0, x0 = yc * CHUNK_XY, xc * CHUNK_XY
            y1, x1 = min(y0 + CHUNK_XY, H), min(x0 + CHUNK_XY, W)
            vol[:, y0:y1, x0:x1] = ch[:, :y1 - y0, :x1 - x0]
    close()
    store = zarr.open(out_zarr, mode="w", shape=(D, H, W), chunks=(min(8, D), 32, 32),
                      dtype="<u2", compressor=None, zarr_format=2)
    upd, close = _pbar(D, "writing zarr layers")
    for li in range(D):
        store[li] = vol[li].astype(np.uint16)
        upd()
    close()
    print(f"[dl] wrote zarr {out_zarr} ({D},{H},{W}) uint16")

    # mask = footprint where ANY layer is nonzero (valid rendered surface)
    mask = (vol.max(axis=0) > 0).astype(np.uint8) * 255
    os.makedirs("masks", exist_ok=True)
    Image.fromarray(mask).save(f"masks/{out_id}.png")
    print(f"[dl] wrote masks/{out_id}.png  valid_frac={(mask > 0).mean():.3f}")


def main():
    ap = argparse.ArgumentParser(description="download a pre-rendered OME-Zarr surface volume")
    ap.add_argument("--url", required=True, help="base .zarr url (no trailing level)")
    ap.add_argument("--level", type=int, default=0)
    ap.add_argument("--mode", choices=["volume", "midslice"], default="volume")
    ap.add_argument("--out-zarr", default=None, help="volume mode: local zarr path (default ves_zarrs2/<id>.zarr)")
    ap.add_argument("--out-png", default=None, help="midslice mode: output png path")
    ap.add_argument("--out-id", default=None, help="volume mode: id for masks/<id>.png + default zarr name")
    ap.add_argument("--workers", type=int, default=16)
    ap.add_argument("--cache-dir", default=None, help="chunk cache dir (default _ves_tmp/dl_<id or level>)")
    args = ap.parse_args()

    if args.mode == "volume":
        if not args.out_id:
            print("[ABORT] volume mode needs --out-id"); return
        out_zarr = args.out_zarr or rf"C:\Users\ChenJeff\Documents\ves_zarrs2\{args.out_id}.zarr"
        cache = args.cache_dir or rf"C:\Users\ChenJeff\Documents\_ves_tmp\dl_{args.out_id}_L{args.level}"
        download(args.url, args.level, "volume", out_zarr, None, args.out_id, args.workers, cache)
    else:
        if not args.out_png:
            print("[ABORT] midslice mode needs --out-png"); return
        tag = os.path.splitext(os.path.basename(args.out_png))[0]
        cache = args.cache_dir or rf"C:\Users\ChenJeff\Documents\_ves_tmp\dl_{tag}_L{args.level}"
        download(args.url, args.level, "midslice", None, args.out_png, None, args.workers, cache)


if __name__ == "__main__":
    main()
