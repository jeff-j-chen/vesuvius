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
import argparse, json, os, shutil, subprocess, sys, time
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
    False if aria2c is not on PATH (caller uses the per-chunk curl path instead)."""
    if shutil.which("aria2c") is None:
        return False
    todo = [(f"{base}/{level}/0/{yc}/{xc}", f"{level}_{yc}_{xc}.raw")
            for (_b, _lvl, yc, xc, _cd) in jobs
            if not os.path.exists(os.path.join(cache_dir, f"{level}_{yc}_{xc}.raw"))]
    if not todo:
        return True
    listfile = os.path.join(cache_dir, "_aria2_urls.txt")
    with open(listfile, "w") as f:
        for url, name in todo:
            f.write(f"{url}\n  dir={cache_dir}\n  out={name}\n")
    j = max(1, min(int(workers), 64))
    print(f"[dl] aria2c: bulk-fetching {len(todo)} chunks with -j{j} (missing chunks verified below)...", flush=True)
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
    return True


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
    if not _aria2_fetch(base, level, jobs, cache_dir, workers):
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
