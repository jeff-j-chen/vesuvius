"""asf_morph cue maps (shift 0, -1, +1) for every patch the add-one domain runs of train_sequence.py train on.

Each patch is read on its learned surface (surface_labels/<id>), window offset 0 (+shift), from the volume the run reads
(<id>.matched for the fine scans). Existing maps are kept unless they are older than the patch's surface labels or mask.

    python3 build_domain_cues.py [--workers 6] [--check] [--skip id,id]
"""
import os
import sys
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

REPO = Path(__file__).resolve().parent
sys.path[:0] = [str(REPO), str(REPO / "old")]
os.chdir(REPO)

DIRS = {0: "./cue_asf_morph", -1: "./cue_asf_morph_dm1", 1: "./cue_asf_morph_dp1"}


def job_config(scroll_id, delta):
    from utils.config import Config

    config = Config()
    config.data.surface_window_offset_by_scroll = {scroll_id: int(config.data.surface_window_offset) + delta}
    config.data.cue_dir = DIRS[delta]
    return config


def output_path(scroll_id, delta):
    config = job_config(scroll_id, delta)
    volume_id = f"{scroll_id}{(config.data.zarr_suffix or {}).get(str(scroll_id), '')}"
    return Path(DIRS[delta]) / f"{volume_id}.npy"


def stale(scroll_id, path):
    sources = [Path("surface_labels") / str(scroll_id) / "depth.npy", Path("masks") / f"{scroll_id}.png"]
    return any(source.exists() and source.stat().st_mtime > path.stat().st_mtime for source in sources)


def build(job):
    scroll_id, delta, force = job
    from utils import precomputed_asf

    return f"{scroll_id} shift {delta:+d}: " + precomputed_asf.build(scroll_id, force=force, config=job_config(scroll_id, delta))


def main():
    import train_sequence

    jobs = []
    skip = {int(s) for s in sys.argv[sys.argv.index("--skip") + 1].split(",")} if "--skip" in sys.argv else set()
    for scroll_id in train_sequence.DOMAIN_PATCH_IDS:
        for delta in DIRS:
            path = output_path(scroll_id, delta)
            if scroll_id in skip and delta == 0 and path.exists():
                continue
            if not path.exists():
                jobs.append((scroll_id, delta, False))
                print(f"missing {path}", flush=True)
            elif stale(scroll_id, path):
                jobs.append((scroll_id, delta, True))
                print(f"stale   {path} (older than its surface labels or mask)", flush=True)
    print(f"{len(jobs)} cue maps to build", flush=True)
    if "--check" in sys.argv or not jobs:
        return
    workers = int(sys.argv[sys.argv.index("--workers") + 1]) if "--workers" in sys.argv else 6
    with ProcessPoolExecutor(max_workers=workers) as pool:
        for line in pool.map(build, jobs):
            print("[domain-cue]", line, flush=True)


if __name__ == "__main__":
    main()
