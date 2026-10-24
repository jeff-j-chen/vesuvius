"""asf_morph cue maps for the depth-MIL pair views: the primary window shifted by -2 and +2 slices.

Depth-MIL reads a second view of the same block 2 slices above/below the primary one, but the cue slice is the single map
of the primary window. These maps give each pair view its own cue (cue_asf_morph_dm2 / _dp2, same file names as cue_asf_morph).

    python3 build_pair_cues.py [--workers 4] [--force]
"""
import os
import sys
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

REPO = Path(__file__).resolve().parent
sys.path[:0] = [str(REPO), str(REPO / "old")]
os.chdir(REPO)

DELTAS = {-2: "./cue_asf_morph_dm2", -1: "./cue_asf_morph_dm1", 1: "./cue_asf_morph_dp1", 2: "./cue_asf_morph_dp2"}


def corpus():
    import train_iter2
    from utils.config import Config

    config = Config()
    train_iter2.configure(config, "idea2_depth_mil")
    ids = [int(s) for s in config.scroll_ids()]
    offsets = {i: int(config.data.surface_window_offset_by_scroll.get(i, config.data.surface_window_offset)) for i in ids}
    return config, ids, offsets


def build(job):
    scroll_id, offset, delta, force = job
    from utils import precomputed_asf
    from utils.config import Config

    import train_iter2

    config = Config()
    train_iter2.configure(config, "idea2_depth_mil")
    config.data.surface_window_offset_by_scroll = {scroll_id: offset + delta}
    config.data.cue_dir = DELTAS[delta]
    return f"{scroll_id} delta {delta:+d} (offset {offset + delta:+d}): " + precomputed_asf.build(scroll_id, force=force, config=config)


if __name__ == "__main__":
    workers = int(sys.argv[sys.argv.index("--workers") + 1]) if "--workers" in sys.argv else 4
    _, ids, offsets = corpus()
    print("training scrolls and primary window offsets:", offsets, flush=True)
    jobs = [(i, offsets[i], d, "--force" in sys.argv) for i in ids for d in DELTAS]
    with ProcessPoolExecutor(max_workers=workers) as pool:
        for line in pool.map(build, jobs):
            print("[pair-cue]", line, flush=True)
