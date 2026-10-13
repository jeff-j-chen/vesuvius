"""Profile the main-process training step of a campaign_letters_scratch arm on a small domain subset.

Real batch size, workers and AND-mask; cProfile around train_epoch shows where the main thread spends time.

    python3 old/profile_train_step.py combo_patch_bag_strict [--samples 1000] [--workers 8]
"""
from __future__ import annotations

import argparse
import cProfile
import os
import pstats
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
os.environ["PYTORCH_NVML_BASED_CUDA_CHECK"] = "1"


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("arm")
    parser.add_argument("--samples", type=int, default=1000, help="max samples per patch")
    parser.add_argument("--workers", type=int, default=None)
    parser.add_argument("--out", default="/tmp/profile_train_step.prof")
    parser.add_argument("--sync-debug", action="store_true",
                        help="count GPU->host synchronising call sites in train.py instead of profiling")
    args = parser.parse_args()
    os.chdir(ROOT)
    import torch
    import campaign_letters_scratch as scratch
    from train import Trainer

    config = scratch.build_config(args.arm, smoke=True)
    weights = dict(zip(config.data.train_scroll_dict, config.data.train_scroll_weights))
    config.data.train_scroll_dict = {d: ids for d, ids in config.data.train_scroll_dict.items()
                                     if d in scratch.SMOKE_DOMAINS}
    config.data.train_scroll_weights = [weights[d] for d in config.data.train_scroll_dict]
    kept = {int(s) for ids in config.data.train_scroll_dict.values() for s in ids}
    config.data.scrolls = [s for s in config.data.scrolls if int(s.scroll_id) in kept]
    config.tra.dann_n_domains = len(config.data.train_scroll_dict)
    config.data.max_samples_per_epoch = args.samples
    config.data.vis_scroll_ids = []
    config.data.label_free_vis_scroll_ids = []
    config.tra.eval_int = config.tra.test_int = 10 ** 6
    config.tra.log_dir = "/tmp/profile_train_step_logs"
    if args.workers is not None:
        config.dl.num_workers = args.workers
    trainer = Trainer(config)
    if args.sync_debug:
        import collections
        import traceback
        import warnings
        sites = collections.Counter()

        def record(message, category, filename, lineno, file=None, line=None):
            if "synchroniz" not in str(message):
                return
            frames = [f for f in traceback.extract_stack() if f.filename.startswith(str(ROOT))]
            sites[" <- ".join(f"{Path(f.filename).name}:{f.lineno}" for f in frames[-3:])] += 1

        warnings.showwarning = record
        warnings.simplefilter("always")
        torch.cuda.set_sync_debug_mode("warn")
        try:
            trainer.train_epoch(None, epoch=0)
        finally:
            torch.cuda.set_sync_debug_mode(0)
            warnings.simplefilter("default")
            steps = len(trainer.train_loader)
            for site, count in sites.most_common(60):
                print(f"{count / steps:7.2f}/step  {site}")
        print(f"[sync-debug] {sum(sites.values()) / steps:.1f} synchronising calls per step", flush=True)
    try:
        steps = len(trainer.train_loader)
        torch.cuda.synchronize()
        profiler = cProfile.Profile()
        start = time.time()
        profiler.enable()
        trainer.train_epoch(None, epoch=0)
        torch.cuda.synchronize()
        profiler.disable()
        elapsed = time.time() - start
        print(f"[profile] {args.arm}: {steps} steps in {elapsed:.1f}s = {steps / elapsed:.2f} it/s "
              f"(batch {config.dl.batch_size}, workers {config.dl.num_workers})", flush=True)
        profiler.dump_stats(args.out)
        stats = pstats.Stats(profiler).sort_stats("tottime")
        stats.print_stats(25)
        stats.sort_stats("cumulative").print_stats(40)
    finally:
        trainer.close()


if __name__ == "__main__":
    main()
