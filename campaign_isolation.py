"""campaign_isolation.py -- leave-one-out (LOO) label-correction campaign.

we have 14 PHerc0139 training fragments. for EACH one, train a model on the OTHER 13 and
run inference on the held-out fragment (which the model never saw) so its inklabels can be
re-annotated from an unbiased prediction. 14 runs, in sequence, each holding out a different
fragment.

config = tsI (medium regularization) from campaign_runner_twostage.py, with these exceptions:
  - n_epochs   = 20
  - eval_int   = 20     (eval figures render once, at the final epoch)
  - test_int   = 9999   (NO periodic test during training)
the held-out fragment is still inferred ONCE at the final epoch via tra.test_on_final=True.

VISUALIZATIONS  (./output/visualizations/<exp_name>/ -- each run gets its own folder):
  - the HELD-OUT fragment's full-size inference figure is ALWAYS saved here (this is what
    you use to re-annotate). independent of --save-vis.
  - with --save-vis, the 13 training-scroll eval figures (prediction + gold inklabel overlay)
    are ALSO saved here. off by default (TensorBoard-only) since 13 figures per run is a lot.

usage:
  python campaign_isolation.py                # all 14 LOO runs (holdout figure only)
  python campaign_isolation.py --save-vis     # also dump the 13 eval figures per run
  python campaign_isolation.py --only w058    # a single holdout (by w-name or zarr id)
  python campaign_isolation.py --from w046    # resume from a given holdout onward
  python campaign_isolation.py --dry-run      # print the plan, run nothing
"""
from __future__ import annotations
import argparse, gc, os, sys, time, traceback
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
os.environ.setdefault("TF_ENABLE_ONEDNN_OPTS", "0")
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")

from utils.config import Config, DEFAULT_SCROLLS, ScrollConfig
from utils.platform import get_zarr_dir, get_default_batch_size, get_default_eval_bs, get_default_workers

INTER_RUN_COOLDOWN_SECS = 120

# zarr id -> short fragment name (for run naming + output folders)
WNAME = {
    20260115000000: "w044", 20250223000000: "w059", 20260206000001: "w047",
    20260115000001: "w056", 20260210000000: "w058", 20260227000000: "w052",
    20260318000000: "w049", 20260325000000: "w046", 20260108000000: "w041",
    20250831000000: "w040", 20260302000000: "w039", 20260306000000: "w038",
    20260310000000: "w037", 20260303000000: "w034",
    20260226000000: "seg46527",
}

# canonical 14 training fragments = the active DEFAULT_SCROLLS (13) + w058 (currently held out
# of DEFAULT_SCROLLS as a sanity fragment). deduped by id so this stays correct even if w058
# is later re-added to DEFAULT_SCROLLS.
_W058 = ScrollConfig(20260210000000, split_axis="x", train_split_frac=0.75)
_by_id = {}
for _s in list(DEFAULT_SCROLLS) + [_W058]:
    _by_id[int(_s.scroll_id)] = _s
FRAGMENTS = list(_by_id.values())   # 14 ScrollConfig objects


def _base_config(exp_name: str) -> Config:
    """tsI (medium-reg) two-stage config, with the isolation-campaign exceptions applied."""
    c = Config()
    c.exp_name = exp_name
    
    # Set platform-aware zarr path
    c.data.zarr_path = get_zarr_dir()
    
    # --- architecture (tsI) ---
    c.model.arch         = "v15_twostage_wide_zgrad"
    c.data.tile_size     = 16
    c.data.depth         = 24
    c.data.train_d_start = 4
    c.data.train_d_end   = 28
    c.data.d_start       = 4
    c.data.d_end         = 28
    c.model.conv1_drop   = 0.1     # tsI c1_drop
    c.model.conv2_drop   = 0.1     # tsI c2_drop
    c.model.head_drop    = 0.3     # tsI h_drop
    # --- training schedule (ISOLATION EXCEPTIONS: 20 epochs, eval@20, no periodic test) ---
    c.tra.n_epochs       = 20
    c.tra.eval_int       = 20
    c.tra.test_int       = 9999
    c.tra.test_on_final  = True    # still infer the held-out fragment ONCE at the final epoch
    c.tra.probe_int      = 10      # tsI value (probe ROIs only exist for a few scrolls -> cheap)
    c.tra.save_int       = 2       # frequent checkpoints (BSODs ongoing)
    c.tra.log_dir        = "./runs_isolation"
    c.tra.deterministic  = False
    # --- loss / reg (tsI) ---
    c.tra.l1_lambda      = 1e-5
    c.tra.weight_decay   = 0.0
    c.tra.ranking_lambda = 0.5
    c.tra.ranking_neg_frac = 1.0
    # --- dataloader + augmentation (tsI medreg) ---
    c.dl.batch_size      = get_default_batch_size()
    c.dl.num_workers     = get_default_workers()
    c.dl.data_aug        = True
    c.dl.channel_mixing_prob = 0.0
    c.dl.flip_prob       = 0.4
    c.dl.rotation_prob   = 0.4
    c.dl.noise_prob      = 0.05
    c.dl.brightness_prob = 0.1
    c.dl.contrast_prob   = 0.1
    c.dl.cutout_prob     = 0.2
    c.dl.cutout_max_frac = 0.15
    c.dl.cutout_n_patches = 2
    c.dl.depth_mask_prob = 0.0
    # --- data plumbing ---
    c.data.mask_memmap       = True
    c.data.ring_negatives    = True
    c.data.ring_label_source = "closed"   # closed ring off (hand-cleaned) eroded map
    c.data.ring_close_r      = 3
    c.data.ring_gap_r        = 3
    c.data.ring_shell_r      = 2
    # --- thermal cooldowns (same as twostage) ---
    c.tra.epoch_cooldown_secs   = 0 if on_linux else 9
    c.tra.val_cooldown_secs     = 0 if on_linux else 12
    c.tra.eval_cooldown_secs    = 0 if on_linux else 60
    c.tra.fig_chunk_cooldown_ms = 0 if on_linux else 60
    c.data.eval_infer_bs = get_default_eval_bs()
    return c


def build_config(holdout: ScrollConfig, save_vis: bool) -> Config:
    """config for one LOO run: train on the other 13, infer on `holdout`."""
    hid   = int(holdout.scroll_id)
    wname = WNAME.get(hid, str(hid))
    c = _base_config(f"cmp_isolation_holdout_{wname}_{hid}")
    c.tra.save_vis = bool(save_vis)
    # train on the other 13 fragments; hold this one out entirely
    c.data.scrolls = [s for s in FRAGMENTS if int(s.scroll_id) != hid]
    # infer on the held-out fragment (rendered once at the final epoch, ALWAYS saved)
    c.data.test_scroll_ids    = []
    c.data.holdout_scroll_ids = [hid]
    os.makedirs("models/isolation", exist_ok=True)
    c.save_final = f"models/isolation/isolation_holdout_{wname}_{hid}_final.pth"
    return c


def cooldown(secs: int, label: str):
    if secs > 0:
        print(f"[COOLDOWN] {label} {secs}s ...", flush=True)
        time.sleep(secs)


def run_holdout(holdout: ScrollConfig, save_vis: bool, dry_run: bool) -> bool:
    hid   = int(holdout.scroll_id)
    wname = WNAME.get(hid, str(hid))
    c = build_config(holdout, save_vis)
    train_ids = [int(s.scroll_id) for s in c.data.scrolls]
    print(f"\n{'='*70}\n[isolation] HOLDOUT {wname} ({hid})\n{'='*70}", flush=True)
    print(f"  train on 13: {[WNAME.get(i, i) for i in train_ids]}")
    print(f"  arch={c.model.arch}  n_epochs={c.tra.n_epochs}  eval_int={c.tra.eval_int}  "
          f"test_int={c.tra.test_int}  test_on_final={c.tra.test_on_final}")
    print(f"  save_vis={c.tra.save_vis}  -> ./output/visualizations/{c.exp_name}/")
    print(f"  save_final={c.save_final}")
    if dry_run:
        print("  [DRY RUN] skipping")
        return True
    from train import Trainer
    try:
        trainer = Trainer(c)
        trainer.run()
        return True
    except Exception:
        print("[ERROR] training raised an exception:", flush=True)
        traceback.print_exc()
        return False
    finally:
        gc.collect()
        try:
            import torch
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        except Exception:
            pass


def _resolve(token: str) -> int | None:
    """map a --only/--from token (w-name or zarr id) to a fragment id present in FRAGMENTS."""
    ids = {int(s.scroll_id) for s in FRAGMENTS}
    name_to_id = {v: k for k, v in WNAME.items()}
    if token in name_to_id and name_to_id[token] in ids:
        return name_to_id[token]
    try:
        if int(token) in ids:
            return int(token)
    except ValueError:
        pass
    return None


def main():
    ap = argparse.ArgumentParser(description="leave-one-out label-correction campaign")
    ap.add_argument("--save-vis", action="store_true",
                    help="also save the 13 training-scroll eval figures per run (holdout figure always saves)")
    ap.add_argument("--only", type=str, default=None, help="run a single holdout (w-name or zarr id)")
    ap.add_argument("--from", dest="from_id", type=str, default=None, help="resume from this holdout onward")
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    order = list(FRAGMENTS)   # LOO order = FRAGMENTS order

    if args.only:
        rid = _resolve(args.only)
        if rid is None:
            print(f"[ABORT] --only '{args.only}' not found; valid: {[WNAME[int(s.scroll_id)] for s in FRAGMENTS]}")
            return
        order = [s for s in FRAGMENTS if int(s.scroll_id) == rid]
    elif args.from_id:
        rid = _resolve(args.from_id)
        if rid is None:
            print(f"[ABORT] --from '{args.from_id}' not found; valid: {[WNAME[int(s.scroll_id)] for s in FRAGMENTS]}")
            return
        start = next(i for i, s in enumerate(FRAGMENTS) if int(s.scroll_id) == rid)
        order = FRAGMENTS[start:]

    names = [WNAME.get(int(s.scroll_id), int(s.scroll_id)) for s in order]
    print(f"[isolation] {len(order)} LOO run(s): {names}  save_vis={args.save_vis}  dry_run={args.dry_run}")

    for i, holdout in enumerate(order):
        ok = run_holdout(holdout, args.save_vis, args.dry_run)
        if not ok:
            print(f"[isolation] holdout {WNAME.get(int(holdout.scroll_id))} FAILED; continuing to next.")
        if not args.dry_run and i < len(order) - 1:
            cooldown(INTER_RUN_COOLDOWN_SECS, "inter-run")

    print("\n[isolation] campaign complete.")


if __name__ == "__main__":
    main()
