"""queue-driven test-fragment evaluation using test_inference.ipynb's own setup/helper cells.

    python3 old/test_eval.py enqueue <zid>        # assembler hook: queue one finished fragment
    python3 old/test_eval.py end                  # no more fragments will arrive
    python3 old/test_eval.py tidy                 # free disk if low (finished render caches, dl_ dirs)
    python3 old/test_eval.py serve <checkpoint>   # evaluate every queued fragment in order, then exit at end
    python3 old/test_eval.py serve-ensemble       # test_inference_and_letters.ipynb: both models, shared z-scores
"""
import fcntl
import os
import shutil
import sys
import time
import traceback
from contextlib import contextmanager
from pathlib import Path

REPO = Path("/vesuvius")
STATE = Path(os.getenv("VES_EVAL_STATE", str(REPO / "_ves_tmp" / "test_eval")))
QUEUE = STATE / "queue.txt"
PREPARE_LOCK = STATE / "prepare.lock"
END = "__END__"
TIDY_BELOW_GB = 60.0
# cells 2-8 of test_inference.ipynb: run selection, imports, composites, figure builder, model, readiness
SETUP_CELLS = range(1, 8)


@contextmanager
def locked(path, mode):
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "a+") as handle:
        fcntl.flock(handle, mode)
        try:
            yield handle
        finally:
            fcntl.flock(handle, fcntl.LOCK_UN)


def enqueue(zid):
    with locked(QUEUE, fcntl.LOCK_EX) as handle:
        handle.write(f"{zid}\n")
    print(f"[eval-queue] queued {zid}", flush=True)


def read_queue():
    if not QUEUE.exists():
        return []
    with locked(QUEUE, fcntl.LOCK_SH) as handle:
        handle.seek(0)
        return [line.strip() for line in handle if line.strip()]


def tidy():
    free = shutil.disk_usage(REPO).free / 1e9
    if free >= TIDY_BELOW_GB:
        return
    zarr_dir = REPO / "ves_zarrs2"
    removed = []
    for path in sorted((REPO / "_ves_tmp").glob("render_*")):
        zid = path.name.removeprefix("render_")
        finished = (zarr_dir / f"{zid}.zarr").is_dir() and (
            (REPO / "masks" / f"{zid}.png").is_file() or (REPO / "masks" / "extras" / f"{zid}.png").is_file()
        )
        if finished:
            shutil.rmtree(path, ignore_errors=True)
            removed.append(path.name)
    for path in sorted((REPO / "_ves_tmp").glob("dl_*")):
        if path.is_dir():
            shutil.rmtree(path, ignore_errors=True)
            removed.append(path.name)
    after = shutil.disk_usage(REPO).free / 1e9
    print(f"[tidy] free {free:.0f} GB < {TIDY_BELOW_GB:.0f} GB: removed {len(removed)} temp dirs -> {after:.0f} GB free",
          flush=True)


def notebook_namespace(checkpoint):
    import matplotlib
    import nbformat

    matplotlib.use("Agg")
    os.environ["VESUVIUS_CHECKPOINT"] = str(checkpoint)
    notebook = nbformat.read(REPO / "test_inference.ipynb", as_version=4)
    code = [cell for cell in notebook.cells if cell.cell_type == "code"]
    namespace = {"__name__": "__main__"}
    for index in SETUP_CELLS:
        exec(compile(code[index - 1].source, f"test_inference.ipynb:cell{index + 1}", "exec"), namespace)
    return namespace


def serve(checkpoint):
    sys.path.insert(0, str(REPO))
    import assemble_test_segments as assembler

    mesh_names = {zid: mesh for zid, mesh, *_ in assembler.FRAGMENTS}
    ns = notebook_namespace(checkpoint)
    run_id = ns["RUN_ID"]
    names = {str(sid): name for name, sid in ns["FRAGMENTS"].items()}
    extras_dir = os.path.join(ns["OUTPUT_DIR"], "test_visualizations", run_id, ns["EXTRAS_SUBDIR"])
    done_path = STATE / f"done_{run_id}.txt"
    done = set(done_path.read_text().split()) if done_path.exists() else set()
    print(f"[eval {run_id}] serving {checkpoint}; {len(done)} fragment(s) already done", flush=True)
    while True:
        queue = read_queue()
        pending = [zid for zid in dict.fromkeys(queue) if zid != END and zid not in done]
        if not pending:
            if END in queue:
                print(f"[eval {run_id}] queue finished: {len(done)} fragment(s) evaluated", flush=True)
                return
            time.sleep(20)
            continue
        zid = pending[0]
        extra = not zid.isdigit()
        scroll_id = zid if extra else int(zid)
        start = time.time()
        try:
            with locked(PREPARE_LOCK, fcntl.LOCK_EX):
                ns["ensure_fragment_ready"](scroll_id)
                mask = ns["np"].array(ns["Image"].open(ns["mask_path_for"](scroll_id)).convert("L"))
                ns["make_composite"](scroll_id, bounds=(0, mask.shape[0], 0, mask.shape[1]))
            with locked(PREPARE_LOCK, fcntl.LOCK_SH):
                if extra:
                    ns["render_fragment"](zid, zid, ns["MODEL"], ns["C"], fast_eval=True,
                                          fast_path=os.path.join(extras_dir, f"{zid}.png"), show=False,
                                          fast_vertical=True)
                else:
                    name = names.get(zid) or mesh_names.get(zid, zid)
                    ns["render_fragment"](name, scroll_id, ns["MODEL"], ns["C"], show=False)
            status = "ok"
        except Exception:
            traceback.print_exc()
            status = "FAILED"
        done.add(zid)
        with open(done_path, "a") as handle:
            handle.write(f"{zid}\n")
        print(f"[eval {run_id}] {zid}: {status} in {time.time() - start:.0f}s "
              f"({len(done)} done, {len(pending) - 1} waiting)", flush=True)


def serve_ensemble():
    """run test_inference_and_letters.ipynb's setup cells (both checkpoints, readiness helpers), then its
    render_fragment on every queued fragment: shared z-score ensemble, 2x2 figure under its OUT_DIR."""
    import matplotlib
    import nbformat

    sys.path.insert(0, str(REPO))
    import assemble_test_segments as assembler

    matplotlib.use("Agg")
    notebook = nbformat.read(REPO / "test_inference_and_letters.ipynb", as_version=4)
    code = [cell for cell in notebook.cells if cell.cell_type == "code"]
    ns = {"__name__": "__main__"}
    # code cells 1-8: checkpoints, run selection, imports, composites, ensemble figure, models, readiness
    for index in range(8):
        exec(compile(code[index].source, f"test_inference_and_letters.ipynb:code{index + 1}", "exec"), ns)
    mesh_names = {zid: mesh for zid, mesh, *_ in assembler.FRAGMENTS}
    names = {str(sid): name for name, sid in ns["FRAGMENTS"].items()}
    out_name = os.path.basename(ns["OUT_DIR"])
    done_path = STATE / f"done_{out_name}.txt"
    done = set(done_path.read_text().split()) if done_path.exists() else set()
    print(f"[eval {out_name}] serving {ns['MODEL_KEYS']} -> {ns['OUT_DIR']}; {len(done)} already done", flush=True)
    while True:
        queue = read_queue()
        pending = [zid for zid in dict.fromkeys(queue) if zid != END and zid not in done]
        if not pending:
            if END in queue:
                print(f"[eval {out_name}] queue finished: {len(done)} fragment(s) evaluated", flush=True)
                return
            time.sleep(20)
            continue
        zid = pending[0]
        extra = not zid.isdigit()
        scroll_id = zid if extra else int(zid)
        start = time.time()
        try:
            with locked(PREPARE_LOCK, fcntl.LOCK_EX):
                ns["ensure_fragment_ready"](scroll_id)
            if extra:
                ns["render_fragment"](zid, zid, out_path=os.path.join(ns["OUT_DIR"], ns["EXTRAS_SUBDIR"], f"{zid}.jpg"))
            else:
                ns["render_fragment"](names.get(zid) or mesh_names.get(zid, zid), scroll_id)
            status = "ok"
        except Exception:
            traceback.print_exc()
            status = "FAILED"
        done.add(zid)
        with open(done_path, "a") as handle:
            handle.write(f"{zid}\n")
        print(f"[eval {out_name}] {zid}: {status} in {time.time() - start:.0f}s "
              f"({len(done)} done, {len(pending) - 1} waiting)", flush=True)


if __name__ == "__main__":
    command = sys.argv[1]
    if command == "enqueue":
        enqueue(sys.argv[2])
    elif command == "end":
        enqueue(END)
    elif command == "tidy":
        tidy()
    elif command == "serve":
        serve(sys.argv[2])
    elif command == "serve-ensemble":
        serve_ensemble()
    else:
        raise SystemExit(__doc__)
