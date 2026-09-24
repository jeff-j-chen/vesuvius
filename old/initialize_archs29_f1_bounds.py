"""create the TensorBoard helper run that anchors patch F1 charts to 0..1."""

from pathlib import Path
import shutil

from torch.utils.tensorboard import SummaryWriter

from campaign_archs_29 import CAMPAIGN28_SCROLLS, LOG_DIR
from utils.visualizer import _patch_character_f1_layout


RUN_NAME = "f1_bounds_0_to_1"


def initialize_bounds_run() -> Path:
    log_dir = Path(LOG_DIR) / RUN_NAME
    if log_dir.exists():
        shutil.rmtree(log_dir)
    patch_ids = list(dict.fromkeys(
        int(scroll.scroll_id) for scroll in CAMPAIGN28_SCROLLS
    ))
    layout = {
        "Patch_Character_F1": _patch_character_f1_layout(CAMPAIGN28_SCROLLS),
    }

    writer = SummaryWriter(str(log_dir))
    writer.add_custom_scalars(layout)
    for patch_id in patch_ids:
        writer.add_scalar(f"Character/Patch/{patch_id}/F1/Train", 1.0, 0)
        writer.add_scalar(f"Character/Patch/{patch_id}/F1/Valid", 0.0, 0)
    writer.flush()
    writer.close()

    print(
        f"initialized {log_dir} with 0..1 F1 anchors for "
        f"{len(patch_ids)} patches at epoch 1"
    )
    return log_dir


if __name__ == "__main__":
    initialize_bounds_run()