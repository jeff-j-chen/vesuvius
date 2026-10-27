"""v8-in entry point (run with nader-fork/.venv): train_resnet3d.py with cuDNN autotuning and channels-last weights.

the input shape never changes, so cuDNN benchmark picks the fastest 3D conv algorithms once; 5D weights go channels_last_3d
(the 2D decoder's 4D weights channels_last) so mixed-precision convs run on tensor-core friendly layouts. Same maths.
  <venv python> v8in_launch.py <train_resnet3d.py arguments>
"""
import os
import sys

import pytorch_lightning as pl
import torch

LIB = os.path.join(os.path.dirname(os.path.abspath(__file__)), "nader-fork", "training", "lib")
sys.path.insert(0, os.path.abspath(os.environ.get("V8IN_TRAIN_LIB", LIB)))


class ChannelsLast(pl.Callback):
    def on_fit_start(self, trainer, pl_module):
        converted = 0
        for parameter in pl_module.parameters():
            fmt = {5: torch.channels_last_3d, 4: torch.channels_last}.get(parameter.dim())
            if fmt is not None:
                parameter.data = parameter.data.contiguous(memory_format=fmt)
                converted += 1
        print(f"[v8in_launch] {converted} conv weights in channels-last layout; cudnn.benchmark="
              f"{torch.backends.cudnn.benchmark}", flush=True)


class PreValidationCheckpoint(pl.Callback):
    """trained-epoch{N}.ckpt after the last optimizer step of every epoch, so a crash in validation keeps the weights."""

    def on_train_batch_end(self, trainer, pl_module, outputs, batch, batch_idx):
        if batch_idx + 1 != trainer.num_training_batches or trainer.checkpoint_callback is None:
            return
        path = os.path.join(trainer.checkpoint_callback.dirpath, f"trained-epoch{trainer.current_epoch}.ckpt")
        trainer.save_checkpoint(path)
        print(f"[v8in_launch] saved {path}", flush=True)


_init = pl.Trainer.__init__


def _trainer_init(self, *args, **kwargs):
    kwargs["benchmark"] = True
    kwargs["callbacks"] = [*(kwargs.get("callbacks") or []), ChannelsLast(), PreValidationCheckpoint()]
    _init(self, *args, **kwargs)


pl.Trainer.__init__ = _trainer_init
torch.backends.cudnn.benchmark = True

import train_resnet3d  # noqa: E402

if __name__ == "__main__":
    train_resnet3d.main()
