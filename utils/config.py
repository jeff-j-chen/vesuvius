from dataclasses import dataclass, field
from typing import Optional
import os
import torch

@dataclass
class DataConfig:
    zarr_path: str = field(default_factory=lambda: os.getenv("VESUVIUS_ZARR_PATH", "C:\\Users\\ChenJeff\\Documents\\ves_zarrs2"))
    scroll1_id: int = 20230827161847
    # scroll1_id: int = 20230702185753
    scroll4_id: int = 20231210132040
    tile_size: int = 32
    depth: int = 8
    d_start: int = 28
    d_end: int = 48

@dataclass
class DataloaderConfig:
    batch_size: int = 96
    num_workers: int = 2
    data_aug: bool = True
    channel_mixing_prob: float = 0.25
    rotation_prob: float = 0.25
    flip_prob: float = 0.25
    noise_prob: float = 0.30
    brightness_prob: float = 0.50
    contrast_prob: float = 0.50

@dataclass
class TrainingConfig:
    n_epochs: int = 30
    lr: float = 1e-4
    weight_decay: float = 0
    # l1_lambda: float = 0
    l1_lambda: float = 7e-6
    grad_norm: float = 0.5
    patience: int = 5
    lr_decay: float = 0.5
    save_int: int = 10
    log_dir: str = './runs'
    eval_int: int = 10
    test_int: int = 30
    probe_int: int = 5

@dataclass
class FinetuneConfig:
    learning_rate: float = 1e-5
    num_epochs: int = 25

@dataclass
class ModelConfig:
    # conv1_drop: float = 0
    # conv2_drop: float = 0
    # fc1_drop: float = 0
    # fc2_drop: float = 0
    conv1_drop: float = 0.0
    conv2_drop: float = 0.05
    fc1_drop: float = 0.2
    fc2_drop: float = 0.1
    pooling: str = "avg"
    gem_p: float = 3.0
    conv3_dilation: int = 1

@dataclass
class HardMining:
    hn_cutoff: float = 0.8
    hp_cutoff: float = 0.45
    hm_frac: float = 0.1

@dataclass
class Config:
    data: DataConfig = field(default_factory=DataConfig)
    dl: DataloaderConfig = field(default_factory=DataloaderConfig)
    tra: TrainingConfig = field(default_factory=TrainingConfig)
    model: ModelConfig = field(default_factory=ModelConfig)
    finetune: FinetuneConfig = field(default_factory=FinetuneConfig)
    hm: HardMining = field(default_factory=HardMining)
    
    # Derived properties
    device: str = "cuda" if torch.cuda.is_available() else "cpu"
    model_dir: str = "models"
    exp_name: Optional[str] = None
