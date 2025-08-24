from dataclasses import dataclass
from typing import Optional
import torch

@dataclass
class DataConfig:
    zarr_path: str = "/media/jeff/SSD_2/ves_zarrs2/"
    # scroll1_id: int = 20230827161847
    scroll1_id: int = 20230702185753
    scroll4_id: int = 20231210132040
    tile_size: int = 32
    depth: int = 8
    d_start: int = 28
    d_end: int = 48

@dataclass
class DataloaderConfig:
    batch_size: int = 64
    num_workers: int = 8
    data_aug: bool = True

@dataclass
class TrainingConfig:
    n_epochs: int = 50
    lr: float = 1e-4
    weight_decay: float = 0
    # l1_lambda: float = 0
    l1_lambda: float = 7e-6
    grad_norm: float = 0.5
    patience: int = 5
    lr_decay: float = 0.5
    save_int: int = 10
    log_dir: str = './runs'
    eval_int: int = 20
    test_int: int = 50

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

@dataclass
class HardMining:
    hn_cutoff: float = 0.8
    hp_cutoff: float = 0.45
    hm_frac: float = 0.1

@dataclass
class Config:
    data: DataConfig = DataConfig()
    dl: DataloaderConfig = DataloaderConfig()
    tra: TrainingConfig = TrainingConfig()
    model: ModelConfig = ModelConfig()
    finetune: FinetuneConfig = FinetuneConfig()
    hm: HardMining = HardMining()
    
    # Derived properties
    device: str = "cuda" if torch.cuda.is_available() else "cpu"
    model_dir: str = "models"
    exp_name: Optional[str] = None
