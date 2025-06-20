from dataclasses import dataclass
from typing import Optional
import torch

@dataclass
class DataConfig:
    segment_id: int = 20230827161847
    tile_size: int = 32
    depth: int = 8
    normalize: bool = True

@dataclass
class DataloaderConfig:
    batch_size: int = 16
    num_workers: int = 4
    shuffle_train: bool = True
    shuffle_valid: bool = False

@dataclass
class TrainingConfig:
    num_epochs: int = 30
    learning_rate: float = 3e-4
    weight_decay: float = 1e-7
    max_grad_norm: float = 1.0
    patience: int = 5
    lr_scheduler_factor: float = 0.3
    save_every_n_epochs: int = 5
    log_dir: str = './runs'
    evaluation_interval: int = 5


@dataclass
class Config:
    data: DataConfig = DataConfig()
    dataloader: DataloaderConfig = DataloaderConfig()
    training: TrainingConfig = TrainingConfig()
    
    # Derived properties
    device: str = "cuda" if torch.cuda.is_available() else "cpu"
    model_dir: str = "models"
    experiment_name: Optional[str] = None
