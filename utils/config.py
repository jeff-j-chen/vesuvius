from dataclasses import dataclass
from typing import Optional
import torch

@dataclass
class DataConfig:
    scroll4_path: str = "scroll4_bottomleft.npz"
    segment_id: int = 20230827161847
    tile_size: int = 32
    depth: int = 8
    normalize: bool = True
    start_level: int = 28
    end_level: int = 48

@dataclass
class DataloaderConfig:
    train_batch_size: int = 128
    train_num_workers: int = 8
    train_shuffle: bool = True
    valid_batch_size: int = 128
    valid_num_workers: int = 8
    valid_shuffle: bool = False
    apply_transforms: bool = True

@dataclass
class TrainingConfig:
    num_epochs: int = 50
    learning_rate: float = 1e-4
    weight_decay: float = 0
    l1_lambda: float = 7e-6
    max_grad_norm: float = 1.0
    patience: int = 5
    lr_scheduler_factor: float = 0.5
    save_every_n_epochs: int = 10
    log_dir: str = './runs'
    evaluation_interval: int = 50
    test_interval: int = 50

@dataclass
class ModelConfig:
    conv1_drop: float = 0.0
    conv2_drop: float = 0.05
    fc1_drop: float = 0.2
    fc2_drop: float = 0.1


@dataclass
class Config:
    data: DataConfig = DataConfig()
    dataloader: DataloaderConfig = DataloaderConfig()
    training: TrainingConfig = TrainingConfig()
    model: ModelConfig = ModelConfig()
    
    # Derived properties
    device: str = "cuda" if torch.cuda.is_available() else "cpu"
    model_dir: str = "models"
    experiment_name: Optional[str] = None
