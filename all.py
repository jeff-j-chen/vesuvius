from operator import itemgetter

import torch
import torch.nn as nn
import torch.optim as optim

from torchvision import datasets, transforms
from torch.optim.lr_scheduler import MultiStepLR

from trainer import Trainer, hooks, configuration
from trainer.utils import setup_system, patch_configs
from trainer.metrics import AccuracyEstimator
from trainer.tensorboard_visualizer import TensorBoardVisualizer

import os
import pandas as pd
import numpy as np
from collections import Counter
from torch.utils.data import Dataset, DataLoader


class InkVolumeDataset(Dataset):
    def __init__(self, volume, labels, tile_size, depth):
        """
        volume: [D, H, W] - 3D volume of grayscale slices
        labels: [H, W] - 2D binary mask shared across depth
        tile_size: size of each 2D tile (height and width)
        depth: number of slices to stack per sample
        """
        self.volume = volume
        self.labels = labels
        self.tile_size = tile_size
        self.depth = depth
        self.D, self.H, self.W = volume.shape

        self.blocks = []
        for d in range(0, self.D - depth + 1, int(depth//2)):
            for y in range(0, self.H - tile_size + 1, tile_size):
                for x in range(0, self.W - tile_size + 1, tile_size):
                    label_tile = labels[y:y+tile_size, x:x+tile_size]
                    if label_tile.shape == (tile_size, tile_size):
                        self.blocks.append((d, y, x))

    def __len__(self):
        return len(self.blocks)

    def __getitem__(self, idx):
        d, y, x = self.blocks[idx]
        
        block = self.volume[d:d+self.depth, y:y+self.tile_size, x:x+self.tile_size]
        label_tile = self.labels[y:y+self.tile_size, x:x+self.tile_size]

        # Convert to tensor and ensure proper normalization
        # Don't divide by 255 again if already normalized
        block = torch.tensor(block, dtype=torch.float32)
        
        # Add channel dimension: [D, H, W] -> [1, D, H, W]
        block = block.unsqueeze(0)

        # Binary label: 1 if any ink present (more robust checking)
        has_ink = np.any(label_tile > 0.5)  # More robust than == 1.0
        label = torch.tensor([float(has_ink)], dtype=torch.float32)

        return block, label


def get_data(batch_size, volume, labels, tile_size, depth, num_workers):
    split_x = int(volume.shape[2] * 0.75)

    train_volume = volume[:, :, :split_x]
    train_labels = labels[:, :split_x]
    valid_volume = volume[:, :, split_x:]
    valid_labels = labels[:, split_x:]

    # Create datasets
    train_dataset = InkVolumeDataset(train_volume, train_labels, tile_size=tile_size, depth=depth)
    valid_dataset = InkVolumeDataset(valid_volume, valid_labels, tile_size=tile_size, depth=depth)

    # Check label distribution
    all_labels = [int(label.item()) for _, label in train_dataset]
    label_counts = Counter(all_labels)
    print(f"Label distribution: {label_counts}")

    pos_weight = torch.tensor([label_counts[0] / label_counts[1]])

    # Create dataloaders
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    valid_loader = DataLoader(valid_dataset, batch_size=batch_size)

    return train_loader, valid_loader


class InkDetector(nn.Module):
    def __init__(self):
        super(InkDetector, self).__init__()
        
        self.features = nn.Sequential(
            nn.Conv3d(1, 32, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            
            nn.Conv3d(32, 64, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool3d(kernel_size=3),

            nn.Conv3d(64, 128, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool3d(1)
        )

        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(128, 128),
            nn.Linear(128, 64),
            nn.Linear(64, 32),
            nn.ReLU(inplace=True),
            nn.Linear(32, 1)
        )

    def forward(self, x):
        x = self.features(x)
        x = self.classifier(x)
        return x


class Experiment:
    def __init__(self, system_config, dataset_config, dataloader_config, optimizer_config):
        self.loader_train, self.loader_test = get_data(
            batch_size=dataloader_config.batch_size,
            num_workers=dataloader_config.num_workers,
            data_root=dataset_config.root_dir
        )

        setup_system(system_config)

        self.model = InkDetector()
        self.loss_fn = nn.BCEWithLogitsLoss(pos_weight=pos_weight.to(device))
        self.metric_fn = AccuracyEstimator(topk=(1, ))
        self.optimizer = optim.SGD(
            self.model.parameters(),
            lr=optimizer_config.learning_rate,
            weight_decay=optimizer_config.weight_decay,
            momentum=optimizer_config.momentum
        )
        self.lr_scheduler = MultiStepLR(
            self.optimizer, milestones=optimizer_config.lr_step_milestones, gamma=optimizer_config.lr_gamma
        )
        self.visualizer = TensorBoardVisualizer()

    def run(self, trainer_config: configuration.TrainerConfig) -> dict:

        device = torch.device(trainer_config.device)
        self.model = self.model.to(device)
        self.loss_fn = self.loss_fn.to(device)

        model_trainer = Trainer(
            model=self.model,
            loader_train=self.loader_train,
            loader_test=self.loader_test,
            loss_fn=self.loss_fn,
            metric_fn=self.metric_fn,
            optimizer=self.optimizer,
            lr_scheduler=self.lr_scheduler,
            device=device,
            data_getter=itemgetter(0),
            target_getter=itemgetter(1),
            stage_progress=trainer_config.progress_bar,
            get_key_metric=itemgetter("top1"),
            visualizer=self.visualizer,
            model_saving_frequency=trainer_config.model_saving_frequency,
            save_dir=trainer_config.model_dir
        )

        # model_trainer.register_hook("end_epoch", hooks.end_epoch_hook_classification)
        self.metrics = model_trainer.fit(trainer_config.epoch_num)
        return self.metrics


dataloader_config, trainer_config = patch_configs()
dataset_config = configuration.DatasetConfig(root_dir="data")
experiment = Experiment(dataset_config=dataset_config, dataloader_config=dataloader_config)
results = experiment.run(trainer_config)


# import cv2
# import numpy as np
# model = FaceDetector()
# checkpoint = torch.load("model_best")
# state_dict = model.state_dict()
# for k1, k2 in zip(state_dict.keys(), checkpoint.keys()):
#     state_dict[k1] = checkpoint[k2]
# model.load_state_dict(state_dict)
# model.eval()

# universal_transforms = transforms.Compose([
#     transforms.ToTensor(),
#     transforms.Normalize((0.4450, ), (0.3000, ))
# ])

# image = cv2.imread("./data/0051.jpg")
# tensor = universal_transforms(image).unsqueeze(0)
# output = model(tensor)
# probs = torch.softmax(output, dim=1)
# print(probs)