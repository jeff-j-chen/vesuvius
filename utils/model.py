import torch
import torch.nn as nn
from .config import Config

class SpatialAttention(nn.Module):
    def __init__(self, kernel_size=7):
        super().__init__()
        self.conv = nn.Conv2d(2, 1, kernel_size=kernel_size, padding=kernel_size // 2)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = torch.mean(x, dim=1, keepdim=True)  # (B, 1, H, W)
        max_out, _ = torch.max(x, dim=1, keepdim=True)  # (B, 1, H, W)
        attn = torch.cat([avg_out, max_out], dim=1)  # (B, 2, H, W)
        attn = self.conv(attn)  # (B, 1, H, W)
        attn = self.sigmoid(attn)  # (B, 1, H, W)
        return x * attn  # Apply attention

class InkDetector(nn.Module):
    def __init__(self, config: Config):
        super(InkDetector, self).__init__()
        self.spatial_attention = SpatialAttention()

        self.features = nn.Sequential(
            # Depthwise conv to compress depth (D=8 → maxed out later)
            nn.Conv3d(1, 32, kernel_size=(3, 1, 1), padding=(1, 0, 0), bias=False),  # (B, 32, 8, 32, 32)
            nn.BatchNorm3d(32),
            nn.ReLU(inplace=True),
            # Depth collapsed later via max
        )

        self.conv2d_path = nn.Sequential(
            # Starting with (B, 32, 32, 32)

            # Add more conv layers with moderate channel growth
            nn.Conv2d(32, 64, kernel_size=3, padding=1, bias=False),   # (B, 64, 32, 32)
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),

            nn.Conv2d(64, 64, kernel_size=3, padding=1, bias=False),   # Added conv layer for depth & nonlinearity (B, 64, 32, 32)
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),

            nn.MaxPool2d(kernel_size=2, stride=1),  # Overlapping pooling (B, 64, 31, 31)

            nn.Conv2d(64, 96, kernel_size=3, padding=1, bias=False),   # (B, 96, 31, 31)
            nn.BatchNorm2d(96),
            nn.ReLU(inplace=True),

            nn.Conv2d(96, 96, kernel_size=3, padding=1, bias=False),   # Added another conv (B, 112, 31, 31)
            nn.BatchNorm2d(96),
            nn.ReLU(inplace=True),
            nn.Dropout(config.model.conv1_drop),

            nn.MaxPool2d(kernel_size=2, stride=1),  # (B, 112, 30, 30)

            nn.Conv2d(96, 128, kernel_size=3, padding=1, bias=False),  # (B, 128, 30, 30)
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.Dropout(config.model.conv2_drop),

            SpatialAttention(),

            nn.AdaptiveAvgPool2d((1, 1))  # (B, 128, 1, 1)
        )

        self.classifier = nn.Sequential(
            nn.Flatten(),  # (B, 128)
            nn.Linear(128, 64, bias=False),
            nn.BatchNorm1d(64),
            nn.ReLU(inplace=True),
            nn.Dropout(config.model.fc1_drop),

            nn.Linear(64, 32, bias=False),
            nn.BatchNorm1d(32),
            nn.ReLU(inplace=True),
            nn.Dropout(config.model.fc2_drop),

            nn.Linear(32, 1)  # Output: (B, 1)
        )
        self.activations = {}

        # Register hooks to capture activations
        self._register_hooks()

    def forward(self, x):
        x = self.features(x)  # (B, 32, 8, 32, 32)
        x = torch.max(x, dim=2).values  # Collapse depth → (B, 32, 32, 32)
        x = self.conv2d_path(x)  # (B, 128, 1, 1)
        x = self.classifier(x)  # (B, 1)
        return x

    def _register_hooks(self):
        def hook(module, input, output):
            self.activations[module] = output.detach()

        for layer in self.features:
            if not isinstance(layer, (nn.Dropout3d, nn.BatchNorm3d)):
                layer.register_forward_hook(hook)
        for layer in self.classifier:
            if not isinstance(layer, (nn.Dropout, nn.BatchNorm1d)):
                layer.register_forward_hook(hook)
        
        

def create_model(config: Config):
    """Create and initialize the model"""
    model = InkDetector(config).to(config.device)
    
    # Initialize weights properly
    def init_weights(m):
        if isinstance(m, (nn.Conv3d, nn.Linear)):
            # Use smaller initialization to prevent aggressive learning
            nn.init.xavier_uniform_(m.weight, gain=0.8)
            if m.bias is not None:
                nn.init.zeros_(m.bias)
        elif isinstance(m, (nn.BatchNorm3d, nn.BatchNorm1d)):
            nn.init.constant_(m.weight, 1)
            nn.init.constant_(m.bias, 0)
    
    
    model.apply(init_weights)
    
    # Count parameters
    model_parameters = filter(lambda p: p.requires_grad, model.parameters())
    params = sum([torch.numel(p) for p in model_parameters])
    print(f"Model parameters: {params:,}")
    
    return model, params