import torch
import torch.nn as nn
from .config import Config

class InkDetector(nn.Module):
    def __init__(self):
        super(InkDetector, self).__init__()
        
        self.features = nn.Sequential(
            nn.Conv3d(1, 64, kernel_size=3, padding=1),
            nn.BatchNorm3d(64),
            nn.ReLU(inplace=True),
            
            nn.Conv3d(64, 196, kernel_size=3, padding=1),
            nn.BatchNorm3d(196),
            nn.ReLU(inplace=True),
            nn.MaxPool3d(kernel_size=2),
            nn.AdaptiveAvgPool3d(1)
        )

        self.classifier = nn.Sequential(
            nn.Flatten(),

            nn.Linear(196, 128),
            nn.ReLU(inplace=True),
            nn.Dropout(0.2),

            nn.Linear(128, 64),
            nn.ReLU(inplace=True),
            nn.Dropout(0.2),

            nn.Linear(64, 64),
            nn.ReLU(inplace=True),
            nn.Dropout(0.2),

            nn.Linear(64, 1)
        )

        self.activations = {}

    def forward(self, x):
        for idx, layer in enumerate(self.features):
            x = layer(x)
            self.activations[f'features_{idx}'] = x.detach()
        
        x = self.classifier[0](x)
        for idx, layer in enumerate(self.classifier[1:], start=1):
            x = layer(x)
            self.activations[f'classifier_{idx}'] = x.detach()
        
        return x
        

def create_model(config: Config):
    """Create and initialize the model"""
    model = InkDetector().to(config.device)
    
    # Initialize weights properly
    def init_weights(m):
        if isinstance(m, (nn.Conv3d, nn.Linear)):
            nn.init.xavier_uniform_(m.weight)
            if m.bias is not None:
                nn.init.zeros_(m.bias)
    
    model.apply(init_weights)
    
    # Count parameters
    model_parameters = filter(lambda p: p.requires_grad, model.parameters())
    params = sum([torch.numel(p) for p in model_parameters])
    print(f"Model parameters: {params:,}")
    
    return model, params