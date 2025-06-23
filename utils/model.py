import torch
import torch.nn as nn
from .config import Config

class InkDetector(nn.Module):
    def __init__(self):
        super(InkDetector, self).__init__()
        
        self.features = nn.Sequential(
            # Input: (B, 1, 8, 32, 32)
            nn.Conv3d(1, 32, kernel_size=(1, 4, 4), padding=1, bias=False), # (B, 32, 8, 32, 32)
            nn.BatchNorm3d(32),
            nn.ReLU(inplace=True),
            nn.Dropout3d(0.3),
            
            nn.Conv3d(32, 64, kernel_size=(2, 3, 3), padding=1, bias=False), # (B, 64, 7, 32, 32)
            nn.BatchNorm3d(64),
            nn.ReLU(inplace=True),
            nn.Dropout3d(0.3),
            
            nn.MaxPool3d(kernel_size=(1, 2, 2)), # (B, 64, 7, 16, 16)
            
            nn.Conv3d(64, 96, kernel_size=(2, 3, 3), padding=1, bias=False), # (B, 96, 6, 16, 16)
            nn.BatchNorm3d(96),
            nn.ReLU(inplace=True),
            nn.Dropout3d(0.4),
            
            nn.MaxPool3d(kernel_size=(1, 2, 2)), # (B, 96, 6, 8, 8)
            
            nn.Conv3d(96, 128, kernel_size=(2, 3, 3), padding=1, bias=False), # (B, 128, 5, 8, 8)
            nn.BatchNorm3d(128),
            nn.ReLU(inplace=True),
            nn.Dropout3d(0.4),
            
            nn.MaxPool3d(kernel_size=(1, 2, 2)), # (B, 128, 5, 4, 4)
            
            nn.AdaptiveAvgPool3d(1) # (B, 128, 1, 1, 1)
        )

        self.classifier = nn.Sequential(
            nn.Flatten(),
            
            nn.Linear(128, 64, bias=False),
            nn.BatchNorm1d(64),
            nn.ReLU(inplace=True),
            nn.Dropout(0.6),

            nn.Linear(64, 32, bias=False),
            nn.BatchNorm1d(32),
            nn.ReLU(inplace=True),
            nn.Dropout(0.6),

            nn.Linear(32, 1)  # Keep bias for final output layer
        )

        self.activations = {}

        # Register hooks to capture activations
        self._register_hooks()
    def _register_hooks(self):
        def hook(module, input, output):
            self.activations[module] = output.detach()

        for layer in self.features:
            if not isinstance(layer, (nn.Dropout3d, nn.BatchNorm3d)):
                layer.register_forward_hook(hook)
        for layer in self.classifier:
            if not isinstance(layer, (nn.Dropout, nn.BatchNorm1d)):
                layer.register_forward_hook(hook)

    def forward(self, x):
        x = self.features(x)
        x = self.classifier(x)
        return x
        
        

def create_model(config: Config):
    """Create and initialize the model"""
    model = InkDetector().to(config.device)
    
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