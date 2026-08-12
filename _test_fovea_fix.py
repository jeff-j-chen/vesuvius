"""Quick test of fovea architecture fix."""
import torch
from utils.model import create_model
from utils.config import Config

# Test fovea with context
config = Config()
config.model.arch = "v16_arch_ctx_fovea"
config.data.tile_size = 16
config.data.depth = 24
config.data.context_size = 64
config.data.context_downsample = 2
config.tra.supcon = True
config.tra.attn_mil = True
config.model.attn_mil = True
config.device = torch.device('cpu')

print("Testing v16_arch_ctx_fovea with context_size=64...")
model, n_params = create_model(config)
print(f"Model created: {n_params:,} parameters")

# Test forward pass
dummy_input = torch.randn(2, 1, 24, 64, 64)
model.eval()

with torch.no_grad():
    output = model(dummy_input)

print(f"Input shape:  {tuple(dummy_input.shape)}")
print(f"Output shape: {tuple(output.shape)}")
print(f"Output range: [{output.min():.3f}, {output.max():.3f}]")
print("OK  Fovea architecture test passed!")
