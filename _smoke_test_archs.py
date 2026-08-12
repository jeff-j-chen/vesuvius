import torch
import sys
from utils.config import Config
from utils.model import create_model

# Test a few representative architectures
test_archs = [
    'v16_arch_ctx_fovea',
    'v16_dual_stream_early',
    'v16_hybrid_depth_per_window',
    'v16_multiscale_pyramid',
    'v16_nonlocal_depth',
    'v16_fpn',
]

print("Testing architecture instantiation...")
for arch in test_archs:
    try:
        c = Config()
        c.model.arch = arch
        c.device = 'cpu'
        c.data.tile_size = 16
        c.data.depth = 24
        c.data.context_size = 48
        c.data.context_downsample = 2
        c.tra.dann = False
        c.tra.supcon = False
        c.model.attn_mil = False
        
        model, params = create_model(c)
        
        # Test forward pass
        dummy_input = torch.randn(2, 1, 24, 48, 48)  # batch=2, channels=1, depth=24, H=48, W=48
        output = model(dummy_input)
        
        print(f"OK  {arch:35s} - params: {params:,} - output: {output.shape}")
    except Exception as e:
        print(f"ERR {arch:35s} - ERROR: {str(e)[:80]}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

print("\nOK  All tested architectures work correctly!")
