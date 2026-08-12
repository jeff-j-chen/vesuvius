from utils.model import create_model
from utils.config import Config
import torch

c = Config()
c.model.arch = 'v16_arch_ctx_fovea'
c.device = 'cpu'
c.data.tile_size = 16
c.data.depth = 24
c.data.context_size = 48
c.data.context_downsample = 2
c.tra.dann = False
c.tra.supcon = False
c.model.attn_mil = False

model, params = create_model(c)
print('Model created:', type(model))
print('Params:', params)
print('Has forward:', hasattr(model, 'forward'))
print('forward attr type:', type(model.forward))

dummy = torch.randn(2, 1, 24, 48, 48)  # batch=2, channels=1, depth=24, H=48, W=48
try:
    out = model.forward(dummy)
    print(f'Success with .forward()! Output shape: {out.shape}')
except Exception as e:
    print(f'Error calling .forward(): {e}')
    import traceback
    traceback.print_exc()

try:
    out = model(dummy)
    print(f'Success with ()! Output shape: {out.shape}')
except Exception as e:
    print(f'Error calling (): {e}')
    import traceback
    traceback.print_exc()
