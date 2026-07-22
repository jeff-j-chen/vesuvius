import numpy as np, os
from PIL import Image
Image.MAX_IMAGE_PIXELS = None
for sid in ['20260115000001', '20260206000001', '20250223000000', '20260115000000']:
    row = [sid]
    for kind in ['masks', 'inklabels', 'eroded_inklabels']:
        p = f'{kind}/{sid}.png'
        if os.path.exists(p):
            a = np.array(Image.open(p))
            u = np.unique(a)
            row.append(f'{kind}: shape={a.shape} dtype={a.dtype} nlevels={len(u)} valid={(a>0).mean():.3f}')
        else:
            row.append(f'{kind}: MISSING')
    print('\n  '.join(row))
    print()
