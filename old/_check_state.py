import os, zarr, numpy as np, json
from PIL import Image
Image.MAX_IMAGE_PIXELS = None

ZARR_DIR = r"C:\Users\ChenJeff\Documents\ves_zarrs2"

# new fragment ids (zarr id = segment timestamp)
NEW = {
    "w058": "20260210000000", "w052": "20260227000000", "w049": "20260318000000",
    "w046": "20260325000000", "w041": "20260108000000", "w040": "20250831000000",
    "w039": "20260302000000", "w038": "20260306000000", "w037": "20260310000000",
    "w034": "20260303000000",
}
# re-rendered test segs to sanity-check too
TEST = {"seg3_20260719202304": "20260719202304", "pherc1203_20260720090842": "20260720090842"}

try:
    norm = json.load(open("norm_cache.json"))
except Exception:
    norm = {}

def zshape(zid):
    p = os.path.join(ZARR_DIR, f"{zid}.zarr")
    if not os.path.isdir(p):
        return None
    try:
        return zarr.open(p, mode="r").shape
    except Exception as e:
        return f"OPEN-ERR({e})"

def pngok(path):
    if not os.path.exists(path):
        return "-"
    try:
        a = np.array(Image.open(path).convert("L"))
        return f"{a.shape} v={(a>0).mean():.3f}"
    except Exception as e:
        return f"ERR({e})"

print(f"{'name':<24}{'zarr':<22}{'mask':<20}{'inklabels':<22}{'eroded':<22}{'norm':<5}")
for name, zid in {**NEW, **TEST}.items():
    zs = zshape(zid)
    print(f"{name:<24}{str(zs):<22}"
          f"{pngok(f'masks/{zid}.png'):<20}"
          f"{pngok(f'inklabels/{zid}.png'):<22}"
          f"{pngok(f'eroded_inklabels/{zid}.png'):<22}"
          f"{'Y' if zid in norm else '-':<5}")
