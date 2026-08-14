"""process_inklabels_w013.py -- generate inklabels + eroded_inklabels for w013 PHerc1667.

reads the already-assembled zarr for dimensions, resizes the 2.4um ink TIF to match,
thresholds + erodes, saves both label files, then generates the overlay figure.

run AFTER the zarr exists:
  python process_inklabels_w013.py
  python process_inklabels_w013.py --overlay-only   # if labels already done
"""
from __future__ import annotations
import argparse, os, sys
import cv2
import numpy as np
from PIL import Image
Image.MAX_IMAGE_PIXELS = None

ZID        = "20240304141531"
INK_THRESH = 0.55
LEFT_FRAC  = 0.25
# zarr is level-2 surface zarr (4x XY downsample of 2.4um): exact dims (28,10400,4975)
H_OUT = 10400
W_OUT = 4975

ZARR_DIR = os.getenv("VESUVIUS_ZARR_PATH",
                     "/vesuvius/ves_zarrs2" if os.name == "posix"
                     else r"C:\Users\ChenJeff\Documents\ves_zarrs2")
TMP = "_ves_tmp"

INK_TIF_PATH = os.path.join(TMP, f"{ZID}_ink_2p4.tif")
ZARR_PATH    = os.path.join(ZARR_DIR, f"{ZID}.zarr")
INK_OUT      = f"inklabels/{ZID}.png"
MASK_PATH    = f"masks/{ZID}.png"
OVERLAY_OUT  = os.path.join(TMP, f"{ZID}_overlay.png")


def process_inklabels():
    import zarr
    if not os.path.isdir(ZARR_PATH):
        print(f"ERROR: zarr not found at {ZARR_PATH}")
        print("run assemble_w013_1667.py first")
        sys.exit(1)

    z = zarr.open(ZARR_PATH, mode="r")
    D, H, W = z.shape
    print(f"zarr shape: ({D}, {H}, {W})")

    if not os.path.exists(INK_TIF_PATH):
        print(f"ERROR: ink TIF not found at {INK_TIF_PATH}")
        print("download it first:")
        print("  curl -o", INK_TIF_PATH, "<ink_url>")
        sys.exit(1)

    sz = os.path.getsize(INK_TIF_PATH)
    print(f"loading ink TIF ({sz/1e9:.2f} GB) ...")
    ink = np.array(Image.open(INK_TIF_PATH))
    print(f"ink TIF shape: {ink.shape}  dtype: {ink.dtype}")

    # level-2 zarr is 4x XY downsample: ink (41600,79600) -> zarr (10400,19900) -> crop W
    ink_h, ink_w = ink.shape[:2]
    W_full_resize = int(round(ink_w * H / ink_h))
    print(f"  resizing ink ({ink_h},{ink_w}) -> ({H},{W_full_resize}) -> crop to ({H},{W})"
          f"  (expected ~19900 x {W})")
    print(f"resizing ink ({ink.shape[0]},{ink.shape[1]}) -> ({H},{W_full}) ...")
    ink_resized = cv2.resize(ink.astype(np.float32), (W_full, H),
                             interpolation=cv2.INTER_AREA)
    ink_crop = ink_resized[:, :W]
    print(f"cropped to ({H},{W})  range [{ink_crop.min():.1f},{ink_crop.max():.1f}]")

    # load mask to zero-out non-surface regions
    mask = None
    if os.path.exists(MASK_PATH):
        mask = np.array(Image.open(MASK_PATH).convert("L")) > 0
        print(f"mask loaded: valid_frac={mask.mean():.3f}")

    # save continuous inklabels only -- eroded_inklabels/ is manually curated, do not touch
    os.makedirs("inklabels", exist_ok=True)
    ink_u8 = np.clip(ink_crop, 0, 255).astype(np.uint8)
    if mask is not None:
        ink_u8 = (ink_u8 * mask).astype(np.uint8)
    Image.fromarray(ink_u8).save(INK_OUT)
    print(f"wrote {INK_OUT}  ink_frac={(ink_u8>128).mean():.4f}")


def gen_overlay():
    import zarr
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    if not os.path.isdir(ZARR_PATH):
        print(f"zarr missing: {ZARR_PATH}")
        return

    z = zarr.open(ZARR_PATH, mode="r")
    D, H, W = z.shape
    mid = D // 2
    print(f"loading midslice (layer {mid}) ...")
    layer = z[mid].astype(np.float32)

    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    fig.suptitle(f"w013 PHerc1667  ZID={ZID}  9.362um isotropic  layer {mid}/{D}")

    axes[0].imshow(layer, cmap="gray", vmin=0, vmax=255)
    axes[0].set_title(f"zarr midslice (layer {mid})")
    axes[0].axis("off")

    if os.path.exists(INK_OUT):
        ink = np.array(Image.open(INK_OUT).convert("L")).astype(np.float32)
        rgb = np.stack([layer / 255.0] * 3, axis=-1)
        rgb[:, :, 0] = np.clip(rgb[:, :, 0] + ink / 255.0 * 0.5, 0, 1)
        axes[1].imshow(rgb)
        axes[1].set_title(f"inklabels overlay  ink_frac={(ink>0).mean():.3f}")
        axes[1].axis("off")
    else:
        axes[1].text(0.5, 0.5, "inklabels missing", ha="center", va="center")
        axes[1].axis("off")

    plt.tight_layout()
    plt.savefig(OVERLAY_OUT, dpi=100, bbox_inches="tight")
    plt.close()
    print(f"overlay saved -> {OVERLAY_OUT}")
    try:
        os.startfile(OVERLAY_OUT)
    except Exception:
        pass


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--overlay-only", action="store_true")
    args = ap.parse_args()

    if args.overlay_only:
        gen_overlay()
        return

    if not os.path.exists(INK_OUT):
        process_inklabels()
    else:
        print(f"labels already exist: {INK_OUT}")

    gen_overlay()


if __name__ == "__main__":
    main()
