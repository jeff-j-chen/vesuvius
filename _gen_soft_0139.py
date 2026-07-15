import numpy as np, cv2
from PIL import Image
Image.MAX_IMAGE_PIXELS=None
sid=20260115000000
src=f"eroded_inklabels/{sid}.png"
img=np.array(Image.open(src).convert("L"))
print("eroded", img.shape, "ink_frac", round(float((img>127).mean()),4))
# a1_blur recipe: dilate 1px (3x3 ellipse) then gaussian blur sigma=15
k=cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(3,3))
dil=cv2.dilate(img,k,iterations=1)
blur=cv2.GaussianBlur(dil,(0,0),sigmaX=15.0,sigmaY=15.0)
import os; os.makedirs("soft_inklabels",exist_ok=True)
Image.fromarray(blur).save(f"soft_inklabels/{sid}.png")
print("wrote soft_inklabels", blur.shape, "mean", round(float(blur.mean()),3), "max", int(blur.max()))
