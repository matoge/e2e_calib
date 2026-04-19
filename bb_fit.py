"""BB fit experiment: render car image + BB overlay.

Usage:
    python bb_fit.py pick <idx_start>           # find car crop + save /tmp/bb_car.png raw
    python bb_fit.py draw <x> <y> <w> <h>       # render /tmp/bb_car.png + BB -> /tmp/bb_overlay.png
"""
import sys, torch
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from pathlib import Path
from dataset_pandaset import PandaSetCalibDataset

RAW  = Path("/tmp/bb_car.png")
OUT  = Path("/tmp/bb_overlay.png")
META = Path("/tmp/bb_meta.txt")

def pick(start=0):
    ds = PandaSetCalibDataset('/tmp/pandaset_cache.pt', split='val')
    for idx in range(start, len(ds)):
        img, true_uvd, dist_uvd = ds[idx]
        is_obj = dist_uvd[:, 3].numpy() > 0.5
        if not is_obj.any(): continue
        t_obj = true_uvd[is_obj, :2].numpy()
        x0,y0 = float(t_obj[:,0].min()), float(t_obj[:,1].min())
        x1,y1 = float(t_obj[:,0].max()), float(t_obj[:,1].max())
        w,h = x1-x0, y1-y0
        if is_obj.sum() < 30: continue
        if min(w,h) < 15: continue
        fig, ax = plt.subplots(figsize=(6,6), dpi=96)
        ax.imshow(img.permute(1,2,0).numpy())
        ax.axis('off'); plt.tight_layout(pad=0)
        plt.savefig(RAW, dpi=96, bbox_inches='tight'); plt.close(fig)
        META.write_text(f"idx={idx} gt_bb=({x0:.1f},{y0:.1f},{w:.1f},{h:.1f}) img_size=128\n")
        print(f"picked idx={idx}  gt_bb=(x={x0:.1f},y={y0:.1f},w={w:.1f},h={h:.1f})")
        print(f"saved {RAW}")
        return
    print("no car found in range")

def draw(x, y, w, h):
    import numpy as np
    from PIL import Image
    img = Image.open(RAW).convert('RGB')
    import matplotlib.image as mpimg
    # Prefer loading the original 128x128 image to avoid matplotlib padding
    ds = PandaSetCalibDataset('/tmp/pandaset_cache.pt', split='val')
    meta = META.read_text()
    idx = int(meta.split()[0].split('=')[1])
    raw_img, true_uvd, dist_uvd = ds[idx]
    arr = raw_img.permute(1,2,0).numpy()
    fig, ax = plt.subplots(figsize=(6,6), dpi=96)
    ax.imshow(arr, extent=[0,128,128,0])
    ax.add_patch(plt.Rectangle((x, y), w, h, fill=False, ec='yellow', lw=2.0))
    # grid for sub-pixel reading
    for g in range(0, 129, 16):
        ax.axvline(g, color='white', lw=0.2, alpha=0.3)
        ax.axhline(g, color='white', lw=0.2, alpha=0.3)
    ax.set_xlim(0,128); ax.set_ylim(128,0)
    ax.set_title(f"BB=(x={x:.1f}, y={y:.1f}, w={w:.1f}, h={h:.1f})", fontsize=10)
    ax.set_xlabel("x (px)"); ax.set_ylabel("y (px)")
    plt.tight_layout(); plt.savefig(OUT, dpi=128, bbox_inches='tight'); plt.close(fig)
    # Compute overlap with GT for numeric feedback
    gtstr = meta.split('gt_bb=')[1].split()[0]  # "(x0,y0,w,h)"
    gt = [float(v) for v in gtstr.strip('()').split(',')]
    gx0,gy0,gw,gh = gt
    gx1,gy1 = gx0+gw, gy0+gh
    x1,y1 = x+w, y+h
    ix0,iy0 = max(x,gx0), max(y,gy0)
    ix1,iy1 = min(x1,gx1), min(y1,gy1)
    inter = max(0, ix1-ix0) * max(0, iy1-iy0)
    area_p = w*h; area_g = gw*gh
    iou = inter / (area_p + area_g - inter) if (area_p + area_g - inter) > 0 else 0.0
    dx0, dy0 = x-gx0, y-gy0
    dx1, dy1 = x1-gx1, y1-gy1
    print(f"saved {OUT}")
    print(f"GT  bb=(x={gx0:.1f}, y={gy0:.1f}, w={gw:.1f}, h={gh:.1f})")
    print(f"IoU={iou:.3f}  edge Δ: left={dx0:+.1f} top={dy0:+.1f} right={dx1:+.1f} bottom={dy1:+.1f}")

if __name__ == "__main__":
    cmd = sys.argv[1]
    if cmd == "pick":
        pick(int(sys.argv[2]) if len(sys.argv) > 2 else 0)
    elif cmd == "draw":
        x,y,w,h = map(float, sys.argv[2:6])
        draw(x,y,w,h)
