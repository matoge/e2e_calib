"""Pre-generate sim3d dataset and save to disk for fast training."""
import time, torch
from sim3d import make_sample

TRAIN_SIZE  = 8000
VAL_SIZE    = 800
MAX_OFFSET  = 8.0
BG_RATIO    = 1
TRAIN_FILE  = "sim3d_train.pt"
VAL_FILE    = "sim3d_val.pt"


def generate(n, seed_offset, label):
    imgs, true_uvds, dist_uvds = [], [], []
    t0 = time.time()
    for i in range(n):
        img, true_uvd, dist_uvd = make_sample(
            seed=seed_offset + i, max_offset=MAX_OFFSET, bg_ratio=BG_RATIO)
        imgs.append(img.mean(dim=0, keepdim=True))   # (1,H,W) grayscale
        true_uvds.append(true_uvd)
        dist_uvds.append(dist_uvd)
        if (i + 1) % 500 == 0:
            elapsed = time.time() - t0
            rem = elapsed / (i + 1) * (n - i - 1)
            print(f"  {label}: {i+1}/{n}  elapsed={elapsed:.0f}s  remaining≈{rem:.0f}s", flush=True)
    return (torch.stack(imgs),
            torch.stack(true_uvds),
            torch.stack(dist_uvds))


print(f"Generating {TRAIN_SIZE} train samples…")
train_data = generate(TRAIN_SIZE, seed_offset=0, label="train")
torch.save(train_data, TRAIN_FILE)
print(f"Saved → {TRAIN_FILE}")

print(f"Generating {VAL_SIZE} val samples…")
val_data = generate(VAL_SIZE, seed_offset=500_000, label="val")
torch.save(val_data, VAL_FILE)
print(f"Saved → {VAL_FILE}")
print("Done.")
