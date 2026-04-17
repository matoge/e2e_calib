"""Train CalibNet at 64x64 resolution."""
import math, time, torch, torch.nn as nn
from dataset import make_image_and_points
from model import CalibNet
from torch.utils.data import Dataset, DataLoader

torch.set_float32_matmul_precision("high")

DEVICE     = torch.device("cuda")
IMG_SIZE   = 64
BATCH_SIZE = 128
EPOCHS     = 60
LR         = 1e-3
CKPT       = "best_model_64.pt"


class Dataset64(Dataset):
    def __init__(self, length, base_seed=0):
        self.length = length
        self.base_seed = base_seed

    def __len__(self): return self.length

    def __getitem__(self, idx):
        img, true_uv, dist_uv = make_image_and_points(
            img_size=IMG_SIZE, seed=self.base_seed + idx)
        return img, true_uv, dist_uv


def epoch_loop(model, loader, optimizer, scaler, train):
    model.train(train)
    total_mse, n = 0.0, 0
    for imgs, true_uv, dist_uv in loader:
        imgs    = imgs.to(DEVICE)
        true_uv = true_uv.to(DEVICE)
        dist_uv = dist_uv.to(DEVICE)
        gt      = true_uv - dist_uv

        with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
            pred   = model(imgs, dist_uv)
            loss   = (pred - gt).norm(dim=-1).mean()

        if train:
            optimizer.zero_grad(set_to_none=True)
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            scaler.step(optimizer); scaler.update()

        total_mse += loss.item(); n += 1
    return total_mse / max(n, 1)


def main():
    print(f"64x64 training", flush=True)
    kw = dict(num_workers=4, pin_memory=True, persistent_workers=True)
    train_loader = DataLoader(Dataset64(8000, 0),       batch_size=BATCH_SIZE, shuffle=True,  **kw)
    val_loader   = DataLoader(Dataset64(800,  200_000), batch_size=BATCH_SIZE, shuffle=False, **kw)

    model     = CalibNet(img_size=IMG_SIZE).to(DEVICE)
    optimizer = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=1e-3)

    def lr_lambda(e):
        if e < 5: return (e + 1) / 5
        return 0.01 + 0.99 * 0.5 * (1 + math.cos(math.pi * (e - 5) / max(1, EPOCHS - 5)))

    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)
    scaler    = torch.GradScaler(device="cuda")
    best_val  = float("inf")
    t0        = time.time()

    for epoch in range(1, EPOCHS + 1):
        tr = epoch_loop(model, train_loader, optimizer, scaler, True)
        with torch.no_grad():
            va = epoch_loop(model, val_loader, optimizer, scaler, False)
        scheduler.step()
        print(f"[{epoch:3d}/{EPOCHS}]  train={tr:.3f}  val={va:.3f}  "
              f"lr={scheduler.get_last_lr()[0]:.2e}  tot={(time.time()-t0)/60:.1f}min", flush=True)
        if va < best_val:
            best_val = va
            torch.save(model.state_dict(), CKPT)
            print(f"  ↳ saved (val={best_val:.4f})", flush=True)

    print(f"\nBest val MSE: {best_val:.4f}px  |  time: {(time.time()-t0)/60:.1f}min", flush=True)


if __name__ == "__main__":
    main()
