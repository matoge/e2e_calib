"""Train CalibNet on 2-object dataset at 64x64."""
import sys, pathlib; sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[2]))
import math, time, torch, torch.nn as nn
from datasets.synthetic import make_image_and_points_multi
from models.model import CalibNet
from torch.utils.data import Dataset, DataLoader

torch.set_float32_matmul_precision("high")

DEVICE     = torch.device("cuda")
IMG_SIZE   = 64
BATCH_SIZE = 128
EPOCHS     = 60
LR         = 1e-3
CKPT       = "best_model_multi_64.pt"


class Multi64Dataset(Dataset):
    def __init__(self, length, base_seed=0):
        self.length = length
        self.base_seed = base_seed

    def __len__(self): return self.length

    def __getitem__(self, idx):
        return make_image_and_points_multi(
            img_size=IMG_SIZE, seed=self.base_seed + idx)


def epoch_loop(model, loader, criterion, optimizer, scaler, train):
    model.train(train)
    total, n = 0.0, 0
    for imgs, true_uv, dist_uv in loader:
        imgs, true_uv, dist_uv = imgs.to(DEVICE), true_uv.to(DEVICE), dist_uv.to(DEVICE)
        gt = true_uv - dist_uv
        with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
            loss = criterion(model(imgs, dist_uv), gt)
        if train:
            optimizer.zero_grad(set_to_none=True)
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            scaler.step(optimizer); scaler.update()
        total += loss.item(); n += 1
    return total / max(n, 1)


def main():
    print(f"Multi-object 64x64 training", flush=True)
    kw = dict(num_workers=4, pin_memory=True, persistent_workers=True)
    train_loader = DataLoader(Multi64Dataset(8000, 0),       batch_size=BATCH_SIZE, shuffle=True,  **kw)
    val_loader   = DataLoader(Multi64Dataset(800,  300_100), batch_size=BATCH_SIZE, shuffle=False, **kw)

    model     = CalibNet(img_size=IMG_SIZE).to(DEVICE)
    optimizer = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=1e-3)
    criterion = nn.SmoothL1Loss(beta=1.0)

    def lr_lambda(e):
        if e < 5: return (e + 1) / 5
        return 0.01 + 0.99 * 0.5 * (1 + math.cos(math.pi * (e - 5) / max(1, EPOCHS - 5)))

    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)
    scaler    = torch.GradScaler(device="cuda")
    best_val  = float("inf")
    t0        = time.time()

    for epoch in range(1, EPOCHS + 1):
        tr = epoch_loop(model, train_loader, criterion, optimizer, scaler, True)
        with torch.no_grad():
            va = epoch_loop(model, val_loader, criterion, optimizer, scaler, False)
        scheduler.step()
        print(f"[{epoch:3d}/{EPOCHS}]  train={tr:.3f}  val={va:.3f}  "
              f"lr={scheduler.get_last_lr()[0]:.2e}  tot={(time.time()-t0)/60:.1f}min", flush=True)
        if va < best_val:
            best_val = va
            torch.save(model.state_dict(), CKPT)
            print(f"  ↳ saved (val={best_val:.4f})", flush=True)

    print(f"\nBest val: {best_val:.4f}px  |  time: {(time.time()-t0)/60:.1f}min", flush=True)


if __name__ == "__main__":
    main()
