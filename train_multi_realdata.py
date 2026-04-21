"""
Train CalibNet (Cross→Self) on REAL Woven dataset (U, V, D format).
Uses the same model as train_multi.py but with real data instead of synthetic.
"""
import math, time, torch, torch.nn as nn
from torch.utils.data import DataLoader
from dataset_woven import WovenCalibDataset
from model import CalibNet

torch.set_float32_matmul_precision("high")
try:
    torch.backends.cuda.matmul.fp32_precision = "tf32"
    torch.backends.cudnn.conv.fp32_precision  = "tf32"
except AttributeError:
    pass

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
BATCH_SIZE = 64
EPOCHS = 80
LR = 1e-3
CKPT = "best_model_multi_realdata.pt"


def epoch_loop(model, loader, criterion, optimizer, scaler, train):
    model.train(train)
    total, n = 0.0, 0
    for batch in loader:
        # Real dataset returns: (img, true_uv, dist_uv, weights, depths)
        imgs, true_uv, dist_uv, weights, depths = batch
        imgs = imgs.to(DEVICE, non_blocking=True)
        true_uv = true_uv.to(DEVICE, non_blocking=True)
        dist_uv = dist_uv.to(DEVICE, non_blocking=True)

        # Clamp to max 256 points per sample
        B, N = true_uv.shape[0], true_uv.shape[1]
        if N > 256:
            true_uv = true_uv[:, :256]
            dist_uv = dist_uv[:, :256]

        # Keep RGB (CalibNet now expects 3 channels)

        gt = true_uv - dist_uv
        with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
            loss = criterion(model(imgs, dist_uv), gt)
        if train:
            optimizer.zero_grad(set_to_none=True)
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            scaler.step(optimizer)
            scaler.update()
        total += loss.item()
        n += 1
    return total / max(n, 1)


def main():
    print("=" * 70)
    print("Training CalibNet (Cross→Self) on REAL Woven dataset")
    print("=" * 70)
    print(f"Device: {DEVICE}")
    print(f"Batch size: {BATCH_SIZE}")
    print(f"Epochs: {EPOCHS}")
    print(f"Learning rate: {LR}")
    print(f"Checkpoint: {CKPT}")
    print()

    # Create real datasets (non-grid mode for U,V,D compatibility)
    sequences = [
        ('llinking_27', 'tf_long2', 'ip654_1337941440921107425_16943630305775105398_1749030654176-1749030664176')
    ]

    print("Initializing training dataset...")
    train_ds = WovenCalibDataset(
        '/home/hfunaya/git/loom/backend/assets/woven_sequence',
        sequences,
        target_labels=['traffic_body', 'sign', 'delineator'],
        max_pitch_deg=0.4,
        max_yaw_deg=0.4,
        max_roll_deg=0.1,
        random_seed=42,
        grid_sampling=False,  # Use non-grid mode for U,V,D format
        max_points=128,
    )

    print("Initializing validation dataset...")
    val_ds = WovenCalibDataset(
        '/home/hfunaya/git/loom/backend/assets/woven_sequence',
        sequences,
        target_labels=['traffic_body', 'sign', 'delineator'],
        max_pitch_deg=0.4,
        max_yaw_deg=0.4,
        max_roll_deg=0.1,
        random_seed=123,  # Different seed for validation
        grid_sampling=False,
        max_points=128,
    )

    print(f"Training samples: {len(train_ds)}")
    print(f"Validation samples: {len(val_ds)}")
    print()

    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True, num_workers=4, pin_memory=True)
    val_loader = DataLoader(val_ds, batch_size=BATCH_SIZE, shuffle=False, num_workers=4, pin_memory=True)

    model = CalibNet().to(DEVICE)
    optimizer = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=1e-3)

    def lr_lambda(e):
        if e < 5:
            return (e + 1) / 5
        return 0.01 + 0.99 * 0.5 * (1 + math.cos(math.pi * (e - 5) / max(1, EPOCHS - 5)))

    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)
    criterion = nn.SmoothL1Loss(beta=1.0)
    scaler = torch.GradScaler(device="cuda")
    best_val = float("inf")
    t0 = time.time()

    print("Starting training...")
    print()

    for epoch in range(1, EPOCHS + 1):
        tr = epoch_loop(model, train_loader, criterion, optimizer, scaler, True)
        with torch.no_grad():
            va = epoch_loop(model, val_loader, criterion, optimizer, scaler, False)
        scheduler.step()

        improved = ""
        if va < best_val:
            best_val = va
            torch.save(model.state_dict(), CKPT)
            improved = " ← NEW BEST!"

        elapsed = time.time() - t0
        print(f"[{epoch:2d}/{EPOCHS}] train={tr:7.4f}  val={va:7.4f}  best={best_val:7.4f}  "
              f"time={elapsed/60:.1f}m{improved}")

    print()
    print("=" * 70)
    print(f"Training complete! Best val loss: {best_val:.4f}")
    print(f"Model saved to: {CKPT}")
    print(f"Total time: {(time.time() - t0) / 60:.1f} minutes")
    print("=" * 70)


if __name__ == "__main__":
    main()
