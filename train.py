"""
train.py  –  RTX 5080-optimised training loop

Optimisations:
  - TF32: torch.set_float32_matmul_precision('high')
  - AMP : torch.autocast(device_type='cuda', dtype=torch.bfloat16)
  - Graph compilation: torch.compile(model, mode='max-autotune')
  - AdamW + CosineAnnealingLR
  - Smooth-L1 loss (Huber)
"""
import time
import math
import torch
import torch.nn as nn

from dataset import build_loaders
from model import CalibNet

# ── RTX 5080 tuning ─────────────────────────────────────────────────────────
torch.set_float32_matmul_precision("high")   # enable TF32
# new API (torch >= 2.9 avoids deprecation warning)
try:
    torch.backends.cuda.matmul.fp32_precision = "tf32"
    torch.backends.cudnn.conv.fp32_precision  = "tf32"
except AttributeError:
    pass

# ── Config ───────────────────────────────────────────────────────────────────
DEVICE      = torch.device("cuda" if torch.cuda.is_available() else "cpu")
BATCH_SIZE  = 64
EPOCHS      = 80
LR          = 1e-3
WARMUP      = 5    # warmup epochs
GRAD_CLIP   = 1.0
CKPT_PATH   = "best_model.pt"
LOG_EVERY   = 20   # batches


def epoch_loop(model, loader, criterion, optimizer, scaler, train: bool):
    model.train(train)
    total_loss = 0.0
    n_batches  = 0

    for imgs, true_uv, dist_uv in loader:
        imgs     = imgs.to(DEVICE, non_blocking=True)
        true_uv  = true_uv.to(DEVICE, non_blocking=True)
        dist_uv  = dist_uv.to(DEVICE, non_blocking=True)

        # ground-truth offset = true_uv - distorted_uv
        gt_offset = true_uv - dist_uv   # (B, N, 2)

        with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
            pred_offset = model(imgs, dist_uv)     # (B, N, 2)
            loss = criterion(pred_offset, gt_offset)

        if train:
            optimizer.zero_grad(set_to_none=True)
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            nn.utils.clip_grad_norm_(model.parameters(), GRAD_CLIP)
            scaler.step(optimizer)
            scaler.update()

        total_loss += loss.item()
        n_batches  += 1

    return total_loss / max(n_batches, 1)


def main():
    print(f"Device : {DEVICE}")
    print(f"PyTorch: {torch.__version__}")

    train_loader, val_loader = build_loaders(
        train_size=8000, val_size=800,
        batch_size=BATCH_SIZE, num_workers=4,
    )

    model = CalibNet().to(DEVICE)

    # ── compile (warm-up will happen on first forward pass) ──────────────────
    print("Compiling model with torch.compile(mode='max-autotune') …")
    model = torch.compile(model, mode="max-autotune")

    optimizer = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=1e-3)

    # linear warmup then cosine decay
    def lr_lambda(epoch):
        if epoch < WARMUP:
            return float(epoch + 1) / WARMUP
        progress = (epoch - WARMUP) / max(1, EPOCHS - WARMUP)
        return 0.01 + 0.99 * 0.5 * (1 + math.cos(math.pi * progress))

    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)
    criterion = nn.SmoothL1Loss(beta=1.0)
    scaler    = torch.GradScaler(device="cuda")

    best_val = float("inf")
    t0 = time.time()

    for epoch in range(1, EPOCHS + 1):
        t_ep = time.time()

        train_loss = epoch_loop(model, train_loader, criterion, optimizer, scaler, train=True)
        with torch.no_grad():
            val_loss = epoch_loop(model, val_loader, criterion, optimizer, scaler, train=False)

        scheduler.step()

        elapsed = time.time() - t_ep
        total_elapsed = time.time() - t0
        lr_now = scheduler.get_last_lr()[0]

        print(
            f"[{epoch:3d}/{EPOCHS}]  "
            f"train={train_loss:.4f}  val={val_loss:.4f}  "
            f"lr={lr_now:.2e}  ep={elapsed:.1f}s  tot={total_elapsed/60:.1f}min"
        )

        if val_loss < best_val:
            best_val = val_loss
            # save raw (uncompiled) module weights
            raw_model = model._orig_mod if hasattr(model, "_orig_mod") else model
            torch.save(raw_model.state_dict(), CKPT_PATH)
            print(f"  ↳ saved checkpoint  (val={best_val:.4f})")

    print(f"\nTraining complete. Best val loss: {best_val:.4f}")
    print(f"Total time: {(time.time()-t0)/60:.1f} min")


if __name__ == "__main__":
    main()
