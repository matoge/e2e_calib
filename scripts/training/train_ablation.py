"""Train ablation model (no self-attention) and save to best_model_no_sa.pt"""
import math, time, torch, torch.nn as nn
from dataset import build_loaders
from model_no_sa import CalibNetNoSA

torch.set_float32_matmul_precision("high")
try:
    torch.backends.cuda.matmul.fp32_precision = "tf32"
    torch.backends.cudnn.conv.fp32_precision  = "tf32"
except AttributeError:
    pass

DEVICE, BATCH_SIZE, EPOCHS, LR = torch.device("cuda"), 64, 80, 1e-3
CKPT = "best_model_no_sa.pt"

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
    train_loader, val_loader = build_loaders(train_size=8000, val_size=800, batch_size=BATCH_SIZE, num_workers=4)
    model = torch.compile(CalibNetNoSA().to(DEVICE), mode="max-autotune")
    optimizer = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=1e-3)
    def lr_lambda(e):
        if e < 5: return (e+1)/5
        return 0.01 + 0.99*0.5*(1+math.cos(math.pi*(e-5)/max(1,EPOCHS-5)))
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)
    criterion = nn.SmoothL1Loss(beta=1.0)
    scaler    = torch.GradScaler(device="cuda")
    best_val  = float("inf")
    for epoch in range(1, EPOCHS+1):
        tr = epoch_loop(model, train_loader, criterion, optimizer, scaler, True)
        with torch.no_grad():
            va = epoch_loop(model, val_loader, criterion, optimizer, scaler, False)
        scheduler.step()
        print(f"[{epoch:3d}/{EPOCHS}]  train={tr:.4f}  val={va:.4f}")
        if va < best_val:
            best_val = va
            raw = model._orig_mod if hasattr(model, "_orig_mod") else model
            torch.save(raw.state_dict(), CKPT)
    print(f"Best val: {best_val:.4f}")

if __name__ == "__main__":
    main()
