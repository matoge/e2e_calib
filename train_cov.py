"""Train CalibNetCov with 2D Gaussian NLL loss."""
import math, time, torch, torch.nn as nn
from dataset import build_loaders
from model_cov import CalibNetCov, gaussian2d_nll

torch.set_float32_matmul_precision("high")
try:
    torch.backends.cuda.matmul.fp32_precision = "tf32"
    torch.backends.cudnn.conv.fp32_precision  = "tf32"
except AttributeError:
    pass

DEVICE, BATCH_SIZE, EPOCHS, LR = torch.device("cuda"), 64, 80, 1e-3
CKPT = "best_model_cov.pt"


def epoch_loop(model, loader, optimizer, scaler, train):
    model.train(train)
    total_nll, total_mse, n = 0.0, 0.0, 0
    for imgs, true_uv, dist_uv in loader:
        imgs     = imgs.to(DEVICE)
        true_uv  = true_uv.to(DEVICE)
        dist_uv  = dist_uv.to(DEVICE)
        gt       = true_uv - dist_uv   # (B,N,2)

        with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
            params = model(imgs, dist_uv)              # (B,N,5)
            loss   = gaussian2d_nll(params, gt)

        if train:
            optimizer.zero_grad(set_to_none=True)
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            scaler.step(optimizer); scaler.update()

        with torch.no_grad():
            mse = (params[..., :2] - gt).norm(dim=-1).mean().item()

        total_nll += loss.item(); total_mse += mse; n += 1

    return total_nll/max(n,1), total_mse/max(n,1)


def main():
    print("Covariance training — 2D Gaussian NLL")
    train_loader, val_loader = build_loaders(
        train_size=8000, val_size=800, batch_size=BATCH_SIZE, num_workers=4)

    model     = torch.compile(CalibNetCov().to(DEVICE), mode="max-autotune")
    optimizer = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=1e-3)

    def lr_lambda(e):
        if e < 5: return (e+1)/5
        return 0.01 + 0.99*0.5*(1+math.cos(math.pi*(e-5)/max(1,EPOCHS-5)))

    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)
    scaler    = torch.GradScaler(device="cuda")
    best_val  = float("inf")
    t0        = time.time()

    for epoch in range(1, EPOCHS+1):
        tr_nll, tr_mse = epoch_loop(model, train_loader, optimizer, scaler, True)
        with torch.no_grad():
            va_nll, va_mse = epoch_loop(model, val_loader, optimizer, scaler, False)
        scheduler.step()

        print(f"[{epoch:3d}/{EPOCHS}]  "
              f"train nll={tr_nll:.3f} mse={tr_mse:.3f}  "
              f"val nll={va_nll:.3f} mse={va_mse:.3f}  "
              f"lr={scheduler.get_last_lr()[0]:.2e}  "
              f"tot={( time.time()-t0)/60:.1f}min")

        if va_nll < best_val:
            best_val = va_nll
            raw = model._orig_mod if hasattr(model, "_orig_mod") else model
            torch.save(raw.state_dict(), CKPT)
            print(f"  ↳ saved (val_nll={best_val:.4f})")

    print(f"\nBest val NLL: {best_val:.4f}  |  time: {(time.time()-t0)/60:.1f}min")


if __name__ == "__main__":
    main()
