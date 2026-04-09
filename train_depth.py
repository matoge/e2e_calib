"""Train CalibNetDepth: 2 objects + background, depth-aware, NLL loss."""
import math, time, torch, torch.nn as nn
from dataset import build_loaders_depth
from model_depth import CalibNetDepth
from model_cov import gaussian2d_nll

torch.set_float32_matmul_precision("high")
try:
    torch.backends.cuda.matmul.fp32_precision = "tf32"
    torch.backends.cudnn.conv.fp32_precision  = "tf32"
except AttributeError:
    pass

DEVICE, BATCH_SIZE, EPOCHS, LR = torch.device("cuda"), 64, 80, 1e-3
BG_RATIO = 1   # BG points per obj point. set 3 or 5 for denser BG
CKPT = "best_model_depth.pt"


def epoch_loop(model, loader, optimizer, scaler, train):
    model.train(train)
    total_nll, total_mse, n = 0.0, 0.0, 0
    for imgs, true_uvd, dist_uvd in loader:
        imgs      = imgs.to(DEVICE)
        true_uvd  = true_uvd.to(DEVICE)
        dist_uvd  = dist_uvd.to(DEVICE)
        gt        = true_uvd[..., :2] - dist_uvd[..., :2]   # (B,N,2) UV offset only

        with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
            params = model(imgs, dist_uvd)
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
    print("Depth-aware covariance training (obj1/obj2/background)")
    train_loader, val_loader = build_loaders_depth(
        train_size=8000, val_size=800, batch_size=BATCH_SIZE, num_workers=4,
        bg_ratio=BG_RATIO)

    model     = torch.compile(CalibNetDepth().to(DEVICE), mode="max-autotune")
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
              f"train nll={tr_nll:+.3f} mse={tr_mse:.3f}  "
              f"val nll={va_nll:+.3f} mse={va_mse:.3f}  "
              f"lr={scheduler.get_last_lr()[0]:.2e}  "
              f"tot={(time.time()-t0)/60:.1f}min")
        if va_nll < best_val:
            best_val = va_nll
            raw = model._orig_mod if hasattr(model, "_orig_mod") else model
            torch.save(raw.state_dict(), CKPT)
            print(f"  ↳ saved (val_nll={best_val:.4f})")

    print(f"\nBest val NLL: {best_val:.4f}  |  time: {(time.time()-t0)/60:.1f}min")


if __name__ == "__main__":
    main()
