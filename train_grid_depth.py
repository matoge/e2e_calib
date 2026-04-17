"""Train CalibNetDepth on grid+depth dataset at 64x64.
Config: edit config_grid_depth.py to change model / checkpoint / training params.
Results saved to experiments/{name}/
"""
import math, time, shutil, logging, sys, torch, torch.nn as nn
from pathlib import Path
from datetime import datetime
from config_grid_depth import CFG
from dataset import GridDepthDataset, collate_grid_depth
from model_depth import CalibNetDepth
from model_cov import gaussian2d_nll
from torch.utils.data import DataLoader

torch.set_float32_matmul_precision("high")

DEVICE = torch.device("cuda")


class Logger:
    """Write to stdout and a log file. File opened/closed per line to avoid fork issues."""
    def __init__(self, log_path: Path):
        self.log_path = log_path
        log_path.write_text("")  # truncate

    def info(self, msg: str):
        ts = datetime.now().strftime("%H:%M:%S")
        line = f"{ts}  {msg}"
        with open(self.log_path, "a", encoding="utf-8") as f:
            f.write(line + "\n")


def epoch_loop(model, loader, optimizer, scaler, train):
    model.train(train)
    total_nll, total_mse, n = 0.0, 0.0, 0
    obj_nll_sum, obj_n = 0.0, 0
    bg_nll_sum,  bg_n  = 0.0, 0
    for imgs, true_uvd, dist_uvd, pad_mask in loader:
        imgs     = imgs.to(DEVICE)
        true_uvd = true_uvd.to(DEVICE)
        dist_uvd = dist_uvd.to(DEVICE)
        pad_mask = pad_mask.to(DEVICE)
        gt       = true_uvd[..., :2] - dist_uvd[..., :2]

        with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
            params = model(imgs, dist_uvd, key_padding_mask=pad_mask)
            valid  = ~pad_mask
            loss   = gaussian2d_nll(params[valid], gt[valid])

        if train:
            optimizer.zero_grad(set_to_none=True)
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            scaler.step(optimizer); scaler.update()

        with torch.no_grad():
            mse = (params[valid][..., :2] - gt[valid]).norm(dim=-1).mean().item()
            # per-group NLL
            depth  = dist_uvd[..., 2]          # (B, N)
            is_bg  = valid & (depth >= 0.95)
            is_obj = valid & (depth <  0.95)
            if is_obj.any():
                obj_nll_sum += gaussian2d_nll(params[is_obj], gt[is_obj]).item()
                obj_n += 1
            if is_bg.any():
                bg_nll_sum += gaussian2d_nll(params[is_bg], gt[is_bg]).item()
                bg_n += 1
        total_nll += loss.item(); total_mse += mse; n += 1

    obj_nll = obj_nll_sum / max(obj_n, 1)
    bg_nll  = bg_nll_sum  / max(bg_n,  1)
    return total_nll / max(n, 1), total_mse / max(n, 1), obj_nll, bg_nll


def main():
    c       = CFG
    name    = c["name"]
    exp_dir = Path("experiments") / name
    exp_dir.mkdir(parents=True, exist_ok=True)

    log  = Logger(exp_dir / "train.log")
    ckpt = exp_dir / "best_model.pt"

    shutil.copy("config_grid_depth.py", exp_dir / "config.py")

    log.info(f"name={name}  n_layers={c['n_layers']} self_first={c['self_first']} "
         f"max_offset={c['max_offset']}  batch={c['batch_size']}  "
         f"epochs={c['epochs']}  lr={c['lr']}→{c['lr_min']}")

    kw = dict(num_workers=4, pin_memory=True, persistent_workers=True,
              collate_fn=collate_grid_depth)
    train_loader = DataLoader(
        GridDepthDataset(c["train_size"], c["img_size"],
                         max_offset=c["max_offset"], random_each_epoch=True),
        batch_size=c["batch_size"], shuffle=True, **kw)
    val_loader = DataLoader(
        GridDepthDataset(c["val_size"], c["img_size"],
                         max_offset=c["max_offset"], base_seed=700_000),
        batch_size=c["batch_size"], shuffle=False, **kw)

    model = CalibNetDepth(
        img_size    = c["img_size"],
        in_channels = c["in_channels"],
        n_layers    = c["n_layers"],
        self_first  = c["self_first"],
    ).to(DEVICE)

    total_params = sum(p.numel() for p in model.parameters())
    log.info(f"params: {total_params/1e6:.2f}M")

    optimizer    = torch.optim.AdamW(model.parameters(), lr=c["lr"], weight_decay=1e-3)
    epochs       = c["epochs"]
    lr_min_ratio = c.get("lr_min", 1e-5) / c["lr"]

    def lr_lambda(e):
        if e < 5: return (e + 1) / 5
        t = (e - 5) / max(1, epochs - 5)
        return lr_min_ratio + (1 - lr_min_ratio) * 0.5 * (1 + math.cos(math.pi * t))

    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)
    scaler    = torch.GradScaler(device="cuda")
    best_val  = float("inf")
    t0        = time.time()

    for epoch in range(1, epochs + 1):
        tr_nll, tr_mse, tr_obj, tr_bg = epoch_loop(model, train_loader, optimizer, scaler, True)
        with torch.no_grad():
            va_nll, va_mse, va_obj, va_bg = epoch_loop(model, val_loader, optimizer, scaler, False)
        scheduler.step()

        log.info(f"[{epoch:3d}/{epochs}]  "
             f"train nll={tr_nll:+.3f}(obj={tr_obj:+.3f} bg={tr_bg:+.3f}) mse={tr_mse:.3f}  "
             f"val nll={va_nll:+.3f}(obj={va_obj:+.3f} bg={va_bg:+.3f}) mse={va_mse:.3f}  "
             f"lr={scheduler.get_last_lr()[0]:.2e}  "
             f"tot={(time.time()-t0)/60:.1f}min")

        if va_nll < best_val:
            best_val = va_nll
            torch.save(model.state_dict(), ckpt)
            log.info(f"  ↳ saved (val_nll={best_val:.4f})")

    log.info(f"Best val NLL: {best_val:.4f}  |  time: {(time.time()-t0)/60:.1f}min")


if __name__ == "__main__":
    main()
