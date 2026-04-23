"""Train CalibNetDepth on Waymo Sign (type=3) dataset."""
import sys, pathlib; sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[2]))
import math, time, torch, torch.nn as nn
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from pathlib import Path
from datetime import datetime
from datasets.waymo import WaymoCalibDataset, collate_waymo, build_cache
from models.model_depth import CalibNetDepth
from models.model_cov import gaussian2d_nll
from torch.utils.data import DataLoader

torch.set_float32_matmul_precision("high")
DEVICE = torch.device("cuda")

CFG = dict(
    name         = "wm_sign_v2",
    n_layers     = 4,
    img_size     = 64,
    in_channels  = 3,
    use_convnext = True,
    use_frustum  = True,
    epochs       = 30,
    batch_size   = 64,
    lr           = 1e-3,
    lr_min       = 1e-6,
)

CACHE = '/tmp/waymo_sign_cache.pt'


def epoch_loop(model, loader, optimizer, scaler, train):
    model.train(train)
    total_nll, total_mse, n = 0.0, 0.0, 0
    obj_nll_s, obj_mse_s, obj_n = 0.0, 0.0, 0
    bg_nll_s,  bg_mse_s,  bg_n  = 0.0, 0.0, 0
    for imgs, true_uvd, dist_uvd, pad_mask in loader:
        imgs     = imgs.to(DEVICE)
        true_uvd = true_uvd.to(DEVICE)
        dist_uvd = dist_uvd.to(DEVICE)
        pad_mask = pad_mask.to(DEVICE)
        gt       = true_uvd[..., :2] - dist_uvd[..., :2]
        with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
            params = model(imgs, dist_uvd[..., :3], key_padding_mask=pad_mask)
            valid  = ~pad_mask
            loss   = gaussian2d_nll(params[valid], gt[valid])
        if train:
            optimizer.zero_grad(set_to_none=True)
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            scaler.step(optimizer); scaler.update()
        with torch.no_grad():
            mse    = (params[valid][..., :2] - gt[valid]).norm(dim=-1).mean().item()
            is_obj = valid & (dist_uvd[..., 3] > 0.5)
            is_bg  = valid & (dist_uvd[..., 3] < 0.5)
            if is_obj.any():
                obj_nll_s += gaussian2d_nll(params[is_obj], gt[is_obj]).item(); obj_n += 1
                obj_mse_s += (params[is_obj][..., :2] - gt[is_obj]).norm(dim=-1).mean().item()
            if is_bg.any():
                bg_nll_s  += gaussian2d_nll(params[is_bg],  gt[is_bg]).item();  bg_n  += 1
                bg_mse_s  += (params[is_bg][...,  :2] - gt[is_bg]).norm(dim=-1).mean().item()
        total_nll += loss.item(); total_mse += mse; n += 1
    obj_nll = obj_nll_s / max(obj_n, 1); obj_mse = obj_mse_s / max(obj_n, 1)
    bg_nll  = bg_nll_s  / max(bg_n,  1); bg_mse  = bg_mse_s  / max(bg_n,  1)
    return total_nll / max(n,1), total_mse / max(n,1), obj_nll, bg_nll, obj_mse, bg_mse


def main():
    c = CFG
    exp_dir = Path("experiments") / c["name"]
    exp_dir.mkdir(parents=True, exist_ok=True)
    log_path = exp_dir / "train.log"
    log_path.write_text("")

    def log(msg):
        ts = datetime.now().strftime("%H:%M:%S")
        line = f"{ts}  {msg}"
        print(line)
        with open(log_path, "a") as f: f.write(line+"\n")

    if not Path(CACHE).exists():
        log("Building Waymo Sign cache...")
        build_cache(CACHE, target_types={3}, frame_sample=0.1)

    kw = dict(num_workers=4, pin_memory=True, persistent_workers=True,
              collate_fn=collate_waymo)
    train_ds = WaymoCalibDataset(CACHE, split='train', max_dist_m=40.0)
    val_ds   = WaymoCalibDataset(CACHE, split='val',   max_dist_m=40.0)
    log(f"train={len(train_ds)} val={len(val_ds)}")

    train_loader = DataLoader(train_ds, batch_size=c["batch_size"], shuffle=True,  **kw)
    val_loader   = DataLoader(val_ds,   batch_size=c["batch_size"], shuffle=False, **kw)

    model = CalibNetDepth(img_size=c["img_size"], in_channels=c["in_channels"],
                          n_layers=c["n_layers"], self_first=False,
                          use_convnext=c.get("use_convnext", False),
                          use_frustum=c.get("use_frustum", False)).to(DEVICE)
    log(f"params: {sum(p.numel() for p in model.parameters())/1e6:.2f}M")

    optimizer = torch.optim.AdamW(model.parameters(), lr=c["lr"], weight_decay=1e-3)
    epochs    = c["epochs"]
    lr_min_r  = c["lr_min"] / c["lr"]
    def lr_lambda(e):
        if e < 5: return (e+1)/5
        t = (e-5)/max(1,epochs-5)
        return lr_min_r + (1-lr_min_r)*0.5*(1+math.cos(math.pi*t))
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)
    scaler    = torch.GradScaler(device="cuda")
    best_val  = float("inf")
    ckpt      = exp_dir / "best_model.pt"
    t0        = time.time()

    for epoch in range(1, epochs+1):
        tr_nll, tr_mse, tr_obj, tr_bg, tr_obj_mse, tr_bg_mse = epoch_loop(
            model, train_loader, optimizer, scaler, True)
        with torch.no_grad():
            va_nll, va_mse, va_obj, va_bg, va_obj_mse, va_bg_mse = epoch_loop(
                model, val_loader, optimizer, scaler, False)
        scheduler.step()
        log(f"[{epoch:3d}/{epochs}]  "
            f"train nll={tr_nll:+.3f}(obj={tr_obj:+.3f} bg={tr_bg:+.3f}) "
            f"mse={tr_mse:.3f}(obj={tr_obj_mse:.3f} bg={tr_bg_mse:.3f})  "
            f"val nll={va_nll:+.3f}(obj={va_obj:+.3f} bg={va_bg:+.3f}) "
            f"mse={va_mse:.3f}(obj={va_obj_mse:.3f} bg={va_bg_mse:.3f})  "
            f"lr={scheduler.get_last_lr()[0]:.2e}  tot={(time.time()-t0)/60:.1f}min")
        if va_nll < best_val:
            best_val = va_nll
            torch.save(model.state_dict(), ckpt)
            log(f"  ↳ saved (val_nll={best_val:.4f})")

    log(f"Best val NLL: {best_val:.4f}  |  time: {(time.time()-t0)/60:.1f}min")

    # ── vis ──────────────────────────────────────────────────────────────────
    model.load_state_dict(torch.load(ckpt, map_location=DEVICE, weights_only=True))
    model.eval()
    vis_dir = exp_dir / "vis"; vis_dir.mkdir(exist_ok=True)
    n_vis = min(12, len(val_ds))
    for vi in range(n_vis):
        img, true_uvd, dist_uvd = val_ds[vi]
        with torch.no_grad():
            pad = torch.zeros(1, true_uvd.shape[0], dtype=torch.bool, device=DEVICE)
            params = model(img.unsqueeze(0).to(DEVICE),
                           dist_uvd.unsqueeze(0).to(DEVICE)[..., :3],
                           key_padding_mask=pad)[0].cpu().float()
        pred_uv = (dist_uvd[:, :2] + params[:, :2]).numpy()
        true_uv = true_uvd[:, :2].numpy()
        dist_uv = dist_uvd[:, :2].numpy()
        is_obj  = dist_uvd[:, 3].numpy() > 0.5

        err_b_obj = float((abs(dist_uv[is_obj]  - true_uv[is_obj])).sum(1).mean())  if is_obj.any()  else float('nan')
        err_a_obj = float((abs(pred_uv[is_obj]   - true_uv[is_obj])).sum(1).mean()) if is_obj.any()  else float('nan')
        err_b_bg  = float((abs(dist_uv[~is_obj] - true_uv[~is_obj])).sum(1).mean()) if (~is_obj).any() else float('nan')
        err_a_bg  = float((abs(pred_uv[~is_obj]  - true_uv[~is_obj])).sum(1).mean()) if (~is_obj).any() else float('nan')

        fig, ax = plt.subplots(figsize=(4, 4), dpi=96)
        ax.imshow(img.permute(1,2,0).numpy())
        for mask, col_gt, col_d, col_p, lbl in [
            (is_obj,  'lime',  'red',    'cyan',        'sign'),
            (~is_obj, 'yellow','orange', 'deepskyblue', 'bg'),
        ]:
            if not mask.any(): continue
            ax.scatter(true_uv[mask,0], true_uv[mask,1], c=col_gt, s=20, marker='x', linewidths=1.2, label=f'GT {lbl}', zorder=3)
            ax.scatter(dist_uv[mask,0], dist_uv[mask,1], c=col_d,  s=10, alpha=0.5, label=f'dist {lbl}', zorder=2)
            ax.scatter(pred_uv[mask,0], pred_uv[mask,1], c=col_p,  s=20, marker='+', linewidths=1.4, label=f'pred {lbl}', zorder=4)
        ax.set_title(f"sign:{err_b_obj:.2f}→{err_a_obj:.2f}  bg:{err_b_bg:.2f}→{err_a_bg:.2f}px", fontsize=6)
        ax.axis('off'); ax.legend(fontsize=4, loc='upper right', framealpha=0.5, ncol=2)
        plt.tight_layout(pad=0.2)
        plt.savefig(vis_dir / f"val_{vi:02d}.png", dpi=96, bbox_inches='tight')
        plt.close(fig)
    log(f"Saved {n_vis} vis → {vis_dir}")


if __name__ == "__main__":
    main()
