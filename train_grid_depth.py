"""Train CalibNetDepth on grid+depth dataset at 64x64.
Config: edit config_grid_depth.py to change model / checkpoint / training params.
Results saved to experiments/{name}/
"""
import math, time, shutil, logging, sys, torch, torch.nn as nn
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from pathlib import Path
from datetime import datetime
from config_grid_depth import CFG
from dataset import GridDepthDataset, collate_grid_depth, make_image_and_points_grid_depth
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
    obj_nll_sum, obj_mse_sum, obj_n = 0.0, 0.0, 0
    bg_nll_sum,  bg_mse_sum,  bg_n  = 0.0, 0.0, 0
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
            mse   = (params[valid][..., :2] - gt[valid]).norm(dim=-1).mean().item()
            depth = dist_uvd[..., 2]
            max_d = depth.max(dim=1, keepdim=True).values
            is_bg  = valid & (depth >= max_d - 1e-3)
            is_obj = valid & (depth <  max_d - 1e-3)
            if is_obj.any():
                obj_nll_sum += gaussian2d_nll(params[is_obj], gt[is_obj]).item()
                obj_mse_sum += (params[is_obj][..., :2] - gt[is_obj]).norm(dim=-1).mean().item()
                obj_n += 1
            if is_bg.any():
                bg_nll_sum += gaussian2d_nll(params[is_bg], gt[is_bg]).item()
                bg_mse_sum += (params[is_bg][..., :2] - gt[is_bg]).norm(dim=-1).mean().item()
                bg_n += 1
        total_nll += loss.item(); total_mse += mse; n += 1

    obj_nll = obj_nll_sum / max(obj_n, 1)
    bg_nll  = bg_nll_sum  / max(bg_n,  1)
    obj_mse = obj_mse_sum / max(obj_n, 1)
    bg_mse  = bg_mse_sum  / max(bg_n,  1)
    return total_nll / max(n, 1), total_mse / max(n, 1), obj_nll, bg_nll, obj_mse, bg_mse


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
    rd = c.get("random_depths", False)
    train_loader = DataLoader(
        GridDepthDataset(c["train_size"], c["img_size"],
                         max_offset=c["max_offset"], random_each_epoch=True,
                         random_depths=rd),
        batch_size=c["batch_size"], shuffle=True, **kw)
    val_loader = DataLoader(
        GridDepthDataset(c["val_size"], c["img_size"],
                         max_offset=c["max_offset"], base_seed=700_000,
                         random_depths=rd),
        batch_size=c["batch_size"], shuffle=False, **kw)

    model = CalibNetDepth(
        img_size     = c["img_size"],
        in_channels  = c["in_channels"],
        n_layers     = c["n_layers"],
        self_first   = c["self_first"],
        kv_self_attn = c.get("kv_self_attn", False),
        cross_temp   = c.get("cross_temp", 1.0),
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

    scheduler  = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)
    scaler     = torch.GradScaler(device="cuda")
    best_val   = float("inf")
    t0         = time.time()
    temp_start = c.get("cross_temp_start", 1.0)
    temp_end   = c.get("cross_temp",       1.0)

    for epoch in range(1, epochs + 1):
        # cosine anneal temperature: temp_start → temp_end
        frac  = (epoch - 1) / max(1, epochs - 1)
        temp  = temp_end + (temp_start - temp_end) * 0.5 * (1 + math.cos(math.pi * frac))
        if hasattr(model, 'set_cross_temp'):
            model.set_cross_temp(temp)

        tr_nll, tr_mse, tr_obj, tr_bg, tr_obj_mse, tr_bg_mse = epoch_loop(model, train_loader, optimizer, scaler, True)
        with torch.no_grad():
            va_nll, va_mse, va_obj, va_bg, va_obj_mse, va_bg_mse = epoch_loop(model, val_loader, optimizer, scaler, False)
        scheduler.step()

        log.info(f"[{epoch:3d}/{epochs}]  "
             f"train nll={tr_nll:+.3f}(obj={tr_obj:+.3f} bg={tr_bg:+.3f}) "
             f"mse={tr_mse:.3f}(obj={tr_obj_mse:.3f} bg={tr_bg_mse:.3f})  "
             f"val nll={va_nll:+.3f}(obj={va_obj:+.3f} bg={va_bg:+.3f}) "
             f"mse={va_mse:.3f}(obj={va_obj_mse:.3f} bg={va_bg_mse:.3f})  "
             f"lr={scheduler.get_last_lr()[0]:.2e}  temp={temp:.3f}  "
             f"tot={(time.time()-t0)/60:.1f}min")

        if va_nll < best_val:
            best_val = va_nll
            torch.save(model.state_dict(), ckpt)
            log.info(f"  ↳ saved (val_nll={best_val:.4f})")

    log.info(f"Best val NLL: {best_val:.4f}  |  time: {(time.time()-t0)/60:.1f}min")

    # ── Val visualization ───────────────────────────────────────────────────
    vis_dir = exp_dir / "vis"
    vis_dir.mkdir(exist_ok=True)
    model.load_state_dict(torch.load(ckpt, map_location=DEVICE, weights_only=True))
    model.eval()
    cmap = plt.cm.tab10
    n_vis = 12
    for vi in range(n_vis):
        seed = 700_000 + vi
        img, true_uvd, dist_uvd = make_image_and_points_grid_depth(
            img_size=c["img_size"], max_offset=c["max_offset"],
            seed=seed, random_depths=rd)
        with torch.no_grad():
            params = model(img.unsqueeze(0).to(DEVICE),
                           dist_uvd.unsqueeze(0).to(DEVICE))[0].cpu().float()
        pred_uv = (dist_uvd[:, :2] + params[:, :2]).clamp(0, c["img_size"] - 1).numpy()
        true_uv = true_uvd[:, :2].numpy()
        dist_uv = dist_uvd[:, :2].numpy()

        # detect groups by depth
        depths = dist_uvd[:, 2]
        groups, start, cur = [], 0, depths[0].item()
        for i in range(1, len(depths)):
            d = depths[i].item()
            if abs(d - cur) > 1e-4:
                groups.append((start, i, cur)); start = i; cur = d
        groups.append((start, len(depths), cur))

        fig, ax = plt.subplots(figsize=(4, 4), dpi=96)
        img_np = img.numpy().transpose(1, 2, 0)
        ax.imshow(img_np, origin="upper", extent=[0, c["img_size"], c["img_size"], 0])
        for gi, (s, e, depth_val) in enumerate(groups):
            col = cmap(gi % 10)
            lbl = "bg" if gi == len(groups) - 1 else f"obj{gi+1}"
            ax.scatter(true_uv[s:e,0], true_uv[s:e,1], color=col, s=25, marker='x',
                       linewidths=1.2, alpha=0.9, label=f"GT {lbl} d={depth_val:.2f}")
            ax.scatter(dist_uv[s:e,0], dist_uv[s:e,1], color=col, s=18, marker='o',
                       alpha=0.6, linewidths=0)
            ax.scatter(pred_uv[s:e,0], pred_uv[s:e,1], color=col, s=25, marker='+',
                       linewidths=1.4, alpha=0.9)
            for dx, dy, px, py in zip(dist_uv[s:e,0], dist_uv[s:e,1],
                                       pred_uv[s:e,0], pred_uv[s:e,1]):
                ax.annotate("", xy=(px, py), xytext=(dx, dy),
                            arrowprops=dict(arrowstyle="->", color=col, lw=0.6, alpha=0.6))
            # depth label at centroid
            cx, cy = dist_uv[s:e,0].mean(), dist_uv[s:e,1].mean()
            ax.text(cx, cy - 2, f"{lbl}\nd={depth_val:.2f}", color=col,
                    fontsize=5, ha='center', va='bottom',
                    bbox=dict(boxstyle='round,pad=0.1', fc='black', alpha=0.5, ec='none'))

        obj_idx = sum(e - s for gi, (s, e, _) in enumerate(groups) if gi < len(groups) - 1)
        s_obj = groups[0][0]; e_obj = groups[-1][0]
        obj_err_b = float(torch.tensor(dist_uv[:e_obj] - true_uv[:e_obj]).norm(dim=1).mean())
        obj_err_a = float(torch.tensor(pred_uv[:e_obj] - true_uv[:e_obj]).norm(dim=1).mean())
        ax.set_title(f"seed={vi}  obj: {obj_err_b:.2f}→{obj_err_a:.2f}px", fontsize=7)
        ax.set_xlim(0, c["img_size"]); ax.set_ylim(c["img_size"], 0)
        ax.set_xticks([]); ax.set_yticks([])
        ax.legend(fontsize=5, loc="upper right", framealpha=0.5)
        plt.tight_layout(pad=0.2)
        plt.savefig(vis_dir / f"val_{vi:02d}.png", dpi=96, bbox_inches="tight")
        plt.close(fig)
    log.info(f"Saved {n_vis} val visualizations → {vis_dir}")


if __name__ == "__main__":
    main()
