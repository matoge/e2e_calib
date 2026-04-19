"""Train CalibNetDepth on nuScenes + PandaSet combined."""
import math, time, torch, torch.nn as nn, numpy as np
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from pathlib import Path
from datetime import datetime
from dataset_nuscenes import NuScenesCalibDataset
from dataset_pandaset import PandaSetCalibDataset
from model_depth import CalibNetDepth
from model_cov import gaussian2d_nll
from torch.utils.data import DataLoader, ConcatDataset

torch.set_float32_matmul_precision("high")
DEVICE = torch.device("cuda")

CFG = dict(
    name         = "ns_ps_v2",
    n_layers     = 4,
    img_size     = 64,
    in_channels  = 3,
    use_convnext = True,
    use_frustum  = True,
    epochs       = 100,
    batch_size   = 64,
    lr           = 1e-3,
    lr_min       = 1e-6,
)

NS_CACHE = '/tmp/nuscenes_static_cache.pt'
PS_CACHE = '/tmp/pandaset_cache.pt'


def collate_fn(batch):
    imgs, true_uvds, dist_uvds = zip(*batch)
    max_n = max(t.shape[0] for t in true_uvds)
    def pad(seqs):
        out  = torch.zeros(len(seqs), max_n, seqs[0].shape[1])
        mask = torch.ones(len(seqs), max_n, dtype=torch.bool)
        for i, s in enumerate(seqs):
            out[i, :s.shape[0]] = s
            mask[i, :s.shape[0]] = False
        return out, mask
    true_t, _    = pad(true_uvds)
    dist_t, mask = pad(dist_uvds)
    return torch.stack(imgs), true_t, dist_t, mask


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
                bg_nll_s  += gaussian2d_nll(params[is_bg], gt[is_bg]).item();  bg_n  += 1
                bg_mse_s  += (params[is_bg][...,  :2] - gt[is_bg]).norm(dim=-1).mean().item()
        total_nll += loss.item(); total_mse += mse; n += 1
    obj_nll = obj_nll_s / max(obj_n, 1); obj_mse = obj_mse_s / max(obj_n, 1)
    bg_nll  = bg_nll_s  / max(bg_n,  1); bg_mse  = bg_mse_s  / max(bg_n,  1)
    return total_nll / max(n, 1), total_mse / max(n, 1), obj_nll, bg_nll, obj_mse, bg_mse


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
        with open(log_path, "a") as f: f.write(line + "\n")

    ns_train = NuScenesCalibDataset(NS_CACHE, split='train')
    ns_val   = NuScenesCalibDataset(NS_CACHE, split='val')
    ps_train = PandaSetCalibDataset(PS_CACHE, split='train')
    ps_val   = PandaSetCalibDataset(PS_CACHE, split='val')

    train_ds = ConcatDataset([ns_train, ps_train])
    val_ds   = ConcatDataset([ns_val,   ps_val])
    log(f"train={len(train_ds)} (ns={len(ns_train)} ps={len(ps_train)})  "
        f"val={len(val_ds)} (ns={len(ns_val)} ps={len(ps_val)})")

    kw = dict(num_workers=4, pin_memory=True, persistent_workers=True, collate_fn=collate_fn)
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
        t = (e-5)/max(1, epochs-5)
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

    # ── visualization & report ────────────────────────────────────────────────
    model.load_state_dict(torch.load(ckpt, map_location=DEVICE, weights_only=True))
    model.eval()
    vis_dir = exp_dir / "vis"; vis_dir.mkdir(exist_ok=True)

    # parse training log for curves
    import re
    log_lines = log_path.read_text().splitlines()
    curve = {'epoch': [], 'tr_nll': [], 'va_nll': [], 'tr_mse': [], 'va_mse': []}
    for line in log_lines:
        m = re.search(r'\[\s*(\d+)/\d+\].*train nll=([+-]?\d+\.\d+).*mse=(\d+\.\d+).*val nll=([+-]?\d+\.\d+).*mse=(\d+\.\d+)', line)
        if m:
            curve['epoch'].append(int(m.group(1)))
            curve['tr_nll'].append(float(m.group(2)))
            curve['tr_mse'].append(float(m.group(3)))
            curve['va_nll'].append(float(m.group(4)))
            curve['va_mse'].append(float(m.group(5)))

    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    axes[0].plot(curve['epoch'], curve['tr_nll'], label='train NLL')
    axes[0].plot(curve['epoch'], curve['va_nll'], label='val NLL')
    axes[0].set_xlabel('epoch'); axes[0].set_ylabel('NLL'); axes[0].legend(); axes[0].grid(True)
    axes[0].set_title('NLL (lower=better)')
    axes[1].plot(curve['epoch'], curve['tr_mse'], label='train MSE')
    axes[1].plot(curve['epoch'], curve['va_mse'], label='val MSE')
    axes[1].set_xlabel('epoch'); axes[1].set_ylabel('MSE (px)'); axes[1].legend(); axes[1].grid(True)
    axes[1].set_title('MSE px (lower=better)')
    plt.tight_layout()
    plt.savefig(vis_dir / 'curves.png', dpi=120); plt.close()

    # sample visualizations
    n_vis = min(16, len(val_ds))
    vis_imgs = []
    err_before_list, err_after_list = [], []
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

        err_b = float(np.linalg.norm(dist_uv - true_uv, axis=1).mean())
        err_a = float(np.linalg.norm(pred_uv - true_uv, axis=1).mean())
        err_before_list.append(err_b); err_after_list.append(err_a)

        fig, ax = plt.subplots(figsize=(3, 3), dpi=96)
        ax.imshow(img.permute(1, 2, 0).numpy())
        for mask, cgt, cd, cp in [(is_obj, 'lime', 'red', 'cyan'), (~is_obj, 'yellow', 'orange', 'deepskyblue')]:
            if not mask.any(): continue
            ax.scatter(true_uv[mask,0], true_uv[mask,1], c=cgt, s=16, marker='x', linewidths=1.2, zorder=4)
            ax.scatter(dist_uv[mask,0], dist_uv[mask,1], c=cd,  s=8,  alpha=0.5, zorder=2)
            ax.scatter(pred_uv[mask,0], pred_uv[mask,1], c=cp,  s=16, marker='+', linewidths=1.2, zorder=3)
        ax.set_title(f'{err_b:.1f}→{err_a:.1f}px', fontsize=7)
        ax.axis('off'); plt.tight_layout(pad=0.2)
        p = vis_dir / f'val_{vi:02d}.png'
        plt.savefig(p, dpi=96, bbox_inches='tight'); plt.close()
        vis_imgs.append(p.name)

    mean_before = float(np.mean(err_before_list))
    mean_after  = float(np.mean(err_after_list))
    improvement = (mean_before - mean_after) / mean_before * 100

    # HTML report
    img_tags = ''.join(f'<img src="vis/{n}" style="width:180px;margin:4px">' for n in vis_imgs)
    html = f"""<!DOCTYPE html><html><head><meta charset="utf-8">
<title>{c["name"]} Report</title>
<style>body{{font-family:monospace;background:#111;color:#eee;padding:24px}}
h1{{color:#7cf}}table{{border-collapse:collapse}}td,th{{padding:6px 14px;border:1px solid #444}}
th{{background:#222}}.good{{color:#7f7}}.bad{{color:#f77}}</style></head><body>
<h1>{c["name"]} — Evaluation Report</h1>
<h2>Summary</h2>
<table>
<tr><th>Metric</th><th>Value</th></tr>
<tr><td>Best val NLL</td><td class="good">{best_val:.4f}</td></tr>
<tr><td>Mean err BEFORE</td><td class="bad">{mean_before:.2f} px</td></tr>
<tr><td>Mean err AFTER</td><td class="good">{mean_after:.2f} px</td></tr>
<tr><td>Improvement</td><td class="good">{improvement:.1f}%</td></tr>
<tr><td>Train samples</td><td>{len(train_ds)}</td></tr>
<tr><td>Val samples</td><td>{len(val_ds)}</td></tr>
<tr><td>Params</td><td>{sum(p.numel() for p in model.parameters())/1e6:.2f}M</td></tr>
</table>
<h2>Training Curves</h2>
<img src="vis/curves.png" style="max-width:800px">
<h2>Validation Samples</h2>
<p>green×=GT&nbsp; red●=distorted&nbsp; cyan+=predicted&nbsp; (title: before→after px)</p>
{img_tags}
</body></html>"""
    (exp_dir / 'report.html').write_text(html)
    log(f"Report → {exp_dir}/report.html  err {mean_before:.2f}→{mean_after:.2f}px ({improvement:.1f}% imp)")


if __name__ == "__main__":
    main()
