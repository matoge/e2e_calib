"""Train CalibNetDepth on nuScenes (ps_v9 parity: 4-layer, 200 ep, object-level split)."""
import math, time, torch, torch.nn as nn
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from pathlib import Path
from datetime import datetime
from dataset_nuscenes import NuScenesCalibDataset, collate_nuscenes
from model_depth import CalibNetDepth
from model_cov import gaussian2d_nll
from torch.utils.data import DataLoader, Subset, ConcatDataset

torch.set_float32_matmul_precision("high")
DEVICE = torch.device("cuda")

CFG = dict(
    name          = "ns_v9_objsplit",
    n_layers      = 4,
    img_size      = 64,
    in_channels   = 3,
    use_convnext  = True,
    use_frustum   = True,
    epochs        = 200,
    batch_size    = 64,
    lr            = 1e-3,
    lr_min        = 1e-6,
    val_fraction  = 0.1,
    split_seed    = 42,
)

CACHE = '/tmp/nuscenes_static_cache.pt'


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


def main(cfg=None):
    c = cfg if cfg is not None else CFG
    exp_dir = Path("experiments") / c["name"]
    exp_dir.mkdir(parents=True, exist_ok=True)
    log_path = exp_dir / "train.log"
    log_path.write_text("")

    (exp_dir / "config.py").write_text(
        "CFG = dict(\n" +
        "".join(f"    {k:<13}= {v!r},\n" for k, v in c.items()) + ")\n")

    def log(msg):
        ts = datetime.now().strftime("%H:%M:%S")
        line = f"{ts}  {msg}"
        print(line)
        with open(log_path, "a") as f: f.write(line+"\n")

    cache = CACHE
    kw = dict(num_workers=4, pin_memory=True, persistent_workers=True,
              collate_fn=collate_nuscenes)
    # Merge train+val, object-level split (ps_v9 parity)
    import random as _r
    tr_full = NuScenesCalibDataset(cache, split='train')
    va_full = NuScenesCalibDataset(cache, split='val')
    full_ds = ConcatDataset([tr_full, va_full])
    idxs = list(range(len(full_ds)))
    _r.Random(c["split_seed"]).shuffle(idxs)
    n_val_obj = int(len(idxs) * c["val_fraction"])
    val_idxs, train_idxs = idxs[:n_val_obj], idxs[n_val_obj:]
    if c.get("min_pts"):
        mp = int(c["min_pts"])
        def _dense(i):
            tr_len = len(tr_full)
            inst = tr_full.instances[i] if i < tr_len else va_full.instances[i-tr_len]
            return inst['pts'].shape[0] >= mp
        train_idxs = [i for i in train_idxs if _dense(i)]
        val_idxs   = [i for i in val_idxs   if _dense(i)]
        log(f"min_pts={mp} filter: kept {len(train_idxs)} train, {len(val_idxs)} val")
    if c.get("subset_size"):
        k = int(c["subset_size"])
        train_idxs = train_idxs[:k]
        val_idxs   = val_idxs[:max(1, k // 10)]
    train_ds = Subset(full_ds, train_idxs)
    val_ds   = Subset(full_ds, val_idxs)
    log(f"object-level split: train={len(train_ds)} val={len(val_ds)} (seed={c['split_seed']})")

    train_loader = DataLoader(train_ds, batch_size=c["batch_size"], shuffle=True,  **kw)
    val_loader   = DataLoader(val_ds,   batch_size=c["batch_size"], shuffle=False, **kw)

    model = CalibNetDepth(img_size=c["img_size"], in_channels=c["in_channels"],
                          n_layers=c["n_layers"], self_first=c.get("self_first", False),
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

    history = {'ep': [], 'tr_nll': [], 'va_nll': [], 'tr_mse': [], 'va_mse': []}

    for epoch in range(1, epochs+1):
        tr_nll, tr_mse, tr_obj, tr_bg, tr_obj_mse, tr_bg_mse = epoch_loop(
            model, train_loader, optimizer, scaler, True)
        with torch.no_grad():
            va_nll, va_mse, va_obj, va_bg, va_obj_mse, va_bg_mse = epoch_loop(
                model, val_loader, optimizer, scaler, False)
        scheduler.step()
        history['ep'].append(epoch)
        history['tr_nll'].append(tr_nll); history['va_nll'].append(va_nll)
        history['tr_mse'].append(tr_mse); history['va_mse'].append(va_mse)
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

    vis_dir = exp_dir / "vis"; vis_dir.mkdir(exist_ok=True)
    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    axes[0].plot(history['ep'], history['tr_nll'], label='train'); axes[0].plot(history['ep'], history['va_nll'], label='val')
    axes[0].set_title('NLL'); axes[0].set_xlabel('epoch'); axes[0].legend(); axes[0].grid(alpha=0.3)
    axes[1].plot(history['ep'], history['tr_mse'], label='train'); axes[1].plot(history['ep'], history['va_mse'], label='val')
    axes[1].set_title('MSE (px)'); axes[1].set_xlabel('epoch'); axes[1].legend(); axes[1].grid(alpha=0.3)
    plt.tight_layout(); plt.savefig(vis_dir / "curves.png", dpi=100); plt.close(fig)


if __name__ == "__main__":
    main()
