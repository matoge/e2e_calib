"""ps_v23: PandaSet calib at sigma=3°/0.3m using v9_lazy cache directly.

No pair-path adapter, no live decode — just the same recipe as ps_v9_lazy
but with widened uniform perturbation. Cache already on NVMe and validated.
"""
import sys, pathlib; sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[2]))
import math, time, torch, torch.nn as nn
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from pathlib import Path
from datetime import datetime
from datasets.pandaset_lazy import PandaSetCalibDatasetLazyVFP, collate_pandaset_vfp
from datasets.pandaset import collate_pandaset
from datasets.pandaset_lazy import PandaSetCalibDatasetLazy
from models.model_depth import CalibNetDepth
from models.model_cov import gaussian2d_nll
from torch.utils.data import DataLoader, Subset, ConcatDataset

torch.set_float32_matmul_precision("high")
DEVICE = torch.device("cuda")

CFG = dict(
    name          = "ps_v23_panda_sigma3",
    n_layers      = 2,
    img_size      = 64,
    in_channels   = 3,
    use_convnext  = False,
    use_frustum   = True,
    epochs        = 100,
    batch_size    = 64,
    lr            = 1e-3,
    lr_min        = 1e-6,
    val_fraction  = 0.1,
    split_seed    = 42,
    sigma_ypr     = 3.0,
    sigma_t       = 0.3,
)


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
        gt = true_uvd[..., :2] - dist_uvd[..., :2]
        with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
            params = model(imgs, dist_uvd[..., :3], key_padding_mask=pad_mask)
            valid = ~pad_mask
            loss = gaussian2d_nll(params[valid], gt[valid])
        if train:
            optimizer.zero_grad(set_to_none=True)
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            scaler.step(optimizer); scaler.update()
        with torch.no_grad():
            mse = (params[valid][..., :2] - gt[valid]).norm(dim=-1).mean().item()
            is_obj = valid & (dist_uvd[..., 3] > 0.5)
            is_bg  = valid & (dist_uvd[..., 3] < 0.5)
            if is_obj.any():
                obj_nll_s += gaussian2d_nll(params[is_obj], gt[is_obj]).item(); obj_n += 1
                obj_mse_s += (params[is_obj][..., :2] - gt[is_obj]).norm(dim=-1).mean().item()
            if is_bg.any():
                bg_nll_s  += gaussian2d_nll(params[is_bg],  gt[is_bg]).item();  bg_n  += 1
                bg_mse_s  += (params[is_bg][..., :2] - gt[is_bg]).norm(dim=-1).mean().item()
        total_nll += loss.item(); total_mse += mse; n += 1
    return (total_nll / max(n,1), total_mse / max(n,1),
            obj_nll_s / max(obj_n, 1), bg_nll_s / max(bg_n, 1),
            obj_mse_s / max(obj_n, 1), bg_mse_s / max(bg_n, 1))


def main():
    c = CFG
    exp_dir = Path("experiments") / c["name"]
    exp_dir.mkdir(parents=True, exist_ok=True)
    log_path = exp_dir / "train.log"
    log_path.write_text("")
    (exp_dir / "config.py").write_text(
        "CFG = dict(\n" + "".join(f"    {k:<13}= {v!r},\n" for k, v in c.items()) + ")\n")

    def log(msg):
        ts = datetime.now().strftime("%H:%M:%S")
        line = f"{ts}  {msg}"
        print(line)
        with open(log_path, "a") as f: f.write(line+"\n")

    cache = '/mnt/nvme6t/e2e_calib_cache/pandaset_mc_s384_lazy'
    log(f"loading cache {cache}")
    tr_full = PandaSetCalibDatasetLazy(cache, split='train',
                                       max_offset_m=c['sigma_t'], max_rot_deg=c['sigma_ypr'])
    va_full = PandaSetCalibDatasetLazy(cache, split='val',
                                       max_offset_m=c['sigma_t'], max_rot_deg=c['sigma_ypr'])
    log(f"train cache: {len(tr_full)}  val cache: {len(va_full)}")

    import random as _r
    full_ds = ConcatDataset([tr_full, va_full])
    idxs = list(range(len(full_ds)))
    _r.Random(c["split_seed"]).shuffle(idxs)
    n_val = int(len(idxs) * c["val_fraction"])
    val_idxs, train_idxs = idxs[:n_val], idxs[n_val:]
    train_ds = Subset(full_ds, train_idxs)
    val_ds   = Subset(full_ds, val_idxs)
    log(f"object-level split: train={len(train_ds)} val={len(val_ds)}")

    kw = dict(num_workers=16, pin_memory=True, persistent_workers=True,
              collate_fn=collate_pandaset)
    train_loader = DataLoader(train_ds, batch_size=c["batch_size"], shuffle=True,  **kw)
    val_loader   = DataLoader(val_ds,   batch_size=c["batch_size"], shuffle=False, **kw)

    model = CalibNetDepth(img_size=c["img_size"], in_channels=c["in_channels"],
                          n_layers=c["n_layers"],
                          use_convnext=c["use_convnext"],
                          use_frustum=c["use_frustum"]).to(DEVICE)
    log(f"params: {sum(p.numel() for p in model.parameters())/1e6:.2f}M")

    optimizer = torch.optim.AdamW(model.parameters(), lr=c["lr"], weight_decay=1e-3)
    epochs = c["epochs"]; lr_min_r = c["lr_min"] / c["lr"]
    def lr_lambda(e):
        if e < 5: return (e+1)/5
        t = (e-5)/max(1, epochs-5)
        return lr_min_r + (1-lr_min_r)*0.5*(1+math.cos(math.pi*t))
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)
    scaler = torch.GradScaler(device="cuda")
    best_val = float("inf"); ckpt = exp_dir / "best_model.pt"; t0 = time.time()
    history = {'ep':[], 'tr_nll':[], 'va_nll':[], 'tr_mse':[], 'va_mse':[]}

    for epoch in range(1, epochs+1):
        tr_nll, tr_mse, tr_obj, tr_bg, tr_obj_mse, tr_bg_mse = epoch_loop(model, train_loader, optimizer, scaler, True)
        with torch.no_grad():
            va_nll, va_mse, va_obj, va_bg, va_obj_mse, va_bg_mse = epoch_loop(model, val_loader, optimizer, scaler, False)
        scheduler.step()
        history['ep'].append(epoch)
        history['tr_nll'].append(tr_nll); history['va_nll'].append(va_nll)
        history['tr_mse'].append(tr_mse); history['va_mse'].append(va_mse)
        log(f"[{epoch:3d}/{epochs}]  train nll={tr_nll:+.3f}(obj={tr_obj:+.3f} bg={tr_bg:+.3f}) "
            f"mse={tr_mse:.3f}(obj={tr_obj_mse:.3f} bg={tr_bg_mse:.3f})  "
            f"val nll={va_nll:+.3f}(obj={va_obj:+.3f} bg={va_bg:+.3f}) "
            f"mse={va_mse:.3f}(obj={va_obj_mse:.3f} bg={va_bg_mse:.3f})  "
            f"lr={scheduler.get_last_lr()[0]:.2e}  tot={(time.time()-t0)/60:.1f}min")
        if va_nll < best_val:
            best_val = va_nll
            torch.save(model.state_dict(), ckpt)
            log(f"  ↳ saved (val_nll={best_val:.4f})")
    log(f"Best val NLL: {best_val:.4f}  |  time: {(time.time()-t0)/60:.1f}min")


if __name__ == '__main__':
    main()
