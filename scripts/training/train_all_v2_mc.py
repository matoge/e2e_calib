"""Joint NS+PS+WM training on multicrop caches.

Successor to all_v2. Same model config, caches under /mnt/nvme6t/e2e_calib_cache/.
"""
import sys, pathlib; sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[2]))
import math, time, torch, torch.nn as nn
import matplotlib; matplotlib.use("Agg")
from pathlib import Path
from datetime import datetime
from datasets.nuscenes import NuScenesCalibDataset
from datasets.pandaset import PandaSetCalibDataset
from datasets.waymo     import WaymoCalibDataset
from models.model_depth import CalibNetDepth
from models.model_cov import gaussian2d_nll
from torch.utils.data import DataLoader, Subset, ConcatDataset

torch.set_float32_matmul_precision("high")
DEVICE = torch.device("cuda")

CFG = dict(
    name          = "all_v3_mc",
    n_layers      = 4,
    img_size      = 64,
    in_channels   = 3,
    use_convnext  = True,
    use_frustum   = True,
    epochs        = 200,
    batch_size    = 64,
    lr            = 1e-3,
    lr_min        = 1e-7,
    n_train_per   = 20000,
    n_val_per     = 500,
    split_seed    = 42,
)

CACHE_DIR = '/mnt/nvme6t/e2e_calib_cache'
NS_CACHE = f'{CACHE_DIR}/nuscenes_mc_cache.pt'
PS_CACHE = f'{CACHE_DIR}/pandaset_mc_cache.pt'
WM_CACHE = f'{CACHE_DIR}/waymo_mc_cache.pt'


def collate_mixed(batch):
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
                bg_nll_s  += gaussian2d_nll(params[is_bg],  gt[is_bg]).item();  bg_n  += 1
                bg_mse_s  += (params[is_bg][...,  :2] - gt[is_bg]).norm(dim=-1).mean().item()
        total_nll += loss.item(); total_mse += mse; n += 1
    obj_nll = obj_nll_s / max(obj_n, 1); obj_mse = obj_mse_s / max(obj_n, 1)
    bg_nll  = bg_nll_s  / max(bg_n,  1); bg_mse  = bg_mse_s  / max(bg_n,  1)
    return total_nll / max(n,1), total_mse / max(n,1), obj_nll, bg_nll, obj_mse, bg_mse


def main(cfg=None):
    import random as _r
    c = cfg or CFG
    exp_dir = Path("experiments") / c["name"]
    exp_dir.mkdir(parents=True, exist_ok=True)
    log_path = exp_dir / "train.log"
    log_path.write_text("")
    (exp_dir / "config.py").write_text(
        "CFG = dict(\n" + "".join(f"    {k:<13}= {v!r},\n" for k, v in c.items()) + ")\n")

    def log(msg):
        ts = datetime.now().strftime("%H:%M:%S")
        line = f"{ts}  {msg}"
        print(line, flush=True)
        with open(log_path, "a") as f: f.write(line+"\n")

    n_tr, n_va = c["n_train_per"], c["n_val_per"]
    rng = _r.Random(c["split_seed"])
    import gc

    def _subset(ds_cls, cache, name):
        """Load cache, sample train/val subsets, then drop the rest to free RAM."""
        tr = ds_cls(cache, split='train')
        va = ds_cls(cache, split='val')
        full = ConcatDataset([tr, va])
        log(f"{name} full: {len(full)}")
        idxs = list(range(len(full))); rng.shuffle(idxs)
        val_part   = idxs[:n_va]
        train_part = idxs[n_va:n_va + n_tr]
        pick = sorted(set(val_part + train_part))
        remap = {old: new for new, old in enumerate(pick)}
        n_tr_split = len(tr)
        kept_instances = [tr.instances[i] if i < n_tr_split
                          else va.instances[i - n_tr_split]
                          for i in pick]
        tr.instances = kept_instances
        va.instances = kept_instances  # share
        tr_idxs_new = [remap[i] for i in train_part]
        va_idxs_new = [remap[i] for i in val_part]
        log(f"{name}: train={len(tr_idxs_new)} val={len(va_idxs_new)}  (dropped {len(full)-len(kept_instances)} unused)")
        del full, idxs, pick, remap
        gc.collect()
        return Subset(tr, tr_idxs_new), Subset(va, va_idxs_new)

    # Load biggest cache first (WM) while the most RAM is available.
    tr_wm, va_wm = _subset(WaymoCalibDataset,    WM_CACHE, 'WM')
    tr_ps, va_ps = _subset(PandaSetCalibDataset, PS_CACHE, 'PS')
    tr_ns, va_ns = _subset(NuScenesCalibDataset, NS_CACHE, 'NS')
    train_ds = ConcatDataset([tr_ns, tr_ps, tr_wm])
    val_ds   = ConcatDataset([va_ns, va_ps, va_wm])
    log(f"joint: train={len(train_ds)} val={len(val_ds)} (seed={c['split_seed']})")

    kw = dict(num_workers=4, pin_memory=True, persistent_workers=True,
              collate_fn=collate_mixed)
    train_loader = DataLoader(train_ds, batch_size=c["batch_size"], shuffle=True,  **kw)
    val_loader   = DataLoader(val_ds,   batch_size=c["batch_size"], shuffle=False, **kw)

    model = CalibNetDepth(img_size=c["img_size"], in_channels=c["in_channels"],
                          n_layers=c["n_layers"], self_first=c.get("self_first", False),
                          use_convnext=c.get("use_convnext", False),
                          use_frustum=c.get("use_frustum", False),
                          deform_mode=c.get("deform_mode", "none"),
                          deform_n_points=c.get("deform_n_points", 4)).to(DEVICE)
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


if __name__ == "__main__":
    main()
