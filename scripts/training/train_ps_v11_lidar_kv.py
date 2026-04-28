"""ps_v11: CalibNetDepth + LiDAR-bank KV concat + pose_emb (vfp scale anchor).

Step 2 of the unified KV plan. KV side becomes:
  KV = concat([image_tokens, point_features])     (no modality emb — encoder
                                                    distributions already differ)
Q (per-point) = PointMLP(uvd) + Frustum + pose_emb
KV adds the SAME pose_emb (broadcast) to image_tokens AND point_features.

pose_emb input = [SE3 6-DoF (=0 for calib), log(vfp)]. vfp varies per sample
(crop_size_in_full_res_px ~ 64..C, output 64×64), giving the model the
scale anchor needed to normalize across cameras / datasets.

Should not regress vs ps_v9_lazy (val_nll 1.8141). If it holds, the same
model + dataset slot in for cross-frame by setting SE3 to the relative
virtual-cam SE(3) and adding a frame_B token bank to KV.
"""
import sys, pathlib; sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[2]))
import argparse, math, time, torch, torch.nn as nn
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from pathlib import Path
from datetime import datetime
from datasets.pandaset_lazy import PandaSetCalibDatasetLazyVFP, collate_pandaset_vfp
from models.model_depth import CalibNetDepth
from models.model_cov import gaussian2d_nll
from torch.utils.data import DataLoader, Subset

torch.set_float32_matmul_precision("high")
DEVICE = torch.device("cuda")

CFG = dict(
    name             = "ps_v11_lidar_kv",
    n_layers         = 2,
    img_size         = 64,
    in_channels      = 3,
    use_convnext     = False,
    use_frustum      = True,
    use_lidar_kv     = True,
    use_pose_emb     = True,
    epochs           = 100,
    batch_size       = 64,
    lr               = 1e-3,
    lr_min           = 1e-6,
    val_fraction     = 0.1,
    split_seed       = 42,
)


def epoch_loop(model, loader, optimizer, scaler, train):
    model.train(train)
    total_nll, total_mse, n = 0.0, 0.0, 0
    obj_nll_s, obj_mse_s, obj_n = 0.0, 0.0, 0
    bg_nll_s,  bg_mse_s,  bg_n  = 0.0, 0.0, 0
    for imgs, true_uvd, dist_uvd, pad_mask, vfp in loader:
        imgs     = imgs.to(DEVICE)
        true_uvd = true_uvd.to(DEVICE)
        dist_uvd = dist_uvd.to(DEVICE)
        pad_mask = pad_mask.to(DEVICE)
        vfp      = vfp.to(DEVICE)
        gt       = true_uvd[..., :2] - dist_uvd[..., :2]
        with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
            params = model(imgs, dist_uvd[..., :3], key_padding_mask=pad_mask, vfp=vfp)
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


def main(cfg=None, clearml=False, clearml_project='e2e_calib/calib', queue=None,
         cache: str = '/mnt/nvme6t/e2e_calib_cache/pandaset_mc_s64_lazy'):
    c = cfg if cfg is not None else CFG
    c = dict(c); c['cache'] = cache       # surface cache path in saved cfg
    cml_task = None
    if clearml:
        from clearml import Task
        cml_task = Task.init(project_name=clearml_project, task_name=c['name'],
                             reuse_last_task_id=False, output_uri=True)
        cml_task.connect(c, name='cfg')
        if queue:
            # Skip auto-detected requirements: rely on agent's system_site_packages=true
            # to pull torch/scipy/etc from the host. Otherwise pip resolves the local
            # Linux+CUDA versions (e.g. scipy==1.16.1) which the remote box may not have.
            cml_task.set_packages([])
            cml_task.execute_remotely(queue_name=queue, exit_process=True)
    # post-connect: trust the cfg dict's cache (ClearML overwrites it with the
    # task's stored hyperparam on remote runs; argparse default would otherwise win).
    cache = c['cache']

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

    kw = dict(num_workers=16, pin_memory=True, persistent_workers=True,
              collate_fn=collate_pandaset_vfp)
    import random as _r
    log(f"loading cache {cache} (lazy)")
    tr_full = PandaSetCalibDatasetLazyVFP(cache, split='train')
    log(f"train cache: {len(tr_full)} instances")
    va_full = PandaSetCalibDatasetLazyVFP(cache, split='val')
    log(f"val   cache: {len(va_full)} instances")
    from torch.utils.data import ConcatDataset
    full_ds = ConcatDataset([tr_full, va_full])
    idxs = list(range(len(full_ds)))
    _r.Random(c["split_seed"]).shuffle(idxs)
    n_val = int(len(idxs) * c["val_fraction"])
    val_idxs, train_idxs = idxs[:n_val], idxs[n_val:]
    train_ds = Subset(full_ds, train_idxs)
    val_ds   = Subset(full_ds, val_idxs)
    log(f"object-level split: train={len(train_ds)} val={len(val_ds)}")

    train_loader = DataLoader(train_ds, batch_size=c["batch_size"], shuffle=True,  **kw)
    val_loader   = DataLoader(val_ds,   batch_size=c["batch_size"], shuffle=False, **kw)

    model = CalibNetDepth(img_size=c["img_size"], in_channels=c["in_channels"],
                          n_layers=c["n_layers"],
                          use_convnext=c.get("use_convnext", False),
                          use_frustum=c.get("use_frustum", False),
                          use_lidar_kv=c.get("use_lidar_kv", False),
                          use_pose_emb=c.get("use_pose_emb", False)).to(DEVICE)
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
    history   = {'ep': [], 'tr_nll': [], 'va_nll': [], 'tr_mse': [], 'va_mse': []}

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
        if cml_task is not None:
            lg = cml_task.get_logger()
            lg.report_scalar('nll', 'train', tr_nll, epoch); lg.report_scalar('nll', 'val', va_nll, epoch)
            lg.report_scalar('mse', 'train', tr_mse, epoch); lg.report_scalar('mse', 'val', va_mse, epoch)
            lg.report_scalar('obj_nll', 'val', va_obj, epoch); lg.report_scalar('bg_nll', 'val', va_bg, epoch)
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
    ap = argparse.ArgumentParser()
    ap.add_argument('--clearml', action='store_true')
    ap.add_argument('--clearml-project', default='e2e_calib/calib')
    ap.add_argument('--queue', default=None,
                    help='If set with --clearml: submit to this queue and exit (remote run).')
    ap.add_argument('--cache', default='/mnt/nvme6t/e2e_calib_cache/pandaset_mc_s64_lazy',
                    help='Lazy disk cache dir. Override on remote hosts (e.g., sakurai2 local SSD).')
    ap.add_argument('--name', default=None, help='override exp name suffix')
    ap.add_argument('--no-pose-emb',  action='store_true', help='ablation: drop pose_emb (no vfp scale anchor)')
    ap.add_argument('--no-lidar-kv',  action='store_true', help='ablation: drop lidar bank from KV')
    args = ap.parse_args()
    cfg = dict(CFG)
    if args.name:        cfg['name'] = args.name
    if args.no_pose_emb: cfg['use_pose_emb']  = False
    if args.no_lidar_kv: cfg['use_lidar_kv']  = False
    main(cfg=cfg, clearml=args.clearml, clearml_project=args.clearml_project,
         queue=args.queue, cache=args.cache)
