"""ps_v12: cross-frame overfit test with the unified arch (CalibNetDepthPair).

Minimal scope:
  - Single PandaSet scene
  - baseline_range = (1, 10)  → within ~1s of motion (10 Hz capture)
  - n_overfit = 64 fixed samples, 100 epochs
  - GOAL: confirm the unified arch handles cross-frame trivially when it's
    given matched frames + a perturbed pose. If train_nll drops cleanly
    and val_nll follows, the design (Q in target frame + per-frame VCPE
    on KV banks) is validated and we can scale up (full PandaSet, larger
    baseline, then multi-camera, then dataset mixing).

Why minimal first:
  v9_lazy (calib) hit val_nll=1.81. v11 (calib + scaffolding for cross-
  frame) hit 1.89 — within noise. The KV-bank + pose_emb plumbing didn't
  break calib. Now we test whether the SAME forward path, fed pair data,
  learns the residual flow in B's frame from a perturbed pose. Tiny
  baseline = tiny per-point flow = the easiest possible overfit task.
"""
import sys, pathlib; sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[2]))

import argparse, math, time
import torch, torch.nn as nn
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from pathlib import Path
from datetime import datetime

from datasets.pandaset_pair import PandaSetCrossFrameDataset
from models.model_pair import CalibNetDepthPair
from models.model_cov import gaussian2d_nll
from torch.utils.data import DataLoader

torch.set_float32_matmul_precision("high")
DEVICE = torch.device("cuda")

CFG = dict(
    name             = "ps_v12_cross_frame_overfit",
    n_layers         = 2,
    img_size         = 64,
    in_channels      = 3,
    use_convnext     = False,
    use_frustum      = True,
    use_lidar_kv     = True,
    use_pose_emb     = True,
    epochs           = 100,
    batch_size       = 16,
    lr               = 1e-3,
    lr_min           = 1e-6,
    n_overfit        = 64,
    virtual_epoch    = 64,
    max_points       = 256,
    crop_min         = 128,
    crop_max         = 256,
    sigma_ypr        = 0.5,
    sigma_t          = 0.05,
    baseline_min     = 1,
    baseline_max     = 10,    # 10 frames @ 10 Hz = 1.0 s
    cameras          = "front_camera",
    scene            = "/mnt/mininas/datasets/pandaset/015",
    scenes_root      = None,    # CLI-only override; ClearML connect() needs the
                                 # key to exist in the dict to fill from stored val
    train_frac       = 0.8,
    num_workers      = 8,
)


def collate_pair(batch):
    """Stack the per-frame and per-pair tensors from `_try_one_nframe(N=2)`."""
    keys = ['patches', 'uvd', 'pad', 'pose_hat_6dof', 'uv_hat', 'uv_gt', 'pad_dir']
    out = {k: torch.stack([b[k] for b in batch], dim=0) for k in keys}
    return out


def step(model, batch):
    """Run forward_pair for the (A=0, B=1) direction and return loss + metrics."""
    patches  = batch['patches'].to(DEVICE)         # (B, 2, 3, H, W)
    uvd      = batch['uvd'].to(DEVICE)             # (B, 2, N, 4)
    pad      = batch['pad'].to(DEVICE)             # (B, 2, N)
    pose_hat = batch['pose_hat_6dof'].to(DEVICE)   # (B, 2, 2, 6)
    uv_hat   = batch['uv_hat'].to(DEVICE)          # (B, 2, 2, N, 2)
    uv_gt    = batch['uv_gt'].to(DEVICE)           # (B, 2, 2, N, 2)
    pad_dir  = batch['pad_dir'].to(DEVICE)         # (B, 2, 2, N)

    image_A = patches[:, 0]
    image_B = patches[:, 1]
    uvd_A   = uvd[:, 0, :, :3]                     # drop is_obj column if present
    uvd_B   = uvd[:, 1, :, :3]
    pad_A   = pad[:, 0]
    pad_B   = pad[:, 1]
    pose_AB = pose_hat[:, 0, 1]                    # 6dof
    uv_B_naive = uv_hat[:, 0, 1]                   # A-pts naive proj to B  (= input)
    uv_B_gt    = uv_gt [:, 0, 1]                   # A-pts GT proj to B    (= target)
    valid      = ~pad_dir[:, 0, 1]                 # which A-pts are visible in B

    # vfp: not in the pair output (yet) — derive a per-batch placeholder from
    # the patches' implied focal. For the overfit run, all crops share the
    # same scene + crop_size sampling distribution, so just pass img_size as
    # a constant placeholder; pose_emb will absorb it as a bias.
    vfp = torch.full((image_A.size(0),), float(64), device=DEVICE)

    target_residual = uv_B_gt - uv_B_naive
    with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
        params = model.forward_pair(
            image_A, image_B, uvd_A, uvd_B, uv_B_naive,
            pose_AB, vfp, pad_A=pad_A, pad_B=pad_B, query_pad=pad_A,
        )
        loss = gaussian2d_nll(params[valid], target_residual[valid])
    with torch.no_grad():
        if valid.any():
            mse = (params[valid][..., :2] - target_residual[valid]).norm(dim=-1).mean().item()
        else:
            mse = float('nan')
    return loss, mse


def epoch_loop(model, loader, optimizer, scaler, train):
    model.train(train)
    total_nll = total_mse = 0.0; n = 0
    for batch in loader:
        loss, mse = step(model, batch)
        if train:
            optimizer.zero_grad(set_to_none=True)
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            scaler.step(optimizer); scaler.update()
        total_nll += loss.item(); total_mse += mse; n += 1
    return total_nll / max(n,1), total_mse / max(n,1)


def main(cfg=None, clearml=False, why='', clearml_project='e2e_calib/cross-frame',
         queue=None):
    c = cfg if cfg is not None else CFG
    cml_task = None
    if clearml:
        from scripts.util.clearml_context import init_with_context
        cml_task = init_with_context(
            project=clearml_project, name=c['name'], cfg=c, why=why,
            baseline={'name': 'ps_v9_lazy (calib)', 'metric': 'val_nll', 'value': 1.8141})
        if queue:
            cml_task.set_packages([])
            cml_task.execute_remotely(queue_name=queue, exit_process=True)

    exp_dir = Path("experiments") / c['name']
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

    overfit = (c.get('n_overfit') or 0) > 0
    ds_kw = dict(
        cameras           = c['cameras'],
        img_size          = c['img_size'],
        max_points        = c['max_points'],
        baseline_range    = (c['baseline_min'], c['baseline_max']),
        sigma_ypr         = c['sigma_ypr'],
        sigma_t           = c['sigma_t'],
        crop_range        = (c['crop_min'], c['crop_max']),
        virtual_epoch_len = c['n_overfit'] if overfit else c.get('virtual_epoch', 10000),
        n_frames          = 2,
        use_stacked       = True,
    )
    if c.get('scenes_root'):
        ds_kw['scenes_root'] = c['scenes_root']
        ds_kw['train_frac']  = c.get('train_frac', 0.8)
        train_ds = PandaSetCrossFrameDataset(split='train', **ds_kw)
        val_kw = dict(ds_kw); val_kw['virtual_epoch_len'] = max(256, ds_kw['virtual_epoch_len'] // 20)
        val_ds   = PandaSetCrossFrameDataset(split='val',   **val_kw)
        log(f"dataset ready  scenes_root={c['scenes_root']}  "
            f"baseline={c['baseline_min']}..{c['baseline_max']}  "
            f"train.virt_ep={ds_kw['virtual_epoch_len']}  val.virt_ep={val_kw['virtual_epoch_len']}")
    else:
        ds_kw['scene_root'] = c['scene']
        ds = PandaSetCrossFrameDataset(**ds_kw)
        log(f"dataset ready  scene={c['scene']}  baseline={c['baseline_min']}..{c['baseline_max']}")
        if overfit:
            log(f"caching {c['n_overfit']} fixed overfit samples ...")
            fixed = [ds[i] for i in range(c['n_overfit'])]
            log(f"cached {len(fixed)} samples")
            class Fixed(torch.utils.data.Dataset):
                def __len__(self): return len(fixed)
                def __getitem__(self, i): return fixed[i]
            train_ds = val_ds = Fixed()
        else:
            train_ds = ds
            val_kw = dict(ds_kw); val_kw['virtual_epoch_len'] = max(256, ds_kw['virtual_epoch_len'] // 20)
            val_ds = PandaSetCrossFrameDataset(**val_kw)

    nw = 0 if overfit else c.get('num_workers', 8)
    train_loader = DataLoader(train_ds, batch_size=c['batch_size'], shuffle=True,
                              collate_fn=collate_pair, num_workers=nw,
                              persistent_workers=(nw > 0))
    val_loader   = DataLoader(val_ds, batch_size=c['batch_size'], shuffle=False,
                              collate_fn=collate_pair, num_workers=nw,
                              persistent_workers=(nw > 0))

    model = CalibNetDepthPair(
        img_size=c['img_size'], in_channels=c['in_channels'],
        n_layers=c['n_layers'], use_convnext=c.get('use_convnext', False),
        use_frustum=c.get('use_frustum', False),
        use_lidar_kv=c.get('use_lidar_kv', False),
        use_pose_emb=c.get('use_pose_emb', False),
    ).to(DEVICE)
    log(f"params: {sum(p.numel() for p in model.parameters())/1e6:.2f}M")

    optimizer = torch.optim.AdamW(model.parameters(), lr=c['lr'], weight_decay=1e-3)
    epochs = c['epochs']
    lr_min_r = c['lr_min'] / c['lr']
    def lr_lambda(e):
        if e < 5: return (e+1)/5
        t = (e-5)/max(1, epochs-5)
        return lr_min_r + (1-lr_min_r)*0.5*(1+math.cos(math.pi*t))
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)
    scaler = torch.GradScaler(device="cuda")
    best_val = float('inf')
    history = {'ep': [], 'tr_nll': [], 'va_nll': [], 'tr_mse': [], 'va_mse': []}
    t0 = time.time()
    for epoch in range(1, epochs+1):
        tr_nll, tr_mse = epoch_loop(model, train_loader, optimizer, scaler, True)
        with torch.no_grad():
            va_nll, va_mse = epoch_loop(model, val_loader, optimizer, scaler, False)
        scheduler.step()
        history['ep'].append(epoch)
        history['tr_nll'].append(tr_nll); history['va_nll'].append(va_nll)
        history['tr_mse'].append(tr_mse); history['va_mse'].append(va_mse)
        log(f"[{epoch:3d}/{epochs}]  train nll={tr_nll:+.3f} mse={tr_mse:.3f}px  "
            f"val nll={va_nll:+.3f} mse={va_mse:.3f}px  "
            f"lr={scheduler.get_last_lr()[0]:.2e}  tot={(time.time()-t0)/60:.1f}min")
        if cml_task is not None:
            lg = cml_task.get_logger()
            lg.report_scalar('nll', 'train', tr_nll, epoch); lg.report_scalar('nll', 'val', va_nll, epoch)
            lg.report_scalar('mse', 'train', tr_mse, epoch); lg.report_scalar('mse', 'val', va_mse, epoch)
        if va_nll < best_val:
            best_val = va_nll
            torch.save(model.state_dict(), exp_dir / "best_model.pt")
            log(f"  ↳ saved (val_nll={best_val:.4f})")

    log(f"Best val NLL: {best_val:.4f}  |  time: {(time.time()-t0)/60:.1f}min")

    vis_dir = exp_dir / "vis"; vis_dir.mkdir(exist_ok=True)
    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    axes[0].plot(history['ep'], history['tr_nll'], label='train'); axes[0].plot(history['ep'], history['va_nll'], label='val')
    axes[0].set_title('NLL'); axes[0].legend(); axes[0].grid(alpha=0.3)
    axes[1].plot(history['ep'], history['tr_mse'], label='train'); axes[1].plot(history['ep'], history['va_mse'], label='val')
    axes[1].set_title('MSE (px)'); axes[1].legend(); axes[1].grid(alpha=0.3)
    plt.tight_layout(); plt.savefig(vis_dir / "curves.png", dpi=100); plt.close(fig)

    if cml_task is not None:
        from scripts.util.clearml_context import write_retrospective
        write_retrospective(cml_task, dict(
            best_val_nll=best_val,
            final_val_nll=history['va_nll'][-1] if history['va_nll'] else None,
            time_min=(time.time() - t0) / 60,
            epochs=len(history['ep']),
        ), baseline={'name': 'ps_v9_lazy (calib)', 'value': 1.8141})


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument('--clearml', action='store_true')
    ap.add_argument('--clearml-project', default='e2e_calib/cross-frame')
    ap.add_argument('--queue', default=None,
                    help='If set with --clearml: submit to this queue and exit (remote run).')
    ap.add_argument('--why', default='')
    ap.add_argument('--name', default=None)
    ap.add_argument('--scene', default=None,
                    help='single scene root (mutually exclusive with --scenes-root)')
    ap.add_argument('--scenes-root', default=None,
                    help='root dir holding many scenes (e.g. /mnt/nvme6t/pandaset)')
    ap.add_argument('--baseline-min', type=int, default=None)
    ap.add_argument('--baseline-max', type=int, default=None)
    ap.add_argument('--virtual-epoch', type=int, default=None,
                    help='samples per epoch when not in n_overfit mode')
    ap.add_argument('--n-overfit', type=int, default=None,
                    help='if 0, run full (multi-scene) mode')
    ap.add_argument('--epochs', type=int, default=None)
    ap.add_argument('--batch-size', type=int, default=None)
    args = ap.parse_args()
    cfg = dict(CFG)
    for k_arg, k_cfg in [('name', 'name'), ('scene', 'scene'),
                         ('baseline_min', 'baseline_min'), ('baseline_max', 'baseline_max'),
                         ('virtual_epoch', 'virtual_epoch'), ('n_overfit', 'n_overfit'),
                         ('epochs', 'epochs'), ('batch_size', 'batch_size')]:
        v = getattr(args, k_arg)
        if v is not None:
            cfg[k_cfg] = v
    if args.scenes_root:
        cfg['scenes_root'] = args.scenes_root
        cfg.pop('scene', None)
    main(cfg=cfg, clearml=args.clearml, clearml_project=args.clearml_project, why=args.why,
         queue=args.queue)
