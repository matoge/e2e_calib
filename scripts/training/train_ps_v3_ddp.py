"""Train CalibNetDepth on PandaSet — HF Accelerate DDP variant of train_ps_v3.py.

Why a separate file:
  The single-GPU train_ps_v3.py is the verified baseline (ps_v506 reached
  val_nll ~2.5). Rather than branching it with `if ddp:` everywhere and risking
  regression, this file is the DDP twin: same config shape, same epoch_loop
  numerics, Accelerator-wrapped model / optimizer / loaders, rank-0 gating on
  ClearML + logging + checkpointing + visualization.

Launch:
  accelerate launch --num_processes=4 --mixed_precision=fp16 \
      scripts/training/train_ps_v3_ddp.py --name ps_ddp_v1 --batch-size 128 ...

Notes:
  - batch-size is PER-PROCESS (global = bs * num_processes).
  - num_workers is per-process; keep reasonable (90/4 ≈ 22 per GPU on dgx2).
  - mixed_precision is driven by accelerate launch flag, not runtime detect,
    so V100 needs --mixed_precision=fp16 explicitly.
"""
import sys, pathlib; sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[2]))
import math, time, torch, torch.nn as nn
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from pathlib import Path
from datetime import datetime
from datasets.pandaset_full import PandaSetCalibDatasetFull, collate_full
from models.model_depth import CalibNetDepth
from models.model_cov import gaussian2d_nll
from torch.utils.data import DataLoader, Subset

from accelerate import Accelerator
from accelerate.utils import set_seed

torch.set_float32_matmul_precision("high")

CFG = dict(
    name          = "ps_v3_ddp",
    n_layers      = 2,
    img_size      = 64,
    in_channels   = 3,
    use_convnext  = False,
    use_frustum   = True,
    epochs        = 100,
    batch_size    = 64,       # per-process
    lr            = 1e-3,
    lr_min        = 1e-6,
    val_fraction  = 0.1,
    split_seed    = 42,
    min_crop_px   = 128,
    max_crop_px   = 512,
    # --cache は必須。 host ごとに path が違うので default は設けない
    # (旧: /mnt/nvme6t/... は dgx1/2 で存在せず、別 dataset に silent
    #  fall-through する事故を起こしていた。 2026-05-04 撤去)。
    cache         = None,
)


def epoch_loop(model, loader, optimizer, accel: Accelerator, train: bool):
    """Same numerics as train_ps_v3.epoch_loop but uses accel.backward.

    GradScaler / autocast are owned by Accelerator. We still use torch.autocast
    for the forward pass — Accelerate's mixed_precision plugs into this.
    """
    model.train(train)
    total_nll, total_mse, n = 0.0, 0.0, 0
    obj_nll_s, obj_n = 0.0, 0
    bg_nll_s,  bg_n  = 0.0, 0
    obj_errs, bg_errs = [], []
    _t_start = time.time()
    _last_log_step = 0
    for imgs, true_uvd, dist_uvd, pad_mask, vfp, dist_uvd_full, pad_full in loader:
        # accel.prepare already puts tensors in device via DataLoader wrap, but
        # the collate returns uint8 images — convert to float on-device here.
        imgs     = imgs.float().div_(255.0)
        gt       = true_uvd[..., :2] - dist_uvd[..., :2]
        # Accelerate manages autocast via its mixed_precision plugin; we just
        # call the model normally.
        # NOTE: dist_uvd_full / pad_full are REQUIRED when frustum_enc is
        # enabled (model raises otherwise). Harmless extra kwargs when off.
        params = model(imgs, dist_uvd[..., :3], key_padding_mask=pad_mask, vfp=vfp,
                       distorted_uvd_full=dist_uvd_full[..., :3], pad_full=pad_full)
        valid  = ~pad_mask
        loss   = gaussian2d_nll(params[valid], gt[valid])
        if train:
            optimizer.zero_grad(set_to_none=True)
            accel.backward(loss)
            # clip_grad_norm_ must go through accel so it unscales first
            accel.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
        with torch.no_grad():
            err_all = (params[valid][..., :2].float() - gt[valid]).norm(dim=-1)
            mse    = err_all.mean().item()
            is_obj = valid & (dist_uvd[..., 3] > 0.5)
            is_bg  = valid & (dist_uvd[..., 3] < 0.5)
            if is_obj.any():
                obj_nll_s += gaussian2d_nll(params[is_obj], gt[is_obj]).item(); obj_n += 1
                obj_errs.append((params[is_obj][..., :2].float() - gt[is_obj]).norm(dim=-1).detach())
            if is_bg.any():
                bg_nll_s  += gaussian2d_nll(params[is_bg],  gt[is_bg]).item();  bg_n  += 1
                bg_errs.append((params[is_bg][..., :2].float() - gt[is_bg]).norm(dim=-1).detach())
        total_nll += loss.item(); total_mse += mse; n += 1
        if train and accel.is_main_process and (n - _last_log_step >= 100):
            _dt = time.time() - _t_start
            # sps is PER-PROCESS here; multiply by num_processes for global throughput
            sps_per = n * imgs.shape[0] / _dt if _dt > 0 else 0
            sps_glob = sps_per * accel.num_processes
            print(f"  step {n}  loss={loss.item():+.3f}  "
                  f"sps/rank={sps_per:.0f}  sps(global)={sps_glob:.0f}", flush=True)
            _last_log_step = n
    obj_nll = obj_nll_s / max(obj_n, 1)
    bg_nll  = bg_nll_s  / max(bg_n,  1)
    import numpy as _np
    def _stats(errs):
        if not errs:
            return float('nan'), float('nan'), float('nan')
        e = torch.cat(errs).cpu().numpy()
        return float(e.mean()), float(_np.median(e)), float(_np.percentile(e, 95))
    obj_mse, obj_med, obj_p95 = _stats(obj_errs)
    bg_mse,  bg_med,  bg_p95  = _stats(bg_errs)
    # epoch-level sps: total samples pushed through forward / wall time (per-rank)
    _dt_total = max(time.time() - _t_start, 1e-6)
    _bs = 0
    try:
        # imgs still in scope from last iter; fall back to 0 if empty loader
        _bs = int(imgs.shape[0])
    except Exception:
        _bs = 0
    sps_rank_epoch = (n * _bs) / _dt_total if n > 0 else 0.0
    return (total_nll / max(n,1), total_mse / max(n,1),
            obj_nll, bg_nll, obj_mse, bg_mse,
            obj_med, obj_p95, bg_med, bg_p95,
            sps_rank_epoch)


def main(cfg=None):
    c = cfg if cfg is not None else CFG
    # One Accelerator per process; mixed_precision comes from accelerate launch.
    accel = Accelerator()
    set_seed(c.get('split_seed', 42) + accel.process_index)

    exp_dir = Path("experiments") / c["name"]
    if accel.is_main_process:
        exp_dir.mkdir(parents=True, exist_ok=True)
        log_path = exp_dir / "train.log"
        log_path.write_text("")
        (exp_dir / "config.py").write_text(
            "CFG = dict(\n" +
            "".join(f"    {k:<13}= {v!r},\n" for k, v in c.items()) + ")\n")
    accel.wait_for_everyone()

    def log(msg):
        if not accel.is_main_process:
            return
        ts = datetime.now().strftime("%H:%M:%S")
        line = f"{ts}  {msg}"
        print(line, flush=True)
        with open(exp_dir / "train.log", "a") as f: f.write(line + "\n")

    log(f"accelerate: num_processes={accel.num_processes} mixed_precision={accel.mixed_precision} "
        f"device={accel.device}")

    cache = c.get('cache')
    if not cache:
        raise SystemExit(
            "[train_ps_v3_ddp] --cache is required. "
            "Host-specific silent fallback was removed (2026-05-04): "
            "explicitly pass --cache /path/to/v3_cache."
        )
    nw = c.get('num_workers', 16)
    pf = c.get('prefetch_factor', 4)
    kw = dict(num_workers=nw, pin_memory=True,
              persistent_workers=(nw > 0),
              prefetch_factor=pf if nw > 0 else None,
              collate_fn=collate_full)
    val_nw = min(4, nw)
    val_kw = dict(num_workers=val_nw, pin_memory=True,
                  persistent_workers=(val_nw > 0),
                  prefetch_factor=pf if val_nw > 0 else None,
                  collate_fn=collate_full)
    import random as _r
    ds_kw = dict(max_offset_m=c.get('max_offset_m', 0.20),
                  max_rot_deg=c.get('max_rot_deg', 0.5),
                  min_crop_px=c.get('min_crop_px', 128),
                  max_crop_px=c.get('max_crop_px', 512),
                  frame_stride=c.get('frame_stride', 1),
                  grid_n=c.get('grid_n', 16),
                  oversample=c.get('oversample', 12))
    log(f"cache={cache}  perturbation=±{ds_kw['max_rot_deg']} deg / ±{ds_kw['max_offset_m']} m  "
        f"crop_px=[{ds_kw['min_crop_px']}, {ds_kw['max_crop_px']}] (full → {c['img_size']})")
    tr_full = PandaSetCalibDatasetFull(cache, split='train', **ds_kw)
    va_full = PandaSetCalibDatasetFull(cache, split='val',   **ds_kw)
    log(f"train cache: {len(tr_full)} inst   val cache: {len(va_full)} inst")

    from torch.utils.data import ConcatDataset
    full_ds = ConcatDataset([tr_full, va_full])
    idxs = list(range(len(full_ds)))
    _r.Random(c["split_seed"]).shuffle(idxs)
    n_val_obj = int(len(idxs) * c["val_fraction"])
    val_idxs, train_idxs = idxs[:n_val_obj], idxs[n_val_obj:]
    train_ds = Subset(full_ds, train_idxs)
    val_ds   = Subset(full_ds, val_idxs)
    log(f"object-level split: train={len(train_ds)} val={len(val_ds)} (seed={c['split_seed']})")

    train_size = c.get('train_size', None)
    val_size   = c.get('val_size',   None)
    from torch.utils.data import RandomSampler
    # NOTE: we do NOT install DistributedSampler here — accel.prepare does it.
    if train_size and train_size < len(train_ds):
        tr_sampler = RandomSampler(train_ds, replacement=False, num_samples=train_size)
        log(f"  train subsample: {train_size}/{len(train_ds)} per epoch")
        train_loader = DataLoader(train_ds, batch_size=c["batch_size"], sampler=tr_sampler, **kw)
    else:
        train_loader = DataLoader(train_ds, batch_size=c["batch_size"], shuffle=True,  **kw)
    if val_size and val_size < len(val_ds):
        val_subset = Subset(val_ds, list(range(val_size)))
        log(f"  val subsample: {val_size}/{len(val_ds)} (deterministic first-N)")
        val_loader = DataLoader(val_subset, batch_size=c["batch_size"], shuffle=False, **val_kw)
    else:
        val_loader   = DataLoader(val_ds,   batch_size=c["batch_size"], shuffle=False, **val_kw)

    model = CalibNetDepth(img_size=c["img_size"], in_channels=c["in_channels"],
                          n_layers=c["n_layers"], self_first=c.get("self_first", False),
                          use_convnext=c.get("use_convnext", False),
                          use_frustum=c.get("use_frustum", True),
                          deform_mode=c.get("deform_mode", "none"))
    # torch.compile BEFORE accel.prepare so DDP wraps the compiled graph.
    # mode='reduce-overhead' is the sweet spot: CUDA graph capture for static
    # shapes, minimal compile time (~30-60s vs max-autotune's 5+ min).
    if c.get('compile', False):
        log(f"torch.compile(mode='reduce-overhead') — first 1-2 steps will be slow (graph capture)")
        model = torch.compile(model, mode='reduce-overhead')
    optimizer = torch.optim.AdamW(model.parameters(), lr=c["lr"], weight_decay=1e-3)

    # Single prepare() pass — Accelerate wraps model w/ DDP, sampler w/ Distributed
    model, optimizer, train_loader, val_loader = accel.prepare(
        model, optimizer, train_loader, val_loader)

    log(f"params: {sum(p.numel() for p in model.parameters())/1e6:.2f}M  "
        f"(world_size={accel.num_processes}, bs/rank={c['batch_size']}, "
        f"global_bs={c['batch_size']*accel.num_processes})")

    epochs    = c["epochs"]
    lr_min_r  = c["lr_min"] / c["lr"]
    def lr_lambda(e):
        if e < 5: return (e+1)/5
        t = (e-5)/max(1,epochs-5)
        return lr_min_r + (1-lr_min_r)*0.5*(1+math.cos(math.pi*t))
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)
    best_val  = float("inf")
    ckpt      = exp_dir / "best_model.pt"
    t0        = time.time()
    history = {'ep': [], 'tr_nll': [], 'va_nll': [], 'tr_mse': [], 'va_mse': []}

    # ClearML (rank-0 only)
    cml_logger = None
    if accel.is_main_process:
        try:
            from clearml import Task as _ClearMLTask
            cml_logger = _ClearMLTask.current_task().get_logger() if _ClearMLTask.current_task() else None
        except Exception:
            cml_logger = None

    # Pre-training cache sanity vis (rank-0 only, then rendezvous).
    if accel.is_main_process:
        try:
            from scripts.util.midtrain_vis import vis_pretrain_run
            vis_pretrain_run(exp_dir, cache, cml_logger=cml_logger, n=10, log=log)
        except Exception as _e:
            log(f"vis_pretrain skipped: {_e}")
    accel.wait_for_everyone()

    for epoch in range(1, epochs+1):
        # DistributedSampler needs set_epoch for proper shuffling across epochs;
        # Accelerate exposes the underlying sampler via train_loader.sampler.
        if hasattr(train_loader, 'set_epoch'):
            train_loader.set_epoch(epoch)
        (tr_nll, tr_mse, tr_obj, tr_bg, tr_obj_mse, tr_bg_mse,
         tr_obj_med, tr_obj_p95, tr_bg_med, tr_bg_p95, tr_sps_rank) = epoch_loop(
            model, train_loader, optimizer, accel, True)
        with torch.no_grad():
            (va_nll, va_mse, va_obj, va_bg, va_obj_mse, va_bg_mse,
             va_obj_med, va_obj_p95, va_bg_med, va_bg_p95, va_sps_rank) = epoch_loop(
                model, val_loader, optimizer, accel, False)
        scheduler.step()
        # tr/va numbers are PER-RANK means; for reporting we average across ranks.
        # sps is also per-rank; global sps = mean-across-ranks × num_processes
        # (each rank processes its own shard at ~rank_sps; total throughput adds).
        stats = torch.tensor([tr_nll, va_nll, tr_mse, va_mse,
                               tr_obj, tr_bg, va_obj, va_bg,
                               tr_sps_rank, va_sps_rank],
                              device=accel.device)
        stats_all = accel.gather(stats.unsqueeze(0)).mean(dim=0).tolist()
        (tr_nll, va_nll, tr_mse, va_mse,
         tr_obj, tr_bg, va_obj, va_bg,
         tr_sps_rank_mean, va_sps_rank_mean) = stats_all
        tr_sps_global = tr_sps_rank_mean * accel.num_processes
        va_sps_global = va_sps_rank_mean * accel.num_processes

        if accel.is_main_process:
            history['ep'].append(epoch)
            history['tr_nll'].append(tr_nll); history['va_nll'].append(va_nll)
            history['tr_mse'].append(tr_mse); history['va_mse'].append(va_mse)
            log(f"[{epoch:3d}/{epochs}]  "
                f"train nll={tr_nll:+.3f}(obj={tr_obj:+.3f} bg={tr_bg:+.3f}) mse={tr_mse:.2f}  "
                f"val nll={va_nll:+.3f}(obj={va_obj:+.3f} bg={va_bg:+.3f}) mse={va_mse:.2f}  "
                f"sps(global)={tr_sps_global:.0f}  "
                f"lr={scheduler.get_last_lr()[0]:.2e}  tot={(time.time()-t0)/60:.1f}min")
            if va_nll < best_val:
                best_val = va_nll
                # unwrap before state_dict so keys don't carry 'module.' prefix
                accel.save(accel.unwrap_model(model).state_dict(), ckpt)
                log(f"  ↳ saved (val_nll={best_val:.4f})")
            if cml_logger is not None:
                try:
                    rs = cml_logger.report_scalar
                    rs('nll', 'train', iteration=epoch, value=tr_nll)
                    rs('nll', 'val',   iteration=epoch, value=va_nll)
                    rs('mse_px', 'train', iteration=epoch, value=tr_mse)
                    rs('mse_px', 'val',   iteration=epoch, value=va_mse)
                    rs('lr', 'lr', iteration=epoch, value=scheduler.get_last_lr()[0])
                    rs('best', 'best_val_nll', iteration=epoch, value=best_val)
                    # throughput: per-rank and global — lets us sanity-check
                    # DDP scaling (global should be ~num_processes × rank) and
                    # compare runs with different world sizes in the same panel
                    rs('sps', 'train_rank',   iteration=epoch, value=tr_sps_rank_mean)
                    rs('sps', 'train_global', iteration=epoch, value=tr_sps_global)
                    rs('sps', 'val_rank',     iteration=epoch, value=va_sps_rank_mean)
                    rs('sps', 'val_global',   iteration=epoch, value=va_sps_global)
                except Exception:
                    pass
            # Per-10-epoch debug images (rank-0 only; needs the unwrapped model
            # so forward signature matches the dataset-level call in the util).
            if epoch % 10 == 0 or epoch == epochs:
                try:
                    from scripts.util.midtrain_vis import midtrain_vis
                    midtrain_vis(
                        accel.unwrap_model(model), exp_dir, cache, epoch,
                        img_size=c["img_size"],
                        min_crop_px=c.get("min_crop_px", 128),
                        max_crop_px=c.get("max_crop_px", 384),
                        cml_logger=cml_logger, device=accel.device,
                        amp_dtype=torch.float16, n=10, log=log)
                except Exception as _e:
                    log(f"vis_ep{epoch:03d} skipped: {_e}")
        accel.wait_for_everyone()

    log(f"Best val NLL: {best_val:.4f}  |  time: {(time.time()-t0)/60:.1f}min")

    if accel.is_main_process:
        vis_dir = exp_dir / "vis"; vis_dir.mkdir(exist_ok=True)
        fig, axes = plt.subplots(1, 2, figsize=(10, 4))
        axes[0].plot(history['ep'], history['tr_nll'], label='train'); axes[0].plot(history['ep'], history['va_nll'], label='val')
        axes[0].set_title('NLL'); axes[0].set_xlabel('epoch'); axes[0].legend(); axes[0].grid(alpha=0.3)
        axes[1].plot(history['ep'], history['tr_mse'], label='train'); axes[1].plot(history['ep'], history['va_mse'], label='val')
        axes[1].set_title('MSE (px)'); axes[1].set_xlabel('epoch'); axes[1].legend(); axes[1].grid(alpha=0.3)
        plt.tight_layout(); plt.savefig(vis_dir / "curves.png", dpi=100); plt.close(fig)
        log("curves saved")


if __name__ == "__main__":
    import multiprocessing as _mp
    try:
        _mp.set_start_method('forkserver', force=True)
    except RuntimeError:
        pass
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument('--name')
    ap.add_argument('--cache')
    ap.add_argument('--rot-deg', type=float, default=None)
    ap.add_argument('--t-m',     type=float, default=None)
    ap.add_argument('--epochs',  type=int,   default=None)
    ap.add_argument('--lr',      type=float, default=None)
    ap.add_argument('--lr-min',  type=float, default=None)
    ap.add_argument('--frame-stride', type=int, default=1)
    ap.add_argument('--grid-n', type=int, default=None)
    ap.add_argument('--workers', type=int,   default=None)
    ap.add_argument('--min-crop-px', type=int, default=None)
    ap.add_argument('--max-crop-px', type=int, default=None)
    ap.add_argument('--oversample', type=int, default=None)
    ap.add_argument('--batch-size', type=int, default=None,
                    help='PER-PROCESS batch size; global = bs * num_processes')
    ap.add_argument('--prefetch-factor', type=int, default=None)
    ap.add_argument('--n-layers', type=int, default=None)
    ap.add_argument('--img-size', type=int, default=None)
    ap.add_argument('--convnext', action='store_true')
    ap.add_argument('--compile', action='store_true',
                    help='wrap model with torch.compile(mode="reduce-overhead") '
                         'BEFORE accel.prepare so DDP wraps the compiled graph. '
                         'First 1-2 steps slow (CUDA graph capture).')
    ap.add_argument('--deform-mode', default=None)
    ap.add_argument('--train-size', type=int, default=None)
    ap.add_argument('--val-size', type=int, default=None)
    ap.add_argument('--no-frustum', action='store_true',
                    help='disable FrustumLocalEncoder (ablation vs CFG default)')
    ap.add_argument('--clearml', action='store_true')
    ap.add_argument('--why',     default='')
    args = ap.parse_args()
    cfg = dict(CFG)
    if args.name:    cfg['name'] = args.name
    if args.cache:   cfg['cache'] = args.cache
    if args.rot_deg is not None: cfg['max_rot_deg']  = args.rot_deg
    if args.t_m     is not None: cfg['max_offset_m'] = args.t_m
    if args.epochs  is not None: cfg['epochs'] = args.epochs
    if args.lr      is not None: cfg['lr']     = args.lr
    if args.lr_min  is not None: cfg['lr_min'] = args.lr_min
    if args.frame_stride and args.frame_stride > 1: cfg['frame_stride'] = args.frame_stride
    if args.oversample is not None: cfg['oversample'] = args.oversample
    if args.grid_n is not None: cfg['grid_n'] = args.grid_n
    if args.workers is not None: cfg['num_workers'] = args.workers
    if args.min_crop_px is not None: cfg['min_crop_px'] = args.min_crop_px
    if args.max_crop_px is not None: cfg['max_crop_px'] = args.max_crop_px
    if args.batch_size is not None: cfg['batch_size'] = args.batch_size
    if args.prefetch_factor is not None: cfg['prefetch_factor'] = args.prefetch_factor
    if args.n_layers is not None: cfg['n_layers'] = args.n_layers
    if args.img_size is not None: cfg['img_size'] = args.img_size
    if args.convnext: cfg['use_convnext'] = True
    if args.compile:  cfg['compile'] = True
    if args.deform_mode is not None: cfg['deform_mode'] = args.deform_mode
    if args.train_size is not None: cfg['train_size'] = args.train_size
    if args.val_size   is not None: cfg['val_size']   = args.val_size
    if args.no_frustum: cfg['use_frustum'] = False

    # ClearML init happens before main so the task is available to cml_logger
    if args.clearml:
        # NOTE: only rank-0 should register the task; accelerate launches all
        # processes simultaneously, so each would race for the same task name.
        # Accelerate sets LOCAL_RANK/RANK env vars before entering user code.
        import os as _os
        if int(_os.environ.get('RANK', '0')) == 0:
            from scripts.util.clearml_context import init_with_context
            init_with_context(
                project='e2e_calib/calib', name=cfg['name'], cfg=cfg,
                why=args.why,
                baseline={'name': 'ps_v506_fp16', 'metric': 'val_nll', 'value': 2.52})
    main(cfg=cfg)
