"""Train CalibNetDepth on PandaSet (jitter-fix verification, 500 epochs)."""
import sys, pathlib; sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[2]))
import os, math, time, torch, torch.nn as nn
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from pathlib import Path
from datetime import datetime
from datasets.pandaset_full import PandaSetCalibDatasetFull, collate_full
from models.model_depth import CalibNetDepth
from models.model_cov import gaussian2d_nll
from torch.utils.data import DataLoader, Subset

torch.set_float32_matmul_precision("high")
DEVICE = torch.device("cuda")

# Auto-select AMP dtype: BF16 needs compute-cap >= 8.0 (Ampere+) for native
# Tensor Core support. On Volta/Turing (V100/T4/RTX20xx) BF16 autocast falls
# through to FP32 Tensor Cores - losing 8-10x throughput vs FP16 TC.
# Blackwell/Hopper/Ada/Ampere (sm80+) -> BF16 (wider exponent, no loss scaling).
if torch.cuda.is_available():
    _cc_major = torch.cuda.get_device_capability(0)[0]
    _AMP_DTYPE = torch.bfloat16 if _cc_major >= 8 else torch.float16
else:
    _AMP_DTYPE = torch.bfloat16
_NEED_SCALER = (_AMP_DTYPE == torch.float16)

# Default = "pixel-only" CalibNet (per-pt Δuv + Σ head only).
# Rationale (2026-05-13): the clspose / fxfy heads added in f3d043b and later
# experiments (vcam pose head, Δfx aug, full-cov CLS) were silently inherited
# by subsequent runs even when not needed; the σ head correlates with depth
# not texture, so BA driven by clspose ended up dominated by near-range road
# pts and skewing calibration estimates. Strip back to 4-layer ConvNeXt +
# deform_sl + frustum, no CLS head, no intrinsic aug. Opt in to clspose / fxfy
# only when explicitly studying frame-pose regression.
CFG = dict(
    name          = "ps_v3_full",
    n_layers      = 4,
    img_size      = 128,
    in_channels   = 3,
    use_convnext  = True,
    use_frustum   = True,
    epochs        = 100,
    batch_size    = 128,
    lr            = 3e-4,
    lr_min        = 1e-7,
    val_fraction  = 0.1,
    split_seed    = 42,
    min_crop_px   = 128,
    max_crop_px   = 384,
    deform_mode   = "sl",
    oversample    = 1,
    num_workers   = 12,
    max_rot_deg   = 1.5,
    max_offset_m  = 0.6,
    val_size      = 8000,
    # Intentionally OFF:
    #   use_frame_pose = False  → no CLS frame-pose head
    #   max_fx_pct     = 0.0    → no intrinsic perturbation augmentation
    #   max_fy_pct     = 0.0
    #   pose_frame     = 'orig' → no virtual-camera label conversion
    # --cache は必須。 host ごとに path が違うので default は設けない
    # (旧: /mnt/nvme6t/... は dgx1/2 で存在せず、別 dataset に silent
    #  fall-through する事故を起こしていた。 2026-05-04 撤去)。
    cache         = None,
)


def epoch_loop(model, loader, optimizer, scaler, train, frame_pose_weight=0.5):
    model.train(train)
    total_nll, total_mse, n = 0.0, 0.0, 0
    pt_nll_s, fr_nll_s, fr_n = 0.0, 0.0, 0  # split per-pt vs frame-pose NLL
    obj_nll_s, obj_n = 0.0, 0
    bg_nll_s,  bg_n  = 0.0, 0
    obj_errs, bg_errs = [], []  # collect per-point L2 to compute median/p95
    import time as _time
    _t_start = _time.time()
    _last_log_step = 0
    for batch in loader:
        # collate_full returns 8-tuple post-2026-05-11 (added pert_6vec); fall back
        # to 7-tuple unpack for backward-compat caches.
        if len(batch) == 8:
            imgs, true_uvd, dist_uvd, pad_mask, vfp, bucket_uvd, bucket_valid, pert_6vec = batch
        else:
            imgs, true_uvd, dist_uvd, pad_mask, vfp, bucket_uvd, bucket_valid = batch
            pert_6vec = None
        # dataset returns uint8 to cut IPC 4x; convert to float on GPU
        imgs         = imgs.to(DEVICE, non_blocking=True).float().div_(255.0)
        true_uvd     = true_uvd.to(DEVICE, non_blocking=True)
        dist_uvd     = dist_uvd.to(DEVICE, non_blocking=True)
        pad_mask     = pad_mask.to(DEVICE, non_blocking=True)
        vfp          = vfp.to(DEVICE, non_blocking=True)
        bucket_uvd   = bucket_uvd.to(DEVICE, non_blocking=True)
        bucket_valid = bucket_valid.to(DEVICE, non_blocking=True)
        if pert_6vec is not None:
            pert_6vec = pert_6vec.to(DEVICE, non_blocking=True)
        gt           = true_uvd[..., :2] - dist_uvd[..., :2]
        with torch.autocast(device_type="cuda", dtype=_AMP_DTYPE):
            params = model(imgs, dist_uvd[..., :3], key_padding_mask=pad_mask, vfp=vfp,
                            bucket_uvd=bucket_uvd, bucket_valid=bucket_valid)
            # When use_frame_pose=True, model returns (per_pt_pred, (μ, log_σ, L)).
            frame_L = None
            if isinstance(params, tuple):
                params, head_out = params
                if len(head_out) == 3:
                    frame_mu, frame_logsig, frame_L = head_out
                else:
                    frame_mu, frame_logsig = head_out
            else:
                frame_mu = frame_logsig = None
            valid  = ~pad_mask
            loss_pt = gaussian2d_nll(params[valid], gt[valid])
            loss = loss_pt
            # Frame-pose NLL: full multivariate Gaussian via Cholesky L (full Σ = L Lᵀ).
            # Falls back to diag if frame_L missing (legacy). Translation in m, ypr in deg.
            loss_fr = None
            if frame_mu is not None and pert_6vec is not None:
                resid = pert_6vec[..., :frame_mu.shape[-1]] - frame_mu  # (B, n_dof)
                if frame_L is not None:
                    # NLL = 0.5 * x^T Σ^{-1} x + log|L| = 0.5 ||L^{-1} x||² + Σ log diag(L)
                    z = torch.linalg.solve_triangular(frame_L, resid.unsqueeze(-1), upper=False).squeeze(-1)
                    fr_nll = 0.5 * (z * z).sum(dim=-1) + frame_logsig.sum(dim=-1)
                else:
                    inv_var = torch.exp(-2.0 * frame_logsig)
                    fr_nll  = 0.5 * (resid * resid * inv_var).sum(dim=-1) + frame_logsig.sum(dim=-1)
                loss_fr = fr_nll.mean()
                loss    = loss + float(frame_pose_weight) * loss_fr
        if train:
            optimizer.zero_grad(set_to_none=True)
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            scaler.step(optimizer); scaler.update()
        with torch.no_grad():
            err_all = (params[valid][..., :2].float() - gt[valid]).norm(dim=-1)
            mse    = err_all.mean().item()
            is_obj = valid & (dist_uvd[..., 3] > 0.5)
            is_bg  = valid & (dist_uvd[..., 3] < 0.5)
            # keep err tensors on GPU; cat+cpu only at epoch end (avoid per-batch sync)
            if is_obj.any():
                obj_nll_s += gaussian2d_nll(params[is_obj], gt[is_obj]).item(); obj_n += 1
                obj_errs.append((params[is_obj][..., :2].float() - gt[is_obj]).norm(dim=-1).detach())
            if is_bg.any():
                bg_nll_s  += gaussian2d_nll(params[is_bg],  gt[is_bg]).item();  bg_n  += 1
                bg_errs.append((params[is_bg][..., :2].float() - gt[is_bg]).norm(dim=-1).detach())
        total_nll += loss.item(); total_mse += mse; n += 1
        pt_nll_s += loss_pt.item()
        if loss_fr is not None:
            fr_nll_s += loss_fr.item(); fr_n += 1
        if train and (n - _last_log_step >= 100):
            _dt = _time.time() - _t_start
            sps = n * imgs.shape[0] / _dt if _dt > 0 else 0
            print(f"  step {n}  loss={loss.item():+.3f}  sps={sps:.0f}", flush=True)
            _last_log_step = n
    obj_nll = obj_nll_s / max(obj_n, 1)
    bg_nll  = bg_nll_s  / max(bg_n,  1)
    # full per-point distribution stats — gives mean/median/p95 that line up
    # with eyeballed vis (median ≈ "typical"); mean inflated by long-tail.
    # torch.quantile errors at numel > 2^24; cat on GPU (= no per-batch sync),
    # then single .cpu().numpy() at epoch end.
    import numpy as _np
    def _stats(errs):
        if not errs:
            return float('nan'), float('nan'), float('nan')
        e = torch.cat(errs).cpu().numpy()
        return float(e.mean()), float(_np.median(e)), float(_np.percentile(e, 95))
    obj_mse, obj_med, obj_p95 = _stats(obj_errs)
    bg_mse,  bg_med,  bg_p95  = _stats(bg_errs)
    pt_nll  = pt_nll_s / max(n, 1)
    fr_nll  = (fr_nll_s / max(fr_n, 1)) if fr_n > 0 else float('nan')
    return (total_nll / max(n,1), total_mse / max(n,1),
            obj_nll, bg_nll, obj_mse, bg_mse,
            obj_med, obj_p95, bg_med, bg_p95,
            pt_nll, fr_nll)


def main(cfg=None):
    c = cfg if cfg is not None else CFG
    exp_dir = Path("experiments") / c["name"]
    exp_dir.mkdir(parents=True, exist_ok=True)
    log_path = exp_dir / "train.log"
    log_path.write_text("")

    # Persist config for WebUI
    (exp_dir / "config.py").write_text(
        "CFG = dict(\n" +
        "".join(f"    {k:<13}= {v!r},\n" for k, v in c.items()) + ")\n")

    def log(msg):
        ts = datetime.now().strftime("%H:%M:%S")
        line = f"{ts}  {msg}"
        print(line)
        with open(log_path, "a") as f: f.write(line+"\n")

    cache = c.get('cache')
    if not cache:
        raise SystemExit(
            "[train_ps_v3] --cache is required. "
            "Host-specific silent fallback was removed (2026-05-04): "
            "explicitly pass --cache /path/to/v3_cache."
        )
    # For incidental single-cache uses (vis_pretrain subprocess, periodic vis),
    # pick the first cache path. ConcatDataset / training uses all of them.
    cache_first = cache[0] if isinstance(cache, (list, tuple)) else cache
    zod_src = c.get('zod_src', None)
    nw = c.get('num_workers', 16)
    pf = c.get('prefetch_factor', 4)
    kw = dict(num_workers=nw, pin_memory=True,
              persistent_workers=(nw > 0),
              prefetch_factor=pf if nw > 0 else None,
              collate_fn=collate_full)
    val_nw = min(4, nw)  # val_loader runs sequentially; idle workers waste RAM
    val_kw = dict(num_workers=val_nw, pin_memory=True,
                  persistent_workers=(val_nw > 0),
                  prefetch_factor=pf if val_nw > 0 else None,
                  collate_fn=collate_full)
    import random as _r
    if zod_src:
        from datasets.zod_full import ZODCalibDataset
        log(f"loading ZOD direct-read from {zod_src} (no cache, KB-distortion projection)")
        ds_kw = dict(max_offset_m=c.get('max_offset_m', 0.20),
                      max_rot_deg=c.get('max_rot_deg', 1.0),
                      min_crop_px=c.get('min_crop_px', 256),
                      max_crop_px=c.get('max_crop_px', 768),
                      grid_n=c.get('grid_n', 16))
        log(f"  perturbation: ±{ds_kw['max_rot_deg']} deg / ±{ds_kw['max_offset_m']} m"
            f"   crop_px=[{ds_kw['min_crop_px']}, {ds_kw['max_crop_px']}] (full-image px → {c['img_size']})")
        tr_full = ZODCalibDataset(zod_src, split='train', img_size=c['img_size'],
                                    oversample=c.get('oversample', 4), **ds_kw)
        log(f"ZOD train: {len(tr_full)} instances")
        va_full = ZODCalibDataset(zod_src, split='val', img_size=c['img_size'],
                                    oversample=c.get('oversample', 4), **ds_kw)
    else:
        log(f"loading cache {cache} (V3 full-image)")
        ds_kw = dict(img_size=c['img_size'],
                      max_offset_m=c.get('max_offset_m', 0.20),
                      max_rot_deg=c.get('max_rot_deg', 0.5),
                      max_fx_pct=c.get('max_fx_pct', 0.0),
                      max_fy_pct=c.get('max_fy_pct', 0.0),
                      pose_frame=c.get('pose_frame', 'orig'),
                      min_crop_px=c.get('min_crop_px', 128),
                      max_crop_px=c.get('max_crop_px', 512),
                      frame_stride=c.get('frame_stride', 1),
                      grid_n=c.get('grid_n', 16),
                      oversample=c.get('oversample', 12),
                      zoom_aug=c.get('zoom_aug', False))
        log(f"  perturbation: ±{ds_kw['max_rot_deg']} deg / ±{ds_kw['max_offset_m']} m"
            f"   fx/fy: ±{ds_kw['max_fx_pct']*100:.2f}% / ±{ds_kw['max_fy_pct']*100:.2f}%"
            f"   crop_px=[{ds_kw['min_crop_px']}, {ds_kw['max_crop_px']}] (full-image px → {c['img_size']})")
        # Multi-cache support: --cache may be a list of paths. Joint training builds
        # ConcatDataset across PS/WM/ZOD caches so the model learns a single calib
        # ズレネット shared across sensor stacks (matches end-goal of zero-shot
        # transfer to company data; memory project_zero_shot_success_bar.md).
        cache_paths = cache if isinstance(cache, (list, tuple)) else [cache]
        ds_kw_val = dict(ds_kw); ds_kw_val['oversample'] = 1  # val never oversamples
        tr_per, va_per = [], []
        for cp in cache_paths:
            t = PandaSetCalibDatasetFull(cp, split='train', **ds_kw)
            v = PandaSetCalibDatasetFull(cp, split='val',   **ds_kw_val)
            tr_per.append(t); va_per.append(v)
            log(f"  + cache {cp}: train={len(t)} val={len(v)}")
        if len(tr_per) == 1:
            tr_full, va_full = tr_per[0], va_per[0]
        else:
            from torch.utils.data import ConcatDataset
            tr_full = ConcatDataset(tr_per)
            va_full = ConcatDataset(va_per)
        log(f"train cache(s) loaded: {len(tr_full)} instances")
    log(f"val cache loaded: {len(va_full)} instances (oversample=1)")
    # Sequence-level split: use the cache's pre-built train/val (scene-disjoint).
    # The previous code re-shuffled cache_train+cache_val with seed and cut at
    # val_fraction, which produced an OBJECT-level split (instances of the
    # same scene leak between train/val) — caused over-optimistic in-domain
    # val_nll because model effectively memorized scene-level features.
    # The cache split (PandaSetCalibDatasetFull(... split='train' / 'val')) is
    # already scene-disjoint per the build pipeline.
    train_ds = tr_full
    val_ds   = va_full
    log(f"sequence-level split (cache pre-built): train={len(train_ds)} val={len(val_ds)}")

    # Optional per-epoch random subsample (RandomSampler with num_samples).
    # Each epoch draws this many random indices (without replacement when
    # num_samples ≤ len(train_ds)) → caps wall-time per epoch independent of
    # cache size. Used for many-short-experiment workflows on the 384 cache.
    train_size = c.get('train_size', None)
    val_size   = c.get('val_size',   None)
    from torch.utils.data import RandomSampler, SequentialSampler
    if train_size and train_size < len(train_ds):
        tr_sampler = RandomSampler(train_ds, replacement=False, num_samples=train_size)
        log(f"  train subsample: {train_size}/{len(train_ds)} per epoch")
        train_loader = DataLoader(train_ds, batch_size=c["batch_size"], sampler=tr_sampler, **kw)
    else:
        train_loader = DataLoader(train_ds, batch_size=c["batch_size"], shuffle=True,  **kw)
    if val_size and val_size < len(val_ds):
        # deterministic first-N (sequential) so val NLL is comparable across runs
        val_subset = Subset(val_ds, list(range(val_size)))
        log(f"  val subsample: {val_size}/{len(val_ds)} (deterministic first-N)")
        val_loader = DataLoader(val_subset, batch_size=c["batch_size"], shuffle=False, **val_kw)
    else:
        val_loader   = DataLoader(val_ds,   batch_size=c["batch_size"], shuffle=False, **val_kw)

    model = CalibNetDepth(img_size=c["img_size"], in_channels=c["in_channels"],
                          n_layers=c["n_layers"], self_first=c.get("self_first", False),
                          use_convnext=c.get("use_convnext", False),
                          use_frustum=c.get("use_frustum", False),
                          frustum_dense=c.get("frustum_dense", False),
                          use_lidar_kv=c.get("use_lidar_kv", False),
                          use_pose_emb=c.get("use_pose_emb", False),
                          deform_mode=c.get("deform_mode", "none"),
                          use_frame_pose=c.get("use_frame_pose", False),
                          frame_pose_dof=c.get("frame_pose_dof", 6),
                          frame_pose_full_cov=c.get("frame_pose_full_cov", False)).to(DEVICE)
    log(f"params: {sum(p.numel() for p in model.parameters())/1e6:.2f}M")
    log(f"amp_dtype={_AMP_DTYPE} scaler_enabled={_NEED_SCALER} device={torch.cuda.get_device_name(0)} cc={torch.cuda.get_device_capability(0)}")

    optimizer = torch.optim.AdamW(model.parameters(), lr=c["lr"], weight_decay=1e-3)
    epochs    = c["epochs"]
    lr_min_r  = c["lr_min"] / c["lr"]
    def lr_lambda(e):
        if e < 5: return (e+1)/5
        t = (e-5)/max(1,epochs-5)
        return lr_min_r + (1-lr_min_r)*0.5*(1+math.cos(math.pi*t))
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)
    scaler    = torch.GradScaler(device="cuda", enabled=_NEED_SCALER)
    best_val  = float("inf")
    ckpt      = exp_dir / "best_model.pt"
    t0        = time.time()

    history = {'ep': [], 'tr_nll': [], 'va_nll': [], 'tr_mse': [], 'va_mse': []}

    # ClearML scalar logger: report metrics per epoch so the web UI shows curves
    # (otherwise only GPU auto-metrics appear). Pulls the active task.
    try:
        from clearml import Task as _ClearMLTask
        cml_logger = _ClearMLTask.current_task().get_logger() if _ClearMLTask.current_task() else None
    except Exception:
        cml_logger = None

    # Pre-training cache sanity vis: 10 random insts with GT projection on full image.
    # Lets a human eyeball the cache before sinking hours into training.
    if zod_src:
        log("vis_pretrain skipped (ZOD direct-read; full-image vis not yet wired)")
    else:
        # subprocess (NOT in-process import) so ClearML-agent's argparse
        # monkeypatch can't leak parent's empty Args/* params into vis_pretrain.
        # Argparse errors there were causing parent to die via SystemExit(2).
        import subprocess as _sp, sys as _sys
        try:
            _sp.run([_sys.executable, 'scripts/visualization/vis_pretrain.py',
                     '--cache', cache_first, '--out', str(exp_dir / 'vis_pretrain'),
                     '--n', '10'], check=False, env={**os.environ,
                                                        'CLEARML_TASK_ID': ''})
            log(f"vis_pretrain → {exp_dir / 'vis_pretrain'}")
            if cml_logger is not None:
                for p in sorted((exp_dir / 'vis_pretrain').glob('*.png')):
                    try: cml_logger.report_image('vis_pretrain', p.stem, iteration=0, local_path=str(p))
                    except Exception: pass
        except (Exception, SystemExit) as e:
            log(f"vis_pretrain skipped: {e}")

    def _midtrain_vis(epoch: int, n: int = 10):
        """Render N obj-centered val tiles with current model output. Thin
        wrapper around scripts.visualization.vis_eval.render_eval_samples so
        post-training demos go through the SAME pipeline (no private vis)."""
        if zod_src:
            from datasets.zod_full import ZODCalibDataset
            ds = ZODCalibDataset(zod_src, split='val',
                                  img_size=c['img_size'],
                                  min_crop_px=c.get('min_crop_px', 256),
                                  max_crop_px=c.get('max_crop_px', 768),
                                  oversample=1)
        else:
            from datasets.pandaset_full import PandaSetCalibDatasetFull
            ds = PandaSetCalibDatasetFull(cache_first, split='val',
                                                  img_size=c['img_size'],
                                                  min_crop_px=c.get('min_crop_px', 128),
                                                  max_crop_px=c.get('max_crop_px', 384),
                                                  max_fx_pct=c.get('max_fx_pct', 0.0),
                                                  max_fy_pct=c.get('max_fy_pct', 0.0),
                                                  max_offset_m=c.get('max_offset_m', 0.20),
                                                  max_rot_deg=c.get('max_rot_deg', 0.5),
                                                  oversample=1)
        from scripts.visualization.vis_eval import render_eval_samples
        return render_eval_samples(
            model=model, ds=ds, out_dir=exp_dir / f'vis_ep{epoch:03d}',
            img_size=int(c['img_size']), device=DEVICE, amp_dtype=_AMP_DTYPE,
            n=n, epoch=epoch, cml_logger=cml_logger, log=log)

    # Optional curriculum: schedule = list of (start_ep, end_ep, rot_deg, t_m).
    # If set, sigma is clamped to that schedule at each epoch start. The dataset's
    # max_rot_deg / max_offset_m are mutated in place — but workers cache the dataset
    # at fork time, so persistent_workers=False is required for the curriculum to
    # take effect mid-run. We restart workers per stage transition by toggling
    # train_loader.dataset attributes and recreating the loader.
    sigma_schedule = c.get('sigma_schedule', None)
    def _sigma_for_epoch(ep):
        if not sigma_schedule:
            return None
        for s in sigma_schedule:
            if s[0] <= ep <= s[1]:
                return s[2], s[3]
        return sigma_schedule[-1][2], sigma_schedule[-1][3]

    # ep0 baseline disabled — in-process matplotlib draw of 10·300+ cuboid
    # edges blocks the main thread for several minutes (forkserver workers
    # can't even start). Skipping until vis_ep000 is moved to subprocess.

    cur_sigma = None
    for epoch in range(1, epochs+1):
        _ep_start = time.time()
        # Stage 2 query drop schedule: linear 0.2 → 0.8 over training.
        if c.get('query_drop', False):
            t = max(0.0, min(1.0, (epoch - 1) / max(1, epochs - 1)))
            qdp = 0.2 + 0.6 * t
            model.query_drop_prob = float(qdp)
            if epoch == 1 or (epoch % 10 == 0):
                log(f"  query_drop_prob={qdp:.2f}")
        if sigma_schedule is not None:
            new_sigma = _sigma_for_epoch(epoch)
            if new_sigma != cur_sigma:
                rot, tm = new_sigma
                tr_full.max_rot_deg = rot;  tr_full.max_offset_m = tm
                va_full.max_rot_deg = rot;  va_full.max_offset_m = tm
                # rebuild loaders with workers=0 OR fresh persistent workers
                # so the in-memory dataset state propagates.
                train_loader = DataLoader(train_ds, batch_size=c["batch_size"],
                                          shuffle=True, **kw)
                val_loader   = DataLoader(val_ds,   batch_size=c["batch_size"],
                                          shuffle=False, **kw)
                cur_sigma = new_sigma
                log(f"  curriculum: sigma → rot={rot} t={tm}")
        fp_w = float(c.get('frame_pose_weight', 0.5))
        (tr_nll, tr_mse, tr_obj, tr_bg, tr_obj_mse, tr_bg_mse,
         tr_obj_med, tr_obj_p95, tr_bg_med, tr_bg_p95,
         tr_pt_nll, tr_fr_nll) = epoch_loop(
            model, train_loader, optimizer, scaler, True, frame_pose_weight=fp_w)
        with torch.no_grad():
            (va_nll, va_mse, va_obj, va_bg, va_obj_mse, va_bg_mse,
             va_obj_med, va_obj_p95, va_bg_med, va_bg_p95,
             va_pt_nll, va_fr_nll) = epoch_loop(
                model, val_loader, optimizer, scaler, False, frame_pose_weight=fp_w)
        scheduler.step()
        history['ep'].append(epoch)
        history['tr_nll'].append(tr_nll); history['va_nll'].append(va_nll)
        history['tr_mse'].append(tr_mse); history['va_mse'].append(va_mse)
        # Log per-pt and frame-pose NLL separately so we can verify the head
        # is actually learning (frame nll must decrease for the head to be useful).
        log(f"[{epoch:3d}/{epochs}]  "
            f"train nll={tr_nll:+.3f}(pt={tr_pt_nll:+.3f} fr={tr_fr_nll:+.3f} obj={tr_obj:+.3f} bg={tr_bg:+.3f}) "
            f"mse={tr_mse:.2f}(obj={tr_obj_mse:.2f}/m{tr_obj_med:.2f}/95p{tr_obj_p95:.1f} "
            f"bg={tr_bg_mse:.2f}/m{tr_bg_med:.2f}/95p{tr_bg_p95:.1f})  "
            f"val nll={va_nll:+.3f}(pt={va_pt_nll:+.3f} fr={va_fr_nll:+.3f} obj={va_obj:+.3f} bg={va_bg:+.3f}) "
            f"mse={va_mse:.2f}(obj={va_obj_mse:.2f}/m{va_obj_med:.2f}/95p{va_obj_p95:.1f} "
            f"bg={va_bg_mse:.2f}/m{va_bg_med:.2f}/95p{va_bg_p95:.1f})  "
            f"lr={scheduler.get_last_lr()[0]:.2e}  tot={(time.time()-t0)/60:.1f}min")
        if va_nll < best_val:
            best_val = va_nll
            torch.save(model.state_dict(), ckpt)
            log(f"  ↳ saved (val_nll={best_val:.4f})")
        # report scalars to ClearML so curves render in the web UI
        if cml_logger is not None:
            try:
                rs = cml_logger.report_scalar
                rs('nll', 'train',     iteration=epoch, value=tr_nll)
                rs('nll', 'val',       iteration=epoch, value=va_nll)
                rs('nll', 'val_obj',   iteration=epoch, value=va_obj)
                rs('nll', 'val_bg',    iteration=epoch, value=va_bg)
                rs('mse_px', 'train',  iteration=epoch, value=tr_mse)
                rs('mse_px', 'val',    iteration=epoch, value=va_mse)
                rs('mse_px', 'val_obj_mean',   iteration=epoch, value=va_obj_mse)
                rs('mse_px', 'val_obj_median', iteration=epoch, value=va_obj_med)
                rs('mse_px', 'val_obj_p95',    iteration=epoch, value=va_obj_p95)
                rs('mse_px', 'val_bg_mean',    iteration=epoch, value=va_bg_mse)
                rs('mse_px', 'val_bg_median',  iteration=epoch, value=va_bg_med)
                rs('mse_px', 'val_bg_p95',     iteration=epoch, value=va_bg_p95)
                rs('lr', 'lr', iteration=epoch, value=scheduler.get_last_lr()[0])
                rs('best', 'best_val_nll', iteration=epoch, value=best_val)
                # epoch wall-clock + sps (samples/sec across train epoch)
                if 'epoch_t0' in globals():
                    pass  # placeholder
                _ep_dt = time.time() - _ep_start
                _ep_sps = (len(train_loader) * c["batch_size"]) / max(_ep_dt, 1e-6)
                rs('sps', 'train_epoch', iteration=epoch, value=_ep_sps)
                rs('time', 'epoch_sec',  iteration=epoch, value=_ep_dt)
            except Exception:
                pass

        if epoch % 10 == 0 or epoch == epochs:
            try: _midtrain_vis(epoch, n=10)
            except Exception as _e: log(f"vis_ep{epoch:03d} skipped: {_e}")

    log(f"Best val NLL: {best_val:.4f}  |  time: {(time.time()-t0)/60:.1f}min")

    # ── curves ──
    vis_dir = exp_dir / "vis"; vis_dir.mkdir(exist_ok=True)
    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    axes[0].plot(history['ep'], history['tr_nll'], label='train'); axes[0].plot(history['ep'], history['va_nll'], label='val')
    axes[0].set_title('NLL'); axes[0].set_xlabel('epoch'); axes[0].legend(); axes[0].grid(alpha=0.3)
    axes[1].plot(history['ep'], history['tr_mse'], label='train'); axes[1].plot(history['ep'], history['va_mse'], label='val')
    axes[1].set_title('MSE (px)'); axes[1].set_xlabel('epoch'); axes[1].legend(); axes[1].grid(alpha=0.3)
    plt.tight_layout(); plt.savefig(vis_dir / "curves.png", dpi=100); plt.close(fig)

    # ── vis (V3 cache uses different schema than vis_ps expects; skip) ──
    n_vis = 48
    try:
        from scripts.visualization.vis_ps_v3 import main as vis_main
        vis_main(c["name"], n_vis=n_vis, cache=cache_first)
        log(f"Saved {n_vis} vis → {vis_dir}")
    except Exception as e:
        n_vis = 0
        log(f"vis: skipped due to error: {e}")

    # ── report.html (overwritten by vis_ps; rebuild to include local best_nll) ──
    vis_imgs = "\n".join(f'      <img src="vis/val_{i:02d}.png" style="width:200px; margin:4px; border:1px solid #333;">' for i in range(n_vis))
    report = f"""<!doctype html><meta charset="utf-8">
<title>{c['name']} — report</title>
<style>body{{background:#0d1117;color:#eee;font-family:sans-serif;padding:20px;}} pre{{background:#161b22;padding:10px;border-radius:6px;}} h2{{border-bottom:1px solid #333;padding-bottom:4px;}}</style>
<h1>{c['name']}</h1>
<p>best val NLL: <b>{best_val:.4f}</b></p>
<pre>{chr(10).join(f'{k:<15}= {v!r}' for k,v in c.items())}</pre>
<h2>Curves</h2>
<img src="vis/curves.png" style="max-width:800px;">
<h2>Val samples</h2>
<div style="display:flex;flex-wrap:wrap;">
{vis_imgs}
</div>
"""
    (exp_dir / "report.html").write_text(report)
    log("Report saved → report.html")


if __name__ == "__main__":
    # Avoid fork-after-thread deadlock: ClearML's reporter thread holds locks
    # at fork time, and DataLoader workers inherit those locks but not the
    # holder thread → futex wait forever. forkserver spawns a clean server
    # process up-front; all workers are forked from that thread-less server.
    import multiprocessing as _mp
    try:
        _mp.set_start_method('forkserver', force=True)
    except RuntimeError:
        pass
    import argparse, copy
    ap = argparse.ArgumentParser()
    ap.add_argument('--name')
    ap.add_argument('--cache', nargs='+',
                    help='cache path(s). Multiple → ConcatDataset for joint '
                         'multi-dataset training (e.g. PS+WM+ZOD).')
    ap.add_argument('--zod-src', default=None,
                    help='ZOD Frames root (e.g. /mnt/nvme6t/zod/frames). '
                         'When set, --cache is ignored and frames are read '
                         'directly via the official zod toolkit (KB-distortion '
                         'projection, no .pt cache).')
    ap.add_argument('--rot-deg', type=float, default=None,
                    help='extrinsic perturbation half-range (deg per axis)')
    ap.add_argument('--t-m',     type=float, default=None,
                    help='extrinsic perturbation half-range (m per axis)')
    ap.add_argument('--epochs',  type=int,   default=None)
    ap.add_argument('--lr',      type=float, default=None)
    ap.add_argument('--lr-min',  type=float, default=None)
    ap.add_argument('--frame-stride', type=int, default=1,
                    help='subsample fnames by this stride at runtime (e.g. 5 → 2Hz from 10Hz cache)')
    ap.add_argument('--grid-n', type=int, default=None,
                    help='point sub-grid size (16=default, 8 for coarser, set with img-size for matching coarse feat map)')
    ap.add_argument('--workers', type=int,   default=None,
                    help='dataloader workers (default 16; sakurai2 may hang at 16, try 4)')
    ap.add_argument('--min-crop-px', type=int, default=None,
                    help='min random crop side in full-image px (default 128)')
    ap.add_argument('--max-crop-px', type=int, default=None,
                    help='max random crop side in full-image px (default 512)')
    ap.add_argument('--oversample', type=int, default=None,
                    help='per-epoch oversample factor (each frame yields N random crops; default 12 → ~90K/ep matches v9_lazy)')
    ap.add_argument('--batch-size', type=int, default=None,
                    help='train/val batch size (default 64; bump to 256 for v2 cache to feed GPU)')
    ap.add_argument('--prefetch-factor', type=int, default=None,
                    help='DataLoader prefetch factor (default 4)')
    ap.add_argument('--n-layers', type=int, default=None,
                    help='cross-attn layers (default 2; all_v1 used 4)')
    ap.add_argument('--img-size', type=int, default=None,
                    help='model input side (px); default 64. e.g. 128 for higher-res experiment')
    ap.add_argument('--convnext', action='store_true',
                    help='use ConvNeXt backbone (=all_v1 1.62M config)')
    ap.add_argument('--deform-mode', default=None,
                    help="deformable cross-attn: 'none' (default) | 'sl' single-level | 'ml' multi-level")
    ap.add_argument('--frustum-dense', action='store_true',
                    help='emit dense gh*gw LiDAR map (zero-pad empty cells + '
                         'learnable per-cell UV embedding) and feed it as '
                         'extra_kv to deform-CA — instead of adding scattered '
                         'frustum tokens to q. Hybrid: DA on image + regular '
                         'softmax CA on the dense LiDAR map.')
    ap.add_argument('--use-lidar-kv', action='store_true',
                    help='Stage 1 (hybrid KV): add a lidar bank to KV using '
                         'initial Q copy as per-pivot lidar features. '
                         'Combined with --use-pose-emb gives the unified '
                         'KV concat per docs/model_progression.md.')
    ap.add_argument('--use-pose-emb', action='store_true',
                    help='Stage 1: VCPE (Virtual Camera Pose Embedding) — '
                         'MLP([SE3, log(vfp)]) → broadcast to Q + KV. SE3=0 '
                         'in calib regime, varies in cross-frame.')
    ap.add_argument('--query-drop', action='store_true',
                    help='Stage 2 (mixed Q): random Bernoulli zero-out of '
                         'query depth (d) so the model learns UV-only Q. '
                         'Schedule: 0.2 → 0.8 over training epochs. '
                         'Combine with --use-pose-emb --frustum-dense for full stage 2.')
    ap.add_argument('--use-frame-pose', action='store_true',
                    help='enable CLS frame-pose head (patch-level 6-DoF + diagonal cov)')
    ap.add_argument('--frame-pose-weight', type=float, default=None,
                    help='loss weight for frame-pose NLL (default 0.5)')
    ap.add_argument('--max-fx-pct', type=float, default=None,
                    help='multiplicative fx perturbation half-range (e.g. 0.02 = ±2%). '
                         'When nonzero, dataset perturbs K[0,0] per sample and the '
                         'frame-pose head learns to regress Δfx as a separate dim.')
    ap.add_argument('--max-fy-pct', type=float, default=None,
                    help='multiplicative fy perturbation half-range (independent of fx). '
                         'PS BA shows fx/fy drift independently per cam → train them separately.')
    ap.add_argument('--frame-pose-full-cov', action='store_true',
                    help='CLS pose head outputs full n_dof×n_dof Cholesky cov '
                         '(off-diagonal captures pose-dim correlations for BA). '
                         'Default: diagonal Σ (compatible with legacy ckpts).')
    ap.add_argument('--pose-frame', choices=('orig','vcam'), default=None,
                    help='Frame in which the CLS pose-head target is expressed. '
                         "'orig' (default): orig camera frame — label depends on "
                         'where in the image the tile was cropped (model needs '
                         'visual context to disambiguate position relative to '
                         'optical axis). '
                         "'vcam': tile-local virtual-camera frame whose optical "
                         'axis is the ray through the tile center → label is '
                         'crop-position-agnostic. roll_vcam=0 (dropped). '
                         'Downstream BA aggregates per-tile (μ_v, Σ_v) via '
                         'J_i = R_orig→vcam_i.')
    ap.add_argument('--zoom-aug', action='store_true',
                    help='depth-dependent zoom-in aug: shrink cs by up to '
                         'scale_max(z) (1.0@z=20m → 2.0@z>=100m). Synthesizes '
                         'telephoto views of distant objects to fill the '
                         'high-resolution far-object regime native lens lacks.')
    ap.add_argument('--train-size', type=int, default=None,
                    help='per-epoch random subsample of train set (caps wall-time). '
                         'e.g. 20000 → 78 batches @ bs=256 → ~14s/ep')
    ap.add_argument('--val-size', type=int, default=None,
                    help='deterministic first-N val subsample for fast eval')
    ap.add_argument('--curriculum', default=None,
                    help='sigma curriculum spec, semicolon-separated stages '
                         'e.g. "1-25:0.5,0.05;26-60:1.0,0.10;61-100:2.0,0.20"')
    ap.add_argument('--clearml', action=argparse.BooleanOptionalAction, default=True,
                    help='log to ClearML (default: on). --no-clearml to disable.')
    ap.add_argument('--queue', default=None,
                    help='ClearML queue name. If set, Task.init then '
                         'execute_remotely() — local process exits, agent on '
                         'that queue picks up the run (e.g. dgx2 / sakurai2).')
    ap.add_argument('--why',     default='')
    args = ap.parse_args()
    cfg = dict(CFG)
    if args.name:    cfg['name'] = args.name
    if args.cache:   cfg['cache'] = args.cache
    if args.zod_src: cfg['zod_src'] = args.zod_src
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
    if args.deform_mode is not None: cfg['deform_mode'] = args.deform_mode
    if args.frustum_dense: cfg['frustum_dense'] = True
    if args.use_lidar_kv:  cfg['use_lidar_kv']  = True
    if args.use_pose_emb:  cfg['use_pose_emb']  = True
    if args.query_drop:    cfg['query_drop']    = True
    if args.use_frame_pose: cfg['use_frame_pose'] = True
    if args.frame_pose_weight is not None: cfg['frame_pose_weight'] = args.frame_pose_weight
    if args.max_fx_pct is not None: cfg['max_fx_pct'] = args.max_fx_pct
    if args.max_fy_pct is not None: cfg['max_fy_pct'] = args.max_fy_pct
    if args.pose_frame is not None: cfg['pose_frame'] = args.pose_frame
    if args.frame_pose_full_cov: cfg['frame_pose_full_cov'] = True
    # Default frame_pose_dof based on pose_frame and fx/fy aug:
    #   pose_frame='orig' + fx/fy aug:  8 (full 6-DoF SE3 + Δfx + Δfy)
    #   pose_frame='orig' no fx/fy:     6 (default of CLSFramePoseHead)
    #   pose_frame='vcam' + fx/fy aug:  7 (5-DoF VCAM + Δfx + Δfy)
    #   pose_frame='vcam' no fx/fy:     5 (5-DoF VCAM only)
    if cfg.get('use_frame_pose', False) and 'frame_pose_dof' not in cfg:
        has_fxfy = cfg.get('max_fx_pct', 0.0) > 0 or cfg.get('max_fy_pct', 0.0) > 0
        if cfg.get('pose_frame', 'orig') == 'vcam':
            cfg['frame_pose_dof'] = 7 if has_fxfy else 5
        elif has_fxfy:
            cfg['frame_pose_dof'] = 8
    if args.zoom_aug: cfg['zoom_aug'] = True
    if args.train_size is not None: cfg['train_size'] = args.train_size
    if args.val_size   is not None: cfg['val_size']   = args.val_size
    if args.curriculum:
        # parse "1-25:0.5,0.05;26-60:1.0,0.10;..."
        stages = []
        for part in args.curriculum.split(';'):
            rng, sigmas = part.strip().split(':')
            ep_lo, ep_hi = (int(x) for x in rng.split('-'))
            rot, tm = (float(x) for x in sigmas.split(','))
            stages.append((ep_lo, ep_hi, rot, tm))
        cfg['sigma_schedule'] = stages

    # optional ClearML reporting (with rich context if --clearml + --why)
    cml_task = None
    if args.clearml or args.queue:
        from scripts.util.clearml_context import init_with_context, write_retrospective
        cml_task = init_with_context(
            project='e2e_calib/calib', name=cfg['name'], cfg=cfg,
            why=args.why, baseline={'name':'ps_v9_lazy', 'metric':'val_nll', 'value':1.7176})
    if args.queue:
        # Submit to remote agent on this queue and exit local process.
        # Agent re-clones at task.script.repository@task.script.version_num
        # and runs main(cfg=cfg) on its host.
        cml_task.execute_remotely(queue_name=args.queue, clone=False, exit_process=True)
    main(cfg=cfg)
