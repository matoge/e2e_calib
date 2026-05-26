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
import torch.multiprocessing as _tmp
try: _tmp.set_sharing_strategy('file_system')
except Exception: pass
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
    img_size      = 128,
    in_channels   = 3,
    use_convnext  = True,
    use_frustum   = True,
    epochs        = 100,
    batch_size    = 64,       # per-process
    lr            = 1e-3,
    lr_min        = 1e-6,
    val_fraction  = 0.1,
    split_seed    = 42,
    min_crop_px   = 128,
    max_crop_px   = 384,
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
    for batch in loader:
        # collate_full evolved 7 → 8 (pert_6vec) → 12-tuple (orig-cam solver
        # tensors) → 13-tuple (δ1 hint for split_pert). DDP path needs the first
        # 7 fields + δ1 (slot 12); the rest are consumed by the BA eval hook.
        imgs, true_uvd, dist_uvd, pad_mask, vfp, bucket_uvd, bucket_valid = batch[:7]
        delta1_se3 = batch[12] if len(batch) >= 13 else None
        # accel.prepare already puts tensors in device via DataLoader wrap, but
        # the collate returns uint8 images — convert to float on-device here.
        imgs     = imgs.float().div_(255.0)
        gt       = true_uvd[..., :2] - dist_uvd[..., :2]
        # Accelerate manages autocast via its mixed_precision plugin; we just
        # call the model normally.
        # NOTE: bucket_uvd / bucket_valid are REQUIRED when frustum_enc is
        # enabled (model raises otherwise). Harmless extra kwargs when off.
        # API renamed 2026-05-04 (commit 01abd02) — flat distorted_uvd_full
        # → bucketed (G², K, 3) layout.
        # dist_uvd cols: [u, v, d, is_obj, intensity]. Model wants UVD(+I) so
        # we slice out the is_obj column (3) and pass [u,v,d, intensity] when
        # use_intensity=True (model.point_mlp in_channels=4), else [u,v,d].
        if getattr(accel.unwrap_model(model), 'use_intensity', False):
            point_in = torch.cat([dist_uvd[..., :3], dist_uvd[..., 4:5]], dim=-1)
        else:
            point_in = dist_uvd[..., :3]
        params = model(imgs, point_in, key_padding_mask=pad_mask, vfp=vfp,
                       bucket_uvd=bucket_uvd, bucket_valid=bucket_valid,
                       pose_emb_se3=delta1_se3)
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
        # Lowered from 100 to 25 so short runs (smoke / small train-size) still
        # see step-level sps. Otherwise users report "SPS not appearing".
        if train and accel.is_main_process and (n - _last_log_step >= 25):
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
    # find_unused_parameters: FALSE by default (fast path). Earlier comment
    # claimed frustum_enc ON + n_layers=2 caused AllReduce hang — that was
    # verified on 2026-05-04 against the pre-bucket (flat distorted_uvd_full)
    # API. After 01abd02 (bucket API) + 1868604 (DDP loop rename) the param
    # usage pattern stabilized. 2026-05-05 regression: True caused 22× slowdown
    # (1552 sps/global → 72 sps). If DDP actually hangs with False, add
    # DistributedDataParallelKwargs(find_unused_parameters=True) back.
    fup = bool(c.get('find_unused_parameters', False))
    if fup:
        from accelerate import DistributedDataParallelKwargs as _DDPK
        accel = Accelerator(kwargs_handlers=[_DDPK(find_unused_parameters=True)])
    else:
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
              collate_fn=collate_full,
              multiprocessing_context='spawn' if nw > 0 else None)
    val_nw = min(4, nw)
    val_kw = dict(num_workers=val_nw, pin_memory=True,
                  persistent_workers=(val_nw > 0),
                  prefetch_factor=pf if val_nw > 0 else None,
                  collate_fn=collate_full,
                  multiprocessing_context='spawn' if val_nw > 0 else None)
    import random as _r
    ds_kw = dict(img_size=c['img_size'],
                  max_offset_m=c.get('max_offset_m', 0.20),
                  max_rot_deg=c.get('max_rot_deg', 0.5),
                  min_crop_px=c.get('min_crop_px', 128),
                  max_crop_px=c.get('max_crop_px', 512),
                  frame_stride=c.get('frame_stride', 1),
                  grid_n=c.get('grid_n', 16),
                  oversample=c.get('oversample', 12),
                  split_pert=c.get('split_pert', False))
    log(f"cache={cache}  perturbation=±{ds_kw['max_rot_deg']} deg / ±{ds_kw['max_offset_m']} m  "
        f"crop_px=[{ds_kw['min_crop_px']}, {ds_kw['max_crop_px']}] (full → {c['img_size']})")
    cache_paths = [p.strip() for p in str(cache).split(',') if p.strip()]
    from torch.utils.data import ConcatDataset
    tr_parts, va_parts = [], []
    for cp in cache_paths:
        # Waymo cache is already huge (5M tiles); oversample=1 keeps it
        # in the same epoch budget as the smaller (kami/woven) caches at
        # the user-supplied oversample. Override per-cache.
        cp_kw = dict(ds_kw)
        if 'waymo' in cp.lower():
            cp_kw['oversample'] = 1
        tr_p = PandaSetCalibDatasetFull(cp, split='train', **cp_kw)
        # Val: optionally restrict pivots to the central vertical band so the
        # set is dominated by interpretable mid-image samples (default 0.5 =
        # central 50% of rows). Set val_center_band=0 to disable.
        va_kw = dict(cp_kw)
        va_kw['center_band'] = float(c.get('val_center_band', 0.5))
        va_p = PandaSetCalibDatasetFull(cp, split='val',   **va_kw)
        log(f"  [{cp}] train={len(tr_p)} val={len(va_p)} (os={cp_kw['oversample']})")
        tr_parts.append(tr_p); va_parts.append(va_p)
    tr_full = ConcatDataset(tr_parts) if len(tr_parts) > 1 else tr_parts[0]
    va_full = ConcatDataset(va_parts) if len(va_parts) > 1 else va_parts[0]
    log(f"train cache: {len(tr_full)} inst   val cache: {len(va_full)} inst")

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
                          frustum_grid_n=c.get("grid_n", 16),
                          frustum_dense=c.get("frustum_dense", False),
                          use_pose_emb=c.get("use_pose_emb", False),
                          deform_mode=c.get("deform_mode", "none"),
                          convnext_n_blocks=c.get("convnext_n_blocks", 2),
                          convnext_fine_d=c.get("convnext_fine_d", None),
                          convnext_stem_d=c.get("convnext_stem_d", None))
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

    # Optional warm-start: load weights from another experiment's best_model.pt.
    # Only the model state_dict is restored; optimizer / scheduler / epoch
    # counter start fresh (use a longer --epochs to budget for the extra
    # training). Useful for "extend N→N+M epochs" without resume bookkeeping.
    init_from = c.get('init_from')
    if init_from:
        ifp = Path(init_from)
        if not ifp.is_absolute():
            ifp = REPO_ROOT / 'experiments' / ifp / 'best_model.pt' \
                  if (REPO_ROOT := Path(__file__).resolve().parents[2]) else ifp
        if not ifp.is_file():
            ifp = Path('experiments') / init_from / 'best_model.pt'
        log(f"warm-start from: {ifp}")
        sd = torch.load(ifp, map_location='cpu', weights_only=True)
        # strict=False tolerates missing/unexpected keys but NOT shape mismatch
        # (e.g. img_size 128→256 changes pos_emb, grid_n 16→32 changes
        # frustum_enc.cell_uv_embed). Drop any tensor whose shape disagrees so
        # they fall back to fresh init while everything else (CNN backbone,
        # attention QKV) is still warm-started.
        cur_sd = accel.unwrap_model(model).state_dict()
        skipped = []
        filt = {}
        for k, v in sd.items():
            if k in cur_sd and tuple(cur_sd[k].shape) != tuple(v.shape):
                skipped.append((k, tuple(v.shape), tuple(cur_sd[k].shape)))
                continue
            filt[k] = v
        miss = accel.unwrap_model(model).load_state_dict(filt, strict=False)
        log(f"warm-start: loaded {len(filt)}/{len(sd)} tensors  "
            f"(skipped {len(skipped)} shape-mismatch, "
            f"{len(miss.missing_keys)} missing, "
            f"{len(miss.unexpected_keys)} unexpected)")
        for k, src, dst in skipped[:8]:
            log(f"  shape-skip {k}: ckpt{src} != model{dst}")
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

    # vis_pretrain is removed; the preflight midtrain_vis below renders the
    # same per-dataset samples but on the actual model (untrained at ep=0)
    # so we get the GT correspondence + Σ ellipse overlays for free.

    # ── BA pose-residual eval setup (rank-0 only, optional) ─────────────────
    # Re-uses _solve_one from scripts/eval/eval_shared_256x800.py: builds
    # B = n_inst × n_per_inst sub-crops with one shared rig-level perturbation,
    # runs frozen-current-model forward + shared-GN, returns δ̂ vs δ_target.
    # The 2026-05-22 256×800 result (ω res 0.45 px @ fx) used the same path
    # against the HEAD ckpt; here we run it as a tiny in-train metric.
    ba_eval_state = None
    if c.get('ba_eval', True) and accel.is_main_process:
        try:
            from scripts.eval import eval_shared_256x800 as _baeval
            ba_idx       = int(c.get('ba_eval_idx', 17))
            # Sweep over multiple perturbation magnitudes per epoch so we see
            # how residual scales with target. Default: 0.5° / 1.0° / 1.5° at
            # matching translation. Caller can override via --ba-eval-levels
            # (parsed below) or set --ba-eval-rot-deg / --ba-eval-t-m for a
            # single-level run.
            ba_levels = c.get('ba_eval_levels')
            if ba_levels is None:
                if 'ba_eval_rot_deg' in c or 'ba_eval_t_m' in c:
                    ba_levels = [(float(c.get('ba_eval_rot_deg', 0.5)),
                                  float(c.get('ba_eval_t_m', 0.05)))]
                else:
                    ba_levels = [(0.5, 0.05), (1.0, 0.10), (1.5, 0.20)]
            ba_n_seeds   = int(c.get('ba_eval_n_seeds', 4))
            ba_n_inst    = int(c.get('ba_eval_n_inst', 20))
            ba_start_ep  = int(c.get('ba_eval_start_ep', 1))
            ba_cache     = c.get('ba_eval_cache', cache_paths[0])
            # cs/n_per_inst: always 256-quadrant split (4-per-inst) so n_inst=200
            # → 800 tiles. Decoupled from c['img_size']: even at img_size=128,
            # eval crops are taken at 256 in original-camera units and then
            # resized to S=img_size for the model — larger crops give more
            # shared-GN Fisher info per tile (project_resolution_hypothesis_512).
            # Caller can override via --ba-eval-cs / --ba-eval-npi.
            ba_cs        = int(c.get('ba_eval_cs', 256))
            ba_npi       = int(c.get('ba_eval_npi', 4))

            ba_ds = PandaSetCalibDatasetFull(
                cache_dir=ba_cache, split='val',
                img_size=c['img_size'],
                min_crop_px=c.get('min_crop_px', 128),
                max_crop_px=c.get('max_crop_px', 512),
                max_offset_m=0.0, max_rot_deg=0.0,
                oversample=1, grid_n=c.get('grid_n', 16),
                center_band=0.0, preload=False,
            )
            inst0 = ba_ds._load_inst(ba_idx)
            assert inst0.get('is_fisheye', False), f'BA eval idx={ba_idx} not fisheye'
            ba_dist_one = inst0['distortion'].clone().detach().to(torch.float32).reshape(1, 4)
            ba_fx = float(inst0['K_full'].numpy()[0, 0])
            _baeval.DEVICE = accel.device
            ba_every_n   = int(c.get('ba_eval_every', 1))
            ba_eval_state = dict(
                mod=_baeval, ds=ba_ds, dist_one=ba_dist_one, fx=ba_fx,
                idx=ba_idx, levels=ba_levels,
                n_seeds=ba_n_seeds, n_inst=ba_n_inst, start_ep=ba_start_ep,
                every_n=ba_every_n,
                cs=ba_cs, npi=ba_npi,
            )
            levels_str = ' / '.join(f'±{r}°,±{t}m' for (r, t) in ba_levels)
            log(f"BA eval: idx={ba_idx} levels=[{levels_str}] "
                f"K={ba_n_seeds} seeds × n_inst={ba_n_inst} × cs={ba_cs}({ba_npi}-per) "
                f"start_ep={ba_start_ep} every_n={ba_every_n}  fx={ba_fx:.1f}px")
        except Exception as _e:
            log(f"BA eval setup skipped: {_e!r}")
            ba_eval_state = None

    def _run_ba_eval(unwrapped, ep):
        """Returns list of per-level dicts (omega_deg, omega_px, t_m, label,
        vis_path). Renders one 3-panel overlay per level (seed=0) so the
        ClearML images tab shows GT / perturbed / corrected at each ep.
        """
        if ba_eval_state is None or ep < ba_eval_state['start_ep']:
            return None
        s = ba_eval_state
        # Skip ep that aren't on the every_n cadence — but always run on
        # ep == start_ep (sanity gate) and on the last epoch.
        every_n = int(s.get('every_n', 1))
        if every_n > 1 and ep != s['start_ep'] and ep != int(c.get('epochs', ep)) \
                and (ep - s['start_ep']) % every_n != 0:
            return None
        import numpy as _np
        was_train = unwrapped.training
        unwrapped.eval()
        out = []
        vis_dir = exp_dir / '_ba_vis' / f'ep{ep:03d}'
        target_inst = s['ds']._load_inst(int(s['idx']))
        try:
            for li, (rot_deg, t_m) in enumerate(s['levels']):
                omegas, ts = [], []
                first_delta = None
                first_ypr = first_tt = None
                for k in range(s['n_seeds']):
                    rng = _np.random.RandomState(1000 + 100 * li + k)
                    ypr_t, t_t = s['mod']._draw_pert(rng, rot_deg=rot_deg, t_m=t_m)
                    rng2 = _np.random.RandomState(2000 + 100 * li + k)
                    d, _, _ = s['mod']._solve_one(
                        unwrapped, s['ds'],
                        target_idx=s['idx'], n_inst=s['n_inst'],
                        cs=s['cs'], n_per_inst=s['npi'], rng=rng2,
                        ypr_target=ypr_t, t_target=t_t,
                        dist_one=s['dist_one'], cfg=c, label=f'ep{ep}-L{li}-s{k}')
                    if k == 0:
                        first_delta, first_ypr, first_tt = d, ypr_t, t_t
                    tgt = _np.array([ypr_t[2], ypr_t[1], ypr_t[0]], dtype=_np.float64)
                    d_np = d.detach().cpu().numpy()
                    omegas.append(_np.linalg.norm(d_np[:3] - tgt))
                    ts.append(_np.linalg.norm(d_np[3:] - t_t))
                omega_mean = float(_np.mean(omegas))
                t_mean     = float(_np.mean(ts))
                omega_px   = float(s['fx'] * _np.tan(_np.deg2rad(omega_mean)))
                label = f'r{rot_deg}_t{t_m}'
                # Render overlay for the seed=0 solve
                vis_path = None
                try:
                    vis_path = vis_dir / f'L{li}_{label}.png'
                    suptitle = (f'ep={ep} idx={s["idx"]}  ±{rot_deg}°/±{t_m}m  '
                                f'ω={omega_mean:.4f}° ({omega_px:.3f}px@fx) '
                                f't={t_mean:.4f}m  (mean over K={s["n_seeds"]} seeds)')
                    s['mod'].render_3panel_overlay(
                        target_inst, first_ypr, first_tt, first_delta,
                        out_path=vis_path, suptitle=suptitle,
                        panel_label=f'BA-corrected (seed0)')
                except Exception as _e:
                    log(f"BA vis ep={ep} L{li} skipped: {_e!r}")
                    vis_path = None
                out.append(dict(omega_deg=omega_mean, omega_px=omega_px,
                                t_m=t_mean, rot_deg=rot_deg, t_m_target=t_m,
                                label=label, vis_path=vis_path))
        finally:
            if was_train: unwrapped.train()
        return out

    # ── PRE-FLIGHT (ep=0): val pass on the untrained model so we fail fast
    # (within minutes) if loader/model is broken. midtrain_vis is gated to
    # rank-0 and the other ranks would otherwise NCCL-AllReduce-timeout
    # (10 min watchdog) waiting for it — keep vis to the regular every-10ep
    # cadence inside the train loop.
    log("preflight: val pass on untrained model")
    with torch.no_grad():
        epoch_loop(model, val_loader, optimizer, accel, False)
    accel.wait_for_everyone()
    # rank-0 vis (small N=5) so debug-samples panel populates immediately.
    # Other ranks wait at the barrier below; with N=5 this stays under
    # NCCL's 600 s watchdog even on slow Lustre.
    if accel.is_main_process:
        try:
            from scripts.util.vis import visualize
            for cp in cache_paths:
                ds_name = Path(cp).name
                per_ds_exp = exp_dir / f'_vis_per_{ds_name}'
                per_ds_exp.mkdir(exist_ok=True)
                visualize(
                    accel.unwrap_model(model), per_ds_exp, cp, 0,
                    ds_kw=dict(ds_kw),
                    n=5, device=accel.device,
                    amp_dtype=torch.float16, log=log)
                if cml_logger is not None:
                    vis_dir = per_ds_exp / f'vis_ep000'
                    for p in sorted(vis_dir.glob('*.png')):
                        try:
                            cml_logger.report_image(
                                f'vis_ep__{ds_name}', p.stem,
                                iteration=0, local_path=str(p))
                        except Exception:
                            pass
        except Exception as _e:
            log(f"preflight vis skipped: {_e}")
    accel.wait_for_everyone()
    log("preflight OK — entering train loop")

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

        # rank-0 BA pose-residual eval. All other ranks idle at the barrier
        # below; keep K small (default 4 seeds × 20 inst × 4 sub-crops = 320
        # tiles per seed) so this stays under a few seconds on V100.
        ba_metrics = None
        if accel.is_main_process and ba_eval_state is not None:
            try:
                ba_metrics = _run_ba_eval(accel.unwrap_model(model), epoch)
            except Exception as _e:
                log(f"BA eval ep={epoch} skipped: {_e!r}")
        accel.wait_for_everyone()
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
            if ba_metrics is not None:
                for m in ba_metrics:
                    log(f"  BA[{m['label']}]  ω={m['omega_deg']:.4f}° "
                        f"({m['omega_px']:.3f}px@fx)  t={m['t_m']:.4f}m")
            if va_nll < best_val:
                best_val = va_nll
                # unwrap before state_dict so keys don't carry 'module.' prefix
                accel.save(accel.unwrap_model(model).state_dict(), ckpt)
                log(f"  ↳ saved (val_nll={best_val:.4f})")
                # Push best.pt to ClearML as an OutputModel so the web UI Models
                # tab carries it (and downstream eval/serve scripts can pull by
                # task id instead of mounting the experiments dir).
                if cml_logger is not None:
                    try:
                        from clearml import OutputModel as _OM, Task as _T
                        _task = _T.current_task()
                        if _task is not None:
                            _om = _OM(task=_task, name=f"{cfg['name']}_best",
                                      framework='PyTorch')
                            _om.update_weights(str(ckpt), upload_uri=None,
                                               iteration=epoch, auto_delete_file=False)
                            # Also expose under the Artifacts tab so people who
                            # navigate Task → Artifacts (not Models) still find
                            # the weights. delete_after_upload=False keeps the
                            # local file for warm-start.
                            _task.upload_artifact(
                                'best_model.pt', artifact_object=str(ckpt),
                                delete_after_upload=False)
                    except Exception as _e:
                        log(f"  ↳ ClearML OutputModel upload skipped: {_e}")
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
                    if ba_metrics is not None:
                        for m in ba_metrics:
                            rs('ba_omega_deg', m['label'], iteration=epoch, value=m['omega_deg'])
                            rs('ba_omega_px',  m['label'], iteration=epoch, value=m['omega_px'])
                            rs('ba_t_m',       m['label'], iteration=epoch, value=m['t_m'])
                            if m.get('vis_path') is not None:
                                try:
                                    cml_logger.report_image(
                                        'ba_overlay', m['label'],
                                        iteration=epoch, local_path=str(m['vis_path']))
                                except Exception:
                                    pass
                except Exception:
                    pass
            # Per-10-epoch debug images (rank-0 only; needs the unwrapped model
            # so forward signature matches the dataset-level call in the util).
            # Per-N-epoch debug images, dataset-split. Class-level lmdb env
            # cache means parent + workers share one env per path → safe.
            if epoch % 10 == 0 or epoch == epochs:
                try:
                    from scripts.util.vis import visualize
                    n_vis = 15 if len(cache_paths) > 1 else 10
                    for cp in cache_paths:
                        ds_name = Path(cp).name
                        per_ds_exp = exp_dir / f'_vis_per_{ds_name}'
                        per_ds_exp.mkdir(exist_ok=True)
                        visualize(
                            accel.unwrap_model(model), per_ds_exp, cp, epoch,
                            ds_kw=dict(ds_kw),
                            n=n_vis, device=accel.device,
                            amp_dtype=torch.float16, log=log)
                        if cml_logger is not None:
                            vis_dir = per_ds_exp / f'vis_ep{epoch:03d}'
                            for p in sorted(vis_dir.glob('*.png')):
                                try:
                                    cml_logger.report_image(
                                        f'vis_ep__{ds_name}', p.stem,
                                        iteration=epoch, local_path=str(p))
                                except Exception:
                                    pass
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
    ap.add_argument('--convnext-n-blocks', type=int, default=None,
                    help='ConvNeXt blocks per stage (default 2). 4 ~= +0.7 M params, +30%% forward.')
    ap.add_argument('--convnext-fine-d', type=int, default=None,
                    help='ConvNeXt fine-stage channels (default = d). Pass 96 for 64→96→128 graduated stem→fine→coarse expansion.')
    ap.add_argument('--convnext-stem-d', type=int, default=None,
                    help='ConvNeXt stem channels (default 64).')
    ap.add_argument('--compile', action='store_true',
                    help='wrap model with torch.compile(mode="reduce-overhead") '
                         'BEFORE accel.prepare so DDP wraps the compiled graph. '
                         'First 1-2 steps slow (CUDA graph capture).')
    ap.add_argument('--deform-mode', default=None)
    ap.add_argument('--train-size', type=int, default=None)
    ap.add_argument('--val-size', type=int, default=None)
    ap.add_argument('--no-frustum', action='store_true',
                    help='disable FrustumLocalEncoder (ablation vs CFG default)')
    ap.add_argument('--ba-eval-start-ep', type=int, default=None,
                     help='enable BA pose-residual eval starting at this epoch (default: 5)')
    ap.add_argument('--ba-eval-every', type=int, default=None,
                     help='run BA eval every N epochs (default: 1). '
                          'Always runs on start_ep and on the last epoch.')
    ap.add_argument('--ba-eval-n-seeds', type=int, default=None)
    ap.add_argument('--ba-eval-n-inst',  type=int, default=None)
    ap.add_argument('--ba-eval-rot-deg', type=float, default=None,
                     help='rot perturbation magnitude for BA eval; defaults to '
                          'train rot magnitude so the metric is in-distribution')
    ap.add_argument('--ba-eval-t-m',     type=float, default=None,
                     help='translation perturbation magnitude for BA eval')
    ap.add_argument('--ba-eval-cs',  type=int, default=None,
                     help='BA eval crop side (orig-cam px). Default 256 — '
                          'decoupled from training img_size so the shared-GN '
                          'Fisher info per tile stays large even at img_size=128.')
    ap.add_argument('--ba-eval-npi', type=int, default=None,
                     help='BA eval n_per_inst (sub-crops per inst). Default 4 '
                          '(quadrant split for cs=256). Use 1 for cs=512.')
    ap.add_argument('--no-ba-eval', action='store_true')
    ap.add_argument('--use-pose-emb', action='store_true',
                    help='enable PoseEmb (effectively log(vfp) bias on Q + img tokens)')
    ap.add_argument('--pose-emb-self-sup', action='store_true',
                    help='self-supervised pose_emb: dataset samples δ = δ1 + δ2 '
                         '(each ±max_offset_m / ±max_rot_deg). δ1 is fed into '
                         'pose_emb as a "known hint"; the network only has to '
                         'regress the δ2-induced reproj residual. With δ1=0 '
                         'this degenerates exactly to legacy calib. Implies '
                         '--use-pose-emb. Stepping stone toward cross-frame.')
    ap.add_argument('--frustum-dense', action='store_true',
                    help='enable FrustumLocalEncoder.forward_dense (cell map with '
                         'zero LiDAR + UV emb for empty cells). With deform_mode=ml '
                         'this becomes a 3rd KV level (coarse, fine, lidar_dense).')
    ap.add_argument('--find-unused-parameters', action='store_true',
                    help='DDP find_unused_parameters=True. Off by default '
                         '(2026-05-05: True caused 22× slowdown). Turn on '
                         'only if DDP AllReduce actually hangs at backward.')
    ap.add_argument('--clearml', action='store_true')
    ap.add_argument('--init-from', default=None,
                    help='warm-start from another exp\'s best_model.pt '
                         '(name only, e.g. km_wv_8gpu_200ep_os4)')
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
    if args.convnext_n_blocks is not None: cfg['convnext_n_blocks'] = args.convnext_n_blocks
    if args.convnext_fine_d   is not None: cfg['convnext_fine_d']   = args.convnext_fine_d
    if args.convnext_stem_d   is not None: cfg['convnext_stem_d']   = args.convnext_stem_d
    if args.compile:  cfg['compile'] = True
    if args.deform_mode is not None: cfg['deform_mode'] = args.deform_mode
    if args.train_size is not None: cfg['train_size'] = args.train_size
    if args.val_size   is not None: cfg['val_size']   = args.val_size
    if args.init_from  is not None: cfg['init_from']  = args.init_from
    if args.no_frustum: cfg['use_frustum'] = False
    if args.use_pose_emb: cfg['use_pose_emb'] = True
    if args.pose_emb_self_sup:
        cfg['use_pose_emb'] = True
        cfg['split_pert']   = True
    if args.frustum_dense: cfg['frustum_dense'] = True
    if args.ba_eval_start_ep is not None: cfg['ba_eval_start_ep'] = args.ba_eval_start_ep
    if args.ba_eval_every    is not None: cfg['ba_eval_every']    = args.ba_eval_every
    if args.ba_eval_n_seeds  is not None: cfg['ba_eval_n_seeds']  = args.ba_eval_n_seeds
    if args.ba_eval_n_inst   is not None: cfg['ba_eval_n_inst']   = args.ba_eval_n_inst
    if args.ba_eval_rot_deg  is not None: cfg['ba_eval_rot_deg']  = args.ba_eval_rot_deg
    if args.ba_eval_t_m      is not None: cfg['ba_eval_t_m']      = args.ba_eval_t_m
    if args.ba_eval_cs       is not None: cfg['ba_eval_cs']       = args.ba_eval_cs
    if args.ba_eval_npi      is not None: cfg['ba_eval_npi']      = args.ba_eval_npi
    if args.no_ba_eval: cfg['ba_eval'] = False
    if args.find_unused_parameters: cfg['find_unused_parameters'] = True

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
