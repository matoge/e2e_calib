"""Train CalibNet2 (CND2) on tiled-cache data — minimal Accelerate DDP trainer.

Forked from train_ps_v3_ddp.py but stripped:
  * model = CalibNet2 (own RoPEPoseEmb, plain cross-attn, single final head)
  * forward signature differs (distorted_uvd / dpose_R / vfp / bucket_uvd)
  * No BA eval hook, no per-dataset breakdown, no warm-start, no ClearML;
    those wire to CalibNetDepth-specific surfaces and aren't worth the
    bloat for the first CND2 run.

Loss: gaussian2d_nll over per_pt[..., :5] vs gt_duv (=true - dist), same as
legacy. CalibNet2 returns (per_pt, W) when use_info_head=True; we ignore W
for the first kick (matches "sigma-head only" baseline).

Launch (DGX2 GPU 3-10 + 15 = 9 processes, fp16):
    CUDA_VISIBLE_DEVICES=3,4,5,6,7,8,9,10,15 \
    /home/hfunaya/.pyenv/versions/3.10.4/bin/python -m accelerate.commands.launch \
        --num_processes=9 --mixed_precision=fp16 \
        scripts/training/train_cnd2_ddp.py \
        --name cnd2_km_os16_50ep \
        --cache /home/hfunaya/cache/kamikado_v3_tiled \
        --epochs 50 --oversample 16 --batch-size 64
"""
import sys, os, pathlib; sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[2]))
import argparse, math, time, torch
import torch.multiprocessing as _tmp
try: _tmp.set_sharing_strategy('file_system')
except Exception: pass
from pathlib import Path
from datetime import datetime

from datasets.pandaset_full import PandaSetCalibDatasetFull, collate_full, collate_pair
from models.calibnet2 import CalibNet2
from models.model_cov import gaussian2d_nll
from torch.utils.data import DataLoader, Subset, ConcatDataset, RandomSampler
import random as _r

from accelerate import Accelerator
from accelerate import DistributedDataParallelKwargs
from accelerate.utils import set_seed
# Local hack: repo has its own `datasets/` package (= datasets.pandaset_full).
# Accelerate's prepare_data_loader does `from datasets import IterableDataset`
# when is_datasets_available() is True; since the repo dir shadows HF datasets
# on sys.path, that import fails. We don't use HF datasets at all → force the
# probe to False before accel.prepare() runs.
import accelerate.utils.imports as _ai
_ai.is_datasets_available = lambda: False
import accelerate.data_loader as _adl
_adl.is_datasets_available = lambda: False

torch.set_float32_matmul_precision("high")


def _R_from_zyx_deg(ypr_deg: torch.Tensor) -> torch.Tensor:
    """Build (B, 3, 3) rotation matrices from intrinsic ZYX Euler in degrees.

    `ypr_deg` is (B, 3) [yaw, pitch, roll]; matches dataset's
    Rotation.from_euler('zyx', ..., degrees=True) convention.
    """
    yaw = ypr_deg[..., 0] * (math.pi / 180.0)
    pit = ypr_deg[..., 1] * (math.pi / 180.0)
    rol = ypr_deg[..., 2] * (math.pi / 180.0)
    cy, sy = torch.cos(yaw), torch.sin(yaw)
    cp, sp = torch.cos(pit), torch.sin(pit)
    cr, sr = torch.cos(rol), torch.sin(rol)
    z = torch.zeros_like(yaw); o = torch.ones_like(yaw)
    Rz = torch.stack([cy, -sy, z, sy, cy, z, z, z, o], dim=-1).view(*yaw.shape, 3, 3)
    Ry = torch.stack([cp, z, sp, z, o, z, -sp, z, cp], dim=-1).view(*yaw.shape, 3, 3)
    Rx = torch.stack([o, z, z, z, cr, -sr, z, sr, cr], dim=-1).view(*yaw.shape, 3, 3)
    return Rz @ Ry @ Rx


def epoch_loop_pair(model, loader, optimizer, accel: Accelerator, train: bool,
                    img_size: int):
    """Cross-frame pair epoch.

    For each batch (dict from collate_pair): forward A's LiDAR Q through the
    shared block stack on KV_A, apply RoPE(R_AB) once, then continue on
    KV_B. Loss = NLL of predicted Δ vs (true_uv_in_B - dist_uv_in_B), where:
      - dist_uv = HAT projection of A's points in B image (= dist_uvd_B)
      - true_uv = GT projection of A's points in B image (= true_uvd_B)
    Both come from the dataset's pair builder which already aligns A↔B
    indices when same_frame_self_sup=True. For genuine cross-frame pairs we
    rely on the same-coordinate alignment that build_window emits.
    """
    model.train(train)
    total_nll, total_mse, n = 0.0, 0.0, 0
    _t_start = time.time()
    _last_log_step = 0
    for batch in loader:
        A = batch['A']; B = batch['B']
        dpose_AB = batch['dpose_AB']                          # (B, 6) GT [tx,ty,tz, ypr_deg]
        # A side: image + LiDAR Q.
        imgs_A, _trueA, distA, padA, vfpA, buA, bvA = A[:7]
        imgs_A = imgs_A.float().div_(255.0)
        # B side: image + bucket KV. trueB / distB describe the SAME A points
        # under POSE_GT (target) and POSE_HAT (sampling anchor) on B image.
        imgs_B, trueB, distB, padB, vfpB, buB, bvB = B[:7]
        imgs_B = imgs_B.float().div_(255.0)
        pertB = B[7]                                          # (B, 8) [tx,ty,tz, ypr, dfx, dfy]
        # PoseEmb input = POSE_HAT_AB = POSE_GT_AB ⊕ ε.  ε is what the dataset
        # sampled in pertB[:, :6].  Compose as: ypr_HAT = ypr_GT + ypr_pert
        # (small-angle approx; exact composition is built dataset-side already
        # via POSE_HAT projection — we just need the network to know HAT).
        ypr_GT  = dpose_AB[..., 3:6]
        ypr_eps = pertB[..., 3:6].to(ypr_GT)
        ypr_HAT = ypr_GT + ypr_eps
        R_AB = _R_from_zyx_deg(ypr_HAT).to(imgs_A.device, dtype=imgs_A.dtype)

        # CalibNet2 input on Q: A's [u, v, d, intensity] (drop is_obj col 3).
        point_in_A = torch.cat([distA[..., :3], distA[..., 4:5]], dim=-1)

        # Dispatch through model.forward(...) so DDP's grad-sync hook fires.
        # CalibNet2.forward routes to forward_cross_frame when mode='cross'.
        out = model(
            imgs_A, point_in_A,
            mode='cross', image_B=imgs_B, R_AB=R_AB,
            vfp=vfpA, vfp_B=vfpB,
            bucket_uvd=buA, bucket_valid=bvA,
            bucket_uvd_B=buB, bucket_valid_B=bvB,
            key_padding_mask=padA,
        )
        per_pt = out[0] if isinstance(out, tuple) else out

        # Target = δ between POSE_GT and POSE_HAT B-projection of A's points.
        # build_window emits A and B in lex-deterministic sub_idx order; in
        # same_frame_self_sup mode the leading min(N_A, N_B) tokens describe
        # the same world points.  Crop to that prefix; mask via padA up to it.
        Nmin = min(per_pt.shape[1], trueB.shape[1])
        per_pt_c = per_pt[:, :Nmin]
        gt = trueB[:, :Nmin, :2] - distB[:, :Nmin, :2]
        valid = ~(padA[:, :Nmin] | padB[:, :Nmin])
        loss = gaussian2d_nll(per_pt_c[valid], gt[valid])
        if train:
            optimizer.zero_grad(set_to_none=True)
            accel.backward(loss)
            accel.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
        with torch.no_grad():
            err = (per_pt_c[valid][..., :2].float() - gt[valid]).norm(dim=-1)
            total_mse += err.mean().item()
        total_nll += loss.item(); n += 1
        if train and accel.is_main_process and (n - _last_log_step >= 25):
            _dt = time.time() - _t_start
            sps_per  = n * imgs_A.shape[0] / _dt if _dt > 0 else 0
            sps_glob = sps_per * accel.num_processes
            print(f"  step {n}  loss={loss.item():+.3f}  "
                  f"sps/rank={sps_per:.0f}  sps(global)={sps_glob:.0f}", flush=True)
            _last_log_step = n
    return (total_nll / max(n, 1), total_mse / max(n, 1))


def epoch_loop(model, loader, optimizer, accel: Accelerator, train: bool,
               img_size: int):
    model.train(train)
    total_nll, total_mse, n = 0.0, 0.0, 0
    _t_start = time.time()
    _last_log_step = 0
    for batch in loader:
        imgs, true_uvd, dist_uvd, pad_mask, vfp, bucket_uvd, bucket_valid = batch[:7]
        imgs = imgs.float().div_(255.0)
        gt   = true_uvd[..., :2] - dist_uvd[..., :2]
        # CalibNet2.use_intensity is True by default; pass [u,v,d,intensity].
        # dist_uvd cols: [u, v, d, is_obj, intensity] → drop is_obj (col 3).
        point_in = torch.cat([dist_uvd[..., :3], dist_uvd[..., 4:5]], dim=-1)
        out = model(imgs, point_in,
                    dpose_R=None, vfp=vfp,
                    bucket_uvd=bucket_uvd, bucket_valid=bucket_valid,
                    key_padding_mask=pad_mask)
        # use_info_head=False here → out is per_pt only.
        per_pt = out[0] if isinstance(out, tuple) else out
        valid  = ~pad_mask
        loss   = gaussian2d_nll(per_pt[valid], gt[valid])
        if train:
            optimizer.zero_grad(set_to_none=True)
            accel.backward(loss)
            accel.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
        with torch.no_grad():
            err = (per_pt[valid][..., :2].float() - gt[valid]).norm(dim=-1)
            total_mse += err.mean().item()
        total_nll += loss.item(); n += 1
        if train and accel.is_main_process and (n - _last_log_step >= 25):
            _dt = time.time() - _t_start
            sps_per  = n * imgs.shape[0] / _dt if _dt > 0 else 0
            sps_glob = sps_per * accel.num_processes
            print(f"  step {n}  loss={loss.item():+.3f}  "
                  f"sps/rank={sps_per:.0f}  sps(global)={sps_glob:.0f}", flush=True)
            _last_log_step = n
    return (total_nll / max(n, 1), total_mse / max(n, 1))


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--name', required=True)
    p.add_argument('--cache', required=True,
                   help='comma-separated v3-tiled cache path(s)')
    p.add_argument('--epochs', type=int, default=50)
    p.add_argument('--batch-size', type=int, default=64)
    p.add_argument('--img-size', type=int, default=128)
    p.add_argument('--lr', type=float, default=1e-3)
    p.add_argument('--lr-min', type=float, default=1e-6)
    p.add_argument('--oversample', type=int, default=16)
    p.add_argument('--rot-deg', type=float, default=1.5)
    p.add_argument('--t-m', type=float, default=0.20)
    p.add_argument('--min-crop-px', type=int, default=256)
    p.add_argument('--max-crop-px', type=int, default=512)
    p.add_argument('--grid-n', type=int, default=16)
    p.add_argument('--workers', type=int, default=8)
    p.add_argument('--prefetch', type=int, default=4)
    p.add_argument('--n-iter', type=int, default=3)
    p.add_argument('--n-heads', type=int, default=4)
    p.add_argument('--val-fraction', type=float, default=0.1)
    p.add_argument('--split-seed', type=int, default=42)
    p.add_argument('--use-info-head', action='store_true',
                   help='enable InfoHead2x2 (still ignored by NLL loss; for ckpt only)')
    p.add_argument('--u-band', type=str, default='',
                   help='comma-separated <cache_path>:<frac> overrides for u_band '
                        '(central horizontal pivot band). 0=disabled, 0.8=keep '
                        'central 80%. Example: '
                        '"/home/hfunaya/cache_v5/tss4_v3_full_iter2kb4_yaw3:0.8"')
    p.add_argument('--frame-stride', type=str, default='',
                   help='comma-separated <cache_path>:<int> per-cache frame_stride '
                        'overrides (sub-sample fnames). 1=keep all, 10=1/10. '
                        'Example: "/home/hfunaya/cache_v5/waymo_v3_full:10"')
    p.add_argument('--per-cache-oversample', type=str, default='',
                   help='comma-separated <cache_path>:<int> per-cache oversample '
                        'overrides. Falls back to --oversample. Example: '
                        '"/home/hfunaya/cache_v5/kamikado_v3_full:8"')
    p.add_argument('--pair-stride', type=int, default=10,
                   help='In pair-mode, max |Δframe| sampled per anchor. The '
                        'dataset emits all (i_A, i_A+δ) pairs with δ ∈ '
                        '[-S..-1, +1..+S] within the same (scene, cam). '
                        'Default 10 (= ±1 sec at 10 Hz).')
    p.add_argument('--pair-mode', action='store_true',
                   help='cross-frame pair training. Dataset emits (A, B, '
                        'dpose_AB) tuples; trainer calls '
                        'CalibNet2.forward_cross_frame(A, B, R_AB) and '
                        'targets the GT projection of A points in B image.')
    p.add_argument('--clearml', action='store_true',
                   help='register a ClearML Task with cfg + why + git context')
    p.add_argument('--clearml-project', type=str, default='e2e_calib/calib',
                   help='ClearML project namespace')
    p.add_argument('--why', type=str, default='',
                   help='WHY blob: rationale, hypothesis, expected outcome '
                        '(stored in ClearML task comment, free-form prose).')
    args = p.parse_args()

    # find_unused_parameters=True: in pair_mode cross-frame path the legacy
    # entry-time pose_emb / info_head / etc. don't see gradients in some
    # batches; without this DDP raises a "reduction not finished" error.
    accel = Accelerator(kwargs_handlers=[
        DistributedDataParallelKwargs(find_unused_parameters=True),
    ])
    set_seed(args.split_seed + accel.process_index)

    # ClearML init (rank-0 only, BEFORE main loop so cml_logger lookups work)
    if args.clearml and int(os.environ.get('RANK', '0')) == 0:
        try:
            from scripts.util.clearml_context import init_with_context
            init_with_context(
                project=args.clearml_project, name=args.name,
                cfg={k: v for k, v in vars(args).items()
                     if not callable(v) and not k.startswith('_')},
                why=args.why,
                baseline={'name': 'cnd2_km_os16_50ep_dgx2_9gpu',
                          'metric': 'val_nll', 'value': 2.4666})
        except Exception as _e:
            print(f'[clearml init failed] {_e}', flush=True)

    exp_dir = Path("experiments") / args.name
    if accel.is_main_process:
        exp_dir.mkdir(parents=True, exist_ok=True)
        log_path = exp_dir / "train.log"
        log_path.write_text("")
        # Save full config (every CLI arg) for reproducibility
        (exp_dir / "config.py").write_text(
            "# auto-generated by train_cnd2_ddp.py at run start\n"
            "CFG = dict(\n" +
            "".join(f"    {k:<20}= {v!r},\n"
                    for k, v in vars(args).items()) + ")\n")
    accel.wait_for_everyone()

    def log(msg):
        if not accel.is_main_process: return
        ts = datetime.now().strftime("%H:%M:%S")
        line = f"{ts}  {msg}"
        print(line, flush=True)
        with open(exp_dir / "train.log", "a") as f: f.write(line + "\n")

    log(f"accelerate: num_processes={accel.num_processes} "
        f"mixed_precision={accel.mixed_precision} device={accel.device}")

    # --- dataset(s) ---
    ds_kw = dict(img_size=args.img_size,
                 max_offset_m=args.t_m, max_rot_deg=args.rot_deg,
                 min_crop_px=args.min_crop_px, max_crop_px=args.max_crop_px,
                 grid_n=args.grid_n, oversample=args.oversample,
                 split_pert=False,
                 pair_mode=bool(args.pair_mode),
                 pair_stride=int(args.pair_stride))
    cache_paths = [s.strip() for s in args.cache.split(',') if s.strip()]
    # Per-cache u_band override
    ub_map = {}
    if getattr(args, 'u_band', ''):
        for tok in args.u_band.split(','):
            tok = tok.strip()
            if not tok: continue
            k, v = tok.rsplit(':', 1)
            ub_map[k.strip()] = float(v)
    # Per-cache oversample override
    os_map = {}
    if getattr(args, 'per_cache_oversample', ''):
        for tok in args.per_cache_oversample.split(','):
            tok = tok.strip()
            if not tok: continue
            k, v = tok.rsplit(':', 1)
            os_map[k.strip()] = int(v)
    tr_parts, va_parts = [], []
    for cp in cache_paths:
        ub = ub_map.get(cp, 0.0)
        os_i = os_map.get(cp, args.oversample)
        kw = {**ds_kw, 'oversample': os_i}
        tr = PandaSetCalibDatasetFull(cp, split='train', u_band=ub, **kw)
        va = PandaSetCalibDatasetFull(cp, split='val',
                                       center_band=0.5, u_band=ub, **kw)
        log(f"  [{cp}] train={len(tr)} val={len(va)} (os={os_i}) u_band={ub}")
        tr_parts.append(tr); va_parts.append(va)
    tr_full = ConcatDataset(tr_parts) if len(tr_parts) > 1 else tr_parts[0]
    va_full = ConcatDataset(va_parts) if len(va_parts) > 1 else va_parts[0]

    full_ds = ConcatDataset([tr_full, va_full])
    # __len__ now equals frame count (oversample handled inside __getitem__).
    # Each idx = 1 frame → list of `oversample` samples in collate. Standard
    # shuffle works at the frame level, so the worker decodes once per frame
    # and slices `oversample` crops out of it.
    idxs = list(range(len(full_ds)))
    _r.Random(args.split_seed).shuffle(idxs)
    n_val = int(len(idxs) * args.val_fraction)
    val_idxs, train_idxs = idxs[:n_val], idxs[n_val:]
    train_ds = Subset(full_ds, train_idxs)
    val_ds   = Subset(full_ds, val_idxs)
    log(f"frame-level split: train={len(train_ds)} val={len(val_ds)} frames")

    nw = args.workers
    kw = dict(num_workers=nw, pin_memory=True,
              persistent_workers=(nw > 0),
              prefetch_factor=args.prefetch if nw > 0 else None,
              collate_fn=collate_pair if args.pair_mode else collate_full,
              multiprocessing_context='spawn' if nw > 0 else None)
    val_nw = min(4, nw)
    val_kw = dict(num_workers=val_nw, pin_memory=True,
                  persistent_workers=(val_nw > 0),
                  prefetch_factor=args.prefetch if val_nw > 0 else None,
                  collate_fn=collate_pair if args.pair_mode else collate_full,
                  multiprocessing_context='spawn' if val_nw > 0 else None)
    # batch_size here = number of FRAMES per batch; collate expands each
    # frame to its `oversample` samples → effective batch = batch_size * os.
    # When configuring --batch-size think in frames, not samples.
    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, **kw)
    val_loader   = DataLoader(val_ds,   batch_size=args.batch_size, shuffle=False, **val_kw)

    # --- model ---
    model = CalibNet2(d=128, img_size=args.img_size, in_channels=3,
                      use_intensity=True, frustum_grid_n=args.grid_n,
                      n_iter=args.n_iter, n_heads=args.n_heads,
                      d_scalar=8, n_type1=40,
                      use_info_head=args.use_info_head)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-3)

    model, optimizer, train_loader, val_loader = accel.prepare(
        model, optimizer, train_loader, val_loader)

    log(f"params: {sum(p.numel() for p in model.parameters())/1e6:.2f}M  "
        f"(world_size={accel.num_processes}, bs/rank={args.batch_size}, "
        f"global_bs={args.batch_size*accel.num_processes})")

    epochs   = args.epochs
    lr_min_r = args.lr_min / args.lr
    def lr_lambda(e):
        if e < 5: return (e + 1) / 5
        t = (e - 5) / max(1, epochs - 5)
        return lr_min_r + (1 - lr_min_r) * 0.5 * (1 + math.cos(math.pi * t))
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)
    best_val = float("inf")
    ckpt = exp_dir / "best_model.pt"
    t0 = time.time()

    # ClearML logger handle (rank-0 only)
    cml_logger = None
    cml_task = None
    if args.clearml and accel.is_main_process:
        try:
            from clearml import Task as _ClearMLTask
            cml_task = _ClearMLTask.current_task()
            if cml_task is not None:
                cml_logger = cml_task.get_logger()
        except Exception as _e:
            log(f"[clearml] logger init failed: {_e}")

    _epoch_fn = epoch_loop_pair if args.pair_mode else epoch_loop
    for ep in range(epochs):
        ep_t = time.time()
        tr_nll, tr_mse = _epoch_fn(model, train_loader, optimizer, accel, True,
                                    args.img_size)
        with torch.no_grad():
            va_nll, va_mse = _epoch_fn(model, val_loader, optimizer, accel, False,
                                        args.img_size)
        scheduler.step()
        if accel.is_main_process:
            elapsed = time.time() - t0
            log(f"ep{ep+1:03d}/{epochs}  "
                f"tr_nll={tr_nll:.4f} tr_mse={tr_mse:.3f}  "
                f"va_nll={va_nll:.4f} va_mse={va_mse:.3f}  "
                f"lr={scheduler.get_last_lr()[0]:.2e}  "
                f"ep_t={time.time()-ep_t:.1f}s  total={elapsed/60:.1f}min")
            if cml_logger is not None:
                rs = cml_logger.report_scalar
                rs(title='loss/nll', series='train', value=tr_nll, iteration=ep+1)
                rs(title='loss/nll', series='val',   value=va_nll, iteration=ep+1)
                rs(title='loss/mse', series='train', value=tr_mse, iteration=ep+1)
                rs(title='loss/mse', series='val',   value=va_mse, iteration=ep+1)
                rs(title='lr', series='lr',
                   value=scheduler.get_last_lr()[0], iteration=ep+1)
            if va_nll < best_val:
                best_val = va_nll
                accel.save(accel.unwrap_model(model).state_dict(), ckpt)
                log(f"  ↳ best val_nll={best_val:.4f}  saved {ckpt}")
                # Upload best.pt as ClearML OutputModel + pair config.py
                if cml_task is not None:
                    try:
                        from clearml import OutputModel as _OM
                        _om = _OM(task=cml_task, name=f'{args.name}_best')
                        _om.update_weights(weights_filename=str(ckpt))
                        cfg_p = exp_dir / 'config.py'
                        if cfg_p.exists():
                            cml_task.upload_artifact('config.py',
                                                     artifact_object=str(cfg_p),
                                                     auto_pickle=False)
                    except Exception as _e:
                        log(f"  ↳ ClearML upload skipped: {_e}")
        accel.wait_for_everyone()


if __name__ == '__main__':
    main()
