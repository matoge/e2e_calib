"""Phase 1.c-unlock — backbone UNFROZEN, DDP across 12 GPUs.

Same contract as overfit_2dof_ba_multiframe.py, but:
  - ALL of CalibNetDepth is trainable (info_head + backbone).
  - HF Accelerate DDP wrap; rank-0 gates eval / logging / saving.
  - Train: KAMIKADO train tiles, fresh (frame, ωx, ωy) each step.
  - Held-out eval: fixed (val frame, pert) batch, rank-0 only.

Launch:
  CUDA_VISIBLE_DEVICES=4,5,6,7,8,9,10,11,12,13,14,15 \\
  accelerate launch --num_processes=12 --mixed_precision=no \\
      scripts/_debug/overfit_2dof_ba_multiframe_unlock_ddp.py \\
      --n-steps 600 --batch 16 --n-eval 512 --tag unlock12

Note: --batch is PER-RANK; global batch = batch * num_processes.
"""
from __future__ import annotations
import argparse
import sys
import time
from pathlib import Path

import numpy as np
import torch

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))

from datasets.pandaset_full import PandaSetCalibDatasetFull, collate_full
from models.model_depth import CalibNetDepth
from scripts.ba.ba_torch import (
    solve_pinhole_xyz, make_info_from_sigma_rho,
)

from accelerate import Accelerator
from accelerate.utils import set_seed


CACHE       = Path.home() / 'cache' / 'kamikado_v3_tiled'
CKPT        = REPO / 'experiments' / 'km_wv_wm_dgx2_n4_img128_8gpu_HEAD' / 'best_model.pt'
EXP_CFG_PATH = REPO / 'experiments' / 'km_wv_wm_dgx2_n4_img128_8gpu_HEAD' / 'config.py'
OUT         = REPO / 'scripts' / '_debug' / '_outputs' / 'overfit_2dof_ba_multiframe_unlock_ddp'

DEFAULT_N_STEPS = 600
DEFAULT_BATCH   = 16   # per-rank
DEFAULT_N_EVAL  = 512
LR              = 1e-4   # backbone unfrozen → lower than 1e-3 frozen-head LR
LOG_EVERY       = 25
EVAL_EVERY      = 50
BA_N_ITER       = 3
DAMPING         = 1e-3
SEED            = 7

DOF_PRESETS = {
    '2dof':  ['omega_x', 'omega_y'],
    '2dof_t': ['tx', 'ty'],   # XY translation only — easy uv-shift sanity
    '3dof':  ['omega_x', 'omega_y', 'omega_z'],
    '4dof':  ['omega_x', 'omega_y', 'tx', 'ty'],
    '6dof':  ['omega_x', 'omega_y', 'omega_z', 'tx', 'ty', 'tz'],
    '10dof': ['omega_x', 'omega_y', 'omega_z', 'tx', 'ty', 'tz',
              'dfx', 'dfy', 'dcx', 'dcy'],
}


def _load_cfg() -> dict:
    src = EXP_CFG_PATH.read_text()
    ns: dict = {}
    exec(src, ns, ns)
    return ns['CFG']


def _build_model(cfg: dict) -> CalibNetDepth:
    return CalibNetDepth(
        img_size=cfg['img_size'],
        in_channels=cfg['in_channels'],
        n_layers=cfg['n_layers'],
        self_first=cfg.get('self_first', False),
        use_convnext=cfg.get('use_convnext', True),
        use_frustum=cfg.get('use_frustum', True),
        deform_mode=cfg.get('deform_mode', 'sl'),
        convnext_n_blocks=cfg.get('convnext_n_blocks', 2),
        convnext_fine_d=cfg.get('convnext_fine_d', None),
        convnext_stem_d=cfg.get('convnext_stem_d', None),
        use_info_head=True,
    )


def _local_to_orig_scale(cs: torch.Tensor, S: float) -> torch.Tensor:
    """Per-sample (cs/S) factor that converts local-128px → original-camera px.
       cs : (B,) crop size in original-camera px;  S = local network px (e.g. 128).
       returns (B, 1, 1) for broadcast against (B, N, 2) duv tensors.
    """
    return (cs / float(S)).reshape(-1, 1, 1)


def _sample_one(ds, idx, rng, *, rot_deg, t_m, dfx_pct, dfy_pct):
    """Sample one perturbation. ranges are *half-ranges* (uniform [-r, +r])."""
    if rot_deg > 0.0:
        omega_x = float(rng.uniform(-rot_deg, rot_deg))
        omega_y = float(rng.uniform(-rot_deg, rot_deg))
        omega_z = float(rng.uniform(-rot_deg, rot_deg))
    else:
        omega_x = omega_y = omega_z = 0.0
    # apply_perturbation_explicit ypr_deg = [yaw=ω_z, pitch=ω_y, roll=ω_x]?
    # The 2-DoF caller used ypr=[0, oy, ox] for (ω_y, ω_x). Keep that mapping
    # and add ω_z in slot 0.
    ypr = np.array([omega_z, omega_y, omega_x], dtype=np.float64)
    if t_m > 0.0:
        t = (rng.uniform(-1.0, 1.0, size=3) * t_m).astype(np.float64)
    else:
        t = np.zeros(3, dtype=np.float64)
    if dfx_pct > 0.0:
        dfx = float(rng.uniform(-dfx_pct, dfx_pct))
    else:
        dfx = 0.0
    if dfy_pct > 0.0:
        dfy = float(rng.uniform(-dfy_pct, dfy_pct))
    else:
        dfy = 0.0
    win = ds.apply_perturbation_explicit(idx, t, ypr,
                                          dfx_pct=dfx, dfy_pct=dfy)
    return win


def _build_batch(ds, idx_iter, B, rng, *, rot_deg, t_m, dfx_pct, dfy_pct,
                 max_tries=64):
    windows = []
    tries = 0
    while len(windows) < B and tries < max_tries * B:
        i = idx_iter()
        win = _sample_one(ds, i, rng, rot_deg=rot_deg, t_m=t_m,
                          dfx_pct=dfx_pct, dfy_pct=dfy_pct)
        tries += 1
        if win is None:
            continue
        windows.append(win)
    if len(windows) < B:
        return None
    return collate_full(windows)


def _move_batch(batch, device):
    return [t.to(device) if torch.is_tensor(t) else t for t in batch]


def _forward_with_q(model, imgs_u8, dist_uvd, pad_mask, vfp,
                    bucket_uvd, bucket_valid, *, no_grad: bool):
    """Forward + capture per-query feature q (input to info_head.mlp[0]).

    Returns (per_pt, q). q is needed because info_head is called twice:
    once inside forward (W_learn used downstream), once for diagnostics.

    For training we set no_grad=False so the backbone graph is kept.
    """
    captured = {}
    real_model = model.module if hasattr(model, 'module') else model

    def _hook(_module, _inputs, _output):
        # detach OK for diagnostics path; but during train we want the
        # actual W_learn from the live forward graph, so we use the
        # captured q only for re-evaluation in eval mode.
        captured['q'] = _inputs[0].detach()

    h = real_model.info_head.mlp[0].register_forward_hook(_hook)
    imgs = imgs_u8.float().div(255.0)
    point_in = torch.cat([dist_uvd[..., :3], dist_uvd[..., 4:5]], dim=-1)
    if no_grad:
        with torch.no_grad():
            out = model(imgs, point_in, key_padding_mask=pad_mask, vfp=vfp,
                        bucket_uvd=bucket_uvd, bucket_valid=bucket_valid)
    else:
        out = model(imgs, point_in, key_padding_mask=pad_mask, vfp=vfp,
                    bucket_uvd=bucket_uvd, bucket_valid=bucket_valid)
    h.remove()
    per_pt, _ = out
    return per_pt, captured['q']


def _prep_inputs(cfg, true_uvd, dist_uvd, pad_mask,
                  pts_cam_orig, duv_orig, K_orig, cs):
    """Build solver inputs in the ORIGINAL-camera frame (m + orig px + parent K).

    Returns
        valid       : (B, N) bool         — same as ~pad_mask
        duv_oracle_local : (B, N, 2)      — local-128px Δuv (for per-pt loss)
        duv_oracle_orig  : (B, N, 2)      — orig-px Δuv (for solver target)
        P0_orig     : (B, N, 3) m, parent-cam frame XYZ (cleared on padding)
        K_orig      : (B, 3, 3) parent intrinsics
        cs          : (B,) crop size in original-camera px
    """
    valid = ~pad_mask
    pad_full = ~valid

    # local-px Δuv (still useful for per-pt loss; the network predicts it)
    duv_oracle_local = (true_uvd[..., :2] - dist_uvd[..., :2]).detach()

    # orig-px Δuv comes straight from the dataset — already tile-invariant
    duv_oracle_orig = duv_orig.detach().clone()

    P0_orig = pts_cam_orig.detach().clone()
    if pad_full.any():
        duv_oracle_local = duv_oracle_local.clone()
        duv_oracle_local[pad_full] = 0.0
        duv_oracle_orig[pad_full] = 0.0
        # Z=1.0 fill for padded slots avoids divide-by-zero in J; gn_step
        # zeros these rows via `valid=` anyway.
        P0_orig[pad_full] = torch.tensor(
            [0.0, 0.0, 1.0], dtype=P0_orig.dtype, device=P0_orig.device)
    return valid, duv_oracle_local, duv_oracle_orig, P0_orig, K_orig, cs


def _solve_target(P0_orig, duv_oracle_orig, valid, K_orig, dofs):
    B, N = P0_orig.shape[:2]
    W_eye = torch.eye(2, device=P0_orig.device,
                       dtype=P0_orig.dtype).expand(B, N, 2, 2)
    with torch.no_grad():
        delta, _ = solve_pinhole_xyz(
            P0_orig, duv_oracle_orig, W_eye, K_orig, dofs,
            valid=valid, n_iter=BA_N_ITER, damping=DAMPING,
        )
    return delta.detach()


def _solve_with(P0_orig, duv_orig, W_orig, K_orig, dofs, valid):
    return solve_pinhole_xyz(
        P0_orig, duv_orig, W_orig, K_orig, dofs,
        valid=valid, n_iter=BA_N_ITER, damping=DAMPING,
    )


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--n-steps', type=int, default=DEFAULT_N_STEPS)
    ap.add_argument('--batch',   type=int, default=DEFAULT_BATCH,
                    help='per-rank batch')
    ap.add_argument('--n-eval',  type=int, default=DEFAULT_N_EVAL)
    ap.add_argument('--tag',     type=str, default=None)
    ap.add_argument('--lr',      type=float, default=LR)
    ap.add_argument('--dofs', type=str, default='2dof',
                    choices=list(DOF_PRESETS.keys()),
                    help='which DoFs to optimise / supervise')
    ap.add_argument('--rot-deg', type=float, default=0.30,
                    help='per-axis rotation half-range (deg). Used for all '
                         'rotation DoFs in the chosen preset.')
    ap.add_argument('--t-m', type=float, default=0.0,
                    help='translation half-range (m). >0 only meaningful '
                         'with 6/10dof.')
    ap.add_argument('--dfx-pct', type=float, default=0.0,
                    help='focal x perturbation half-range (fraction). '
                         'Only used by 10dof.')
    ap.add_argument('--dfy-pct', type=float, default=0.0,
                    help='focal y perturbation half-range (fraction). '
                         'Only used by 10dof.')
    ap.add_argument('--loss', type=str, default='mse',
                    choices=['mse', 'nll'],
                    help='mse = ||δ-δ_gt||²;  '
                         'nll = ½(δ-δ_gt)^T H (δ-δ_gt) − ½ log det H, '
                         'where H = J^T W J + λI is the closed-form '
                         'information matrix of the GN solve.')
    ap.add_argument('--freeze-backbone', action='store_true',
                    help='if set, only info_head is trained — backbone '
                         'AND per-point Δuv/σ head stay at the σ-baseline '
                         'ckpt. Equivalent to Phase 1.c-frozen but with '
                         'configurable --dofs / --loss.')
    ap.add_argument('--perpt-weight', type=float, default=0.0,
                    help='if >0, also supervise per-anchor Δuv against '
                         'the analytic oracle: '
                         'L_total = L_pose + λ · mean((duv_pred - duv_oracle)²) '
                         'on valid anchors. Anchors W = info_head output is '
                         'untouched. λ=0 = pose loss only (1.d-style).')
    args = ap.parse_args()

    accel = Accelerator()
    set_seed(SEED + accel.process_index)
    rng = np.random.RandomState(SEED + accel.process_index)
    device = accel.device

    out_dir = OUT if args.tag is None else (OUT.parent / f'overfit_2dof_ba_multiframe_unlock_ddp_{args.tag}')
    if accel.is_main_process:
        out_dir.mkdir(parents=True, exist_ok=True)

    DOFS = DOF_PRESETS[args.dofs]
    pert_kw = dict(rot_deg=args.rot_deg, t_m=args.t_m,
                   dfx_pct=args.dfx_pct, dfy_pct=args.dfy_pct)

    cfg = _load_cfg()
    if accel.is_main_process:
        print(f"[mfu] cfg: img={cfg['img_size']} layers={cfg['n_layers']} "
              f"deform={cfg.get('deform_mode')} convnext={cfg.get('use_convnext')}")
        print(f"[mfu] DoFs={args.dofs}  ({DOFS})  loss={args.loss}  "
              f"rot=±{args.rot_deg}deg  t=±{args.t_m}m  "
              f"dfx=±{args.dfx_pct}  dfy=±{args.dfy_pct}")
        print(f"[mfu] world_size={accel.num_processes}  "
              f"per-rank batch={args.batch}  global batch={args.batch*accel.num_processes}")

    ds_train = PandaSetCalibDatasetFull(
        cache_dir=CACHE, split='train',
        img_size=cfg['img_size'],
        min_crop_px=cfg['min_crop_px'], max_crop_px=cfg['max_crop_px'],
        max_offset_m=0.0, max_rot_deg=0.0,
        oversample=1, grid_n=cfg.get('grid_n', 16),
        center_band=0.0, preload=False,
    )
    ds_val = PandaSetCalibDatasetFull(
        cache_dir=CACHE, split='val',
        img_size=cfg['img_size'],
        min_crop_px=cfg['min_crop_px'], max_crop_px=cfg['max_crop_px'],
        max_offset_m=0.0, max_rot_deg=0.0,
        oversample=1, grid_n=cfg.get('grid_n', 16),
        center_band=0.0, preload=False,
    )
    if accel.is_main_process:
        print(f"[mfu] train tiles: {len(ds_train.fnames)}   val tiles: {len(ds_val.fnames)}")

    model = _build_model(cfg).to(device)
    sd = torch.load(CKPT, map_location='cpu', weights_only=False)
    if isinstance(sd, dict) and ('state_dict' in sd or 'model' in sd):
        sd = sd.get('state_dict') or sd.get('model')
    missing, unexpected = model.load_state_dict(sd, strict=False)
    other_missing = [k for k in missing if not k.startswith('info_head')]
    assert len(other_missing) == 0, f"missing: {other_missing[:5]}"
    if args.freeze_backbone:
        for n, p in model.named_parameters():
            if not n.startswith('info_head'):
                p.requires_grad = False
    n_total = sum(p.numel() for p in model.parameters())
    n_trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    if accel.is_main_process:
        mode = ('info_head ONLY (backbone+Δuv head frozen)'
                if args.freeze_backbone else 'ALL trainable, backbone unfrozen')
        print(f"[mfu] params: total={n_total/1e6:.2f}M  "
              f"trainable={n_trainable/1e6:.2f}M  ({mode})")

    train_params = [p for p in model.parameters() if p.requires_grad]
    opt = torch.optim.AdamW(train_params, lr=args.lr, weight_decay=0.0)
    # cosine to ~1e-7 over the full run (matches train_ps_v3 style: lr_min=1e-7)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(
        opt, T_max=args.n_steps, eta_min=1e-7,
    )

    model, opt = accel.prepare(model, opt)

    # ─── BUILD HELD-OUT VAL EVAL BATCH (rank 0 only) ─────────────────
    eval_batch = None
    eval_tensors = None
    delta_target_eval = None
    n_eval = args.n_eval
    if accel.is_main_process:
        eval_rng = np.random.RandomState(SEED + 1000)
        val_idx_pool = list(range(len(ds_val.fnames)))
        eval_rng.shuffle(val_idx_pool)
        eval_pool_iter = iter(val_idx_pool)

        def _eval_idx_iter():
            nonlocal eval_pool_iter
            try:
                return next(eval_pool_iter)
            except StopIteration:
                eval_rng.shuffle(val_idx_pool)
                eval_pool_iter = iter(val_idx_pool)
                return next(eval_pool_iter)

        print(f"[mfu] building held-out eval batch (N={n_eval} val tiles × 1 pert) ...")
        t_eval_build = time.time()
        eval_batch = _build_batch(ds_val, _eval_idx_iter, n_eval, eval_rng,
                                   **pert_kw)
        assert eval_batch is not None, "could not build eval batch"
        print(f"[mfu]   eval batch built in {time.time()-t_eval_build:.1f}s")
        e = _move_batch(eval_batch, device)
        (e_imgs, e_true_uvd, e_dist_uvd, e_pad, e_vfp,
         e_bucket_uvd, e_bucket_valid, _,
         e_pts_cam_orig, e_duv_orig, e_K_orig, e_cs) = e
        (e_valid, e_duv_oracle_local, e_duv_oracle_orig,
         e_P0_orig, e_K_orig, e_cs) = _prep_inputs(
            cfg, e_true_uvd, e_dist_uvd, e_pad,
            e_pts_cam_orig, e_duv_orig, e_K_orig, e_cs)
        delta_target_eval = _solve_target(
            e_P0_orig, e_duv_oracle_orig, e_valid, e_K_orig, DOFS)
        eval_tensors = dict(
            imgs=e_imgs, true_uvd=e_true_uvd, dist_uvd=e_dist_uvd,
            pad=e_pad, vfp=e_vfp, bucket_uvd=e_bucket_uvd,
            bucket_valid=e_bucket_valid, valid=e_valid,
            duv_oracle_local=e_duv_oracle_local,
            duv_oracle_orig=e_duv_oracle_orig,
            P0_orig=e_P0_orig, K_orig=e_K_orig, cs=e_cs,
            padfull=~e_valid,
        )

    EVAL_CHUNK = 32
    SOLVE_CHUNK = 32

    def _eval_all_W(model_unwrapped):
        """Returns dict of MSE under W=I, W=σ, W=learned + delta_target ref."""
        et = eval_tensors
        n = n_eval
        per_pt_list, q_list = [], []
        for s in range(0, n, EVAL_CHUNK):
            sl = slice(s, min(s + EVAL_CHUNK, n))
            pp, qq = _forward_with_q(
                model_unwrapped,
                et['imgs'][sl], et['dist_uvd'][sl], et['pad'][sl],
                et['vfp'][sl], et['bucket_uvd'][sl], et['bucket_valid'][sl],
                no_grad=True,
            )
            per_pt_list.append(pp); q_list.append(qq)
        per_pt = torch.cat(per_pt_list, dim=0)
        q      = torch.cat(q_list,      dim=0)
        duv_pred_local = per_pt[..., :2].detach()
        if et['padfull'].any():
            duv_pred_local = duv_pred_local.clone()
            duv_pred_local[et['padfull']] = 0.0
        sx = per_pt[..., 2].exp()
        sy = per_pt[..., 3].exp()
        rho = per_pt[..., 4]
        W_sigma_local = make_info_from_sigma_rho(sx, sy, rho).detach()
        W_eye_local   = torch.eye(2, device=device).expand(n, q.shape[1], 2, 2)
        with torch.no_grad():
            W_learn_local = model_unwrapped.info_head(q)

        # Convert local-128px → original-camera px for the solver path.
        # duv_orig = duv_local · (cs/S);  W_orig = W_local · (S/cs)²
        scale_l2o = _local_to_orig_scale(et['cs'], cfg['img_size'])  # (B,1,1)
        duv_pred_orig = duv_pred_local * scale_l2o
        inv_l2o = (1.0 / scale_l2o).reshape(-1, 1, 1, 1)              # (B,1,1,1)
        W_eye_orig   = W_eye_local   * inv_l2o.pow(2)
        W_sigma_orig = W_sigma_local * inv_l2o.pow(2)
        W_learn_orig = W_learn_local * inv_l2o.pow(2)

        def _mse_for(W_full):
            d_parts = []
            for s in range(0, n, SOLVE_CHUNK):
                sl = slice(s, min(s + SOLVE_CHUNK, n))
                d_part, _ = _solve_with(
                    et['P0_orig'][sl], duv_pred_orig[sl], W_full[sl],
                    et['K_orig'][sl], DOFS, et['valid'][sl])
                d_parts.append(d_part)
            d_full = torch.cat(d_parts, dim=0)
            return (d_full - delta_target_eval).pow(2).mean(dim=-1).mean().item()

        with torch.no_grad():
            mse_I  = _mse_for(W_eye_orig)
            mse_s  = _mse_for(W_sigma_orig)
            mse_l  = _mse_for(W_learn_orig)
            d_zero = torch.zeros_like(delta_target_eval)
            mse_zero = (d_zero - delta_target_eval).pow(2).mean(dim=-1).mean().item()
            logdet = torch.linalg.det(W_learn_orig).clamp_min(1e-12).log().mean().item()
        return mse_zero, mse_I, mse_s, mse_l, logdet

    if accel.is_main_process:
        unwrapped = accel.unwrap_model(model)
        unwrapped.eval()
        mse_zero, mse_w1, mse_w2, mse_w3_init, ld_init = _eval_all_W(unwrapped)
        print(f"[mfu] HELD-OUT BASELINE  do-nothing: {mse_zero:.6e}  "
              f"W=I: {mse_w1:.6e}  W=σ: {mse_w2:.6e}  "
              f"W=learned(init frozen-head): {mse_w3_init:.6e}  "
              f"⟨log det W⟩={ld_init:+.2f}")

    # ─── streaming TRAIN ─────────────────────────────────────────────
    train_idx_pool = list(range(len(ds_train.fnames)))
    rng.shuffle(train_idx_pool)
    train_iter = iter(train_idx_pool)

    def _train_idx_iter():
        nonlocal train_iter
        try:
            return next(train_iter)
        except StopIteration:
            rng.shuffle(train_idx_pool)
            train_iter = iter(train_idx_pool)
            return next(train_iter)

    history = []
    t0 = time.time()
    model.train()
    for step in range(args.n_steps + 1):
        batch = _build_batch(ds_train, _train_idx_iter, args.batch, rng,
                              **pert_kw)
        if batch is None:
            if accel.is_main_process:
                print(f"  step {step}: skipped (could not build train batch)", flush=True)
            continue
        b = _move_batch(batch, device)
        (imgs, true_uvd, dist_uvd, pad_mask, vfp,
         bucket_uvd, bucket_valid, _,
         pts_cam_orig, duv_orig, K_orig, cs) = b
        (valid, duv_oracle_local, duv_oracle_orig,
         P0_orig, K_orig, cs) = _prep_inputs(
            cfg, true_uvd, dist_uvd, pad_mask,
            pts_cam_orig, duv_orig, K_orig, cs)
        delta_target = _solve_target(P0_orig, duv_oracle_orig, valid, K_orig, DOFS)

        # forward through DDP-wrapped model so backbone gradients flow
        per_pt, q = _forward_with_q(
            model, imgs, dist_uvd, pad_mask, vfp, bucket_uvd, bucket_valid,
            no_grad=False,
        )
        duv_pred_local = per_pt[..., :2]
        if (~valid).any():
            mask = (~valid).unsqueeze(-1).float()
            duv_pred_local = duv_pred_local * (1.0 - mask)

        # W_learn from the SAME forward (info_head is in-graph). Both duv and W
        # are network outputs in local-128px units; convert to orig-camera px
        # before entering the solver so δ comes out in parent-camera SE3.
        unwrapped = accel.unwrap_model(model)
        W_learn_local = unwrapped.info_head(q)

        scale_l2o = _local_to_orig_scale(cs, cfg['img_size'])         # (B,1,1)
        inv_l2o = (1.0 / scale_l2o).reshape(-1, 1, 1, 1)              # (B,1,1,1)
        duv_pred_orig = duv_pred_local * scale_l2o
        W_learn_orig  = W_learn_local  * inv_l2o.pow(2)

        delta_pred, H_pred = _solve_with(
            P0_orig, duv_pred_orig, W_learn_orig, K_orig, DOFS, valid)
        diff = delta_pred - delta_target          # (B, K)
        if args.loss == 'mse':
            pose_loss = diff.pow(2).sum(dim=-1).mean()
        else:  # nll
            # ½ diff^T H diff − ½ log det H
            #   = ½ Mahalanobis² + ½ log det Σ_pose
            maha = torch.einsum('bi,bij,bj->b', diff, H_pred, diff)
            sign, logabsdet = torch.linalg.slogdet(H_pred)
            pose_loss = (0.5 * maha - 0.5 * logabsdet).mean()

        if args.perpt_weight > 0.0:
            v = valid.unsqueeze(-1).float()
            n_valid = v.sum().clamp(min=1.0)
            # per-pt supervision lives in local-128px space (network's native
            # output units). Δuv_orig = Δuv_local · (cs/S) is exact, so the two
            # losses are off by a per-sample² factor — consistent across
            # samples with cs ≈ const.
            perpt_loss = ((duv_pred_local - duv_oracle_local).pow(2) * v).sum() / (2.0 * n_valid)
            loss = pose_loss + args.perpt_weight * perpt_loss
        else:
            perpt_loss = torch.zeros((), device=device)
            loss = pose_loss

        opt.zero_grad(set_to_none=True)
        accel.backward(loss)
        accel.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
        sched.step()

        if step % LOG_EVERY == 0 and accel.is_main_process:
            sps = (step + 1) / max(1e-3, time.time() - t0)
            cur_lr = opt.param_groups[0]['lr']
            print(f"  step {step:>4}  loss={loss.item():.4e}  "
                  f"pose={pose_loss.item():.4e}  perpt={perpt_loss.item():.4e}  "
                  f"lr={cur_lr:.2e}  "
                  f"({sps:.2f} step/s, ~{sps*accel.num_processes:.0f} sample/s global)",
                  flush=True)

        if step % EVAL_EVERY == 0 or step == args.n_steps:
            accel.wait_for_everyone()
            if accel.is_main_process:
                unwrapped = accel.unwrap_model(model)
                unwrapped.eval()
                _mse_zero, _mse_I, _mse_s, _mse_l, _ld = _eval_all_W(unwrapped)
                history.append((step, loss.item(), _mse_l, _ld, _mse_I, _mse_s))
                print(f"  >> EVAL step {step:>4}  do-nothing={_mse_zero:.4e}  "
                      f"W=I={_mse_I:.4e}  W=σ={_mse_s:.4e}  "
                      f"W=learn={_mse_l:.4e}  ⟨log det W⟩={_ld:+.2f}",
                      flush=True)
                model.train()
            accel.wait_for_everyone()

    accel.wait_for_everyone()
    if accel.is_main_process:
        unwrapped = accel.unwrap_model(model)
        unwrapped.eval()
        mse_zero, mse_w1, mse_w2, mse_w3, _ld = _eval_all_W(unwrapped)
        print(f"\n[mfu] HELD-OUT FINAL  do-nothing: {mse_zero:.6e}  "
              f"W=I: {mse_w1:.6e}  W=σ: {mse_w2:.6e}  "
              f"W=learned: {mse_w3:.6e}  ⟨log det W⟩={_ld:+.2f}")
        print(f"[mfu] PASS vs uniform: {mse_w3 < mse_w1}   "
              f"PASS vs σ:       {mse_w3 < mse_w2}   "
              f"PASS vs do-nothing: {mse_w3 < mse_zero}")

        # plots
        steps  = [h[0] for h in history]
        trnL   = [h[1] for h in history]
        evMSE  = [h[2] for h in history]
        logdet = [h[3] for h in history]

        fig, axes = plt.subplots(1, 2, figsize=(13, 4.4))
        ax = axes[0]
        ax.semilogy(steps, evMSE, '-o', color='tab:red', ms=3,
                     label=f'learned W (val held-out N={n_eval})')
        ax.semilogy(steps, trnL,  '-x', color='tab:orange', ms=3, alpha=0.5,
                     label='train loss')
        ax.axhline(mse_zero, ls=':',  color='tab:green',  label=f'do-nothing ({mse_zero:.2e})')
        ax.axhline(mse_w1,   ls='--', color='tab:gray',   label=f'W=I        ({mse_w1:.2e})')
        ax.axhline(mse_w2,   ls='--', color='tab:purple', label=f'W=σ-head ({mse_w2:.2e})')
        ax.set_xlabel('step'); ax.set_ylabel('δ-MSE (deg²)')
        ax.set_title(f'(a) UNFROZEN backbone — DDP {accel.num_processes}×')
        ax.grid(which='both', alpha=0.3); ax.legend(loc='best', fontsize=8)

        ax = axes[1]
        bars = ['do-nothing', 'W=I', 'W=σ', 'W=learn']
        vals = [mse_zero, mse_w1, mse_w2, mse_w3]
        colors = ['tab:green', 'tab:gray', 'tab:purple', 'tab:red']
        ax.bar(bars, vals, color=colors)
        for i, v in enumerate(vals):
            ax.text(i, v, f'{v:.2e}', ha='center', va='bottom', fontsize=8)
        ax.set_yscale('log')
        ax.set_ylabel('δ-MSE (deg²)')
        ax.set_title(f'(b) final δ-MSE on val held-out N={n_eval}\n'
                     f'unlock DDP{accel.num_processes}: {args.n_steps} steps × B={args.batch}/rank')
        ax.grid(axis='y', alpha=0.3)

        fig.suptitle(
            f'Phase 1.c-unlock: backbone UNFROZEN + DDP {accel.num_processes}×\n'
            f'Frozen baseline ckpt → train ALL of CalibNetDepth thru BA gradient.',
            y=1.04, fontsize=10,
        )
        fig.tight_layout()
        out_png = out_dir / 'curves.png'
        fig.savefig(out_png, dpi=110, bbox_inches='tight')
        plt.close(fig)
        print(f"[mfu] wrote → {out_png}")

        # save unwrapped state dicts (whole model + info_head only for compat
        # with render_sigma_ellipse_compare.py which expects info_head.pt)
        unwrapped = accel.unwrap_model(model)
        torch.save(unwrapped.state_dict(), out_dir / 'model.pt')
        torch.save(unwrapped.info_head.state_dict(), out_dir / 'info_head.pt')
        summary = {
            'n_steps': args.n_steps, 'batch': args.batch, 'n_eval': n_eval,
            'world_size': accel.num_processes,
            'mse_zero': mse_zero, 'mse_W_I': mse_w1, 'mse_W_sigma': mse_w2,
            'mse_W_learned': mse_w3,
            'history': history,
        }
        torch.save(summary, out_dir / 'summary.pt')
        print(f"[mfu] wrote → {out_dir / 'info_head.pt'}, model.pt, summary.pt")
        print("[mfu] done.")


if __name__ == '__main__':
    main()
