"""Phase 1.b-orig — 1-frame STREAMING perturbations, configurable DoFs,
   ORIG-camera solver (solve_pinhole_xyz), single GPU.

Anchor: 1 val tile (default idx=17, the kamikado_v3_tiled highway tile we've
been visualising). Per step we re-sample fresh random perturbations
(rotation / translation / focal) from this same tile, so the InfoHead2x2 has
no fixed memorisation surface — it must learn a frame-conditional mapping
q → W.

Same frozen ckpt as Phase 1.b. Backbone + Δuv head are FROZEN; only
info_head trains. The solver runs entirely in the original-camera frame:
    duv_orig = duv_local · (cs/S)
    W_orig   = W_local   · (S/cs)²        (W = Σ⁻¹ → inverse scale²)

Pass criterion (held-out): final δ-MSE  <  W=I baseline  AND  ≤ W=σ baseline.

Outputs land in scripts/_debug/_outputs/overfit_6dof_ba_stream_orig_{tag}/.
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
    solve_kb_xyz, solve_kb_xyz_shared, make_info_from_sigma_rho,
)

try:
    from clearml import Task as _ClearMLTask
except Exception:  # pragma: no cover
    _ClearMLTask = None


# ─── config ───────────────────────────────────────────────────────────
CACHE       = Path.home() / 'cache' / 'kamikado_v3_tiled'
CKPT        = REPO / 'experiments' / 'km_wv_wm_dgx2_n4_img128_8gpu_HEAD' / 'best_model.pt'
EXP_CFG_PATH = REPO / 'experiments' / 'km_wv_wm_dgx2_n4_img128_8gpu_HEAD' / 'config.py'
OUT         = REPO / 'scripts' / '_debug' / '_outputs' / 'overfit_6dof_ba_stream_orig'

DEFAULT_N_STEPS = 600
DEFAULT_BATCH   = 16
DEFAULT_N_EVAL  = 200
LR              = 1e-3
LOG_EVERY       = 25
EVAL_EVERY      = 25
BA_N_ITER       = 6
DAMPING         = 1e-3
SEED            = 7

DOF_PRESETS = {
    '2dof':   ['omega_x', 'omega_y'],
    '2dof_t': ['tx', 'ty'],
    '3dof':   ['omega_x', 'omega_y', 'omega_z'],
    '4dof':   ['omega_x', 'omega_y', 'tx', 'ty'],
    '6dof':   ['omega_x', 'omega_y', 'omega_z', 'tx', 'ty', 'tz'],
    '10dof':  ['omega_x', 'omega_y', 'omega_z', 'tx', 'ty', 'tz',
               'dfx', 'dfy', 'dcx', 'dcy'],
}

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


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


def _freeze_except_info_head(model: CalibNetDepth):
    n_train = 0
    n_total = 0
    for name, p in model.named_parameters():
        n_total += p.numel()
        if name.startswith('info_head'):
            p.requires_grad_(True)
            n_train += p.numel()
        else:
            p.requires_grad_(False)
    return n_train, n_total


def _unfreeze_all(model: CalibNetDepth):
    n_train = 0
    n_total = 0
    for _name, p in model.named_parameters():
        n_total += p.numel()
        p.requires_grad_(True)
        n_train += p.numel()
    return n_train, n_total


def _local_to_orig_scale(cs: torch.Tensor, S: float) -> torch.Tensor:
    return (cs / float(S)).reshape(-1, 1, 1)


def _draw_pert(rng, *, rot_deg, t_m, dfx_pct, dfy_pct):
    """Draw one (ypr, t, dfx, dfy) perturbation tuple."""
    if rot_deg > 0.0:
        omega_x = float(rng.uniform(-rot_deg, rot_deg))
        omega_y = float(rng.uniform(-rot_deg, rot_deg))
        omega_z = float(rng.uniform(-rot_deg, rot_deg))
    else:
        omega_x = omega_y = omega_z = 0.0
    ypr = np.array([omega_z, omega_y, omega_x], dtype=np.float64)
    if t_m > 0.0:
        t = (rng.uniform(-1.0, 1.0, size=3) * t_m).astype(np.float64)
    else:
        t = np.zeros(3, dtype=np.float64)
    dfx = float(rng.uniform(-dfx_pct, dfx_pct)) if dfx_pct > 0.0 else 0.0
    dfy = float(rng.uniform(-dfy_pct, dfy_pct)) if dfy_pct > 0.0 else 0.0
    return ypr, t, dfx, dfy


def _sample_one(ds, idx, rng, *, rot_deg, t_m, dfx_pct, dfy_pct, pert=None):
    if pert is None:
        ypr, t, dfx, dfy = _draw_pert(rng, rot_deg=rot_deg, t_m=t_m,
                                       dfx_pct=dfx_pct, dfy_pct=dfy_pct)
    else:
        ypr, t, dfx, dfy = pert
    return ds.apply_perturbation_explicit(idx, t, ypr,
                                           dfx_pct=dfx, dfy_pct=dfy)


def _build_batch(ds, idx, B, rng, *, rot_deg, t_m, dfx_pct, dfy_pct,
                 max_tries=64):
    windows = []
    tries = 0
    while len(windows) < B and tries < max_tries * B:
        win = _sample_one(ds, idx, rng, rot_deg=rot_deg, t_m=t_m,
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


def _forward_capture_q(model, imgs_u8, dist_uvd, pad_mask, vfp,
                       bucket_uvd, bucket_valid, *, no_grad: bool = True):
    captured = {}

    def _hook(_module, _inputs, _output):
        # in unfreeze mode we need q with grad so info_head receives gradient
        captured['q'] = _inputs[0] if not no_grad else _inputs[0].detach()

    h = model.info_head.mlp[0].register_forward_hook(_hook)
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
    per_pt, _W_unused = out
    return per_pt, captured['q']


def _prep_inputs(true_uvd, dist_uvd, pad_mask,
                 pts_cam_orig, duv_orig, K_orig, cs):
    valid = ~pad_mask
    pad_full = ~valid
    duv_oracle_local = (true_uvd[..., :2] - dist_uvd[..., :2]).detach()
    duv_oracle_orig = duv_orig.detach().clone()
    P0_orig = pts_cam_orig.detach().clone()
    if pad_full.any():
        duv_oracle_local = duv_oracle_local.clone()
        duv_oracle_local[pad_full] = 0.0
        duv_oracle_orig[pad_full] = 0.0
        P0_orig[pad_full] = torch.tensor(
            [0.0, 0.0, 1.0], dtype=P0_orig.dtype, device=P0_orig.device)
    return valid, duv_oracle_local, duv_oracle_orig, P0_orig, K_orig, cs


def _build_prior_diag(dofs, *, sigma_rot_deg, sigma_t_m, sigma_f_pct):
    """λ_k = 1/σ_k² for the k-th DoF in `dofs`. Solver internals carry ω in
    DEG and t in M, so prior magnitudes follow those units directly:
        rotation axes (omega_*): λ = 1/σ_rot_deg²       (deg⁻²)
        translation axes (t*) : λ = 1/σ_t_m²            (m⁻²)
        focal/center axes    : λ = 1/σ_f_pct²           (% units)
    Set any sigma to 0 to disable that group."""
    vals = []
    for nm in dofs:
        if nm.startswith('omega'):
            sig = sigma_rot_deg
        elif nm in ('tx', 'ty', 'tz'):
            sig = sigma_t_m
        else:  # dfx, dfy, dcx, dcy
            sig = sigma_f_pct
        vals.append(0.0 if sig <= 0.0 else 1.0 / (float(sig) ** 2))
    return torch.tensor(vals, dtype=torch.float32)


def _solve_target(P0_orig, duv_oracle_orig, valid, K_orig, dist, dofs,
                   prior_diag=None, *, shared=False):
    B, N = P0_orig.shape[:2]
    W_eye = torch.eye(2, device=P0_orig.device,
                       dtype=P0_orig.dtype).expand(B, N, 2, 2)
    with torch.no_grad():
        if shared:
            delta, _ = solve_kb_xyz_shared(
                P0_orig, duv_oracle_orig, W_eye, K_orig, dist, dofs,
                valid=valid, n_iter=BA_N_ITER, damping=DAMPING,
                prior_diag=prior_diag,
            )
        else:
            delta, _ = solve_kb_xyz(
                P0_orig, duv_oracle_orig, W_eye, K_orig, dist, dofs,
                valid=valid, n_iter=BA_N_ITER, damping=DAMPING,
                prior_diag=prior_diag,
            )
    return delta.detach()


def _solve_with(P0_orig, duv_orig, W_orig, K_orig, dist, dofs, valid,
                 prior_diag=None, *, shared=False):
    if shared:
        return solve_kb_xyz_shared(
            P0_orig, duv_orig, W_orig, K_orig, dist, dofs,
            valid=valid, n_iter=BA_N_ITER, damping=DAMPING,
            prior_diag=prior_diag,
        )
    return solve_kb_xyz(
        P0_orig, duv_orig, W_orig, K_orig, dist, dofs,
        valid=valid, n_iter=BA_N_ITER, damping=DAMPING,
        prior_diag=prior_diag,
    )


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--idx', type=int, default=17,
                    help='val tile index to anchor on (default: 17 = highway tile)')
    ap.add_argument('--n-steps', type=int, default=DEFAULT_N_STEPS)
    ap.add_argument('--batch',   type=int, default=DEFAULT_BATCH)
    ap.add_argument('--n-eval',  type=int, default=DEFAULT_N_EVAL)
    ap.add_argument('--lr',      type=float, default=LR)
    ap.add_argument('--tag',     type=str, default=None)
    ap.add_argument('--dofs',    type=str, default='6dof',
                    choices=list(DOF_PRESETS.keys()))
    ap.add_argument('--rot-deg', type=float, default=0.30)
    ap.add_argument('--t-m',     type=float, default=0.05)
    ap.add_argument('--dfx-pct', type=float, default=0.0)
    ap.add_argument('--dfy-pct', type=float, default=0.0)
    ap.add_argument('--clearml', action='store_true',
                    help='log scalars + final plots to ClearML')
    ap.add_argument('--clearml-project', type=str,
                    default='e2e_calib/principled_ml')
    ap.add_argument('--unfreeze', action='store_true',
                    help='unfreeze backbone + Δuv head; train everything end-to-end')
    ap.add_argument('--perpt-weight', type=float, default=1.0,
                    help='weight on per-point Δuv MSE aux loss (only used '
                         'with --unfreeze; default 1.0)')
    ap.add_argument('--pose-loss', type=str, default='mse',
                    choices=['mse', 'nll'],
                    help='pose loss: mse = ||δ−δ*||²,  nll = ½ Mahalanobis − ½ log det H')
    ap.add_argument('--multi', action='store_true',
                    help='draw a random tile idx from TRAIN split every step '
                         '(instead of fixed --idx). Eval batch also draws '
                         'random val tiles when this is on.')
    ap.add_argument('--prior-rot-deg', type=float, default=3.0,
                    help='σ for rotation Gaussian prior in DEGREES. '
                         'λ_ω = 1/σ². Set 0 to disable. Default 3°.')
    ap.add_argument('--prior-t-m', type=float, default=0.2,
                    help='σ for translation Gaussian prior in METERS. '
                         'λ_t = 1/σ². Set 0 to disable. Default 0.2 m.')
    ap.add_argument('--prior-f-pct', type=float, default=0.0,
                    help='σ for focal % prior. Default 0 (no prior on '
                         'dfx/dfy/dcx/dcy axes).')
    ap.add_argument('--w-diag-min', type=float, default=1e-2,
                    help='lower clamp on info-head L diagonals (a, b). '
                         'orig-px world; 1e-2 ↔ σ_uv ≤ 10 px. '
                         'Set 0 to disable (legacy softplus path).')
    ap.add_argument('--w-diag-max', type=float, default=4.0,
                    help='upper clamp on info-head L diagonals (a, b). '
                         '4.0 ↔ σ_uv ≥ 0.5 px (no super-confident points). '
                         'Set 0 to disable.')
    ap.add_argument('--shared', action='store_true',
                    help='aggregate B tiles into ONE solve sharing a single '
                         'rig-level δ (same ypr/t applied to all tiles). '
                         'Aperture vanishes via H-aggregation across cameras.')
    args = ap.parse_args()

    torch.manual_seed(SEED)
    np.random.seed(SEED)
    tag = args.tag or f'idx{args.idx}_{args.dofs}'
    out_dir = OUT.parent / f'overfit_6dof_ba_stream_orig_{tag}'
    out_dir.mkdir(parents=True, exist_ok=True)
    rng = np.random.RandomState(SEED)

    cml_task = None
    cml_logger = None
    if args.clearml:
        if _ClearMLTask is None:
            print('[stream-orig] WARN: --clearml passed but clearml not importable')
        else:
            cml_task = _ClearMLTask.init(
                project_name=args.clearml_project,
                task_name=f'overfit_orig_{tag}',
                task_type=_ClearMLTask.TaskTypes.optimizer,
                reuse_last_task_id=False,
                auto_connect_frameworks={'matplotlib': False, 'pytorch': False},
            )
            cml_task.connect({
                'idx': args.idx, 'dofs': args.dofs,
                'n_steps': args.n_steps, 'batch': args.batch,
                'n_eval': args.n_eval, 'lr': args.lr,
                'rot_deg': args.rot_deg, 't_m': args.t_m,
                'dfx_pct': args.dfx_pct, 'dfy_pct': args.dfy_pct,
                'ba_n_iter': BA_N_ITER, 'damping': DAMPING, 'seed': SEED,
            })
            cml_logger = cml_task.get_logger()
            print(f'[stream-orig] ClearML task: {cml_task.get_output_log_web_page()}')

    cfg = _load_cfg()
    DOFS = DOF_PRESETS[args.dofs]
    pert_kw = dict(rot_deg=args.rot_deg, t_m=args.t_m,
                   dfx_pct=args.dfx_pct, dfy_pct=args.dfy_pct)
    prior_diag = _build_prior_diag(
        DOFS, sigma_rot_deg=args.prior_rot_deg,
        sigma_t_m=args.prior_t_m, sigma_f_pct=args.prior_f_pct,
    ).to(DEVICE)
    print(f"[stream-orig] physical prior σ_ω={args.prior_rot_deg}° "
          f"σ_t={args.prior_t_m}m σ_f={args.prior_f_pct}%  "
          f"→ λ_diag={prior_diag.cpu().numpy()}")
    print(f"[stream-orig] cfg: img={cfg['img_size']} layers={cfg['n_layers']} "
          f"deform={cfg.get('deform_mode')} convnext={cfg.get('use_convnext')}")
    print(f"[stream-orig] DoFs={args.dofs} ({DOFS})  "
          f"rot=±{args.rot_deg}deg  t=±{args.t_m}m  "
          f"dfx=±{args.dfx_pct}  dfy=±{args.dfy_pct}")

    ds = PandaSetCalibDatasetFull(
        cache_dir=CACHE,
        split='val',
        img_size=cfg['img_size'],
        min_crop_px=cfg['min_crop_px'],
        max_crop_px=cfg['max_crop_px'],
        max_offset_m=0.0,
        max_rot_deg=0.0,
        oversample=1,
        grid_n=cfg.get('grid_n', 16),
        center_band=0.0,
        preload=False,
    )
    print(f"[stream-orig] dataset: {len(ds.fnames)} val instances")

    train_ds = None
    if args.multi or args.shared:
        train_ds = PandaSetCalibDatasetFull(
            cache_dir=CACHE,
            split='train',
            img_size=cfg['img_size'],
            min_crop_px=cfg['min_crop_px'],
            max_crop_px=cfg['max_crop_px'],
            max_offset_m=0.0,
            max_rot_deg=0.0,
            oversample=1,
            grid_n=cfg.get('grid_n', 16),
            center_band=0.0,
            preload=False,
        )
        mode_tag = 'SHARED' if args.shared else '--multi'
        print(f"[stream-orig] {mode_tag} ON: train pool = {len(train_ds.fnames)} tiles, "
              f"eval = random val tiles (--idx ignored)")

    base_idx = int(args.idx)
    win0 = ds.apply_perturbation_explicit(base_idx, np.zeros(3), np.zeros(3))
    assert win0 is not None, f"idx={base_idx} returns None"
    print(f"[stream-orig] anchor val frame idx={base_idx} ({ds.fnames[base_idx]})")

    inst0 = ds._load_inst(base_idx)
    assert inst0.get('is_fisheye', False) and 'distortion' in inst0, \
        f"idx={base_idx} is not fisheye / has no KB distortion — KB solver path requires k1..k4"
    dist_one = inst0['distortion'].clone().detach().to(torch.float32).reshape(1, 4)
    print(f"[stream-orig] KB dist (k1..k4) = {dist_one[0].numpy()}")

    model = _build_model(cfg).to(DEVICE)
    sd = torch.load(CKPT, map_location='cpu', weights_only=False)
    if isinstance(sd, dict) and ('state_dict' in sd or 'model' in sd):
        sd = sd.get('state_dict') or sd.get('model')
    missing, unexpected = model.load_state_dict(sd, strict=False)
    other_missing = [k for k in missing if not k.startswith('info_head')]
    assert len(other_missing) == 0, f"missing keys: {other_missing[:5]}"
    if args.unfreeze:
        n_train, n_total = _unfreeze_all(model)
        mode = 'UNFREEZE (all params + per-pt MSE aux)'
        opt_params = [p for p in model.parameters() if p.requires_grad]
    else:
        n_train, n_total = _freeze_except_info_head(model)
        mode = 'FROZEN (info_head only)'
        opt_params = list(model.info_head.parameters())
    print(f"[stream-orig] mode: {mode}")
    print(f"[stream-orig] params: {n_train/1e6:.2f}M trainable / "
          f"{n_total/1e6:.2f}M total")

    info_head = model.info_head
    if args.w_diag_min > 0.0 and args.w_diag_max > 0.0:
        info_head.w_diag_min = float(args.w_diag_min)
        info_head.w_diag_max = float(args.w_diag_max)
        print(f"[stream-orig] InfoHead diag clamp ON: L_diag ∈ "
              f"[{args.w_diag_min}, {args.w_diag_max}]  "
              f"⇒ σ_uv ∈ [{1.0/args.w_diag_max**0.5:.2f}, "
              f"{1.0/args.w_diag_min**0.5:.2f}] px (orig)")
    else:
        info_head.w_diag_min = None
        info_head.w_diag_max = None
        print(f"[stream-orig] InfoHead diag clamp OFF (legacy softplus)")
    opt = torch.optim.AdamW(opt_params, lr=args.lr, weight_decay=0.0)

    # ─── BASELINES (held-out) ──────────────────────────────────────
    eval_rng = np.random.RandomState(SEED + 1000)
    if args.shared:
        # eval batch = N different RANDOM val tiles, ONE shared perturbation
        # applied to all of them (matches the shared training contract).
        shared_pert = _draw_pert(eval_rng, **pert_kw)
        e_wins = []
        e_tries = 0
        while len(e_wins) < args.n_eval and e_tries < 16 * args.n_eval:
            ridx = int(eval_rng.randint(0, len(ds.fnames)))
            w = _sample_one(ds, ridx, eval_rng, pert=shared_pert, **pert_kw)
            e_tries += 1
            if w is not None:
                e_wins.append(w)
        assert len(e_wins) == args.n_eval, \
            f"could not build shared eval batch ({len(e_wins)}/{args.n_eval})"
        eval_batch = collate_full(e_wins)
        print(f"[stream-orig] EVAL: {args.n_eval} random val tiles, "
              f"SHARED δ  ypr={shared_pert[0]} t={shared_pert[1]}")
    elif args.multi:
        # eval batch = N different RANDOM val tiles, each with its own
        # random perturbation. Mirrors training's tile diversity instead
        # of pinning to args.idx.
        e_wins = []
        e_tries = 0
        while len(e_wins) < args.n_eval and e_tries < 16 * args.n_eval:
            ridx = int(eval_rng.randint(0, len(ds.fnames)))
            w = _sample_one(ds, ridx, eval_rng, **pert_kw)
            e_tries += 1
            if w is not None:
                e_wins.append(w)
        assert len(e_wins) == args.n_eval, \
            f"could not build multi-tile eval batch ({len(e_wins)}/{args.n_eval})"
        eval_batch = collate_full(e_wins)
        print(f"[stream-orig] EVAL: {args.n_eval} random val tiles "
              f"(--multi mode)")
    else:
        eval_batch = _build_batch(ds, base_idx, args.n_eval, eval_rng, **pert_kw)
    assert eval_batch is not None, "could not build eval batch"
    e = _move_batch(eval_batch, DEVICE)
    (e_imgs, e_true_uvd, e_dist_uvd, e_pad, e_vfp,
     e_bucket_uvd, e_bucket_valid, _,
     e_pts_cam_orig, e_duv_orig, e_K_orig, e_cs) = e
    (e_valid, e_duv_oracle_local, e_duv_oracle_orig,
     e_P0_orig, e_K_orig, e_cs) = _prep_inputs(
        e_true_uvd, e_dist_uvd, e_pad,
        e_pts_cam_orig, e_duv_orig, e_K_orig, e_cs)
    e_dist = dist_one.to(DEVICE).expand(args.n_eval, 4).contiguous()
    delta_target_eval = _solve_target(
        e_P0_orig, e_duv_oracle_orig, e_valid, e_K_orig, e_dist, DOFS,
        prior_diag=prior_diag, shared=args.shared)

    model.eval()
    e_per_pt, e_q = _forward_capture_q(
        model, e_imgs, e_dist_uvd, e_pad, e_vfp, e_bucket_uvd, e_bucket_valid)
    e_duv_pred_local = e_per_pt[..., :2].detach()
    e_padfull = ~e_valid
    if e_padfull.any():
        e_duv_pred_local = e_duv_pred_local.clone()
        e_duv_pred_local[e_padfull] = 0.0
    e_sx = e_per_pt[..., 2].exp()
    e_sy = e_per_pt[..., 3].exp()
    e_rho = e_per_pt[..., 4]
    e_W_sigma_local = make_info_from_sigma_rho(e_sx, e_sy, e_rho).detach()
    e_W_eye_local = torch.eye(2, device=DEVICE).expand(
        args.n_eval, e_q.shape[1], 2, 2)

    # local → orig conversion (held-out)
    e_scale_l2o = _local_to_orig_scale(e_cs, cfg['img_size'])           # (B,1,1)
    e_inv_l2o   = (1.0 / e_scale_l2o).reshape(-1, 1, 1, 1)              # (B,1,1,1)
    e_duv_pred_orig = e_duv_pred_local * e_scale_l2o
    e_W_eye_orig    = e_W_eye_local    * e_inv_l2o.pow(2)
    e_W_sigma_orig  = e_W_sigma_local  * e_inv_l2o.pow(2)

    def _nll_per_sample(delta, delta_target, H):
        """½ (δ-δ*)ᵀ H (δ-δ*) − ½ log det H, averaged over batch.
        Treats target as a delta-spike, so KL(target ‖ N(δ, H⁻¹)) up to const = NLL.
        Lower is better."""
        diff = delta - delta_target
        maha = torch.einsum('bi,bij,bj->b', diff, H, diff)
        sign, logabsdet = torch.linalg.slogdet(H)
        return (0.5 * maha - 0.5 * logabsdet).mean().item(), \
               (0.5 * maha).mean().item(), \
               (0.5 * logabsdet).mean().item()

    def _nll_one(delta, delta_target, H):
        """Same scoring rule as `_nll_per_sample` for a single shared answer.
        Inputs: δ (K,), δ* (K,), H (K, K)."""
        diff = delta - delta_target
        maha = torch.einsum('i,ij,j->', diff, H, diff)
        sign, logabsdet = torch.linalg.slogdet(H)
        return (0.5 * maha - 0.5 * logabsdet).item(), \
               (0.5 * maha).item(), \
               (0.5 * logabsdet).item()

    with torch.no_grad():
        d_zero = torch.zeros_like(delta_target_eval)
        if args.shared:
            H_ref = torch.eye(len(DOFS), device=DEVICE)
            mse_zero = (d_zero - delta_target_eval).pow(2).mean().item()
            nll_zero, mh_zero, ld_zero = _nll_one(
                d_zero, delta_target_eval, H_ref)
        else:
            H_ref = torch.eye(len(DOFS), device=DEVICE).expand(
                args.n_eval, len(DOFS), len(DOFS))
            mse_zero = (d_zero - delta_target_eval).pow(2).mean(dim=-1).mean().item()
            nll_zero, mh_zero, ld_zero = _nll_per_sample(
                d_zero, delta_target_eval, H_ref)

        d_w1, H_w1 = _solve_with(e_P0_orig, e_duv_pred_orig, e_W_eye_orig,
                                  e_K_orig, e_dist, DOFS, e_valid,
                                  prior_diag=prior_diag, shared=args.shared)
        if args.shared:
            mse_w1 = (d_w1 - delta_target_eval).pow(2).mean().item()
            nll_w1, mh_w1, ld_w1 = _nll_one(d_w1, delta_target_eval, H_w1)
        else:
            mse_w1 = (d_w1 - delta_target_eval).pow(2).mean(dim=-1).mean().item()
            nll_w1, mh_w1, ld_w1 = _nll_per_sample(d_w1, delta_target_eval, H_w1)

        d_w2, H_w2 = _solve_with(e_P0_orig, e_duv_pred_orig, e_W_sigma_orig,
                                  e_K_orig, e_dist, DOFS, e_valid,
                                  prior_diag=prior_diag, shared=args.shared)
        if args.shared:
            mse_w2 = (d_w2 - delta_target_eval).pow(2).mean().item()
            nll_w2, mh_w2, ld_w2 = _nll_one(d_w2, delta_target_eval, H_w2)
        else:
            mse_w2 = (d_w2 - delta_target_eval).pow(2).mean(dim=-1).mean().item()
            nll_w2, mh_w2, ld_w2 = _nll_per_sample(d_w2, delta_target_eval, H_w2)
    print(f"[stream-orig] HELD-OUT BASELINE δ-MSE  "
          f"do-nothing: {mse_zero:.4e}  W=I: {mse_w1:.4e}   W=σ: {mse_w2:.4e}")
    print(f"[stream-orig] HELD-OUT BASELINE NLL    "
          f"do-nothing(H=I): {nll_zero:+.3f} (½M={mh_zero:+.3f} ½logdet={ld_zero:+.3f})")
    print(f"[stream-orig]                          "
          f"W=I:             {nll_w1:+.3f} (½M={mh_w1:+.3f} ½logdet={ld_w1:+.3f})")
    print(f"[stream-orig]                          "
          f"W=σ:             {nll_w2:+.3f} (½M={mh_w2:+.3f} ½logdet={ld_w2:+.3f})")
    if cml_logger is not None:
        cml_logger.report_single_value('mse_baseline_do_nothing', mse_zero)
        cml_logger.report_single_value('mse_baseline_W_I', mse_w1)
        cml_logger.report_single_value('mse_baseline_W_sigma', mse_w2)
        cml_logger.report_single_value('nll_baseline_do_nothing_HI', nll_zero)
        cml_logger.report_single_value('nll_baseline_W_I', nll_w1)
        cml_logger.report_single_value('nll_baseline_W_sigma', nll_w2)

    # ─── streaming train ──────────────────────────────────────────
    history = []
    t0 = time.time()
    info_head.train()
    for step in range(args.n_steps + 1):
        if args.shared:
            # 1 perturbation drawn ONCE per step, applied to B different
            # random train tiles. Solver collapses them into a single δ.
            step_pert = _draw_pert(rng, **pert_kw)
            wins = []
            tries = 0
            while len(wins) < args.batch and tries < 8 * args.batch:
                rand_idx = int(rng.randint(0, len(train_ds.fnames)))
                w = _sample_one(train_ds, rand_idx, rng,
                                pert=step_pert, **pert_kw)
                tries += 1
                if w is not None:
                    wins.append(w)
            batch = collate_full(wins) if len(wins) == args.batch else None
        elif args.multi:
            # build a batch of B samples, each from a DIFFERENT random train tile
            wins = []
            tries = 0
            while len(wins) < args.batch and tries < 8 * args.batch:
                rand_idx = int(rng.randint(0, len(train_ds.fnames)))
                w = _sample_one(train_ds, rand_idx, rng, **pert_kw)
                tries += 1
                if w is not None:
                    wins.append(w)
            batch = collate_full(wins) if len(wins) == args.batch else None
        else:
            batch = _build_batch(ds, base_idx, args.batch, rng, **pert_kw)
        if batch is None:
            print(f"  step {step}: skipped", flush=True)
            continue
        b = _move_batch(batch, DEVICE)
        (imgs, true_uvd, dist_uvd, pad_mask, vfp,
         bucket_uvd, bucket_valid, _,
         pts_cam_orig, duv_orig, K_orig, cs) = b
        (valid, duv_oracle_local, duv_oracle_orig,
         P0_orig, K_orig, cs) = _prep_inputs(
            true_uvd, dist_uvd, pad_mask,
            pts_cam_orig, duv_orig, K_orig, cs)
        b_dist = dist_one.to(DEVICE).expand(P0_orig.shape[0], 4).contiguous()
        delta_target = _solve_target(P0_orig, duv_oracle_orig, valid, K_orig,
                                      b_dist, DOFS, prior_diag=prior_diag,
                                      shared=args.shared)

        per_pt, q = _forward_capture_q(
            model, imgs, dist_uvd, pad_mask, vfp, bucket_uvd, bucket_valid,
            no_grad=not args.unfreeze)
        duv_pred_local = per_pt[..., :2]
        if not args.unfreeze:
            duv_pred_local = duv_pred_local.detach()
        if (~valid).any():
            duv_pred_local = torch.where(
                valid.unsqueeze(-1), duv_pred_local,
                torch.zeros_like(duv_pred_local))

        # info_head is trained — single live forward through it (graph kept).
        W_learn_local = info_head(q)

        scale_l2o = _local_to_orig_scale(cs, cfg['img_size'])
        inv_l2o = (1.0 / scale_l2o).reshape(-1, 1, 1, 1)
        duv_pred_orig = duv_pred_local * scale_l2o
        W_learn_orig  = W_learn_local  * inv_l2o.pow(2)

        delta_pred, H_pred = _solve_with(
            P0_orig, duv_pred_orig, W_learn_orig, K_orig, b_dist, DOFS, valid,
            prior_diag=prior_diag, shared=args.shared)
        diff = delta_pred - delta_target
        if args.shared:
            if args.pose_loss == 'mse':
                pose_loss = diff.pow(2).mean()
            else:  # nll
                maha = torch.einsum('i,ij,j->', diff, H_pred, diff)
                sign, logabsdet = torch.linalg.slogdet(H_pred)
                pose_loss = 0.5 * maha - 0.5 * logabsdet
        elif args.pose_loss == 'mse':
            pose_loss = diff.pow(2).sum(dim=-1).mean()
        else:  # nll
            maha = torch.einsum('bi,bij,bj->b', diff, H_pred, diff)
            sign, logabsdet = torch.linalg.slogdet(H_pred)
            pose_loss = (0.5 * maha - 0.5 * logabsdet).mean()
        loss = pose_loss
        aux_val = 0.0
        if args.unfreeze and args.perpt_weight > 0.0:
            r = per_pt[..., :2] - duv_oracle_local
            aux = r.pow(2).sum(dim=-1)
            aux = aux.masked_select(valid).mean()
            loss = loss + args.perpt_weight * aux
            aux_val = float(aux.detach().item())
        opt.zero_grad(set_to_none=True)
        loss.backward()
        if args.unfreeze:
            torch.nn.utils.clip_grad_norm_(opt_params, 1.0)
        else:
            torch.nn.utils.clip_grad_norm_(info_head.parameters(), 1.0)
        opt.step()

        if step % LOG_EVERY == 0 or step == args.n_steps:
            info_head.eval()
            if args.unfreeze:
                model.eval()
            with torch.no_grad():
                # re-forward backbone each eval if it's been training.
                if args.unfreeze:
                    e_per_pt_now, e_q_now = _forward_capture_q(
                        model, e_imgs, e_dist_uvd, e_pad, e_vfp,
                        e_bucket_uvd, e_bucket_valid, no_grad=True)
                    e_duv_pred_local_now = e_per_pt_now[..., :2]
                    e_padfull_now = ~e_valid
                    if e_padfull_now.any():
                        e_duv_pred_local_now = e_duv_pred_local_now.clone()
                        e_duv_pred_local_now[e_padfull_now] = 0.0
                    e_duv_pred_orig_now = e_duv_pred_local_now * e_scale_l2o
                else:
                    e_q_now = e_q
                    e_duv_pred_orig_now = e_duv_pred_orig
                W_l_local = info_head(e_q_now)
                W_l_orig  = W_l_local * e_inv_l2o.pow(2)
                d_l, H_l = _solve_with(
                    e_P0_orig, e_duv_pred_orig_now, W_l_orig,
                    e_K_orig, e_dist, DOFS, e_valid,
                    prior_diag=prior_diag, shared=args.shared)
                if args.shared:
                    eval_mse = (d_l - delta_target_eval).pow(2).mean().item()
                    eval_nll, eval_mh, eval_ld = _nll_one(
                        d_l, delta_target_eval, H_l)
                else:
                    eval_mse = (d_l - delta_target_eval).pow(2).mean(dim=-1).mean().item()
                    eval_nll, eval_mh, eval_ld = _nll_per_sample(
                        d_l, delta_target_eval, H_l)
                logdet = torch.linalg.det(W_l_orig).clamp_min(1e-12).log().mean().item()
            info_head.train()
            if args.unfreeze:
                model.train()
            sps = (step + 1) / max(1e-3, time.time() - t0)
            history.append((step, loss.item(), eval_mse, logdet, eval_nll))
            aux_str = f"  perpt_MSE={aux_val:.3e}" if args.unfreeze and args.perpt_weight > 0.0 else ""
            print(f"  step {step:>4}  loss={loss.item():.4e}{aux_str}  "
                  f"EVAL δ-MSE={eval_mse:.4e} (vs0={mse_zero:.4e} σ={mse_w2:.4e})  "
                  f"NLL={eval_nll:+.3f} (vs0={nll_zero:+.3f} σ={nll_w2:+.3f})  "
                  f"½M={eval_mh:+.3f} ½logdetH={eval_ld:+.3f}  "
                  f"⟨log det W⟩={logdet:+.2f}  "
                  f"({sps:.2f} step/s)", flush=True)
            if cml_logger is not None:
                cml_logger.report_scalar('loss', 'train', loss.item(), step)
                cml_logger.report_scalar('mse', 'W_learned', eval_mse, step)
                cml_logger.report_scalar('mse', 'do_nothing', mse_zero, step)
                cml_logger.report_scalar('mse', 'W_I', mse_w1, step)
                cml_logger.report_scalar('mse', 'W_sigma', mse_w2, step)
                cml_logger.report_scalar('nll', 'W_learned', eval_nll, step)
                cml_logger.report_scalar('nll', 'do_nothing', nll_zero, step)
                cml_logger.report_scalar('nll', 'W_I', nll_w1, step)
                cml_logger.report_scalar('nll', 'W_sigma', nll_w2, step)
                cml_logger.report_scalar('logdet_W_orig', 'mean',
                                          logdet, step)

    # ─── FINAL eval ────────────────────────────────────────────────
    info_head.eval()
    if args.unfreeze:
        model.eval()
    with torch.no_grad():
        if args.unfreeze:
            e_per_pt_fin, e_q_fin = _forward_capture_q(
                model, e_imgs, e_dist_uvd, e_pad, e_vfp,
                e_bucket_uvd, e_bucket_valid, no_grad=True)
            e_duv_pred_local_fin = e_per_pt_fin[..., :2]
            e_padfull_fin = ~e_valid
            if e_padfull_fin.any():
                e_duv_pred_local_fin = e_duv_pred_local_fin.clone()
                e_duv_pred_local_fin[e_padfull_fin] = 0.0
            e_duv_pred_orig_fin = e_duv_pred_local_fin * e_scale_l2o
        else:
            e_q_fin = e_q
            e_duv_pred_orig_fin = e_duv_pred_orig
        W_final_local = info_head(e_q_fin)
        W_final_orig  = W_final_local * e_inv_l2o.pow(2)
        d_final, H_final = _solve_with(
            e_P0_orig, e_duv_pred_orig_fin, W_final_orig,
            e_K_orig, e_dist, DOFS, e_valid,
            prior_diag=prior_diag, shared=args.shared)
        if args.shared:
            mse_w3 = (d_final - delta_target_eval).pow(2).mean().item()
            nll_w3, mh_w3, ld_w3 = _nll_one(
                d_final, delta_target_eval, H_final)
        else:
            mse_w3 = (d_final - delta_target_eval).pow(2).mean(dim=-1).mean().item()
            nll_w3, mh_w3, ld_w3 = _nll_per_sample(
                d_final, delta_target_eval, H_final)
    print(f"\n[stream-orig] HELD-OUT FINAL δ-MSE  do-nothing: {mse_zero:.4e}  "
          f"W=I: {mse_w1:.4e}   W=σ: {mse_w2:.4e}   W=learned: {mse_w3:.4e}")
    print(f"[stream-orig] HELD-OUT FINAL NLL    do-nothing(H=I): {nll_zero:+.3f}  "
          f"W=I: {nll_w1:+.3f}   W=σ: {nll_w2:+.3f}   "
          f"W=learned: {nll_w3:+.3f} (½M={mh_w3:+.3f} ½logdet={ld_w3:+.3f})")
    print(f"[stream-orig] PASS vs do-nothing(MSE): {mse_w3 < mse_zero}   "
          f"PASS vs uniform: {mse_w3 < mse_w1}   "
          f"PASS vs σ: {mse_w3 <= mse_w2 * 1.05}")
    print(f"[stream-orig] PASS vs do-nothing(NLL): {nll_w3 < nll_zero}   "
          f"PASS vs σ(NLL): {nll_w3 < nll_w2}")

    # ─── plot ──────────────────────────────────────────────────────
    steps  = [h[0] for h in history]
    trnL   = [h[1] for h in history]
    evMSE  = [h[2] for h in history]

    fig, axes = plt.subplots(1, 2, figsize=(13, 4.4))
    ax = axes[0]
    ax.semilogy(steps, evMSE, '-o', color='tab:red', ms=3,
                 label=f'learned W (held-out N={args.n_eval})')
    ax.semilogy(steps, trnL,  '-x', color='tab:orange', ms=3, alpha=0.5,
                 label='train loss')
    ax.axhline(mse_zero, ls=':',  color='tab:green',  label=f'do-nothing ({mse_zero:.2e})')
    ax.axhline(mse_w1,   ls='--', color='tab:gray',   label=f'W=I        ({mse_w1:.2e})')
    ax.axhline(mse_w2,   ls='--', color='tab:purple', label=f'W=σ-head ({mse_w2:.2e})')
    ax.set_xlabel('step'); ax.set_ylabel('δ-MSE')
    ax.set_title(f'(a) 1-frame STREAM (orig solver) — idx={base_idx}, {args.dofs}')
    ax.grid(which='both', alpha=0.3); ax.legend(loc='best', fontsize=8)

    ax = axes[1]
    bars = ['do-nothing', 'W=I', 'W=σ', 'W=learn']
    vals = [mse_zero, mse_w1, mse_w2, mse_w3]
    colors = ['tab:green', 'tab:gray', 'tab:purple', 'tab:red']
    ax.bar(bars, vals, color=colors)
    for i, v in enumerate(vals):
        ax.text(i, v, f'{v:.2e}', ha='center', va='bottom', fontsize=8)
    ax.set_yscale('log')
    ax.set_ylabel('δ-MSE')
    ax.set_title(f'(b) final δ-MSE on held-out N={args.n_eval}\n'
                 f'{args.n_steps} steps × B={args.batch}')
    ax.grid(axis='y', alpha=0.3)

    fig.suptitle(
        f'Phase 1.b-orig: 1-frame STREAM perts on val tile {base_idx} '
        f'({args.dofs}, orig-cam solver)\n'
        f'Frozen CalibNetDepth + InfoHead2x2; '
        f'rot=±{args.rot_deg}deg t=±{args.t_m}m '
        f'dfx=±{args.dfx_pct} dfy=±{args.dfy_pct}',
        y=1.04, fontsize=10,
    )
    fig.tight_layout()
    out_png = out_dir / 'curves.png'
    fig.savefig(out_png, dpi=110, bbox_inches='tight')
    plt.close(fig)
    print(f"[stream-orig] wrote → {out_png}")

    sd_out = out_dir / 'info_head.pt'
    torch.save(info_head.state_dict(), sd_out)
    summary = {
        'idx': base_idx, 'dofs': args.dofs, 'DOFS': DOFS,
        'n_steps': args.n_steps, 'batch': args.batch, 'n_eval': args.n_eval,
        'rot_deg': args.rot_deg, 't_m': args.t_m,
        'dfx_pct': args.dfx_pct, 'dfy_pct': args.dfy_pct,
        'mse_zero': mse_zero, 'mse_W_I': mse_w1,
        'mse_W_sigma': mse_w2, 'mse_W_learned': mse_w3,
        'history': history,
    }
    torch.save(summary, out_dir / 'summary.pt')
    print(f"[stream-orig] wrote → {sd_out}, summary.pt")

    print("[stream-orig] done.")


if __name__ == '__main__':
    main()
