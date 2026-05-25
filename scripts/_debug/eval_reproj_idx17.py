"""Sanity check: does the 30% δ-MSE win actually reduce REPROJECTION error?

Loads the info_head.pt saved by overfit_6dof_ba_stream_orig.py
(default: scripts/_debug/_outputs/overfit_6dof_ba_stream_orig_idx17_6dof_kb/info_head.pt)
and measures pixel reprojection error after applying the solved δ to P0:

  err_px = || project(R(δ)·P0 + t(δ); K, dist) - (uv0 + duv_oracle) ||

for four variants:
  - do-nothing : δ = 0
  - W=I        : solver with identity weight
  - W=σ        : solver with σ-head info matrix (trained NLL legacy way)
  - W=learned  : solver with info_head from the run

Prints per-quantile px error and saves a 4-panel overlay on idx=17 showing
target_uv (yellow ✕), projected uv (green ○), and the residual line, so we
can eyeball whether the truck region (the foreground tile content the user
cares about) is actually sub-pixel under W=learned.
"""
from __future__ import annotations
import argparse
import sys
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
    solve_kb_xyz, make_info_from_sigma_rho, project_kb,
    _apply_extrinsic, _K_with_delta, _split_delta,
)

CACHE = Path.home() / 'cache' / 'kamikado_v3_tiled'
CKPT  = REPO / 'experiments' / 'km_wv_wm_dgx2_n4_img128_8gpu_HEAD' / 'best_model.pt'
EXP_CFG_PATH = REPO / 'experiments' / 'km_wv_wm_dgx2_n4_img128_8gpu_HEAD' / 'config.py'
DEFAULT_HEAD = (REPO / 'scripts' / '_debug' / '_outputs'
                / 'overfit_6dof_ba_stream_orig_idx17_6dof_kb' / 'info_head.pt')
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

DOF_PRESETS = {
    '2dof':   ['omega_x', 'omega_y'],
    '3dof':   ['omega_x', 'omega_y', 'omega_z'],
    '6dof':   ['omega_x', 'omega_y', 'omega_z', 'tx', 'ty', 'tz'],
}


def _load_cfg():
    src = EXP_CFG_PATH.read_text(); ns = {}; exec(src, ns, ns); return ns['CFG']


def _build_model(cfg):
    return CalibNetDepth(
        img_size=cfg['img_size'], in_channels=cfg['in_channels'],
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


def _apply_delta_and_project(P0_orig, delta, K_orig, dist, dofs):
    """Forward apply δ to (P0, K) and project with KB.  Returns uv (B,N,2)."""
    d = _split_delta(delta, dofs)
    omega = torch.stack([d['omega_x'], d['omega_y'], d['omega_z']], dim=-1)
    t_v   = torch.stack([d['tx'], d['ty'], d['tz']], dim=-1)
    P_lin = _apply_extrinsic(P0_orig, omega, t_v)
    K_lin = _K_with_delta(K_orig, d['dfx'], d['dfy'], d['dcx'], d['dcy'])
    return project_kb(P_lin, K_lin, dist)


def _reproj_err(uv_pred_orig, uv_target_orig, valid):
    """Per-point pixel residual norm. Returns (B,N) with NaN at invalids."""
    err = torch.linalg.vector_norm(uv_pred_orig - uv_target_orig, dim=-1)
    err = err.masked_fill(~valid, float('nan'))
    return err


def _build_eval_batch(ds, idx, n_eval, *, rot_deg, t_m, seed):
    rng = np.random.RandomState(seed)
    wins = []
    while len(wins) < n_eval:
        ox = float(rng.uniform(-rot_deg, rot_deg))
        oy = float(rng.uniform(-rot_deg, rot_deg))
        oz = float(rng.uniform(-rot_deg, rot_deg))
        ypr = np.array([oz, oy, ox], dtype=np.float64)
        t = (rng.uniform(-1.0, 1.0, size=3) * t_m).astype(np.float64) \
            if t_m > 0 else np.zeros(3)
        win = ds.apply_perturbation_explicit(idx, t, ypr)
        if win is None:
            continue
        wins.append(win)
    return collate_full(wins)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--idx', type=int, default=17)
    ap.add_argument('--head-pt', type=Path, default=DEFAULT_HEAD)
    ap.add_argument('--dofs', type=str, default='6dof', choices=list(DOF_PRESETS.keys()))
    ap.add_argument('--rot-deg', type=float, default=0.30)
    ap.add_argument('--t-m', type=float, default=0.05)
    ap.add_argument('--n-eval', type=int, default=200)
    ap.add_argument('--seed', type=int, default=7 + 1000,
                    help='same RNG as overfit script eval_rng')
    ap.add_argument('--out', type=Path,
                    default=REPO / 'scripts' / '_debug' / '_outputs'
                            / 'reproj_check_idx17.png')
    args = ap.parse_args()

    cfg = _load_cfg()
    DOFS = DOF_PRESETS[args.dofs]
    print(f"[reproj] head={args.head_pt}")
    print(f"[reproj] dofs={args.dofs}  rot=±{args.rot_deg}deg  t=±{args.t_m}m  "
          f"N={args.n_eval}  idx={args.idx}")

    ds = PandaSetCalibDatasetFull(
        cache_dir=CACHE, split='val',
        img_size=cfg['img_size'],
        min_crop_px=cfg['min_crop_px'], max_crop_px=cfg['max_crop_px'],
        max_offset_m=0.0, max_rot_deg=0.0,
        oversample=1, grid_n=cfg.get('grid_n', 16),
        center_band=0.0, preload=False,
    )
    inst0 = ds._load_inst(int(args.idx))
    assert inst0.get('is_fisheye', False), 'idx={} not fisheye'.format(args.idx)
    dist_one = inst0['distortion'].clone().detach().to(torch.float32).reshape(1, 4)

    model = _build_model(cfg).to(DEVICE)
    sd = torch.load(CKPT, map_location='cpu', weights_only=False)
    if isinstance(sd, dict) and ('state_dict' in sd or 'model' in sd):
        sd = sd.get('state_dict') or sd.get('model')
    model.load_state_dict(sd, strict=False)
    head_sd = torch.load(args.head_pt, map_location=DEVICE, weights_only=False)
    model.info_head.load_state_dict(head_sd)
    model.eval()
    for p in model.parameters():
        p.requires_grad_(False)
    print(f"[reproj] loaded ckpt + info_head from {args.head_pt}")

    # Build the same held-out eval batch the train script used
    batch = _build_eval_batch(ds, int(args.idx), args.n_eval,
                               rot_deg=args.rot_deg, t_m=args.t_m,
                               seed=args.seed)
    moved = [t.to(DEVICE) if torch.is_tensor(t) else t for t in batch]
    (imgs, true_uvd, dist_uvd, pad_mask, vfp,
     bucket_uvd, bucket_valid, _,
     pts_cam_orig, duv_orig, K_orig, cs) = moved
    valid = ~pad_mask
    pad_full = ~valid
    B, N = pts_cam_orig.shape[:2]
    P0_orig = pts_cam_orig.detach().clone()
    duv_oracle_orig = duv_orig.detach().clone()
    if pad_full.any():
        duv_oracle_orig[pad_full] = 0.0
        P0_orig[pad_full] = torch.tensor([0.0, 0.0, 1.0],
                                          dtype=P0_orig.dtype,
                                          device=P0_orig.device)
    dist = dist_one.to(DEVICE).expand(B, 4).contiguous()

    # ---- forward (capture q for info_head) ---------------------------
    captured = {}
    def _hk(_m, _i, _o):
        captured['q'] = _i[0].detach()
    h = model.info_head.mlp[0].register_forward_hook(_hk)
    point_in = torch.cat([dist_uvd[..., :3], dist_uvd[..., 4:5]], dim=-1)
    img_norm = imgs.float().div(255.0)
    with torch.no_grad():
        out = model(img_norm, point_in, key_padding_mask=pad_mask, vfp=vfp,
                    bucket_uvd=bucket_uvd, bucket_valid=bucket_valid)
    h.remove()
    per_pt = out[0]
    q = captured['q']

    duv_pred_local = per_pt[..., :2].detach()
    if pad_full.any():
        duv_pred_local = duv_pred_local.clone()
        duv_pred_local[pad_full] = 0.0
    sx = per_pt[..., 2].exp(); sy = per_pt[..., 3].exp(); rho = per_pt[..., 4]
    W_sigma_local = make_info_from_sigma_rho(sx, sy, rho).detach()
    eye2 = torch.eye(2, device=DEVICE).expand(B, N, 2, 2)
    W_eye_local = eye2.clone()
    with torch.no_grad():
        W_learn_local = model.info_head(q).detach()

    scale_l2o = (cs / float(cfg['img_size'])).reshape(-1, 1, 1)         # (B,1,1)
    inv_l2o = (1.0 / scale_l2o).reshape(-1, 1, 1, 1)
    duv_pred_orig = duv_pred_local * scale_l2o
    W_eye_orig    = W_eye_local    * inv_l2o.pow(2)
    W_sigma_orig  = W_sigma_local  * inv_l2o.pow(2)
    W_learn_orig  = W_learn_local  * inv_l2o.pow(2)

    # ---- target: uv0 + duv_oracle (in original-px coords) ------------
    uv0_orig = project_kb(P0_orig, K_orig, dist)
    uv_target_orig = uv0_orig + duv_oracle_orig

    # ---- solve under each W -------------------------------------------
    solver_kw = dict(n_iter=6, damping=1e-3, valid=valid)
    with torch.no_grad():
        delta_zero = torch.zeros(B, len(DOFS), device=DEVICE)
        delta_wI, _   = solve_kb_xyz(P0_orig, duv_pred_orig, W_eye_orig,
                                      K_orig, dist, DOFS, **solver_kw)
        delta_wS, _   = solve_kb_xyz(P0_orig, duv_pred_orig, W_sigma_orig,
                                      K_orig, dist, DOFS, **solver_kw)
        delta_wL, _   = solve_kb_xyz(P0_orig, duv_pred_orig, W_learn_orig,
                                      K_orig, dist, DOFS, **solver_kw)
        # also: oracle (target Δuv exactly + W=I) — best the geometry permits
        delta_oracle, _ = solve_kb_xyz(P0_orig, duv_oracle_orig, W_eye_orig,
                                        K_orig, dist, DOFS, **solver_kw)

    # ---- compute reprojection error -----------------------------------
    def _eval_variant(name, delta):
        uv_pred = _apply_delta_and_project(P0_orig, delta, K_orig, dist, DOFS)
        err = _reproj_err(uv_pred, uv_target_orig, valid)
        flat = err[~torch.isnan(err)]
        return uv_pred, flat

    variants = [
        ('do-nothing',       delta_zero),
        ('W=I',              delta_wI),
        ('W=σ',              delta_wS),
        ('W=learned',        delta_wL),
        ('oracle (Δuv=GT)',  delta_oracle),
    ]
    rows = []
    for name, dl in variants:
        uv_pred, flat = _eval_variant(name, dl)
        flat_np = flat.cpu().numpy()
        rows.append((name, dl, uv_pred, flat_np))

    print()
    print(f"  {'variant':18s}  {'mean':>8s}  {'median':>8s}  "
          f"{'p90':>8s}  {'p95':>8s}  {'max':>8s}     [orig-px reproj err]")
    for name, _dl, _uv, flat_np in rows:
        if flat_np.size == 0:
            print(f"  {name:18s}  (no valid points)")
            continue
        print(f"  {name:18s}  "
              f"{flat_np.mean():8.3f}  "
              f"{np.median(flat_np):8.3f}  "
              f"{np.percentile(flat_np, 90):8.3f}  "
              f"{np.percentile(flat_np, 95):8.3f}  "
              f"{flat_np.max():8.3f}")
    print()

    # ---- viz on a single sample (sample 0) -----------------------------
    img0 = imgs[0].permute(1, 2, 0).cpu().numpy().astype('uint8')
    S = img0.shape[0]
    cs0 = float(cs[0].item())
    # convert orig-px → local-px for display
    o2l = float(S) / cs0
    valid0 = valid[0].cpu().numpy()
    uv_target_local = (uv_target_orig[0].cpu().numpy() * o2l)
    uv0_local = (uv0_orig[0].cpu().numpy() * o2l)

    fig, axes = plt.subplots(1, len(rows), figsize=(4.0 * len(rows), 4.4))
    for ax, (name, _dl, uv_pred_orig_all, _flat_np) in zip(axes, rows):
        ax.imshow(img0)
        uv_pred_local = uv_pred_orig_all[0].cpu().numpy() * o2l
        # residual lines: target → projected (showing how far off we still are)
        for k in np.where(valid0)[0]:
            ax.plot([uv_target_local[k, 0], uv_pred_local[k, 0]],
                    [uv_target_local[k, 1], uv_pred_local[k, 1]],
                    color='cyan', lw=0.4, alpha=0.6, zorder=2)
        ax.scatter(uv0_local[valid0, 0], uv0_local[valid0, 1],
                    s=14, facecolors='none', edgecolors='red',
                    linewidths=0.7, zorder=4, label='uv0 (dist input)')
        ax.scatter(uv_target_local[valid0, 0], uv_target_local[valid0, 1],
                    s=22, c='yellow', marker='x', linewidths=1.0,
                    zorder=6, label='uv_target (oracle)')
        ax.scatter(uv_pred_local[valid0, 0], uv_pred_local[valid0, 1],
                    s=14, facecolors='none', edgecolors='lime',
                    linewidths=0.9, zorder=7, label='uv after solve')
        # per-sample reproj err
        err0 = _reproj_err(uv_pred_orig_all[:1],
                            uv_target_orig[:1], valid[:1])
        flat0 = err0[~torch.isnan(err0)].cpu().numpy()
        m = flat0.mean() if flat0.size else float('nan')
        med = np.median(flat0) if flat0.size else float('nan')
        ax.set_xlim(0, S); ax.set_ylim(S, 0); ax.axis('off')
        ax.set_title(f'{name}\nreproj px (sample 0)  mean {m:.2f}  med {med:.2f}',
                     fontsize=10)
    fig.suptitle(
        f'Reprojection check on val tile {args.idx} ({ds.fnames[args.idx]})  '
        f'— {args.dofs}, rot=±{args.rot_deg}°, t=±{args.t_m}m, N={args.n_eval}\n'
        f'cyan line = residual (target - solved-projection)',
        y=1.02, fontsize=11,
    )
    fig.tight_layout()
    args.out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.out, dpi=120, bbox_inches='tight')
    plt.close(fig)
    print(f"[reproj] wrote → {args.out}")


if __name__ == '__main__':
    main()
