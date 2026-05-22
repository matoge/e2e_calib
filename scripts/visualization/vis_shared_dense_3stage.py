"""3-stage DENSE reproj overlay on a single kamikado tile, using a δ
solved from N=200 random val tiles aggregated through a SHARED Gauss-
Newton step (solve_kb_xyz_shared).

Mirrors the OUT_DENSE figure in scripts/_debug/ba_one_frame_vis.py
(yellow=GT / red=perturbed / green=corrected) but instead of solving
on 1 scene, the δ comes from B=N_shared tiles → so this is the
'200 frames lift the aperture' version.

W is the σ-head from the frozen ckpt (no info_head trained tonight).
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
    solve_kb_xyz_shared, make_info_from_sigma_rho, project_kb,
    _apply_extrinsic, _K_with_delta, _split_delta,
)

CACHE = Path.home() / 'cache' / 'kamikado_v3_tiled'
CKPT  = REPO / 'experiments' / 'km_wv_wm_dgx2_n4_img128_8gpu_HEAD' / 'best_model.pt'
EXP_CFG_PATH = REPO / 'experiments' / 'km_wv_wm_dgx2_n4_img128_8gpu_HEAD' / 'config.py'
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

DOFS = ['omega_x', 'omega_y', 'omega_z', 'tx', 'ty', 'tz']
PRIOR_DIAG = torch.tensor(
    [1.0/9.0, 1.0/9.0, 1.0/9.0, 25.0, 25.0, 25.0], dtype=torch.float32)
BA_N_ITER = 6
DAMPING   = 1e-3


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


def _draw_pert(rng, *, rot_deg, t_m):
    ox = float(rng.uniform(-rot_deg, rot_deg))
    oy = float(rng.uniform(-rot_deg, rot_deg))
    oz = float(rng.uniform(-rot_deg, rot_deg))
    ypr = np.array([oz, oy, ox], dtype=np.float64)
    t = (rng.uniform(-1.0, 1.0, size=3) * t_m).astype(np.float64) \
        if t_m > 0.0 else np.zeros(3, dtype=np.float64)
    return ypr, t


def _project_with_delta(P0_orig, delta, K_orig, dist):
    """Apply a single shared δ (K,) to a batched P0 (B,N,3) and project."""
    delta_b = delta.unsqueeze(0).expand(P0_orig.shape[0], -1)
    d = _split_delta(delta_b, DOFS)
    omega = torch.stack([d['omega_x'], d['omega_y'], d['omega_z']], dim=-1)
    t_v   = torch.stack([d['tx'], d['ty'], d['tz']], dim=-1)
    P_lin = _apply_extrinsic(P0_orig, omega, t_v)
    K_lin = _K_with_delta(K_orig, d['dfx'], d['dfy'], d['dcx'], d['dcy'])
    return project_kb(P_lin, K_lin, dist)


def _per_sample_err(uv_pred, uv_target, valid):
    e = torch.linalg.vector_norm(uv_pred - uv_target, dim=-1)
    e = e.masked_fill(~valid, float('nan'))
    return e[~torch.isnan(e)]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--idx', type=int, default=17,
                    help='val tile to draw (panels show this tile only)')
    ap.add_argument('--n-shared', type=int, default=200)
    ap.add_argument('--rot-deg', type=float, default=0.30)
    ap.add_argument('--t-m',     type=float, default=0.05)
    ap.add_argument('--seed',    type=int,   default=7 + 1000)
    ap.add_argument('--out', type=Path,
                    default=REPO / 'scripts' / '_debug' / '_outputs'
                            / 'vis_shared_dense_3stage_idx17.png')
    args = ap.parse_args()

    cfg = _load_cfg()
    print(f"[3stage] idx={args.idx}  N_shared={args.n_shared}  "
          f"rot=±{args.rot_deg}°  t=±{args.t_m}m  seed={args.seed}")

    ds = PandaSetCalibDatasetFull(
        cache_dir=CACHE, split='val',
        img_size=cfg['img_size'],
        min_crop_px=cfg['min_crop_px'], max_crop_px=cfg['max_crop_px'],
        max_offset_m=0.0, max_rot_deg=0.0,
        oversample=1, grid_n=cfg.get('grid_n', 16),
        center_band=0.0, preload=False,
    )
    inst0 = ds._load_inst(int(args.idx))
    assert inst0.get('is_fisheye', False), f'idx={args.idx} not fisheye'
    dist_one = inst0['distortion'].clone().detach().to(torch.float32).reshape(1, 4)

    model = _build_model(cfg).to(DEVICE)
    sd = torch.load(CKPT, map_location='cpu', weights_only=False)
    if isinstance(sd, dict) and ('state_dict' in sd or 'model' in sd):
        sd = sd.get('state_dict') or sd.get('model')
    model.load_state_dict(sd, strict=False)
    model.eval()
    for p in model.parameters():
        p.requires_grad_(False)

    rng = np.random.RandomState(args.seed)
    ypr_shared, t_shared = _draw_pert(rng, rot_deg=args.rot_deg, t_m=args.t_m)
    print(f"[3stage] shared δ_target  ypr={ypr_shared}  t={t_shared}")
    print(f"[3stage]   (=[ω_x={ypr_shared[2]:+.4f}° ω_y={ypr_shared[1]:+.4f}° "
          f"ω_z={ypr_shared[0]:+.4f}° tx={t_shared[0]:+.4f}m "
          f"ty={t_shared[1]:+.4f}m tz={t_shared[2]:+.4f}m])")

    # target tile = sample 0, then N-1 random val tiles
    wins = [ds.apply_perturbation_explicit(int(args.idx), t_shared, ypr_shared)]
    assert wins[0] is not None, f'target idx={args.idx} returned None'
    tries = 0
    while len(wins) < args.n_shared and tries < 16 * args.n_shared:
        ridx = int(rng.randint(0, len(ds.fnames)))
        w = ds.apply_perturbation_explicit(ridx, t_shared, ypr_shared)
        tries += 1
        if w is not None:
            wins.append(w)
    assert len(wins) == args.n_shared, \
        f"could not build shared batch ({len(wins)}/{args.n_shared})"

    moved = [t.to(DEVICE) if torch.is_tensor(t) else t for t in collate_full(wins)]
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

    point_in = torch.cat([dist_uvd[..., :3], dist_uvd[..., 4:5]], dim=-1)
    img_norm = imgs.float().div(255.0)
    with torch.no_grad():
        out = model(img_norm, point_in, key_padding_mask=pad_mask, vfp=vfp,
                    bucket_uvd=bucket_uvd, bucket_valid=bucket_valid)
    per_pt = out[0]
    duv_pred_local = per_pt[..., :2].detach()
    if pad_full.any():
        duv_pred_local = duv_pred_local.clone()
        duv_pred_local[pad_full] = 0.0
    sx  = per_pt[..., 2].exp()
    sy  = per_pt[..., 3].exp()
    rho = per_pt[..., 4]
    W_sigma_local = make_info_from_sigma_rho(sx, sy, rho).detach()

    scale_l2o = (cs / float(cfg['img_size'])).reshape(-1, 1, 1)
    inv_l2o = (1.0 / scale_l2o).reshape(-1, 1, 1, 1)
    duv_pred_orig = duv_pred_local * scale_l2o
    W_sigma_orig  = W_sigma_local  * inv_l2o.pow(2)

    prior = PRIOR_DIAG.to(DEVICE)
    with torch.no_grad():
        delta_shared, _ = solve_kb_xyz_shared(
            P0_orig, duv_pred_orig, W_sigma_orig, K_orig, dist, DOFS,
            valid=valid, n_iter=BA_N_ITER, damping=DAMPING, prior_diag=prior,
        )
    print(f"[3stage] solved δ̂           = {delta_shared.cpu().numpy()}")

    # uv tensors for sample 0 only
    uv0_orig         = project_kb(P0_orig, K_orig, dist)
    uv_target_orig   = uv0_orig + duv_oracle_orig
    uv_corrected_orig = _project_with_delta(P0_orig, delta_shared, K_orig, dist)

    err_pert = _per_sample_err(uv0_orig[0], uv_target_orig[0], valid[0])
    err_corr = _per_sample_err(uv_corrected_orig[0], uv_target_orig[0], valid[0])
    print(f"[3stage] sample0 reproj-px  perturbed: mean={err_pert.mean().item():.2f} "
          f"med={err_pert.median().item():.2f}  ||  "
          f"corrected: mean={err_corr.mean().item():.2f} "
          f"med={err_corr.median().item():.2f}")

    img0 = imgs[0].permute(1, 2, 0).cpu().numpy().astype('uint8')
    S = img0.shape[0]
    cs0 = float(cs[0].item())
    o2l = float(S) / cs0
    valid0 = valid[0].cpu().numpy()

    uv_target_local    = (uv_target_orig[0].cpu().numpy() * o2l)
    uv0_local          = (uv0_orig[0].cpu().numpy() * o2l)
    uv_corrected_local = (uv_corrected_orig[0].cpu().numpy() * o2l)

    panels = [
        (f'GT  ({int(valid0.sum())} pts)',
            uv_target_local,    'yellow'),
        (f'perturbed   |reproj|  mean={err_pert.mean().item():.2f}px '
         f'med={err_pert.median().item():.2f}',
            uv0_local,          'red'),
        (f'shared {args.n_shared}-tile BA-corrected  '
         f'δ̂[ω={delta_shared[0].item():+.3f},{delta_shared[1].item():+.3f},'
         f'{delta_shared[2].item():+.3f}° '
         f't={delta_shared[3].item():+.3f},{delta_shared[4].item():+.3f},'
         f'{delta_shared[5].item():+.3f}m]  '
         f'|reproj| mean={err_corr.mean().item():.2f}px '
         f'med={err_corr.median().item():.2f}',
            uv_corrected_local, 'lime'),
    ]
    fig, axes = plt.subplots(3, 1, figsize=(S/120, 3*S/120), dpi=200)
    for ax, (title, uv_local, colr) in zip(axes, panels):
        ax.imshow(img0)
        ax.scatter(uv_local[valid0, 0], uv_local[valid0, 1],
                    s=14, facecolors='none', edgecolors=colr,
                    linewidths=0.9, zorder=5)
        ax.set_xlim(0, S); ax.set_ylim(S, 0); ax.axis('off')
        ax.set_title(title, fontsize=9)
    fig.suptitle(
        f'Shared {args.n_shared}-tile GN δ on val tile {args.idx} '
        f'({ds.fnames[args.idx]})\n'
        f'frozen σ-head W, 6-DoF, rot=±{args.rot_deg}° t=±{args.t_m}m',
        y=1.0, fontsize=10,
    )
    fig.tight_layout()
    args.out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.out, dpi=200, bbox_inches='tight')
    plt.close(fig)
    print(f"[3stage] wrote → {args.out}")


if __name__ == '__main__':
    main()
