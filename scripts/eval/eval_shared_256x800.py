"""200 random val instances × 4 sub-crops (256×256 each) = 800 tiles, all
sharing one rig-level δ_target. Solve δ̂ via frozen σ-head + shared GN, compare
against the existing 200×512 baseline.

Why: per-tile σ_uv (local-px) → σ_orig scales as cs/S. Halving cs (512→256) →
2× tighter σ_orig per point → 4× Fisher info per tile. Quadrupling tile count
(200→800) → another 4×. Combined ~16× → σ_δ ~ 4× tighter, which (per user's
hypothesis) should drag ω residual from 0.05° (= fx 換算 1.6 px) toward sub-px.

We do NOT retrain. The σ-head from km_wv_wm_dgx2_n4_img128_8gpu_HEAD was
trained on 512-crop inputs; running it on 256-crops is OOD and we want to see
how badly that bites.

Tile geometry: each val instance pt has IH=IW=512 (no random pivot at eval).
We split each into 4 non-overlapping quadrants (u0,v0) ∈ {(0,0),(256,0),
(0,256),(256,256)} with cs=256, then resample each to S=img_size=128.
"""
from __future__ import annotations
import argparse
import sys
from pathlib import Path

import numpy as np
import torch
from scipy.spatial.transform import Rotation

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
DAMPING = 1e-3


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


def _is_obj_full(inst):
    if 'is_obj' in inst:
        v = inst['is_obj']
        return v.numpy().astype(bool) if hasattr(v, 'numpy') else np.asarray(v, dtype=bool)
    cubs = inst.get('cuboids', [])
    pts = inst['pts'].numpy() if hasattr(inst['pts'], 'numpy') else np.asarray(inst['pts'])
    from datasets.pandaset_full import _is_obj_per_point
    return _is_obj_per_point(pts, cubs).astype(bool)


def _build_subwin(ds, inst, t_delta, ypr_deg, *, u0, v0, cs):
    """Same logic as PandaSetCalibDatasetFull.apply_perturbation_explicit but
    with caller-supplied (u0,v0,cs) so we can carve sub-crops out of one tile."""
    if 'jpg_bytes' in inst:
        IH, IW = int(inst['IH']), int(inst['IW'])
    else:
        IH, IW = int(inst['img'].shape[-2]), int(inst['img'].shape[-1])
    K = inst['K_full'].numpy()
    pts = inst['pts'].numpy()
    cp = inst['cam_pos'].numpy()
    R_gt = inst['R_gt'].numpy()
    intensity = inst['intensity'].numpy() if hasattr(inst['intensity'], 'numpy') \
                else np.asarray(inst['intensity'])
    intensity = np.clip(np.asarray(intensity, dtype=np.float32), 0.0, 1.0)

    if 'uv_full' in inst and 'z_cam' in inst:
        uv_full = inst['uv_full'].numpy()
        z = inst['z_cam'].numpy()
    else:
        T_gt = inst['T_gt'].numpy()
        homo = np.column_stack([pts, np.ones(len(pts))])
        pts_cam_gt = (T_gt @ homo.T)[:3].T
        z = pts_cam_gt[:, 2].astype(np.float32)
        uv_full = ((K @ pts_cam_gt.T)[:2] / np.maximum(pts_cam_gt[:, 2:].T, 1e-6)).T.astype(np.float32)
    is_obj_full = _is_obj_full(inst)

    tile_u0 = int(inst.get('tile_u0', 0))
    tile_v0 = int(inst.get('tile_v0', 0))
    if tile_u0 or tile_v0:
        uv_full = uv_full - np.array([tile_u0, tile_v0], dtype=np.float32)
    img_full = None if 'jpg_bytes' in inst else inst['img']

    pad_px = int(cs * 0.10)
    in_pad = ((uv_full[:, 0] >= u0 - pad_px) & (uv_full[:, 0] < u0 + cs + pad_px) &
              (uv_full[:, 1] >= v0 - pad_px) & (uv_full[:, 1] < v0 + cs + pad_px) &
              (z > 0.5))
    cand_idx = np.where(in_pad)[0]
    if len(cand_idx) < ds.min_pts:
        return None
    if len(cand_idx) > ds.n_full:
        cand_idx = np.random.choice(cand_idx, size=ds.n_full, replace=False)
    pts_c = pts[cand_idx]
    intens_c = intensity[cand_idx]
    uv_gt_c = uv_full[cand_idx]

    ypr = np.asarray(ypr_deg, dtype=np.float64).reshape(3)
    t = np.asarray(t_delta, dtype=np.float64).reshape(3)
    R_off = R_gt @ Rotation.from_euler('zyx', ypr, degrees=True).as_matrix()
    cp_off = cp + t
    K_pert = K.copy()
    pert_vec = np.array([t[0], t[1], t[2], ypr[0], ypr[1], ypr[2], 0.0, 0.0],
                          dtype=np.float32)
    return ds.build_window(
        inst, pts_c, intens_c, uv_gt_c, cand_idx, is_obj_full,
        u0, v0, cs, K, R_off, cp_off, K_pert, cp, pert_vec,
        tile_u0, tile_v0, img_full, IW, IH,
    )


def _solve_one(model, ds_imgs, *, target_idx, n_inst, cs, n_per_inst,
                rng, ypr_target, t_target, dist_one, cfg, label):
    """Build B = n_inst × n_per_inst sub-crops with a SHARED rig-level
    perturbation, run model + shared GN, return solved δ̂."""
    if cs == 512:
        # Whole-tile, single sub-crop per instance: trivially the existing path.
        u0v0_list = [(0, 0)]
        assert n_per_inst == 1
    elif cs == 256:
        u0v0_list = [(0, 0), (256, 0), (0, 256), (256, 256)]
        assert n_per_inst == 4
    else:
        raise ValueError(f'unsupported cs={cs}')

    # Always seed sample 0 from the user-requested target idx. All sub-crops
    # of the target idx come first, then random instances fill to n_inst.
    target_inst = ds_imgs._load_inst(int(target_idx))
    wins = []
    for (u0, v0) in u0v0_list:
        w = _build_subwin(ds_imgs, target_inst, t_target, ypr_target,
                          u0=u0, v0=v0, cs=cs)
        if w is not None:
            wins.append(w)
    assert len(wins) >= 1, f'target idx={target_idx} returned no sub-crops'

    # Random val instances
    target_b = n_inst * n_per_inst
    tries = 0
    while len(wins) < target_b and tries < 32 * target_b:
        ridx = int(rng.randint(0, len(ds_imgs.fnames)))
        inst_r = ds_imgs._load_inst(ridx)
        for (u0, v0) in u0v0_list:
            if len(wins) >= target_b:
                break
            w = _build_subwin(ds_imgs, inst_r, t_target, ypr_target,
                              u0=u0, v0=v0, cs=cs)
            tries += 1
            if w is not None:
                wins.append(w)
    assert len(wins) == target_b, \
        f'[{label}] could not build batch ({len(wins)}/{target_b})'

    moved = [t.to(DEVICE) if torch.is_tensor(t) else t for t in collate_full(wins)]
    (imgs, true_uvd, dist_uvd, pad_mask, vfp,
     bucket_uvd, bucket_valid, _,
     pts_cam_orig, duv_orig, K_orig, cs_b) = moved
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

    use_intensity = getattr(model, 'use_intensity', True)
    if use_intensity:
        point_in = torch.cat([dist_uvd[..., :3], dist_uvd[..., 4:5]], dim=-1)
    else:
        point_in = dist_uvd[..., :3]
    img_norm = imgs.float().div(255.0)
    with torch.no_grad():
        out = model(img_norm, point_in, key_padding_mask=pad_mask, vfp=vfp,
                    bucket_uvd=bucket_uvd, bucket_valid=bucket_valid)
    per_pt = out[0] if isinstance(out, tuple) else out
    duv_pred_local = per_pt[..., :2].detach()
    if pad_full.any():
        duv_pred_local = duv_pred_local.clone()
        duv_pred_local[pad_full] = 0.0
    sx = per_pt[..., 2].exp()
    sy = per_pt[..., 3].exp()
    rho = per_pt[..., 4]
    W_sigma_local = make_info_from_sigma_rho(sx, sy, rho).detach()

    scale_l2o = (cs_b / float(cfg['img_size'])).reshape(-1, 1, 1)
    inv_l2o = (1.0 / scale_l2o).reshape(-1, 1, 1, 1)
    duv_pred_orig = duv_pred_local * scale_l2o
    W_sigma_orig = W_sigma_local * inv_l2o.pow(2)

    prior = PRIOR_DIAG.to(DEVICE)
    with torch.no_grad():
        delta_shared, H_last = solve_kb_xyz_shared(
            P0_orig, duv_pred_orig, W_sigma_orig, K_orig, dist, DOFS,
            valid=valid, n_iter=BA_N_ITER, damping=DAMPING, prior_diag=prior,
        )
    return delta_shared, B, H_last


def _residual_str(delta, ypr_target, t_target):
    """Pose residual = solved - target, in cam-frame (ω_x, ω_y, ω_z, tx, ty, tz).
    ypr_target is [ω_z, ω_y, ω_x] (zyx euler) so we re-order to xyz for compare."""
    target_xyz = np.array([ypr_target[2], ypr_target[1], ypr_target[0]], dtype=np.float64)
    d = delta.cpu().numpy()
    res_w = d[:3] - target_xyz
    res_t = d[3:] - t_target
    return (f'  ω̂=[{d[0]:+.4f},{d[1]:+.4f},{d[2]:+.4f}]°  '
            f't̂=[{d[3]:+.4f},{d[4]:+.4f},{d[5]:+.4f}]m\n'
            f'  res ω=[{res_w[0]:+.4f},{res_w[1]:+.4f},{res_w[2]:+.4f}]°  '
            f't=[{res_t[0]:+.4f},{res_t[1]:+.4f},{res_t[2]:+.4f}]m')


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--idx', type=int, default=17)
    ap.add_argument('--n-shared-512', type=int, default=200)
    ap.add_argument('--n-shared-256', type=int, default=200,
                    help='# of source instances (× 4 sub-crops = total tiles)')
    ap.add_argument('--rot-deg', type=float, default=0.30)
    ap.add_argument('--t-m', type=float, default=0.05)
    ap.add_argument('--seed', type=int, default=7 + 1000)
    args = ap.parse_args()

    cfg = _load_cfg()
    print(f'[256x800] idx={args.idx}  rot=±{args.rot_deg}°  '
          f't=±{args.t_m}m  seed={args.seed}')
    print(f'[256x800] cfg img_size={cfg["img_size"]}  '
          f'min_crop_px={cfg["min_crop_px"]}  max_crop_px={cfg["max_crop_px"]}')

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

    # Same RNG draws the same δ_target so 256x800 vs 512x200 are comparable.
    rng = np.random.RandomState(args.seed)
    ypr_target, t_target = _draw_pert(rng, rot_deg=args.rot_deg, t_m=args.t_m)
    target_xyz = np.array([ypr_target[2], ypr_target[1], ypr_target[0]], dtype=np.float64)
    print(f'[256x800] δ_target  ω=[{target_xyz[0]:+.4f},{target_xyz[1]:+.4f},'
          f'{target_xyz[2]:+.4f}]°  t=[{t_target[0]:+.4f},{t_target[1]:+.4f},'
          f'{t_target[2]:+.4f}]m')

    print()
    print('─── 200×512 (baseline) ' + '─' * 40)
    rng_a = np.random.RandomState(args.seed + 1)
    delta_a, Ba, Ha = _solve_one(
        model, ds, target_idx=args.idx, n_inst=args.n_shared_512,
        cs=512, n_per_inst=1, rng=rng_a,
        ypr_target=ypr_target, t_target=t_target, dist_one=dist_one,
        cfg=cfg, label='200x512')
    print(f'  B={Ba}')
    print(_residual_str(delta_a, ypr_target, t_target))

    print()
    print(f'─── {args.n_shared_256}×4=({args.n_shared_256*4})×256 (split sub-crops) ' + '─' * 12)
    rng_b = np.random.RandomState(args.seed + 1)  # same instance draws as 512
    delta_b, Bb, Hb = _solve_one(
        model, ds, target_idx=args.idx, n_inst=args.n_shared_256,
        cs=256, n_per_inst=4, rng=rng_b,
        ypr_target=ypr_target, t_target=t_target, dist_one=dist_one,
        cfg=cfg, label='Nx256')
    print(f'  B={Bb}')
    print(_residual_str(delta_b, ypr_target, t_target))

    # Compare residual norms
    target_xyz_t = torch.tensor(target_xyz, dtype=delta_a.dtype, device=delta_a.device)
    t_target_t = torch.tensor(t_target, dtype=delta_a.dtype, device=delta_a.device)
    full_target = torch.cat([target_xyz_t, t_target_t])
    res_a = delta_a - full_target
    res_b = delta_b - full_target
    fx = float(inst0['K_full'].numpy()[0, 0])
    print()
    print('─── Summary ' + '─' * 50)
    print(f'  fx (parent) = {fx:.1f} px')
    print(f'  ω-norm (deg)  512×200 = {res_a[:3].norm().item():.4f}     '
          f'256×{Bb} = {res_b[:3].norm().item():.4f}')
    print(f'  ω-norm (px-equiv via fx·tan)  '
          f'512×200 = {fx*np.tan(np.deg2rad(res_a[:3].norm().item())):.3f}     '
          f'256×{Bb} = {fx*np.tan(np.deg2rad(res_b[:3].norm().item())):.3f}')
    print(f'  t-norm (m)    512×200 = {res_a[3:].norm().item():.4f}     '
          f'256×{Bb} = {res_b[3:].norm().item():.4f}')


if __name__ == '__main__':
    main()
