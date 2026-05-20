"""Multi-frame closed-form BA: stack H, b from N consecutive frames of one
kamikado scene, solve once for the rig δ̂.

Same per-frame pipeline as ba_one_frame_vis.py (frame_to_tiles → per-tile
inference → centre-band σ-stratified TOP100 → KB closed-form GN), but
instead of solving each frame in isolation we extract the per-frame
H_t = Σ_i J_iᵀ Σ_i⁻¹ J_i and b_t = Σ_i J_iᵀ Σ_i⁻¹ r_i and accumulate:

    H_total = Σ_t H_t    b_total = Σ_t b_t    δ̂ = H_total⁻¹ b_total

The same synthetic perturbation (yaw=+1°, pitch=+0.5°) is applied
uniformly across all frames so the rig drift is constant — exactly
the multi-frame fusion contract.
"""
import sys
from pathlib import Path
REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))

import numpy as np
import torch
from scipy.spatial.transform import Rotation

from scripts.data.adapters.kamikado import load_frame, list_frames
from scripts.data.tile_cutter import frame_to_tiles
from datasets.pandaset_full import PandaSetCalibDatasetFull, collate_full
from scripts.inference.infer_calib import load_calib_model
from scripts.ba.ba_kb_jac import solve_dofs_kb, kb_jacobian, project_kb

EXP = 'km_wv_wm_n4_img128_cs256_512_200ep_dgx1_16gpu_resume'
SCENE_ROOT = '/raw_kamikado/scenes'
SCENE = 'points_ip664_D_20260226_224648_d005_3000_3020'
TILE = 512
STRIDE = 448
S_IN = 128
N_FRAMES = 6                         # how many frames to fuse


# Reuse the in-memory dataset wrapper from ba_one_frame_vis (avoids
# duplicating 50 lines).
from scripts._debug.ba_one_frame_vis import _InMemoryDS  # type: ignore


def _band_topk_pool(uv_obs, par_obs, z_obs, K_full, pW, pH,
                     band_u_frac=0.50, band_v_frac=0.25,
                     topk=100, gu=8, gv=4, per_cell=5):
    """Centre-band + σ-stratified TOP-K, identical to ba_one_frame_vis."""
    band_u = band_u_frac * pW / 2
    band_v = band_v_frac * pH / 2
    in_band = (np.abs(uv_obs[:, 0] - K_full[0, 2]) < band_u) & \
              (np.abs(uv_obs[:, 1] - K_full[1, 2]) < band_v)
    uv_b   = uv_obs[in_band]
    par_b  = par_obs[in_band]
    z_b    = z_obs[in_band]
    sigma_mean = 0.5 * (par_b[:, 2] + par_b[:, 3])
    order = np.argsort(sigma_mean)
    cell_count = np.zeros((gu, gv), dtype=np.int32)
    sel = []
    cu = pW / gu; cv = pH / gv
    for i in order:
        gx = int(min(uv_b[i, 0] / cu, gu - 1))
        gy = int(min(uv_b[i, 1] / cv, gv - 1))
        if cell_count[gx, gy] >= per_cell:
            continue
        sel.append(i); cell_count[gx, gy] += 1
        if len(sel) >= topk:
            break
    keep = np.asarray(sel, dtype=np.int64)
    return uv_b[keep], par_b[keep], z_b[keep]


def per_frame_pool(model, device, cf, rpy_deg: np.ndarray, t: np.ndarray):
    """Run frame_to_tiles + per-tile inference + parent-coord pooling +
    centre-band TOP-K. Return (uv_obs, par_obs, z_obs, K_full, pW, pH)."""
    parent = np.asarray(cf.img)
    pH, pW = parent.shape[:2]

    tile_insts = frame_to_tiles(cf, tile_w=TILE, tile_h=TILE, stride=STRIDE)
    ds = _InMemoryDS(tile_insts, img_size=S_IN, min_crop_px=TILE, max_crop_px=TILE)
    samples = []
    for i in range(len(tile_insts)):
        s = ds.apply_perturbation_explicit(i, t, rpy_deg)
        if s is not None:
            samples.append((i, s))
    if not samples:
        return None

    batch = collate_full([s for _, s in samples])
    imgs, true_uvd, dist_uvd, pad_mask, vfp, b_uvd, b_v = batch[:7]
    imgs_g = imgs.to(device).float().div(255.0)
    pad_g = pad_mask.to(device); vfp_g = vfp.to(device)
    b_uvd_g = b_uvd.to(device); b_v_g = b_v.to(device)
    use_int = bool(getattr(model, 'use_intensity', False))
    pin = (torch.cat([dist_uvd[..., :3], dist_uvd[..., 4:5]], -1)
           if use_int else dist_uvd[..., :3]).to(device)
    with torch.no_grad(), torch.autocast(device_type='cuda', dtype=torch.float16):
        params = model(imgs_g, pin, key_padding_mask=pad_g, vfp=vfp_g,
                        bucket_uvd=b_uvd_g, bucket_valid=b_v_g)
    p = params.float().cpu().numpy()
    dist_uv_b = dist_uvd[..., :2].numpy()
    pred_uv_b = dist_uv_b + p[..., :2]
    sx_b = np.exp(p[..., 2]); sy_b = np.exp(p[..., 3])
    rho_b = np.tanh(p[..., 4])

    K_full = tile_insts[samples[0][0]]['K_full']
    K_full = K_full.numpy() if hasattr(K_full, 'numpy') else np.asarray(K_full)
    scale_p = TILE / S_IN

    uv_list, par_list, z_list = [], [], []
    for k, (i, _) in enumerate(samples):
        ti = tile_insts[i]
        valid = ~((dist_uv_b[k, :, 0] == 0) & (dist_uv_b[k, :, 1] == 0))
        if not valid.any():
            continue
        u_p = dist_uv_b[k, valid, 0] * scale_p + ti['tile_u0']
        v_p = dist_uv_b[k, valid, 1] * scale_p + ti['tile_v0']
        gu_p = (true_uvd[k, valid, 0].numpy()) * scale_p + ti['tile_u0']
        gv_p = (true_uvd[k, valid, 1].numpy()) * scale_p + ti['tile_v0']
        du_p = (pred_uv_b[k, valid, 0] - dist_uv_b[k, valid, 0]) * scale_p
        dv_p = (pred_uv_b[k, valid, 1] - dist_uv_b[k, valid, 1]) * scale_p
        su_p = sx_b[k, valid] * scale_p
        sv_p = sy_b[k, valid] * scale_p
        rho_p = rho_b[k, valid]
        d_eucl = (true_uvd[k, valid, 2].numpy() * 100.0).astype(np.float64)
        _fx, _fy, _cx, _cy = K_full[0, 0], K_full[1, 1], K_full[0, 2], K_full[1, 2]
        _xn = (gu_p - _cx) / _fx
        _yn = (gv_p - _cy) / _fy
        d_proxy = d_eucl / np.sqrt(1.0 + _xn * _xn + _yn * _yn)
        uv_list.append(np.stack([u_p, v_p], axis=1))
        par_list.append(np.stack([du_p, dv_p, su_p, sv_p, rho_p], axis=1))
        z_list.append(d_proxy)

    if not uv_list:
        return None
    uv_obs = np.concatenate(uv_list)
    par_obs = np.concatenate(par_list)
    z_obs = np.concatenate(z_list)

    uv_b, par_b, z_b = _band_topk_pool(uv_obs, par_obs, z_obs, K_full, pW, pH)
    return uv_b, par_b.astype(np.float64), z_b.astype(np.float64), K_full, np.asarray(cf.dist, dtype=np.float64)


def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = load_calib_model(EXP).eval().to(device)

    scene = f'{SCENE_ROOT}/{SCENE}'
    frames = list_frames(Path(scene))
    mid = len(frames) // 2
    half = N_FRAMES // 2
    pick = frames[mid - half: mid - half + N_FRAMES]
    print(f'using {len(pick)} frames around mid={mid}: {pick}')

    rpy_deg = np.array([0.0, +1.0, +0.5], dtype=np.float64)  # [roll, yaw, pitch] (zyx)
    t = np.zeros(3, dtype=np.float64)
    pert_xyz = np.array([rpy_deg[2], rpy_deg[1], rpy_deg[0]])  # (ω_x, ω_y, ω_z)
    print(f'shared perturbation: rpy={rpy_deg} deg, t={t} m')
    print(f'  GT in (ω_x, ω_y, ω_z): {pert_xyz}')

    # Solve each frame independently and accumulate H, b. Stay 2-DoF
    # (pitch, yaw) to mirror the stable 1-frame configuration; multi-
    # frame translation requires real parallax which one scene barely
    # provides.
    dof = ['omega_x', 'omega_y']
    H_total = np.zeros((2, 2))
    b_total = np.zeros(2)
    per_frame_results = []
    for fidx in pick:
        cf = load_frame(scene, fidx)
        pool = per_frame_pool(model, device, cf, rpy_deg, t)
        if pool is None:
            print(f'  frame {fidx}: no usable pool, skip')
            continue
        uv_b, par_b, z_b, K_full, dist_kb = pool
        # Run solver once per frame to populate _last_H, _last_b. Use
        # warm-start at zero so different frames produce comparable H, b
        # at the same linearisation point.
        # Run 10-iter GN (Huber on) to log per-frame solution quality,
        # then re-run a SINGLE iteration at δ=0 so _last_H, _last_b
        # represent the linearisation at the same global origin across
        # frames — that's what the fused normal equation needs.
        delta_t = solve_dofs_kb(uv_b, par_b, z_b, K_full, dist_kb, dof,
                                 damping=1e-3, huber_k=2.5, n_iter=10,
                                 x0=np.zeros(len(dof)))
        # Linearise at the origin (n_iter=1, no Huber) to get H, b
        # describing the same linear system every frame contributes to.
        _ = solve_dofs_kb(uv_b, par_b, z_b, K_full, dist_kb, dof,
                           damping=1e-12, huber_k=None, n_iter=1,
                           x0=np.zeros(len(dof)))
        H_t = solve_dofs_kb._last_H.copy()
        b_t = solve_dofs_kb._last_b.copy()
        H_total += H_t
        b_total += b_t
        per_frame_results.append((fidx, len(uv_b), delta_t.copy()))
        print(f'  frame {fidx}: pool={len(uv_b)}  '
              f'δ_t[ω_x={delta_t[0]:+.3f}  ω_y={delta_t[1]:+.3f}]')

    # Damped solve of stacked normal equations.
    H_reg = H_total + 1e-3 * np.eye(len(dof))
    delta_fused = np.linalg.solve(H_reg, b_total)
    cov = np.linalg.inv(H_reg)
    gt = np.array([pert_xyz[0], pert_xyz[1]])  # ω_x = pitch, ω_y = yaw
    print('\n  [Multi-frame fused, 2-DoF (pitch, yaw)]')
    print('  DoF        δ̂          ±σ        |  GT       residual')
    for j, n in enumerate(dof):
        res = gt[j] - delta_fused[j]
        print(f'  {n:8s}  {delta_fused[j]:+9.4f}  ±{np.sqrt(max(cov[j,j],0.0)):.4f}'
              f'  | {gt[j]:+.4f}   {res:+.4f}')
    # Sanity: the average of per-frame δ_t should match the fused result
    # (it does only when all H_t are similar; differences come from
    # information-weighted averaging built into the H,b accumulation).
    avg = np.mean([d for _, _, d in per_frame_results], axis=0)
    print(f'\n  Mean of per-frame δ_t (informational baseline): '
          f'ω_x={avg[0]:+.4f}  ω_y={avg[1]:+.4f}')


if __name__ == '__main__':
    main()
