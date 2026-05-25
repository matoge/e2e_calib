"""Pair-frame geometry overlay sanity check.

Builds (frame_A, frame_B) crops from a PandaSet tile pair using the
same code path as PandaSetCalibDatasetFull.build_window, where:

  * frame_A:  random pivot from in-image LiDAR pts (cs ∈ [256, 512]),
              full δ_A SE3 perturbation, VCAM_A = ray(pivot)→z rotation.
  * frame_B:  cs_B = cs_A (VFP-identity rule), δ_B = 0 (LiDAR perfect),
              crop_B centred on uv_B = T_gt_B-projection of the same
              world pivot, VCAM_B = ray(uv_B)→z rotation.

Visual check (saved to docs/assets/2026-05-24_pair_overlay/):
  - panel A: full frame_A + crop_A bbox + pivot uv + perturbed/GT lidar
  - panel B: full frame_B + crop_B bbox + uv_B + GT lidar
  - panel A_crop: cropped frame_A 256-square + lidar overlay
  - panel B_crop: cropped frame_B 256-square + lidar overlay
  - prints: Δpose_AB (ypr deg + t m), VCAM_A vs VCAM_B angle, cs, uv_B in-image.

Run with:
  /home/hfunaya/.pyenv/versions/3.10.4/bin/python \
      scripts/_debug/_pair_overlay_check.py
"""
from __future__ import annotations

import os
import sys
import random
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from scipy.spatial.transform import Rotation

# Make repo root importable.
ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from datasets.pandaset_full import PandaSetCalibDatasetFull, decode_inst_img  # noqa: E402

CACHE_DIR = '/home/hfunaya/cache/pandaset_v3_tiled'
OUT_DIR = ROOT / 'docs/assets/2026-05-24_pair_overlay'
OUT_DIR.mkdir(parents=True, exist_ok=True)

IMG_SIZE = 256
MIN_CROP = 256
MAX_CROP = 512
SEED = 0


def vcam_R(uc: float, vc: float, K: np.ndarray) -> np.ndarray:
    """Rotation orig → vcam where vcam's +z passes through pixel (uc, vc).

    Same construction as pandaset_full.PandaSetCalibDatasetFull.__getitem__
    pose_frame='vcam' branch.
    """
    ray = np.array([(uc - K[0, 2]) / K[0, 0],
                    (vc - K[1, 2]) / K[1, 1],
                    1.0], dtype=np.float64)
    r_i = ray / (np.linalg.norm(ray) + 1e-12)
    z_ax = np.array([0., 0., 1.])
    axis = np.cross(r_i, z_ax)
    an = np.linalg.norm(axis)
    if an < 1e-9:
        return np.eye(3) if r_i[2] > 0 else -np.eye(3)
    axis = axis / an
    angle = float(np.arccos(np.clip(r_i @ z_ax, -1.0, 1.0)))
    return Rotation.from_rotvec(axis * angle).as_matrix()


def world_to_cam(pts_w: np.ndarray, R_cam_world: np.ndarray,
                 cam_pos: np.ndarray) -> np.ndarray:
    """world → cam-frame XYZ.  pts_w (N,3), R_cam_world (3,3) is cam→world.
    Convention here matches pandaset_full: x_cam = R^T (x_w - cam_pos)."""
    return (pts_w - cam_pos[None, :]) @ R_cam_world  # equiv. (R^T (x-c))^T


def project_pinhole(pts_cam: np.ndarray, K: np.ndarray) -> np.ndarray:
    z = np.maximum(pts_cam[:, 2], 1e-6)
    u = pts_cam[:, 0] / z * K[0, 0] + K[0, 2]
    v = pts_cam[:, 1] / z * K[1, 1] + K[1, 2]
    return np.stack([u, v], axis=-1)


def draw_dot(arr: np.ndarray, u: float, v: float, color, size: int = 2) -> None:
    H, W, _ = arr.shape
    iu, iv = int(round(u)), int(round(v))
    for du in range(-size, size + 1):
        for dv in range(-size, size + 1):
            x, y = iu + du, iv + dv
            if 0 <= x < W and 0 <= y < H:
                arr[y, x] = color


def draw_rect(arr: np.ndarray, u0: int, v0: int, cs: int, color,
              thick: int = 2) -> None:
    H, W, _ = arr.shape
    u1, v1 = u0 + cs - 1, v0 + cs - 1
    for t in range(thick):
        for u in range(max(0, u0 - t), min(W, u1 + t + 1)):
            for v in (v0 - t, v1 + t):
                if 0 <= v < H:
                    arr[v, u] = color
        for v in range(max(0, v0 - t), min(H, v1 + t + 1)):
            for u in (u0 - t, u1 + t):
                if 0 <= u < W:
                    arr[v, u] = color


def main():
    random.seed(SEED)
    np.random.seed(SEED)

    # Two consecutive tiles of the same scene (front_camera, frame 0 vs 1).
    fname_A = '00003000_t0.pt'
    fname_B = '00003001_t0.pt'

    # Use the dataset class only as a build_window provider — drive it
    # ourselves so we know the chosen pivot/cs/δ exactly.
    ds = PandaSetCalibDatasetFull(
        cache_dir=CACHE_DIR, split='train',
        img_size=IMG_SIZE,
        min_crop_px=MIN_CROP, max_crop_px=MAX_CROP,
        max_offset_m=0.20, max_rot_deg=1.0,
        pose_frame='orig',          # we compute VCAM ourselves below
        oversample=1, frame_stride=1,
        grid_n=16, k_per_cell=8,
        preload=False,
    )

    inst_A = torch.load(Path(CACHE_DIR) / 'inst' / fname_A, weights_only=False)
    inst_B = torch.load(Path(CACHE_DIR) / 'inst' / fname_B, weights_only=False)
    assert inst_A['scene'] == inst_B['scene']
    assert inst_A['cam'] == inst_B['cam']
    assert inst_A['frame'] + 1 == inst_B['frame']

    K = inst_A['K_full'].numpy()
    assert torch.allclose(inst_A['K_full'], inst_B['K_full']), 'pair K mismatch'

    IH = int(inst_A['IH']); IW = int(inst_A['IW'])
    tile_u0 = int(inst_A.get('tile_u0', 0))
    tile_v0 = int(inst_A.get('tile_v0', 0))

    pts_w_A = inst_A['pts'].numpy()                         # (N_A, 3) world coords
    R_gt_A  = inst_A['R_gt'].numpy()                        # cam→world
    cp_A    = inst_A['cam_pos'].numpy()
    R_gt_B  = inst_B['R_gt'].numpy()
    cp_B    = inst_B['cam_pos'].numpy()

    # ── world→A-cam to pick a pivot from A's in-image lidar ──
    pts_cam_A_gt = world_to_cam(pts_w_A, R_gt_A, cp_A)
    z_A = pts_cam_A_gt[:, 2]
    uv_A_full = project_pinhole(pts_cam_A_gt, K)            # parent-image px
    uv_A_loc  = uv_A_full - np.array([tile_u0, tile_v0])    # tile-local
    valid_A = ((z_A > 0.5)
               & (uv_A_loc[:, 0] >= 0) & (uv_A_loc[:, 0] < IW)
               & (uv_A_loc[:, 1] >= 0) & (uv_A_loc[:, 1] < IH))

    # cs_A ∈ [MIN_CROP, MAX_CROP] uniform (clipped to tile size)
    cs_hi = min(MAX_CROP, IW, IH)
    cs_lo = min(MIN_CROP, cs_hi)
    cs = int(np.random.randint(cs_lo, cs_hi + 1))

    # Pivot: random pick from A's lidar that admits a cs-crop fitting both
    # in the A tile AND with its B-projection lying inside the B tile.
    pivot_pool: list[int] = []
    pts_cam_B_pre = world_to_cam(pts_w_A, R_gt_B, cp_B)
    z_B_pre = pts_cam_B_pre[:, 2]
    uv_Bproj_full = project_pinhole(pts_cam_B_pre, K)
    uv_Bproj_loc  = uv_Bproj_full - np.array([tile_u0, tile_v0])
    # Valid pivot candidate: in-image on A, AND its B-projection falls
    # inside the B tile (so crop_B centred on uv_B is a valid window).
    valid_pivot = (valid_A
                   & (z_B_pre > 0.5)
                   & (uv_Bproj_loc[:, 0] >= 0) & (uv_Bproj_loc[:, 0] < IW)
                   & (uv_Bproj_loc[:, 1] >= 0) & (uv_Bproj_loc[:, 1] < IH))
    pool = np.where(valid_pivot)[0]
    if len(pool) == 0:
        raise RuntimeError('no pivot candidate visible in both A and B')
    pi = pool[np.random.randint(len(pool))]
    pivot_w   = pts_w_A[pi]
    pivot_uvA = uv_A_loc[pi]                                # tile-local
    pu_A, pv_A = float(pivot_uvA[0]), float(pivot_uvA[1])

    # crop_A centred on pivot_uvA (clipped to tile bounds)
    u0_A = int(np.clip(pu_A - cs / 2, 0, IW - cs))
    v0_A = int(np.clip(pv_A - cs / 2, 0, IH - cs))

    # δ_A sampling (pose_frame='orig' for clarity in the debug view)
    rot_deg = 1.0
    off_m   = 0.20
    t_delta_A = (np.random.rand(3) * 2 - 1) * off_m
    ypr_A     = (np.random.rand(3) * 2 - 1) * rot_deg

    # Build candidate index list = same in_pad mask the dataset uses.
    pad_px = int(cs * 0.10)
    in_pad_A = ((uv_A_loc[:, 0] >= u0_A - pad_px) &
                (uv_A_loc[:, 0] <  u0_A + cs + pad_px) &
                (uv_A_loc[:, 1] >= v0_A - pad_px) &
                (uv_A_loc[:, 1] <  v0_A + cs + pad_px) &
                (z_A > 0.5))
    cand_idx_A = np.where(in_pad_A)[0]
    if len(cand_idx_A) > ds.n_full:
        cand_idx_A = np.random.choice(cand_idx_A, size=ds.n_full, replace=False)

    pts_c_A    = pts_w_A[cand_idx_A]
    intens_A   = inst_A['intensity'].numpy()[cand_idx_A].astype(np.float32)
    intens_A   = np.clip(intens_A, 0.0, 1.0)
    uv_gt_c_A  = uv_A_loc[cand_idx_A].astype(np.float32)
    is_obj_A   = inst_A['is_obj'].numpy().astype(bool)

    # R_off / cp_off for δ_A in original-camera frame
    R_off_A = R_gt_A @ Rotation.from_euler('zyx', ypr_A, degrees=True).as_matrix()
    cp_off_A = cp_A + t_delta_A
    K_pert_A = K.copy()

    pert_vec_A = np.array([t_delta_A[0], t_delta_A[1], t_delta_A[2],
                           ypr_A[0], ypr_A[1], ypr_A[2], 0.0, 0.0],
                          dtype=np.float32)

    out_A = ds.build_window(
        inst_A, pts_c_A, intens_A, uv_gt_c_A, cand_idx_A, is_obj_A,
        u0_A, v0_A, cs, K, R_off_A, cp_off_A, K_pert_A, cp_A, pert_vec_A,
        tile_u0, tile_v0, None, IW, IH,
    )
    if out_A is None:
        raise RuntimeError('build_window(A) returned None — re-roll seed')
    (img_A, true_A, dist_A, vfp_A, *_rest_A) = out_A

    # ── frame_B: project the SAME world pivot into B's GT camera ──
    pivot_cam_B = world_to_cam(pivot_w[None, :], R_gt_B, cp_B)[0]
    pivot_uvB_parent = project_pinhole(pivot_cam_B[None, :], K)[0]
    pivot_uvB = pivot_uvB_parent - np.array([tile_u0, tile_v0])
    pu_B, pv_B = float(pivot_uvB[0]), float(pivot_uvB[1])
    inb_B = (0 <= pu_B < IW) and (0 <= pv_B < IH) and (pivot_cam_B[2] > 0.5)
    print(f'pivot uv_A (tile-local): {pu_A:.1f}, {pv_A:.1f}')
    print(f'pivot uv_B (tile-local): {pu_B:.1f}, {pv_B:.1f}   in_image={inb_B}')

    # cs_B = cs_A (VFP-identity rule); centre crop_B on uv_B
    u0_B = int(np.clip(pu_B - cs / 2, 0, IW - cs))
    v0_B = int(np.clip(pv_B - cs / 2, 0, IH - cs))

    # δ_B = 0 (perfect LiDAR for v1)
    pts_w_B = inst_B['pts'].numpy()
    intens_B_full = np.clip(inst_B['intensity'].numpy().astype(np.float32), 0.0, 1.0)
    pts_cam_B_gt = world_to_cam(pts_w_B, R_gt_B, cp_B)
    z_B = pts_cam_B_gt[:, 2]
    uv_B_full = project_pinhole(pts_cam_B_gt, K)
    uv_B_loc = uv_B_full - np.array([tile_u0, tile_v0])
    pad_px_B = int(cs * 0.10)
    in_pad_B = ((uv_B_loc[:, 0] >= u0_B - pad_px_B) &
                (uv_B_loc[:, 0] <  u0_B + cs + pad_px_B) &
                (uv_B_loc[:, 1] >= v0_B - pad_px_B) &
                (uv_B_loc[:, 1] <  v0_B + cs + pad_px_B) &
                (z_B > 0.5))
    cand_idx_B = np.where(in_pad_B)[0]
    if len(cand_idx_B) > ds.n_full:
        cand_idx_B = np.random.choice(cand_idx_B, size=ds.n_full, replace=False)
    pts_c_B   = pts_w_B[cand_idx_B]
    intens_B  = intens_B_full[cand_idx_B]
    uv_gt_c_B = uv_B_loc[cand_idx_B].astype(np.float32)
    is_obj_B  = inst_B['is_obj'].numpy().astype(bool)
    R_off_B   = R_gt_B          # δ_B = 0
    cp_off_B  = cp_B
    K_pert_B  = K.copy()
    pert_vec_B = np.zeros(8, dtype=np.float32)
    out_B = ds.build_window(
        inst_B, pts_c_B, intens_B, uv_gt_c_B, cand_idx_B, is_obj_B,
        u0_B, v0_B, cs, K, R_off_B, cp_off_B, K_pert_B, cp_B, pert_vec_B,
        tile_u0, tile_v0, None, IW, IH,
    )
    if out_B is None:
        raise RuntimeError('build_window(B) returned None — re-roll seed')
    (img_B, true_B, dist_B, vfp_B, *_rest_B) = out_B

    # ── VCAM rotations ──
    uc_A = u0_A + 0.5 * cs + tile_u0
    vc_A = v0_A + 0.5 * cs + tile_v0
    R_o_vA = vcam_R(uc_A, vc_A, K)
    uc_B = u0_B + 0.5 * cs + tile_u0
    vc_B = v0_B + 0.5 * cs + tile_v0
    R_o_vB = vcam_R(uc_B, vc_B, K)
    R_AB_vcam = R_o_vB @ R_o_vA.T
    ang_AB = float(np.degrees(np.linalg.norm(Rotation.from_matrix(R_AB_vcam).as_rotvec())))

    # ── Δpose_AB in original-camera (ego-cam-A → ego-cam-B) ──
    # x_cam_B = R_gt_B^T (x_w - cp_B); x_w = R_gt_A x_cam_A + cp_A.
    # ⇒ x_cam_B = R_gt_B^T R_gt_A x_cam_A + R_gt_B^T (cp_A - cp_B).
    R_AB = R_gt_B.T @ R_gt_A
    t_AB = R_gt_B.T @ (cp_A - cp_B)
    ypr_AB = Rotation.from_matrix(R_AB).as_euler('zyx', degrees=True)
    print('Δpose_AB (cam_A → cam_B): ypr_deg=({:+.3f}, {:+.3f}, {:+.3f})  '
          't_m=({:+.3f}, {:+.3f}, {:+.3f})  |t|={:.3f}m'
          .format(ypr_AB[0], ypr_AB[1], ypr_AB[2],
                  t_AB[0], t_AB[1], t_AB[2], float(np.linalg.norm(t_AB))))
    print(f'cs={cs}px, vfp_A={vfp_A.item():.1f}px, vfp_B={vfp_B.item():.1f}px '
          f'(should match)')
    print(f'crop_A u0/v0={u0_A}/{v0_A}, crop_B u0/v0={u0_B}/{v0_B}')
    print(f'VCAM_A vs VCAM_B rotation magnitude: {ang_AB:.3f} deg')

    # ── Visualisations ──
    YELLOW  = np.array([255, 220,   0], dtype=np.uint8)
    RED     = np.array([240,  40,  40], dtype=np.uint8)
    GREEN   = np.array([ 30, 220,  60], dtype=np.uint8)
    BLUE    = np.array([ 60, 130, 255], dtype=np.uint8)

    fullA = decode_inst_img(inst_A).permute(1, 2, 0).numpy().copy()  # (H, W, 3) uint8
    fullB = decode_inst_img(inst_B).permute(1, 2, 0).numpy().copy()
    # Overlay all visible LiDAR (small red), pivot (yellow), crop bbox (blue), perturbed_A (green).
    for k in np.where(valid_A)[0]:
        draw_dot(fullA, uv_A_loc[k, 0], uv_A_loc[k, 1], RED, size=0)
    # Perturbed lidar in A (after δ_A): re-project pts_w_A via R_off_A/cp_off_A.
    pts_cam_A_off = world_to_cam(pts_w_A, R_off_A, cp_off_A)
    z_off = pts_cam_A_off[:, 2]
    uv_off_full = project_pinhole(pts_cam_A_off, K)
    uv_off_loc  = uv_off_full - np.array([tile_u0, tile_v0])
    valid_off = ((z_off > 0.5) & (uv_off_loc[:, 0] >= 0) & (uv_off_loc[:, 0] < IW)
                 & (uv_off_loc[:, 1] >= 0) & (uv_off_loc[:, 1] < IH))
    for k in np.where(valid_off)[0]:
        draw_dot(fullA, uv_off_loc[k, 0], uv_off_loc[k, 1], GREEN, size=0)
    draw_rect(fullA, u0_A, v0_A, cs, BLUE)
    draw_dot(fullA, pu_A, pv_A, YELLOW, size=4)

    valid_B = ((z_B > 0.5)
               & (uv_B_loc[:, 0] >= 0) & (uv_B_loc[:, 0] < IW)
               & (uv_B_loc[:, 1] >= 0) & (uv_B_loc[:, 1] < IH))
    for k in np.where(valid_B)[0]:
        draw_dot(fullB, uv_B_loc[k, 0], uv_B_loc[k, 1], RED, size=0)
    draw_rect(fullB, u0_B, v0_B, cs, BLUE)
    if inb_B:
        draw_dot(fullB, pu_B, pv_B, YELLOW, size=4)

    Image.fromarray(fullA).save(OUT_DIR / 'frame_A_full.png')
    Image.fromarray(fullB).save(OUT_DIR / 'frame_B_full.png')

    # cropped views (img_A is (3, S, S) uint8 from build_window)
    crop_A_arr = img_A.permute(1, 2, 0).numpy().copy()
    crop_B_arr = img_B.permute(1, 2, 0).numpy().copy()
    # GT pivot in crop-local px (S coords)
    scale = IMG_SIZE / cs
    pu_A_loc = (pu_A - u0_A) * scale
    pv_A_loc = (pv_A - v0_A) * scale
    pu_B_loc = (pu_B - u0_B) * scale
    pv_B_loc = (pv_B - v0_B) * scale
    # Mark all true_uvd (yellow=GT) and dist_uvd (green=δ-perturbed) for A.
    for r in range(true_A.shape[0]):
        u, v = float(true_A[r, 0]), float(true_A[r, 1])
        if 0 <= u < IMG_SIZE and 0 <= v < IMG_SIZE:
            draw_dot(crop_A_arr, u, v, YELLOW, size=1)
    for r in range(dist_A.shape[0]):
        u, v = float(dist_A[r, 0]), float(dist_A[r, 1])
        if 0 <= u < IMG_SIZE and 0 <= v < IMG_SIZE:
            draw_dot(crop_A_arr, u, v, GREEN, size=1)
    # Pivot last (on top), bigger.
    draw_dot(crop_A_arr, pu_A_loc, pv_A_loc, BLUE, size=5)
    # B: only GT (δ_B = 0)
    for r in range(true_B.shape[0]):
        u, v = float(true_B[r, 0]), float(true_B[r, 1])
        if 0 <= u < IMG_SIZE and 0 <= v < IMG_SIZE:
            draw_dot(crop_B_arr, u, v, YELLOW, size=1)
    if inb_B:
        draw_dot(crop_B_arr, pu_B_loc, pv_B_loc, BLUE, size=5)
    Image.fromarray(crop_A_arr).save(OUT_DIR / 'crop_A.png')
    Image.fromarray(crop_B_arr).save(OUT_DIR / 'crop_B.png')

    # Side-by-side composite for at-a-glance check.
    pad = 8
    H = max(fullA.shape[0], fullB.shape[0])
    Wf = fullA.shape[1] + pad + fullB.shape[1]
    canvas_top = np.full((H, Wf, 3), 32, dtype=np.uint8)
    canvas_top[:fullA.shape[0], :fullA.shape[1]] = fullA
    canvas_top[:fullB.shape[0], fullA.shape[1] + pad:fullA.shape[1] + pad + fullB.shape[1]] = fullB
    Hc = IMG_SIZE
    Wc = IMG_SIZE * 2 + pad
    canvas_bot = np.full((Hc, Wc, 3), 32, dtype=np.uint8)
    canvas_bot[:, :IMG_SIZE] = crop_A_arr
    canvas_bot[:, IMG_SIZE + pad:IMG_SIZE * 2 + pad] = crop_B_arr
    # stack vertically with a small spacer
    spacer = np.full((pad, max(Wf, Wc), 3), 32, dtype=np.uint8)
    top_pad = np.full((H, max(Wf, Wc), 3), 32, dtype=np.uint8)
    top_pad[:, :Wf] = canvas_top
    bot_pad = np.full((Hc, max(Wf, Wc), 3), 32, dtype=np.uint8)
    bot_pad[:, :Wc] = canvas_bot
    composite = np.concatenate([top_pad, spacer, bot_pad], axis=0)
    Image.fromarray(composite).save(OUT_DIR / 'composite.png')
    print(f'wrote: {OUT_DIR}/composite.png')


if __name__ == '__main__':
    main()
