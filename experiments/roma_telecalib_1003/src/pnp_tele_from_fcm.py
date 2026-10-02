"""Multi-frame PnP: recover R, t of TELE camera relative to FCM.

Inputs per frame:
    - fcm image  (tss4_fcm/*.jpg)                — KB4 fisheye
    - tele image (tss4_tele/*.jpg)               — KB4 fisheye (near-pinhole)
    - lidar      (vls128_rear_axle/*.npz)        — in vehicle rear-axle frame
    - setting-*.json                             — fcm/tele intrinsics + mp/rot

Pipeline:
    1. Undistort fcm/tele with per-cam virtual pinhole K_v.
    2. RoMaV2 dense match → high-conf correspondences (pxA on FCM, pxB on TELE).
    3. Project lidar into undistorted FCM (lidar → fcm cam frame → K_v_fcm).
    4. For each FCM px correspondence, find the nearest projected lidar point
       (within radius_px) → recover 3D in FCM cam frame.
    5. Collect (3D_fcm, 2D_tele) over all frames.
    6. cv2.solvePnPRansac → (R, t) of TELE in FCM frame.

Call:
    python pnp_tele_from_fcm.py \\
        --seq /path/to/sequence=.../ \\
        --frames 10,50,120 \\
        --out results/pnp.json
"""
import argparse
import json
import sys
from pathlib import Path

import cv2
import numpy as np
import torch
from scipy.spatial import cKDTree
from scipy.spatial.transform import Rotation

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parent / 'RoMaV2' / 'src'))
from tele_fcm_roma import _load_setting, _cam_block, _kb_undistort
from romav2 import RoMaV2


# ───────────────────────────────────────────────────────────────────────────
# Vehicle (rear-axle, ISO8855 FLU: x=forward, y=left, z=up) → camera (FRD)
R_TO_RDF = np.array([[0, -1, 0], [0, 0, -1], [1, 0, 0]], dtype=np.float64)


def _T_cam_from_veh(cam_cfg):
    """Camera extrinsic (vehicle-rear-axle → camera-FRD).
       mp = camera position in vehicle frame.
       rot = (roll, pitch, yaw) euler zyx intrinsic? Check with build script..."""
    mp  = np.asarray(cam_cfg['mp'],  dtype=np.float64)
    rot = np.asarray(cam_cfg['rot'], dtype=np.float64)
    R_cv_world = Rotation.from_euler('zyx', rot[::-1]).as_matrix()
    # Camera-FRD = R_to_rdf @ (world-frame → cam optical)
    R_cam_from_veh = R_TO_RDF @ np.linalg.inv(R_cv_world)
    t_cam_in_veh   = (R_TO_RDF @ (-mp)).astype(np.float64)
    return R_cam_from_veh, t_cam_in_veh


def _lidar_to_fcm_cam(lidar_xyz, R_cam_from_veh, t_cam_in_veh):
    """lidar (N,3) in rear-axle → fcm cam frame."""
    return (R_cam_from_veh @ lidar_xyz.T).T + t_cam_in_veh


def _project_pinhole(pts_cam, K):
    """(N,3) cam-frame → (N,2) uv, mask Z>0."""
    Z = pts_cam[:, 2]
    valid = Z > 0.1
    uv = np.full((len(pts_cam), 2), -1.0)
    uv[valid, 0] = pts_cam[valid, 0] * K[0, 0] / Z[valid] + K[0, 2]
    uv[valid, 1] = pts_cam[valid, 1] * K[1, 1] / Z[valid] + K[1, 2]
    return uv, valid


def _load_lidar(npz_path):
    d = np.load(npz_path, allow_pickle=True)
    return np.stack([d['xs'], d['ys'], d['zs']], -1).astype(np.float64)


def _one_frame(seq, k, K_fcm, D_fcm, K_tele, D_tele,
                K_v_fcm, K_v_tele, out_wh,
                R_cam_from_veh_fcm, t_cam_in_veh_fcm,
                model, conf):
    """Return (obj_pts_fcm (M,3), img_pts_tele (M,2)) for frame k, or None."""
    w, h = out_wh
    fcm_files  = sorted((seq / 'tss4_fcm').glob('*.jpg'))
    tele_files = sorted((seq / 'tss4_tele').glob('*.jpg'))
    lidar_files = sorted((seq / 'vls128_rear_axle').glob('*.npz'))
    if k >= min(len(fcm_files), len(tele_files), len(lidar_files)):
        return None, None, None
    im_fcm  = cv2.imread(str(fcm_files[k]))
    im_tele = cv2.imread(str(tele_files[k]))
    im_fcm_u  = _kb_undistort(im_fcm,  K_fcm,  D_fcm,  K_v_fcm,  (w, h))
    im_tele_u = _kb_undistort(im_tele, K_tele, D_tele, K_v_tele, (w, h))
    tmpd = Path('/tmp/_pnp_pair'); tmpd.mkdir(exist_ok=True)
    f_u = tmpd / 'f.jpg'; t_u = tmpd / 't.jpg'
    cv2.imwrite(str(f_u), im_fcm_u); cv2.imwrite(str(t_u), im_tele_u)

    preds = model.match(str(f_u), str(t_u))
    matches, overlaps, _, _ = model.sample(preds, 10000)
    pxA, pxB = model.to_pixel_coordinates(matches, h, w, h, w)
    pxA = pxA.cpu().numpy(); pxB = pxB.cpu().numpy()
    cert = overlaps.cpu().numpy()
    m = cert > conf
    pxA, pxB, cert = pxA[m], pxB[m], cert[m]
    print(f'[f{k:04d}] {len(pxA)} high-conf correspondences  '
          f'(mean cert {cert.mean():.3f})')

    lidar = _load_lidar(lidar_files[k])
    pts_fcm = _lidar_to_fcm_cam(lidar, R_cam_from_veh_fcm, t_cam_in_veh_fcm)
    uv_fcm, in_front = _project_pinhole(pts_fcm, K_v_fcm)
    inside = ((uv_fcm[:, 0] >= 0) & (uv_fcm[:, 0] < w) &
              (uv_fcm[:, 1] >= 0) & (uv_fcm[:, 1] < h) & in_front)
    uv_fcm_i = uv_fcm[inside]
    pts_fcm_i = pts_fcm[inside]
    print(f'[f{k:04d}] lidar: {len(uv_fcm_i)}/{len(lidar)} in undist FCM')

    # KDTree over lidar uv → nearest for each pxA
    tree = cKDTree(uv_fcm_i)
    d, idx = tree.query(pxA, distance_upper_bound=4.0)
    valid = np.isfinite(d)
    pts_3d  = pts_fcm_i[idx[valid]]
    pts_tele = pxB[valid]
    print(f'[f{k:04d}] {valid.sum()} pxA ⟷ nearest-lidar within 4px')
    return pts_3d, pts_tele, dict(n_corr=int(len(pxA)),
                                    n_pnp=int(valid.sum()),
                                    frame=int(k))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--seq', required=True)
    ap.add_argument('--frames', default='10,50,120',
                    help='comma-separated frame indices')
    ap.add_argument('--out', required=True)
    ap.add_argument('--conf', type=float, default=0.3)
    ap.add_argument('--fcm-focal',  type=float, default=3000.0)
    ap.add_argument('--tele-focal', type=float, default=4500.0)
    ap.add_argument('--out-wh', type=int, nargs=2, default=[1920, 1200])
    args = ap.parse_args()

    seq = Path(args.seq)
    setting = _load_setting(seq)
    K_fcm,  D_fcm  = _cam_block(setting, 'fcm')
    K_tele, D_tele = _cam_block(setting, 'tele')
    R_fcm, t_fcm   = _T_cam_from_veh(setting['fcm'])
    R_tele, t_tele = _T_cam_from_veh(setting['tele'])
    # Initial guess for TELE in FCM frame (from setting). PnP recovers refined.
    R_tele_from_fcm_init = R_tele @ R_fcm.T
    t_tele_from_fcm_init = t_tele - R_tele_from_fcm_init @ t_fcm
    print(f'[init] R_tele_from_fcm (deg zyx) = '
          f'{Rotation.from_matrix(R_tele_from_fcm_init).as_euler("zyx", degrees=True)}')
    print(f'[init] t_tele_from_fcm (m)       = {t_tele_from_fcm_init}')

    w, h = args.out_wh
    K_v_fcm  = np.array([[args.fcm_focal,  0, w/2],
                          [0, args.fcm_focal,  h/2], [0,0,1]], np.float64)
    K_v_tele = np.array([[args.tele_focal, 0, w/2],
                          [0, args.tele_focal, h/2], [0,0,1]], np.float64)

    torch.set_float32_matmul_precision('highest')
    model = RoMaV2(); model.apply_setting('precise')

    all_3d, all_2d, per_frame = [], [], []
    for ks in args.frames.split(','):
        k = int(ks.strip())
        pts3d, pts2d, diag = _one_frame(seq, k, K_fcm, D_fcm, K_tele, D_tele,
                                           K_v_fcm, K_v_tele, (w, h),
                                           R_fcm, t_fcm, model, args.conf)
        if pts3d is None: continue
        all_3d.append(pts3d); all_2d.append(pts2d); per_frame.append(diag)

    obj = np.concatenate(all_3d, 0).astype(np.float64)
    img = np.concatenate(all_2d, 0).astype(np.float64)
    print(f'[pnp] total pairs: {len(obj)}')

    ok, rvec, tvec, inl = cv2.solvePnPRansac(
        objectPoints=obj.reshape(-1, 1, 3),
        imagePoints=img.reshape(-1, 1, 2),
        cameraMatrix=K_v_tele, distCoeffs=None,
        iterationsCount=1000, reprojectionError=3.0,
        confidence=0.999, flags=cv2.SOLVEPNP_EPNP)
    R_est, _ = cv2.Rodrigues(rvec)
    t_est = tvec.ravel()
    print(f'[pnp] inliers: {len(inl)}/{len(obj)}')
    print(f'[pnp] R_tele_from_fcm (deg zyx) = '
          f'{Rotation.from_matrix(R_est).as_euler("zyx", degrees=True)}')
    print(f'[pnp] t_tele_from_fcm (m)       = {t_est}')

    delta_R = R_est @ R_tele_from_fcm_init.T
    delta_deg = Rotation.from_matrix(delta_R).as_euler('zyx', degrees=True)
    delta_t = t_est - t_tele_from_fcm_init
    print(f'[Δ init→est] rot (deg zyx) = {delta_deg}')
    print(f'[Δ init→est] t (m)         = {delta_t}')

    out = {
        'per_frame': per_frame,
        'n_pairs': len(obj),
        'n_inliers': int(len(inl)),
        'R_init': R_tele_from_fcm_init.tolist(),
        't_init': t_tele_from_fcm_init.tolist(),
        'R_est':  R_est.tolist(),
        't_est':  t_est.tolist(),
        'delta_rot_deg_zyx': delta_deg.tolist(),
        'delta_t_m': delta_t.tolist(),
        'K_v_fcm':  K_v_fcm.tolist(),
        'K_v_tele': K_v_tele.tolist(),
    }
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(out, indent=2))
    print(f'[out] {args.out}')


if __name__ == '__main__':
    main()
