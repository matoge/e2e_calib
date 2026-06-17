"""WovenParser: a drop-in replacement for examples/datasets/colmap.Parser
that reads a single woven_sequence directory + recalibration.json.

Provides exactly the attributes that gsplat 1.5.3's simple_trainer expects:
    image_names, image_paths, camtoworlds, camera_ids, Ks_dict, params_dict,
    imsize_dict, mask_dict, points, points_err, points_rgb, point_indices,
    transform, scene_scale, mapx_dict, mapy_dict, roi_undist_dict, factor,
    test_every, extconf

KB4 native path: feed simple_trainer with --camera_model fisheye --with_ut
                 --with_eval3d. We populate `params_dict[0]` with k1..k4 so
                 the rasterizer reads them. Image undistortion is DISABLED
                 (mapx_dict/roi_undist_dict left empty so colmap.Dataset's
                 cv2.remap branch is not triggered — see the patched Dataset).
"""
from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np
from scipy.spatial.transform import Rotation as _R

import sys
_HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(_HERE))
sys.path.insert(0, str(_HERE.parent / 'webui_kb_fit'))
import lidar_deskew as DK


RECALIB_DEFAULT = Path(os.environ.get(
    'WOVEN_RECALIB_JSON',
    '/home/hfunaya/git/loom/backend/assets/woven_sequence/'
    'llinking_26/recalibration.json'))

R_TO_RDF = np.array([[0, -1, 0], [0, 0, -1], [1, 0, 0]], dtype=np.float64)


def _load_calib(recalib: Path, vehicle: str) -> dict:
    d = json.loads(recalib.read_text())[vehicle]
    fcm = d['fcm']
    poslv = d['poslv']
    fx = fy = float(fcm['kb']['focal_length'])
    cx, cy = fcm['cc']
    K = np.array([[fx, 0, cx], [0, fy, cy], [0, 0, 1]], dtype=np.float64)
    dist = np.array([fcm['kb'][f'k{i}'] for i in (1, 2, 3, 4)],
                    dtype=np.float64)
    W, H = int(fcm['resolution'][0]), int(fcm['resolution'][1])

    mp_fcm = np.asarray(fcm['mp'], dtype=np.float64)
    roll, pitch, yaw = fcm['rot']
    R_fcm = _R.from_euler('zyx', [yaw, pitch, roll]).as_matrix()
    mp_poslv = np.asarray(poslv['mp'], dtype=np.float64)
    roll_p, pitch_p, yaw_p = poslv['rot']
    R_poslv = _R.from_euler('zyx', [yaw_p, pitch_p, roll_p]).as_matrix()

    R_w2c_static = R_TO_RDF @ R_fcm.T @ R_poslv
    t_w2c_static = R_TO_RDF @ R_fcm.T @ mp_poslv - R_TO_RDF @ mp_fcm
    T_pscam_from_rear_static = np.eye(4)
    T_pscam_from_rear_static[:3, :3] = R_w2c_static
    T_pscam_from_rear_static[:3, 3] = t_w2c_static
    T_rear_from_pscam_static = np.linalg.inv(T_pscam_from_rear_static)
    return dict(K=K, dist=dist, W=W, H=H,
                T_rear_from_pscam_static=T_rear_from_pscam_static)


class WovenParser:
    """A Parser-shaped object that simple_trainer.py and Dataset can consume."""

    def __init__(
        self,
        seq_dir: Path,
        vehicle: str = '248',
        recalib_json: Path = RECALIB_DEFAULT,
        masks_dir: Optional[Path] = None,
        factor: int = 1,
        test_every: int = 8,
        normalize: bool = False,
        sample_per_frame: int = 80_000,
        pose_ckpt: Optional[Path] = None,
        gicp_T_world: Optional[Path] = None,
        lidar_dir_name: str = 'vls128_rear_axle',
    ):
        self.factor = factor
        self.test_every = test_every
        self.extconf = {'spiral_radius_scale': 1.0, 'no_factor_suffix': False}
        self.mask_dict: Dict[int, Optional[np.ndarray]] = {}
        # per-frame mask paths (image_name -> mask png path), consumed by
        # the patched Dataset.__getitem__ -> data["mask"]
        self.frame_mask_paths: Dict[str, Path] = {}

        self.seq_dir = Path(seq_dir)
        if masks_dir is not None:
            masks_dir = Path(masks_dir)
        cam_files = sorted((self.seq_dir / 'tss4_fcm').glob('*.jpg'))
        lid_files = sorted((self.seq_dir / lidar_dir_name).glob('*.npz'))
        if len(cam_files) != len(lid_files):
            raise RuntimeError(
                f'cam={len(cam_files)} != lid={len(lid_files)} in {seq_dir}')
        n = len(cam_files)
        if n == 0:
            raise RuntimeError(f'no frames in {seq_dir}')

        calib = _load_calib(recalib_json, vehicle)
        K = calib['K'].astype(np.float64)
        dist = calib['dist'].astype(np.float64)
        W_full, H_full = calib['W'], calib['H']

        # downsample by factor (gsplat expects K and image to match the actual
        # image size on disk; WovenParser ships full-res by default)
        if factor != 1:
            K = K.copy()
            K[0, :] /= factor
            K[1, :] /= factor
            W = W_full // factor
            H = H_full // factor
        else:
            W, H = W_full, H_full

        # POSLV poses (skip if gicp_T_world is provided AND no poslv files exist)
        pq = self.seq_dir / 'poslv' / 'POS.parquet'
        csv = self.seq_dir / 'poslv' / 'poslv.csv'
        camtoworlds = np.zeros((n, 4, 4), dtype=np.float64)
        cam_t_secs = [float(j.stem.split('_')[1]) / 1000.0 for j in cam_files]
        T_rear_world_per_frame = np.zeros((n, 4, 4), dtype=np.float64)
        if (pq.is_file() or csv.is_file()):
            ts_a, e, n_, u, roll, pitch, head = DK.load_poslv_poses(
                pq if pq.is_file() else csv)
            T_poslv_from_rear_at_t0 = DK.interp_pose(
                np.array([cam_t_secs[0]]), ts_a, e, n_, u, roll, pitch, head)[0]
            T_rear0_from_poslv = np.linalg.inv(T_poslv_from_rear_at_t0)
            for i, jpg in enumerate(cam_files):
                T_poslv_from_rear_i = DK.interp_pose(
                    np.array([cam_t_secs[i]]), ts_a, e, n_, u, roll, pitch, head)[0]
                T_world_from_rear_i = T_rear0_from_poslv @ T_poslv_from_rear_i
                T_rear_world_per_frame[i] = T_world_from_rear_i
                camtoworlds[i] = T_world_from_rear_i @ calib[
                    'T_rear_from_pscam_static']
        else:
            # POSLV unavailable: identity placeholder, expect gicp_T_world to overwrite
            assert gicp_T_world is not None, (
                f'no POSLV files in {self.seq_dir}/poslv/ AND gicp_T_world=None')
            print(f'[WovenParser] POSLV missing; relying entirely on gicp_T_world')
            for i in range(n):
                T_rear_world_per_frame[i] = np.eye(4)
                camtoworlds[i] = np.eye(4) @ calib['T_rear_from_pscam_static']

        # gicp_T_world : npy file containing per-frame T_world_from_rear (n, 4, 4)
        # anchored at frame 0. Replaces POSLV trajectory (= same as pinhole parser).
        if gicp_T_world is not None:
            T_wr_gicp = np.load(str(gicp_T_world))
            assert T_wr_gicp.shape == (n, 4, 4), f'gicp shape {T_wr_gicp.shape} vs {n}'
            T_rear_world_per_frame = T_wr_gicp.astype(np.float64)
            T_rear_from_pscam_static = calib['T_rear_from_pscam_static']
            for i in range(n):
                camtoworlds[i] = T_rear_world_per_frame[i] @ T_rear_from_pscam_static
            print(f'[WovenParser] gicp_T_world baked from {gicp_T_world}')

        # pose_ckpt : bake per-frame 9-dim pose_adjust deltas (rot_6d + trans 3)
        # from a previous simple_trainer run into camtoworlds.
        if pose_ckpt is not None:
            import torch as _t
            ck = _t.load(str(pose_ckpt), map_location='cpu', weights_only=False)
            emb = ck['pose_adjust']['embeds.weight'].numpy()  # [n_train, 9]
            train_idx = [i for i in range(n) if i % test_every != 0]
            val_idx   = [i for i in range(n) if i % test_every == 0]
            _ID6 = np.array([1., 0., 0., 0., 1., 0.])
            def _r6(d):
                a1, a2 = d[:3], d[3:]
                b1 = a1 / np.linalg.norm(a1)
                b2 = a2 - (b1 * a2).sum() * b1
                b2 = b2 / np.linalg.norm(b2)
                return np.stack([b1, b2, np.cross(b1, b2)], axis=-1)
            def _apply(c2w, d9):
                R = _r6(d9[3:9] + _ID6)
                T = np.eye(4); T[:3, :3] = R; T[:3, 3] = d9[:3]
                return c2w @ T
            new_c2w = camtoworlds.copy()
            for di, fi in enumerate(train_idx):
                new_c2w[fi] = _apply(camtoworlds[fi], emb[di])
            ta = np.asarray(train_idx)
            for vi in val_idx:
                diffs = np.abs(ta - vi)
                o = np.argsort(diffs)[:2]
                d0, d1 = diffs[o[0]], diffs[o[1]]
                if d0 + d1 == 0:
                    d_emb = emb[o[0]]
                else:
                    w0 = d1 / (d0 + d1)
                    w1 = d0 / (d0 + d1)
                    d_emb = emb[o[0]] * w0 + emb[o[1]] * w1
                new_c2w[vi] = _apply(camtoworlds[vi], d_emb)
            T_pscam_from_rear = np.linalg.inv(calib['T_rear_from_pscam_static'])
            new_rw = np.zeros_like(T_rear_world_per_frame)
            for i in range(n):
                new_rw[i] = new_c2w[i] @ T_pscam_from_rear
            camtoworlds = new_c2w
            T_rear_world_per_frame = new_rw
            print(f'[WovenParser] pose_ckpt baked from {pose_ckpt} '
                  f'(train delta mean trans {np.linalg.norm(emb[:,:3], axis=1).mean()*100:.2f} cm)')

        image_names = [jpg.stem for jpg in cam_files]
        image_paths = [str(jpg) for jpg in cam_files]
        camera_ids = [0] * n
        Ks_dict = {0: K.astype(np.float64)}
        params_dict = {0: dist.astype(np.float64)}
        imsize_dict = {0: (W, H)}
        self.mask_dict[0] = None

        # Permanent vignette + dashboard mask (vehicle-specific, all frames same).
        # Polygons mark INVALID regions (vignette + dashboard); we want a
        # mask where True=valid (= keep in loss).
        try:
            import json as _json
            from pathlib import Path as _Path
            perm_path = _Path('/host_e2e_calib/assets/permanent_masks') / f'v{vehicle}_tss4_fcm.json'
            if not perm_path.is_file():
                # also try host path (when running outside container)
                perm_path = _Path(__file__).resolve().parents[2] / 'assets' / 'permanent_masks' / f'v{vehicle}_tss4_fcm.json'
            if perm_path.is_file():
                pdef = _json.loads(perm_path.read_text())
                W_p, H_p = pdef['image_size']
                m = np.ones((H, W), dtype=bool)
                # build at native polygon resolution then resize
                m_full = np.full((H_p, W_p), 255, dtype=np.uint8)
                for poly in pdef['polygons']:
                    pts = np.array(poly, dtype=np.int32).reshape(-1, 1, 2)
                    cv2.fillPoly(m_full, [pts], 0)  # 0 = invalid
                if (W_p, H_p) != (W, H):
                    m_full = cv2.resize(m_full, (W, H),
                                        interpolation=cv2.INTER_NEAREST)
                self.mask_dict[0] = (m_full > 127)
                kept = float((m_full > 127).mean())
                print(f'[WovenParser] permanent mask loaded from {perm_path.name} '
                      f'(keep ratio {kept:.3f})')
        except Exception as _e:
            print(f'[WovenParser] permanent mask load failed: {_e}')

        # populate per-frame mask paths (image_name -> path)
        if masks_dir is not None:
            for jpg in cam_files:
                cand = masks_dir / f'{jpg.stem}.png'
                if cand.is_file():
                    self.frame_mask_paths[jpg.stem] = cand

        # ── point cloud (LiDAR + RGB) ──────────────────────────────────────
        rng = np.random.default_rng(0)
        pts_world_list = []
        rgb_list = []
        for i, (jpg, npz) in enumerate(zip(cam_files, lid_files)):
            d = np.load(npz)
            xs, ys, zs = d['xs'], d['ys'], d['zs']
            # rear_axle frame -> world (frame 0)
            pts_rear = np.stack([xs, ys, zs], axis=-1).astype(np.float64)
            T_w_r = T_rear_world_per_frame[i]
            pts_w = (T_w_r[:3, :3] @ pts_rear.T + T_w_r[:3, 3:4]).T

            # Project into the (full-resolution) fisheye camera to get a
            # color and to apply the dynamic mask drop
            T_w2c = np.linalg.inv(camtoworlds[i])
            pts_c = (T_w2c[:3, :3] @ pts_w.T + T_w2c[:3, 3:4]).T
            Z = pts_c[:, 2]
            in_front = Z > 0.5
            if not in_front.any():
                continue
            pcv = pts_c[in_front].reshape(-1, 1, 3)
            uv_full, _ = cv2.fisheye.projectPoints(
                pcv, np.zeros(3), np.zeros(3),
                calib['K'].astype(np.float64),
                calib['dist'].astype(np.float64))
            uv_full = uv_full.reshape(-1, 2)
            in_b = ((uv_full[:, 0] >= 0) & (uv_full[:, 0] < W_full)
                    & (uv_full[:, 1] >= 0) & (uv_full[:, 1] < H_full))
            idx_front = np.where(in_front)[0]
            keep_idx = idx_front[in_b]
            if len(keep_idx) == 0:
                continue
            uv_kept = uv_full[in_b]

            # mask drop (white = keep)
            if masks_dir is not None:
                mp = masks_dir / f'{jpg.stem}.png'
                if mp.is_file():
                    m = cv2.imread(str(mp), cv2.IMREAD_GRAYSCALE)
                    if m.shape[:2] != (H_full, W_full):
                        m = cv2.resize(m, (W_full, H_full),
                                        interpolation=cv2.INTER_NEAREST)
                    u_int = np.clip(np.round(uv_kept[:, 0]).astype(np.int32),
                                     0, W_full - 1)
                    v_int = np.clip(np.round(uv_kept[:, 1]).astype(np.int32),
                                     0, H_full - 1)
                    keep_mask = m[v_int, u_int] > 127
                    keep_idx = keep_idx[keep_mask]
                    uv_kept = uv_kept[keep_mask]
            if len(keep_idx) == 0:
                continue

            # subsample
            if len(keep_idx) > sample_per_frame:
                sel = rng.choice(len(keep_idx), sample_per_frame,
                                 replace=False)
                keep_idx = keep_idx[sel]
                uv_kept = uv_kept[sel]

            # color via the full-res image
            img = cv2.imread(image_paths[i])
            rgb_img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            u = np.clip(np.round(uv_kept[:, 0]).astype(np.int32),
                         0, W_full - 1)
            v = np.clip(np.round(uv_kept[:, 1]).astype(np.int32),
                         0, H_full - 1)
            cols = rgb_img[v, u].astype(np.uint8)
            pts_world_list.append(pts_w[keep_idx])
            rgb_list.append(cols)

        points = np.concatenate(pts_world_list, axis=0).astype(np.float32)
        points_rgb = np.concatenate(rgb_list, axis=0).astype(np.uint8)
        points_err = np.zeros(len(points), dtype=np.float32)
        point_indices: Dict[str, np.ndarray] = {
            name: np.empty((0,), dtype=np.int32) for name in image_names
        }

        # scene_scale: gsplat uses max distance from camera centroid.
        camera_locations = camtoworlds[:, :3, 3]
        scene_center = camera_locations.mean(axis=0)
        dists = np.linalg.norm(camera_locations - scene_center, axis=1)
        self.scene_scale = float(np.max(dists)) if dists.size > 0 else 1.0
        # If cameras are nearly co-located (50 frames of slow driving), the
        # default scene_scale could be tiny. gsplat's lr is `1.6e-4 *
        # scene_scale`, so floor it.
        if self.scene_scale < 1.0:
            self.scene_scale = 1.0

        self.image_names = image_names
        self.image_paths = image_paths
        self.camtoworlds = camtoworlds.astype(np.float32)
        self.camera_ids = camera_ids
        self.Ks_dict = Ks_dict
        self.params_dict = params_dict
        self.imsize_dict = imsize_dict
        self.points = points
        self.points_err = points_err
        self.points_rgb = points_rgb
        self.point_indices = point_indices
        self.transform = np.eye(4, dtype=np.float64)

        # KB4 native path: leave undistortion maps EMPTY. Dataset will skip
        # cv2.remap when these are missing.
        self.mapx_dict: Dict[int, np.ndarray] = {}
        self.mapy_dict: Dict[int, np.ndarray] = {}
        self.roi_undist_dict: Dict[int, list] = {}

        print(f'[WovenParser] {n} frames  {W}×{H}  fx={K[0,0]:.2f} '
              f'k={list(dist)}')
        print(f'[WovenParser] {len(points)} init points,  '
              f'scene_scale={self.scene_scale:.3f}')


if __name__ == '__main__':
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument('--seq-dir', type=Path, required=True)
    ap.add_argument('--vehicle', default='248')
    ap.add_argument('--masks-dir', type=Path, default=None)
    ap.add_argument('--factor', type=int, default=1)
    args = ap.parse_args()
    p = WovenParser(args.seq_dir, vehicle=args.vehicle,
                     masks_dir=args.masks_dir, factor=args.factor)
    print(f'OK: {len(p.image_names)} frames, {len(p.points)} points, '
          f'scene_scale={p.scene_scale}, K[0]=\n{p.Ks_dict[0]}')
