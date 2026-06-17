"""WovenParserPinhole: Parser for the pre-undistorted+cropped+halved 2500x700
pinhole woven_pandaset_pylon/001_half data.

Inputs:
  - /raid/home/hfunaya/woven_pandaset_pylon/001_half/camera/front_camera/*.jpg
  - .../intrinsics.json (fx, fy, cx, cy, width, height; PINHOLE)
  - per-frame masks (downsampled to 2500x700)

Output: a Parser-shaped object compatible with simple_trainer.py PINHOLE
codepath. params_dict[0] = empty array (no distortion). World frame = rear_axle
of frame 0 (same as WovenParser fisheye), so we can reuse the camtoworlds
construction.
"""
from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Dict, List, Optional

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


class WovenParserPinhole:
    def __init__(
        self,
        pandaset_dir: Path,        # e.g. .../woven_pandaset_pylon/001_half
        seq_dir: Path,             # source woven sequence (for POSLV + LiDAR)
        vehicle: str = '248',
        recalib_json: Path = RECALIB_DEFAULT,
        masks_dir: Optional[Path] = None,
        factor: int = 1,
        test_every: int = 8,
        normalize: bool = False,
        sample_per_frame: int = 200_000,
        aggregate_radius: int = 0,
        pose_ckpt: Optional[Path] = None,
        gicp_T_world: Optional[Path] = None,
        lidar_dir_name: str = 'vls128_rear_axle',
    ):
        self.factor = factor
        self.test_every = test_every
        self.extconf = {'spiral_radius_scale': 1.0, 'no_factor_suffix': False}
        self.mask_dict: Dict[int, Optional[np.ndarray]] = {0: None}
        self.frame_mask_paths: Dict[str, Path] = {}

        pandaset_dir = Path(pandaset_dir)
        seq_dir = Path(seq_dir)

        cam_dir = pandaset_dir / 'camera' / 'front_camera'
        intr = json.loads((cam_dir / 'intrinsics.json').read_text())
        fx, fy, cx, cy = intr['fx'], intr['fy'], intr['cx'], intr['cy']
        W, H = int(intr['width']), int(intr['height'])
        K = np.array([[fx, 0, cx], [0, fy, cy], [0, 0, 1]], dtype=np.float64)

        # cropped pinhole jpgs (already undistorted)
        jpgs = sorted(cam_dir.glob('[0-9][0-9].jpg'))
        if len(jpgs) == 0:
            jpgs = sorted(cam_dir.glob('*.jpg'))
        n = len(jpgs)
        if n == 0:
            raise RuntimeError(f'no jpgs in {cam_dir}')

        # Need fcm pose for camtoworlds (POSLV chain)
        rec = json.loads(recalib_json.read_text())[vehicle]
        fcm = rec['fcm']; poslv = rec['poslv']
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

        # POSLV trajectory (skip if poslv files absent AND gicp_T_world given)
        pq = seq_dir / 'poslv' / 'POS.parquet'
        csv = seq_dir / 'poslv' / 'poslv.csv'
        cam_files_src = sorted((seq_dir / 'tss4_fcm').glob('*.jpg'))
        if len(cam_files_src) != n:
            raise RuntimeError(
                f'pandaset jpgs n={n} != source cam jpgs n={len(cam_files_src)}')
        camtoworlds = np.zeros((n, 4, 4), dtype=np.float64)
        T_rear_world_per_frame = np.zeros((n, 4, 4), dtype=np.float64)
        if pq.is_file() or csv.is_file():
            ts_a, e, n_, u, roll_v, pitch_v, head = DK.load_poslv_poses(
                pq if pq.is_file() else csv)

            def _interp_pose_enu(ts_q):
                from scipy.spatial.transform import Rotation as _Rot
                e_q = np.interp(ts_q, ts_a, e)
                n_q = np.interp(ts_q, ts_a, n_)
                u_q = np.interp(ts_q, ts_a, u)
                r_q = np.interp(ts_q, ts_a, roll_v)
                p_q = np.interp(ts_q, ts_a, pitch_v)
                h_q = np.interp(ts_q, ts_a, head)
                yaw_enu = (np.pi / 2.0) - h_q
                R_wb = _Rot.from_euler(
                    'zyx', np.stack([yaw_enu, p_q, r_q], axis=-1)).as_matrix()
                T = np.tile(np.eye(4), (len(ts_q), 1, 1))
                T[:, :3, :3] = R_wb
                T[:, :3, 3] = np.stack([e_q, n_q, u_q], axis=-1)
                return T

            cam_t0 = float(cam_files_src[0].stem.split('_')[1]) / 1000.0
            T_poslv_from_rear_at_t0 = _interp_pose_enu(np.array([cam_t0]))[0]
            T_rear0_from_poslv = np.linalg.inv(T_poslv_from_rear_at_t0)
            for i, jpg_src in enumerate(cam_files_src):
                cam_t = float(jpg_src.stem.split('_')[1]) / 1000.0
                T_poslv_from_rear_i = _interp_pose_enu(np.array([cam_t]))[0]
                T_world_from_rear_i = T_rear0_from_poslv @ T_poslv_from_rear_i
                T_rear_world_per_frame[i] = T_world_from_rear_i
                camtoworlds[i] = T_world_from_rear_i @ T_rear_from_pscam_static
        else:
            assert gicp_T_world is not None, (
                f'no POSLV files in {seq_dir}/poslv/ AND gicp_T_world=None')
            print(f'[WovenParserPinhole] POSLV missing; relying on gicp_T_world')
            for i in range(n):
                T_rear_world_per_frame[i] = np.eye(4)
                camtoworlds[i] = np.eye(4) @ T_rear_from_pscam_static

        # Bake pose_adjust deltas from a previous checkpoint into camtoworlds
        # (and per-frame rear_world used for LiDAR lift-up). After this, the
        # parser's c2w IS the pose-corrected c2w; trainer can run with
        # pose_opt off / lr 0 and get the right anchor uv too.
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

            # train frames: direct delta
            new_c2w = camtoworlds.copy()
            for di, fi in enumerate(train_idx):
                new_c2w[fi] = _apply(camtoworlds[fi], emb[di])
            # val frames: linear interp between two nearest train deltas
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
            # Update rear_world to keep T_w_from_rear consistent with the
            # corrected c2w: T_w_r = c2w @ inv(T_rear_from_pscam_static)
            T_pscam_from_rear = np.linalg.inv(T_rear_from_pscam_static)
            new_rw = np.zeros_like(T_rear_world_per_frame)
            for i in range(n):
                new_rw[i] = new_c2w[i] @ T_pscam_from_rear
            camtoworlds = new_c2w
            T_rear_world_per_frame = new_rw
            print(f'[WovenPinhole] pose_ckpt baked from {pose_ckpt} '
                  f'(train delta mean trans {np.linalg.norm(emb[:,:3], axis=1).mean()*100:.2f} cm)')

        # gicp_T_world : (n, 4, 4) per-frame T_world_from_rear, anchored at
        # frame 0. The keys (= image stems) are stamped at the LiDAR sweep
        # time, so this matrix == "T_world_from_rear at LiDAR-time".
        # We then interp it to the *fcm camera shutter time* for c2w used by
        # GS, while raw LiDAR points keep using the un-shifted (LiDAR-time)
        # matrix — that's the per-stream-timestamp story the user wants.
        if gicp_T_world is not None:
            T_wr_gicp = np.load(str(gicp_T_world))
            assert T_wr_gicp.shape == (n, 4, 4), f'gicp shape {T_wr_gicp.shape} vs {n}'
            T_rear_world_per_frame = T_wr_gicp.astype(np.float64)

            # Per-frame camera shutter ts from metadata.json::timestamps.
            T_rear_world_at_camera = T_rear_world_per_frame.copy()
            try:
                meta_p = seq_dir / 'metadata.json'
                meta = json.loads(meta_p.read_text()) if meta_p.is_file() else {}
                ts_block = meta.get('timestamps', {})
                # build sorted (key, lidar_ts_ns, camera_ts_ns)
                keys = sorted(ts_block.keys())
                lidar_ts = np.array([ts_block[k]['lidar_timestamp_ns']
                                     for k in keys], dtype=np.float64)
                cam_ts = np.array([ts_block[k]['camera_timestamps']['fcm']
                                   for k in keys], dtype=np.float64)
                if len(keys) == n:
                    from scipy.spatial.transform import Rotation as _Rot, Slerp as _Slerp
                    # SLERP+linear T_world_from_rear at camera times along
                    # the LiDAR-ts knots.
                    R_knots = _Rot.from_matrix(T_rear_world_per_frame[:, :3, :3])
                    slerp = _Slerp(lidar_ts, R_knots)
                    for i in range(n):
                        t_q = float(np.clip(cam_ts[i], lidar_ts[0], lidar_ts[-1]))
                        R_q = slerp([t_q]).as_matrix()[0]
                        # linear interp of translation
                        t_xyz = np.stack([
                            np.interp(t_q, lidar_ts, T_rear_world_per_frame[:, k, 3])
                            for k in range(3)])
                        Tc = np.eye(4); Tc[:3, :3] = R_q; Tc[:3, 3] = t_xyz
                        T_rear_world_at_camera[i] = Tc
                    print(f'[WovenPinhole] camera-time pose interp via metadata '
                          f'(camera_t − lidar_t mean = '
                          f'{(cam_ts - lidar_ts).mean()*1e-6:+.2f} ms)')
            except Exception as _e:
                print(f'[WovenPinhole] WARN: camera-time pose interp failed '
                      f'({_e}); using LiDAR-time pose for c2w too')

            for i in range(n):
                # c2w uses camera-time rear→world  → image is observed at this pose
                camtoworlds[i] = T_rear_world_at_camera[i] @ T_rear_from_pscam_static
            print(f'[WovenPinhole] gicp_T_world baked from {gicp_T_world}')

        image_names = [j.stem for j in jpgs]
        image_paths = [str(j) for j in jpgs]
        camera_ids = [0] * n
        Ks_dict = {0: K.astype(np.float64)}
        params_dict = {0: np.zeros(0, dtype=np.float64)}  # PINHOLE (no distortion)
        imsize_dict = {0: (W, H)}

        if masks_dir is not None:
            masks_dir = Path(masks_dir)
            for jpg, src in zip(jpgs, cam_files_src):
                cand = masks_dir / f'{src.stem}.png'
                if cand.is_file():
                    self.frame_mask_paths[jpg.stem] = cand

        # ── point cloud (LiDAR + RGB sampled from undistorted images) ──
        rng = np.random.default_rng(0)
        pts_world_list = []
        rgb_list = []
        lid_files = sorted((seq_dir / lidar_dir_name).glob('*.npz'))
        for i, npz in enumerate(lid_files):
            d = np.load(npz)
            xs, ys, zs = d['xs'], d['ys'], d['zs']
            pts_rear = np.stack([xs, ys, zs], axis=-1).astype(np.float64)
            # `T_rear_world_per_frame[i]` carries the same timestamp as the
            # LiDAR sweep (poses dict key = LiDAR ts). Lifting raw LiDAR
            # points with it puts them in world at LiDAR time.
            T_w_r = T_rear_world_per_frame[i]
            pts_w = (T_w_r[:3, :3] @ pts_rear.T + T_w_r[:3, 3:4]).T
            T_w2c = np.linalg.inv(camtoworlds[i])
            pts_c = (T_w2c[:3, :3] @ pts_w.T + T_w2c[:3, 3:4]).T
            Z = pts_c[:, 2]
            in_front = Z > 0.5
            if not in_front.any():
                continue
            pcv = pts_c[in_front]
            uvh = (K @ pcv.T).T
            uv = uvh[:, :2] / uvh[:, 2:3]
            in_b = ((uv[:, 0] >= 0) & (uv[:, 0] < W)
                    & (uv[:, 1] >= 0) & (uv[:, 1] < H))
            idx_front = np.where(in_front)[0]
            keep_idx = idx_front[in_b]
            uv_kept = uv[in_b]
            if len(keep_idx) == 0:
                continue

            # mask drop using full-res masks (3840x1952)
            jpg = jpgs[i]
            if jpg.stem in self.frame_mask_paths:
                mp = self.frame_mask_paths[jpg.stem]
                m = cv2.imread(str(mp), cv2.IMREAD_GRAYSCALE)
                if m.shape[:2] != (H, W):
                    m = cv2.resize(m, (W, H), interpolation=cv2.INTER_NEAREST)
                u_int = np.clip(np.round(uv_kept[:, 0]).astype(np.int32),
                                 0, W - 1)
                v_int = np.clip(np.round(uv_kept[:, 1]).astype(np.int32),
                                 0, H - 1)
                keep_mask = m[v_int, u_int] > 127
                keep_idx = keep_idx[keep_mask]
                uv_kept = uv_kept[keep_mask]
            if len(keep_idx) == 0:
                continue

            if len(keep_idx) > sample_per_frame:
                sel = rng.choice(len(keep_idx), sample_per_frame, replace=False)
                keep_idx = keep_idx[sel]
                uv_kept = uv_kept[sel]

            img = cv2.imread(image_paths[i])
            rgb_img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            u_i = np.clip(np.round(uv_kept[:, 0]).astype(np.int32), 0, W - 1)
            v_i = np.clip(np.round(uv_kept[:, 1]).astype(np.int32), 0, H - 1)
            cols = rgb_img[v_i, u_i].astype(np.uint8)
            pts_world_list.append(pts_w[keep_idx])
            rgb_list.append(cols)

        # build cumulative offsets so we know which slice of `points` came
        # from which frame; that is exactly the per-frame visible-points list
        # the colmap Dataset.load_depths expects in `point_indices`.
        sizes = [len(x) for x in pts_world_list]
        offsets = np.cumsum([0] + sizes)
        points = np.concatenate(pts_world_list, axis=0).astype(np.float32)
        points_rgb = np.concatenate(rgb_list, axis=0).astype(np.uint8)
        points_err = np.zeros(len(points), dtype=np.float32)
        point_indices: Dict[str, np.ndarray] = {}
        for i, name in enumerate(image_names):
            point_indices[name] = np.arange(offsets[i], offsets[i + 1],
                                             dtype=np.int32)

        camera_locations = camtoworlds[:, :3, 3]
        scene_center = camera_locations.mean(axis=0)
        dists = np.linalg.norm(camera_locations - scene_center, axis=1)
        self.scene_scale = float(np.max(dists)) if dists.size > 0 else 1.0
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
        self.mapx_dict: Dict[int, np.ndarray] = {}
        self.mapy_dict: Dict[int, np.ndarray] = {}
        self.roi_undist_dict: Dict[int, list] = {}

        print(f'[WovenPinhole] {n} frames  {W}×{H}  fx={K[0,0]:.2f} '
              f'cx={K[0,2]:.2f} cy={K[1,2]:.2f}')
        print(f'[WovenPinhole] {len(points)} init pts  '
              f'scene_scale={self.scene_scale:.3f}')


if __name__ == '__main__':
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument('--pandaset-dir', type=Path, required=True)
    ap.add_argument('--seq-dir', type=Path, required=True)
    ap.add_argument('--vehicle', default='248')
    ap.add_argument('--masks-dir', type=Path, default=None)
    args = ap.parse_args()
    p = WovenParserPinhole(args.pandaset_dir, args.seq_dir,
                            vehicle=args.vehicle, masks_dir=args.masks_dir)
    print(f'OK: {len(p.image_names)} frames, {len(p.points)} pts')
