"""Build a lazy disk-backed PandaSet cache (v2 layout: 384x384 full-res crops,
3 closest objects + 6 stratified random per (frame, cam), all 6 cameras).

Difference vs v1 (pandaset_mc_s64):
  - CACHE_IMG = 384 (was 192) and the crop is taken at FULL-RES 384x384 — NO
    resize. So a far-distance object with a 30-px bbox is shown inside a
    384-px patch with surrounding context, not stretched 6× from a 60-px
    bbox.
  - Per (frame, cam) we store 3 + 6 = 9 instances (was 1 per obj, leading to
    overlapping pixel duplication for crowded frames).
  - Object selection: top 3 closest by depth-to-camera (= largest bbox proxy).
  - Stratified random: image divided 2x3 = 6 cells, one random 384x384 crop
    per cell (clipped to image bounds).

Each instance file mirrors the v1 schema as much as possible:
  inst.pt = {
    'img_cache': (3, 384, 384) uint8,
    'pts':       (Nlidar, 3) float32,         # full lidar visible in this cam
    'cam_pos':   (3,) float32,
    'R_gt':      (3, 3) float32,
    'T_gt':      (4, 4) float32,
    'K_full':    (3, 3) float32,
    'u0', 'v0', 'crop_size':  full-res crop top-left + side (== 384),
    # one of these (mirrors v1):
    'obj_pos','obj_dims','obj_yaw'   # for the 3 object-anchored crops
    'obj_bbox': zeros(4)              # for the 6 stratified-random crops
  }
"""
from __future__ import annotations
import sys, pathlib; sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[2]))

import argparse, json, pickle, random
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from scipy.spatial.transform import Rotation

from datasets.pandaset import (
    USEFUL_LABELS, _quat_pos_to_mat, _project, _bbox2d_of_cuboid,
)


CACHE_IMG_V2 = 384
GRID_RAND = (2, 3)        # 2 rows × 3 cols = 6 stratified cells
N_OBJ      = 3
N_RAND     = GRID_RAND[0] * GRID_RAND[1]
CAM_NAMES  = ['front_camera', 'front_left_camera', 'front_right_camera',
              'back_camera',  'left_camera',       'right_camera']


def _save_inst(d: dict, out_dir: Path, idx: int):
    out_dir.mkdir(parents=True, exist_ok=True)
    torch.save(d, out_dir / f'{idx:07d}.pt')


def _crop_clamp(img: Image.Image, uc: float, vc: float, side: int,
                IW: int, IH: int) -> tuple[np.ndarray, float, float]:
    """Take a `side x side` crop centered on (uc, vc), clipped to image bounds.
    Pads with zeros for any out-of-image area. Returns (img_np, u0, v0)."""
    half = side // 2
    u0 = int(round(uc - half)); v0 = int(round(vc - half))
    u0 = max(0, min(IW - side, u0))
    v0 = max(0, min(IH - side, v0))
    box = (u0, v0, u0 + side, v0 + side)
    arr = np.array(img.crop(box), dtype=np.uint8)
    if arr.shape[0] != side or arr.shape[1] != side:
        # very small image; pad
        pad = np.zeros((side, side, 3), dtype=np.uint8)
        h = min(side, arr.shape[0]); w = min(side, arr.shape[1])
        pad[:h, :w] = arr[:h, :w]
        arr = pad
    return arr, float(u0), float(v0)


def _process_frame_cam(scene: str, sc_dir: Path, cam_name: str, fi: int,
                        all_poses: list, K_full: np.ndarray, cuboids,
                        IW: int, IH: int, side: int,
                        n_obj: int, n_rand: int, rng: random.Random,
                        min_pts: int) -> list[dict]:
    """Emit up to n_obj + n_rand instance dicts for one (scene, fi, cam)."""
    cam_dir = sc_dir / 'camera' / cam_name
    img_path = cam_dir / f'{fi:02d}.jpg'
    if not img_path.exists():
        return []

    cam_pose = all_poses[fi]
    pose_mat = _quat_pos_to_mat(cam_pose['heading'], cam_pose['position'])
    cam_pos  = np.array([cam_pose['position']['x'], cam_pose['position']['y'],
                         cam_pose['position']['z']], dtype=np.float32)

    lidar_df  = pickle.load(open(sc_dir / 'lidar' / f'{fi:02d}.pkl', 'rb'))
    lidar_df  = lidar_df[lidar_df['d'] == 0]
    pts_world = lidar_df[['x','y','z']].values.astype(np.float32)
    uv_gt, z_gt = _project(pts_world, pose_mat, K_full)
    vis = (z_gt > 0.5) & (uv_gt[:,0] >= 0) & (uv_gt[:,0] < IW) & \
          (uv_gt[:,1] >= 0) & (uv_gt[:,1] < IH)
    if vis.sum() < min_pts:
        return []
    pts_vis = pts_world[vis]
    uv_vis  = uv_gt[vis]
    z_vis   = z_gt[vis]

    img_full = Image.open(img_path).convert('RGB')

    R_gt = Rotation.from_quat([cam_pose['heading']['x'], cam_pose['heading']['y'],
                                cam_pose['heading']['z'], cam_pose['heading']['w']
                                ]).as_matrix().astype(np.float32)
    T_gt = np.linalg.inv(pose_mat).astype(np.float32)
    common = dict(
        pts     = torch.from_numpy(pts_vis),
        cam_pos = torch.from_numpy(cam_pos),
        R_gt    = torch.from_numpy(R_gt),
        T_gt    = torch.from_numpy(T_gt),
        K_full  = torch.from_numpy(K_full.astype(np.float32)),
        crop_size = float(side),
        # provenance — so any inst can be traced back to its source jpg.
        # Without this, debugging "this patch looks wrong" requires matching
        # cam_pos against pandaset metadata after-the-fact (expensive).
        scene   = scene,
        cam     = cam_name,
        frame   = int(fi),
    )
    half = side // 2

    out: list[dict] = []

    # 3 closest object-anchored crops (depth ascending)
    obj_candidates = []
    for _, obj in cuboids.iterrows():
        pos  = np.array([obj['position.x'], obj['position.y'], obj['position.z']])
        dims = np.array([obj['dimensions.x'], obj['dimensions.y'], obj['dimensions.z']])
        yaw  = obj['yaw']
        bbox = _bbox2d_of_cuboid(pos, dims, yaw, pose_mat, K_full)
        if bbox is None: continue
        u_min, v_min, u_max, v_max = bbox
        uc, vc = (u_min+u_max)/2, (v_min+v_max)/2
        if not (0 <= uc < IW and 0 <= vc < IH): continue
        obj_pos_cam = (pose_mat[:3,:3].T @ (pos - cam_pos)).astype(np.float32)
        depth = float(obj_pos_cam[2])
        if depth < 0.5 or depth > 80: continue
        obj_candidates.append((depth, uc, vc, pos, dims, yaw))
    obj_candidates.sort(key=lambda t: t[0])
    for depth, uc, vc, pos, dims, yaw in obj_candidates[:n_obj]:
        arr, u0, v0 = _crop_clamp(img_full, uc, vc, side, IW, IH)
        in_box = (uv_vis[:, 0] >= u0) & (uv_vis[:, 0] < u0 + side) & \
                 (uv_vis[:, 1] >= v0) & (uv_vis[:, 1] < v0 + side)
        if in_box.sum() < min_pts: continue
        d = dict(common)
        d['img_cache'] = torch.from_numpy(arr).permute(2, 0, 1)
        d['u0'], d['v0'] = u0, v0
        d['obj_pos']  = torch.from_numpy(pos.astype(np.float32))
        d['obj_dims'] = torch.from_numpy(dims.astype(np.float32))
        d['obj_yaw']  = float(yaw)
        out.append(d)

    # n_rand stratified random crops (one per grid cell)
    rows, cols = GRID_RAND
    cell_w = IW / cols
    cell_h = IH / rows
    half_s = side / 2
    for r in range(rows):
        for c in range(cols):
            cu_min = max(half_s, c * cell_w)
            cu_max = min(IW - half_s, (c + 1) * cell_w)
            cv_min = max(half_s, r * cell_h)
            cv_max = min(IH - half_s, (r + 1) * cell_h)
            if cu_max <= cu_min or cv_max <= cv_min:
                continue
            uc = rng.uniform(cu_min, cu_max)
            vc = rng.uniform(cv_min, cv_max)
            arr, u0, v0 = _crop_clamp(img_full, uc, vc, side, IW, IH)
            in_box = (uv_vis[:, 0] >= u0) & (uv_vis[:, 0] < u0 + side) & \
                     (uv_vis[:, 1] >= v0) & (uv_vis[:, 1] < v0 + side)
            if in_box.sum() < min_pts: continue
            d = dict(common)
            d['img_cache'] = torch.from_numpy(arr).permute(2, 0, 1)
            d['u0'], d['v0'] = u0, v0
            d['obj_bbox'] = torch.zeros(4)
            out.append(d)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--root', default='/mnt/nvme6t/pandaset')
    ap.add_argument('--out',  default='/mnt/nvme6t/e2e_calib_cache/pandaset_v2_384')
    ap.add_argument('--max-scenes', type=int, default=None)
    ap.add_argument('--cams', default=','.join(CAM_NAMES),
                    help='comma-separated camera list (default = all 6)')
    ap.add_argument('--side',  type=int, default=CACHE_IMG_V2)
    ap.add_argument('--n-obj', type=int, default=N_OBJ)
    ap.add_argument('--n-rand',type=int, default=N_RAND)
    ap.add_argument('--val-fraction', type=float, default=0.15)
    ap.add_argument('--seed',  type=int, default=42)
    ap.add_argument('--min-pts', type=int, default=8)
    args = ap.parse_args()

    cams = [c.strip() for c in args.cams.split(',') if c.strip()]
    root = Path(args.root)
    out  = Path(args.out)
    inst_dir = out / 'inst'
    inst_dir.mkdir(parents=True, exist_ok=True)
    rng  = random.Random(args.seed)

    scenes = sorted(p.name for p in root.iterdir() if p.is_dir())
    rng.shuffle(scenes)
    if args.max_scenes:
        scenes = scenes[:args.max_scenes]
    n_val = max(0, int(len(scenes) * args.val_fraction))

    train_files: list[str] = []
    val_files:   list[str] = []
    instance_idx = 0

    IW, IH = 1920, 1080
    print(f'building v2 cache  side={args.side}  obj={args.n_obj}  rand={args.n_rand}')
    print(f'  scenes: {len(scenes)} (val={n_val})  cams: {cams}')

    for si, scene in enumerate(scenes):
        sc_dir = root / scene
        if not sc_dir.is_dir(): continue
        cuboid_files = sorted((sc_dir / 'annotations' / 'cuboids').glob('*.pkl'))
        lidar_files  = sorted((sc_dir / 'lidar').glob('*.pkl'))
        n_frames = min(len(cuboid_files), len(lidar_files))
        is_val = si < n_val

        scene_count = 0
        for cam_name in cams:
            cam_dir = sc_dir / 'camera' / cam_name
            if not cam_dir.exists(): continue
            with open(cam_dir / 'poses.json') as f:
                all_poses = json.load(f)
            with open(cam_dir / 'intrinsics.json') as f:
                intr = json.load(f)
            K_full = np.array([[intr['fx'], 0, intr['cx']],
                               [0, intr['fy'], intr['cy']],
                               [0, 0, 1]], dtype=np.float64)
            n_frames_cam = min(n_frames, len(all_poses))
            for fi in range(n_frames_cam):
                cuboids = pickle.load(open(cuboid_files[fi], 'rb'))
                cuboids = cuboids[cuboids['label'].isin(USEFUL_LABELS)]
                instances = _process_frame_cam(
                    scene, sc_dir, cam_name, fi, all_poses, K_full, cuboids,
                    IW, IH, args.side, args.n_obj, args.n_rand,
                    rng, args.min_pts)
                for inst in instances:
                    fname = f'{instance_idx:07d}.pt'
                    torch.save(inst, inst_dir / fname)
                    (val_files if is_val else train_files).append(fname)
                    instance_idx += 1
                    scene_count += 1
        print(f'  [{si+1}/{len(scenes)}] {scene}  +{scene_count} inst   '
              f'(total: train={len(train_files)} val={len(val_files)})')

    torch.save({'train': train_files, 'val': val_files,
                 'side': args.side, 'cams': cams,
                 'n_obj': args.n_obj, 'n_rand': args.n_rand},
                out / 'meta.pt')
    total_files = len(train_files) + len(val_files)
    print(f'done.  total instances: {total_files}  ({len(train_files)} train + {len(val_files)} val)')
    print(f'output: {out}')


if __name__ == '__main__':
    main()
