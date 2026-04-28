"""Streaming per-instance PandaSet lazy cache builder.

Reproduces datasets.pandaset.build_cache (cuboid-anchored crops + per-frame
random crops) but emits per-instance .pt files directly without an in-RAM
accumulation. Lets us go to cache_img=384 without 100GB RAM blowup.

Layout:
    {out}/inst/{0000000.pt, ...}   # one file per instance (~440 KB at 384²)
    {out}/meta.pt                  # {'train': [fname, ...], 'val': [...]}
"""
import sys, pathlib; sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[2]))
import argparse, gzip, json, pickle, random, time
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from scipy.spatial.transform import Rotation

from datasets.pandaset import (USEFUL_LABELS, _quat_pos_to_mat, _project,
                                _bbox2d_of_cuboid)


def _load_pkl(path: Path):
    if path.suffix == '.gz':
        with gzip.open(path, 'rb') as f:
            return pickle.load(f)
    return pickle.load(open(path, 'rb'))


def _process_scene(args_tuple):
    (scene_root_str, scene_name, cache_img, bbox_scale, min_pts, random_crops,
     out_inst_str, gid_start) = args_tuple
    sc_dir = Path(scene_root_str) / scene_name
    out_inst = Path(out_inst_str)
    cam_dir = sc_dir / 'camera' / 'front_camera'
    if not cam_dir.exists():
        return scene_name, []
    with open(cam_dir / 'poses.json') as f:
        all_poses = json.load(f)
    with open(cam_dir / 'intrinsics.json') as f:
        intr = json.load(f)
    K_full = np.array([[intr['fx'], 0, intr['cx']],
                       [0, intr['fy'], intr['cy']],
                       [0, 0, 1]], dtype=np.float64)

    cb_dir = sc_dir / 'annotations' / 'cuboids'
    ld_dir = sc_dir / 'lidar'
    cuboid_files = sorted(list(cb_dir.glob('*.pkl')) + list(cb_dir.glob('*.pkl.gz')))
    lidar_files  = sorted(list(ld_dir.glob('*.pkl')) + list(ld_dir.glob('*.pkl.gz')))
    n_frames = min(len(all_poses), len(cuboid_files), len(lidar_files))

    IW, IH = 1920, 1080
    out_files = []
    gid = gid_start
    rng = random.Random(0xfeedbeef ^ hash(scene_name))

    for fi in range(n_frames):
        cam_pose = all_poses[fi]
        pose_mat = _quat_pos_to_mat(cam_pose['heading'], cam_pose['position'])
        cam_pos = np.array([cam_pose['position']['x'],
                            cam_pose['position']['y'],
                            cam_pose['position']['z']], dtype=np.float32)

        lidar_df = _load_pkl(lidar_files[fi])
        if 'd' in lidar_df.columns:
            lidar_df = lidar_df[lidar_df['d'] == 0]
        pts_world = lidar_df[['x', 'y', 'z']].values.astype(np.float32)

        uv_gt, z_gt = _project(pts_world, pose_mat, K_full)
        vis = (z_gt > 0.5) & (uv_gt[:, 0] >= 0) & (uv_gt[:, 0] < IW) & \
              (uv_gt[:, 1] >= 0) & (uv_gt[:, 1] < IH)
        if vis.sum() < min_pts:
            continue
        pts_vis = pts_world[vis]

        img_full = Image.open(cam_dir / f'{fi:02d}.jpg').convert('RGB')
        cuboids = _load_pkl(cuboid_files[fi])
        if 'label' in cuboids.columns:
            cuboids = cuboids[cuboids['label'].isin(USEFUL_LABELS)]

        R_gt = Rotation.from_quat([cam_pose['heading']['x'],
                                   cam_pose['heading']['y'],
                                   cam_pose['heading']['z'],
                                   cam_pose['heading']['w']]).as_matrix().astype(np.float32)
        T_gt = np.linalg.inv(_quat_pos_to_mat(cam_pose['heading'],
                              cam_pose['position'])).astype(np.float32)

        n_obj_this_frame = 0
        for _, obj in cuboids.iterrows():
            pos = np.array([obj['position.x'], obj['position.y'], obj['position.z']])
            dims = np.array([obj['dimensions.x'], obj['dimensions.y'], obj['dimensions.z']])
            yaw = float(obj['yaw'])
            bbox = _bbox2d_of_cuboid(pos, dims, yaw, pose_mat, K_full)
            if bbox is None: continue
            u_min, v_min, u_max, v_max = bbox
            uc, vc = (u_min + u_max) / 2, (v_min + v_max) / 2
            if not (0 <= uc < IW and 0 <= vc < IH): continue
            bw, bh = u_max - u_min, v_max - v_min
            crop_size = max(bw, bh) * bbox_scale
            crop_size = max(crop_size, 64)
            half = crop_size / 2
            u0 = float(np.clip(uc - half, 0, IW - crop_size))
            v0 = float(np.clip(vc - half, 0, IH - crop_size))
            crop_size = float(crop_size)

            box = (int(u0), int(v0), int(u0 + crop_size), int(v0 + crop_size))
            img_cache = np.array(
                img_full.crop(box).resize((cache_img, cache_img), Image.BILINEAR),
                dtype=np.uint8)
            inst = {
                'img_cache': torch.from_numpy(img_cache).permute(2, 0, 1),
                'pts':       torch.from_numpy(pts_vis),
                'cam_pos':   torch.from_numpy(cam_pos),
                'R_gt':      torch.from_numpy(R_gt),
                'T_gt':      torch.from_numpy(T_gt),
                'K_full':    torch.from_numpy(K_full.astype(np.float32)),
                'u0': u0, 'v0': v0, 'crop_size': crop_size,
                'obj_pos':   torch.from_numpy(pos.astype(np.float32)),
                'obj_dims':  torch.from_numpy(dims.astype(np.float32)),
                'obj_yaw':   yaw,
            }
            fname = f'{gid:08d}.pt'
            torch.save(inst, out_inst / fname)
            out_files.append(fname)
            gid += 1
            n_obj_this_frame += 1

        if random_crops and n_obj_this_frame > 0:
            uv_v, z_v = _project(pts_vis, pose_mat, K_full)
            rcs = int(IW * 0.10)
            for _ in range(n_obj_this_frame):
                ru0 = rng.randint(0, IW - rcs)
                rv0 = rng.randint(0, IH - rcs)
                in_c = ((uv_v[:, 0] >= ru0) & (uv_v[:, 0] < ru0 + rcs) &
                        (uv_v[:, 1] >= rv0) & (uv_v[:, 1] < rv0 + rcs) &
                        (z_v > 0.5))
                if in_c.sum() < min_pts: continue
                box_r = (ru0, rv0, ru0 + rcs, rv0 + rcs)
                img_r = np.array(
                    img_full.crop(box_r).resize((cache_img, cache_img), Image.BILINEAR),
                    dtype=np.uint8)
                inst = {
                    'img_cache': torch.from_numpy(img_r).permute(2, 0, 1),
                    'pts':       torch.from_numpy(pts_vis),
                    'cam_pos':   torch.from_numpy(cam_pos),
                    'R_gt':      torch.from_numpy(R_gt),
                    'T_gt':      torch.from_numpy(T_gt),
                    'K_full':    torch.from_numpy(K_full.astype(np.float32)),
                    'u0': float(ru0), 'v0': float(rv0), 'crop_size': float(rcs),
                    'obj_bbox':  torch.zeros(4),
                }
                fname = f'{gid:08d}.pt'
                torch.save(inst, out_inst / fname)
                out_files.append(fname)
                gid += 1

    return scene_name, out_files


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--root',       default='/mnt/nvme6t/pandaset')
    ap.add_argument('--out',        default='/mnt/nvme6t/e2e_calib_cache/pandaset_mc_s384_lazy')
    ap.add_argument('--cache-img',  type=int,   default=384)
    ap.add_argument('--bbox-scale', type=float, default=3.0)
    ap.add_argument('--min-pts',    type=int,   default=8)
    ap.add_argument('--val-frac',   type=float, default=0.15)
    ap.add_argument('--seed',       type=int,   default=42)
    ap.add_argument('--max-scenes', type=int,   default=None)
    ap.add_argument('--workers',    type=int,   default=12)
    ap.add_argument('--random-crops', action='store_true', default=False)
    args = ap.parse_args()

    root = Path(args.root); out = Path(args.out)
    inst_dir = out / 'inst'; inst_dir.mkdir(parents=True, exist_ok=True)

    scenes = sorted(p.name for p in root.iterdir() if p.is_dir())
    rng = random.Random(args.seed); rng.shuffle(scenes)
    if args.max_scenes:
        scenes = scenes[:args.max_scenes]
    n_val = max(1, int(len(scenes) * args.val_frac))
    val_set = set(scenes[:n_val])

    print(f'Build cache_img={args.cache_img} from {len(scenes)} scenes  '
          f'(val={n_val}, train={len(scenes)-n_val})  workers={args.workers}', flush=True)
    GID_PER_SCENE = 20_000
    meta = {'train': [], 'val': []}
    t0 = time.time()
    futures = []
    with ProcessPoolExecutor(max_workers=args.workers) as ex:
        for si, sc in enumerate(scenes):
            split = 'val' if sc in val_set else 'train'
            futures.append((split, ex.submit(
                _process_scene,
                (str(root), sc, args.cache_img, args.bbox_scale, args.min_pts,
                 args.random_crops, str(inst_dir), si * GID_PER_SCENE))))
        total = 0
        for i, (split, fut) in enumerate(futures):
            sc, files = fut.result()
            meta[split].extend(files)
            total += len(files)
            elapsed = time.time() - t0
            print(f'  [{i+1}/{len(futures)}] {sc} +{len(files)} ({split}) | total={total} '
                  f'rate={total/max(elapsed,1e-3):.0f}/s elapsed={elapsed/60:.1f}min',
                  flush=True)
    torch.save(meta, out / 'meta.pt')
    print(f'\ndone: train={len(meta["train"])}  val={len(meta["val"])}  total={total}')
    print(f'output: {out}')


if __name__ == '__main__':
    main()
