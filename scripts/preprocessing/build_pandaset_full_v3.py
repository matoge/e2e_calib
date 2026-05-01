"""V3 cache: full-image (decoded JPG) + per-frame metadata.

No crop pre-baked. Per (scene, cam, frame) we save:
  img:       (3, IH, IW) uint8                 — decoded image
  pts:       (N_vis, 3) float32                — visible (in-image) lidar pts (world)
  cam_pos:   (3,)                              — camera world position
  R_gt:      (3, 3)                            — cam-to-world rotation
  T_gt:      (4, 4)                            — world-to-cam (inv pose)
  K_full:    (3, 3)                            — intrinsic
  cuboids:   list[dict] of useful labels       — for is_obj computation per-crop at __getitem__ time
  scene/cam/frame                              — provenance

Storage (front_camera, 103 scenes, ~80 frames each):
  ~80 frames × 6.2 MB image + ~0.4 MB lidar ≈ 7 MB/frame
  total ≈ 103 × 80 × 7 MB ≈ 58 GB

Crop is decided at __getitem__ time → no center bias from cache.
"""
import argparse, sys, os, gzip, pickle, json, time
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
import numpy as np
import torch
from PIL import Image
from scipy.spatial.transform import Rotation
from concurrent.futures import ProcessPoolExecutor, as_completed

from datasets.pandaset import (USEFUL_LABELS, _quat_pos_to_mat, _project)


def _load_pkl(path: Path):
    if path.suffix == '.gz':
        with gzip.open(path, 'rb') as f:
            return pickle.load(f)
    return pickle.load(open(path, 'rb'))


def _process_scene(args_tuple):
    (scene_root, scene_name, cam_name, out_dir, min_pts, gid_start, stride) = args_tuple
    sc_dir = Path(scene_root) / scene_name
    cam_dir = sc_dir / 'camera' / cam_name
    if not cam_dir.exists():
        return scene_name, []
    poses = json.load(open(cam_dir / 'poses.json'))
    intr  = json.load(open(cam_dir / 'intrinsics.json'))
    K = np.array([[intr['fx'], 0, intr['cx']],
                  [0, intr['fy'], intr['cy']],
                  [0, 0, 1]], dtype=np.float64)

    cb_dir = sc_dir / 'annotations' / 'cuboids'
    ld_dir = sc_dir / 'lidar'
    # PandaSet ships BOTH 00.pkl and 00.pkl.gz (identical content).
    # If we sorted both together, cubs[44]=22.pkl while image fi=44 → frame mismatch.
    # Prefer .pkl; fall back to .pkl.gz only when .pkl missing.
    def _frame_files(d: Path) -> list[Path]:
        by_stem = {}
        for f in list(d.glob('*.pkl')) + list(d.glob('*.pkl.gz')):
            stem = f.name.split('.', 1)[0]  # '44' from '44.pkl' or '44.pkl.gz'
            if stem not in by_stem or f.suffix == '.pkl':
                by_stem[stem] = f
        return [by_stem[k] for k in sorted(by_stem.keys(), key=int)]

    cubs = _frame_files(cb_dir)
    lids = _frame_files(ld_dir)
    n = min(len(poses), len(cubs), len(lids))

    IW, IH = 1920, 1080
    out_files = []
    gid = gid_start

    for fi in range(0, n, stride):
        cp = poses[fi]
        pose_mat = _quat_pos_to_mat(cp['heading'], cp['position'])
        cam_pos = np.array([cp['position']['x'], cp['position']['y'], cp['position']['z']],
                           dtype=np.float32)

        df = _load_pkl(lids[fi])
        if 'd' in df.columns:
            df = df[df['d'] == 0]
        pts_world = df[['x','y','z']].values.astype(np.float32)
        uv, z = _project(pts_world, pose_mat, K)
        vis = (z > 0.5) & (uv[:,0] >= 0) & (uv[:,0] < IW) & (uv[:,1] >= 0) & (uv[:,1] < IH)
        if vis.sum() < min_pts:
            continue
        pts_vis = pts_world[vis]

        img_path = cam_dir / f'{fi:02d}.jpg'
        if not img_path.exists():
            continue
        # Store JPEG bytes inline (~200KB) instead of decoded raw uint8 (~6MB).
        # 30× cache shrink: 50GB → ~1.6GB. Decode is done at __getitem__ time
        # via TurboJPEG (~10ms/sample full-decode, faster with crop region).
        jpg_bytes = img_path.read_bytes()
        # Read dims from header without decoding pixels
        with Image.open(img_path) as _im:
            IW, IH = _im.size  # (W, H) — flip to (H, W) below

        cubs_df = _load_pkl(cubs[fi])
        if 'label' in cubs_df.columns:
            cubs_df = cubs_df[cubs_df['label'].isin(USEFUL_LABELS)]
        cub_list = []
        for _, obj in cubs_df.iterrows():
            cub_list.append(dict(
                pos  = np.array([obj['position.x'], obj['position.y'], obj['position.z']], dtype=np.float32),
                dims = np.array([obj['dimensions.x'], obj['dimensions.y'], obj['dimensions.z']], dtype=np.float32),
                yaw  = float(obj['yaw']),
                label= str(obj['label']),
            ))

        R_gt = Rotation.from_quat([cp['heading']['x'], cp['heading']['y'],
                                    cp['heading']['z'], cp['heading']['w']
                                    ]).as_matrix().astype(np.float32)
        T_gt = np.linalg.inv(pose_mat).astype(np.float32)

        inst = dict(
            jpg_bytes = jpg_bytes,
            IH        = int(IH),
            IW        = int(IW),
            pts      = torch.from_numpy(pts_vis),
            cam_pos  = torch.from_numpy(cam_pos),
            R_gt     = torch.from_numpy(R_gt),
            T_gt     = torch.from_numpy(T_gt),
            K_full   = torch.from_numpy(K.astype(np.float32)),
            cuboids  = cub_list,
            scene    = scene_name,
            cam      = cam_name,
            frame    = int(fi),
        )
        fname = f'{gid:08d}.pt'
        torch.save(inst, Path(out_dir) / fname)
        out_files.append(fname)
        gid += 1

    return scene_name, out_files


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--root',    default='/mnt/nvme6t/pandaset')
    ap.add_argument('--out',     default='/mnt/nvme6t/e2e_calib_cache/pandaset_v3_full')
    ap.add_argument('--cam',     default='front_camera')
    ap.add_argument('--workers', type=int, default=8)
    ap.add_argument('--val-frac',type=float, default=0.15)
    ap.add_argument('--seed',    type=int,   default=42)
    ap.add_argument('--min-pts', type=int,   default=8)
    ap.add_argument('--max-scenes', type=int, default=None)
    ap.add_argument('--stride', type=int, default=1, help='frame stride (5 → 2Hz)')
    args = ap.parse_args()

    out_dir = Path(args.out)
    inst_dir = out_dir / 'inst'
    inst_dir.mkdir(parents=True, exist_ok=True)

    root = Path(args.root)
    scenes = sorted([p.name for p in root.iterdir() if p.is_dir() and p.name.isdigit()])
    import random
    rng = random.Random(args.seed); rng.shuffle(scenes)
    if args.max_scenes:
        scenes = scenes[:args.max_scenes]
    n_val = max(1, int(len(scenes) * args.val_frac))
    val_scenes = set(scenes[:n_val])
    print(f'cam={args.cam} scenes={len(scenes)} val={n_val} train={len(scenes)-n_val} workers={args.workers}', flush=True)

    # estimate gid stride: ~80 frames per scene, conservative buffer
    gid_stride = 200
    train_files, val_files = [], []
    t0 = time.time()
    if args.workers <= 1:
        for si, sc in enumerate(scenes):
            _, fnames = _process_scene((str(root), sc, args.cam, str(inst_dir), args.min_pts, si * gid_stride, args.stride))
            (val_files if sc in val_scenes else train_files).extend(fnames)
            print(f'[{si+1}/{len(scenes)}] {sc} +{len(fnames)} ({time.time()-t0:.0f}s)', flush=True)
    else:
        argv = [(str(root), sc, args.cam, str(inst_dir), args.min_pts, si * gid_stride, args.stride)
                for si, sc in enumerate(scenes)]
        with ProcessPoolExecutor(max_workers=args.workers) as ex:
            futures = {ex.submit(_process_scene, a): a[1] for a in argv}
            done = 0
            for fut in as_completed(futures):
                sc, fnames = fut.result()
                (val_files if sc in val_scenes else train_files).extend(fnames)
                done += 1
                print(f'[{done}/{len(scenes)}] {sc} +{len(fnames)} ({time.time()-t0:.0f}s)', flush=True)

    meta = {'train': sorted(train_files), 'val': sorted(val_files), 'cam': args.cam}
    torch.save(meta, out_dir / 'meta.pt')
    print(f'\nsaved meta.pt: train={len(meta["train"])} val={len(meta["val"])} -> {out_dir}', flush=True)


if __name__ == '__main__':
    main()
