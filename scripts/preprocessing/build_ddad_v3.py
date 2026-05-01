"""V3 full-image cache for DDAD (TRI DGP format).

Schema matches build_pandaset_full_v3 / build_waymo_v3 / build_nuscenes_v3:
  jpg_bytes : raw JPEG bytes inline
  IH, IW    : ints
  pts       : (N, 3) float32 — lidar pts in WORLD frame
  cam_pos   : (3,)            — camera world position
  R_gt      : (3, 3)          — cam→world rotation
  T_gt      : (4, 4)          — world→cam (inv pose)
  K_full    : (3, 3)          — pinhole intrinsics
  cuboids   : []              — DDAD scene JSON has no 3D boxes by default
  scene, cam, frame           — provenance

DDAD is 10Hz native. --stride 10 → 1Hz.

Usage:
  python build_ddad_v3.py --max-scenes 5             # smoke
  python build_ddad_v3.py                            # all scenes, 1Hz, 6 cams
"""
import argparse, json, sys, time
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
import numpy as np
import torch
from PIL import Image
from scipy.spatial.transform import Rotation
from concurrent.futures import ProcessPoolExecutor, as_completed


def _pose_to_T(pose) -> np.ndarray:
    q = pose['rotation']; t = pose['translation']
    R = Rotation.from_quat([q['qx'], q['qy'], q['qz'], q['qw']]).as_matrix()
    M = np.eye(4); M[:3, :3] = R; M[:3, 3] = [t['x'], t['y'], t['z']]
    return M


def _process_scene(args_tuple):
    scene_dir, out_dir, cams_keep, stride, max_frames, gid_start = args_tuple
    scene_dir = Path(scene_dir)
    scene_name = scene_dir.name
    out_dir = Path(out_dir)
    inst_dir = out_dir / 'inst'
    inst_dir.mkdir(parents=True, exist_ok=True)

    scene_jsons = list(scene_dir.glob('scene_*.json'))
    if not scene_jsons:
        return scene_name, 0
    sc = json.loads(scene_jsons[0].read_text())

    calib_paths = list((scene_dir / 'calibration').glob('*.json'))
    if not calib_paths:
        return scene_name, 0
    calib_key = sc['samples'][0].get('calibration_key', '')
    cal_path = scene_dir / 'calibration' / f'{calib_key}.json'
    if not cal_path.exists(): cal_path = calib_paths[0]
    cal = json.loads(cal_path.read_text())

    # cam_name → K (3x3); skip LiDAR rows (all-zero intrinsics)
    K_per_cam = {}
    for i, name in enumerate(cal['names']):
        intr = cal['intrinsics'][i]
        if intr.get('fx', 0) == 0: continue
        K = np.array([[intr['fx'], 0, intr['cx']],
                      [0, intr['fy'], intr['cy']],
                      [0, 0, 1]], dtype=np.float32)
        K_per_cam[name] = K

    if cams_keep:
        cams = [c for c in K_per_cam.keys() if c in cams_keep]
    else:
        cams = list(K_per_cam.keys())
    if not cams: return scene_name, 0

    data_by_key = {d['key']: d for d in sc['data']}
    samples = sc['samples']
    if stride > 1: samples = samples[::stride]
    if max_frames: samples = samples[:max_frames]

    written = 0
    gid = gid_start
    for fi, sample in enumerate(samples):
        # Find lidar + cam datums for this sample
        lidar_d = None
        cam_d_by_name = {}
        for dk in sample['datum_keys']:
            d = data_by_key.get(dk)
            if d is None: continue
            sensor = d['id']['name']
            if sensor == 'LIDAR': lidar_d = d
            elif sensor in cams: cam_d_by_name[sensor] = d
        if lidar_d is None or not cam_d_by_name: continue

        # lidar in world
        lnpz_path = scene_dir / lidar_d['datum']['point_cloud']['filename']
        if not lnpz_path.exists(): continue
        try:
            lz = np.load(lnpz_path)
            arr = lz[list(lz.keys())[0]]
            pts_lid = arr[:, :3].astype(np.float32)
        except Exception:
            continue
        T_lid2w = _pose_to_T(lidar_d['datum']['point_cloud']['pose'])
        pts_world = (T_lid2w[:3, :3] @ pts_lid.T + T_lid2w[:3, 3:4]).T.astype(np.float32)

        for cam_name, cam_d in cam_d_by_name.items():
            jpg_path = scene_dir / cam_d['datum']['image']['filename']
            if not jpg_path.exists(): continue
            try:
                jpg_bytes = jpg_path.read_bytes()
                with Image.open(jpg_path) as _im:
                    IW, IH = _im.size
            except Exception:
                continue

            T_cam2w = _pose_to_T(cam_d['datum']['image']['pose'])
            T_gt = np.linalg.inv(T_cam2w).astype(np.float32)
            R_gt = T_cam2w[:3, :3].astype(np.float32)
            cam_pos = T_cam2w[:3, 3].astype(np.float32)
            K = K_per_cam[cam_name]

            # Filter visible
            homo = np.column_stack([pts_world, np.ones(len(pts_world))])
            pcam = (T_gt @ homo.T)[:3].T
            z = pcam[:, 2]
            uv = ((K @ pcam.T)[:2] / np.maximum(pcam[:, 2:].T, 1e-6)).T
            vis = (z > 0.5) & (uv[:, 0] >= 0) & (uv[:, 0] < IW) & (uv[:, 1] >= 0) & (uv[:, 1] < IH)
            if vis.sum() < 16: continue
            pts_vis = pts_world[vis]

            inst = dict(
                jpg_bytes = jpg_bytes,
                IH = int(IH), IW = int(IW),
                pts      = torch.from_numpy(pts_vis),
                cam_pos  = torch.from_numpy(cam_pos),
                R_gt     = torch.from_numpy(R_gt),
                T_gt     = torch.from_numpy(T_gt),
                K_full   = torch.from_numpy(K),
                cuboids  = [],
                scene = scene_name, cam = cam_name, frame = int(fi),
            )
            torch.save(inst, inst_dir / f'{gid:08d}.pt')
            gid += 1
            written += 1

    return scene_name, written


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--src',   default='/mnt/nvme6t/ddad_ext/ddad_train_val')
    ap.add_argument('--out',   default='/mnt/nvme6t/e2e_calib_cache/ddad_v3_full')
    ap.add_argument('--cams',  default='', help='comma list, empty=all (typically CAMERA_01..06 + 05/09)')
    ap.add_argument('--workers', type=int, default=4)
    ap.add_argument('--max-scenes', type=int, default=None)
    ap.add_argument('--max-frames', type=int, default=None)
    ap.add_argument('--stride',     type=int, default=10, help='10 → 1Hz from 10Hz')
    ap.add_argument('--val-frac',   type=float, default=0.15)
    ap.add_argument('--seed',       type=int, default=42)
    args = ap.parse_args()

    src = Path(args.src)
    if not src.exists():
        print(f'src not found: {src}', flush=True); sys.exit(1)
    out = Path(args.out); out.mkdir(parents=True, exist_ok=True)

    cams_keep = [c for c in args.cams.split(',') if c] if args.cams else None
    scenes = sorted(p for p in src.iterdir() if p.is_dir())
    if args.max_scenes: scenes = scenes[:args.max_scenes]
    print(f'scenes={len(scenes)}  cams={cams_keep or "ALL"}  stride={args.stride}  out={out}', flush=True)

    gid_stride = 4000
    argv = [(str(s), str(out), cams_keep, args.stride, args.max_frames, i * gid_stride)
            for i, s in enumerate(scenes)]

    t0 = time.time()
    written_total = 0
    with ProcessPoolExecutor(max_workers=args.workers) as ex:
        futs = {ex.submit(_process_scene, a): a[0] for a in argv}
        done = 0
        for fut in as_completed(futs):
            name, n = fut.result()
            written_total += n; done += 1
            print(f'[{done}/{len(scenes)}] {Path(name).name}: +{n}  total={written_total} '
                  f'({time.time()-t0:.0f}s)', flush=True)

    # train/val split (scene-level)
    fnames = sorted(p.name for p in (out / 'inst').glob('*.pt'))
    import random
    rng = random.Random(args.seed); rng.shuffle(scenes)
    n_val = max(1, int(len(scenes) * args.val_frac))
    val_scene_names = set(s.name for s in scenes[:n_val])

    train_files, val_files = [], []
    for f in fnames:
        inst = torch.load(out / 'inst' / f, weights_only=False)
        if inst.get('scene') in val_scene_names: val_files.append(f)
        else: train_files.append(f)
    torch.save({'train': train_files, 'val': val_files}, out / 'meta.pt')
    print(f'meta.pt: train={len(train_files)} val={len(val_files)}  → {out}', flush=True)


if __name__ == '__main__':
    main()
