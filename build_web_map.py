"""Dump PandaSet scene accumulation as compact binary + metadata JSON
for Babylon.js web viewer.

Output:
  web_viewer/data/{scene}.bin   binary: N × (x,y,z : f32) then N × (color_idx : u16)
                                 color_idx encodes frame index in [0, n_frames)
  web_viewer/data/{scene}.json  metadata: counts, trajectory, center, bbox, n_frames
"""
import json, pickle, time
from pathlib import Path
import numpy as np

ROOT = Path('/mnt/mininas/datasets/pandaset')
OUT = Path('web_viewer/data'); OUT.mkdir(parents=True, exist_ok=True)


def voxel_downsample(pts, extras, voxel=0.15):
    q = np.floor(pts / voxel).astype(np.int64)
    key = q[:, 0]*73856093 ^ q[:, 1]*19349663 ^ q[:, 2]*83492791
    _, idx = np.unique(key, return_index=True)
    return pts[idx], extras[idx]


def build(scene, stride=1, voxel=0.15, max_range=70):
    poses = json.load(open(ROOT/scene/'camera'/'front_camera'/'poses.json'))
    intr = json.load(open(ROOT/scene/'camera'/'front_camera'/'intrinsics.json'))
    n_frames = len(list((ROOT/scene/'lidar').glob('*.pkl')))
    frames = list(range(0, n_frames, stride))

    all_pts = []
    all_frame = []
    t0 = time.time()
    for fi in frames:
        lidar = pickle.load(open(ROOT/scene/'lidar'/f'{fi:02d}.pkl','rb'))
        pts = lidar[['x','y','z']].values.astype(np.float32)
        cam = np.array([poses[fi]['position'][k] for k in 'xyz'], dtype=np.float32)
        d = np.linalg.norm(pts - cam, axis=1)
        m = (d < max_range) & (d > 2)
        pts = pts[m]
        all_pts.append(pts)
        all_frame.append(np.full(len(pts), fi, dtype=np.int32))
    pts = np.concatenate(all_pts)
    frame = np.concatenate(all_frame).astype(np.float32)[:, None]
    pts, extras = voxel_downsample(pts, frame, voxel)
    frame = extras[:, 0].astype(np.uint16)
    print(f'{scene}: {len(pts)//1000}k pts, {len(frames)} frames ({time.time()-t0:.0f}s)')

    # center pointcloud to origin for viewer convenience
    center = pts.mean(0)
    pts_c = pts - center
    bbox_min = pts_c.min(0); bbox_max = pts_c.max(0)

    # ego trajectory
    traj = np.array([[p['position'][k] for k in 'xyz'] for p in poses], dtype=np.float32)
    traj_c = traj - center

    # write binary: xyz (Nx3 f32) || frame (N u16)
    bin_path = OUT / f'{scene}.bin'
    with open(bin_path, 'wb') as f:
        f.write(pts_c.astype(np.float32).tobytes())
        f.write(frame.astype(np.uint16).tobytes())

    meta = {
        'scene': scene,
        'n_points': int(len(pts_c)),
        'n_frames': int(len(frames)),
        'frames': [int(fi) for fi in frames],
        'voxel': float(voxel),
        'center_world': [float(x) for x in center],
        'bbox_min': [float(x) for x in bbox_min],
        'bbox_max': [float(x) for x in bbox_max],
        'trajectory': [[float(x) for x in p] for p in traj_c],
        'intrinsics': intr,
        'layout': {
            'positions': {'type': 'f32', 'offset': 0, 'count': len(pts_c), 'stride': 12},
            'frame': {'type': 'u16', 'offset': len(pts_c)*12, 'count': len(pts_c), 'stride': 2},
        },
    }
    meta_path = OUT / f'{scene}.json'
    meta_path.write_text(json.dumps(meta, indent=None))
    print(f'  wrote {bin_path} ({bin_path.stat().st_size/1024/1024:.1f} MB)  +  {meta_path}')


for s in ['015', '006', '021']:
    build(s, stride=1, voxel=0.2)

# manifest of available scenes
scenes = []
for s in ['015', '006', '021']:
    m = json.loads((OUT/f'{s}.json').read_text())
    scenes.append({'scene': s, 'n_points': m['n_points'], 'n_frames': m['n_frames']})
(OUT/'manifest.json').write_text(json.dumps({'scenes': scenes}, indent=2))
print('wrote manifest.json')
