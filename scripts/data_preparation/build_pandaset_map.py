"""Accumulate full-scene PandaSet LiDAR into a single world-frame point cloud.
Save as .ply (loadable in Babylon.js, Potree, CloudCompare, MeshLab etc.),
and render BEV + side views."""
import sys, pathlib; sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[2]))
import json, pickle, time
from pathlib import Path
import numpy as np
from scipy.spatial.transform import Rotation
import matplotlib; matplotlib.use('Agg')
import matplotlib.pyplot as plt

ROOT = Path('/mnt/mininas/datasets/pandaset')
OUT = Path('experiments/maps'); OUT.mkdir(parents=True, exist_ok=True)


def voxel_downsample_with_color(pts, colors, voxel=0.1):
    q = np.floor(pts / voxel).astype(np.int64)
    key = q[:, 0]*73856093 ^ q[:, 1]*19349663 ^ q[:, 2]*83492791
    _, idx = np.unique(key, return_index=True)
    return pts[idx], colors[idx]


def build_map(scene, stride=1, voxel=0.15, max_range=70):
    """Accumulate all frames of a scene into world-frame point cloud."""
    poses = json.load(open(ROOT/scene/'camera'/'front_camera'/'poses.json'))
    all_pts = []
    all_col = []  # color by source frame index
    all_dev = []
    all_int = []
    n_frames = len(list((ROOT/scene/'lidar').glob('*.pkl')))
    frames = list(range(0, n_frames, stride))
    t0 = time.time()
    for fi in frames:
        lidar = pickle.load(open(ROOT/scene/'lidar'/f'{fi:02d}.pkl','rb'))
        pts_w = lidar[['x','y','z']].values.astype(np.float32)
        dev = lidar['d'].values.astype(np.uint8)
        # intensity col name varies; try 'i' or 'intensity'
        if 'i' in lidar.columns:
            intens = lidar['i'].values.astype(np.float32)
        else:
            intens = np.zeros(len(pts_w), dtype=np.float32)
        # filter by range from ego
        cam = np.array([poses[fi]['position'][k] for k in 'xyz'], dtype=np.float32)
        d = np.linalg.norm(pts_w - cam, axis=1)
        m = (d < max_range) & (d > 2)
        pts_w, dev, intens = pts_w[m], dev[m], intens[m]
        all_pts.append(pts_w)
        all_col.append(np.full(len(pts_w), fi, dtype=np.int32))
        all_dev.append(dev)
        all_int.append(intens)
    pts = np.concatenate(all_pts)
    col = np.concatenate(all_col)
    dev = np.concatenate(all_dev)
    intens = np.concatenate(all_int)
    print(f'{scene}: {len(pts)//1000}k pts raw, {len(frames)} frames ({time.time()-t0:.0f}s)')
    # voxel downsample keeping nearest point per voxel (retains color)
    pts_ds, extras_ds = voxel_downsample_with_color(
        pts, np.stack([col.astype(np.float32), dev.astype(np.float32), intens], axis=1), voxel)
    col_ds = extras_ds[:, 0].astype(np.int32)
    dev_ds = extras_ds[:, 1].astype(np.uint8)
    int_ds = extras_ds[:, 2]
    print(f'  → {len(pts_ds)//1000}k after voxel {voxel} m')
    return pts_ds, col_ds, dev_ds, int_ds, frames


def save_ply(path, pts, rgb):
    """Simple binary-little-endian PLY writer."""
    header = (
        'ply\nformat binary_little_endian 1.0\n'
        f'element vertex {len(pts)}\n'
        'property float x\nproperty float y\nproperty float z\n'
        'property uchar red\nproperty uchar green\nproperty uchar blue\n'
        'end_header\n'
    )
    arr = np.empty(len(pts), dtype=[('x','f4'),('y','f4'),('z','f4'),
                                      ('r','u1'),('g','u1'),('b','u1')])
    arr['x'], arr['y'], arr['z'] = pts[:,0], pts[:,1], pts[:,2]
    arr['r'], arr['g'], arr['b'] = rgb[:,0], rgb[:,1], rgb[:,2]
    with open(path, 'wb') as f:
        f.write(header.encode())
        f.write(arr.tobytes())
    print(f'  wrote {path}  ({path.stat().st_size/1024/1024:.1f} MB)')


def colorize_by_frame(frame_idx, n_frames, cmap='plasma'):
    import matplotlib.cm as cm
    norm = (frame_idx - 0) / max(n_frames - 1, 1)
    return (cm.get_cmap(cmap)(norm)[:, :3] * 255).astype(np.uint8)


def colorize_by_height(z, vmin=None, vmax=None, cmap='turbo'):
    import matplotlib.cm as cm
    if vmin is None: vmin = np.percentile(z, 2)
    if vmax is None: vmax = np.percentile(z, 98)
    norm = np.clip((z - vmin) / (vmax - vmin + 1e-9), 0, 1)
    return (cm.get_cmap(cmap)(norm)[:, :3] * 255).astype(np.uint8)


PICKS = [
    ('015', 'straight 27 km/h'),
    ('006', 'left-turn 12.8°/s peak'),
    ('021', 'fast straight 52 km/h'),
]

for scene, label in PICKS:
    pts, frame_col, dev, intens, frames = build_map(scene, stride=1, voxel=0.15)

    # save two flavors: color by frame (time), color by height
    rgb_frame = colorize_by_frame(frame_col, len(frames))
    rgb_height = colorize_by_height(pts[:, 2])
    save_ply(OUT / f'{scene}_map_by_frame.ply', pts, rgb_frame)
    save_ply(OUT / f'{scene}_map_by_height.ply', pts, rgb_height)

    # BEV + side
    fig, axes = plt.subplots(1, 2, figsize=(18, 8), dpi=100)
    # BEV: x vs y, color by height
    ax = axes[0]
    z_norm = np.clip((pts[:,2] - np.percentile(pts[:,2],2)) /
                     (np.percentile(pts[:,2],98) - np.percentile(pts[:,2],2) + 1e-9), 0, 1)
    # downsample for plotting
    n_plot = min(200_000, len(pts))
    ix = np.random.default_rng(0).permutation(len(pts))[:n_plot]
    ax.scatter(pts[ix,0], pts[ix,1], c=z_norm[ix], cmap='turbo', s=0.3, alpha=0.6, edgecolors='none')
    # plot ego trajectory
    poses = json.load(open(ROOT/scene/'camera'/'front_camera'/'poses.json'))
    traj = np.array([[p['position']['x'], p['position']['y']] for p in poses])
    ax.plot(traj[:,0], traj[:,1], 'w-', lw=2, alpha=0.9, label='ego trajectory')
    ax.plot(traj[0,0], traj[0,1], 'go', ms=10, label='start')
    ax.plot(traj[-1,0], traj[-1,1], 'rs', ms=10, label='end')
    ax.set_aspect('equal'); ax.set_title(f'{scene} BEV — {label}', fontsize=12, fontweight='bold')
    ax.set_xlabel('world X (m)'); ax.set_ylabel('world Y (m)')
    ax.legend(); ax.grid(alpha=0.3)

    # side view: x vs z (along travel)
    ax = axes[1]
    # project onto direction of ego travel
    dxy = traj[-1] - traj[0]; hdg = np.arctan2(dxy[1], dxy[0])
    c, s = np.cos(-hdg), np.sin(-hdg)
    rot = np.array([[c, -s], [s, c]])
    pts_r = pts[:, :2] @ rot.T
    traj_r = traj @ rot.T
    ax.scatter(pts_r[ix,0], pts[ix,2], c=z_norm[ix], cmap='turbo', s=0.3, alpha=0.6, edgecolors='none')
    ax.plot(traj_r[:,0], np.full(len(traj_r), pts[:,2].min()+0.5), 'w-', lw=2)
    ax.set_title(f'{scene} side (along-travel) x-z'); ax.set_xlabel('travel (m)'); ax.set_ylabel('z (m)')
    ax.grid(alpha=0.3)

    fig.suptitle(f'PandaSet {scene} accumulated map  |  {len(pts)//1000}k pts  |  voxel 0.15m',
                  fontsize=13, fontweight='bold')
    plt.tight_layout()
    out = OUT / f'{scene}_map_views.png'
    plt.savefig(out, dpi=100, bbox_inches='tight')
    plt.close(fig)
    print(f'  wrote {out}')
