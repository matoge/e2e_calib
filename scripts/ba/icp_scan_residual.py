"""Frame-to-frame ICP to measure ego-pose residual.

Given two consecutive LiDAR scans in world frame (using dataset-provided poses),
run trimmed-point-to-point ICP to find the small residual transform that best
aligns them. If pose is perfect, ICP correction should be ~0. If drift exists,
we see non-zero translation/rotation residual.

Compare across scenes:
- 015 (straight 27 km/h): expect small residuals
- 006 (left turn 12.8°/s): expect larger residuals if motion-comp imperfect
"""
import sys, pathlib; sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[2]))
import pickle, time
from pathlib import Path
import numpy as np
import json
from scipy.spatial import cKDTree
from scipy.spatial.transform import Rotation
import matplotlib; matplotlib.use('Agg')
import matplotlib.pyplot as plt

ROOT = Path('/mnt/mininas/datasets/pandaset')
OUT = Path('experiments/icp_residual'); OUT.mkdir(parents=True, exist_ok=True)


def voxel_downsample(pts, voxel=0.2):
    """Quick voxel grid: quantize + unique."""
    q = np.floor(pts / voxel).astype(np.int64)
    # combine 3 ints to one hash
    key = q[:, 0] * 73856093 ^ q[:, 1] * 19349663 ^ q[:, 2] * 83492791
    _, idx = np.unique(key, return_index=True)
    return pts[idx]


def umeyama(src, tgt, with_scale=False):
    """Kabsch/Umeyama rigid transform from src → tgt. Returns (R, t)."""
    mu_s = src.mean(0); mu_t = tgt.mean(0)
    S = src - mu_s; T = tgt - mu_t
    H = S.T @ T / len(S)
    U, _, Vt = np.linalg.svd(H)
    d = np.sign(np.linalg.det(Vt.T @ U.T))
    D = np.diag([1, 1, d])
    R = Vt.T @ D @ U.T
    t = mu_t - R @ mu_s
    return R, t


def icp_p2p(src, tgt, max_iter=20, max_corr_dist=1.0, trim_ratio=0.7):
    """Point-to-point ICP with trimmed correspondences.
    src, tgt: (N,3) pre-downsampled points in (roughly) same frame.
    Returns: final (R, t), fitness (trimmed mean residual), n_iter.
    """
    tree = cKDTree(tgt)
    R_acc = np.eye(3); t_acc = np.zeros(3)
    src_cur = src.copy()
    last_err = np.inf
    for it in range(max_iter):
        d, nn = tree.query(src_cur, k=1, distance_upper_bound=max_corr_dist)
        valid = np.isfinite(d)
        if valid.sum() < 50:
            break
        # trim: keep closest trim_ratio fraction
        d_valid = d[valid]
        thresh = np.partition(d_valid, int(len(d_valid)*trim_ratio))[int(len(d_valid)*trim_ratio)]
        keep = valid & (d <= thresh)
        if keep.sum() < 50:
            break
        src_pts = src_cur[keep]
        tgt_pts = tgt[nn[keep]]
        R, t = umeyama(src_pts, tgt_pts)
        src_cur = (R @ src_cur.T).T + t
        R_acc = R @ R_acc; t_acc = R @ t_acc + t
        err = np.mean(np.linalg.norm(src_pts - tgt_pts, axis=1))
        if abs(last_err - err) < 1e-4:
            return R_acc, t_acc, err, it + 1
        last_err = err
    return R_acc, t_acc, last_err, max_iter


def load_scan_world(scene, fi, voxel=0.3, max_range=60):
    """Load LiDAR in world frame, downsampled."""
    lidar = pickle.load(open(ROOT/scene/'lidar'/f'{fi:02d}.pkl','rb'))
    pts_w = lidar[lidar['d']==0][['x','y','z']].values.astype(np.float32)
    # filter by range from ego (pose position)
    poses = json.load(open(ROOT/scene/'camera'/'front_camera'/'poses.json'))
    cam = np.array([poses[fi]['position'][k] for k in 'xyz'])
    d = np.linalg.norm(pts_w - cam, axis=1)
    pts_w = pts_w[(d < max_range) & (d > 2)]
    pts_w = voxel_downsample(pts_w, voxel)
    return pts_w


def run_scene(scene, frames, voxel=0.3):
    """Run ICP between consecutive frames, return per-transition residuals."""
    poses = json.load(open(ROOT/scene/'camera'/'front_camera'/'poses.json'))
    out = []
    t0 = time.time()
    scans = {}
    for fi in frames:
        scans[fi] = load_scan_world(scene, fi, voxel)
    print(f'{scene}: loaded {len(scans)} scans, {sum(len(s) for s in scans.values())//1000}k pts ({time.time()-t0:.0f}s)')

    for fa, fb in zip(frames[:-1], frames[1:]):
        src, tgt = scans[fa], scans[fb]
        R, t, err, ni = icp_p2p(src, tgt, max_iter=25, max_corr_dist=1.5, trim_ratio=0.6)
        rotvec = Rotation.from_matrix(R).as_rotvec()
        dyaw = np.rad2deg(rotvec[2])  # z-axis rotation = yaw (world frame)
        t_mag = np.linalg.norm(t) * 100  # cm
        # also compute expected pose-based delta
        pa = np.array([poses[fa]['position'][k] for k in 'xyz'])
        pb = np.array([poses[fb]['position'][k] for k in 'xyz'])
        ego_dt = np.linalg.norm(pb - pa)  # m traveled between frames
        qa = Rotation.from_quat([poses[fa]['heading'][k] for k in 'xyzw'])
        qb = Rotation.from_quat([poses[fb]['heading'][k] for k in 'xyzw'])
        yaw_rate = np.rad2deg((qb*qa.inv()).as_rotvec()[2]) / 0.1
        out.append((fa, fb, err, t_mag, dyaw, ego_dt, yaw_rate, ni))
    return out


SCENES = [
    ('015', list(range(10, 40)), 'straight 27 km/h'),
    ('006', list(range(0, 30)),  'LEFT TURN 12.8°/s peak'),
    ('021', list(range(10, 40)), 'fast straight 52 km/h'),
]

fig, axes = plt.subplots(3, 2, figsize=(16, 10), dpi=110)
results = {}
for row, (scene, frames, label) in enumerate(SCENES):
    t0 = time.time()
    res = run_scene(scene, frames)
    results[scene] = res
    r = np.array([(a, err, tm, dy, ed, yr) for a,b,err,tm,dy,ed,yr,_ in res])
    # translation residual (cm) and yaw residual (deg) vs frame index
    ax = axes[row, 0]
    ax.plot(r[:,0], r[:,2], 'o-', color='C0', label='ICP |Δt| (cm)', lw=1.5)
    ax.set_ylabel('|Δt| (cm)', color='C0'); ax.tick_params(axis='y', labelcolor='C0')
    ax2 = ax.twinx()
    ax2.plot(r[:,0], r[:,3], 's-', color='C3', label='ICP Δyaw (deg)', lw=1.5)
    ax2.set_ylabel('Δyaw (deg)', color='C3'); ax2.tick_params(axis='y', labelcolor='C3')
    ax.set_xlabel('frame'); ax.set_title(f'{scene} — {label}  ICP residual per transition',
                                          fontsize=11, fontweight='bold')
    ax.grid(alpha=0.3)
    # ego yaw rate for context
    ax = axes[row, 1]
    ax.plot(r[:,0], r[:,5], 'o-', color='gray', label='pose yaw rate (deg/s)')
    ax.axhline(0, color='k', lw=0.5)
    ax.set_xlabel('frame'); ax.set_ylabel('ego yaw rate (°/s)')
    ax.set_title(f'{scene} ego yaw rate (for reference)')
    ax.grid(alpha=0.3)
    ax.legend()

    print(f'{scene} summary: median |Δt|={np.median(r[:,2]):.1f} cm  '
           f'median |Δyaw|={np.median(np.abs(r[:,3])):.3f}°  '
           f'max |Δt|={r[:,2].max():.1f} cm  max |Δyaw|={np.abs(r[:,3]).max():.3f}°')

fig.suptitle('ICP-measured residual pose between consecutive LiDAR scans '
             '(after applying dataset-provided poses).\n'
             'Large residual = dataset pose has drift OR motion-comp imperfect.',
             fontsize=12, fontweight='bold')
plt.tight_layout(rect=[0,0,1,0.96])
out = OUT / 'icp_residual_by_frame.png'
plt.savefig(out, dpi=110, bbox_inches='tight')
plt.close(fig)
print(f'\nwrote {out}')
