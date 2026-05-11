"""Visualize ZOD frames from curated list — high-res, single-frame PNGs.

Renders lidar projection on the front DNAT camera with depth-coded scatter
overlay, one PNG per frame at the native 3848×2168 image resolution.

⚠ DOES NOT call `build_zod_v3._ego_motion_apply`. That function's per-point
motion-compensation pipeline produces VISIBLY MISALIGNED projections (5-20 px
drift, easily seen on pole/sign edges). The cause is suspected: either
- `pt_ts` interpretation (units? abs vs relative to cam shutter?)
- `_camera_shutter_ts` returning wrong base time
- ego_motion.json timestamp frame (Unix vs GPS) mismatch with pt_ts

Until that's debugged, the scan-wise (static T_vl + T_cv only, no MC) projection
is the correct reference. The 100ms scan span causes minor drift on edges of
the FOV but the front-cam DNAT field is narrow enough that scanwise is clean.

The reproduced V3 cache `zod_v3_tiled` was built with the broken MC and may
contain misaligned uv_full labels — re-build needed after MC is fixed.

Usage:
    python scripts/visualization/vis_zod_curated.py \\
        --curated /mnt/nvme6t/zod/frames/curated_v2_y1_a2_s10_80.txt \\
        --out experiments/_cache_check/zod_curated/hires_v3 \\
        --n 30 --max-id 30000
"""
import argparse, json, random
import numpy as np
from pathlib import Path
from PIL import Image
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt


def render_frame(frame_dir: Path, out_path: Path, speed: float, yaw: float, accel: float):
    """Scanwise (NO MC) projection + KB fisheye → hi-res PNG with overlay text."""
    calib = json.loads((frame_dir / 'calibration.json').read_text())['FC']
    K = np.asarray(calib['intrinsics'])[:3, :3]
    dist = np.asarray(calib['distortion'])
    T_vc = np.asarray(calib['extrinsics'])         # camera-to-vehicle
    T_vl = np.asarray(calib['lidar_extrinsics'])   # lidar-to-vehicle
    IW, IH = int(calib['image_dimensions'][0]), int(calib['image_dimensions'][1])

    lidars = sorted((frame_dir / 'lidar_velodyne').glob('*.npy'))
    if not lidars:
        return False
    # core sweep at the camera shutter time
    sweep = np.load(lidars[len(lidars) // 2], allow_pickle=False)
    pts_l = np.stack([sweep['x'], sweep['y'], sweep['z']], axis=1)
    homo = np.column_stack([pts_l, np.ones(len(pts_l))])
    pts_v = (T_vl @ homo.T).T[:, :3]
    T_cv = np.linalg.inv(T_vc)
    pts_c = (T_cv[:3, :3] @ pts_v.T).T + T_cv[:3, 3]

    # Kannala-Brandt forward projection
    x, y, z = pts_c[:, 0], pts_c[:, 1], pts_c[:, 2]
    r = np.sqrt(x * x + y * y)
    theta = np.arctan2(r, np.maximum(z, 1e-6))
    k1, k2, k3, k4 = dist
    t2 = theta * theta
    td = theta * (1 + k1 * t2 + k2 * t2 ** 2 + k3 * t2 ** 3 + k4 * t2 ** 4)
    rs = np.where(r > 1e-9, r, 1.0)
    u = K[0, 0] * (td * x / rs) + K[0, 2]
    v = K[1, 1] * (td * y / rs) + K[1, 2]
    valid = (z > 0.5) & (u >= 0) & (u < IW) & (v >= 0) & (v < IH)

    img_jpg = sorted((frame_dir / 'camera_front_dnat').glob('*.jpg'))[0]
    img = np.asarray(Image.open(img_jpg).convert('RGB'))

    DPI = 150
    fig, ax = plt.subplots(figsize=(IW / DPI, IH / DPI), dpi=DPI)
    ax.imshow(img)
    ax.scatter(u[valid], v[valid], c=z[valid], s=4, cmap='turbo',
               vmin=2, vmax=60, alpha=0.55, marker='.', linewidths=0)
    ax.text(20, IH - 30,
            f'{frame_dir.name}  yaw={yaw:.2f}°/s  speed={speed:.1f}km/h  accel={accel:.2f}m/s²',
            fontsize=20, color='white',
            bbox=dict(facecolor='black', alpha=0.7, edgecolor='none', pad=4))
    ax.set_xlim(0, IW); ax.set_ylim(IH, 0); ax.axis('off')
    ax.set_position([0, 0, 1, 1])
    plt.savefig(out_path, dpi=DPI, bbox_inches='tight', pad_inches=0)
    plt.close(fig)
    return True


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--zod-root', default='/mnt/nvme6t/zod/frames/single_frames')
    ap.add_argument('--curated', default='/mnt/nvme6t/zod/frames/curated_v2_y1_a2_s10_80.txt',
                    help='TSV produced by curate_zod_frames.py: <fid>\\t<sp>\\t<yaw>\\t<accel>')
    ap.add_argument('--out', default='experiments/_cache_check/zod_curated/hires_v3')
    ap.add_argument('--n', type=int, default=30)
    ap.add_argument('--max-id', type=int, default=30000,
                    help='only keep curated frames with id<max_id (lidar tarball coverage)')
    ap.add_argument('--seed', type=int, default=7)
    args = ap.parse_args()

    root = Path(args.zod_root)
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    random.seed(args.seed)
    rows = []
    with open(args.curated) as f:
        for line in f:
            parts = line.strip().split('\t')
            if len(parts) >= 4 and int(parts[0]) < args.max_id:
                rows.append((parts[0], float(parts[1]), float(parts[2]), float(parts[3])))
    print(f'curated with-lidar pool: {len(rows)}  sampling: {args.n}')
    samples = random.sample(rows, args.n)

    ok = 0
    for i, (fid, sp, yaw, ac) in enumerate(samples):
        if render_frame(root / fid, out_dir / f'zod_{i:02d}_{fid}.png', sp, yaw, ac):
            ok += 1
    print(f'rendered {ok}/{args.n} → {out_dir}')


if __name__ == '__main__':
    main()
