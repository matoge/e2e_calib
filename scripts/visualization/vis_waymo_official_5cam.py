"""Waymo OFFICIAL projection — uses datasets.waymo_lcp helpers.

Loads Waymo's precomputed lidar_camera_projection (LCP) and renders per-cam
overlays at high resolution. Color = lidar range (m).
"""
import sys, pathlib, argparse, io
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[2]))
import numpy as np, pandas as pd, pyarrow.parquet as pq
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from PIL import Image

from datasets.waymo import WAYMO_DIR
from datasets.waymo_lcp import CAM_NAMES, ALL_LASERS, project_frame

ap = argparse.ArgumentParser()
ap.add_argument('--seg-idx', type=int, default=0)
ap.add_argument('--frame', type=int, default=20)
ap.add_argument('--lcp-root', default='/mnt/nvme6t/waymo_lcp')
ap.add_argument('--all-lasers', action='store_true', help='use all 5 lidars (default: TOP only)')
ap.add_argument('--max-d', type=float, default=60.0)
ap.add_argument('--dpi', type=int, default=180)
ap.add_argument('--out', default='vis_waymo_official_5cam.png')
args = ap.parse_args()

LASERS = ALL_LASERS if args.all_lasers else (1,)
MAX_D  = args.max_d

segs = sorted(f.stem for f in (WAYMO_DIR/'lidar').glob('*.parquet'))
seg  = segs[args.seg_idx]
print(f'segment[{args.seg_idx}]: {seg}', flush=True)

# pick frame ts (use TOP_LIDAR's frame ts as canonical)
ts_list = sorted(pq.read_table(WAYMO_DIR/'lidar'/f'{seg}.parquet',
                               columns=['key.frame_timestamp_micros'],
                               filters=[('key.laser_name', '=', 1)]
                               ).to_pandas()['key.frame_timestamp_micros'].unique())
ts_lidar = int(ts_list[min(args.frame, len(ts_list)-1)])
print(f'ts_lidar={ts_lidar}', flush=True)

per_cam = project_frame(seg, ts_lidar,
                         waymo_root=WAYMO_DIR,
                         lcp_root=args.lcp_root,
                         lasers=LASERS)

# load cam images at this frame
print(f'reading camera_image...', flush=True)
cam_df = pq.read_table(WAYMO_DIR/'camera_image'/f'{seg}.parquet',
                       filters=[('key.frame_timestamp_micros', '=', ts_lidar)]).to_pandas()

fig, axes = plt.subplots(1, 5, figsize=(40, 8))
for ax, (cid, cname) in zip(axes, CAM_NAMES.items()):
    rows = cam_df[cam_df['key.camera_name']==cid]
    if len(rows)==0:
        ax.set_title(f'{cname}: no row'); ax.axis('off'); continue
    img = Image.open(io.BytesIO(bytes(rows.iloc[0]['[CameraImageComponent].image']))).convert('RGB')
    ax.imshow(img)
    pcm = per_cam[cid]
    if len(pcm['uv']):
        ax.scatter(pcm['uv'][:,0], pcm['uv'][:,1],
                   c=pcm['depth'].clip(0, MAX_D), cmap='turbo',
                   s=0.5, alpha=0.25, vmin=0, vmax=MAX_D)
    ax.set_title(f'{cname}  {len(pcm["uv"])} pts', fontsize=10)
    ax.axis('off')

fig.suptitle(f'Waymo OFFICIAL projection (lidar_camera_projection)  seg={seg[:30]}…  frame={args.frame}', fontsize=11)
plt.tight_layout()
plt.savefig(args.out, dpi=args.dpi, bbox_inches='tight')
print(f'saved → {args.out}')
