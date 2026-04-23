"""Waymo full-image sanity: CAM_FRONT + TOP LiDAR projection + 3D boxes (strict FOV)."""
import sys, pathlib; sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[2]))
import io, numpy as np, pandas as pd, math
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from PIL import Image
from datasets.waymo import (WAYMO_DIR, FRONT_CAM, TOP_LIDAR,
                           _load_cam_cal, _load_lidar_cal,
                           _decode_range_image, _project)

FRAMES = [0, 40, 100]

segs = sorted(f.stem for f in (WAYMO_DIR/'lidar').glob('*.parquet'))
seg  = segs[0]
print(f'segment: {seg}')

fu, fv, cu, cv, IW, IH, T_veh_from_cam = _load_cam_cal(seg)
T_cam_from_veh = np.linalg.inv(T_veh_from_cam)
T_veh_from_lidar, incl, az_corr = _load_lidar_cal(seg)

lidar_df = pd.read_parquet(WAYMO_DIR/'lidar'/f'{seg}.parquet')
cam_df   = pd.read_parquet(WAYMO_DIR/'camera_image'/f'{seg}.parquet')
box_df   = pd.read_parquet(WAYMO_DIR/'lidar_box'/f'{seg}.parquet')
lidar_top = lidar_df[lidar_df['key.laser_name'] == TOP_LIDAR]
cam_front = cam_df[cam_df['key.camera_name'] == FRONT_CAM]
timestamps = sorted(lidar_top['key.frame_timestamp_micros'].unique())

TYPE_COLOR = {1:'yellow', 2:'cyan', 3:'orange', 4:'magenta'}
TYPE_NAME  = {1:'Vehicle', 2:'Pedestrian', 3:'Sign', 4:'Cyclist'}
EDGES = [(0,1),(1,2),(2,3),(3,0),(4,5),(5,6),(6,7),(7,4),(0,4),(1,5),(2,6),(3,7)]

def box_corners_veh(bx, by, bz, sx, sy, sz, heading):
    ch, sh = math.cos(heading), math.sin(heading)
    hx, hy, hz = sx/2, sy/2, sz/2
    # ordering: 0-3 top face CCW, 4-7 bottom face CCW (matches EDGES)
    c = np.array([[ hx, hy, hz],[ hx,-hy, hz],[-hx,-hy, hz],[-hx, hy, hz],
                   [ hx, hy,-hz],[ hx,-hy,-hz],[-hx,-hy,-hz],[-hx, hy,-hz]], dtype=np.float32)
    R = np.array([[ch,-sh,0],[sh,ch,0],[0,0,1]], dtype=np.float32)
    return (R @ c.T).T + np.array([bx,by,bz], dtype=np.float32)


fig, axes = plt.subplots(len(FRAMES), 1, figsize=(20, 11*len(FRAMES)), dpi=70)
if len(FRAMES) == 1: axes = [axes]

for ax, fi in zip(axes, FRAMES):
    ts = timestamps[min(fi, len(timestamps)-1)]

    lr = lidar_top[lidar_top['key.frame_timestamp_micros']==ts].iloc[0]
    pts_s = _decode_range_image(lr['[LiDARComponent].range_image_return1.values'],
                                 lr['[LiDARComponent].range_image_return1.shape'],
                                 incl, az_corr)
    pts_h   = np.hstack([pts_s, np.ones((len(pts_s),1), dtype=np.float32)])
    pts_veh = (T_veh_from_lidar @ pts_h.T).T[:,:3].astype(np.float32)

    cr = cam_front[cam_front['key.frame_timestamp_micros']==ts].iloc[0]
    img = Image.open(io.BytesIO(bytes(cr['[CameraImageComponent].image']))).convert('RGB')

    uv, depth = _project(pts_veh, T_cam_from_veh, fu, fv, cu, cv)
    vis = (depth > 0.5) & (uv[:,0]>=0) & (uv[:,0]<IW) & (uv[:,1]>=0) & (uv[:,1]<IH)

    ax.imshow(img)
    sc = ax.scatter(uv[vis,0], uv[vis,1], c=depth[vis], cmap='turbo',
                    s=2, alpha=0.7, vmin=0, vmax=80)

    boxes = box_df[box_df['key.frame_timestamp_micros']==ts]
    counts = {}
    for _, b in boxes.iterrows():
        t = int(b['[LiDARBoxComponent].type'])
        bx=float(b['[LiDARBoxComponent].box.center.x']); by=float(b['[LiDARBoxComponent].box.center.y']); bz=float(b['[LiDARBoxComponent].box.center.z'])
        sx=float(b['[LiDARBoxComponent].box.size.x']);   sy=float(b['[LiDARBoxComponent].box.size.y']);   sz=float(b['[LiDARBoxComponent].box.size.z'])
        hdg=float(b['[LiDARBoxComponent].box.heading'])
        corners_veh = box_corners_veh(bx, by, bz, sx, sy, sz, hdg)
        uv_c, d_c = _project(corners_veh, T_cam_from_veh, fu, fv, cu, cv)
        if (d_c <= 0.5).any(): continue
        in_img = ((uv_c[:,0]>=0) & (uv_c[:,0]<IW) & (uv_c[:,1]>=0) & (uv_c[:,1]<IH))
        if in_img.sum() < 4: continue
        col = TYPE_COLOR.get(t, 'white')
        for i,j in EDGES:
            ax.plot([uv_c[i,0],uv_c[j,0]], [uv_c[i,1],uv_c[j,1]],
                    color=col, lw=1.5, alpha=0.9)
        counts[t] = counts.get(t,0) + 1

    label = " ".join(f"{TYPE_NAME.get(k,'?')}={v}" for k,v in counts.items())
    ax.set_xlim(0, IW); ax.set_ylim(IH, 0); ax.axis('off')
    ax.set_title(f'frame={fi}  lidar={int(vis.sum())}/{len(pts_veh)}  {label}', fontsize=13)

cbar = fig.colorbar(sc, ax=axes, shrink=0.5, label='depth (m)')
out = 'vis_waymo_full_proj.png'
plt.savefig(out, dpi=70, bbox_inches='tight')
print(f"saved → {out}")
