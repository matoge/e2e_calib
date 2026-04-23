"""Visualize Waymo: only LiDAR points inside each 3D box, projected onto image."""
import sys, pathlib; sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[2]))
import io, math, numpy as np, pandas as pd
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from PIL import Image
from datasets.waymo import (WAYMO_DIR, FRONT_CAM, TOP_LIDAR, TARGET_TYPES,
                            _load_cam_cal, _load_lidar_cal,
                            _decode_range_image, _project, _bbox2d_of_box)

SEG    = None
FRAMES = [0, 30, 60, 100, 150]

segs = sorted(f.stem for f in (WAYMO_DIR / 'lidar').glob('*.parquet'))
seg  = SEG or segs[0]
print(f'Segment: {seg}')

fu, fv, cu, cv, IW, IH, T_veh_from_cam = _load_cam_cal(seg)
T_cam_from_veh = np.linalg.inv(T_veh_from_cam)
T_veh_from_lidar, incl, az_corr = _load_lidar_cal(seg)

lidar_df = pd.read_parquet(WAYMO_DIR / 'lidar'        / f'{seg}.parquet')
cam_df   = pd.read_parquet(WAYMO_DIR / 'camera_image' / f'{seg}.parquet')
box_df   = pd.read_parquet(WAYMO_DIR / 'lidar_box'    / f'{seg}.parquet')
lidar_top = lidar_df[lidar_df['key.laser_name']  == TOP_LIDAR]
cam_front = cam_df  [cam_df ['key.camera_name']  == FRONT_CAM]
timestamps = sorted(lidar_top['key.frame_timestamp_micros'].unique())

TYPE_COLOR = {1:'cyan', 2:'lime', 3:'orange', 4:'magenta'}
TYPE_NAME  = {1:'Vehicle', 2:'Pedestrian', 3:'Sign', 4:'Cyclist'}
EDGES = [(0,1),(1,2),(2,3),(3,0),(4,5),(5,6),(6,7),(7,4),(0,4),(1,5),(2,6),(3,7)]

def box_corners_veh(bx, by, bz, sx, sy, sz, heading):
    ch, sh = math.cos(heading), math.sin(heading)
    hx, hy, hz = sx/2, sy/2, sz/2
    corners = np.array([[ hx, hy, hz],[ hx, hy,-hz],[ hx,-hy, hz],[ hx,-hy,-hz],
                         [-hx, hy, hz],[-hx, hy,-hz],[-hx,-hy, hz],[-hx,-hy,-hz]], dtype=np.float32)
    R = np.array([[ch,-sh,0],[sh,ch,0],[0,0,1]], dtype=np.float32)
    return (R @ corners.T).T + np.array([bx,by,bz], dtype=np.float32)

def pts_in_box(pts_veh, bx, by, bz, sx, sy, sz, heading):
    ch, sh = math.cos(heading), math.sin(heading)
    R_obj = np.array([[ch, sh, 0],[-sh, ch, 0],[0, 0, 1]], dtype=np.float32)
    local = (R_obj @ (pts_veh - np.array([bx,by,bz], np.float32)).T).T
    hx, hy, hz = sx/2, sy/2, sz/2
    return ((np.abs(local[:,0]) <= hx) &
            (np.abs(local[:,1]) <= hy) &
            (np.abs(local[:,2]) <= hz))

for fi in FRAMES:
    ts = timestamps[min(fi, len(timestamps)-1)]

    lr = lidar_top[lidar_top['key.frame_timestamp_micros']==ts].iloc[0]
    pts_s = _decode_range_image(lr['[LiDARComponent].range_image_return1.values'],
                                 lr['[LiDARComponent].range_image_return1.shape'], incl, az_corr)
    pts_h   = np.hstack([pts_s, np.ones((len(pts_s),1), dtype=np.float32)])
    pts_veh = (T_veh_from_lidar @ pts_h.T).T[:,:3].astype(np.float32)

    cr = cam_front[cam_front['key.frame_timestamp_micros']==ts].iloc[0]
    img = Image.open(io.BytesIO(bytes(cr['[CameraImageComponent].image']))).convert('RGB')

    boxes = box_df[(box_df['key.frame_timestamp_micros']==ts) &
                   (box_df['[LiDARBoxComponent].type'].isin(TARGET_TYPES))]

    # ── full frame ────────────────────────────────────────────────────────────
    fig, ax = plt.subplots(figsize=(19, 11), dpi=100)
    ax.imshow(img)

    crops = []
    n_drawn = 0
    for _, b in boxes.iterrows():
        t    = int(b['[LiDARBoxComponent].type'])
        col  = TYPE_COLOR.get(t, 'white')
        bx   = float(b['[LiDARBoxComponent].box.center.x'])
        by   = float(b['[LiDARBoxComponent].box.center.y'])
        bz   = float(b['[LiDARBoxComponent].box.center.z'])
        sx   = float(b['[LiDARBoxComponent].box.size.x'])
        sy   = float(b['[LiDARBoxComponent].box.size.y'])
        sz   = float(b['[LiDARBoxComponent].box.size.z'])
        hdg  = float(b['[LiDARBoxComponent].box.heading'])

        mask = pts_in_box(pts_veh, bx, by, bz, sx, sy, sz, hdg)
        if mask.sum() == 0:
            continue
        uv_obj, depth_obj = _project(pts_veh[mask], T_cam_from_veh, fu, fv, cu, cv)
        vis = (depth_obj > 0.1) & (uv_obj[:,0]>=0) & (uv_obj[:,0]<IW) & \
                                   (uv_obj[:,1]>=0) & (uv_obj[:,1]<IH)
        if vis.sum() == 0:
            continue

        # scatter on full frame
        d_vis = depth_obj[vis]
        ax.scatter(uv_obj[vis,0], uv_obj[vis,1],
                   c=d_vis, cmap='jet_r',
                   vmin=d_vis.min(), vmax=d_vis.max(),
                   s=14, alpha=0.85, linewidths=0, zorder=3)
        n_drawn += 1

        # 3D box wireframe
        corners_veh = box_corners_veh(bx, by, bz, sx, sy, sz, hdg)
        uv_c, d_c   = _project(corners_veh, T_cam_from_veh, fu, fv, cu, cv)
        if (d_c <= 0).any():
            pass
        else:
            for i, j in EDGES:
                ax.plot([uv_c[i,0], uv_c[j,0]], [uv_c[i,1], uv_c[j,1]],
                        color=col, lw=1.5, alpha=0.85)

        # collect crop info
        u0c = uv_obj[vis,0].min(); u1c = uv_obj[vis,0].max()
        v0c = uv_obj[vis,1].min(); v1c = uv_obj[vis,1].max()
        pad = max(u1c-u0c, v1c-v0c) * 0.5 + 15
        u0 = max(0, u0c-pad); v0 = max(0, v0c-pad)
        u1 = min(IW, u1c+pad); v1 = min(IH, v1c+pad)
        crops.append(dict(
            img=img.crop((int(u0),int(v0),int(u1),int(v1))),
            uv=uv_obj[vis], dist=d_vis,
            u0=u0, v0=v0,
            label=TYPE_NAME.get(t,'?'),
            obj_dist=math.sqrt(bx*bx+by*by+bz*bz),
        ))

    ax.set_xlim(0, IW); ax.set_ylim(IH, 0); ax.axis('off')
    ax.set_title(f'frame={fi}  objects={n_drawn}  (LiDAR=inside 3D box, color=depth)', fontsize=12)
    plt.tight_layout(pad=0.2)
    out_full = f'vis_waymo_f{fi:03d}_full.png'
    plt.savefig(out_full, dpi=100, bbox_inches='tight')
    plt.close()
    print(f'  frame {fi}: {n_drawn} objects → {out_full}')

    # ── per-object crops ──────────────────────────────────────────────────────
    if not crops:
        continue
    nc = len(crops); cols_n = min(nc, 5); rows_n = (nc+cols_n-1)//cols_n
    fig2, axes = plt.subplots(rows_n, cols_n, figsize=(cols_n*4, rows_n*4), dpi=100)
    axes = np.array(axes).flatten() if nc > 1 else [axes]
    for i, cr2 in enumerate(crops):
        axi = axes[i]
        axi.imshow(cr2['img'])
        d = cr2['dist']
        axi.scatter(cr2['uv'][:,0]-cr2['u0'], cr2['uv'][:,1]-cr2['v0'],
                    c=d, cmap='jet_r', vmin=d.min(), vmax=d.max(),
                    s=30, alpha=0.9, linewidths=0, zorder=2)
        axi.set_title(f"{cr2['label']}  {cr2['obj_dist']:.0f}m  ({len(d)}pts)", fontsize=8)
        axi.axis('off')
    for axi in axes[nc:]:
        axi.axis('off')
    plt.suptitle(f'frame={fi}  LiDAR inside 3D box  color=depth(jet_r)', fontsize=10)
    plt.tight_layout(pad=0.4)
    out_crop = f'vis_waymo_f{fi:03d}_crops.png'
    fig2.savefig(out_crop, dpi=100, bbox_inches='tight')
    plt.close(fig2)
    print(f'           crops → {out_crop}')

print('Done.')
