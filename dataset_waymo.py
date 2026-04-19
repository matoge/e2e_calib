"""WaymoCalibDataset — camera-LiDAR calibration dataset from Waymo Open Dataset v2.

Coordinate conventions:
  - Vehicle frame: X=forward, Y=left, Z=up
  - Camera frame (Waymo):  X=forward (depth), Y=left, Z=up  (non-standard!)
    Projection: u = fu * (-cy) / cx + cu,  v = fv * (-cz) / cx + cv
  - Extrinsics stored as T_vehicle_from_sensor  (both camera and lidar)
  - Lidar boxes in vehicle frame

Target object types used: 1=Vehicle, 2=Pedestrian  (skip 3=Sign, 4=Cyclist rare)
"""
import io, os, random, math
import numpy as np
import pandas as pd
from pathlib import Path
from PIL import Image
import torch
from torch.utils.data import Dataset
from scipy.spatial.transform import Rotation

WAYMO_DIR   = Path('/mnt/nvme6t/waymo/training')
TARGET_TYPES = {1, 2}      # Vehicle, Pedestrian
FRONT_CAM    = 1
TOP_LIDAR    = 1
IMG_SIZE     = 64
BBOX_SCALE   = 2.0
MIN_PTS      = 4

# ── helpers ─────────────────────────────────────────────────────────────────

def _load_cam_cal(seg_name):
    df = pd.read_parquet(WAYMO_DIR / 'camera_calibration' / f'{seg_name}.parquet')
    row = df[df['key.camera_name'] == FRONT_CAM].iloc[0]
    fu = float(row['[CameraCalibrationComponent].intrinsic.f_u'])
    fv = float(row['[CameraCalibrationComponent].intrinsic.f_v'])
    cu = float(row['[CameraCalibrationComponent].intrinsic.c_u'])
    cv = float(row['[CameraCalibrationComponent].intrinsic.c_v'])
    IW = int(row['[CameraCalibrationComponent].width'])
    IH = int(row['[CameraCalibrationComponent].height'])
    T_veh_from_cam = np.array(row['[CameraCalibrationComponent].extrinsic.transform'],
                               dtype=np.float64).reshape(4, 4)
    return fu, fv, cu, cv, IW, IH, T_veh_from_cam


def _load_lidar_cal(seg_name):
    df = pd.read_parquet(WAYMO_DIR / 'lidar_calibration' / f'{seg_name}.parquet')
    row = df[df['key.laser_name'] == TOP_LIDAR].iloc[0]
    T_veh_from_lidar = np.array(row['[LiDARCalibrationComponent].extrinsic.transform'],
                                 dtype=np.float64).reshape(4, 4)
    incl = np.array(row['[LiDARCalibrationComponent].beam_inclination.values'],
                    dtype=np.float64)
    if incl is None or len(incl) == 0:
        lo = float(row['[LiDARCalibrationComponent].beam_inclination.min'])
        hi = float(row['[LiDARCalibrationComponent].beam_inclination.max'])
        incl = np.linspace(lo, hi, 64)
    # Row 0 of range image = top beam; incl is stored bottom→top, so reverse.
    incl = incl[::-1].copy()
    # az_correction accounts for LiDAR mounting yaw (from official Waymo decode).
    az_correction = math.atan2(T_veh_from_lidar[1, 0], T_veh_from_lidar[0, 0])
    return T_veh_from_lidar, incl, az_correction


def _decode_range_image(ri_vals, ri_shape, incl, az_correction=0.0):
    """Range image → (N, 3) XYZ in lidar sensor frame.

    Uses the official Waymo azimuth formula:
      azimuth[col] = π*(1 - 2*(col+0.5)/W) - az_correction
    where az_correction = atan2(T_veh_from_lidar[1,0], T_veh_from_lidar[0,0]).
    incl must already be reversed (row 0 = top beam).
    """
    ri = np.array(ri_vals, dtype=np.float32).reshape(ri_shape)
    H, W = ri.shape[:2]
    r = ri[:, :, 0]
    valid = r > 0
    if valid.sum() == 0:
        return np.zeros((0, 3), dtype=np.float32)
    cols = np.arange(W, dtype=np.float64)
    azimuth = np.pi * (1.0 - 2.0 * (cols + 0.5) / W) - az_correction
    cos_e = np.cos(incl[:, None]).astype(np.float32)
    sin_e = np.sin(incl[:, None]).astype(np.float32)
    cos_a = np.cos(azimuth[None, :]).astype(np.float32)
    sin_a = np.sin(azimuth[None, :]).astype(np.float32)
    xs = r * cos_e * cos_a
    ys = r * cos_e * sin_a
    zs = r * sin_e
    return np.stack([xs[valid], ys[valid], zs[valid]], axis=1)


def _project(pts_veh, T_cam_from_veh, fu, fv, cu, cv):
    """(N,3) vehicle → (N,2) UV + (N,) depth.  Waymo: depth=cx, u=-cy/cx, v=-cz/cx"""
    h = np.hstack([pts_veh, np.ones((len(pts_veh), 1), dtype=pts_veh.dtype)])
    c = (T_cam_from_veh @ h.T).T          # (N,4)
    depth = c[:, 0]
    safe  = np.where(np.abs(depth) > 1e-6, depth, 1e-6)
    u = fu * (-c[:, 1]) / safe + cu
    v = fv * (-c[:, 2]) / safe + cv
    return np.stack([u, v], axis=1), depth


def _bbox2d_of_box(cx, cy, cz, sx, sy, sz, heading, T_cam_from_veh, fu, fv, cu, cv):
    """3D box center+size+heading (vehicle frame) → 2D bbox or None."""
    cos_h, sin_h = math.cos(heading), math.sin(heading)
    hx, hy, hz = sx / 2, sy / 2, sz / 2
    corners = np.array([
        [ hx,  hy,  hz], [ hx,  hy, -hz], [ hx, -hy,  hz], [ hx, -hy, -hz],
        [-hx,  hy,  hz], [-hx,  hy, -hz], [-hx, -hy,  hz], [-hx, -hy, -hz],
    ], dtype=np.float32)
    R = np.array([[cos_h, -sin_h, 0], [sin_h, cos_h, 0], [0, 0, 1]], dtype=np.float32)
    corners_w = (R @ corners.T).T + np.array([cx, cy, cz], dtype=np.float32)
    uv, depth = _project(corners_w, T_cam_from_veh, fu, fv, cu, cv)
    if (depth <= 0).any():
        return None
    return uv[:, 0].min(), uv[:, 1].min(), uv[:, 0].max(), uv[:, 1].max()


# ── cache builder ────────────────────────────────────────────────────────────

def _process_seg(args):
    seg, obj_types, random_crops, frame_sample, tmp_dir = args
    out_path = Path(tmp_dir) / f'{seg}.pt'
    try:
        fu, fv, cu, cv, IW, IH, T_veh_from_cam = _load_cam_cal(seg)
        T_cam_from_veh_gt = np.linalg.inv(T_veh_from_cam).astype(np.float32)
        T_veh_from_lidar, incl, az_corr = _load_lidar_cal(seg)

        lidar_df = pd.read_parquet(WAYMO_DIR / 'lidar'         / f'{seg}.parquet')
        cam_df   = pd.read_parquet(WAYMO_DIR / 'camera_image'  / f'{seg}.parquet')
        box_df   = pd.read_parquet(WAYMO_DIR / 'lidar_box'     / f'{seg}.parquet')

        lidar_top = lidar_df[lidar_df['key.laser_name'] == TOP_LIDAR]
        cam_front = cam_df[cam_df['key.camera_name']   == FRONT_CAM]

        timestamps = sorted(lidar_top['key.frame_timestamp_micros'].unique())
        if frame_sample < 1.0:
            k = max(1, round(len(timestamps) * frame_sample))
            timestamps = random.sample(timestamps, k)

        instances = []
        for ts in timestamps:
            lidar_row = lidar_top[lidar_top['key.frame_timestamp_micros'] == ts]
            cam_row   = cam_front[cam_front['key.frame_timestamp_micros'] == ts]
            if lidar_row.empty or cam_row.empty:
                continue

            lr = lidar_row.iloc[0]
            pts_sensor = _decode_range_image(
                lr['[LiDARComponent].range_image_return1.values'],
                lr['[LiDARComponent].range_image_return1.shape'], incl, az_corr)
            if len(pts_sensor) < 10:
                continue
            pts_h   = np.hstack([pts_sensor, np.ones((len(pts_sensor), 1), dtype=np.float32)])
            pts_veh = (T_veh_from_lidar @ pts_h.T).T[:, :3].astype(np.float32)

            uv_gt, z_gt = _project(pts_veh, T_cam_from_veh_gt, fu, fv, cu, cv)
            vis = (z_gt > 0.5) & (uv_gt[:,0] >= 0) & (uv_gt[:,0] < IW) & \
                  (uv_gt[:,1] >= 0) & (uv_gt[:,1] < IH)
            if vis.sum() < MIN_PTS:
                continue
            pts_vis   = pts_veh[vis]
            uv_gt_vis = uv_gt[vis]

            img_bytes = bytes(cam_row.iloc[0]['[CameraImageComponent].image'])
            img_full  = Image.open(io.BytesIO(img_bytes)).convert('RGB')

            boxes = box_df[(box_df['key.frame_timestamp_micros'] == ts) &
                            (box_df['[LiDARBoxComponent].type'].isin(obj_types))]

            for _, b in boxes.iterrows():
                bx = float(b['[LiDARBoxComponent].box.center.x'])
                by = float(b['[LiDARBoxComponent].box.center.y'])
                bz = float(b['[LiDARBoxComponent].box.center.z'])
                sx = float(b['[LiDARBoxComponent].box.size.x'])
                sy = float(b['[LiDARBoxComponent].box.size.y'])
                sz = float(b['[LiDARBoxComponent].box.size.z'])
                heading = float(b['[LiDARBoxComponent].box.heading'])

                bbox = _bbox2d_of_box(bx, by, bz, sx, sy, sz, heading,
                                      T_cam_from_veh_gt, fu, fv, cu, cv)
                if bbox is None:
                    continue
                u_min, v_min, u_max, v_max = bbox
                uc, vc = (u_min + u_max) / 2, (v_min + v_max) / 2
                if not (0 <= uc < IW and 0 <= vc < IH):
                    continue
                bw, bh = u_max - u_min, v_max - v_min
                crop_size = max(max(bw, bh) * BBOX_SCALE, 32)
                half = crop_size / 2
                u0 = float(np.clip(uc - half, 0, IW - crop_size))
                v0 = float(np.clip(vc - half, 0, IH - crop_size))
                box_px = (int(u0), int(v0), int(u0 + crop_size), int(v0 + crop_size))
                img_64 = np.array(img_full.crop(box_px).resize((IMG_SIZE, IMG_SIZE),
                                                                 Image.BILINEAR), dtype=np.uint8)
                ch, sh = math.cos(heading), math.sin(heading)
                R_obj = np.array([[ch, sh, 0], [-sh, ch, 0], [0, 0, 1]], dtype=np.float32)

                # 2× crop ROI; 2px grid in the 64×64-view → 64×64 cells over ROI
                margin = crop_size * 0.5
                roi_w  = crop_size + 2 * margin
                in_roi = ((uv_gt_vis[:, 0] >= u0 - margin) & (uv_gt_vis[:, 0] < u0 + crop_size + margin) &
                          (uv_gt_vis[:, 1] >= v0 - margin) & (uv_gt_vis[:, 1] < v0 + crop_size + margin))
                pts_roi = pts_vis[in_roi]
                uv_roi  = uv_gt_vis[in_roi]
                if len(pts_roi) >= MIN_PTS:
                    GS = 64
                    gu = u0 - margin + (np.arange(GS) + 0.5) * (roi_w / GS)
                    gv = v0 - margin + (np.arange(GS) + 0.5) * (roi_w / GS)
                    ggu, ggv = np.meshgrid(gu, gv)
                    ggu = ggu.ravel()[:, None]; ggv = ggv.ravel()[:, None]
                    d2 = (uv_roi[:, 0][None] - ggu) ** 2 + (uv_roi[:, 1][None] - ggv) ** 2
                    g_sel = sorted(set(d2.argmin(axis=1).tolist()))
                    pts_samp = pts_roi[g_sel]
                else:
                    pts_samp = pts_roi

                pts_local = (R_obj @ (pts_samp - np.array([bx, by, bz], dtype=np.float32)).T).T
                half3 = np.array([sx, sy, sz], dtype=np.float32) / 2
                is_obj_full = ((np.abs(pts_local[:, 0]) <= half3[0]) &
                               (np.abs(pts_local[:, 1]) <= half3[1]) &
                               (np.abs(pts_local[:, 2]) <= half3[2]))
                instances.append(dict(
                    seg=seg,
                    img_64=torch.from_numpy(img_64).permute(2, 0, 1),
                    u0=u0, v0=v0, crop_size=float(crop_size),
                    obj_pos=np.array([bx, by, bz], dtype=np.float32),
                    obj_dims=np.array([sx, sy, sz], dtype=np.float32),
                    obj_heading=heading,
                    is_obj_full=is_obj_full,
                    pts_vis=pts_samp,
                    T_cam_gt=T_cam_from_veh_gt,
                    T_veh_from_cam=T_veh_from_cam.astype(np.float32),
                    fu=fu, fv=fv, cu=cu, cv=cv, IW=IW, IH=IH,
                    random_crop=False,
                ))

            if random_crops:
                rand_crop_size = int(IW * 0.10)
                n_rand = max(1, len(boxes) // 2)
                for _ in range(n_rand * 3):
                    ru0 = random.randint(0, IW - rand_crop_size)
                    rv0 = random.randint(0, IH - rand_crop_size)
                    r_margin = rand_crop_size * 0.5
                    in_roi_r = ((uv_gt_vis[:, 0] >= ru0 - r_margin) & (uv_gt_vis[:, 0] < ru0 + rand_crop_size + r_margin) &
                                (uv_gt_vis[:, 1] >= rv0 - r_margin) & (uv_gt_vis[:, 1] < rv0 + rand_crop_size + r_margin))
                    if in_roi_r.sum() < MIN_PTS:
                        continue
                    in_c = ((uv_gt_vis[in_roi_r, 0] >= ru0) & (uv_gt_vis[in_roi_r, 0] < ru0 + rand_crop_size) &
                            (uv_gt_vis[in_roi_r, 1] >= rv0) & (uv_gt_vis[in_roi_r, 1] < rv0 + rand_crop_size))
                    if in_c.sum() < MIN_PTS:
                        continue
                    pts_roi_r = pts_vis[in_roi_r]; uv_roi_r = uv_gt_vis[in_roi_r]
                    roi_w_r = rand_crop_size + 2 * r_margin
                    GS = 64
                    gu_r = ru0 - r_margin + (np.arange(GS) + 0.5) * (roi_w_r / GS)
                    gv_r = rv0 - r_margin + (np.arange(GS) + 0.5) * (roi_w_r / GS)
                    ggu_r, ggv_r = np.meshgrid(gu_r, gv_r)
                    ggu_r = ggu_r.ravel()[:, None]; ggv_r = ggv_r.ravel()[:, None]
                    d2_r = (uv_roi_r[:, 0][None] - ggu_r) ** 2 + (uv_roi_r[:, 1][None] - ggv_r) ** 2
                    g_sel_r = sorted(set(d2_r.argmin(axis=1).tolist()))
                    pts_samp_r = pts_roi_r[g_sel_r]
                    is_obj_r   = np.zeros(len(pts_samp_r), dtype=bool)
                    box_r = (ru0, rv0, ru0 + rand_crop_size, rv0 + rand_crop_size)
                    img_r = np.array(img_full.crop(box_r).resize((IMG_SIZE, IMG_SIZE),
                                                                   Image.BILINEAR), dtype=np.uint8)
                    instances.append(dict(
                        seg=seg,
                        img_64=torch.from_numpy(img_r).permute(2, 0, 1),
                        u0=float(ru0), v0=float(rv0), crop_size=float(rand_crop_size),
                        obj_pos=None, obj_dims=None, obj_heading=None,
                        is_obj_full=is_obj_r,
                        pts_vis=pts_samp_r,
                        T_cam_gt=T_cam_from_veh_gt,
                        T_veh_from_cam=T_veh_from_cam.astype(np.float32),
                        fu=fu, fv=fv, cu=cu, cv=cv, IW=IW, IH=IH,
                        random_crop=True,
                    ))
                    break
        torch.save(instances, out_path)
        return str(out_path), len(instances)
    except Exception as e:
        print(f'  skip {seg}: {e}', flush=True)
        return None, 0


def build_cache(cache_path: str, max_segs: int = None, random_crops: bool = True,
                target_types: set = None, frame_sample: float = 1.0,
                num_workers: int = 4):
    import tempfile
    from multiprocessing import Pool
    segs_lidar = sorted(f.stem for f in (WAYMO_DIR / 'lidar').glob('*.parquet'))
    segs_cam   = {f.stem for f in (WAYMO_DIR / 'camera_image').glob('*.parquet')}
    segs_box   = {f.stem for f in (WAYMO_DIR / 'lidar_box').glob('*.parquet')}
    segs = [s for s in segs_lidar if s in segs_cam and s in segs_box]
    if max_segs:
        segs = segs[:max_segs]
    obj_types = target_types if target_types is not None else TARGET_TYPES
    print(f'Building Waymo cache: {len(segs)} segments  types={obj_types}  workers={num_workers} → {cache_path}', flush=True)

    with tempfile.TemporaryDirectory() as tmp_dir:
        args = [(seg, obj_types, random_crops, frame_sample, tmp_dir) for seg in segs]
        tmp_files = []
        total = 0
        with Pool(num_workers) as pool:
            for si, (fpath, n) in enumerate(pool.imap_unordered(_process_seg, args)):
                if fpath:
                    tmp_files.append(fpath)
                    total += n
                if (si + 1) % 10 == 0:
                    print(f'  [{si+1}/{len(segs)}] ~{total} instances', flush=True)

        print(f'Combining {len(tmp_files)} segment files...', flush=True)
        instances = []
        for fp in tmp_files:
            instances.extend(torch.load(fp, weights_only=False))

    print(f'Total instances: {len(instances)}', flush=True)
    torch.save(instances, cache_path)
    return instances


# ── Dataset ──────────────────────────────────────────────────────────────────

class WaymoCalibDataset(Dataset):
    def __init__(self, cache_path: str, split: str = 'train',
                 img_size: int = IMG_SIZE, max_offset_m: float = 0.20,
                 max_rot_deg: float = 0.5, min_pts: int = MIN_PTS,
                 max_dist_m: float = None):
        instances = torch.load(cache_path, weights_only=False)
        if max_dist_m is not None:
            import numpy as np
            instances = [inst for inst in instances
                         if inst['obj_pos'] is None or
                         np.linalg.norm(inst['obj_pos']) <= max_dist_m]
        n = len(instances)
        if split == 'train':
            self.instances = instances[:int(n * 0.9)]
        else:
            self.instances = instances[int(n * 0.9):]
        self.img_size    = img_size
        self.max_offset  = max_offset_m
        self.max_rot     = max_rot_deg
        self.min_pts     = min_pts

    def __len__(self): return len(self.instances)

    def __getitem__(self, idx):
        inst = self.instances[idx]
        S         = self.img_size
        crop_size = inst['crop_size']
        u0, v0    = inst['u0'], inst['v0']
        fu, fv    = inst['fu'], inst['fv']
        cu, cv    = inst['cu'], inst['cv']
        scale     = S / crop_size

        pts_vis    = inst['pts_vis']          # (N, 3) vehicle frame
        T_cam_gt   = inst['T_cam_gt']         # (4,4)
        T_veh_from_cam = inst['T_veh_from_cam']
        is_obj_full = inst['is_obj_full']     # (N,) bool

        # Random perturbation of camera pose
        for _ in range(10):
            t_delta = (np.random.rand(3) * 2 - 1) * self.max_offset
            ypr     = (np.random.rand(3) * 2 - 1) * self.max_rot
            R_gt    = T_veh_from_cam[:3, :3].T   # cam_from_veh rotation
            R_off   = R_gt @ Rotation.from_euler('zyx', ypr, degrees=True).as_matrix()
            t_gt    = T_veh_from_cam[:3, 3]
            t_off   = t_gt + t_delta
            # Build perturbed T_cam_from_veh
            T_off   = np.eye(4, dtype=np.float32)
            T_off[:3, :3] = R_off.T
            T_off[:3,  3] = -(R_off.T @ t_off)

            # Project with perturbed pose
            h = np.hstack([pts_vis, np.ones((len(pts_vis), 1), dtype=np.float32)])
            c_off = (T_off @ h.T).T
            depth_off = c_off[:, 0]
            u_off = fu * (-c_off[:, 1]) / depth_off + cu
            v_off = fv * (-c_off[:, 2]) / depth_off + cv

            in_crop = ((u_off >= u0) & (u_off < u0 + crop_size) &
                       (v_off >= v0) & (v_off < v0 + crop_size) &
                       (depth_off > 0.5))
            if in_crop.sum() >= self.min_pts:
                break
        else:
            return self[random.randint(0, len(self) - 1)]

        u_off_c = (u_off[in_crop] - u0) * scale
        v_off_c = (v_off[in_crop] - v0) * scale

        # 32×32 grid (2px in 64-view): bin each pt to its cell, keep nearest-to-center.
        grid, cell = 32, float(S) / 32
        ci = np.clip((v_off_c / cell).astype(np.int64), 0, grid - 1)
        cj = np.clip((u_off_c / cell).astype(np.int64), 0, grid - 1)
        cell_id = ci * grid + cj
        du = u_off_c - (cj + 0.5) * cell
        dv = v_off_c - (ci + 0.5) * cell
        d2 = du * du + dv * dv
        sel = []
        for cid in np.unique(cell_id):
            members = np.where(cell_id == cid)[0]
            sel.append(int(members[d2[members].argmin()]))
        sel = np.array(sorted(sel), dtype=np.int64)
        u_off_c = u_off_c[sel]
        v_off_c = v_off_c[sel]
        idx_in  = np.where(in_crop)[0][sel]

        # GT projection
        h2 = np.hstack([pts_vis[idx_in], np.ones((len(idx_in), 1), dtype=np.float32)])
        c_gt = (T_cam_gt @ h2.T).T
        d_gt = c_gt[:, 0]
        u_gt_c = (fu * (-c_gt[:, 1]) / d_gt + cu - u0) * scale
        v_gt_c = (fv * (-c_gt[:, 2]) / d_gt + cv - v0) * scale

        dist_m  = (np.linalg.norm(pts_vis[idx_in], axis=1) / 100.0).astype(np.float32)
        is_obj  = is_obj_full[idx_in].astype(np.float32)

        dist_uvd = np.stack([u_off_c.astype(np.float32), v_off_c.astype(np.float32),
                              dist_m, is_obj], axis=1)
        true_uvd = np.stack([u_gt_c.astype(np.float32),  v_gt_c.astype(np.float32),
                              dist_m, is_obj], axis=1)

        img_crop = inst['img_64'].float() / 255.0
        return img_crop, torch.from_numpy(true_uvd), torch.from_numpy(dist_uvd)


# ── collate ──────────────────────────────────────────────────────────────────

def collate_waymo(batch):
    imgs, true_list, dist_list = zip(*batch)
    imgs = torch.stack(imgs)
    max_n = max(t.shape[0] for t in true_list)
    def pad(tensors):
        out  = torch.zeros(len(tensors), max_n, tensors[0].shape[1])
        mask = torch.ones(len(tensors), max_n, dtype=torch.bool)
        for i, t in enumerate(tensors):
            out[i, :t.shape[0]] = t
            mask[i, :t.shape[0]] = False
        return out, mask
    true_t, mask = pad(true_list)
    dist_t, _    = pad(dist_list)
    return imgs, true_t, dist_t, mask


# ── main: build cache ─────────────────────────────────────────────────────────

if __name__ == '__main__':
    import sys
    cache = sys.argv[1] if len(sys.argv) > 1 else '/tmp/waymo_cache.pt'
    max_segs = int(sys.argv[2]) if len(sys.argv) > 2 else None
    build_cache(cache, max_segs=max_segs)
