"""nuScenes calibration dataset — same cache/interface as dataset_pandaset.py.

Uses CAM_FRONT + LIDAR_TOP.
Target classes: vehicle.car, human.pedestrian.adult,
                movable_object.trafficcone, movable_object.barrier

Cache phase: img at CACHE_IMG×CACHE_IMG with bbox_scale=3.0 (bbox in center 1/3).
__getitem__: random sub-window in cache-image pixels, then bilinear-resize to img_size.
"""
import json, random
import numpy as np
from pathlib import Path
from PIL import Image
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset
from scipy.spatial.transform import Rotation

CACHE_IMG = 192  # cache image resolution (square)

_NS_CTX = {}  # shared context for multiprocessing workers

def _ns_init(ctx):
    global _NS_CTX
    _NS_CTX = ctx

def _ns_process_scene(args):
    scene, is_val = args
    ctx   = _NS_CTX
    rng_s = random.Random(ctx['seed'] ^ hash(scene['token']))
    result = []
    tok = scene['first_sample_token']
    while tok:
        if ctx['frame_sample'] < 1.0 and rng_s.random() > ctx['frame_sample']:
            tok = ctx['samples'][tok]['next']; continue
        sample   = ctx['samples'][tok]
        sd_map   = ctx['sd_by_sample'].get(tok, {})
        cam_sd   = sd_map.get('CAM_FRONT')
        lidar_sd = sd_map.get('LIDAR_TOP')
        if cam_sd is None or lidar_sd is None:
            tok = sample['next']; continue

        cam_cal  = ctx['cal_sensors'][cam_sd['calibrated_sensor_token']]
        ep_cam   = ctx['ego_poses'][cam_sd['ego_pose_token']]
        T_cam2w  = _sensor_to_world(cam_cal, ep_cam)
        T_w2cam  = np.linalg.inv(T_cam2w).astype(np.float32)
        K        = np.array(cam_cal['camera_intrinsic'], dtype=np.float64)
        cam_pos  = T_cam2w[:3, 3].astype(np.float32)

        lidar_cal = ctx['cal_sensors'][lidar_sd['calibrated_sensor_token']]
        ep_lidar  = ctx['ego_poses'][lidar_sd['ego_pose_token']]
        T_lidar2w = _sensor_to_world(lidar_cal, ep_lidar)

        bin_path  = ctx['root'] / lidar_sd['filename']
        pts_lidar = np.frombuffer(bin_path.read_bytes(),
                                  dtype=np.float32).reshape(-1, 5)[:, :3]
        pts_world_h = np.hstack([pts_lidar, np.ones((len(pts_lidar),1), dtype=np.float32)])
        pts_world   = (T_lidar2w @ pts_world_h.T).T[:, :3].astype(np.float32)

        IW, IH   = ctx['IW'], ctx['IH']
        cache_img = ctx['cache_img']
        uv_gt, z_gt = _project(pts_world, T_w2cam, K)
        vis = (z_gt > 0.5) & (uv_gt[:,0]>=0) & (uv_gt[:,0]<IW) & \
              (uv_gt[:,1]>=0) & (uv_gt[:,1]<IH)
        if vis.sum() < ctx['min_pts']:
            tok = sample['next']; continue
        pts_vis = pts_world[vis]

        img_full = Image.open(ctx['root'] / cam_sd['filename']).convert('RGB')

        n_before_frame = len(result)
        for a in ctx['ann_by_sample'].get(tok, []):
            cat = ctx['cats'].get(ctx['instances'][a['instance_token']]['category_token'], '')
            if cat not in ctx['use_cats']: continue

            pos      = np.array(a['translation'], dtype=np.float32)
            size_wlh = np.array(a['size'],        dtype=np.float32)
            quat     = a['rotation']

            bbox = _bbox2d(pos, size_wlh, quat, T_w2cam, K)
            if bbox is None: continue
            u_min, v_min, u_max, v_max = bbox
            uc, vc = (u_min+u_max)/2, (v_min+v_max)/2
            if not (0 <= uc < IW and 0 <= vc < IH): continue

            bw, bh    = u_max-u_min, v_max-v_min
            crop_size = max(max(bw,bh)*ctx['bbox_scale'], 32)
            half      = crop_size/2
            u0 = float(np.clip(uc-half, 0, IW-crop_size))
            v0 = float(np.clip(vc-half, 0, IH-crop_size))
            crop_size = float(crop_size)

            box       = (int(u0), int(v0), int(u0+crop_size), int(v0+crop_size))
            img_cache = np.array(img_full.crop(box).resize((cache_img, cache_img),
                                                            Image.BILINEAR), dtype=np.uint8)
            obj_yaw = Rotation.from_quat([quat[1],quat[2],quat[3],quat[0]]).as_euler('zyx')[0]

            result.append({
                'img_cache': torch.from_numpy(img_cache).permute(2,0,1),  # (3, C, C)
                'pts':       torch.from_numpy(pts_vis),
                'cam_pos':   torch.from_numpy(cam_pos),
                'R_gt':      torch.from_numpy(T_cam2w[:3,:3].copy()),
                'T_gt':      torch.from_numpy(T_w2cam),
                'K_full':    torch.from_numpy(K.astype(np.float32)),
                'u0': u0, 'v0': v0, 'crop_size': crop_size,
                'obj_pos':  torch.from_numpy(pos),
                'obj_dims': torch.from_numpy(size_wlh),
                'obj_yaw':  float(obj_yaw),
                'is_val':   is_val,
            })

        if ctx['random_crops']:
            n_obj = len(result) - n_before_frame
            rand_crop_size = int(IW * 0.15)     # bigger so sub-window still has resolution
            for _ in range(max(1, int(n_obj * 0.5))):
                ru0 = rng_s.randint(0, IW-rand_crop_size)
                rv0 = rng_s.randint(0, IH-rand_crop_size)
                in_c = ((uv_gt[vis,0]>=ru0) & (uv_gt[vis,0]<ru0+rand_crop_size) &
                        (uv_gt[vis,1]>=rv0) & (uv_gt[vis,1]<rv0+rand_crop_size))
                if in_c.sum() < ctx['min_pts']: continue
                box_r = (ru0, rv0, ru0+rand_crop_size, rv0+rand_crop_size)
                img_r = np.array(img_full.crop(box_r).resize((cache_img, cache_img),
                                                              Image.BILINEAR), dtype=np.uint8)
                result.append({
                    'img_cache': torch.from_numpy(img_r).permute(2,0,1),
                    'pts':       torch.from_numpy(pts_vis),
                    'cam_pos':   torch.from_numpy(cam_pos),
                    'R_gt':      torch.from_numpy(T_cam2w[:3,:3].copy()),
                    'T_gt':      torch.from_numpy(T_w2cam),
                    'K_full':    torch.from_numpy(K.astype(np.float32)),
                    'u0': float(ru0), 'v0': float(rv0), 'crop_size': float(rand_crop_size),
                    'obj_pos': torch.zeros(3), 'obj_dims': torch.zeros(3), 'obj_yaw': 0.0,
                    'is_val':  is_val,
                })

        tok = sample['next']

    if ctx['max_per_scene'] and len(result) > ctx['max_per_scene']:
        rng_s.shuffle(result)
        result = result[:ctx['max_per_scene']]

    print(f"  {scene['name']}: {len(result)} instances", flush=True)
    return result

USEFUL_CATS = {
    'vehicle.car', 'human.pedestrian.adult',
    'movable_object.trafficcone', 'movable_object.barrier',
}


# ── geometry helpers ─────────────────────────────────────────────────────────

def _quat_mat(q_wxyz):
    """[w,x,y,z] → 3×3 rotation matrix."""
    return Rotation.from_quat([q_wxyz[1], q_wxyz[2], q_wxyz[3], q_wxyz[0]]).as_matrix()


def _sensor_to_world(cal, ego_pose):
    """calibrated_sensor + ego_pose → 4×4 sensor-to-world."""
    R_s = _quat_mat(cal['rotation']); t_s = np.array(cal['translation'])
    R_e = _quat_mat(ego_pose['rotation']); t_e = np.array(ego_pose['translation'])
    T = np.eye(4); T[:3,:3]=R_e; T[:3,3]=t_e
    S = np.eye(4); S[:3,:3]=R_s; S[:3,3]=t_s
    return (T @ S).astype(np.float32)


def _project(pts_world, T_world2cam, K):
    pts_cam = (T_world2cam[:3,:3] @ pts_world.T + T_world2cam[:3,3:]).T
    z = pts_cam[:,2]
    safe = np.where(np.abs(z) > 1e-6, z, 1e-6)
    uv = (K @ pts_cam.T)[:2] / safe
    return uv.T, z


def _bbox2d(pos, size_wlh, quat_wxyz, T_world2cam, K):
    """Project 3D box corners to image → (u0,v0,u1,v1) or None."""
    w, l, h = size_wlh[0]/2, size_wlh[1]/2, size_wlh[2]/2
    R = _quat_mat(quat_wxyz)
    corners_local = np.array([[sx*l, sy*w, sz*h]
                               for sx in [-1,1] for sy in [-1,1] for sz in [-1,1]])
    corners_world = (R @ corners_local.T).T + np.array(pos)
    uv, z = _project(corners_world, T_world2cam, K)
    front = z > 0.1
    if front.sum() < 4: return None
    uv = uv[front]
    return uv[:,0].min(), uv[:,1].min(), uv[:,0].max(), uv[:,1].max()


# ── cache builder ─────────────────────────────────────────────────────────────

def build_cache(nuscenes_root: str,
                cache_path: str,
                version: str = 'v1.0-mini',
                val_fraction: float = 0.15,
                seed: int = 42,
                max_per_scene: int = None,
                random_crops: bool = False,
                bbox_scale: float = 3.0,
                min_pts: int = 8,
                target_cats: set = None,
                frame_sample: float = 1.0,
                num_workers: int = 16,
                max_scenes: int = None,
                cache_img: int = CACHE_IMG):

    root    = Path(nuscenes_root)
    meta    = root / version

    # load all metadata
    scenes      = json.load(open(meta/'scene.json'))
    samples     = {s['token']: s for s in json.load(open(meta/'sample.json'))}
    sample_data = json.load(open(meta/'sample_data.json'))
    anns_all    = json.load(open(meta/'sample_annotation.json'))
    instances   = {i['token']: i for i in json.load(open(meta/'instance.json'))}
    cats        = {c['token']: c['name'] for c in json.load(open(meta/'category.json'))}
    cal_sensors = {c['token']: c for c in json.load(open(meta/'calibrated_sensor.json'))}
    sensors_map = {s['token']: s for s in json.load(open(meta/'sensor.json'))}
    ego_poses   = {e['token']: e for e in json.load(open(meta/'ego_pose.json'))}

    # index sample_data by sample token and channel
    sd_by_sample = {}
    for sd in sample_data:
        cs = cal_sensors[sd['calibrated_sensor_token']]
        ch = sensors_map[cs['sensor_token']]['channel']
        sd_by_sample.setdefault(sd['sample_token'], {})[ch] = sd

    # index annotations by sample token
    ann_by_sample = {}
    for a in anns_all:
        ann_by_sample.setdefault(a['sample_token'], []).append(a)

    IW, IH = 1600, 900

    rng = random.Random(seed)
    rng.shuffle(scenes)
    if max_scenes:
        scenes = scenes[:max_scenes]
    n_val = max(0, int(len(scenes) * val_fraction))

    use_cats = target_cats if target_cats is not None else USEFUL_CATS

    # shared context passed to each worker (fork-safe on Linux)
    _ctx = dict(root=root, samples=samples, sd_by_sample=sd_by_sample,
                ann_by_sample=ann_by_sample, instances=instances, cats=cats,
                cal_sensors=cal_sensors, ego_poses=ego_poses,
                IW=IW, IH=IH, use_cats=use_cats, bbox_scale=bbox_scale,
                min_pts=min_pts, random_crops=random_crops, cache_img=cache_img,
                frame_sample=frame_sample, max_per_scene=max_per_scene, seed=seed)

    from multiprocessing import Pool
    scene_args = [(scene, si < n_val) for si, scene in enumerate(scenes)]
    train_instances, val_instances = [], []
    with Pool(num_workers, initializer=_ns_init, initargs=(_ctx,)) as pool:
        for result in pool.imap_unordered(_ns_process_scene, scene_args):
            for inst in result:
                if inst.pop('is_val'):
                    val_instances.append(inst)
                else:
                    train_instances.append(inst)

    torch.save({'train': train_instances, 'val': val_instances}, cache_path)
    print(f"Saved → {cache_path}  train={len(train_instances)} val={len(val_instances)}", flush=True)


# ── build_sample (explicit perturbation, for BA) ─────────────────────────────

def build_sample(inst, ypr, t_delta, img_size: int = 64,
                 min_pts: int = 8, sub=None):
    """Apply explicit (ypr deg, t_delta m) perturbation to a cached NS instance.
    Returns (img_crop, true_uvd, dist_uvd, idx_in) or None if too few points."""
    pts = inst['pts'].numpy()
    cp  = inst['cam_pos'].numpy()
    R_gt = inst['R_gt'].numpy()
    T_gt = inst['T_gt'].numpy()
    K    = inst['K_full'].numpy()
    img_cache = inst.get('img_cache', inst.get('img_64'))
    C    = int(img_cache.shape[-1])
    S    = img_size

    if sub is None:
        u_sub, v_sub, s_sub = 0, 0, C
    else:
        u_sub, v_sub, s_sub = int(sub[0]), int(sub[1]), int(sub[2])

    R_off = R_gt @ Rotation.from_euler('zyx', ypr, degrees=True).as_matrix()
    cp_off = cp + t_delta
    T_off = np.eye(4, dtype=np.float32)
    T_off[:3, :3] = R_off.T
    T_off[:3,  3] = -(R_off.T @ cp_off)

    cs_per_px = float(inst['crop_size']) / C
    cu0 = float(inst['u0']) + u_sub * cs_per_px
    cv0 = float(inst['v0']) + v_sub * cs_per_px
    crop_size = s_sub * cs_per_px

    pts_cam_off = T_off[:3, :3] @ pts.T + T_off[:3, 3:]
    z_off  = pts_cam_off[2]
    uv_off = ((K @ pts_cam_off)[:2] / z_off).T

    in_crop = ((uv_off[:,0] >= cu0) & (uv_off[:,0] < cu0 + crop_size) &
               (uv_off[:,1] >= cv0) & (uv_off[:,1] < cv0 + crop_size) &
               (z_off > 0.5))
    if in_crop.sum() < min_pts:
        return None

    scale = S / crop_size
    uv_d_crop = np.stack([(uv_off[in_crop,0] - cu0) * scale,
                           (uv_off[in_crop,1] - cv0) * scale], axis=1)

    grid, cell = 16, float(S) / 16
    ci = np.clip((uv_d_crop[:,1] / cell).astype(np.int64), 0, grid - 1)
    cj = np.clip((uv_d_crop[:,0] / cell).astype(np.int64), 0, grid - 1)
    cell_id = ci * grid + cj
    du = uv_d_crop[:,0] - (cj + 0.5) * cell
    dv = uv_d_crop[:,1] - (ci + 0.5) * cell
    d2 = du*du + dv*dv
    sel = []
    for cid in np.unique(cell_id):
        members = np.where(cell_id == cid)[0]
        sel.append(int(members[d2[members].argmin()]))
    sel = sorted(sel)

    idx_in  = np.where(in_crop)[0][sel]
    pts_sel = pts[idx_in]

    pts_cam_gt = T_gt[:3,:3] @ pts_sel.T + T_gt[:3,3:]
    uv_gt = ((K @ pts_cam_gt)[:2] / pts_cam_gt[2]).T
    uv_gt_c  = np.stack([(uv_gt[:,0] - cu0) * scale,
                          (uv_gt[:,1] - cv0) * scale], axis=1)
    uv_off_c = uv_d_crop[sel]

    dist_m = (np.linalg.norm(pts_sel - cp, axis=1) / 100.0).astype(np.float32)

    if inst['obj_dims'].norm() > 0:
        obj_pos  = inst['obj_pos'].numpy()
        obj_dims = inst['obj_dims'].numpy()
        obj_yaw  = inst['obj_yaw']
        c_y, s_y = np.cos(obj_yaw), np.sin(obj_yaw)
        R_obj = np.array([[c_y, s_y, 0], [-s_y, c_y, 0], [0, 0, 1]], dtype=np.float32)
        pts_local = (R_obj @ (pts_sel - obj_pos).T).T
        half = obj_dims / 2.0
        is_obj = ((np.abs(pts_local[:,0]) <= half[1]) &
                  (np.abs(pts_local[:,1]) <= half[0]) &
                  (np.abs(pts_local[:,2]) <= half[2])).astype(np.float32)
    else:
        is_obj = np.zeros(len(pts_sel), dtype=np.float32)

    true_uvd = np.concatenate([uv_gt_c.astype(np.float32),  dist_m[:,None], is_obj[:,None]], axis=1)
    dist_uvd = np.concatenate([uv_off_c.astype(np.float32), dist_m[:,None], is_obj[:,None]], axis=1)

    sub_img  = img_cache[:, v_sub:v_sub+s_sub, u_sub:u_sub+s_sub].float().unsqueeze(0)
    img_crop = F.interpolate(sub_img, size=(S, S), mode='bilinear',
                              align_corners=False).squeeze(0) / 255.0

    return img_crop, torch.from_numpy(true_uvd), torch.from_numpy(dist_uvd), idx_in


# ── Dataset ───────────────────────────────────────────────────────────────────

class NuScenesCalibDataset(Dataset):
    def __init__(self, cache_path: str, split: str = 'train',
                 img_size: int = 64, max_offset_m: float = 0.20,
                 max_rot_deg: float = 0.5, min_pts: int = 8, max_tries: int = 20):
        data = torch.load(cache_path, weights_only=False)
        self.instances   = data[split]
        self.img_size    = img_size
        self.max_offset_m = max_offset_m
        self.max_rot_deg  = max_rot_deg
        self.min_pts      = min_pts
        self.max_tries    = max_tries

    def __len__(self): return len(self.instances)

    def _sample_sub(self, inst, C):
        """Random sub-window (u, v, s) in cache-image pixel coords.
        For obj crops, constrain sub to still contain the bbox (center 1/3)."""
        s = int(np.random.randint(self.img_size, C + 1))
        is_obj = inst['obj_dims'].norm() > 0
        if is_obj:
            lo = max(0, 2 * C // 3 - s)
            hi = min(C - s, C // 3)
            if hi < lo: hi = lo
            u = int(np.random.randint(lo, hi + 1))
            v = int(np.random.randint(lo, hi + 1))
        else:
            u = int(np.random.randint(0, C - s + 1))
            v = int(np.random.randint(0, C - s + 1))
        return (u, v, s)

    def __getitem__(self, idx):
        inst      = self.instances[idx % len(self.instances)]
        pts       = inst['pts'].numpy()
        cp        = inst['cam_pos'].numpy()
        R_gt      = inst['R_gt'].numpy()
        T_gt      = inst['T_gt'].numpy()
        K         = inst['K_full'].numpy()
        # Support both new (img_cache, C×C) and legacy (img_64, 64×64).
        img_cache = inst.get('img_cache', inst.get('img_64'))
        C         = int(img_cache.shape[-1])
        S         = self.img_size

        for _ in range(self.max_tries):
            # random perturbation
            t_delta = (np.random.rand(3)*2-1) * self.max_offset_m
            ypr     = (np.random.rand(3)*2-1) * self.max_rot_deg
            R_off   = R_gt @ Rotation.from_euler('zyx', ypr, degrees=True).as_matrix()
            cp_off  = cp + t_delta
            T_off   = np.eye(4, dtype=np.float32)
            T_off[:3,:3] = R_off.T
            T_off[:3, 3] = -(R_off.T @ cp_off)

            # sub-window → derive effective (u0, v0, crop_size) in full-res px
            u_sub, v_sub, s_sub = self._sample_sub(inst, C)
            cs_per_px = float(inst['crop_size']) / C
            cu0 = float(inst['u0']) + u_sub * cs_per_px
            cv0 = float(inst['v0']) + v_sub * cs_per_px
            crop_size = s_sub * cs_per_px

            # project with distorted pose
            pts_cam_off = T_off[:3,:3] @ pts.T + T_off[:3,3:]
            z_off  = pts_cam_off[2]
            uv_off = ((K @ pts_cam_off)[:2] / z_off).T

            in_crop = ((uv_off[:,0]>=cu0) & (uv_off[:,0]<cu0+crop_size) &
                       (uv_off[:,1]>=cv0) & (uv_off[:,1]<cv0+crop_size) &
                       (z_off > 0.5))
            if in_crop.sum() >= self.min_pts:
                break
        else:
            return self[random.randint(0, len(self)-1)]

        scale = S / crop_size
        uv_d_crop = np.stack([(uv_off[in_crop,0]-cu0)*scale,
                               (uv_off[in_crop,1]-cv0)*scale], axis=1)

        # 16×16 grid: bin each pt to its cell, keep nearest-to-center per cell.
        grid, cell = 16, float(S)/16
        ci = np.clip((uv_d_crop[:,1] / cell).astype(np.int64), 0, grid-1)
        cj = np.clip((uv_d_crop[:,0] / cell).astype(np.int64), 0, grid-1)
        cell_id = ci * grid + cj
        du = uv_d_crop[:,0] - (cj + 0.5) * cell
        dv = uv_d_crop[:,1] - (ci + 0.5) * cell
        d2 = du*du + dv*dv
        sel = []
        for cid in np.unique(cell_id):
            members = np.where(cell_id == cid)[0]
            sel.append(int(members[d2[members].argmin()]))
        sel = sorted(sel)

        idx_in  = np.where(in_crop)[0][sel]
        pts_sel = pts[idx_in]

        pts_cam_gt = T_gt[:3,:3] @ pts_sel.T + T_gt[:3,3:]
        uv_gt      = ((K @ pts_cam_gt)[:2] / pts_cam_gt[2]).T
        uv_gt_c  = np.stack([(uv_gt[:,0]-cu0)*scale,
                              (uv_gt[:,1]-cv0)*scale], axis=1)
        uv_off_c = uv_d_crop[sel]

        dist_m = (np.linalg.norm(pts_sel - cp, axis=1) / 100.0).astype(np.float32)

        # is_obj: 3D cuboid check
        if inst['obj_dims'].norm() > 0:
            obj_pos  = inst['obj_pos'].numpy()
            obj_dims = inst['obj_dims'].numpy()
            obj_yaw  = inst['obj_yaw']
            c_y, s_y = np.cos(obj_yaw), np.sin(obj_yaw)
            R_obj    = np.array([[c_y,s_y,0],[-s_y,c_y,0],[0,0,1]], dtype=np.float32)
            pts_local = (R_obj @ (pts_sel - obj_pos).T).T
            half = obj_dims / 2.0
            # nuScenes Box: local_x = length, local_y = width, local_z = height
            # obj_dims = [w, l, h] so half = [w/2, l/2, h/2]
            is_obj = ((np.abs(pts_local[:,0])<=half[1]) &
                      (np.abs(pts_local[:,1])<=half[0]) &
                      (np.abs(pts_local[:,2])<=half[2])).astype(np.float32)
        else:
            is_obj = np.zeros(len(pts_sel), dtype=np.float32)

        true_uvd = np.concatenate([uv_gt_c.astype(np.float32),  dist_m[:,None], is_obj[:,None]], axis=1)
        dist_uvd = np.concatenate([uv_off_c.astype(np.float32), dist_m[:,None], is_obj[:,None]], axis=1)

        sub_img  = img_cache[:, v_sub:v_sub+s_sub, u_sub:u_sub+s_sub].float().unsqueeze(0)
        img_crop = F.interpolate(sub_img, size=(S, S), mode='bilinear',
                                  align_corners=False).squeeze(0) / 255.0

        return img_crop, torch.from_numpy(true_uvd), torch.from_numpy(dist_uvd)


# ── collate (same as pandaset) ────────────────────────────────────────────────

def collate_nuscenes(batch):
    imgs, true_uvds, dist_uvds = zip(*batch)
    imgs = torch.stack(imgs)
    def pad(seqs):
        N = max(s.shape[0] for s in seqs)
        t = torch.zeros(len(seqs), N, seqs[0].shape[1])
        m = torch.ones(len(seqs), N, dtype=torch.bool)
        for i, s in enumerate(seqs):
            t[i, :s.shape[0]] = s; m[i, :s.shape[0]] = False
        return t, m
    true_t, _ = pad(true_uvds)
    dist_t, mask = pad(dist_uvds)
    return imgs, true_t, dist_t, mask


# ── smoke test ────────────────────────────────────────────────────────────────

if __name__ == '__main__':
    cache = '/tmp/nuscenes_mini_cache.pt'
    if not Path(cache).exists():
        print("Building cache...")
        build_cache('/mnt/backup/nuscenes', cache, version='v1.0-mini',
                    val_fraction=0.2, random_crops=False)
    ds = NuScenesCalibDataset(cache, split='train')
    print(f"instances: {len(ds)}")
    img, t, d = ds[0]
    print(f"img={img.shape}  true_uvd={t.shape}")
    print(f"shift={(t[:,:2]-d[:,:2]).norm(dim=1).mean():.2f}px")
    print(f"depth range: {t[:,2].min():.3f}–{t[:,2].max():.3f}")
    print(f"obj ratio: {t[:,3].mean():.3f}")
