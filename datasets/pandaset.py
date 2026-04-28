"""PandaSet calibration dataset — object-instance based.

Cache phase (once):
    For each frame × each visible cuboid object (car, person, etc.):
        img_cache  : (3, CACHE_IMG, CACHE_IMG) uint8  — GT image crop, large-ish
        pts_world  : (N, 3) float32                   — LiDAR visible in front cam
        cam_pos    : (3,) float32
        K_full     : (3,3) float32                    — intrinsics (1920×1080)
        u0, v0     : float                            — cached crop TL in full-res px
        crop_size  : float                            — cached crop side in full-res px
    Objects are cached at bbox_scale=3.0 (bbox occupies center 1/3 of cache).
    Random crops (bg) cached at ~192 full-res px, saved at CACHE_IMG×CACHE_IMG.

__getitem__ (fresh random each call):
    1. Random sub-window in cache-image pixels: s ∈ [img_size, CACHE_IMG]
       - for obj crops, sub-window is constrained to still contain bbox
       - for random crops, sub-window is fully free
    2. Derive new (u0, v0, crop_size) in full-res coords from the sub-window
    3. Random YPR+translation perturbation on camera pose
    4. Project all pts_world with DISTORTED pose → find points in crop window
    5. 16×16 grid sample → dist_uvd (crop-local coords, [0, img_size])
    6. GT project same points → true_uvd
    7. Sub-crop cache image and bilinear-resize to img_size×img_size
"""
import json, pickle, random
import numpy as np
from pathlib import Path
from PIL import Image
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset
from scipy.spatial.transform import Rotation

CACHE_IMG = 192  # cache image resolution (square)

USEFUL_LABELS = {
    'Car', 'Pickup Truck', 'Semi-truck', 'Motorcycle',
    'Other Vehicle - Uncommon', 'Other Vehicle - Construction Vehicle',
    'Pedestrian', 'Personal Mobility Device',
}


# ── geometry helpers ─────────────────────────────────────────────────────────

def _quat_pos_to_mat(heading, position) -> np.ndarray:
    q = [heading['x'], heading['y'], heading['z'], heading['w']]
    R = Rotation.from_quat(q).as_matrix()
    M = np.eye(4); M[:3, :3] = R
    M[:3, 3] = [position['x'], position['y'], position['z']]
    return M


def _project(pts_world, pose_mat, K):
    T = np.linalg.inv(pose_mat)
    pts_cam = T[:3, :3] @ pts_world.T + T[:3, 3:]
    z = pts_cam[2]
    uv = (K @ pts_cam)[:2] / z
    return uv.T, z   # (N,2), (N,)


def _bbox2d_of_cuboid(pos, dims, yaw, pose_mat, K):
    """Project 3D cuboid corners to image, return 2D bbox (u_min,v_min,u_max,v_max) or None."""
    dx, dy, dz = dims[0]/2, dims[1]/2, dims[2]/2
    Rc = Rotation.from_euler('z', yaw).as_matrix()
    corners = np.array([[sx*dx, sy*dy, sz*dz]
                        for sx in [-1,1] for sy in [-1,1] for sz in [-1,1]])
    corners_world = (Rc @ corners.T).T + pos
    uv, z = _project(corners_world, pose_mat, K)
    front = z > 0.1
    if front.sum() < 4:
        return None
    uv = uv[front]
    return uv[:,0].min(), uv[:,1].min(), uv[:,0].max(), uv[:,1].max()


# ── cache builder ─────────────────────────────────────────────────────────────

def build_cache(pandaset_root: str,
                cache_path: str,
                val_fraction: float = 0.15,
                seed: int = 42,
                max_scenes: int = None,
                max_per_scene: int = None,
                random_crops: bool = False,
                bbox_scale: float = 3.0,
                min_pts: int = 8,
                frame_sample: float = 1.0,
                cache_img: int = CACHE_IMG):
    root   = Path(pandaset_root)
    scenes = sorted(p.name for p in root.iterdir() if p.is_dir())
    rng    = random.Random(seed); rng.shuffle(scenes)
    if max_scenes:
        scenes = scenes[:max_scenes]
    n_val = max(0, int(len(scenes) * val_fraction))

    IW, IH = 1920, 1080
    train_instances, val_instances = [], []

    for si, scene in enumerate(scenes):
        split_list = val_instances if si < n_val else train_instances
        sc_dir  = root / scene
        cam_dir = sc_dir / 'camera' / 'front_camera'
        if not cam_dir.exists():
            continue

        with open(cam_dir / 'poses.json') as f:
            all_poses = json.load(f)
        with open(cam_dir / 'intrinsics.json') as f:
            intr = json.load(f)
        K_full = np.array([[intr['fx'], 0, intr['cx']],
                           [0, intr['fy'], intr['cy']],
                           [0, 0, 1]], dtype=np.float64)

        cuboid_files = sorted((sc_dir / 'annotations' / 'cuboids').glob('*.pkl'))
        lidar_files  = sorted((sc_dir / 'lidar').glob('*.pkl'))
        n_frames     = min(len(all_poses), len(cuboid_files), len(lidar_files))

        frames = list(range(n_frames))
        if frame_sample < 1.0:
            k = max(1, round(len(frames) * frame_sample))
            frames = random.Random(seed ^ hash(scene)).sample(frames, k)
        n_before = len(split_list)
        for fi in frames:
            n_before_frame = len(split_list)
            cam_pose = all_poses[fi]
            pose_mat = _quat_pos_to_mat(cam_pose['heading'], cam_pose['position'])
            cam_pos  = np.array([cam_pose['position']['x'],
                                 cam_pose['position']['y'],
                                 cam_pose['position']['z']], dtype=np.float32)

            # load LiDAR (d=0 = Pandar64)
            lidar_df  = pickle.load(open(lidar_files[fi], 'rb'))
            lidar_df  = lidar_df[lidar_df['d'] == 0]
            pts_world = lidar_df[['x','y','z']].values.astype(np.float32)

            # keep LiDAR visible in GT front camera
            uv_gt, z_gt = _project(pts_world, pose_mat, K_full)
            vis = (z_gt > 0.5) & \
                  (uv_gt[:,0] >= 0) & (uv_gt[:,0] < IW) & \
                  (uv_gt[:,1] >= 0) & (uv_gt[:,1] < IH)
            if vis.sum() < min_pts:
                continue
            pts_vis = pts_world[vis]

            # load full-res image
            img_full = Image.open(cam_dir / f'{fi:02d}.jpg').convert('RGB')

            # load cuboid annotations for this frame
            cuboids = pickle.load(open(cuboid_files[fi], 'rb'))
            cuboids = cuboids[cuboids['label'].isin(USEFUL_LABELS)]

            for _, obj in cuboids.iterrows():
                pos  = np.array([obj['position.x'], obj['position.y'], obj['position.z']])
                dims = np.array([obj['dimensions.x'], obj['dimensions.y'], obj['dimensions.z']])
                yaw  = obj['yaw']

                bbox = _bbox2d_of_cuboid(pos, dims, yaw, pose_mat, K_full)
                if bbox is None:
                    continue
                u_min, v_min, u_max, v_max = bbox

                # object center must be inside image
                uc, vc = (u_min+u_max)/2, (v_min+v_max)/2
                if not (0 <= uc < IW and 0 <= vc < IH):
                    continue

                # square crop at bbox_scale × bbox
                bw, bh   = u_max - u_min, v_max - v_min
                crop_size = max(bw, bh) * bbox_scale
                crop_size = max(crop_size, 64)   # at least 64px in full res
                half      = crop_size / 2
                u0 = float(np.clip(uc - half, 0, IW - crop_size))
                v0 = float(np.clip(vc - half, 0, IH - crop_size))
                crop_size = float(crop_size)

                # crop and resize image to CACHE_IMG
                box = (int(u0), int(v0), int(u0+crop_size), int(v0+crop_size))
                img_cache = np.array(img_full.crop(box).resize((cache_img, cache_img), Image.BILINEAR),
                                     dtype=np.uint8)

                R_gt = Rotation.from_quat([cam_pose['heading']['x'],
                                           cam_pose['heading']['y'],
                                           cam_pose['heading']['z'],
                                           cam_pose['heading']['w']]).as_matrix().astype(np.float32)
                T_gt = np.linalg.inv(_quat_pos_to_mat(cam_pose['heading'],
                                     cam_pose['position'])).astype(np.float32)

                split_list.append({
                    'img_cache': torch.from_numpy(img_cache).permute(2,0,1),  # (3,CACHE_IMG,CACHE_IMG)
                    'pts':       torch.from_numpy(pts_vis),
                    'cam_pos':   torch.from_numpy(cam_pos),
                    'R_gt':      torch.from_numpy(R_gt),    # (3,3)
                    'T_gt':      torch.from_numpy(T_gt),    # (4,4) world→cam
                    'K_full':    torch.from_numpy(K_full.astype(np.float32)),
                    'u0': u0, 'v0': v0, 'crop_size': crop_size,
                    # 3D cuboid for accurate is_obj check
                    'obj_pos':  torch.from_numpy(pos.astype(np.float32)),   # (3,)
                    'obj_dims': torch.from_numpy(dims.astype(np.float32)),  # (3,)
                    'obj_yaw':  float(yaw),
                })

            # random crops with enough LiDAR points (same # as object instances)
            if random_crops:
                n_obj    = len(split_list) - n_before_frame
                _R_gt = Rotation.from_quat([cam_pose['heading']['x'],
                                            cam_pose['heading']['y'],
                                            cam_pose['heading']['z'],
                                            cam_pose['heading']['w']]).as_matrix().astype(np.float32)
                _T_gt = np.linalg.inv(pose_mat).astype(np.float32)
                rand_crop_size = int(IW * 0.10)
                uv_gt_all, z_gt_all = _project(pts_vis, pose_mat, K_full)
                for _ in range(n_obj * 1):
                    ru0 = rng.randint(0, IW - rand_crop_size)
                    rv0 = rng.randint(0, IH - rand_crop_size)
                    in_c = ((uv_gt_all[:,0] >= ru0) & (uv_gt_all[:,0] < ru0+rand_crop_size) &
                            (uv_gt_all[:,1] >= rv0) & (uv_gt_all[:,1] < rv0+rand_crop_size) &
                            (z_gt_all > 0.5))
                    if in_c.sum() < min_pts: continue
                    box_r = (ru0, rv0, ru0+rand_crop_size, rv0+rand_crop_size)
                    img_r = np.array(img_full.crop(box_r).resize((cache_img, cache_img), Image.BILINEAR), dtype=np.uint8)
                    split_list.append({
                        'img_cache': torch.from_numpy(img_r).permute(2,0,1),
                        'pts':      torch.from_numpy(pts_vis),
                        'cam_pos':  torch.from_numpy(cam_pos),
                        'R_gt':     torch.from_numpy(_R_gt),
                        'T_gt':     torch.from_numpy(_T_gt),
                        'K_full':   torch.from_numpy(K_full.astype(np.float32)),
                        'u0': float(ru0), 'v0': float(rv0), 'crop_size': float(rand_crop_size),
                        'obj_bbox': torch.zeros(4),
                    })

        # subsample if too many instances from this scene
        if max_per_scene and len(split_list) - n_before > max_per_scene:
            scene_instances = split_list[n_before:]
            rng.shuffle(scene_instances)          # in-place on the sublist copy
            del split_list[n_before:]
            split_list.extend(scene_instances[:max_per_scene])

        n_added = len(split_list) - n_before
        print(f"  {scene}: +{n_added} instances  "
              f"(train={len(train_instances)} val={len(val_instances)})", flush=True)

    torch.save({'train': train_instances, 'val': val_instances}, cache_path)
    print(f"Saved → {cache_path}  "
          f"train={len(train_instances)} val={len(val_instances)}")


# ── sample builder (shared with BA / eval / training) ───────────────────────

def build_sample(inst, ypr, t_delta, img_size: int = 64, min_pts: int = 8,
                 sub=None):
    """Apply a given perturbation (ypr deg, t_delta m) to a cached instance and
    produce (img_crop, true_uvd, dist_uvd, idx_in).

    ``sub`` (u_sub, v_sub, s) is a sub-window inside the cached image (in
    cache-image pixel coords, 0..CACHE_IMG). When ``None`` (default) the full
    cached crop is used — useful for reproducing a deterministic crop (BA,
    eval, hero rendering). Training __getitem__ passes a random sub.

    Returns None if the distorted crop has fewer than ``min_pts`` LiDAR points.
    """
    pts       = inst['pts'].numpy()
    cp        = inst['cam_pos'].numpy()
    R_gt      = inst['R_gt'].numpy()
    T_gt      = inst['T_gt'].numpy()
    K         = inst['K_full'].numpy()
    # Support both new (img_cache, CACHE_IMG×CACHE_IMG) and legacy (img_64, 64×64)
    img_cache = inst.get('img_cache', inst.get('img_64'))    # (3, C, C)
    C         = int(img_cache.shape[-1])

    if sub is None:
        u_sub, v_sub, s_sub = 0, 0, C
    else:
        u_sub, v_sub, s_sub = int(sub[0]), int(sub[1]), int(sub[2])

    # derive the effective full-res crop from the sub-window
    cs_per_px = float(inst['crop_size']) / C
    u0        = float(inst['u0']) + u_sub * cs_per_px
    v0        = float(inst['v0']) + v_sub * cs_per_px
    crop_size = s_sub * cs_per_px
    S         = img_size

    R_off = R_gt @ Rotation.from_euler('zyx', ypr, degrees=True).as_matrix()
    cp_off = cp + t_delta
    T_off = np.eye(4, dtype=np.float32)
    T_off[:3, :3] = R_off.T
    T_off[:3,  3] = -(R_off.T @ cp_off)

    pts_cam_off = T_off[:3, :3] @ pts.T + T_off[:3, 3:]
    z_off       = pts_cam_off[2]
    uv_off      = ((K @ pts_cam_off)[:2] / z_off).T

    cu0, cv0 = u0, v0
    in_crop = ((uv_off[:,0] >= cu0) & (uv_off[:,0] < cu0+crop_size) &
               (uv_off[:,1] >= cv0) & (uv_off[:,1] < cv0+crop_size) &
               (z_off > 0.5))
    if in_crop.sum() < min_pts:
        return None

    scale = S / crop_size
    uv_d_crop = np.stack([(uv_off[in_crop,0]-cu0) * scale,
                          (uv_off[in_crop,1]-cv0) * scale], axis=1)
    grid, cell = 16, float(S) / 16
    sel = []
    for gi in range(grid):
        for gj in range(grid):
            d2 = ((uv_d_crop[:,0]-(gj+.5)*cell)**2 +
                  (uv_d_crop[:,1]-(gi+.5)*cell)**2)
            sel.append(int(d2.argmin()))
    sel = sorted(set(sel))

    idx_in  = np.where(in_crop)[0][sel]
    pts_sel = pts[idx_in]

    pts_cam_gt = T_gt[:3, :3] @ pts_sel.T + T_gt[:3, 3:]
    uv_gt      = ((K @ pts_cam_gt)[:2] / pts_cam_gt[2]).T
    uv_gt_c  = np.stack([(uv_gt[:,0]-cu0) * scale,
                          (uv_gt[:,1]-cv0) * scale], axis=1)
    uv_off_c = uv_d_crop[sel]

    dist_m = (np.linalg.norm(pts_sel - cp, axis=1) / 100.0).astype(np.float32)

    if 'obj_pos' in inst:
        obj_pos  = inst['obj_pos'].numpy()
        obj_dims = inst['obj_dims'].numpy()
        obj_yaw  = inst['obj_yaw']
        c_y, s_y = np.cos(obj_yaw), np.sin(obj_yaw)
        R_obj = np.array([[c_y, s_y, 0],
                          [-s_y, c_y, 0],
                          [0,    0,   1]], dtype=np.float32)
        pts_local = (R_obj @ (pts_sel - obj_pos).T).T
        half = obj_dims / 2.0
        is_obj = ((np.abs(pts_local[:,0]) <= half[0]) &
                  (np.abs(pts_local[:,1]) <= half[1]) &
                  (np.abs(pts_local[:,2]) <= half[2])).astype(np.float32)
    else:
        obj_bbox = inst['obj_bbox'].numpy()
        obj_u0c  = (obj_bbox[0] - cu0) * scale
        obj_v0c  = (obj_bbox[1] - cv0) * scale
        obj_u1c  = (obj_bbox[2] - cu0) * scale
        obj_v1c  = (obj_bbox[3] - cv0) * scale
        is_obj   = ((uv_gt_c[:,0] >= obj_u0c) & (uv_gt_c[:,0] < obj_u1c) &
                    (uv_gt_c[:,1] >= obj_v0c) & (uv_gt_c[:,1] < obj_v1c)
                    ).astype(np.float32)

    true_uvd = np.concatenate([uv_gt_c.astype(np.float32),  dist_m[:,None], is_obj[:,None]], axis=1)
    dist_uvd = np.concatenate([uv_off_c.astype(np.float32), dist_m[:,None], is_obj[:,None]], axis=1)

    sub_img  = img_cache[:, v_sub:v_sub+s_sub, u_sub:u_sub+s_sub].float().unsqueeze(0)
    img_crop = F.interpolate(sub_img, size=(S, S), mode='bilinear',
                             align_corners=False).squeeze(0) / 255.0

    # virtual focal length in cropped+resampled-image pixels (= scale anchor):
    #   vfp = fx_full * S / crop_size_full_px
    # img_size=64, crop_size 128 → vfp = fx/2 (wider FOV per pixel).
    vfp = float(K[0, 0]) * S / crop_size

    return img_crop, torch.from_numpy(true_uvd), torch.from_numpy(dist_uvd), idx_in, np.float32(vfp)


# ── dataset ──────────────────────────────────────────────────────────────────

class PandaSetCalibDataset(Dataset):
    def __init__(self,
                 cache_path: str,
                 split: str = 'train',
                 img_size: int = 64,
                 max_offset_m: float = 0.20,
                 max_rot_deg: float = 0.5,
                 min_pts: int = 8,
                 max_tries: int = 20):
        data = torch.load(cache_path, weights_only=False)
        self.instances    = data[split]
        self.img_size     = img_size
        self.max_offset_m = max_offset_m
        self.max_rot_deg  = max_rot_deg
        self.min_pts      = min_pts
        self.max_tries    = max_tries

    def __len__(self): return len(self.instances)

    def _sample_sub(self, inst):
        """Random sub-window (u, v, s) in cache-image pixel coords.
        For obj crops, constrain sub to still contain the bbox (center 1/3)."""
        C = CACHE_IMG
        s = int(np.random.randint(self.img_size, C + 1))
        if 'obj_pos' in inst:
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
        inst = self.instances[idx % len(self.instances)]
        for _ in range(self.max_tries):
            t_delta = (np.random.rand(3)*2-1) * self.max_offset_m
            ypr     = (np.random.rand(3)*2-1) * self.max_rot_deg
            sub     = self._sample_sub(inst)
            out = build_sample(inst, ypr, t_delta,
                               img_size=self.img_size, min_pts=self.min_pts,
                               sub=sub)
            if out is not None:
                img_crop, true_uvd, dist_uvd, *_ = out
                return img_crop, true_uvd, dist_uvd
        return self[random.randint(0, len(self)-1)]


# ── collate ───────────────────────────────────────────────────────────────────

def collate_pandaset(batch):
    imgs, true_uvds, dist_uvds = zip(*batch)
    max_n = max(t.shape[0] for t in true_uvds)
    def pad(tensors):
        out  = torch.zeros(len(tensors), max_n, tensors[0].shape[1])
        mask = torch.ones(len(tensors), max_n, dtype=torch.bool)
        for i, t in enumerate(tensors):
            out[i,:len(t)] = t; mask[i,:len(t)] = False
        return out, mask
    true_t, _    = pad(true_uvds)
    dist_t, mask = pad(dist_uvds)
    return torch.stack(imgs), true_t, dist_t, mask


# ── smoke test ────────────────────────────────────────────────────────────────

if __name__ == '__main__':
    cache = '/tmp/pandaset_inst_cache.pt'
    if not Path(cache).exists():
        print("Building instance cache (1 scene)...")
        build_cache('/mnt/mininas/datasets/pandaset', cache,
                    val_fraction=0.0, max_scenes=1)
    ds = PandaSetCalibDataset(cache, split='train')
    print(f"instances: {len(ds)}")
    img, t, d = ds[0]
    print(f"img={img.shape}  true_uvd={t.shape}")
    print(f"shift={(t[:,:2]-d[:,:2]).norm(dim=1).mean():.2f}px")
    print(f"depth range: {t[:,2].min():.3f}–{t[:,2].max():.3f}")
