"""V3 full-image lazy dataset.

Per __getitem__:
  - Load instance (full image + lidar + cuboids).
  - Pick random crop_size ∈ [min_crop, max_crop] full-image px.
  - Pick random (u0, v0) such that crop fits, AND crop has >= min_pts visible lidar.
  - Compute is_obj per point via 3D box-membership against ALL cuboids in inst.
  - Return (img_crop_64, true_uvd, dist_uvd) like the V1 lazy dataset.
"""
import random
from pathlib import Path
import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset
from scipy.spatial.transform import Rotation
try:
    import turbojpeg as _tj
    # PyTurboJPEG class API. One instance per worker process (not thread-safe
    # across workers, but torch DataLoader forks one process per worker so
    # each gets its own). Bench on dgx2 (Xeon 8168): PIL full decode 14.5ms,
    # TJ full decode 6.9ms, TJ crop+decode (384px) 4.5ms → 3.2× vs PIL.
    _TJ_INST = _tj.TurboJPEG()
    _TJ_PF_RGB = _tj.TJPF_RGB
    _HAVE_TJ = True
except Exception:
    _tj = None
    _TJ_INST = None
    _TJ_PF_RGB = None
    _HAVE_TJ = False
# MCU block size for JPEG; used for crop-aligned partial decode.
# 16px matches 4:2:0 chroma subsampling. Crop x,y must be multiples of
# _MCU; width/height are trimmed to image bounds by libjpeg-turbo.
_MCU = 16


def decode_inst_img(inst: dict) -> torch.Tensor:
    """Return (3, H, W) uint8 torch tensor regardless of cache schema.
    New schema: inst['jpg_bytes'] + inst['IH']/['IW'].  Legacy: inst['img']."""
    if 'jpg_bytes' in inst:
        if _HAVE_TJ:
            arr = np.asarray(_TJ_INST.decode(inst['jpg_bytes'], pixel_format=_TJ_PF_RGB))
        else:
            import io
            from PIL import Image as _PIL
            arr = np.asarray(_PIL.open(io.BytesIO(inst['jpg_bytes'])).convert('RGB'),
                             dtype=np.uint8)
        return torch.from_numpy(arr).permute(2, 0, 1).contiguous()
    return inst['img']


def _is_obj_per_point(pts_sel: np.ndarray, cuboids: list) -> np.ndarray:
    """Return (N,) float32 mask: 1 if a point lies inside ANY cuboid (yaw-rotated AABB).

    Vectorized across both points and cuboids: one einsum + two reductions,
    no Python loop over cuboids.
    """
    N = len(pts_sel)
    if not cuboids or N == 0:
        return np.zeros(N, dtype=np.float32)
    M = len(cuboids)
    poss = np.stack([np.asarray(c['pos'],  dtype=np.float32) for c in cuboids])  # (M,3)
    dims = np.stack([np.asarray(c['dims'], dtype=np.float32) for c in cuboids])  # (M,3)
    yaws = np.fromiter((float(c['yaw']) for c in cuboids), dtype=np.float32, count=M)
    cy, sy = np.cos(yaws), np.sin(yaws)
    # Per-cuboid 3x3 rotation: yaw about z, with the same convention as the loop version
    R = np.zeros((M, 3, 3), dtype=np.float32)
    R[:, 0, 0] = cy;  R[:, 0, 1] = sy
    R[:, 1, 0] = -sy; R[:, 1, 1] = cy
    R[:, 2, 2] = 1.0
    # delta = pts (N,3) - pos (M,3) → (M, N, 3); local = R @ delta.T per cuboid
    delta = pts_sel.astype(np.float32, copy=False)[None, :, :] - poss[:, None, :]
    local = np.einsum('mij,mnj->mni', R, delta)                                 # (M,N,3)
    half  = (dims * 0.5)[:, None, :]                                              # (M,1,3)
    inside = np.all(np.abs(local) <= half, axis=-1)                               # (M,N)
    return inside.any(axis=0).astype(np.float32)                                  # (N,)


class PandaSetCalibDatasetFull(Dataset):
    """Full-image V3 lazy dataset.

    Args:
        cache_dir: contains meta.pt + inst/*.pt (with full-frame images)
        split: 'train' or 'val'
        img_size: model input side (px), default 64
        min_crop_px: minimum random-crop side in full-image px
        max_crop_px: maximum random-crop side in full-image px
        max_offset_m: extrinsic translation perturbation half-range
        max_rot_deg:  extrinsic YPR perturbation half-range
        min_pts: minimum number of lidar pts to keep a sample
        max_tries: re-sample random crop up to this many times before fallback
    """
    def __init__(self,
                 cache_dir: str | Path,
                 split: str = 'train',
                 img_size: int = 64,
                 min_crop_px: int = 128,
                 max_crop_px: int = 512,
                 max_offset_m: float = 0.20,
                 max_rot_deg: float = 0.5,
                 min_pts: int = 8,
                 max_tries: int = 8,
                 oversample: int = 12,
                 frame_stride: int = 1,
                 grid_n: int = 16):
        self.cache_dir = Path(cache_dir)
        self.inst_dir  = self.cache_dir / 'inst'
        meta = torch.load(self.cache_dir / 'meta.pt', weights_only=False)
        assert split in ('train', 'val')
        self.fnames    = list(meta[split])
        if frame_stride > 1:
            self.fnames = self.fnames[::frame_stride]
        self.img_size  = int(img_size)
        self.min_crop_px = int(min_crop_px)
        self.max_crop_px = int(max_crop_px)
        self.max_offset_m = float(max_offset_m)
        self.max_rot_deg  = float(max_rot_deg)
        self.min_pts   = int(min_pts)
        self.max_tries = int(max_tries)
        self.oversample = int(oversample)
        self.grid_n     = int(grid_n)

    def __len__(self):
        return len(self.fnames) * self.oversample

    def _load_inst(self, idx: int) -> dict:
        # idx is in [0, len_fnames * oversample); modulo to wrap to file index
        return torch.load(self.inst_dir / self.fnames[idx % len(self.fnames)], weights_only=False)

    def __getitem__(self, idx: int):
        inst = self._load_inst(idx)
        # New cache stores jpg_bytes; old cache stored decoded uint8 'img'
        if 'jpg_bytes' in inst:
            IH, IW = int(inst['IH']), int(inst['IW'])
        else:
            IH, IW = int(inst['img'].shape[-2]), int(inst['img'].shape[-1])
        K = inst['K_full'].numpy()
        pts = inst['pts'].numpy()
        cp  = inst['cam_pos'].numpy()
        R_gt = inst['R_gt'].numpy()
        cubs = inst.get('cuboids', [])

        # Cached: uv_full (N,2), z_cam (N,), is_obj (N,) — computed once at build / inject time
        if 'uv_full' in inst and 'z_cam' in inst:
            uv_full = inst['uv_full'].numpy()
            z = inst['z_cam'].numpy()
        else:
            T_gt = inst['T_gt'].numpy()
            homo = np.column_stack([pts, np.ones(len(pts))])
            pts_cam_gt = (T_gt @ homo.T)[:3].T
            z = pts_cam_gt[:, 2].astype(np.float32)
            uv_full = ((K @ pts_cam_gt.T)[:2] / np.maximum(pts_cam_gt[:, 2:].T, 1e-6)).T.astype(np.float32)
        if 'is_obj' in inst:
            is_obj_full = inst['is_obj'].numpy().astype(bool)
        else:
            is_obj_full = _is_obj_per_point(pts, cubs).astype(bool)

        valid_in_image = ((z > 0.5) &
                          (uv_full[:,0] >= 0) & (uv_full[:,0] < IW) &
                          (uv_full[:,1] >= 0) & (uv_full[:,1] < IH))
        obj_idxs = np.where(is_obj_full & valid_in_image)[0]
        bg_mask  = (~is_obj_full) & valid_in_image
        # 10x5 grid for bg-pivot stratification
        GU, GV = 10, 5
        cell_w = IW / GU; cell_h = IH / GV
        cell_u = np.clip((uv_full[:, 0] / cell_w).astype(int), 0, GU-1)
        cell_v = np.clip((uv_full[:, 1] / cell_h).astype(int), 0, GV-1)
        cell_id_full = cell_v * GU + cell_u
        bg_cells = np.unique(cell_id_full[bg_mask]) if bg_mask.any() else np.array([], dtype=int)

        S = self.img_size
        # If new cache: defer JPEG decode until crop is chosen (partial decode on the
        # cropped region only). Old cache: image is already decoded as a tensor.
        img_full = None if 'jpg_bytes' in inst else inst['img']  # (3, H, W) uint8 or None

        for _ in range(self.max_tries):
            # Pick pivot first so cs can scale with depth (close pivots need
            # larger crops to capture object edges; far pivots fit fine in small).
            if len(obj_idxs) > 0 and (len(bg_cells) == 0 or np.random.rand() < 0.5):
                i = obj_idxs[np.random.randint(len(obj_idxs))]
                pu, pv = uv_full[i]
            elif len(bg_cells) > 0:
                c = bg_cells[np.random.randint(len(bg_cells))]
                idxs = np.where(bg_mask & (cell_id_full == c))[0]
                i = idxs[np.random.randint(len(idxs))]
                pu, pv = uv_full[i]
            else:
                continue
            # Depth-aware cs sampling: <20m → up to 768 px (keep whole vehicle in frame),
            # else stay narrow [min_crop_px, max_crop_px]
            piv_z = float(z[i])
            if piv_z < 20.0:
                cs_lo, cs_hi = max(self.min_crop_px, 256), min(768, IW, IH)
            else:
                cs_lo, cs_hi = self.min_crop_px, self.max_crop_px
            cs = int(np.random.randint(cs_lo, cs_hi + 1))
            cs = min(cs, IW, IH)
            u0 = int(np.clip(pu - cs/2, 0, IW - cs))
            v0 = int(np.clip(pv - cs/2, 0, IH - cs))

            # Pre-filter to crop+10% padding using cached uv_full → cap at 2000 pts
            pad_px = int(cs * 0.10)
            in_pad = ((uv_full[:, 0] >= u0 - pad_px) & (uv_full[:, 0] < u0 + cs + pad_px) &
                      (uv_full[:, 1] >= v0 - pad_px) & (uv_full[:, 1] < v0 + cs + pad_px) &
                      (z > 0.5))
            cand_idx = np.where(in_pad)[0]
            if len(cand_idx) < self.min_pts:
                continue
            if len(cand_idx) > 2000:
                cand_idx = np.random.choice(cand_idx, size=2000, replace=False)
            pts_c = pts[cand_idx]                       # (M<=2000, 3)
            uv_gt_c = uv_full[cand_idx]                 # (M, 2) full-image px

            # Project candidates with perturbed pose
            t_delta = (np.random.rand(3) * 2 - 1) * self.max_offset_m
            ypr     = (np.random.rand(3) * 2 - 1) * self.max_rot_deg
            R_off = R_gt @ Rotation.from_euler('zyx', ypr, degrees=True).as_matrix()
            cp_off = cp + t_delta
            R_inv = R_off.T.astype(np.float32)
            t_inv = (-(R_off.T @ cp_off)).astype(np.float32)
            pts_cam_off = pts_c @ R_inv.T + t_inv       # (M, 3)
            z_off = pts_cam_off[:, 2]
            uv_off_c = (pts_cam_off[:, :2] * (np.array([K[0,0], K[1,1]], dtype=np.float32))) / \
                       np.maximum(z_off[:, None], 1e-6) + np.array([K[0,2], K[1,2]], dtype=np.float32)

            in_crop_off = ((uv_off_c[:, 0] >= u0) & (uv_off_c[:, 0] < u0 + cs) &
                           (uv_off_c[:, 1] >= v0) & (uv_off_c[:, 1] < v0 + cs) &
                           (z_off > 0.5))
            if in_crop_off.sum() < self.min_pts:
                continue

            # 16x16 sub-grid representative selection — fully vectorized
            scale = S / cs
            uv_local = np.stack([(uv_off_c[in_crop_off, 0] - u0) * scale,
                                 (uv_off_c[in_crop_off, 1] - v0) * scale], axis=1)
            grid_n = self.grid_n
            cell_S = float(S) / grid_n
            ci_u = np.clip((uv_local[:, 0] / cell_S).astype(int), 0, grid_n - 1)
            ci_v = np.clip((uv_local[:, 1] / cell_S).astype(int), 0, grid_n - 1)
            cell_id = ci_v * grid_n + ci_u
            cu_c = (ci_u + 0.5) * cell_S
            cv_c = (ci_v + 0.5) * cell_S
            d2 = (uv_local[:, 0] - cu_c) ** 2 + (uv_local[:, 1] - cv_c) ** 2
            order = np.lexsort((d2, cell_id))           # primary cell_id, secondary d2
            _, first_pos = np.unique(cell_id[order], return_index=True)
            sel = order[first_pos]                        # one rep per occupied cell
            sub_idx = np.where(in_crop_off)[0][sel]      # idx into cand_idx
            pts_sel = pts_c[sub_idx]                     # (Nrep, 3)
            uv_gt_sel  = uv_gt_c[sub_idx]
            uv_off_sel = uv_off_c[sub_idx]

            uv_gt_loc  = ((uv_gt_sel  - np.array([u0, v0], dtype=np.float32)) * scale).astype(np.float32)
            uv_off_loc = ((uv_off_sel - np.array([u0, v0], dtype=np.float32)) * scale).astype(np.float32)
            dist_m = (np.linalg.norm(pts_sel - cp, axis=1) / 100.0).astype(np.float32)
            is_obj = is_obj_full[cand_idx[sub_idx]].astype(np.float32)

            true_uvd = np.concatenate([uv_gt_loc,  dist_m[:, None], is_obj[:, None]], axis=1)
            dist_uvd = np.concatenate([uv_off_loc, dist_m[:, None], is_obj[:, None]], axis=1)

            if img_full is None:
                # TurboJPEG partial decode of just the crop region.
                # PyTurboJPEG API: TurboJPEG().crop(jpeg_bytes, x, y, w, h)
                # returns a new JPEG with only the cropped region; .decode()
                # then pixel-decodes that smaller JPEG. Bench: 4.5ms for 384px
                # vs PIL full-decode 14.5ms (3.2× faster).
                ju0 = (u0 // _MCU) * _MCU
                jv0 = (v0 // _MCU) * _MCU
                ju1 = min(IW, ((u0 + cs + _MCU - 1) // _MCU) * _MCU)
                jv1 = min(IH, ((v0 + cs + _MCU - 1) // _MCU) * _MCU)
                jw, jh = ju1 - ju0, jv1 - jv0
                if _HAVE_TJ:
                    cropped = _TJ_INST.crop(inst['jpg_bytes'], ju0, jv0, jw, jh,
                                             preserve=False)
                    arr = np.asarray(_TJ_INST.decode(cropped, pixel_format=_TJ_PF_RGB))[:jh, :jw]
                else:
                    import io
                    from PIL import Image as _PILImage
                    full = np.asarray(_PILImage.open(io.BytesIO(inst['jpg_bytes'])).convert('RGB'),
                                      dtype=np.uint8)
                    arr = full[jv0:jv1, ju0:ju1]
                # Slice to exact (cs, cs) inside the MCU-padded region
                arr = arr[v0 - jv0:v0 - jv0 + cs, u0 - ju0:u0 - ju0 + cs]
                img_crop = torch.from_numpy(arr.copy()).permute(2, 0, 1).contiguous().float().unsqueeze(0)
            else:
                img_crop = img_full[:, v0:v0+cs, u0:u0+cs].float().unsqueeze(0)
            img_crop = F.interpolate(img_crop, size=(S, S), mode='bilinear',
                                      align_corners=False).squeeze(0)
            # uint8 で渡す → IPC 4x 軽量化、GPU 側で .float()/255.0 する
            img_crop = img_crop.clamp_(0, 255).to(torch.uint8)

            vfp = float(K[0, 0]) * S / cs
            self._last_crop = dict(u0=int(u0), v0=int(v0), cs=int(cs),
                                    scene=inst.get('scene'), frame=int(inst.get('frame', -1)))
            return (img_crop, torch.from_numpy(true_uvd), torch.from_numpy(dist_uvd),
                    torch.tensor(vfp, dtype=torch.float32))

        return self[random.randint(0, len(self) - 1)]


def collate_full(batch):
    """Pad ragged uvd tensors and stack img/vfp."""
    imgs, trues, dists, vfps = zip(*batch)
    imgs = torch.stack(imgs)            # (B, 3, S, S)
    vfps = torch.stack(vfps)            # (B,)
    Nmax = max(t.shape[0] for t in trues)
    B = len(trues)
    Cdim = trues[0].shape[1]
    true_p = torch.zeros(B, Nmax, Cdim)
    dist_p = torch.zeros(B, Nmax, Cdim)
    pad    = torch.ones(B, Nmax, dtype=torch.bool)
    for k, (t, d) in enumerate(zip(trues, dists)):
        n = t.shape[0]
        true_p[k, :n] = t
        dist_p[k, :n] = d
        pad[k, :n] = False
    return imgs, true_p, dist_p, pad, vfps
