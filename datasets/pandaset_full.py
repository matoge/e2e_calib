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

# ── numpy 1.x ↔ 2.x pickle compat shim ────────────────────────────────────
# V3 cache (waymo_v3_tiled, pandaset_v3_full) was pickled on a host with
# numpy 2.x. The payload references `numpy._core.*` (private module introduced
# in numpy 2.0). When we unpickle on the training container (nvcr 24.02 →
# numpy 1.24.4, path not bumpable because torch 2.3.0a0 was built against the
# numpy 1.x ABI), unpickle fails with `ModuleNotFoundError: numpy._core`.
# Register `numpy._core` as an alias for `numpy.core` at import time so
# `torch.load(..., weights_only=False)` finds the expected callables.
# Restricted to the modules that actually appear in our cache payloads
# (multiarray / umath / numeric) — other submodules lazily resolve if/when
# a future cache version pickles new types.
import sys as _sys
try:
    import numpy.core as _np_core
    _sys.modules.setdefault("numpy._core", _np_core)
    for _sub in ("multiarray", "umath", "numeric", "_multiarray_umath",
                 "fromnumeric", "_methods"):
        try:
            _m = __import__(f"numpy.core.{_sub}", fromlist=[_sub])
            _sys.modules.setdefault(f"numpy._core.{_sub}", _m)
        except Exception:
            pass
except Exception:
    pass

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
                 max_fx_pct: float = 0.0,
                 max_fy_pct: float = 0.0,
                 pose_frame: str = 'orig',
                 min_pts: int = 8,
                 max_tries: int = 8,
                 oversample: int = 12,
                 frame_stride: int = 1,
                 grid_n: int = 16,
                 n_full: int = 1024,
                 k_per_cell: int = 8,
                 zoom_aug: bool = False):
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
        # Multiplicative fx/fy perturbation half-range (e.g. 0.02 = ±2%).
        # When non-zero, sampled δ_fx, δ_fy ∈ [-pct, +pct] are folded into the
        # projection K used for `dist_uvd` and appended to the per-sample pert
        # vec as dims 6,7 (so the optional CLS frame-pose head with n_dof=8 can
        # regress Δfx_pct, Δfy_pct alongside the 6-DoF SE3 perturbation).
        self.max_fx_pct = float(max_fx_pct)
        self.max_fy_pct = float(max_fy_pct)
        # pose_frame: 'orig' → pert_vec is in original camera frame (legacy).
        # 'vcam' → pert is converted to the tile's virtual-camera frame whose
        # optical axis is the ray through the tile center. This makes the
        # 6-DoF label crop-position-AGNOSTIC: two tiles at left vs right edges
        # with identical observations get identical labels. roll dim is left
        # as 0 in vcam mode (rotation around VCAM optical axis is mostly
        # confounded with t_x/t_y in tile-local space — explicit user advice).
        # Downstream BA aggregates per-tile (μ_vcam, Σ_vcam) via Jacobian
        # J_i = R_orig→vcam_i to recover orig-frame δ.
        assert pose_frame in ('orig', 'vcam'), f'bad pose_frame={pose_frame}'
        self.pose_frame = pose_frame
        self.min_pts   = int(min_pts)
        self.max_tries = int(max_tries)
        self.oversample = int(oversample)
        self.grid_n     = int(grid_n)
        self.n_full     = int(n_full)
        self.k_per_cell = int(k_per_cell)
        # depth-dependent zoom-in aug. When True, far pivots (z>=20m) randomly
        # shrink cs by up to scale_max(z): 1.0 at 20m → 2.0 at 100m+. This
        # synthesizes "telephoto view of distant objects" without needing
        # wider source crops — fills the high-resolution far-object regime
        # that PS / Waymo data lacks at native lens.
        self.zoom_aug  = bool(zoom_aug)

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

        # Tile-mode insts: uv_full is in PARENT-image coords + (tile_u0, tile_v0)
        # is the tile origin. Subtract once → everything downstream operates in
        # tile-local coords (0..tile_w, 0..tile_h). For full-frame insts these
        # keys are absent and we treat (0, 0) as the origin.
        tile_u0 = int(inst.get('tile_u0', 0))
        tile_v0 = int(inst.get('tile_v0', 0))
        if tile_u0 or tile_v0:
            uv_full = uv_full - np.array([tile_u0, tile_v0], dtype=np.float32)
        # in_box: 1.0 if a pt is strictly inside the tile (not in the padding
        # ring). Pivot candidates must be in_box to keep the random crop inside
        # the tile image. The padding-ring pts stay around as frustum context
        # only — if absent (legacy full-frame insts) treat all valid pts as
        # in_box.
        if 'in_box' in inst:
            in_box = inst['in_box'].numpy().astype(bool)
        else:
            in_box = np.ones(len(uv_full), dtype=bool)

        valid_in_image = ((z > 0.5) &
                          (uv_full[:,0] >= 0) & (uv_full[:,0] < IW) &
                          (uv_full[:,1] >= 0) & (uv_full[:,1] < IH))
        # Pivots: must be valid AND in_box. Frustum context still uses all pts.
        obj_idxs = np.where(is_obj_full & valid_in_image & in_box)[0]
        bg_mask  = (~is_obj_full) & valid_in_image & in_box
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
            # Depth-dependent zoom-in aug: shrink cs by up to scale_max(z) so the
            # final image has the same content but VFP-equivalent of "viewing
            # this region from `scale` x farther". scale_max(z) goes from 1.0 at
            # z<20m (no zoom) up to 2.0 at z>=100m. Far pivots only — close-up
            # zoom on near objects creates artifacts (see project_uvemb_query
            # _curriculum stage 4 design rationale).
            if getattr(self, 'zoom_aug', False) and piv_z >= 20.0:
                t = min(1.0, (piv_z - 20.0) / 80.0)         # 0 at 20m, 1 at 100m+
                scale_max = 1.0 + t * 1.0                   # 1.0 → 2.0
                # 20m → 1.0–1.2 (effectively 0.05–0.20), 100m → 1.0–2.0
                cap = 1.0 + t * (scale_max - 1.0)
                scale = float(np.random.uniform(1.0, max(1.0, scale_max)))
                cs = max(self.min_crop_px // 4, int(cs / scale))
            u0 = int(np.clip(pu - cs/2, 0, IW - cs))
            v0 = int(np.clip(pv - cs/2, 0, IH - cs))

            # Pre-filter to crop+10% padding using cached uv_full → cap to n_full
            pad_px = int(cs * 0.10)
            in_pad = ((uv_full[:, 0] >= u0 - pad_px) & (uv_full[:, 0] < u0 + cs + pad_px) &
                      (uv_full[:, 1] >= v0 - pad_px) & (uv_full[:, 1] < v0 + cs + pad_px) &
                      (z > 0.5))
            cand_idx = np.where(in_pad)[0]
            if len(cand_idx) < self.min_pts:
                continue
            if len(cand_idx) > self.n_full:
                cand_idx = np.random.choice(cand_idx, size=self.n_full, replace=False)
            pts_c = pts[cand_idx]                       # (M<=2000, 3)
            uv_gt_c = uv_full[cand_idx]                 # (M, 2) full-image px

            # Perturbation sampling:
            #   pose_frame='orig': sample (t, ypr) uniform in orig camera axes (legacy)
            #   pose_frame='vcam': sample uniform in tile's VCAM frame so labels are
            #     crop-position-AGNOSTIC (identical observations → identical labels
            #     regardless of tile location). Convert to orig for projection.
            if self.pose_frame == 'vcam':
                u_c = u0 + 0.5 * cs
                v_c = v0 + 0.5 * cs
                ray = np.array([(u_c - K[0,2]) / K[0,0],
                                 (v_c - K[1,2]) / K[1,1],
                                 1.0], dtype=np.float64)
                r_i = ray / (np.linalg.norm(ray) + 1e-12)
                z_ax = np.array([0., 0., 1.])
                axis = np.cross(r_i, z_ax)
                an = np.linalg.norm(axis)
                if an < 1e-9:
                    R_o_v = np.eye(3) if r_i[2] > 0 else -np.eye(3)
                else:
                    axis = axis / an
                    angle = float(np.arccos(np.clip(r_i @ z_ax, -1.0, 1.0)))
                    R_o_v = Rotation.from_rotvec(axis * angle).as_matrix()
                # Sample in VCAM frame: t_v ∈ R³, (yaw_v, pitch_v) ∈ R² (roll_v=0)
                t_vcam = (np.random.rand(3) * 2 - 1) * self.max_offset_m
                ypr_vcam = np.zeros(3, dtype=np.float64)
                ypr_vcam[0] = (np.random.rand() * 2 - 1) * self.max_rot_deg  # yaw
                ypr_vcam[1] = (np.random.rand() * 2 - 1) * self.max_rot_deg  # pitch
                # roll_vcam left 0 (confounded with t_x/t_y per tile)
                # Convert to orig for projection
                R_pert_vcam = Rotation.from_euler('zyx', ypr_vcam, degrees=True).as_matrix()
                R_pert_orig = R_o_v.T @ R_pert_vcam @ R_o_v
                t_delta = R_o_v.T @ t_vcam
                ypr = Rotation.from_matrix(R_pert_orig).as_euler('zyx', degrees=True)
            else:
                t_delta = (np.random.rand(3) * 2 - 1) * self.max_offset_m
                ypr     = (np.random.rand(3) * 2 - 1) * self.max_rot_deg
            R_off = R_gt @ Rotation.from_euler('zyx', ypr, degrees=True).as_matrix()
            cp_off = cp + t_delta
            # Intrinsic fx/fy multiplicative perturbation (independent — left/right
            # PS cams show fx/fy don't drift together, so train them separately).
            dfx_pct = float(np.random.uniform(-self.max_fx_pct, self.max_fx_pct)) \
                       if self.max_fx_pct > 0 else 0.0
            dfy_pct = float(np.random.uniform(-self.max_fy_pct, self.max_fy_pct)) \
                       if self.max_fy_pct > 0 else 0.0
            K_pert = K.copy()
            if dfx_pct != 0.0: K_pert[0, 0] = K[0, 0] * (1.0 + dfx_pct)
            if dfy_pct != 0.0: K_pert[1, 1] = K[1, 1] * (1.0 + dfy_pct)
            # 8-vec perturbation label for the optional CLS frame-pose head.
            # Layout: (tx, ty, tz, yaw_deg, pitch_deg, roll_deg, dfx_pct, dfy_pct)
            # — translation in meters, ypr in DEGREES, fx/fy as fractional percent.
            # pose_frame='vcam': pert_vec is in the tile's VCAM frame (sampled
            # directly above), roll_vcam=0. The model learns one frame-agnostic
            # mapping (input → VCAM 5-DoF); downstream BA aggregates per-tile
            # (μ_vcam, Σ_vcam) via J_i = R_orig→vcam_i to recover orig δ.
            if self.pose_frame == 'vcam':
                pert_vec = np.array([t_vcam[0], t_vcam[1], t_vcam[2],
                                      ypr_vcam[0], ypr_vcam[1], 0.0,
                                      dfx_pct, dfy_pct], dtype=np.float32)
            else:
                pert_vec = np.array([t_delta[0], t_delta[1], t_delta[2],
                                      ypr[0], ypr[1], ypr[2],
                                      dfx_pct, dfy_pct], dtype=np.float32)
            R_inv = R_off.T.astype(np.float32)
            t_inv = (-(R_off.T @ cp_off)).astype(np.float32)
            pts_cam_off = pts_c @ R_inv.T + t_inv       # (M, 3)
            z_off = pts_cam_off[:, 2]
            if inst.get('is_fisheye', False) and 'distortion' in inst:
                # Kannala-Brandt fisheye (e.g. ZOD). Pinhole projection would
                # diverge at the edges where theta > arctan(image_diagonal/2 fx).
                # Re-project via the same lens model the cache used at build time.
                _dist = inst['distortion'].numpy() if hasattr(inst['distortion'], 'numpy') \
                        else np.asarray(inst['distortion'], dtype=np.float32)
                _x, _y, _z = pts_cam_off[:, 0], pts_cam_off[:, 1], pts_cam_off[:, 2]
                _r = np.sqrt(_x * _x + _y * _y)
                _theta = np.arctan2(_r, np.maximum(_z, 1e-6))
                _t2 = _theta * _theta
                _td = _theta * (1.0 + _dist[0] * _t2 + _dist[1] * _t2 ** 2
                                    + _dist[2] * _t2 ** 3 + _dist[3] * _t2 ** 4)
                _r_safe = np.where(_r > 1e-9, _r, 1.0)
                _u = K_pert[0, 0] * (_td * _x / _r_safe) + K_pert[0, 2]
                _v = K_pert[1, 1] * (_td * _y / _r_safe) + K_pert[1, 2]
                uv_off_c = np.stack([_u, _v], axis=-1).astype(np.float32)
            else:
                uv_off_c = (pts_cam_off[:, :2] * (np.array([K_pert[0,0], K_pert[1,1]], dtype=np.float32))) / \
                           np.maximum(z_off[:, None], 1e-6) + np.array([K_pert[0,2], K_pert[1,2]], dtype=np.float32)
            # Tile mode: K_full is unchanged (parent coords) so the freshly-projected
            # uv_off_c lives in parent image coords. Subtract the tile origin so it
            # matches the already-tile-local uv_full / u0 / v0.
            if tile_u0 or tile_v0:
                uv_off_c = uv_off_c - np.array([tile_u0, tile_v0], dtype=np.float32)

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
                # TurboJPEG partial decode of just the crop region (~4.5ms for 384px MCU-aligned).
                # DDAD stores PNG bytes in 'jpg_bytes' — TJ chokes on those, fall back to PIL.
                blob = inst['jpg_bytes']
                is_jpeg = (len(blob) > 2 and blob[0] == 0xff and blob[1] == 0xd8)
                ju0 = (u0 // _MCU) * _MCU
                jv0 = (v0 // _MCU) * _MCU
                ju1 = min(IW, ((u0 + cs + _MCU - 1) // _MCU) * _MCU)
                jv1 = min(IH, ((v0 + cs + _MCU - 1) // _MCU) * _MCU)
                jw, jh = ju1 - ju0, jv1 - jv0
                if _HAVE_TJ and is_jpeg:
                    cropped = _TJ_INST.crop(blob, ju0, jv0, jw, jh, preserve=False)
                    arr = np.asarray(_TJ_INST.decode(cropped, pixel_format=_TJ_PF_RGB))[:jh, :jw]
                else:
                    import io
                    from PIL import Image as _PILImage
                    full = np.asarray(_PILImage.open(io.BytesIO(blob)).convert('RGB'),
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

            # Dense raw point cloud for frustum encoder per-cell context
            # (uv_off in local crop px, depth normalized /100 to match dist_uvd convention).
            uv_full_loc = np.stack([(uv_off_c[in_crop_off, 0] - u0) * scale,
                                     (uv_off_c[in_crop_off, 1] - v0) * scale], axis=1).astype(np.float32)
            d_full = (z_off[in_crop_off] / 100.0).astype(np.float32)
            uvd_full_raw = np.concatenate([uv_full_loc, d_full[:, None]], axis=1)

            # ── Bucketed cell layout: pre-bin pts into a fixed (G², K) grid so
            # the model can read 3×3 neighbor cells per query in O(K)·O(9) work
            # instead of brute-force O(Nkv) per query. K is fixed per cell so
            # collate is a vanilla stack — no per-batch ragged padding.
            G = int(self.grid_n)
            K_per_cell = int(getattr(self, 'k_per_cell', 8))
            cell_S = float(S) / G
            cu = np.clip((uv_full_loc[:, 0] / cell_S).astype(np.int32), 0, G - 1)
            cv = np.clip((uv_full_loc[:, 1] / cell_S).astype(np.int32), 0, G - 1)
            cell_id = cv * G + cu                                    # (n_raw,)
            n_raw = uvd_full_raw.shape[0]
            # Random shuffle so that "first K per cell" picks K random pts when
            # the cell is over-full. Cheap O(n_raw) permutation.
            shuf = np.random.permutation(n_raw)
            sorted_idx = shuf[np.argsort(cell_id[shuf], kind='stable')]
            sorted_uvd = uvd_full_raw[sorted_idx]                    # (n_raw, 3)
            sorted_cid = cell_id[sorted_idx]                         # (n_raw,)
            # within-cell rank: 0,1,2,... per cell. Take only those <K.
            counts = np.bincount(sorted_cid, minlength=G * G)
            cell_starts = np.zeros(G * G + 1, dtype=np.int64)
            cell_starts[1:] = counts.cumsum()
            intra = np.arange(n_raw, dtype=np.int64) - cell_starts[sorted_cid]
            keep_mask = intra < K_per_cell
            slots = intra[keep_mask]
            cells = sorted_cid[keep_mask]
            bucket_uvd  = np.zeros((G * G, K_per_cell, 3), dtype=np.float32)
            bucket_valid = np.zeros((G * G, K_per_cell), dtype=bool)
            bucket_uvd[cells, slots]  = sorted_uvd[keep_mask]
            bucket_valid[cells, slots] = True

            self._last_crop = dict(u0=int(u0), v0=int(v0), cs=int(cs),
                                    scene=inst.get('scene'), frame=int(inst.get('frame', -1)))
            return (img_crop, torch.from_numpy(true_uvd), torch.from_numpy(dist_uvd),
                    torch.tensor(vfp, dtype=torch.float32),
                    torch.from_numpy(bucket_uvd), torch.from_numpy(bucket_valid),
                    torch.from_numpy(pert_vec))

        return self[random.randint(0, len(self) - 1)]


def collate_full(batch):
    """Stack img/vfp + (G², K, 3) bucketed lidar grid + per-sample pert vec.
    Per-batch ragged padding only on the per-pivot true/dist tensors; the
    lidar bucket is fixed-size. pert_vec is per-sample (B, N_DOF) — 6-DoF SE3
    perturbation + (when N_DOF=8) Δfx_pct, Δfy_pct for the CLS frame-pose head."""
    # Tolerate older 6-tuple samples (no pert) for backward compat — pad with zeros.
    imgs    = torch.stack([s[0] for s in batch])
    trues   = [s[1] for s in batch]
    dists   = [s[2] for s in batch]
    vfps    = torch.stack([s[3] for s in batch])
    b_uvds  = torch.stack([s[4] for s in batch])
    b_valids= torch.stack([s[5] for s in batch])
    if len(batch[0]) >= 7:
        # Each sample's pert may be 6-vec (legacy) or 8-vec (with fx/fy). Pad to 8.
        v0 = batch[0][6]
        n_dof = v0.shape[-1] if hasattr(v0, 'shape') else len(v0)
        if n_dof < 8:
            pad = torch.zeros(8 - n_dof, dtype=torch.float32)
            pert_6vec = torch.stack([torch.cat([s[6], pad]) for s in batch])
        else:
            pert_6vec = torch.stack([s[6] for s in batch])     # (B, 8)
    else:
        pert_6vec = torch.zeros(len(batch), 8, dtype=torch.float32)
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
    return imgs, true_p, dist_p, pad, vfps, b_uvds, b_valids, pert_6vec
