"""raw frame → δ̂ pipeline for the v0.2 endpoints.

Same σ-head + closed-form GN that powers v0.1, but the input is a raw
kamikado frame (image PNG/JPG + points_V_*.txt + calib.calib JSON)
instead of an LMDB tile. The kamikado adapter + tile_cutter give us a
list of inst dicts with the same field names as the LMDB instances, so
we can reuse `_build_subwin` from eval_shared_256x800 unchanged.
"""
from __future__ import annotations

import io
import json
import time
from pathlib import Path
from typing import Iterable

import numpy as np
import torch
from PIL import Image

import scripts.eval.eval_shared_256x800 as ess
from datasets.pandaset_full import collate_full
from scripts.ba.ba_torch import solve_kb_xyz_shared, make_info_from_sigma_rho
from scripts.data.adapters.kamikado import (
    _load_calib as _kamikado_load_calib,
    _read_points_V as _kamikado_read_points,
    TILE_LAYOUT,
)
from scripts.data.calib_frame import CalibFrame
from scripts.data.tile_cutter import frame_to_tiles
from scripts.util.projection import project_lidar_into_image


_Z_MIN = 0.5
_INTENSITY_DIVISOR = 128.0  # kamikado ip664


def load_calib_bytes(calib_bytes: bytes):
    """Parse uploaded calib.calib bytes the same way the kamikado adapter does.

    Returns (K, dist4, T_SV) — vehicle→sensor transform.
    """
    from scipy.spatial.transform import Rotation
    j = json.loads(calib_bytes.decode("utf-8"))
    intr = j["calibration"]["intrinsics"]
    K = np.array(intr["camera_model"]["pinhole_parameters"]
                  ["matrix_image_camera"]["matrix"]).T
    K = np.ascontiguousarray(K, dtype=np.float64)
    dist = np.asarray(intr["distortion_model"]["generic_fisheye_parameters"]
                       ["coefficients"], dtype=np.float64)
    extr = j["calibration"]["extrinsics"]["transform_VS"]
    quat = extr["so3"]
    R_VS = Rotation.from_quat(
        [quat["x"], quat["y"], quat["z"], quat["w"]]).as_matrix()
    t_VS = np.asarray(extr["translation"]["matrix"][0], dtype=np.float64)
    T_VS = np.eye(4)
    T_VS[:3, :3] = R_VS
    T_VS[:3, 3] = t_VS
    T_SV = np.linalg.inv(T_VS)
    return K, dist, T_SV


def load_points_text(text: str) -> np.ndarray:
    """Parse uploaded points_V_*.txt bytes → (N, 4) float32 (x, y, z, intensity)."""
    arr = np.loadtxt(io.StringIO(text), comments="#",
                     usecols=(0, 1, 2, 3), dtype=np.float32)
    return np.atleast_2d(arr)


def build_calib_frame(*, img_arr: np.ndarray, pts_V: np.ndarray,
                      K: np.ndarray, dist: np.ndarray, T_SV: np.ndarray,
                      scene_id: str, frame_id: int, cam_id: str = "fcm",
                      ) -> CalibFrame:
    """Mirror of kamikado.load_frame, but with already-loaded arrays."""
    H, W = img_arr.shape[:2]
    if pts_V.size == 0:
        cf = CalibFrame(
            img=img_arr, K=K, is_fisheye=True, dist=dist,
            pts_cam=np.zeros((0, 3), np.float32),
            intensity=np.zeros((0,), np.float32),
            uv_full=np.zeros((0, 2), np.float32),
            z_cam=np.zeros((0,), np.float32),
            is_obj=np.zeros((0,), np.float32),
            scene_id=scene_id, frame_id=int(frame_id), cam_id=cam_id)
        cf.validate()
        return cf
    _, pts_cam, uv_full, z_cam, intens_raw = project_lidar_into_image(
        pts_V, K, T_SV, W, H, is_fisheye=True, dist=dist, z_min=_Z_MIN)
    intensity = np.clip(intens_raw / _INTENSITY_DIVISOR, 0.0, 1.0).astype(np.float32)
    is_obj = np.zeros(len(pts_cam), dtype=np.float32)
    cf = CalibFrame(
        img=img_arr, K=K, is_fisheye=True, dist=dist,
        pts_cam=pts_cam.astype(np.float32),
        intensity=intensity,
        uv_full=uv_full.astype(np.float32),
        z_cam=z_cam.astype(np.float32),
        is_obj=is_obj,
        scene_id=scene_id, frame_id=int(frame_id), cam_id=cam_id,
    )
    cf.validate()
    return cf


def _build_inst_subwin(ds, inst, t_delta, ypr_deg, *, u0, v0, cs):
    """`_build_subwin` works against any inst dict with the v3 cache schema —
    raw tiles from frame_to_tiles match that schema exactly. No edits needed.
    """
    return ess._build_subwin(ds, inst, t_delta, ypr_deg, u0=u0, v0=v0, cs=cs)


def solve_from_calib_frame(model, ds, cf: CalibFrame, *, ypr_target, t_target,
                            cs: int, n_per_inst: int,
                            tile_layout: dict | None = None,
                            min_pts: int = 8):
    """Slice a CalibFrame into tiles, then run σ-head + shared GN.

    Returns (delta, B, tiles_total, tiles_used).

    Pipeline:
        cf → frame_to_tiles → list[inst]
        for each inst: _build_subwin(u0, v0, cs)  (1 or 4 sub-crops)
        batch all sub-crops → model forward → make_info → shared GN
    """
    layout = dict(TILE_LAYOUT)
    if tile_layout:
        layout.update(tile_layout)

    tiles = frame_to_tiles(cf, **layout, min_pts=min_pts)
    if not tiles:
        raise RuntimeError("frame produced no tiles (check min_pts / coverage)")

    if cs == 256:
        u0v0_list = [(0, 0), (256, 0), (0, 256), (256, 256)][:n_per_inst]
    elif cs == 512:
        u0v0_list = [(0, 0)]
    else:
        raise ValueError(f"unsupported cs={cs}")

    wins = []
    for inst in tiles:
        for (u0, v0) in u0v0_list:
            w = _build_inst_subwin(ds, inst, t_target, ypr_target,
                                    u0=u0, v0=v0, cs=cs)
            if w is not None:
                wins.append(w)
    if not wins:
        raise RuntimeError(
            f"no usable sub-crops (tiles={len(tiles)}, cs={cs}, min_pts={min_pts})")

    moved = [t.to(ess.DEVICE) if torch.is_tensor(t) else t
             for t in collate_full(wins)]
    (imgs, _true_uvd, dist_uvd, pad_mask, vfp,
     bucket_uvd, bucket_valid, _,
     pts_cam_orig, duv_orig, K_orig, cs_b) = moved
    valid = ~pad_mask
    pad_full = ~valid
    B, _N = pts_cam_orig.shape[:2]
    P0_orig = pts_cam_orig.detach().clone()
    if pad_full.any():
        P0_orig[pad_full] = torch.tensor([0.0, 0.0, 1.0],
                                           dtype=P0_orig.dtype,
                                           device=P0_orig.device)
    dist_one = torch.from_numpy(cf.dist.astype(np.float32)
                                 ).reshape(1, 4).to(ess.DEVICE)
    dist = dist_one.expand(B, 4).contiguous()

    use_intensity = getattr(model, "use_intensity", True)
    if use_intensity:
        point_in = torch.cat([dist_uvd[..., :3], dist_uvd[..., 4:5]], dim=-1)
    else:
        point_in = dist_uvd[..., :3]
    img_norm = imgs.float().div(255.0)
    with torch.no_grad():
        out = model(img_norm, point_in, key_padding_mask=pad_mask, vfp=vfp,
                    bucket_uvd=bucket_uvd, bucket_valid=bucket_valid)
    per_pt = out[0] if isinstance(out, tuple) else out
    duv_pred_local = per_pt[..., :2].detach()
    if pad_full.any():
        duv_pred_local = duv_pred_local.clone()
        duv_pred_local[pad_full] = 0.0
    sx = per_pt[..., 2].exp()
    sy = per_pt[..., 3].exp()
    rho = per_pt[..., 4]
    W_sigma_local = make_info_from_sigma_rho(sx, sy, rho).detach()

    cfg = ds._cfg if hasattr(ds, "_cfg") else None  # not used; keep parity
    img_size_S = float(getattr(model, "img_size", 128))
    scale_l2o = (cs_b / img_size_S).reshape(-1, 1, 1)
    inv_l2o = (1.0 / scale_l2o).reshape(-1, 1, 1, 1)
    duv_pred_orig = duv_pred_local * scale_l2o
    W_sigma_orig = W_sigma_local * inv_l2o.pow(2)

    prior = ess.PRIOR_DIAG.to(ess.DEVICE)
    with torch.no_grad():
        delta_shared, _H = solve_kb_xyz_shared(
            P0_orig, duv_pred_orig, W_sigma_orig, K_orig, dist, ess.DOFS,
            valid=valid, n_iter=ess.BA_N_ITER, damping=ess.DAMPING,
            prior_diag=prior,
        )
    return delta_shared, int(B), len(tiles), len(wins)


def load_kamikado_frame_from_disk(scene_dir: Path | str,
                                    frame_idx: int) -> CalibFrame:
    """Convenience for the GT-demo path: load a (scene, frame) pair from
    /raw/kamikado/scenes/<scene>/."""
    from scripts.data.adapters.kamikado import load_frame as _kf_load
    return _kf_load(Path(scene_dir), int(frame_idx))
