"""Shared LiDAR→camera projection helpers.

Used by tile-cache build scripts AND CaaaS inference so the input to the
model is byte-identical between training and inference. Anything that
projects a LiDAR sweep into the camera image plane MUST go through here.
"""
from __future__ import annotations

import numpy as np


def project_kannala(pts_cam: np.ndarray, K: np.ndarray,
                     dist: np.ndarray) -> np.ndarray:
    """Kannala-Brandt forward projection (4-coeff fisheye).

    pts_cam : (N, 3) float — camera-frame XYZ
    K       : (3, 3) float — pinhole intrinsics (cx, cy, fx, fy)
    dist    : (4,)    float — k1..k4

    Returns (N, 2) float32 uv in image px. Caller must filter z > 0 and
    in-image bounds afterward.
    """
    x, y, z = pts_cam[:, 0], pts_cam[:, 1], pts_cam[:, 2]
    r = np.sqrt(x * x + y * y)
    theta = np.arctan2(r, np.maximum(z, 1e-6))
    k1, k2, k3, k4 = dist
    t2 = theta * theta
    theta_d = theta * (1.0 + k1 * t2 + k2 * t2 ** 2
                       + k3 * t2 ** 3 + k4 * t2 ** 4)
    r_safe = np.where(r > 1e-9, r, 1.0)
    fx, fy = float(K[0, 0]), float(K[1, 1])
    cx, cy = float(K[0, 2]), float(K[1, 2])
    u = fx * (theta_d * x / r_safe) + cx
    v = fy * (theta_d * y / r_safe) + cy
    return np.stack([u, v], axis=-1).astype(np.float32)


def project_pinhole(pts_cam: np.ndarray, K: np.ndarray) -> np.ndarray:
    """Standard pinhole forward projection. pts_cam (N,3), K (3,3) → uv (N,2)."""
    z = np.maximum(pts_cam[:, 2], 1e-6)
    u = pts_cam[:, 0] / z * K[0, 0] + K[0, 2]
    v = pts_cam[:, 1] / z * K[1, 1] + K[1, 2]
    return np.stack([u, v], axis=-1).astype(np.float32)


def unproject_pinhole(uv: np.ndarray, depth: np.ndarray,
                       K: np.ndarray) -> np.ndarray:
    """Reverse of project_pinhole. uv (N,2), depth (N,), K (3,3) → pts_cam (N,3).

    Used by datasets that ship pre-projected (uv, depth) tables (e.g. Waymo
    LCP) and need to recover camera-frame XYZ for downstream use.
    """
    fx, fy, cx, cy = K[0, 0], K[1, 1], K[0, 2], K[1, 2]
    xc = (uv[:, 0] - cx) * depth / fx
    yc = (uv[:, 1] - cy) * depth / fy
    return np.stack([xc, yc, depth], axis=-1).astype(np.float32)


def lidar_to_cam(pts_xyz: np.ndarray, T_cam_lidar: np.ndarray) -> np.ndarray:
    """(N, 3) LiDAR-frame XYZ → (N, 3) camera-frame XYZ via 4×4 transform."""
    N = pts_xyz.shape[0]
    h = np.column_stack([pts_xyz.astype(np.float64),
                          np.ones(N, dtype=np.float64)])
    return (T_cam_lidar.astype(np.float64) @ h.T).T[:, :3].astype(np.float32)


def project_lidar_into_image(pts_xyzi: np.ndarray, K: np.ndarray,
                               T_cam_lidar: np.ndarray, IW: int, IH: int,
                               *, is_fisheye: bool = False,
                               dist: np.ndarray | None = None,
                               z_min: float = 0.5,
                               pad_px: int = 0):
    """Full pipeline used by every tile-cache builder and by CaaaS.

    Inputs
    ------
    pts_xyzi    : (N, 4) — LiDAR-frame xyz + intensity
    K           : (3, 3)
    T_cam_lidar : (4, 4) — LiDAR→camera rigid transform
    IW, IH      : parent-image size in px
    is_fisheye  : True → use Kannala (`dist` required), else pinhole
    dist        : (4,) Kannala k1..k4 (required if is_fisheye)
    z_min       : drop points closer than this in camera frame

    Returns
    -------
    keep        : (N,) bool — points that landed in the image with z > z_min
    pts_cam     : (M, 3) — camera-frame xyz of kept points
    uv          : (M, 2) — pixel coords in parent image
    z           : (M,)   — depth in metres
    intensity   : (M,)   — raw LiDAR intensity (no normalisation; the
                           dataset/CaaaS layer normalises per-sensor)
    """
    if is_fisheye and dist is None:
        raise ValueError("is_fisheye=True requires dist (4 Kannala coeffs)")
    pts_xyz = pts_xyzi[:, :3]
    intensity_all = pts_xyzi[:, 3].astype(np.float32)
    pts_cam_all = lidar_to_cam(pts_xyz, T_cam_lidar)
    if is_fisheye:
        uv_all = project_kannala(pts_cam_all, np.asarray(K, dtype=np.float64),
                                   np.asarray(dist, dtype=np.float64))
    else:
        uv_all = project_pinhole(pts_cam_all, np.asarray(K, dtype=np.float64))
    z_all = pts_cam_all[:, 2].astype(np.float32)
    p = int(pad_px)
    keep = ((z_all > z_min)
            & (uv_all[:, 0] >= -p) & (uv_all[:, 0] < IW + p)
            & (uv_all[:, 1] >= -p) & (uv_all[:, 1] < IH + p))
    return (keep, pts_cam_all[keep], uv_all[keep],
            z_all[keep], intensity_all[keep])
