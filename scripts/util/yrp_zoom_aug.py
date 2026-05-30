"""YRP + zoom homography aug for self-sup CalibNet2.

Used inside dataset.__getitem__ to fabricate a paired (orig, warped) sample
where the warp is a pure 2D image-plane similarity composed with a 3D YRP
camera rotation expressed as homography. t = 0 (no translation), so depth is
unchanged and a single frame is enough to define the analytical UV residual.

Geometry:
  H_yrp = K @ R(yaw, pitch, roll) @ K_inv          (3D rotation, t=0 case)
  H_2d  = T(pivot) @ diag(s, s, 1) @ T(-pivot)     (2D image-plane similarity)
  H     = H_2d @ H_yrp

Apply same H to image (grid_sample pull-back) and LiDAR uv (forward push).
LiDAR D is invariant (no parallax under t=0 / 2D similarity).

PoseEmb injection rule (consumer-side, not enforced here):
  - YRP (R)  → type-1 RoPE
  - zoom (s) → identity lock (focal_aug = 0)
  - 2D pivot translation → identity lock
  - 80%: zoom=1, pivot=n/a, R injected as hint
  - 20%: zoom ∈ [0.5, 2] log-uniform, pivot ∈ image, drop (R = I)

`sample_yrp_zoom_aug` returns the metadata; the dataset uses {H, R_aug,
drop_flag} downstream.
"""
from __future__ import annotations

from typing import Tuple

import numpy as np
import torch
import torch.nn.functional as F


# ---------------------------------------------------------------------------
# Homography composition (numpy, deterministic)
# ---------------------------------------------------------------------------

def _R_yrp(yaw_deg: float, pitch_deg: float, roll_deg: float) -> np.ndarray:
    """ZYX intrinsic Euler: R = R_yaw(z) @ R_pitch(y) @ R_roll(x).

    yaw is around camera-z (optical axis), pitch around camera-y, roll
    around camera-x. Sign convention: positive yaw rotates the image
    counter-clockwise when viewed from the +z (looking along optical axis).
    """
    cy, sy = np.cos(np.deg2rad(yaw_deg)), np.sin(np.deg2rad(yaw_deg))
    cp, sp = np.cos(np.deg2rad(pitch_deg)), np.sin(np.deg2rad(pitch_deg))
    cr, sr = np.cos(np.deg2rad(roll_deg)), np.sin(np.deg2rad(roll_deg))
    Rz = np.array([[cy, -sy, 0], [sy, cy, 0], [0, 0, 1]], dtype=np.float64)
    Ry = np.array([[cp, 0, sp], [0, 1, 0], [-sp, 0, cp]], dtype=np.float64)
    Rx = np.array([[1, 0, 0], [0, cr, -sr], [0, sr, cr]], dtype=np.float64)
    return Rz @ Ry @ Rx


def compose_H(yrp_deg: Tuple[float, float, float],
              zoom_s: float,
              pivot_uv: Tuple[float, float],
              K: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """Compose H = H_2d @ H_yrp.

    Args:
        yrp_deg: (yaw, pitch, roll) in degrees, ZYX intrinsic.
        zoom_s: scalar scale (>0). 1.0 = no zoom.
        pivot_uv: (u, v) in pixels. Ignored if zoom_s == 1 (still safe to pass).
        K: (3, 3) intrinsic matrix.

    Returns:
        (H, R_yrp): both (3, 3) float64. R_yrp is the 3D rotation matrix
        (for downstream PoseEmb / RoPE consumption).
    """
    K = np.asarray(K, dtype=np.float64)
    K_inv = np.linalg.inv(K)
    R_yrp = _R_yrp(*yrp_deg)
    H_yrp = K @ R_yrp @ K_inv
    pu, pv = pivot_uv
    T_neg = np.array([[1, 0, -pu], [0, 1, -pv], [0, 0, 1]], dtype=np.float64)
    T_pos = np.array([[1, 0, pu], [0, 1, pv], [0, 0, 1]], dtype=np.float64)
    S = np.diag([zoom_s, zoom_s, 1.0]).astype(np.float64)
    H_2d = T_pos @ S @ T_neg
    H = H_2d @ H_yrp
    return H, R_yrp


# ---------------------------------------------------------------------------
# Image warp (pull-back) and LiDAR uv push (forward)
# ---------------------------------------------------------------------------

def apply_H_image(img_t: torch.Tensor, H: np.ndarray,
                  *, padding_mode: str = 'border') -> torch.Tensor:
    """Pull-back warp of an image by H.

    out[u, v] = img[H_inv @ (u, v, 1)]   in homogeneous coords.

    Args:
        img_t: (3, H, W) float in [0, 1].
        H: (3, 3) homography mapping orig → warped (so we use H_inv to pull).

    Returns:
        warped (3, H, W).
    """
    _, Hh, Ww = img_t.shape
    H_t = torch.as_tensor(H, dtype=torch.float32, device=img_t.device)
    H_inv = torch.linalg.inv(H_t)
    yy, xx = torch.meshgrid(
        torch.arange(Hh, device=img_t.device, dtype=torch.float32),
        torch.arange(Ww, device=img_t.device, dtype=torch.float32),
        indexing='ij',
    )
    ones = torch.ones_like(xx)
    pix = torch.stack([xx, yy, ones], dim=-1)  # (H, W, 3)
    src = pix @ H_inv.T                         # (H, W, 3)
    src_u = src[..., 0] / src[..., 2]
    src_v = src[..., 1] / src[..., 2]
    grid_x = (src_u / (Ww - 1)) * 2 - 1
    grid_y = (src_v / (Hh - 1)) * 2 - 1
    grid = torch.stack([grid_x, grid_y], dim=-1)[None]
    out = F.grid_sample(img_t[None], grid, align_corners=True,
                        padding_mode=padding_mode)[0]
    return out


def apply_H_uv(uv: np.ndarray, H: np.ndarray) -> np.ndarray:
    """Forward warp of point uv by H. uv: (N, 2). Returns (N, 2)."""
    uv = np.asarray(uv, dtype=np.float64)
    H = np.asarray(H, dtype=np.float64)
    ones = np.ones((uv.shape[0], 1), dtype=np.float64)
    homog = np.concatenate([uv, ones], axis=1)         # (N, 3)
    warped = homog @ H.T                                # (N, 3)
    return (warped[:, :2] / warped[:, 2:3]).astype(np.float32)


# ---------------------------------------------------------------------------
# Sampler (CFG-style 80/20 hint vs drop)
# ---------------------------------------------------------------------------

def sample_yrp_zoom_aug(rng: np.random.Generator,
                        K: np.ndarray,
                        img_HW: Tuple[int, int],
                        *,
                        yrp_deg: Tuple[float, float, float] = (5.0, 5.0, 2.0),
                        zoom_log_range: Tuple[float, float] = (
                            float(np.log(0.5)), float(np.log(2.0))),
                        hint_drop_p: float = 0.2) -> dict:
    """Sample one aug: 80% pure-YRP hint / 20% YRP+zoom drop.

    Returns dict with:
      H        : (3,3) homography (image pull-back / lidar uv push)
      R_aug    : (3,3) the 3D rotation (= R_yrp), for PoseEmb / RoPE
      yrp_deg  : (yaw, pitch, roll) in deg actually drawn
      zoom_s   : scalar scale
      pivot_uv : (u, v)
      drop     : bool — if True, consumer must NOT inject PoseEmb (R=I, intrinsic 0)
    """
    Hh, Ww = img_HW
    yaw   = float(rng.uniform(-yrp_deg[0], yrp_deg[0]))
    pitch = float(rng.uniform(-yrp_deg[1], yrp_deg[1]))
    roll  = float(rng.uniform(-yrp_deg[2], yrp_deg[2]))
    is_drop = bool(rng.random() < hint_drop_p)
    if is_drop:
        zoom_s = float(np.exp(rng.uniform(*zoom_log_range)))
        pivot_uv = (float(rng.uniform(0.0, Ww)),
                    float(rng.uniform(0.0, Hh)))
    else:
        zoom_s = 1.0
        pivot_uv = (Ww / 2.0, Hh / 2.0)
    H, R_aug = compose_H((yaw, pitch, roll), zoom_s, pivot_uv, K)
    return dict(H=H, R_aug=R_aug,
                yrp_deg=(yaw, pitch, roll), zoom_s=zoom_s,
                pivot_uv=pivot_uv, drop=is_drop)
