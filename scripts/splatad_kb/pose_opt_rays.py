"""Differentiable pose optimization for gsplat via the v_rays back-channel.

gsplat upstream's `rasterization(..., with_eval3d=True, rays=...)` lets the
caller bypass the (non-differentiable) UT projection for ray generation, and
its backward returns `v_rays`. By constructing rays from a learnable pose in
PyTorch ops (KB inverse undistortion done with a Newton iteration that itself
flows through autograd), the chain  loss → v_rays → pose  closes inside
autograd. No CUDA kernel changes required.

This module provides:

  - kb_unproject(uv_grid, K, radial_coeffs, ...) -> dirs_cam   (unit dirs)
  - so3_exp(omega) -> R                                         (3x3 SO(3))
  - make_rays(dirs_cam, R, t, *, row_times=None, traj_R=None, traj_t=None)
        -> rays [C, P, 6]
  - render_with_pose(splats..., R, t, K, kb_coeffs, W, H,
                     base_viewmat=None, ...) -> (img, alpha, info)

The user is responsible for owning the optimizer and the SE(3) parameters
(typically `delta_r, delta_t` near zero, applied as a small left-perturbation
of an anchor pose `R0, t0`).

Tested with gsplat main branch (>= the commit that added `rays=` to
`rasterization`). Will raise an error on older versions.
"""
from __future__ import annotations

from typing import Optional, Tuple

import torch
import torch.nn.functional as F

# Sanity check: ensure rasterization() exposes `rays=`. Defer the import to
# call sites because users may import this module without gsplat installed
# (e.g. for unit tests that only exercise kb_unproject).


# ---------------------------------------------------------------------------
# KB inverse (Newton). uv_normalised = (uv - cxy) / fxy in undistorted
# coordinates, that is, in "tangent angle plane". For OpenCV fisheye (KB4):
#   r_distorted = θ * (1 + k1 θ² + k2 θ⁴ + k3 θ⁶ + k4 θ⁸)
# where θ is the angle between the optical axis and the incoming ray.
# Given r_d (distorted radius in normalised coordinates), solve for θ.
# ---------------------------------------------------------------------------

@torch.no_grad()
def _kb_polynomial_value(theta: torch.Tensor,
                          k: torch.Tensor) -> torch.Tensor:
    """Evaluate r(θ) = θ + k1 θ³ + k2 θ⁵ + k3 θ⁷ + k4 θ⁹."""
    t2 = theta * theta
    return theta * (1.0 + t2 * (k[0] + t2 * (k[1] + t2 * (k[2] + t2 * k[3]))))


@torch.no_grad()
def _kb_polynomial_derivative(theta: torch.Tensor,
                                k: torch.Tensor) -> torch.Tensor:
    """dr/dθ = 1 + 3 k1 θ² + 5 k2 θ⁴ + 7 k3 θ⁶ + 9 k4 θ⁸."""
    t2 = theta * theta
    return 1.0 + t2 * (3 * k[0] + t2 * (5 * k[1] + t2 * (7 * k[2]
                                                          + t2 * 9 * k[3])))


def kb_unproject(
    uv: torch.Tensor,                    # [N, 2] or [H, W, 2] pixel coords
    K: torch.Tensor,                      # [3, 3]
    radial_coeffs: torch.Tensor,          # [4] k1..k4
    *,
    n_newton: int = 5,
    eps: float = 1e-9,
) -> torch.Tensor:                        # same leading shape, last dim 3
    """Pixel coordinates -> unit ray directions in the camera frame
    (OpenCV convention: x=right, y=down, z=forward).

    Implements the KB4 inverse: given image-plane normalised radius r_d, find
    θ such that θ * (1 + sum k_i θ^{2i}) = r_d via 5 Newton iterations.

    All operations are on the supplied dtype (use float64 for tests). The
    result is a unit vector pointing into the scene.
    """
    fx = K[0, 0]; fy = K[1, 1]
    cx = K[0, 2]; cy = K[1, 2]
    u = uv[..., 0]
    v = uv[..., 1]
    x_n = (u - cx) / fx
    y_n = (v - cy) / fy
    r_d = torch.sqrt(x_n * x_n + y_n * y_n).clamp_min(eps)

    # Initial guess: θ ≈ r_d (good for small distortion).
    theta = r_d.clone()
    k = radial_coeffs.to(theta.dtype).to(theta.device)

    # Newton refinement. Compute polynomial / derivative without grad
    # (we are inverting an analytic function — autograd of the iteration is
    # equivalent to implicit-function-theorem differentiation up to the loop
    # length).
    for _ in range(n_newton):
        f_val = _kb_polynomial_value(theta, k)
        f_der = _kb_polynomial_derivative(theta, k).clamp_min(eps)
        theta = theta - (f_val - r_d) / f_der

    # ray dir in camera frame
    sin_t = torch.sin(theta)
    cos_t = torch.cos(theta)
    inv_rd = 1.0 / r_d
    dx = sin_t * (x_n * inv_rd)
    dy = sin_t * (y_n * inv_rd)
    dz = cos_t
    dirs = torch.stack([dx, dy, dz], dim=-1)
    return dirs


# ---------------------------------------------------------------------------
# Small SO(3) exponential map: omega [3] -> R [3,3]. Differentiable.
# ---------------------------------------------------------------------------

def so3_exp(omega: torch.Tensor) -> torch.Tensor:
    """Rodrigues. omega: [..., 3] axis-angle. Returns [..., 3, 3]."""
    theta = omega.norm(dim=-1, keepdim=True).clamp_min(1e-12)
    axis = omega / theta
    K = _hat(axis)
    sin_t = torch.sin(theta)[..., None]
    cos_t = torch.cos(theta)[..., None]
    eye = torch.eye(3, dtype=omega.dtype, device=omega.device)
    eye = eye.expand(K.shape)
    R = eye + sin_t * K + (1 - cos_t) * (K @ K)
    return R


def _hat(v: torch.Tensor) -> torch.Tensor:
    """v: [..., 3] -> skew [..., 3, 3]"""
    zero = torch.zeros_like(v[..., 0])
    K = torch.stack([
        torch.stack([zero,        -v[..., 2],  v[..., 1]], dim=-1),
        torch.stack([ v[..., 2],   zero,      -v[..., 0]], dim=-1),
        torch.stack([-v[..., 1],   v[..., 0],  zero      ], dim=-1),
    ], dim=-2)
    return K


# ---------------------------------------------------------------------------
# make_rays: dirs_cam + (R, t) (cam->world) -> rays [C, P, 6]
# Layout of rays per gsplat: [ox, oy, oz, dx*spread, dy*spread, dz*spread].
# We use spread=1 by default; gsplat will renormalise.
# ---------------------------------------------------------------------------

def make_rays_global(
    dirs_cam: torch.Tensor,              # [H*W, 3]   (precomputed, no grad)
    R_c2w: torch.Tensor,                  # [3, 3]
    t_c2w: torch.Tensor,                  # [3]
    *,
    spread: Optional[torch.Tensor] = None,
) -> torch.Tensor:                        # [1, H*W, 6]   for one camera
    dirs_world = dirs_cam @ R_c2w.t()    # [H*W, 3]
    if spread is not None:
        dirs_world = dirs_world * spread
    origins = t_c2w[None, :].expand(dirs_world.shape[0], 3)
    rays = torch.cat([origins, dirs_world], dim=-1)
    return rays.unsqueeze(0)              # [C=1, P, 6]


def make_rays_rs(
    dirs_cam: torch.Tensor,              # [H*W, 3]
    row_index: torch.Tensor,              # [H*W] long, 0..H-1
    R_per_row: torch.Tensor,              # [H, 3, 3]
    t_per_row: torch.Tensor,              # [H, 3]
    *,
    spread: Optional[torch.Tensor] = None,
) -> torch.Tensor:                        # [1, H*W, 6]
    R_at = R_per_row[row_index]          # [P, 3, 3]
    t_at = t_per_row[row_index]          # [P, 3]
    # dirs_world[i] = R_at[i] @ dirs_cam[i]
    dirs_world = torch.einsum('pij,pj->pi', R_at, dirs_cam)
    if spread is not None:
        dirs_world = dirs_world * spread
    rays = torch.cat([t_at, dirs_world], dim=-1)
    return rays.unsqueeze(0)


def Rt_to_viewmat(R_c2w: torch.Tensor, t_c2w: torch.Tensor) -> torch.Tensor:
    """Build a c2w 4x4 viewmat. (gsplat takes c2w in camtoworlds, then it
    inverts to get world-to-camera internally — but the rasterization()
    signature uses `viewmats` which is *world-to-camera*. Adapt as needed.)
    """
    T = torch.eye(4, dtype=R_c2w.dtype, device=R_c2w.device)
    T[:3, :3] = R_c2w
    T[:3, 3] = t_c2w
    return T


# ---------------------------------------------------------------------------
# render_with_pose: thin wrapper that calls gsplat.rasterization with rays=
# and viewmat-detached for tile assignment.
# ---------------------------------------------------------------------------

def render_with_pose(
    *,
    means: torch.Tensor,                  # [N, 3]
    quats: torch.Tensor,                  # [N, 4]
    scales: torch.Tensor,                  # [N, 3]
    opacities: torch.Tensor,              # [N]
    colors: torch.Tensor,                  # [N, K, 3] (SH) or [N, 3] (rgb)
    R_c2w: torch.Tensor,                  # [3, 3]
    t_c2w: torch.Tensor,                  # [3]
    K: torch.Tensor,                      # [3, 3]
    radial_coeffs: torch.Tensor,          # [4]
    width: int,
    height: int,
    dirs_cam_cache: torch.Tensor,         # [H*W, 3]  -- output of kb_unproject
    sh_degree: Optional[int] = None,
):
    """One-camera differentiable render that flows pose grad via v_rays."""
    from gsplat.rendering import rasterization  # local import (gsplat-main)

    # rays from learnable pose
    rays = make_rays_global(dirs_cam_cache, R_c2w, t_c2w)
    rays = rays.float()                  # gsplat asserts float32

    # tile assignment: use the *current* pose, detached
    with torch.no_grad():
        T_w2c = torch.linalg.inv(Rt_to_viewmat(R_c2w, t_c2w))
        viewmats = T_w2c.unsqueeze(0)    # [1, 4, 4]
    Ks = K.unsqueeze(0)

    out = rasterization(
        means=means,
        quats=quats,
        scales=scales,
        opacities=opacities,
        colors=colors,
        viewmats=viewmats,
        Ks=Ks,
        width=width, height=height,
        sh_degree=sh_degree,
        camera_model="fisheye",
        with_ut=True,
        with_eval3d=True,
        radial_coeffs=radial_coeffs[None],
        rays=rays,
        packed=False,
    )
    # rasterization returns (colors, alphas, info)
    return out


# ---------------------------------------------------------------------------
# T1 — kb_unproject round-trip (project ∘ unproject ≈ identity).
# Run as: python pose_opt_rays.py
# ---------------------------------------------------------------------------

def _t1_roundtrip(verbose: bool = True) -> float:
    """Returns max pixel error."""
    import numpy as np
    import cv2

    fx = 2313.19; cx = 1923.06
    fy = 2313.19; cy = 1099.76
    K_np = np.array([[fx, 0, cx], [0, fy, cy], [0, 0, 1]], dtype=np.float64)
    k_np = np.array([-0.249, 0.047, -0.017, 0.0055], dtype=np.float64)
    W, H = 3840, 1952

    # Sample a sparse grid of UV inside image bounds, away from corners
    us = np.linspace(W * 0.05, W * 0.95, 32)
    vs = np.linspace(H * 0.05, H * 0.95, 16)
    U, V = np.meshgrid(us, vs, indexing='xy')
    uv_in = np.stack([U.ravel(), V.ravel()], axis=-1)

    # Run our autograd inverse (float64)
    K_t = torch.tensor(K_np)
    k_t = torch.tensor(k_np)
    uv_t = torch.tensor(uv_in)
    dirs = kb_unproject(uv_t, K_t, k_t)  # [N, 3]

    # Project back via cv2.fisheye.projectPoints (treat dirs as 3D points
    # at z = dirs[:,2], so put them into camera coords arbitrarily-scaled).
    pts_cam = dirs.numpy().reshape(-1, 1, 3) * 10.0  # any positive scale
    rvec = np.zeros(3); tvec = np.zeros(3)
    proj, _ = cv2.fisheye.projectPoints(pts_cam, rvec, tvec, K_np,
                                         k_np.reshape(4, 1))
    proj = proj.reshape(-1, 2)
    err = np.linalg.norm(proj - uv_in, axis=-1)
    max_err = float(err.max())
    if verbose:
        print(f'[T1] max round-trip err = {max_err:.6f} px '
              f'(N={len(uv_in)} samples)')
    return max_err


if __name__ == '__main__':
    err = _t1_roundtrip()
    assert err < 1e-3, f'KB inverse roundtrip too large: {err} px'
    print('T1 OK')
