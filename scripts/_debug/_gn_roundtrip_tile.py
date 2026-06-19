"""Controlled GN roundtrip: does the solver reproduce a KNOWN pose from correct
UV + a known perturbation, and how does it depend on tile off-axis position?

Fully synthetic (no dataset, no network). Knobs the ONE thing in question:
the unprojection (range→Z vs range-as-Z) and the tile's principal-point offset
(center tile: optical axis inside; edge tile: optical axis far outside).

Per (tile, unproject-mode):
  1. true 3D P in cam frame (controlled depth + tile FOV)
  2. range=‖P‖, uv=project(P)           ← the "sensor" obs
  3. uv_target = project(R_gt·P + t_gt)  ← perfect per-point target (known δ_gt)
  4. pts_cam = unproject(uv, range)      ← buggy or fixed
  5. GN(pts_cam, uv_target) → recover (ω,t); compare to δ_gt
fixed+exact geometry must roundtrip δ_gt to ~0; buggy must blow up on the edge
tile (range≫Z off-axis).
"""
import sys; sys.path.insert(0, '/home/hiro/git/e2e_calib')
import numpy as np, torch
from scripts.ba.ba_torch import (pinhole_jacobian, project_pinhole, gn_step,
                                  make_info_from_sigma_rho, _apply_extrinsic)

torch.set_default_dtype(torch.float64)
S = 128.0; fx = fy = 350.0
N = 300
DOF = ('omega_x', 'omega_y', 'omega_z', 'tx', 'ty', 'tz')
prior = torch.tensor([1/9., 1/9., 1/9., 1/0.09, 1/0.09, 1/0.09])
# known perturbation to recover (axis-angle deg, metres)
omega_gt = torch.tensor([[0.6, -0.4, 0.5]]); t_gt = torch.tensor([[0.10, -0.08, 0.12]])

torch.manual_seed(0)


def K_for(cx_l, cy_l):
    K = torch.zeros(1, 3, 3)
    K[0, 0, 0] = fx; K[0, 1, 1] = fy; K[0, 0, 2] = cx_l; K[0, 1, 2] = cy_l; K[0, 2, 2] = 1.0
    return K


def unproject(uv, rng, K, mode):
    cx = K[..., 0, 2:3]; cy = K[..., 1, 2:3]
    u, v = uv[..., 0], uv[..., 1]
    if mode == 'buggy':
        Z = rng; X = (u - cx) * rng / fx; Y = (v - cy) * rng / fy
    else:
        rx = (u - cx) / fx; ry = (v - cy) / fy
        Z = rng / torch.sqrt(rx*rx + ry*ry + 1.0); X = rx * Z; Y = ry * Z
    return torch.stack([X, Y, Z], -1)


def gn_solve(pts_cam, uv_target, K):
    W = make_info_from_sigma_rho(torch.full((1, N), 2.0), torch.full((1, N), 2.0), torch.zeros(1, N))
    omega = torch.zeros(1, 3); t = torch.zeros(1, 3)
    for _ in range(10):
        pts_t = _apply_extrinsic(pts_cam, omega, t)
        uv_proj = project_pinhole(pts_t, K)
        r = uv_target - uv_proj
        Xp, Yp, Zp = pts_t.unbind(-1)
        J = pinhole_jacobian(Xp, Yp, Zp, K, uv_proj.detach(), DOF)
        d, H = gn_step(J, W, r, damping=1e-3, prior_diag=prior)
        omega = omega + d[:, :3]; t = t + d[:, 3:]
    return omega, t


print(f'{"tile":10s} {"mode":6s} {"rot_err(°)":>11s} {"t_err(m)":>9s} {"range/Z mean":>13s}')
for tile, (cx_l, cy_l) in [('center', (64., 64.)), ('edge', (-200., 64.)), ('far_edge', (-450., 64.))]:
    K = K_for(cx_l, cy_l)
    # true 3D filling the tile FOV at varied depth
    u = torch.rand(1, N) * S; v = torch.rand(1, N) * S; Z = 5 + torch.rand(1, N) * 35
    X = (u - cx_l) * Z / fx; Y = (v - cy_l) * Z / fy
    P = torch.stack([X, Y, Z], -1)                       # true cam-frame 3D
    rng = torch.linalg.norm(P, dim=-1)                   # slant range ‖P‖
    uv = project_pinhole(P, K)                           # sensor uv (== u,v)
    P_pert = _apply_extrinsic(P, omega_gt, t_gt)
    uv_target = project_pinhole(P_pert, K)               # perfect target under δ_gt
    rzr = (rng / Z).mean().item()
    for mode in ['buggy', 'fixed']:
        pts_cam = unproject(uv, rng, K, mode)
        om, t = gn_solve(pts_cam, uv_target, K)
        re = (om - omega_gt).abs().mean().item(); te = (t - t_gt).abs().mean().item()
        print(f'{tile:10s} {mode:6s} {re:11.4f} {te:9.4f} {rzr:13.3f}')
