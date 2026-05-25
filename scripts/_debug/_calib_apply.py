"""Single source of truth for "apply a fit-json calib to a dataset inst".

Used by:
- scripts/_debug/_tss4_full_frame_stats.py
- scripts/_debug/_tss4_t19_stats.py

A fit-json contains some/all of:
  omega_deg : (3,) extrinsic R correction (deg, axis-angle)
  fx_fit, fy_fit, cx_fit, cy_fit : intrinsic
  dist_fit  : KB-N radial coefs (any length, k1..kN)
  tangential_p : (2,) Brown p1, p2 (optional)
  delta_t_m : (3,) extrinsic translation correction in m (optional)

`load_fit(path)` → FitParams (dataclass-ish dict)
`apply_fit_to_inst(inst, fit, log_tag=None)` → new inst with corrected
   T_gt / K_full / distortion / tangential_p / uv_full / z_cam.

The KB-N + tangential forward projection is `proj_kbN` — the canonical
forward used by these scripts. `pandaset_full.PandaSetCalibDatasetFull`
already supports KB-N + optional `inst['tangential_p']` so we just stuff
the arrays into the inst dict and let the dataset re-render.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Optional

import numpy as np
import torch
from scipy.spatial.transform import Rotation as _R


def proj_kbN(pts_cam: np.ndarray, K: np.ndarray,
             D: np.ndarray, p: Optional[np.ndarray] = None) -> np.ndarray:
    """KB-N forward + Brown tangential. Arbitrary len(D)."""
    X, Y, Z = pts_cam[:, 0], pts_cam[:, 1], pts_cam[:, 2]
    r = np.sqrt(X * X + Y * Y)
    theta = np.arctan2(r, np.maximum(Z, 1e-9))
    t2 = theta * theta
    poly = np.ones_like(theta)
    tp = t2.copy()
    for ki in D:
        poly = poly + ki * tp
        tp = tp * t2
    theta_d = theta * poly
    r_safe = np.where(r > 1e-9, r, 1.0)
    Xp = theta_d * X / r_safe
    Yp = theta_d * Y / r_safe
    if p is not None:
        r2p = Xp * Xp + Yp * Yp
        du_t = 2 * p[0] * Xp * Yp + p[1] * (r2p + 2 * Xp * Xp)
        dv_t = p[0] * (r2p + 2 * Yp * Yp) + 2 * p[1] * Xp * Yp
        Xp = Xp + du_t
        Yp = Yp + dv_t
    u = K[0, 0] * Xp + K[0, 2]
    v = K[1, 1] * Yp + K[1, 2]
    return np.stack([u, v], axis=-1)


class FitParams:
    """Parsed fit-json. Use load_fit() to construct."""

    def __init__(self, fj: dict):
        self.raw = fj
        self.R_om = _R.from_rotvec(np.deg2rad(np.asarray(fj['omega_deg']))).as_matrix()
        self.K_new = np.array([[fj['fx_fit'], 0.0, fj['cx_fit']],
                               [0.0, fj['fy_fit'], fj['cy_fit']],
                               [0.0, 0.0, 1.0]], dtype=np.float64)
        self.D_new = np.asarray(fj['dist_fit'], dtype=np.float64)
        self.p_new = (np.asarray(fj['tangential_p'], dtype=np.float64)
                      if 'tangential_p' in fj else None)
        self.dt_new = (np.asarray(fj['delta_t_m'], dtype=np.float64)
                       if 'delta_t_m' in fj else np.zeros(3))
        self.omega_deg = np.asarray(fj['omega_deg'], dtype=np.float64)
        self.fx_init = fj.get('fx_init', fj['fx_fit'])
        self.fy_init = fj.get('fy_init', fj['fy_fit'])

    def summary(self, tag: str = 'apply') -> list[str]:
        lines = [
            f'[{tag}]   ω={self.omega_deg.round(4).tolist()}°  '
            f'fx*={self.K_new[0,0]/self.fx_init:.4f} '
            f'fy*={self.K_new[1,1]/self.fy_init:.4f}  '
            f'k_fit(N={self.D_new.size})={self.D_new.round(4).tolist()}'
        ]
        if self.p_new is not None:
            lines.append(f'[{tag}]   tangential p={self.p_new.round(5).tolist()} APPLIED')
        if np.any(self.dt_new != 0):
            lines.append(f'[{tag}]   Δt={(self.dt_new*1000).round(1).tolist()}mm APPLIED')
        return lines


def load_fit(path: Path) -> FitParams:
    return FitParams(json.loads(Path(path).read_text()))


def apply_fit_to_inst(inst: dict, fit: FitParams) -> dict:
    """In-place style: returns the same dict with K_full / T_gt / distortion /
    tangential_p / uv_full / z_cam re-baked from `fit`. Caller is responsible
    for printing the summary once via `fit.summary()`.
    """
    pts_w = inst['pts'].numpy().astype(np.float64)
    T_gt = inst['T_gt'].numpy().astype(np.float64)

    T_new = T_gt.copy()
    T_new[:3, :3] = fit.R_om @ T_gt[:3, :3]
    T_new[:3, 3]  = fit.R_om @ T_gt[:3, 3] + fit.dt_new

    inst['T_gt']       = torch.from_numpy(T_new.astype(np.float32))
    inst['K_full']     = torch.from_numpy(fit.K_new.astype(np.float32))
    inst['distortion'] = torch.from_numpy(fit.D_new.astype(np.float32))
    if fit.p_new is not None:
        inst['tangential_p'] = torch.from_numpy(fit.p_new.astype(np.float32))

    homo = np.column_stack([pts_w, np.ones(len(pts_w))])
    pts_cam_new = (T_new @ homo.T)[:3].T
    if 'uv_full' in inst:
        uv_new = proj_kbN(pts_cam_new, fit.K_new, fit.D_new, p=fit.p_new)
        inst['uv_full'] = torch.from_numpy(uv_new.astype(np.float32))
    if 'z_cam' in inst:
        inst['z_cam'] = torch.from_numpy(pts_cam_new[:, 2].astype(np.float32))
    return inst


def make_warp_closure(path: Path, log_tag: str = 'apply'):
    """Convenience: load fit, print summary, return warp_inst(inst) closure.
    Returns (warp_inst, fit). If path is None, returns (None, None).
    """
    if path is None:
        return None, None
    fit = load_fit(path)
    print(f'[{log_tag}] APPLY fit from {Path(path).name}')
    for line in fit.summary(log_tag):
        print(line)

    def warp_inst(inst):
        return apply_fit_to_inst(inst, fit)

    return warp_inst, fit
