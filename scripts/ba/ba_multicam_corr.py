"""Pinhole 外部パラメータ BA の numpy 参照実装 (ヤコビアン + 1 ステップ GN)。

学習・推論の GN は scripts/ba/ba_torch.py (torch) で、これは tests/ba/ の
一致テストの参照として残している。

以前はこのファイルにスライディングタイル推論 (build_model / load_frame /
build_bucket / infer_tiles / render_before_after / main) が同居していて、
学習とは別の推論経路になっていた。2026-10-07 に推論を
scripts/inference/infer_calib.py の 1 本に寄せたので削った。
"""
import numpy as np
from scipy.spatial.transform import Rotation

# ── BA DoF library: name → (du(X,Y,Z,uv,K), dv(X,Y,Z,uv,K)) ───────────────
# All angles in DEGREES (d2r applied in solver), translations in METERS, fx
# in PIXELS. Sign convention: positive delta = correction to APPLY on top of
# the declared world→cam SE(3) / intrinsics so the projection lines up.
# Keep the legacy 3-DoF aliases ('df_common' under both 'dfx' and 'df_common'
# names) for back-compat with existing configs.
_D2R = np.pi / 180.0


def _jac_omega_x(X, Y, Z, uv, K):
    fx, fy = K[0, 0], K[1, 1]
    return -(fx * X * Y) / (Z * Z) * _D2R, (-fy - (fy * Y * Y) / (Z * Z)) * _D2R


def _jac_omega_y(X, Y, Z, uv, K):
    fx, fy = K[0, 0], K[1, 1]
    return (fx + (fx * X * X) / (Z * Z)) * _D2R, (fy * X * Y) / (Z * Z) * _D2R


def _jac_omega_z(X, Y, Z, uv, K):
    # Rotation about cam optical axis. Cross-couples with center offset but
    # PandaSet/Waymo lens roll is usually <0.01°.
    fx, fy = K[0, 0], K[1, 1]
    return -fx * Y / Z * _D2R, fy * X / Z * _D2R


def _jac_tx(X, Y, Z, uv, K):
    return K[0, 0] / Z, np.zeros_like(Z)


def _jac_ty(X, Y, Z, uv, K):
    return np.zeros_like(Z), K[1, 1] / Z


def _jac_tz(X, Y, Z, uv, K):
    # Mostly coupled with depth scale and dfx; include only when you really
    # mean a longitudinal mount-position bias (rare on automotive rigs).
    return -K[0, 0] * X / (Z * Z), -K[1, 1] * Y / (Z * Z)


def _jac_dfx(X, Y, Z, uv, K):
    # dfx is a FRACTIONAL intrinsic perturbation: fx_new = fx · (1 + dfx).
    # Then u_new = (u - cx)·(1 + dfx) + cx, so ∂u/∂dfx = (u - cx).
    # Earlier version divided by fx; that was a sign of mixed conventions
    # (absolute-px vs fractional). Production samples dfx as a fraction
    # (max_fx_pct), so the Jacobian here must match that.
    cx = K[0, 2]
    return (uv[:, 0] - cx), np.zeros_like(Z)


def _jac_dfy(X, Y, Z, uv, K):
    cy = K[1, 2]
    return np.zeros_like(Z), (uv[:, 1] - cy)


def _jac_dcx(X, Y, Z, uv, K):
    # ∂u/∂(Δcx) = 1, ∂v/∂(Δcx) = 0. cx perturbation is in pixels.
    return np.ones_like(Z), np.zeros_like(Z)


def _jac_dcy(X, Y, Z, uv, K):
    # ∂u/∂(Δcy) = 0, ∂v/∂(Δcy) = 1. cy perturbation is in pixels.
    return np.zeros_like(Z), np.ones_like(Z)


def _jac_df_common(X, Y, Z, uv, K):
    # Legacy 3-DoF "Δfx" — actually a common Δf in px applied symmetrically
    # to fx and fy (cameras with fixed aspect). Kept for back-compat with
    # the 3-DoF result tables in memory project_ps_calib_full_picture.
    fx, fy, cx, cy = K[0, 0], K[1, 1], K[0, 2], K[1, 2]
    return (uv[:, 0] - cx) / fx, (uv[:, 1] - cy) / fy


DOF_JAC = {
    "omega_x":   _jac_omega_x,
    "omega_y":   _jac_omega_y,
    "omega_z":   _jac_omega_z,
    "tx":        _jac_tx,
    "ty":        _jac_ty,
    "tz":        _jac_tz,
    "dfx":       _jac_dfx,
    "dfy":       _jac_dfy,
    "dcx":       _jac_dcx,
    "dcy":       _jac_dcy,
    "df_common": _jac_df_common,
}

# Default DoF lists for the legacy `param` string aliases.
_DOF_PRESETS = {
    "3dof": ["omega_x", "omega_y", "df_common"],
    "5dof": ["omega_x", "omega_y", "tx", "ty", "df_common"],
    "6dof": ["omega_x", "omega_y", "omega_z", "tx", "ty", "df_common"],
    "7dof": ["omega_x", "omega_y", "omega_z", "tx", "ty", "tz", "df_common"],
    # Pure extrinsic 6-DoF (no intrinsic): for the CaaaS sequence endpoint.
    "6dof_ext": ["omega_x", "omega_y", "omega_z", "tx", "ty", "tz"],
}


def resolve_dof_list(ba_cfg: dict) -> list:
    """ba.dof (explicit list) wins; else ba.param string preset; else 3dof."""
    if isinstance(ba_cfg.get("dof"), list) and ba_cfg["dof"]:
        return list(ba_cfg["dof"])
    return list(_DOF_PRESETS.get(ba_cfg.get("param", "3dof"),
                                  _DOF_PRESETS["3dof"]))


def solve_dofs(uv: np.ndarray, par: np.ndarray, z: np.ndarray, K: np.ndarray,
                dof_names: list, damping: float = 1e-3,
                huber_k: float | None = None, n_iter: int = 1):
    """Linearized GN over a config-supplied list of DoFs. Returns δ of
    len(dof_names) in declaration order.

    If `huber_k` is set, runs IRLS for `n_iter` iterations with a Huber
    M-estimator on the per-point Mahalanobis distance — same H = ΣJᵀWJ
    pipeline, just with W_i scaled by w(d_i) = min(1, k/d_i) where
    d_i² = r_iᵀ W_i r_i (post-correction residual). Plain 1-step (no
    outlier handling) when huber_k is None.
    """
    for name in dof_names:
        if name not in DOF_JAC:
            raise KeyError(f"unknown DoF '{name}' — valid: {sorted(DOF_JAC)}")
    fx, fy, cx, cy = K[0, 0], K[1, 1], K[0, 2], K[1, 2]
    X = (uv[:, 0] - cx) * z / fx
    Y = (uv[:, 1] - cy) * z / fy
    Z = z
    Jus, Jvs = [], []
    for name in dof_names:
        ju, jv = DOF_JAC[name](X, Y, Z, uv, K)
        Jus.append(np.broadcast_to(ju, Z.shape))
        Jvs.append(np.broadcast_to(jv, Z.shape))
    J_u = np.column_stack(Jus)
    J_v = np.column_stack(Jvs)
    r_u0, r_v0 = par[:, 0], par[:, 1]
    su, sv, rho = par[:, 2], par[:, 3], par[:, 4]
    det = su * su * sv * sv * (1 - rho * rho)
    Wuu0 = (sv * sv) / det
    Wvv0 = (su * su) / det
    Wuv0 = -(rho * su * sv) / det
    n = len(dof_names)

    delta = np.zeros(n)
    weights = np.ones_like(r_u0)
    iters = max(1, int(n_iter)) if huber_k is not None else 1
    for it in range(iters):
        Wuu = Wuu0 * weights
        Wvv = Wvv0 * weights
        Wuv = Wuv0 * weights
        H = np.zeros((n, n)); b = np.zeros(n)
        for i in range(n):
            for j in range(n):
                H[i, j] = ((J_u[:, i] * Wuu * J_u[:, j]).sum()
                           + (J_v[:, i] * Wvv * J_v[:, j]).sum()
                           + (J_u[:, i] * Wuv * J_v[:, j]).sum()
                           + (J_v[:, i] * Wuv * J_u[:, j]).sum())
            b[i] = ((J_u[:, i] * Wuu * r_u0).sum()
                    + (J_v[:, i] * Wvv * r_v0).sum()
                    + (J_u[:, i] * Wuv * r_v0).sum()
                    + (J_v[:, i] * Wuv * r_u0).sum())
        H += damping * np.eye(n)
        delta = np.linalg.solve(H, b)
        if huber_k is None:
            break
        # Post-correction residuals & Mahalanobis distance per point
        ru = r_u0 - J_u @ delta
        rv = r_v0 - J_v @ delta
        d2 = (ru * Wuu0 * ru) + (rv * Wvv0 * rv) + 2.0 * (ru * Wuv0 * rv)
        d = np.sqrt(np.maximum(d2, 1e-12))
        weights = np.where(d <= huber_k, 1.0, huber_k / d)
    # Cov(δ) = H^{-1}; useful for downstream (CaaaS, sequence-fuse).
    solve_dofs._last_cov = np.linalg.inv(H)
    solve_dofs._last_H = H
    solve_dofs._last_b = b
    solve_dofs._last_weights = weights
    return delta


# Back-compat thin wrappers — keep old call sites working.
def solve_3dof(uv, par, z, K, damping=1e-3):
    return solve_dofs(uv, par, z, K, _DOF_PRESETS["3dof"], damping)


def solve_5dof(uv, par, z, K, damping=1e-3):
    return solve_dofs(uv, par, z, K, _DOF_PRESETS["5dof"], damping)


def delta_to_dict(delta: np.ndarray, dof_names: list) -> dict:
    """Map solver output → human-readable dict keyed by DoF name with
    unit-aware values. Angles → degrees, translations → meters, fx → px."""
    out = {}
    for i, name in enumerate(dof_names):
        out[name] = float(delta[i])
    return out


def make_T_corr_from_dofs(dof_vals: dict) -> np.ndarray:
    """Build SE(3) T_corr from the DoF-value dict. Rotation = ZYX Euler in
    degrees (omega_z, omega_y, omega_x); translation in meters."""
    ox = dof_vals.get("omega_x", 0.0)
    oy = dof_vals.get("omega_y", 0.0)
    oz = dof_vals.get("omega_z", 0.0)
    # Compose in same order make_T_corr did for back-compat: xy Euler when
    # no z. With z present, use ZYX standard right-handed.
    if abs(oz) < 1e-12:
        R = Rotation.from_euler("xy", [ox, oy], degrees=True).as_matrix()
    else:
        R = Rotation.from_euler("zyx", [oz, oy, ox], degrees=True).as_matrix()
    T = np.eye(4); T[:3, :3] = R
    T[0, 3] = dof_vals.get("tx", 0.0)
    T[1, 3] = dof_vals.get("ty", 0.0)
    T[2, 3] = dof_vals.get("tz", 0.0)
    return T


def make_T_corr(ox_deg: float, oy_deg: float,
                tx_m: float = 0.0, ty_m: float = 0.0) -> np.ndarray:
    """Legacy positional API kept for callers outside this module."""
    return make_T_corr_from_dofs({
        "omega_x": ox_deg, "omega_y": oy_deg,
        "tx": tx_m, "ty": ty_m,
    })
