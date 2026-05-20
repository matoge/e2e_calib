"""Generate teaching figures for docs/blog/2026-05-19_diffba_library.md.

Story:
  Fig 1: 1-D linearisation curve  uv(δ) vs tangent  (ω_y axis)
  Fig 2: GN convergence  ‖δ̂ - δ_true‖ vs n_iter for 2-DoF / 6-DoF / 10-DoF
  Fig 3: KB vs pinhole  uv vs δ at FOV centre vs edge (why KB Jacobian matters)
  Fig 4: Noise robustness  emp_std vs CRLB across σ_uv ∈ {0.5, 1, 2, 5} px
  Fig 5: residual map  uv_pred − uv_target on a tile  before/after 1 GN step

Outputs land in docs/assets/2026-05-19_diffba/*.png. No /tmp.
"""
from __future__ import annotations
import sys
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))

from scipy.spatial.transform import Rotation
from scripts.ba.ba_multicam_corr import solve_dofs
from scripts.ba.ba_kb_jac import solve_dofs_kb, project_kb

OUT = REPO / 'docs' / 'assets' / '2026-05-19_diffba'
OUT.mkdir(parents=True, exist_ok=True)
_D2R = np.pi / 180.0

K_PIN = np.array([[800.0, 0.0, 320.0],
                   [0.0, 800.0, 240.0],
                   [0.0, 0.0,   1.0]], dtype=np.float64)

K_KB = np.array([[1200.0, 0.0, 1024.0],
                  [0.0, 1200.0, 768.0],
                  [0.0, 0.0,    1.0]], dtype=np.float64)
DIST_KB = np.array([-0.05, 0.01, -0.002, 0.0005], dtype=np.float64)


def _pinhole(P, K):
    fx, fy, cx, cy = K[0, 0], K[1, 1], K[0, 2], K[1, 2]
    return np.stack([fx * P[:, 0] / P[:, 2] + cx,
                     fy * P[:, 1] / P[:, 2] + cy], axis=1)


def _R(rotvec_deg):
    return Rotation.from_rotvec(np.deg2rad(rotvec_deg)).as_matrix()


def _build_par(duv):
    par = np.zeros((len(duv), 5)); par[:, 0:2] = duv
    par[:, 2] = par[:, 3] = 1.0; par[:, 4] = 0.0
    return par


# ─── Fig 1: 1-D linearisation curve  ─────────────────────────────────
def fig1_linearisation_curve():
    """For ONE point at the FOV centre and ONE point at the edge, plot
    u(ω_y) over a WIDE sweep so the non-linearity is visible, with the
    analytic tangent at ω=0 overlaid. The point of the picture is to show
    that locally (small δ) the curve and the tangent are indistinguishable
    — and to show the regime where they start to diverge."""
    pts_cam = np.array([
        [0.0, 0.0, 10.0],     # FOV centre
        [3.5, 0.0, 10.0],     # FOV edge   (X/Z = 0.35)
    ])
    omegas = np.linspace(-15.0, 15.0, 121)
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.6))
    for ax, P0, label in zip(axes, pts_cam, ['centre  (X/Z = 0.0)',
                                              'edge   (X/Z = 0.35)']):
        u_curve = []
        for w in omegas:
            P_pert = (_R([0.0, w, 0.0]) @ P0)
            u_curve.append(_pinhole(P_pert[None], K_PIN)[0, 0])
        u_curve = np.array(u_curve)
        u0 = _pinhole(P0[None], K_PIN)[0, 0]
        # analytic tangent slope ∂u/∂ω_y |_{ω=0} = (fx + fx X²/Z²) · _D2R
        X, _, Z = P0
        slope = (K_PIN[0, 0] + K_PIN[0, 0] * X * X / (Z * Z)) * _D2R
        u_lin = u0 + slope * omegas

        ax.plot(omegas, u_curve, 'b-',  lw=2, label='true  u(ω_y)')
        ax.plot(omegas, u_lin,   'r--', lw=2, label='analytic tangent at ω=0')
        ax.axvline(0.0, color='k', lw=0.5, ls=':')
        ax.scatter([0.0], [u0], c='k', s=40, zorder=5)
        # Mark the ±1° window — the regime BA actually operates in.
        ax.axvspan(-1.0, 1.0, color='orange', alpha=0.15,
                    label='BA operating window (±1°)')
        ax.set_xlabel('ω_y  [deg]')
        ax.set_ylabel('u  [px]')
        ax.set_title(f'pinhole  ·  {label}')
        ax.legend(loc='best', fontsize=9)
        ax.grid(alpha=0.3)
    fig.suptitle('Fig 1.  What the BA linearisation does:  '
                 'replace the curve by its tangent at δ=0\n'
                 '(inside the ±1° BA operating window the two are '
                 'indistinguishable)', y=1.02)
    fig.tight_layout()
    fig.savefig(OUT / 'fig1_linearisation_curve.png', dpi=140, bbox_inches='tight')
    plt.close(fig)
    print(f'  wrote {OUT / "fig1_linearisation_curve.png"}')


# ─── Fig 2: GN convergence at increasing DoF count  ──────────────────
def fig2_gn_convergence():
    """Run 1, 2, 3-step GN at 2-DoF (ω_x, ω_y), 6-DoF (extrinsic), 10-DoF
    (extrinsic+intrinsic). Plot max |δ̂ − δ_true| on a log axis. Quadratic
    convergence appears as a triangular cascade: each step squares the error."""
    rng = np.random.RandomState(7)
    Z = rng.uniform(5, 30, 2000)
    X = rng.uniform(-1, 1, 2000) * Z * 0.35
    Y = rng.uniform(-0.5, 0.5, 2000) * Z * 0.35
    P_true = np.stack([X, Y, Z], axis=1)

    delta_2  = np.array([0.10, 0.15])
    delta_6  = np.array([0.10, 0.15, 0.05, 0.02, -0.03, 0.04])
    delta_10 = np.array([0.10, 0.15, 0.05,
                          0.02, -0.03, 0.04,
                          0.01, -0.005, 1.5, -0.8])
    cases = [
        ('2-DoF  (ω_x, ω_y)',          ['omega_x', 'omega_y'],                   delta_2),
        ('6-DoF  (extrinsic)',          ['omega_x','omega_y','omega_z','tx','ty','tz'], delta_6),
        ('10-DoF (extrinsic+intrinsic)', ['omega_x','omega_y','omega_z','tx','ty','tz',
                                          'dfx','dfy','dcx','dcy'],              delta_10),
    ]

    fig, ax = plt.subplots(1, 1, figsize=(7, 4.2))
    n_iters = list(range(1, 6))
    for label, dof, delta_true in cases:
        # Apply δ_true to (P_true, K_PIN) → (P_pert, K_pert), project both.
        P_pert = P_true @ _R(delta_true[:3] if len(dof) >= 3 else
                              [delta_true[0], delta_true[1], 0.0]).T
        if len(dof) >= 6:
            P_pert = P_pert + delta_true[3:6]
        K_pert = K_PIN.copy()
        if 'dfx' in dof:
            K_pert[0, 0] *= (1.0 + delta_true[dof.index('dfx')])
            K_pert[1, 1] *= (1.0 + delta_true[dof.index('dfy')])
            K_pert[0, 2] += delta_true[dof.index('dcx')]
            K_pert[1, 2] += delta_true[dof.index('dcy')]
        uv_obs = _pinhole(P_pert, K_pert)

        # Walk from truth side toward perturbation
        errs = []
        idx = {nm: dof.index(nm) for nm in dof}
        get = lambda dc, nm: float(dc[idx[nm]]) if nm in dof else 0.0
        delta_cum = np.zeros(len(dof))
        for n in n_iters:
            P_lin = P_true @ _R([get(delta_cum, 'omega_x'),
                                  get(delta_cum, 'omega_y'),
                                  get(delta_cum, 'omega_z')]).T
            P_lin = P_lin + np.array([get(delta_cum, 'tx'),
                                       get(delta_cum, 'ty'),
                                       get(delta_cum, 'tz')])
            K_lin = K_PIN.copy()
            K_lin[0, 0] *= (1.0 + get(delta_cum, 'dfx'))
            K_lin[1, 1] *= (1.0 + get(delta_cum, 'dfy'))
            K_lin[0, 2] += get(delta_cum, 'dcx')
            K_lin[1, 2] += get(delta_cum, 'dcy')
            uv_lin = _pinhole(P_lin, K_lin)
            par = _build_par(uv_obs - uv_lin)
            step = solve_dofs(uv_lin, par, P_lin[:, 2], K_lin, dof_names=dof,
                                damping=0.0, huber_k=None, n_iter=1)
            delta_cum = delta_cum + step
            err = np.abs(np.abs(delta_cum) - np.abs(delta_true)).max()
            errs.append(err)
        ax.semilogy(n_iters, errs, '-o', label=label, lw=2)

    ax.set_xlabel('GN iterations')
    ax.set_ylabel('max |δ̂ − δ_true|   (deg / m / px / frac)')
    ax.set_title('Fig 2.  Gauss-Newton convergence on clean data  '
                 '(quadratic: each step squares the error)')
    ax.set_xticks(n_iters)
    ax.set_ylim(1e-17, 1.0)
    ax.grid(which='both', alpha=0.3)
    ax.legend(loc='best')
    fig.tight_layout()
    fig.savefig(OUT / 'fig2_gn_convergence.png', dpi=140, bbox_inches='tight')
    plt.close(fig)
    print(f'  wrote {OUT / "fig2_gn_convergence.png"}')


# ─── Fig 3: KB vs pinhole at the FOV edge  ───────────────────────────
def fig3_kb_vs_pinhole():
    """Pinhole tangent applied to a fisheye projection grossly *under*-
    estimates the response near the edge. Plot u(ω_y) for both projections
    at the FOV edge and overlay the pinhole tangent vs the KB tangent."""
    P0 = np.array([2.5, 0.0, 5.0])  # |X/Z| = 0.5  → near KB edge
    omegas = np.linspace(-2.0, 2.0, 81)

    u_pin, u_kb = [], []
    for w in omegas:
        P = _R([0.0, w, 0.0]) @ P0
        u_pin.append(_pinhole(P[None], K_KB)[0, 0])
        u_kb .append(project_kb(P[None], K_KB, DIST_KB)[0, 0])
    u_pin = np.array(u_pin); u_kb = np.array(u_kb)

    # Tangents at ω=0
    u0_pin = _pinhole(P0[None], K_KB)[0, 0]
    u0_kb  = project_kb(P0[None], K_KB, DIST_KB)[0, 0]
    X, _, Z = P0
    slope_pin = (K_KB[0, 0] + K_KB[0, 0] * X * X / (Z * Z)) * _D2R
    # Numeric KB slope (same as ba_kb_jac analytic, just for plotting)
    eps = 1e-4
    Pp = _R([0.0, eps, 0.0]) @ P0
    Pm = _R([0.0, -eps, 0.0]) @ P0
    slope_kb = (project_kb(Pp[None], K_KB, DIST_KB)[0, 0]
                - project_kb(Pm[None], K_KB, DIST_KB)[0, 0]) / (2 * eps)
    u_pin_lin = u0_pin + slope_pin * omegas
    u_kb_lin  = u0_kb  + slope_kb  * omegas

    fig, axes = plt.subplots(1, 2, figsize=(11, 4.2))
    axes[0].plot(omegas, u_pin, 'b-', lw=2, label='pinhole u(ω_y)')
    axes[0].plot(omegas, u_pin_lin, 'r--', lw=2, label='pinhole tangent')
    axes[0].set_title('Pinhole projection  ·  edge of pinhole image')
    axes[1].plot(omegas, u_kb, 'b-', lw=2, label='KB fisheye u(ω_y)')
    axes[1].plot(omegas, u_pin_lin - u0_pin + u0_kb, 'g--', lw=2,
                  label='pinhole tangent (WRONG slope here)')
    axes[1].plot(omegas, u_kb_lin, 'r--', lw=2, label='KB tangent')
    axes[1].set_title('KB fisheye projection  ·  same 3-D point')
    for ax in axes:
        ax.axvline(0.0, color='k', lw=0.5, ls=':')
        ax.set_xlabel('ω_y  [deg]')
        ax.set_ylabel('u  [px]')
        ax.legend(loc='best')
        ax.grid(alpha=0.3)
    fig.suptitle('Fig 3.  Why the KB Jacobian is needed near the FOV edge: '
                 'pinhole tangent under-estimates the true slope', y=1.02)
    fig.tight_layout()
    fig.savefig(OUT / 'fig3_kb_vs_pinhole.png', dpi=140, bbox_inches='tight')
    plt.close(fig)
    print(f'  wrote {OUT / "fig3_kb_vs_pinhole.png"}')


# ─── Fig 4: noise robustness vs CRLB  ────────────────────────────────
def fig4_noise_vs_crlb():
    """For 6-DoF pinhole, run 50 noise trials at each σ ∈ {0.5, 1, 2, 5} px
    with 3-step GN, plot empirical std(δ̂) vs CRLB √diag(H⁻¹)·σ. Empirical
    should sit on the CRLB line, demonstrating the solver hits the
    information bound."""
    rng_pts = np.random.RandomState(7)
    Z = rng_pts.uniform(5, 30, 300)
    X = (0.25 + rng_pts.uniform(-0.05, 0.05, 300)) * Z
    Y = (0.10 + rng_pts.uniform(-0.05, 0.05, 300)) * Z
    P_true = np.stack([X, Y, Z], axis=1)
    dof = ['omega_x', 'omega_y', 'omega_z', 'tx', 'ty', 'tz']
    delta_true = np.array([0.10, 0.15, 0.05, 0.02, -0.03, 0.04])
    sigmas = [0.5, 1.0, 2.0, 5.0]
    n_trials = 50

    P_pert = P_true @ _R(delta_true[:3]).T + delta_true[3:6]
    uv_pert_clean = _pinhole(P_pert, K_PIN)

    rows = []
    for sigma in sigmas:
        ests = []
        for s in range(n_trials):
            rng = np.random.RandomState(1000 + s)
            uv_obs = uv_pert_clean + rng.normal(0.0, sigma, uv_pert_clean.shape)
            delta_cum = np.zeros(6)
            for _ in range(3):
                P_lin = P_true @ _R(delta_cum[:3]).T + delta_cum[3:6]
                uv_lin = _pinhole(P_lin, K_PIN)
                par = _build_par(uv_obs - uv_lin)
                step = solve_dofs(uv_lin, par, P_lin[:, 2], K_PIN, dof_names=dof,
                                    damping=0.0, huber_k=None, n_iter=1)
                delta_cum = delta_cum + step
            ests.append(delta_cum)
        ests = np.array(ests)
        emp_std = np.std(ests, axis=0)
        crlb = np.sqrt(np.diag(solve_dofs._last_cov)) * sigma
        rows.append((sigma, emp_std, crlb))

    fig, ax = plt.subplots(1, 1, figsize=(7, 5))
    colours = ['tab:blue','tab:orange','tab:green','tab:red','tab:purple','tab:brown']
    for k, nm in enumerate(dof):
        xs = [r[2][k] for r in rows]
        ys = [r[1][k] for r in rows]
        ax.loglog(xs, ys, '-o', color=colours[k], label=nm)
    lo, hi = 1e-5, 1e-1
    ax.plot([lo, hi], [lo, hi], 'k--', lw=1, label='emp = CRLB (Cramér-Rao)')
    ax.set_xlim(lo, hi); ax.set_ylim(lo, hi)
    ax.set_xlabel('CRLB  √diag(H⁻¹) · σ_uv')
    ax.set_ylabel('empirical std(δ̂)  over 50 trials')
    ax.set_title('Fig 4.  6-DoF closed-form GN hits the information bound\n'
                 '(σ_uv ∈ {0.5, 1, 2, 5} px)')
    ax.grid(which='both', alpha=0.3)
    ax.legend(loc='best', fontsize=9, ncol=2)
    fig.tight_layout()
    fig.savefig(OUT / 'fig4_noise_vs_crlb.png', dpi=140, bbox_inches='tight')
    plt.close(fig)
    print(f'  wrote {OUT / "fig4_noise_vs_crlb.png"}')


# ─── Fig 5: residual map before / after one GN step  ────────────────
def fig5_residual_map():
    """Tile of points: show uv_pred − uv_target before and after 1 GN step,
    quivers on the image plane. The 'before' pattern is the structured Δuv
    field from a 6-DoF perturbation; the 'after' pattern is white noise
    around zero (= solver caught the structure)."""
    rng = np.random.RandomState(42)
    Z = rng.uniform(5, 30, 300)
    X = (0.25 + rng.uniform(-0.05, 0.05, 300)) * Z
    Y = (0.10 + rng.uniform(-0.05, 0.05, 300)) * Z
    P_true = np.stack([X, Y, Z], axis=1)
    delta_true = np.array([0.10, 0.15, 0.05, 0.02, -0.03, 0.04])
    P_pert = P_true @ _R(delta_true[:3]).T + delta_true[3:6]
    uv_true = _pinhole(P_true, K_PIN)
    uv_pert = _pinhole(P_pert, K_PIN)
    sigma = 0.3
    uv_obs = uv_pert + np.random.RandomState(5).normal(0, sigma, uv_pert.shape)

    dof = ['omega_x', 'omega_y', 'omega_z', 'tx', 'ty', 'tz']
    par = _build_par(uv_obs - uv_true)
    delta_hat = solve_dofs(uv_true, par, P_true[:, 2], K_PIN, dof_names=dof,
                            damping=0.0, huber_k=None, n_iter=1)
    P_corr = P_true @ _R(delta_hat[:3]).T + delta_hat[3:6]
    uv_corr = _pinhole(P_corr, K_PIN)

    # Compute residual at n_iter = 1 and n_iter = 3 (and at 0 = before).
    delta_cum = np.zeros(6)
    res_panels = [('BEFORE\n(residual = observation − truth-side projection)',
                    uv_obs - uv_true)]
    for n in range(1, 4):
        P_lin = P_true @ _R(delta_cum[:3]).T + delta_cum[3:6]
        uv_lin = _pinhole(P_lin, K_PIN)
        par = _build_par(uv_obs - uv_lin)
        step = solve_dofs(uv_lin, par, P_lin[:, 2], K_PIN, dof_names=dof,
                            damping=0.0, huber_k=None, n_iter=1)
        delta_cum = delta_cum + step
        if n in (1, 3):
            P_c = P_true @ _R(delta_cum[:3]).T + delta_cum[3:6]
            uv_c = _pinhole(P_c, K_PIN)
            res_panels.append((f'AFTER {n} GN step{"s" if n > 1 else ""}\n'
                                '(residual = observation − corrected projection)',
                                uv_obs - uv_c))

    # Tile bbox + 30 px pad for visualisation; full sensor view loses the dots.
    u_min, u_max = uv_true[:, 0].min() - 30, uv_true[:, 0].max() + 30
    v_min, v_max = uv_true[:, 1].min() - 30, uv_true[:, 1].max() + 30
    fig, axes = plt.subplots(1, 3, figsize=(16, 5))
    qscale = 0.15  # smaller = longer arrow; tuned for visibility
    noise_floor = np.sqrt(2) * sigma                   # √2·σ = iid 2-D noise RMS
    for ax, (title, res) in zip(axes, res_panels):
        ax.quiver(uv_true[:, 0], uv_true[:, 1], res[:, 0], res[:, 1],
                   angles='xy', scale_units='xy', scale=qscale, width=0.004,
                   color='tab:red', alpha=0.9)
        ax.scatter(uv_true[:, 0], uv_true[:, 1], s=6, c='k', alpha=0.5)
        ax.set_xlim(u_min, u_max); ax.set_ylim(v_max, v_min)
        ax.set_aspect('equal')
        rms = np.sqrt((res ** 2).sum(axis=1).mean())
        ax.set_title(f'{title}\n  RMS = {rms:.3f} px')
        ax.set_xlabel('u  [px]'); ax.set_ylabel('v  [px]')
        ax.grid(alpha=0.3)
    fig.suptitle(f'Fig 5.  Δuv residual field as GN iterates  '
                  f'(6-DoF, δ_true ≈ 0.1° / few cm,  σ_uv = {sigma} px '
                  f'→ noise floor √2·σ = {noise_floor:.3f} px)\n'
                  'In this small-δ regime the linearisation error O(δ²) '
                  'is below σ, so step 1 already lands at the noise floor.  '
                  'See Fig 2 for the multi-step quadratic convergence on '
                  'clean data (no noise to mask the lin. error).', y=1.06)
    fig.tight_layout()
    fig.savefig(OUT / 'fig5_residual_map.png', dpi=140, bbox_inches='tight')
    plt.close(fig)
    print(f'  wrote {OUT / "fig5_residual_map.png"}')


if __name__ == '__main__':
    print(f'Output dir: {OUT}')
    fig1_linearisation_curve()
    fig2_gn_convergence()
    fig3_kb_vs_pinhole()
    fig4_noise_vs_crlb()
    fig5_residual_map()
    print('done')
