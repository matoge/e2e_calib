"""Check whether the overfit'd network learned a 'BA-consistent' Δuv field.

Claim: pose loss only constrains the 6-D row space of J (= the projection of
Δuv onto the BA Jacobian column space). The remaining (2N - 6) directions
are null space — anything in there is invisible to the loss.

Decomposition (per-DoF, at the linearisation point used by BA):
    J·δ_target ∈ col(J)        ← what BA "sees" as the right answer
    Δuv_oracle = J·δ_target + ε_proj_residual    (KB non-linearity at δ ≈ 1°)
    Δuv_learned = J·δ̂_proj  + ε_null
                = (BA-equivalent component) + (null-space component)

If the network is BA-consistent, the SAME J·δ̂_proj must reproduce the
loss-equivalent solution; ε_null can be arbitrarily large yet loss → 0.

Output:
    docs/assets/2026-05-19_diffba/overfit_nullspace.png
    Console: norms in row vs null space.
"""
from __future__ import annotations
import sys
from pathlib import Path
import numpy as np
import torch
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from scipy.spatial.transform import Rotation

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))

# Re-import the overfit module to share definitions (TileToTokenHead,
# build_W_from_L, gather_token, DELTA_PERT, DOF, LOSS_SCALE, …).
import importlib.util, sys as _s
spec = importlib.util.spec_from_file_location(
    'overfit', str(REPO / 'scripts' / '_debug' / 'overfit_one_image_ba.py'))
m = importlib.util.module_from_spec(spec); _s.modules['overfit'] = m
spec.loader.exec_module(m)

from scripts.data.adapters.kamikado import load_frame
from scripts.ba.ba_kb_jac import project_kb
from scripts.ba.ba_torch import solve_kb, kb_jacobian, project_kb as project_kb_t

OUT = REPO / 'docs' / 'assets' / '2026-05-19_diffba'

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def train(model, img_t, uv_p_t, uv_l_t, z_t, K_t, dist_t,
          delta_target, loss_scale, n_iter=200, lr=1e-3):
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    for it in range(n_iter + 1):
        opt.zero_grad()
        duv_map, L_map = model(img_t)
        duv = m.gather_token(duv_map, uv_l_t, m.TILE)
        L_raw = m.gather_token(L_map.contiguous(), uv_l_t, m.TILE)
        W = m.build_W_from_L(L_raw)
        delta_hat, _ = solve_kb(
            uv_p_t.unsqueeze(0), duv.unsqueeze(0), W.unsqueeze(0),
            z_t.unsqueeze(0), K_t, dist_t, m.DOF,
            n_iter=2, damping=1e-3,
        )
        delta_hat = delta_hat[0]
        loss = (((delta_hat - delta_target) * loss_scale) ** 2).mean()
        loss.backward(); opt.step()
    return model, duv.detach(), W.detach(), delta_hat.detach()


def main():
    torch.manual_seed(7); np.random.seed(7)

    # ─── Reproduce the overfit setup verbatim ─────────────────────────
    cf = load_frame(m.SCENE, m.FRAME)
    img_full = cf.img.astype(np.float32) / 255.0
    H_full, W_full = img_full.shape[:2]
    K = cf.K.astype(np.float64); dist = np.asarray(cf.dist, np.float64)
    pts_cam = cf.pts_cam.astype(np.float64)
    cu, cv = W_full // 2, H_full // 2 + m.DV
    u0 = cu - m.TILE // 2; v0 = cv - m.TILE // 2
    u1 = u0 + m.TILE;       v1 = v0 + m.TILE
    R_pert = Rotation.from_rotvec(np.deg2rad(m.DELTA_PERT[:3])).as_matrix()
    pts_cam_pert = pts_cam @ R_pert.T + m.DELTA_PERT[3:6]
    uv_pert_all = project_kb(pts_cam_pert, K, dist)
    z_pert_all  = pts_cam_pert[:, 2]
    in_tile = ((uv_pert_all[:, 0] >= u0) & (uv_pert_all[:, 0] < u1)
                & (uv_pert_all[:, 1] >= v0) & (uv_pert_all[:, 1] < v1)
                & (z_pert_all > 0.5))
    uv_pert = uv_pert_all[in_tile]; z_pert = z_pert_all[in_tile]
    pts_cam_in = pts_cam[in_tile]
    uv_local = uv_pert.copy(); uv_local[:, 0] -= u0; uv_local[:, 1] -= v0

    img_t  = torch.from_numpy(img_full[v0:v1, u0:u1].copy()).permute(2,0,1).unsqueeze(0).to(DEVICE)
    uv_p_t = torch.from_numpy(uv_pert).float().to(DEVICE)
    uv_l_t = torch.from_numpy(uv_local).float().to(DEVICE)
    z_t    = torch.from_numpy(z_pert).float().to(DEVICE)
    K_t    = torch.from_numpy(K).float().unsqueeze(0).to(DEVICE)
    dist_t = torch.from_numpy(dist).float().unsqueeze(0).to(DEVICE)
    delta_target = torch.tensor(-m.DELTA_PERT, dtype=torch.float32, device=DEVICE)
    loss_scale = m.LOSS_SCALE.to(DEVICE)

    model = m.TileToTokenHead(d=96, n_blocks=2).to(DEVICE)
    model, duv_learned, W_learned, delta_hat = train(
        model, img_t, uv_p_t, uv_l_t, z_t, K_t, dist_t,
        delta_target, loss_scale, n_iter=200, lr=1e-3)
    print(f'final |δ̂ − target|_max = '
           f'{(delta_hat - delta_target).abs().max().item():.4f}')

    # ─── Build the BA Jacobian J at the linearisation point used inside
    #     the solver. solve_kb does 2 iterations; for our analysis we
    #     evaluate J at the FINAL linearisation point (= δ_lin = δ̂_step1).
    #     For loss = 0, J·(δ_target − δ̂) ≈ 0, so all that matters for the
    #     decomposition is J(δ̂_final) itself.
    fx = K[0,0]; fy = K[1,1]; cx = K[0,2]; cy = K[1,2]
    X0 = (uv_pert[:, 0] - cx) * z_pert / fx
    Y0 = (uv_pert[:, 1] - cy) * z_pert / fy
    Z0 = z_pert
    delta_hat_np = delta_hat.cpu().numpy().astype(np.float64)
    R_hat = Rotation.from_rotvec(np.deg2rad(delta_hat_np[:3])).as_matrix()
    pts_lin = np.stack([X0, Y0, Z0], axis=1) @ R_hat.T + delta_hat_np[3:6]
    Xc, Yc, Zc = pts_lin[:, 0], pts_lin[:, 1], pts_lin[:, 2]

    # numpy KB Jacobian — same chain as scripts.ba.ba_kb_jac.
    from scripts.ba.ba_kb_jac import kb_jacobian as kb_jac_np
    Ju, Jv = kb_jac_np(Xc, Yc, Zc, K, dist, m.DOF)        # (N, 6) each
    N = len(uv_pert)
    K_dim = len(m.DOF)
    # Stack to (2N, K) by interleaving (u, v) per point.
    J = np.zeros((2 * N, K_dim))
    J[0::2] = Ju
    J[1::2] = Jv

    # ─── Decompose Δuv_learned and Δuv_oracle into J row space + null ─
    # Whitening with W: BA solves min ‖W^{1/2}(Δuv − J·δ)‖². The "row space"
    # is the column span of W^{1/2} J in 2N space. Null space = orthogonal
    # complement.
    # For a clean geometric reading, do the decomposition without W (= L₂):
    # Δuv_row = J (JᵀJ)⁻¹ Jᵀ Δuv  ;  Δuv_null = Δuv − Δuv_row.
    # If the network is BA-consistent, |Δuv_row vs J·δ̂| should be tiny.
    duv_learned_np = duv_learned.cpu().numpy()             # (N, 2)
    duv_oracle_np = project_kb(pts_cam_in, K, dist) - uv_pert    # (N, 2)
    duv_learned_flat = duv_learned_np.reshape(-1)          # (2N,)
    duv_oracle_flat  = duv_oracle_np.reshape(-1)

    JTJ_inv = np.linalg.inv(J.T @ J)
    proj = lambda x: J @ (JTJ_inv @ (J.T @ x))
    duv_learned_row  = proj(duv_learned_flat)
    duv_learned_null = duv_learned_flat - duv_learned_row
    duv_oracle_row   = proj(duv_oracle_flat)
    duv_oracle_null  = duv_oracle_flat - duv_oracle_row

    # Reconstruction Jδ_target / Jδ̂ for direct comparison.
    Jdelta_target = J @ delta_target.cpu().numpy().astype(np.float64)
    Jdelta_hat    = J @ delta_hat_np

    def _norm(x): return float(np.sqrt((x ** 2).sum() / N))   # px-RMS
    print('\n=== Δuv decomposition (RMS px) ===')
    print(f'  ‖Δuv_oracle‖              = {_norm(duv_oracle_flat):.3f}')
    print(f'  ‖Δuv_oracle.row‖          = {_norm(duv_oracle_row):.3f}'
           f'   (= what BA acts on)')
    print(f'  ‖Δuv_oracle.null‖         = {_norm(duv_oracle_null):.3f}'
           f'   (≈ KB non-linearity at δ ≈ 1°)')
    print(f'  ‖Δuv_learned‖             = {_norm(duv_learned_flat):.3f}')
    print(f'  ‖Δuv_learned.row‖         = {_norm(duv_learned_row):.3f}'
           f'   (= what BA acts on)')
    print(f'  ‖Δuv_learned.null‖        = {_norm(duv_learned_null):.3f}'
           f'   (= NULL-SPACE drift, invisible to pose loss)')
    print(f'  ‖J·δ_target‖              = {_norm(Jdelta_target):.3f}')
    print(f'  ‖J·δ̂‖                    = {_norm(Jdelta_hat):.3f}')
    print(f'  ‖Δuv_learned.row − J·δ̂‖  = {_norm(duv_learned_row - Jdelta_hat):.3e}'
           f'   (should be ≈ 0 if BA-consistent)')
    print(f'  ‖Δuv_oracle.row  − J·δ_target‖ = {_norm(duv_oracle_row - Jdelta_target):.3e}'
           f'   (≈ 0 ⇒ oracle satisfies BA at this lin. pt; '
           f'ε is the KB non-linearity)')
    # Cosine similarity (row component)
    cos_row = float((duv_learned_row @ duv_oracle_row)
                    / (np.linalg.norm(duv_learned_row)
                        * np.linalg.norm(duv_oracle_row) + 1e-12))
    cos_null = float((duv_learned_null @ duv_oracle_null)
                    / (np.linalg.norm(duv_learned_null)
                        * np.linalg.norm(duv_oracle_null) + 1e-12))
    print(f'  cos(learned.row,  oracle.row)  = {cos_row:+.4f}'
           f'   (≈ +1 ⇒ same 6-D solution)')
    print(f'  cos(learned.null, oracle.null) = {cos_null:+.4f}'
           f'   (≈ 0  ⇒ null component is unconstrained)')

    # ─── Plot: 3 panels.
    #   (a) Δuv RAW: learned vs oracle (per-pt, per-axis) → scatter
    #       diverges from y=x ⇔ network found an alternative valid Δuv
    #   (b) Δuv ROW component (= J·(JᵀJ)⁻¹·Jᵀ·Δuv): learned vs oracle
    #       y=x ⇔ same 6-D BA solution
    #   (c) RMS bar chart: row vs null components
    fig, axes = plt.subplots(1, 3, figsize=(17.5, 4.6))

    # (a) raw Δuv
    duv_l_2 = duv_learned_np
    duv_o_2 = duv_oracle_np
    axes[0].scatter(duv_o_2[:, 0], duv_l_2[:, 0], s=4, alpha=0.35,
                     label='Δu', color='tab:blue')
    axes[0].scatter(duv_o_2[:, 1], duv_l_2[:, 1], s=4, alpha=0.35,
                     label='Δv', color='tab:orange')
    lim_a = max(abs(duv_o_2).max(), abs(duv_l_2).max()) * 1.05
    axes[0].plot([-lim_a, lim_a], [-lim_a, lim_a], 'k--', lw=1, label='y = x  (= oracle match)')
    axes[0].set_xlim(-lim_a, lim_a); axes[0].set_ylim(-lim_a, lim_a)
    axes[0].set_aspect('equal')
    axes[0].set_xlabel('Δuv_oracle  [px]')
    axes[0].set_ylabel('Δuv_learned  [px]')
    axes[0].set_title('(a) RAW Δuv  per-pt, per-axis\n'
                       '→ scatter ⇒ network found a different Δuv field')
    axes[0].legend(loc='best'); axes[0].grid(alpha=0.3)

    # (b) row component
    duv_learned_row_2 = duv_learned_row.reshape(N, 2)
    duv_oracle_row_2  = duv_oracle_row.reshape(N, 2)
    axes[1].scatter(duv_oracle_row_2[:, 0], duv_learned_row_2[:, 0],
                     s=4, alpha=0.4, label='Δu  row(J)', color='tab:blue')
    axes[1].scatter(duv_oracle_row_2[:, 1], duv_learned_row_2[:, 1],
                     s=4, alpha=0.4, label='Δv  row(J)', color='tab:orange')
    lim_b = max(abs(duv_oracle_row_2).max(), abs(duv_learned_row_2).max()) * 1.05
    axes[1].plot([-lim_b, lim_b], [-lim_b, lim_b], 'k--', lw=1, label='y = x')
    axes[1].set_xlim(-lim_b, lim_b); axes[1].set_ylim(-lim_b, lim_b)
    axes[1].set_aspect('equal')
    axes[1].set_xlabel('Δuv_oracle  · row(J) projection  [px]')
    axes[1].set_ylabel('Δuv_learned · row(J) projection  [px]')
    axes[1].set_title(f'(b) ROW-space component  (= what BA sees)\n'
                       f'cos = {cos_row:+.4f}  ⇒ same 6-DoF solution')
    axes[1].legend(loc='best'); axes[1].grid(alpha=0.3)

    # (c) RMS bars
    cats = ['oracle\n(uv_GT − uv_pert)', 'learned\n(network out)']
    rows = [_norm(duv_oracle_row), _norm(duv_learned_row)]
    nuls = [_norm(duv_oracle_null), _norm(duv_learned_null)]
    x = np.arange(len(cats))
    w = 0.35
    axes[2].bar(x - w/2, rows, w, label='row(J)  (BA-visible)',
                  color='tab:blue')
    axes[2].bar(x + w/2, nuls, w, label='null(J)  (loss-invisible)',
                  color='tab:red', alpha=0.85)
    axes[2].set_xticks(x); axes[2].set_xticklabels(cats)
    axes[2].set_ylabel('RMS  [px]')
    axes[2].set_title('(c) RMS by subspace\n'
                       'oracle null is tiny (only KB non-linearity);\n'
                       'learned null is free → drifts away from oracle')
    axes[2].legend(loc='best'); axes[2].grid(axis='y', alpha=0.3)

    fig.suptitle(
        'Pose loss only constrains the row(J) ⊆ R^{2N} subspace (= 6-DoF). '
        'Network finds *some* Δuv whose row(J) component matches the oracle\n'
        'but is free to drift in the null(J) direction. (a) shows the raw '
        'Δuv disagreeing with y=x; (b) shows their row(J) projections agreeing.',
        y=1.04, fontsize=10)
    fig.tight_layout()
    out_path = OUT / 'overfit_nullspace.png'
    fig.savefig(out_path, dpi=120, bbox_inches='tight')
    plt.close(fig)
    print(f'\nwrote → {out_path}')


if __name__ == '__main__':
    main()
