"""Fig 6 — Real-frame BA on a 512×512 tile (truck region) with ORACLE Δuv.

Story: same experiment as Fig 5, but on a real fisheye image. The "network"
is replaced by an oracle that knows the true GT pose, so Δuv = uv_GT - uv_pert
exactly (then we add N(0, σ=1px) noise). This isolates the BA layer from the
unknowns of network quality and shows that on real data, with a perfect
predictor + sensor noise, the closed-form 6-DoF KB solver hits the noise
floor in 1-2 GN steps just like the synthetic experiment.

Crop: centre of `points_ip664_D_20260226_224648_d005_3000_3020/image_0.png`,
512×512 about (W/2, H/2 + 300) — captures the two trucks ahead on the road.

Output: docs/assets/2026-05-19_diffba/fig6_real_frame_ba.png
"""
from __future__ import annotations
import sys
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from scipy.spatial.transform import Rotation

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))

from scripts.data.adapters.kamikado import load_frame
from scripts.ba.ba_kb_jac import solve_dofs_kb, project_kb

OUT = REPO / 'docs' / 'assets' / '2026-05-19_diffba'
OUT.mkdir(parents=True, exist_ok=True)

SCENE = Path.home() / 'cache' / 'kamikado' / 'scenes' / \
        'points_ip664_D_20260226_224648_d005_3000_3020'
FRAME = 0
TILE = 512               # px (square)
DV = 300                 # tile centre = (W/2, H/2 + DV)
SIGMA_UV = 1.0           # px sensor noise on the oracle Δuv
RNG_SEED = 7
# Visualisation: show ALL in-tile pts as small semi-transparent dots so the
# perturbation field reads as a coherent shift, not a scattergram.

# Visible perturbation: ~1° rotation + ~0.3 m translation. This is
# intentionally bigger than Fig 5's 0.1° so the red/green/yellow split is
# obvious to the eye on a 512-px crop.
DELTA_TRUE = np.array([1.0, 1.5, 0.5,        # ω_x, ω_y, ω_z  [deg]
                        0.20, -0.30, 0.40],   # tx, ty, tz     [m]
                       dtype=np.float64)
DOF = ['omega_x', 'omega_y', 'omega_z', 'tx', 'ty', 'tz']


def main():
    cf = load_frame(SCENE, FRAME)
    img = cf.img
    K = cf.K.astype(np.float64)
    dist = np.asarray(cf.dist, dtype=np.float64)
    pts_cam = cf.pts_cam.astype(np.float64)
    H, W = img.shape[:2]
    print(f'frame {W}×{H}  pts {len(pts_cam)}  fx={K[0,0]:.1f} cx={K[0,2]:.1f} cy={K[1,2]:.1f}')

    # Tile bbox (square, centred at (W/2, H/2 + DV))
    cu, cv = W // 2, H // 2 + DV
    u0 = cu - TILE // 2; v0 = cv - TILE // 2
    u1 = u0 + TILE;       v1 = v0 + TILE
    print(f'tile: u∈[{u0},{u1}] v∈[{v0},{v1}]  centre=({cu},{cv})')

    # GT projection of all LiDAR pts (KB) → keep those that fall inside the tile
    uv_gt_all = project_kb(pts_cam, K, dist)
    in_tile = ((uv_gt_all[:, 0] >= u0) & (uv_gt_all[:, 0] < u1)
                & (uv_gt_all[:, 1] >= v0) & (uv_gt_all[:, 1] < v1)
                & (pts_cam[:, 2] > 0.5))
    pts3 = pts_cam[in_tile]
    uv_gt = uv_gt_all[in_tile]
    print(f'in-tile pts: {len(pts3)} / {len(pts_cam)}')

    # Apply δ_true to camera pose: pts_pert = R(δ_ω) · pts + δ_t  (cam-frame)
    R = Rotation.from_rotvec(np.deg2rad(DELTA_TRUE[:3])).as_matrix()
    pts3_pert = pts3 @ R.T + DELTA_TRUE[3:6]
    uv_pert = project_kb(pts3_pert, K, dist)

    # Oracle Δuv = uv_gt - uv_pert  (perfect network knows where the point
    # SHOULD be).  Add N(0, σ) sensor noise.
    rng = np.random.RandomState(RNG_SEED)
    duv_oracle = (uv_gt - uv_pert) + rng.normal(0.0, SIGMA_UV, uv_gt.shape)
    print(f'oracle |Δuv|: mean {np.linalg.norm(uv_gt - uv_pert, axis=1).mean():.2f} px '
           f'median {np.median(np.linalg.norm(uv_gt - uv_pert, axis=1)):.2f}  '
           f'+ N(0,{SIGMA_UV}px) noise')

    # BA pool: solver expects par = [Δu, Δv, σ_x, σ_y, ρ] in PARENT-px coords,
    # with uv = the observed (perturbed) projection — i.e. uv_target = uv + Δuv.
    par = np.zeros((len(uv_pert), 5))
    par[:, 0:2] = duv_oracle
    par[:, 2] = par[:, 3] = SIGMA_UV          # isotropic σ = 1 px
    par[:, 4] = 0.0                            # ρ = 0
    z_kb = pts3_pert[:, 2]                     # cam-Z at the perturbed pose

    # Solve at n_iter ∈ {1, 2, 3} so we can compare to Fig 5.
    results = {}
    damping = 1e-3
    for n_iter in (1, 2, 3):
        delta_hat = solve_dofs_kb(uv_pert, par, z_kb, K, dist, DOF,
                                    damping=damping, huber_k=None, n_iter=n_iter)
        # Apply δ̂ as an INVERSE correction to recover GT-side projection
        # (same convention as Fig 5: δ̂ moves perturbed → GT).
        R_hat = Rotation.from_rotvec(np.deg2rad(delta_hat[:3])).as_matrix()
        pts3_corr = pts3_pert @ R_hat.T + delta_hat[3:6]
        uv_corr = project_kb(pts3_corr, K, dist)
        # Residual after correction = uv_target (= uv_pert + Δuv_oracle) − uv_corr.
        # Equivalently, against the noiseless GT it is uv_corr − uv_gt.
        uv_target = uv_pert + duv_oracle
        res = uv_target - uv_corr
        rms_target = np.sqrt((res ** 2).sum(axis=1).mean())
        rms_gt     = np.sqrt(((uv_corr - uv_gt) ** 2).sum(axis=1).mean())
        derr = np.abs(np.abs(delta_hat) - np.abs(DELTA_TRUE)).max()
        results[n_iter] = dict(
            delta=delta_hat, uv_corr=uv_corr,
            res=res, rms_target=rms_target, rms_gt=rms_gt, derr=derr,
        )
        print(f'  n_iter={n_iter}  RMS(uv_corr-target)={rms_target:.4f} px  '
               f'RMS(uv_corr-GT)={rms_gt:.4f} px  '
               f'|δ̂-δ_true|_max={derr:.3e}')

    noise_floor = np.sqrt(2) * SIGMA_UV
    print(f'noise floor √2·σ = {noise_floor:.4f} px')

    # ── Plot: 3 panels (GT yellow / Perturbed red / Recovered green) on the
    # tile crop. ALL in-tile pts, small + low-alpha so the structured shift
    # in the middle panel reads as a coherent field.
    crop = img[v0:v1, u0:u1]

    # Use n_iter=1 for the recovered panel (already at noise floor; identical
    # to n=2/3 by Fig 2's quadratic-convergence saturation argument).
    uv_rec = results[1]['uv_corr']
    rms_rec_vs_gt = np.sqrt(((uv_rec - uv_gt) ** 2).sum(axis=1).mean())

    fig, axes = plt.subplots(1, 3, figsize=(15, 5.6))
    panels = [
        ('GT  (yellow)',
            uv_gt, 'gold', '0 px (definition)'),
        (f'Perturbed by δ_true = ~1°, ~0.3 m  (red)',
            uv_pert, 'red',
            f'mean |Δuv| = {np.linalg.norm(uv_gt - uv_pert, axis=1).mean():.1f} px '
            f'(structured shift)'),
        (f'Recovered  R(δ̂)·P_pert + t̂  (green, 1 GN step)',
            uv_rec, 'limegreen',
            f'RMS vs GT = {rms_rec_vs_gt:.3f} px '
            f'(noise floor √2·σ = {noise_floor:.2f} px)'),
    ]
    for ax, (title, pts_uv, colr, sub_msg) in zip(axes, panels):
        ax.imshow(crop, extent=[u0, u1, v1, v0])
        ax.scatter(pts_uv[:, 0], pts_uv[:, 1],
                    s=2.5, c=colr, marker='o',
                    linewidths=0, alpha=0.45)
        ax.set_xlim(u0, u1); ax.set_ylim(v1, v0)
        ax.set_aspect('equal')
        ax.set_title(f'{title}\n{sub_msg}', fontsize=10)
        ax.set_xticks([]); ax.set_yticks([])

    fig.suptitle(
        f'Fig 6.  Project → perturb → analytically recover, on a real fisheye tile.  '
        f'(all {len(uv_gt)} in-tile LiDAR pts shown;  σ_uv = {SIGMA_UV} px noise on oracle Δuv)\n'
        f'KB closed-form 6-DoF GN, 1 step  ·  scene = {SCENE.name}',
        y=1.02, fontsize=10,
    )
    fig.tight_layout()
    out_path = OUT / 'fig6_real_frame_ba.png'
    fig.savefig(out_path, dpi=140, bbox_inches='tight')
    plt.close(fig)
    print(f'wrote → {out_path}')

    # ── Print pose summary table for the user.
    print('\n  Pose summary (deg / m):')
    print(f'    {"":<14} {"ω_x":>10} {"ω_y":>10} {"ω_z":>10} {"tx":>10} {"ty":>10} {"tz":>10}')
    print(f'    {"GT":<14} ' + ' '.join(f'{0.0:>10.4f}' for _ in range(6)))
    print(f'    {"perturbed":<14} ' + ' '.join(f'{v:>+10.4f}' for v in DELTA_TRUE))
    for n in (1, 2, 3):
        print(f'    {"BA δ̂ (n=" + str(n) + ")":<14} '
              + ' '.join(f'{v:>+10.4f}' for v in results[n]['delta']))


if __name__ == '__main__':
    main()
