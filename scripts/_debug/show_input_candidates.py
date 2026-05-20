"""Visualise candidate network inputs for the random-δ overfit.

The point: the previous run only fed the RGB tile (no projection info),
so the network had no way to know δ_pert and learning UV was impossible.
Below are 3 candidate input encodings — all share the same 512×512 tile
crop, with the perturbed LiDAR projection overlaid in different ways.

Output: docs/assets/2026-05-19_diffba/input_candidates.png
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
from scripts.ba.ba_kb_jac import project_kb

OUT = REPO / 'docs' / 'assets' / '2026-05-19_diffba'
SCENE = Path.home() / 'cache' / 'kamikado' / 'scenes' / \
        'points_ip664_D_20260226_224648_d005_3000_3020'
FRAME = 0
TILE = 512
DV = 300
DELTA = np.array([1.0, 1.5, 0.0, 0.0, 0.0, 0.0], dtype=np.float64)  # 2-DoF


def main():
    cf = load_frame(SCENE, FRAME)
    img = cf.img.astype(np.float32) / 255.0
    H, W = img.shape[:2]
    K = cf.K.astype(np.float64); dist = np.asarray(cf.dist, np.float64)
    pts_cam = cf.pts_cam.astype(np.float64)
    cu, cv = W // 2, H // 2 + DV
    u0 = cu - TILE // 2; v0 = cv - TILE // 2
    u1 = u0 + TILE;       v1 = v0 + TILE

    # Perturbed LiDAR projection
    R = Rotation.from_rotvec(np.deg2rad(DELTA[:3])).as_matrix()
    pts_pert = pts_cam @ R.T + DELTA[3:6]
    uv_pert = project_kb(pts_pert, K, dist)
    z_pert = pts_pert[:, 2]
    in_tile = ((uv_pert[:, 0] >= u0) & (uv_pert[:, 0] < u1)
                & (uv_pert[:, 1] >= v0) & (uv_pert[:, 1] < v1)
                & (z_pert > 0.5))
    uv = uv_pert[in_tile] - np.array([u0, v0])
    z  = z_pert[in_tile]
    print(f'in-tile pts: {in_tile.sum()}')

    crop = img[v0:v1, u0:u1].copy()

    # Candidate A: RGB only (what the previous run actually used — useless).
    cand_a = crop.copy()

    # Build per-pixel 1/z map (max-blend so nearer pts win when stamps overlap).
    rad = 1
    uvi = uv.round().astype(int)
    H_t, W_t = TILE, TILE
    invz = np.zeros((H_t, W_t), dtype=np.float32)
    invz_pt = 1.0 / np.clip(z, 1.0, 60.0)
    for du in range(-rad, rad + 1):
        for dv in range(-rad, rad + 1):
            ui = np.clip(uvi[:, 0] + du, 0, W_t - 1)
            vi = np.clip(uvi[:, 1] + dv, 0, H_t - 1)
            np.maximum.at(invz, (vi, ui), invz_pt)

    # Candidate B: RGB with a 1/z heatmap (matplotlib magma colormap)
    # alpha-blended on top wherever a point projects.  3 channels, the
    # "color" of each dot now encodes depth instead of being plain red.
    norm = invz / max(invz.max(), 1e-6)            # (H,W) ∈ [0,1]
    cmap = plt.get_cmap('magma')
    rgba = cmap(norm)[..., :3].astype(np.float32)   # (H,W,3)
    alpha_map = (norm > 0).astype(np.float32) * 0.65   # (H,W)
    cand_b = crop.copy()
    a = alpha_map[..., None]
    cand_b = a * rgba + (1 - a) * cand_b

    # Candidate C: separate channels — RGB + presence mask + 1/z map (5ch).
    mask = (invz > 0).astype(np.float32)

    fig, axes = plt.subplots(1, 4, figsize=(20, 5.2))
    axes[0].imshow(cand_a); axes[0].set_title('(A) RGB only — what was used\n'
                                                'no δ_pert info → cannot learn',
                                                fontsize=10)
    axes[1].imshow(np.clip(cand_b, 0, 1))
    axes[1].set_title('(B) RGB + 1/z heatmap overlay  (3ch)\n'
                        'magma colormap on perturbed projection — depth-coded dots',
                        fontsize=10)
    axes[2].imshow(mask, cmap='gray'); axes[2].set_title('(C-extra) point-presence mask  (1ch)',
                                                            fontsize=10)
    axes[3].imshow(invz, cmap='magma'); axes[3].set_title('(C-extra) 1/z heatmap  (1ch)\n'
                                                            'C = RGB + mask + 1/z (5ch)',
                                                            fontsize=10)
    for ax in axes:
        ax.set_xticks([]); ax.set_yticks([])
    fig.suptitle(f'Network input candidates  ·  δ_pert = (ωx={DELTA[0]:.1f}°, ωy={DELTA[1]:.1f}°)  '
                  f'·  {in_tile.sum()} in-tile pts', y=1.02, fontsize=11)
    fig.tight_layout()
    out = OUT / 'input_candidates.png'
    fig.savefig(out, dpi=120, bbox_inches='tight')
    plt.close(fig)
    print(f'wrote → {out}')


if __name__ == '__main__':
    main()
