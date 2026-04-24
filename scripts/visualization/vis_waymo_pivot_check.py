"""Quick sanity: pick a Waymo-converted scene, grab N pairs, project pivot
(patch_A center's 3D point) into frame B under GT pose. If the green ★ on
B lands on the same physical feature as the red ★ on A, the converter's
pose/intrinsics/cam-convention are correct.

Run: python scripts/visualization/vis_waymo_pivot_check.py
"""
import sys, pathlib
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[2]))

from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import ConnectionPatch

from datasets.pandaset_pair import _SceneData, _project

import sys as _sys
SCENE = _sys.argv[1] if len(_sys.argv) > 1 else '/mnt/nvme6t/waymo_ps/10017090168044687777_6380_000_6400_000'
OUT   = Path(_sys.argv[2]) if len(_sys.argv) > 2 else Path('experiments/waymo_pivot_check.png')
N_PAIRS = 4
BASELINES = [1, 5, 10, 20]


def main():
    scn = _SceneData(Path(SCENE))
    scn.precompute_all(preload_images=True)

    fig, axes = plt.subplots(N_PAIRS, 2, figsize=(12, 5.5 * N_PAIRS), dpi=100)
    fig.patch.set_facecolor('#f6f4ed')
    if N_PAIRS == 1:
        axes = axes[None, :]

    rng = np.random.default_rng(0)
    for ri, bl in enumerate(BASELINES):
        fi_A = int(rng.integers(5, scn.n_frames - bl - 5))
        fi_B = fi_A + bl

        # Use A's LiDAR in world frame; project to A (own) AND to B.
        pts_w_A, uv_A, z_A, inv_A = scn.frame_data(fi_A)
        # Project the SAME A-frame world points into camera B using scn.T_w2c[fi_B]
        uv_B_of_A, z_B_of_A = _project(pts_w_A, scn.T_w2c[fi_B], scn.K)
        inv_B_of_A = ((z_B_of_A > 1.0) &
                      (uv_B_of_A[:, 0] > 0) & (uv_B_of_A[:, 0] < scn.IW) &
                      (uv_B_of_A[:, 1] > 0) & (uv_B_of_A[:, 1] < scn.IH))
        both = inv_A & inv_B_of_A
        if both.sum() == 0:
            print(f'[bl={bl}] no co-visible points, skipping')
            continue
        center_A = np.array([scn.IW / 2, scn.IH / 2])
        covis = np.where(both)[0]
        d = np.linalg.norm(uv_A[covis] - center_A, axis=1)
        piv_idx = int(covis[np.argsort(d)[0]])

        uv_Apx = uv_A[piv_idx]
        uv_Bpx = uv_B_of_A[piv_idx]
        z_Bpx  = z_B_of_A[piv_idx]
        print(f'[bl={bl}] fi_A={fi_A} fi_B={fi_B} pivot idx {piv_idx}: '
              f'A uv=({uv_Apx[0]:.0f},{uv_Apx[1]:.0f}) z={z_A[piv_idx]:.1f}m, '
              f'B uv=({uv_Bpx[0]:.0f},{uv_Bpx[1]:.0f}) z={z_Bpx:.1f}m')

        ax_A, ax_B = axes[ri]
        ax_A.imshow(scn.load_image(fi_A))
        ax_B.imshow(scn.load_image(fi_B))
        for ax in (ax_A, ax_B):
            ax.set_xticks([]); ax.set_yticks([])

        ax_A.set_title(f'A (fi={fi_A})  red ★ = pivot', fontsize=11, loc='left')
        ax_B.set_title(f'B (fi={fi_B}, Δ{bl:+d})  green ★ = pivot projected under GT',
                        fontsize=11, loc='left')

        # A: pivot red star
        ax_A.plot(*uv_Apx, marker='*', color='#c13c14', markersize=30,
                   markeredgecolor='white', mew=2.5, zorder=10)
        # B: pivot green star at GT projection
        ax_B.plot(*uv_Bpx, marker='*', color='#0fa550', markersize=30,
                   markeredgecolor='white', mew=2.5, zorder=10)
        # cross-panel dotted line
        fig.add_artist(ConnectionPatch(
            xyA=uv_Apx, xyB=uv_Bpx,
            coordsA='data', coordsB='data',
            axesA=ax_A, axesB=ax_B,
            color='#888', lw=1.2, ls=':', alpha=0.8, zorder=4))

        # sprinkle 30 supporting co-visible points for context (small dots)
        sample = rng.choice(covis, size=min(30, len(covis)), replace=False)
        for i in sample:
            ax_A.plot(*uv_A[i], 'o', color='#1e6fff', markersize=4,
                       alpha=0.6, markeredgecolor='white', mew=0.5, zorder=5)
            ax_B.plot(*uv_B_of_A[i], 'o', color='#1e6fff', markersize=4,
                       alpha=0.6, markeredgecolor='white', mew=0.5, zorder=5)

    OUT.parent.mkdir(parents=True, exist_ok=True)
    plt.suptitle(f'Waymo pivot-projection sanity ({Path(SCENE).name})\n'
                  f'blue ● = 30 random co-visible LiDAR points (should align too)',
                  fontsize=13, y=0.995)
    plt.tight_layout()
    plt.savefig(OUT, dpi=100, bbox_inches='tight', facecolor='#f6f4ed')
    print(f'saved → {OUT}')


if __name__ == '__main__':
    main()
