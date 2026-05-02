"""Frustum-encoder neighborhood debug vis.

Loads a single V3 crop, replicates the FrustumLocalEncoder's box-filter +
UV-top-k selection, and renders WHICH dense raw points each query actually
pulls into its local neighborhood. Proof that:
  (a) full_uvd is wired in (otherwise no points appear),
  (b) the box radius scales with cell_px = img_size / grid_n,
  (c) the k=8 chosen neighbors really are around each query.

Usage:
  python scripts/visualization/vis_frustum_neighbors.py \
      --cache /dev/shm/nuscenes_v3_full --idx 0 --out /tmp/frustum_vis.png
"""
import sys, pathlib
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[2]))
import argparse
from pathlib import Path
import numpy as np
import torch
import matplotlib; matplotlib.use('Agg')
import matplotlib.pyplot as plt

from datasets.pandaset_full import PandaSetCalibDatasetFull


def select_neighbors(query_uvd: np.ndarray, full_uvd: np.ndarray, full_pad: np.ndarray,
                      r_uv: float, r_d: float, k: int):
    """Replicate FrustumLocalEncoder.forward selection logic.
    Returns (N_q, k) idx into full_uvd; -1 = no neighbor."""
    Nq = query_uvd.shape[0]
    Nkv = full_uvd.shape[0]
    rel = full_uvd[None, :, :] - query_uvd[:, None, :]   # (Nq, Nkv, 3)
    in_box = ((np.abs(rel[..., 0]) < r_uv) &
              (np.abs(rel[..., 1]) < r_uv) &
              (np.abs(rel[..., 2]) < r_d))
    # exclude exact self-coordinate matches & padded slots
    self_match = (rel[..., 0] == 0) & (rel[..., 1] == 0) & (rel[..., 2] == 0)
    in_box = in_box & ~self_match & ~full_pad[None, :]
    uv_d2 = rel[..., 0] ** 2 + rel[..., 1] ** 2
    uv_d2 = np.where(in_box, uv_d2, 1e9)
    k_eff = min(k, Nkv)
    idx = np.argpartition(uv_d2, k_eff - 1, axis=1)[:, :k_eff]
    # mark invalid
    valid = np.take_along_axis(uv_d2, idx, axis=1) < 1e8
    idx = np.where(valid, idx, -1)
    return idx, valid


def render(crop_img, dist_uvd, uvd_full, pad_full, S, grid_n, r_uv_cells, r_d, k,
            highlight_idxs=None, out_path='/tmp/frustum.png'):
    cell_px = float(S) / float(grid_n)
    r_uv_px = r_uv_cells * cell_px

    # selected neighbor lookup
    qm = dist_uvd[:, :3].astype(np.float32)
    fm = uvd_full.astype(np.float32)
    nb_idx, nb_valid = select_neighbors(qm, fm, pad_full, r_uv_px, r_d, k)

    n_neighbors = nb_valid.sum(axis=1)
    print(f'queries={len(qm)}  full_pts={(~pad_full).sum()}/{len(fm)}  '
          f'cell_px={cell_px:.2f}  r_uv_px={r_uv_px:.2f}')
    print(f'neighbors per query: min={n_neighbors.min()} mean={n_neighbors.mean():.1f} max={n_neighbors.max()}')

    if highlight_idxs is None:
        order = np.argsort(-n_neighbors)
        highlight_idxs = list(order[:6])
    n_panels = len(highlight_idxs) + 1
    cols = 4
    rows = (n_panels + cols - 1) // cols
    fig, axes = plt.subplots(rows, cols, figsize=(4 * cols, 4 * rows), dpi=110)
    axes = axes.flatten()

    crop_np = crop_img.permute(1, 2, 0).cpu().numpy() if hasattr(crop_img, 'permute') else crop_img
    full_uv_valid = fm[~pad_full]

    # overview
    ax = axes[0]
    ax.imshow(crop_np)
    ax.scatter(full_uv_valid[:, 0], full_uv_valid[:, 1], c='yellow', s=4,
               marker='o', alpha=0.5, label=f'full ({len(full_uv_valid)})')
    ax.scatter(qm[:, 0], qm[:, 1], c='lime', s=20, marker='x', linewidths=1.2,
               label=f'query ({len(qm)})')
    ax.set_xlim(0, S); ax.set_ylim(S, 0); ax.axis('off')
    ax.set_title(f'OVERVIEW  cell={cell_px:.1f}px  r_uv={r_uv_px:.1f}px  '
                 f'k={k}  nbr/q μ={n_neighbors.mean():.1f}', fontsize=9)
    ax.legend(loc='upper right', fontsize=7)

    for i, qi in enumerate(highlight_idxs):
        ax = axes[1 + i]
        ax.imshow(crop_np)
        ax.scatter(full_uv_valid[:, 0], full_uv_valid[:, 1], c='yellow', s=3,
                   marker='o', alpha=0.3)
        # box
        u, v = qm[qi, 0], qm[qi, 1]
        ax.add_patch(plt.Rectangle((u - r_uv_px, v - r_uv_px),
                                    2 * r_uv_px, 2 * r_uv_px,
                                    fill=False, ec='red', lw=1.2, alpha=0.8))
        # query
        ax.scatter([u], [v], c='lime', s=80, marker='x', linewidths=2.0)
        # neighbors
        nb_q = nb_idx[qi]
        nb_q = nb_q[nb_q >= 0]
        if len(nb_q):
            nb_uv = fm[nb_q]
            ax.scatter(nb_uv[:, 0], nb_uv[:, 1], c='cyan', s=40,
                       facecolors='none', edgecolors='cyan', linewidths=1.5)
            for nu, nv, _ in nb_uv:
                ax.plot([u, nu], [v, nv], '-', color='cyan', lw=0.8, alpha=0.7)
        ax.set_xlim(max(0, u - 2 * r_uv_px), min(S, u + 2 * r_uv_px))
        ax.set_ylim(min(S, v + 2 * r_uv_px), max(0, v - 2 * r_uv_px))
        ax.set_title(f'q[{qi}] @({u:.1f},{v:.1f})  nb={len(nb_q)}', fontsize=8)
        ax.axis('off')

    for j in range(1 + len(highlight_idxs), len(axes)):
        axes[j].axis('off')
    plt.tight_layout()
    plt.savefig(out_path, dpi=110, bbox_inches='tight'); plt.close(fig)
    print(f'saved → {out_path}')


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--cache', required=True, type=Path)
    ap.add_argument('--idx', type=int, default=0)
    ap.add_argument('--img-size', type=int, default=64)
    ap.add_argument('--grid-n', type=int, default=16)
    ap.add_argument('--r-uv-cells', type=float, default=1.5)
    ap.add_argument('--r-d', type=float, default=0.004)
    ap.add_argument('--k', type=int, default=8)
    ap.add_argument('--n-full', type=int, default=1024)
    ap.add_argument('--out', default='/tmp/frustum_vis.png')
    args = ap.parse_args()

    ds = PandaSetCalibDatasetFull(args.cache, split='val', img_size=args.img_size,
                                    min_crop_px=128, max_crop_px=384, oversample=1,
                                    grid_n=args.grid_n, n_full=args.n_full)
    img, true_uvd, dist_uvd, vfp, uvd_full, pad_full = ds[args.idx]
    print(f'crop@idx={args.idx}: img={tuple(img.shape)} dist={dist_uvd.shape[0]}q  '
          f'full={(~pad_full).sum().item()}/{len(pad_full)}  vfp={float(vfp):.0f}')

    render(img, dist_uvd.numpy(), uvd_full.numpy(), pad_full.numpy(),
            S=args.img_size, grid_n=args.grid_n,
            r_uv_cells=args.r_uv_cells, r_d=args.r_d, k=args.k,
            out_path=args.out)


if __name__ == '__main__':
    main()
