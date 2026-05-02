"""Frustum-encoder neighborhood debug vis.

Loads a single V3 crop, runs the REAL FrustumLocalEncoder.forward (the same
class CalibNetDepth uses at training time), and visualizes the actual
top-k indices it selected. No reimplementation — the live forward writes
`_last_topk_idx` / `_last_valid` / `_last_r_uv_px` and we read them out.

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
from models.model_depth import FrustumLocalEncoder, D_DIM


def render(crop_img, dist_uvd, uvd_full, pad_full, S, grid_n, r_uv_cells, r_d, k,
            highlight_idxs=None, out_path='/tmp/frustum.png'):
    # build encoder, run real forward, read its instrumentation
    enc = FrustumLocalEncoder(d_out=D_DIM, r_uv_cells=r_uv_cells, r_d=r_d,
                                k=k, grid_n=grid_n).eval()
    q_t = torch.from_numpy(dist_uvd[:, :3].astype(np.float32))[None]   # (1, Nq, 3)
    f_t = torch.from_numpy(uvd_full.astype(np.float32))[None]          # (1, Nkv, 3)
    p_t = torch.from_numpy(pad_full.astype(bool))[None]                # (1, Nkv)
    with torch.no_grad():
        _ = enc(q_t, full_uvd=f_t, full_pad_mask=p_t, img_size=S)
    topk = enc._last_topk_idx[0].cpu().numpy()    # (Nq, k)
    valid = enc._last_valid[0].cpu().numpy()      # (Nq, k)
    r_uv_px = enc._last_r_uv_px
    cell_px = enc._last_cell_px

    qm = dist_uvd[:, :3].astype(np.float32)
    fm = uvd_full.astype(np.float32)
    n_neighbors = valid.sum(axis=1)
    print(f'queries={len(qm)}  full_pts={(~pad_full).sum()}/{len(fm)}  '
          f'cell_px={cell_px:.2f}  r_uv_px={r_uv_px:.2f}')
    print(f'live forward neighbors per query: min={n_neighbors.min()} '
          f'mean={n_neighbors.mean():.1f} max={n_neighbors.max()}')

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

    def draw_grid(ax, lw=0.4, color='magenta', alpha=0.35):
        for i in range(grid_n + 1):
            x = i * cell_px
            ax.axvline(x, color=color, lw=lw, alpha=alpha, zorder=1)
            ax.axhline(x, color=color, lw=lw, alpha=alpha, zorder=1)

    ax = axes[0]
    ax.imshow(crop_np)
    draw_grid(ax)
    ax.scatter(full_uv_valid[:, 0], full_uv_valid[:, 1], c='yellow', s=4,
               marker='o', alpha=0.6, label=f'full ({len(full_uv_valid)})', zorder=2)
    ax.scatter(qm[:, 0], qm[:, 1], c='lime', s=22, marker='x', linewidths=1.2,
               label=f'query ({len(qm)})', zorder=4)
    ax.set_xlim(0, S); ax.set_ylim(S, 0); ax.axis('off')
    ax.set_title(f'OVERVIEW (LIVE forward)  grid={grid_n}×{grid_n} cell={cell_px:.1f}px  '
                 f'box=±{r_uv_px:.1f}px  r_d={r_d}({r_d*100:.1f}m)  k={k}  '
                 f'nbr/q μ={n_neighbors.mean():.1f}', fontsize=8)
    ax.legend(loc='upper right', fontsize=7)

    for i, qi in enumerate(highlight_idxs):
        ax = axes[1 + i]
        ax.imshow(crop_np)
        draw_grid(ax, lw=0.6, alpha=0.5)
        ax.scatter(full_uv_valid[:, 0], full_uv_valid[:, 1], c='yellow', s=4,
                   marker='o', alpha=0.5, zorder=2)
        u, v = qm[qi, 0], qm[qi, 1]
        ci = int(min(grid_n - 1, max(0, u // cell_px)))
        cj = int(min(grid_n - 1, max(0, v // cell_px)))
        ax.add_patch(plt.Rectangle((ci * cell_px, cj * cell_px),
                                    cell_px, cell_px, fill=True,
                                    facecolor='magenta', alpha=0.18, lw=0, zorder=1))
        ax.add_patch(plt.Rectangle((u - r_uv_px, v - r_uv_px),
                                    2 * r_uv_px, 2 * r_uv_px,
                                    fill=False, ec='red', lw=1.4, alpha=0.85, zorder=3))
        ax.scatter([u], [v], c='lime', s=110, marker='x', linewidths=2.4, zorder=5)
        nb_q = topk[qi][valid[qi]]
        if len(nb_q):
            nb_uv = fm[nb_q]
            ax.scatter(nb_uv[:, 0], nb_uv[:, 1], s=70,
                       facecolors='none', edgecolors='cyan', linewidths=1.8, zorder=6)
            for nu, nv, nz in nb_uv:
                ax.plot([u, nu], [v, nv], '-', color='cyan', lw=1.0, alpha=0.8, zorder=4)
        zoom = 3 * cell_px
        ax.set_xlim(max(0, u - zoom), min(S, u + zoom))
        ax.set_ylim(min(S, v + zoom), max(0, v - zoom))
        ax.set_title(f'q[{qi}] cell({ci},{cj}) @({u:.1f},{v:.1f}) z={qm[qi,2]*100:.1f}m  '
                     f'nb={int(valid[qi].sum())}', fontsize=8)
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
