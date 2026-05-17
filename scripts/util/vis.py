"""Visualisation that uses the SAME forward path as epoch_loop's val
pass. No reinventing — the only thing this adds on top is matplotlib
PNG dump, the model forward and per-pt err numbers come straight from
the same DataLoader+collate path that produces train.log val mse.

Public API:
    visualize(model, exp_dir, cache, epoch, n=10, log=print, ...)
"""
from __future__ import annotations

import io
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from torch.utils.data import DataLoader, Subset

from datasets.pandaset_full import PandaSetCalibDatasetFull, collate_full


def _build_loader(cache: str, ds_kw: dict, n: int) -> DataLoader:
    ds = PandaSetCalibDatasetFull(cache, split='val', **ds_kw)
    return DataLoader(Subset(ds, list(range(n))), batch_size=n,
                       num_workers=0, collate_fn=collate_full,
                       shuffle=False), ds


def _render(img_t, true_uv, dist_uv, pred_uv, sx, sy, rho, out_path: Path,
             title: str = ''):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib.patches import Ellipse
    img = img_t.permute(1, 2, 0).cpu().numpy().astype(np.uint8)
    S = img.shape[0]
    fig, ax = plt.subplots(figsize=(6, 6), dpi=110)
    ax.imshow(img)
    valid = ~((dist_uv[:, 0] == 0) & (dist_uv[:, 1] == 0))
    if valid.any():
        # arrow1: GT → dist (perturbation that was applied)
        for k in np.where(valid)[0]:
            ax.annotate('', xy=(dist_uv[k, 0], dist_uv[k, 1]),
                         xytext=(true_uv[k, 0], true_uv[k, 1]),
                         arrowprops=dict(arrowstyle='->', color='orange',
                                          lw=0.5, alpha=0.7), zorder=2)
        # arrow2: dist → pred (model's correction Δ)
        for k in np.where(valid)[0]:
            ax.annotate('', xy=(pred_uv[k, 0], pred_uv[k, 1]),
                         xytext=(dist_uv[k, 0], dist_uv[k, 1]),
                         arrowprops=dict(arrowstyle='->', color='cyan',
                                          lw=0.5, alpha=0.85), zorder=3)
        # σ ellipse around each pred
        for k in np.where(valid)[0]:
            cu, cv = pred_uv[k]
            sxk, syk, rk = float(sx[k]), float(sy[k]), float(rho[k])
            cov = np.array([[sxk*sxk, rk*sxk*syk],
                             [rk*sxk*syk, syk*syk]])
            w, V = np.linalg.eigh(cov)
            ang = np.degrees(np.arctan2(V[1, 1], V[0, 1]))
            e = Ellipse((cu, cv),
                         2 * np.sqrt(max(w[1], 1e-6)),
                         2 * np.sqrt(max(w[0], 1e-6)),
                         angle=ang, facecolor='none', edgecolor='lime',
                         lw=0.4, alpha=0.5, zorder=4)
            ax.add_patch(e)
        # markers (drawn last so they sit on top of arrows)
        ax.scatter(true_uv[valid, 0], true_uv[valid, 1], s=26, c='yellow',
                    marker='x', linewidths=1.1, zorder=5, label='GT')
        ax.scatter(dist_uv[valid, 0], dist_uv[valid, 1], s=20,
                    facecolors='none', edgecolors='red', linewidths=0.9,
                    zorder=6, label='dist (input)')
        ax.scatter(pred_uv[valid, 0], pred_uv[valid, 1], s=20,
                    facecolors='none', edgecolors='lime', linewidths=0.9,
                    zorder=7, label='pred')
    ax.set_xlim(0, S); ax.set_ylim(S, 0); ax.axis('off')
    ax.set_title(title, fontsize=8)
    ax.legend(loc='lower right', fontsize=7)
    plt.tight_layout(pad=0.2)
    fig.savefig(out_path, dpi=96)
    plt.close(fig)


def visualize(model, exp_dir, cache: str, epoch: int,
               ds_kw: dict, n: int = 10, device='cuda',
               amp_dtype=torch.float16, log=print) -> None:
    """Run the trainer's val forward path on `n` val samples and dump
    one PNG each. Logs per-sample err so train.log values stay
    comparable.
    """
    out_dir = Path(exp_dir) / f'vis_ep{epoch:03d}'
    out_dir.mkdir(parents=True, exist_ok=True)
    for old in out_dir.glob('*.png'):
        old.unlink()

    loader, ds = _build_loader(cache, ds_kw, n)
    batch = next(iter(loader))
    imgs, true_uvd, dist_uvd, pad_mask, vfp, b_uvd, b_v = batch[:7]
    imgs_g = imgs.to(device).float().div(255.0)
    true_uvd_g = true_uvd.to(device); dist_uvd_g = dist_uvd.to(device)
    pad_mask_g = pad_mask.to(device); vfp_g = vfp.to(device)
    b_uvd_g = b_uvd.to(device); b_v_g = b_v.to(device)
    use_intensity = bool(getattr(model, 'use_intensity', False))
    point_in = (torch.cat([dist_uvd_g[..., :3], dist_uvd_g[..., 4:5]], -1)
                  if use_intensity else dist_uvd_g[..., :3])
    with torch.no_grad(), torch.autocast(device_type='cuda', dtype=amp_dtype):
        params = model(imgs_g, point_in, key_padding_mask=pad_mask_g,
                        vfp=vfp_g, bucket_uvd=b_uvd_g, bucket_valid=b_v_g)
    p = params.float().cpu().numpy()        # (B, Nmax, 5)
    dist_uv_b = dist_uvd[..., :2].numpy()
    true_uv_b = true_uvd[..., :2].numpy()
    pred_uv_b = dist_uv_b + p[..., :2]
    sx_b = np.exp(p[..., 2]); sy_b = np.exp(p[..., 3])
    rho_b = np.tanh(p[..., 4])

    for k in range(imgs.shape[0]):
        valid = ~((dist_uv_b[k, :, 0] == 0) & (dist_uv_b[k, :, 1] == 0))
        if not valid.any():
            continue
        err_pre  = float(np.linalg.norm(dist_uv_b[k, valid] - true_uv_b[k, valid],
                                          axis=1).mean())
        err_post = float(np.linalg.norm(pred_uv_b[k, valid] - true_uv_b[k, valid],
                                          axis=1).mean())
        title = (f'ep{epoch:03d} k={k:02d}  N={int(valid.sum())}  '
                 f'pre={err_pre:.2f}→post={err_post:.2f}px')
        log(f'  vis k={k:02d} N={int(valid.sum())}  pre={err_pre:.2f}  post={err_post:.2f}')
        _render(imgs[k], true_uv_b[k], dist_uv_b[k], pred_uv_b[k],
                 sx_b[k], sy_b[k], rho_b[k],
                 out_dir / f'idx{k:02d}.png', title)
