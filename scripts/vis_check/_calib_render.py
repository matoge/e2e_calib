"""Render one calib-mode (single-frame) sample for debug-sample upload.

Visualises three projections per panel:
- HAT input (dist_uvd) — red x — what the model gets
- GT (true_uvd)       — lime dots — what calib should land on
- PRED (dist + Δuv)   — cyan dots — what the model output

Plus residual lines:
- yellow thin: dist → true (the bias the model must learn)
- cyan thin:   dist → pred (the bias the model predicted)

Returns a small stats dict (n_ok, err_hat, err_pred) so the caller can
log scalars alongside the image.
"""
from __future__ import annotations
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np


def render_one_calib(img_chw, true_uvd, dist_uvd, pred_duv,
                     out_path: Path,
                     img_size: int,
                     pert_vec=None,
                     title_prefix: str = '',
                     pred_sigma=None) -> dict:
    """
    img_chw   : (3, H, W) uint8 numpy or torch tensor
    true_uvd  : (N, 4 or 5) GT-projection [u, v, d, ...]
    dist_uvd  : (N, 4 or 5) HAT-projection (model input) [u, v, d, ...]
    pred_duv  : (N, 2)      predicted (Δu, Δv) — added to dist to get pred
    pert_vec  : (8,) optional perturbation params for title
    pred_sigma: (N, 3) optional [log_sx, log_sy, rho] — drawn as 1-sigma
                ellipses on the predicted points. Without it you cannot see
                whether a bad window KNOWS it is bad; with it, the windows the
                chi2 gate rejects are visibly the ones with fat ellipses.
    """
    if hasattr(img_chw, 'numpy'):
        img_chw = img_chw.numpy()
    if hasattr(true_uvd, 'numpy'):
        true_uvd = true_uvd.numpy()
    if hasattr(dist_uvd, 'numpy'):
        dist_uvd = dist_uvd.numpy()
    if hasattr(pred_duv, 'numpy'):
        pred_duv = pred_duv.numpy()
    if pred_sigma is not None and hasattr(pred_sigma, 'numpy'):
        pred_sigma = pred_sigma.numpy()

    img = img_chw.transpose(1, 2, 0).astype(np.uint8)
    H, W = img.shape[:2]

    in_img = ((true_uvd[:, 0] >= 0) & (true_uvd[:, 0] < W) &
              (true_uvd[:, 1] >= 0) & (true_uvd[:, 1] < H))
    n_ok = int(in_img.sum())
    if n_ok == 0:
        # nothing in frame — still write a placeholder
        fig, ax = plt.subplots(figsize=(6, 6))
        ax.imshow(img); ax.axis('off')
        ax.set_title(f'{title_prefix}  (no points in frame)', fontsize=9)
        fig.savefig(out_path, dpi=110, bbox_inches='tight')
        plt.close(fig)
        return dict(n_ok=0, err_hat=float('nan'), err_pred=float('nan'))

    t = true_uvd[in_img, :2]
    d = dist_uvd[in_img, :2]
    pred = d + pred_duv[in_img, :2]

    err_hat  = float(np.linalg.norm(t - d,   axis=-1).mean())
    err_pred = float(np.linalg.norm(t - pred, axis=-1).mean())

    fig, ax = plt.subplots(figsize=(7, 7))
    ax.imshow(img); ax.axis('off')
    # residual lines (drawn first, below scatter)
    for j in range(len(t)):
        ax.plot([d[j, 0], t[j, 0]], [d[j, 1], t[j, 1]],
                '-', color='yellow', lw=0.4, alpha=0.4)
        ax.plot([d[j, 0], pred[j, 0]], [d[j, 1], pred[j, 1]],
                '-', color='cyan', lw=0.4, alpha=0.6)
    ax.scatter(d[:, 0],    d[:, 1],    s=8, marker='x',
               c='red',  alpha=0.5, label='dist (HAT input)')
    ax.scatter(t[:, 0],    t[:, 1],    s=8,
               c='lime', alpha=0.7, label='true (GT)')
    ax.scatter(pred[:, 0], pred[:, 1], s=8,
               c='cyan', alpha=0.7, label='pred (dist + Δuv)')

    sig_txt = ''
    if pred_sigma is not None:
        from matplotlib.patches import Ellipse
        ps = pred_sigma[in_img]
        sx = np.exp(np.clip(ps[:, 0], -6, 6))
        sy = np.exp(np.clip(ps[:, 1], -6, 6))
        rho = np.tanh(ps[:, 2]) * 0.99 if np.abs(ps[:, 2]).max() > 1.0 else ps[:, 2]
        for j in range(len(pred)):
            cov = np.array([[sx[j]**2, rho[j]*sx[j]*sy[j]],
                            [rho[j]*sx[j]*sy[j], sy[j]**2]])
            w_, V = np.linalg.eigh(cov)
            w_ = np.sqrt(np.maximum(w_, 1e-12))
            ang = np.degrees(np.arctan2(V[1, -1], V[0, -1]))
            ax.add_patch(Ellipse((pred[j, 0], pred[j, 1]),
                                 width=2*w_[-1], height=2*w_[0], angle=ang,
                                 fill=False, ec='cyan', lw=0.5, alpha=0.35))
        sig_txt = (f'   sigma median {np.median(np.sqrt(sx*sy)):.2f}px'
                   f'  (p90 {np.percentile(np.sqrt(sx*sy), 90):.2f})')

    title = (f'{title_prefix}  N={n_ok}\n'
             f'HAT→GT={err_hat:.2f}px   PRED→GT={err_pred:.2f}px{sig_txt}')
    if pert_vec is not None:
        pv = pert_vec.numpy() if hasattr(pert_vec, 'numpy') else pert_vec
        title += (f'\npert: t=({pv[0]:+.2f},{pv[1]:+.2f},{pv[2]:+.2f})m  '
                  f'ypr=({pv[3]:+.2f},{pv[4]:+.2f},{pv[5]:+.2f})°')
    ax.set_title(title, fontsize=9)
    ax.legend(loc='upper right', fontsize=8)
    ax.set_xlim(0, W); ax.set_ylim(H, 0)
    fig.tight_layout()
    fig.savefig(out_path, dpi=110, bbox_inches='tight')
    plt.close(fig)
    return dict(n_ok=n_ok, err_hat=err_hat, err_pred=err_pred)
