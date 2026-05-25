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
    # batch_size=1 so `ds._last_crop` after each yield reflects this sample;
    # we collect per-sample crops + decoded full images for the thumbnail
    # inset. Batching is fine to drop here — vis is N=10ish, not perf-critical.
    return DataLoader(Subset(ds, list(range(n))), batch_size=1,
                       num_workers=0, collate_fn=collate_full,
                       shuffle=False), ds


def _render(img_t, true_uv, dist_uv, pred_uv, sx, sy, rho, out_path: Path,
             title: str = '', thumb_full: np.ndarray | None = None,
             crop_box: tuple | None = None):
    """Render one calibration sample. If thumb_full + crop_box given, draw a
    small full-image preview in the upper-left with a red rectangle showing
    where this crop was taken from."""
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib.patches import Ellipse, Rectangle
    img = img_t.permute(1, 2, 0).cpu().numpy().astype(np.uint8)
    S = img.shape[0]
    # Layout: left thumbnail (where in the full frame this crop is) +
    # right main panel (the prediction image).
    fig = plt.figure(figsize=(8, 6), dpi=110)
    if crop_box is not None:
        ax_t = fig.add_axes([0.02, 0.10, 0.30, 0.80])
        ax = fig.add_axes([0.36, 0.04, 0.62, 0.92])
    else:
        ax_t = None
        ax = fig.add_axes([0.04, 0.04, 0.92, 0.92])
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
    ax.legend(loc='lower right', fontsize=7, framealpha=0.9)
    # Thumbnail: black canvas with a red rectangle = where this crop came
    # from inside the FULL parent frame. crop_box is (u0_parent, v0_parent,
    # cs, parent_W, parent_H) — when parent_W/H aren't known we fall back
    # to (cs, cs) which makes the rectangle fill the box (still informative
    # for crop_in_tile cases where u0/v0 are tile-local).
    if ax_t is not None and crop_box is not None:
        u0, v0, cs = crop_box[:3]
        IW = crop_box[3] if len(crop_box) > 3 else cs
        IH = crop_box[4] if len(crop_box) > 4 else cs
        ax_t.set_facecolor('black')
        ax_t.add_patch(Rectangle((u0, v0), cs, cs, fill=False,
                                   edgecolor='red', linewidth=1.6))
        ax_t.set_xticks([]); ax_t.set_yticks([])
        ax_t.set_xlim(0, IW); ax_t.set_ylim(IH, 0)
        ax_t.set_aspect('equal')
        for s in ax_t.spines.values(): s.set_edgecolor('white'); s.set_linewidth(0.5)
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
    use_intensity = bool(getattr(model, 'use_intensity', False))
    k = 0
    for batch in loader:
        # batch_size=1 so each iteration is one sample with its own
        # ds._last_crop. Decode the full frame for the thumbnail inset.
        imgs, true_uvd, dist_uvd, pad_mask, vfp, b_uvd, b_v = batch[:7]
        crop = getattr(ds, '_last_crop', None)
        # Resolve parent-frame extent + tile offset so the thumbnail rectangle
        # lands at the true position inside the full sensor image. Parent
        # size: 2*cx, 2*cy from K_full (typical pinhole). Tile origin:
        # inst['tile_u0'/'tile_v0'] (0 if cache wasn't tiled).
        parent_box = None
        if crop is not None:
            try:
                inst = ds._load_inst(k)
                K_full = inst['K_full'].numpy()
                pW = int(round(K_full[0, 2] * 2))
                pH = int(round(K_full[1, 2] * 2))
                tu0 = int(inst.get('tile_u0', 0))
                tv0 = int(inst.get('tile_v0', 0))
                parent_box = (crop['u0'] + tu0, crop['v0'] + tv0,
                              crop['cs'], pW, pH)
            except Exception:
                parent_box = None
        imgs_g = imgs.to(device).float().div(255.0)
        dist_uvd_g = dist_uvd.to(device)
        pad_mask_g = pad_mask.to(device); vfp_g = vfp.to(device)
        b_uvd_g = b_uvd.to(device); b_v_g = b_v.to(device)
        point_in = (torch.cat([dist_uvd_g[..., :3], dist_uvd_g[..., 4:5]], -1)
                      if use_intensity else dist_uvd_g[..., :3])
        with torch.no_grad(), torch.autocast(device_type='cuda', dtype=amp_dtype):
            out = model(imgs_g, point_in, key_padding_mask=pad_mask_g,
                        vfp=vfp_g, bucket_uvd=b_uvd_g, bucket_valid=b_v_g)
        per_pt = out[0] if isinstance(out, tuple) else out
        # InfoHead2x2 経路では tuple の最後が (B, N, 2, 2) の W 行列。
        # σ/ρ 経路 (per_pt[..., 2..4]) との後方互換のためどちらにも対応。
        W_t = None
        if isinstance(out, tuple):
            last = out[-1]
            if torch.is_tensor(last) and last.dim() == 4 and last.shape[-2:] == (2, 2):
                W_t = last
        p = per_pt[0].float().cpu().numpy()
        dist_uv = dist_uvd[0, :, :2].numpy()
        true_uv = true_uvd[0, :, :2].numpy()
        pred_uv = dist_uv + p[:, :2]
        if W_t is not None:
            # Σ = W⁻¹ ; clamp det for fp16 robustness, then derive σx, σy, ρ
            Wnp = W_t[0].float().cpu().numpy()
            a = Wnp[..., 0, 0]; b = Wnp[..., 1, 1]; c = Wnp[..., 0, 1]
            det = np.maximum(a * b - c * c, 1e-8)
            cov_xx = b / det; cov_yy = a / det; cov_xy = -c / det
            sx = np.sqrt(np.maximum(cov_xx, 1e-8))
            sy = np.sqrt(np.maximum(cov_yy, 1e-8))
            rho = cov_xy / (sx * sy + 1e-8)
        else:
            sx = np.exp(p[:, 2]); sy = np.exp(p[:, 3]); rho = np.tanh(p[:, 4])
        valid = ~((dist_uv[:, 0] == 0) & (dist_uv[:, 1] == 0))
        if not valid.any():
            k += 1; continue
        err_pre  = float(np.linalg.norm(dist_uv[valid] - true_uv[valid], axis=1).mean())
        err_post = float(np.linalg.norm(pred_uv[valid] - true_uv[valid], axis=1).mean())
        crop_box = parent_box
        cs_title = f' cs={int(crop["cs"])}' if crop is not None else ''
        title = (f'ep{epoch:03d} k={k:02d}{cs_title}  N={int(valid.sum())}  '
                 f'pre={err_pre:.2f}→post={err_post:.2f}px')
        log(f'  vis k={k:02d} N={int(valid.sum())}  pre={err_pre:.2f}  post={err_post:.2f}')
        _render(imgs[0], true_uv, dist_uv, pred_uv,
                 sx, sy, rho,
                 out_dir / f'idx{k:02d}.png', title,
                 thumb_full=None, crop_box=crop_box)
        k += 1
