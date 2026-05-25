"""Render one perturbed sample on a chosen val tile, using the trainer's
existing arrow-style overlay. Shows:
    yellow X    : GT projection
    red O       : dist (perturbed input — what the network sees)
    cyan arrow  : dist → pred (network's Δuv correction)
    orange arrow: GT → dist (the perturbation we added)
    green ring  : pred + σ ellipse (per-point NLL covariance)

This is the existing per-point picture from `scripts/util/vis.py:_render`
on the SAME ckpt + tile we use in overfit_2dof_ba_stream.py — to confirm
the network already moves dist back toward GT.
"""
from __future__ import annotations
import argparse
import sys
from pathlib import Path

import numpy as np
import torch

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))

from datasets.pandaset_full import PandaSetCalibDatasetFull, collate_full
from models.model_depth import CalibNetDepth
from scripts.util.vis import _render

CACHE = Path.home() / 'cache' / 'kamikado_v3_tiled'
CKPT  = REPO / 'experiments' / 'km_wv_wm_dgx2_n4_img128_8gpu_HEAD' / 'best_model.pt'
EXP_CFG_PATH = REPO / 'experiments' / 'km_wv_wm_dgx2_n4_img128_8gpu_HEAD' / 'config.py'
OUT  = REPO / 'scripts' / '_debug' / '_outputs' / 'dist_predict'

PITCH_DEG = 0.30   # ω_x (cam-x rotation) applied for the visualization
YAW_DEG   = 0.30   # ω_y
SEED = 7

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def _load_cfg():
    src = EXP_CFG_PATH.read_text(); ns = {}; exec(src, ns, ns); return ns['CFG']


def _build_model(cfg):
    return CalibNetDepth(
        img_size=cfg['img_size'],
        in_channels=cfg['in_channels'],
        n_layers=cfg['n_layers'],
        self_first=cfg.get('self_first', False),
        use_convnext=cfg.get('use_convnext', True),
        use_frustum=cfg.get('use_frustum', True),
        deform_mode=cfg.get('deform_mode', 'sl'),
        convnext_n_blocks=cfg.get('convnext_n_blocks', 2),
        convnext_fine_d=cfg.get('convnext_fine_d', None),
        convnext_stem_d=cfg.get('convnext_stem_d', None),
        use_info_head=True,
    )


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--idx', type=int, required=True)
    ap.add_argument('--ox', type=float, default=PITCH_DEG)
    ap.add_argument('--oy', type=float, default=YAW_DEG)
    args = ap.parse_args()

    torch.manual_seed(SEED); np.random.seed(SEED)
    OUT.mkdir(parents=True, exist_ok=True)

    cfg = _load_cfg()
    ds = PandaSetCalibDatasetFull(
        cache_dir=CACHE, split='val',
        img_size=cfg['img_size'],
        min_crop_px=cfg['min_crop_px'],
        max_crop_px=cfg['max_crop_px'],
        max_offset_m=0.0, max_rot_deg=0.0,
        oversample=1, grid_n=cfg.get('grid_n', 16),
        center_band=0.0, preload=False,
    )
    print(f"[vis] dataset: {len(ds.fnames)} val instances")

    # ypr layout: (cam-z=0, cam-y=ω_y, cam-x=ω_x)  — same as stream.
    ypr = np.array([0.0, float(args.oy), float(args.ox)], dtype=np.float64)
    win = ds.apply_perturbation_explicit(int(args.idx), np.zeros(3), ypr)
    assert win is not None, f"idx={args.idx} returns None"
    print(f"[vis] tile {args.idx} ({ds.fnames[args.idx]})  ω=(x,y)=({args.ox:+.2f},{args.oy:+.2f})°")

    batch = collate_full([win])
    imgs, true_uvd, dist_uvd, pad_mask, vfp, bucket_uvd, bucket_valid, _pert = [
        t.to(DEVICE) if torch.is_tensor(t) else t for t in batch
    ]

    model = _build_model(cfg).to(DEVICE)
    sd = torch.load(CKPT, map_location='cpu', weights_only=False)
    if isinstance(sd, dict) and ('state_dict' in sd or 'model' in sd):
        sd = sd.get('state_dict') or sd.get('model')
    model.load_state_dict(sd, strict=False)
    model.eval()

    img_norm = imgs.float().div(255.0)
    point_in = torch.cat([dist_uvd[..., :3], dist_uvd[..., 4:5]], dim=-1)
    with torch.no_grad():
        out = model(img_norm, point_in, key_padding_mask=pad_mask, vfp=vfp,
                    bucket_uvd=bucket_uvd, bucket_valid=bucket_valid)
    if isinstance(out, tuple):
        per_pt = out[0]
    else:
        per_pt = out
    duv     = per_pt[..., :2].cpu().numpy()[0]                # (N, 2)
    log_sx  = per_pt[..., 2].cpu().numpy()[0]
    log_sy  = per_pt[..., 3].cpu().numpy()[0]
    rho     = per_pt[..., 4].cpu().numpy()[0]
    sx = np.exp(log_sx); sy = np.exp(log_sy)

    true_uv = true_uvd[0, :, :2].cpu().numpy()
    dist_uv = dist_uvd[0, :, :2].cpu().numpy()
    pred_uv = dist_uv + duv

    # mask invalid: collate_full pads with (0,0) — _render already filters
    out_path = OUT / f'tile{args.idx:02d}_ox{args.ox:+.2f}_oy{args.oy:+.2f}.png'
    _render(imgs[0], true_uv, dist_uv, pred_uv, sx, sy, rho,
            out_path,
            title=(f'tile {args.idx} ({ds.fnames[args.idx]})  '
                   f'ω=({args.ox:+.2f}°,{args.oy:+.2f}°)\n'
                   f'orange = GT→dist (perturbation), cyan = dist→pred (network Δuv)'))
    print(f"[vis] wrote → {out_path}")

    # ─── per-point residual heatmap ‖pred - GT‖ ────────────────────
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    valid = ~((dist_uv[:, 0] == 0) & (dist_uv[:, 1] == 0))
    res = np.linalg.norm(pred_uv - true_uv, axis=1)               # px
    res_clip = np.clip(res, 0, np.percentile(res[valid], 95))      # cap outliers
    img_np = imgs[0].permute(1, 2, 0).cpu().numpy().astype(np.uint8)

    fig, axes = plt.subplots(1, 2, figsize=(11, 5.4))
    for ax, vec, ttl in [
        (axes[0], np.linalg.norm(dist_uv - true_uv, axis=1),
            'pre-network: ‖dist − GT‖ (perturbation residual)'),
        (axes[1], res_clip,
            'post-network: ‖pred − GT‖ (residual after Δuv correction)'),
    ]:
        ax.imshow(img_np)
        if valid.any():
            sc = ax.scatter(dist_uv[valid, 0], dist_uv[valid, 1],
                            c=vec[valid], cmap='hot', s=14, edgecolors='none',
                            alpha=0.85, vmin=0, vmax=res_clip[valid].max())
            cb = plt.colorbar(sc, ax=ax, fraction=0.045, pad=0.02)
            cb.set_label('px', fontsize=8); cb.ax.tick_params(labelsize=7)
        ax.set_title(ttl, fontsize=9)
        ax.axis('off')
    fig.suptitle(
        f'tile {args.idx} ({ds.fnames[args.idx]})  ω=({args.ox:+.2f}°,{args.oy:+.2f}°)\n'
        f'left: how far the input is from GT  |  right: how far the network output is from GT',
        y=0.99, fontsize=10,
    )
    fig.tight_layout()
    out_resid = OUT / f'tile{args.idx:02d}_ox{args.ox:+.2f}_oy{args.oy:+.2f}_residual.png'
    fig.savefig(out_resid, dpi=110, bbox_inches='tight')
    plt.close(fig)
    print(f"[vis] wrote → {out_resid}")
    print(f"[vis] residual stats — pre: mean={np.linalg.norm(dist_uv-true_uv,axis=1)[valid].mean():.3f}px  "
          f"post: mean={res[valid].mean():.3f}px  "
          f"post p50={np.median(res[valid]):.3f}px  p90={np.percentile(res[valid],90):.3f}px")


if __name__ == '__main__':
    main()
