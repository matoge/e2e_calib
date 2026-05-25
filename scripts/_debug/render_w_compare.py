"""Side-by-side comparison of σ-head log det W vs learned log det W on a
chosen anchor tile, using the saved info_head.pt from a prior stream run.

Layout: top row = source image + depth (small thumbnails), bottom row =
big σ-head | learned panels with shared color range so the two can be
compared at a glance.
"""
from __future__ import annotations
import argparse
import sys
from pathlib import Path

import numpy as np
import torch

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))

from datasets.pandaset_full import PandaSetCalibDatasetFull, collate_full
from models.model_depth import CalibNetDepth
from scripts.ba.ba_torch import make_info_from_sigma_rho

CACHE = Path.home() / 'cache' / 'kamikado_v3_tiled'
CKPT  = REPO / 'experiments' / 'km_wv_wm_dgx2_n4_img128_8gpu_HEAD' / 'best_model.pt'
EXP_CFG_PATH = REPO / 'experiments' / 'km_wv_wm_dgx2_n4_img128_8gpu_HEAD' / 'config.py'
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def _load_cfg():
    src = EXP_CFG_PATH.read_text(); ns = {}; exec(src, ns, ns); return ns['CFG']


def _build_model(cfg):
    return CalibNetDepth(
        img_size=cfg['img_size'], in_channels=cfg['in_channels'],
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
    ap.add_argument('--head-pt', type=Path, required=True,
                    help='path to saved info_head.pt (state_dict)')
    ap.add_argument('--ox', type=float, default=0.30)
    ap.add_argument('--oy', type=float, default=0.30)
    ap.add_argument('--out', type=Path, required=True)
    args = ap.parse_args()

    cfg = _load_cfg()
    ds = PandaSetCalibDatasetFull(
        cache_dir=CACHE, split='val',
        img_size=cfg['img_size'],
        min_crop_px=cfg['min_crop_px'], max_crop_px=cfg['max_crop_px'],
        max_offset_m=0.0, max_rot_deg=0.0,
        oversample=1, grid_n=cfg.get('grid_n', 16),
        center_band=0.0, preload=False,
    )
    ypr = np.array([0.0, float(args.oy), float(args.ox)], dtype=np.float64)
    win = ds.apply_perturbation_explicit(int(args.idx), np.zeros(3), ypr)
    assert win is not None
    print(f"[wcmp] tile {args.idx} ({ds.fnames[args.idx]})")

    batch = collate_full([win])
    imgs, true_uvd, dist_uvd, pad_mask, vfp, bucket_uvd, bucket_valid, _ = [
        t.to(DEVICE) if torch.is_tensor(t) else t for t in batch
    ]

    model = _build_model(cfg).to(DEVICE)
    sd = torch.load(CKPT, map_location='cpu', weights_only=False)
    if isinstance(sd, dict) and ('state_dict' in sd or 'model' in sd):
        sd = sd.get('state_dict') or sd.get('model')
    model.load_state_dict(sd, strict=False)
    model.eval()
    head_sd = torch.load(args.head_pt, map_location=DEVICE, weights_only=False)
    model.info_head.load_state_dict(head_sd)
    print(f"[wcmp] loaded info_head from {args.head_pt}")

    # capture q via hook
    captured = {}
    def _hk(_m, _i, _o):
        captured['q'] = _i[0].detach()
    h = model.info_head.mlp[0].register_forward_hook(_hk)
    point_in = torch.cat([dist_uvd[..., :3], dist_uvd[..., 4:5]], dim=-1)
    img_norm = imgs.float().div(255.0)
    with torch.no_grad():
        out = model(img_norm, point_in, key_padding_mask=pad_mask, vfp=vfp,
                    bucket_uvd=bucket_uvd, bucket_valid=bucket_valid)
    h.remove()
    per_pt = out[0]
    q = captured['q']

    # σ → W
    sx = per_pt[..., 2].exp(); sy = per_pt[..., 3].exp(); rho = per_pt[..., 4]
    W_sigma = make_info_from_sigma_rho(sx, sy, rho).detach()
    with torch.no_grad():
        W_learn = model.info_head(q)

    valid = ~pad_mask
    ld_s = torch.linalg.det(W_sigma).clamp_min(1e-12).log()
    ld_l = torch.linalg.det(W_learn).clamp_min(1e-12).log()
    ld_s_np = ld_s[0][valid[0]].cpu().numpy()
    ld_l_np = ld_l[0][valid[0]].cpu().numpy()
    uv_one = dist_uvd[0, :, :2].cpu().numpy()
    z_one  = dist_uvd[0, :, 2].cpu().numpy() * 100.0
    v_one  = valid[0].cpu().numpy()
    img_one = imgs[0].permute(1, 2, 0).cpu().numpy().astype('float32') / 255.0
    img_one = np.clip(img_one, 0, 1)

    # shared colour range for the σ vs learned panels
    vmin = float(min(ld_s_np.min(), ld_l_np.min()))
    vmax = float(max(ld_s_np.max(), ld_l_np.max()))
    print(f"[wcmp] log det W range — σ:[{ld_s_np.min():+.2f}, {ld_s_np.max():+.2f}]  "
          f"learned:[{ld_l_np.min():+.2f}, {ld_l_np.max():+.2f}]")
    corr = float(np.corrcoef(ld_s_np, ld_l_np)[0, 1])
    print(f"[wcmp] Pearson r = {corr:+.3f}")

    fig = plt.figure(figsize=(15, 11))
    gs = fig.add_gridspec(2, 2, height_ratios=[1, 2.0], hspace=0.18, wspace=0.08)

    ax0 = fig.add_subplot(gs[0, 0])
    ax0.imshow(img_one); ax0.set_title('source image', fontsize=12); ax0.axis('off')

    ax1 = fig.add_subplot(gs[0, 1])
    ax1.imshow(img_one)
    if v_one.any():
        sc = ax1.scatter(uv_one[v_one, 0], uv_one[v_one, 1],
                         c=z_one[v_one], cmap='plasma', s=14,
                         edgecolors='none', alpha=0.9)
        cb = plt.colorbar(sc, ax=ax1, fraction=0.045, pad=0.02)
        cb.set_label('depth Z (m)', fontsize=10)
    ax1.set_title('per-query depth', fontsize=12); ax1.axis('off')

    for col, (ld_np, name, cmap) in enumerate([
        (ld_s_np, 'σ-head log det W (per-point NLL trained)', 'viridis'),
        (ld_l_np, 'learned log det W (pose-trust, no direct supervision)', 'viridis'),
    ]):
        ax = fig.add_subplot(gs[1, col])
        ax.imshow(img_one)
        sc = ax.scatter(uv_one[v_one, 0], uv_one[v_one, 1], c=ld_np,
                        cmap=cmap, s=28, edgecolors='none', alpha=0.92,
                        vmin=vmin, vmax=vmax)
        cb = plt.colorbar(sc, ax=ax, fraction=0.045, pad=0.02)
        cb.set_label('log det W  (high = trusted, low = suppressed)', fontsize=10)
        ax.set_title(name, fontsize=13)
        ax.axis('off')

    fig.suptitle(
        f'σ-head vs learned head — per-query trust on val tile {args.idx} '
        f'({ds.fnames[args.idx]})\n'
        f'shared colour range across both bottom panels  |  Pearson r(log det W) = {corr:+.3f}',
        y=0.97, fontsize=13,
    )
    args.out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.out, dpi=120, bbox_inches='tight')
    plt.close(fig)
    print(f"[wcmp] wrote → {args.out}")


if __name__ == '__main__':
    main()
