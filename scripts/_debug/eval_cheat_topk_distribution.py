"""For each sample, pick K=20 points with smallest |duv_pred - duv_oracle|
   and dump:
     1. distribution of those residuals (orig px) — per-sample stats
     2. distribution of duv_oracle magnitudes on selected vs unselected pts
     3. (sample 0) where on the image those 20 picked points lie
"""
from __future__ import annotations
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

CACHE = Path.home() / 'cache' / 'kamikado_v3_tiled'
CKPT  = REPO / 'experiments' / 'km_wv_wm_dgx2_n4_img128_8gpu_HEAD' / 'best_model.pt'
EXP_CFG_PATH = REPO / 'experiments' / 'km_wv_wm_dgx2_n4_img128_8gpu_HEAD' / 'config.py'
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

IDX = 17; ROT = 0.30; T_M = 0.05; N_EVAL = 200; SEED = 7 + 1000
K = 20


def _load_cfg():
    src = EXP_CFG_PATH.read_text(); ns = {}; exec(src, ns, ns); return ns['CFG']


def _build(cfg):
    return CalibNetDepth(
        img_size=cfg['img_size'], in_channels=cfg['in_channels'],
        n_layers=cfg['n_layers'], self_first=cfg.get('self_first', False),
        use_convnext=cfg.get('use_convnext', True),
        use_frustum=cfg.get('use_frustum', True),
        deform_mode=cfg.get('deform_mode', 'sl'),
        convnext_n_blocks=cfg.get('convnext_n_blocks', 2),
        convnext_fine_d=cfg.get('convnext_fine_d', None),
        convnext_stem_d=cfg.get('convnext_stem_d', None),
        use_info_head=True,
    )


def main():
    cfg = _load_cfg()
    ds = PandaSetCalibDatasetFull(
        cache_dir=CACHE, split='val', img_size=cfg['img_size'],
        min_crop_px=cfg['min_crop_px'], max_crop_px=cfg['max_crop_px'],
        max_offset_m=0.0, max_rot_deg=0.0, oversample=1,
        grid_n=cfg.get('grid_n', 16), center_band=0.0, preload=False,
    )
    rng = np.random.RandomState(SEED)
    wins = []
    while len(wins) < N_EVAL:
        ox = float(rng.uniform(-ROT, ROT)); oy = float(rng.uniform(-ROT, ROT))
        oz = float(rng.uniform(-ROT, ROT))
        ypr = np.array([oz, oy, ox], dtype=np.float64)
        t = (rng.uniform(-1.0, 1.0, size=3) * T_M).astype(np.float64)
        win = ds.apply_perturbation_explicit(IDX, t, ypr)
        if win is None:
            continue
        wins.append(win)
    batch = collate_full(wins)
    moved = [t.to(DEVICE) if torch.is_tensor(t) else t for t in batch]
    (imgs, true_uvd, dist_uvd, pad_mask, vfp,
     bucket_uvd, bucket_valid, _,
     pts_cam_orig, duv_orig, K_orig, cs) = moved
    valid = ~pad_mask
    pad_full = ~valid

    model = _build(cfg).to(DEVICE)
    sd = torch.load(CKPT, map_location='cpu', weights_only=False)
    if isinstance(sd, dict) and ('state_dict' in sd or 'model' in sd):
        sd = sd.get('state_dict') or sd.get('model')
    model.load_state_dict(sd, strict=False)
    model.eval()
    point_in = torch.cat([dist_uvd[..., :3], dist_uvd[..., 4:5]], dim=-1)
    with torch.no_grad():
        per_pt, _ = model(imgs.float().div(255.0), point_in,
                           key_padding_mask=pad_mask, vfp=vfp,
                           bucket_uvd=bucket_uvd, bucket_valid=bucket_valid)
    duv_pred_local = per_pt[..., :2]
    if pad_full.any():
        duv_pred_local = duv_pred_local.clone()
        duv_pred_local[pad_full] = 0.0
    scale_l2o = (cs / float(cfg['img_size'])).reshape(-1, 1, 1)
    duv_pred_orig = duv_pred_local * scale_l2o

    score = torch.linalg.vector_norm(duv_pred_orig - duv_orig, dim=-1)  # (B,N)
    score_v = torch.where(valid, score, torch.full_like(score, float('inf')))

    idx_top = torch.topk(score_v, K, dim=-1, largest=False).indices  # (B,K)
    sel = torch.zeros_like(valid); sel.scatter_(1, idx_top, True); sel &= valid

    sel_resid = score[sel].cpu().numpy()       # (B*K,)  selected residuals
    unsel = valid & ~sel
    uns_resid = score[unsel].cpu().numpy()

    oracle_mag = torch.linalg.vector_norm(duv_orig, dim=-1)  # (B,N)
    sel_mag = oracle_mag[sel].cpu().numpy()
    uns_mag = oracle_mag[unsel].cpu().numpy()

    print(f"K={K} per-sample (over {sel.shape[0]} samples)")
    print(f"  selected (K closest)  residual:  "
          f"mean {sel_resid.mean():.3f}  med {np.median(sel_resid):.3f}  "
          f"p95 {np.percentile(sel_resid, 95):.3f}  max {sel_resid.max():.3f} px")
    print(f"  unselected           residual:  "
          f"mean {uns_resid.mean():.3f}  med {np.median(uns_resid):.3f}  "
          f"p95 {np.percentile(uns_resid, 95):.3f}  max {uns_resid.max():.3f} px")
    print(f"  selected   |duv_oracle|:  mean {sel_mag.mean():.3f}  "
          f"med {np.median(sel_mag):.3f}  p95 {np.percentile(sel_mag, 95):.3f} px")
    print(f"  unselected |duv_oracle|:  mean {uns_mag.mean():.3f}  "
          f"med {np.median(uns_mag):.3f}  p95 {np.percentile(uns_mag, 95):.3f} px")

    # per-sample max selected residual
    score_sel = torch.where(sel, score, torch.full_like(score, -float('inf')))
    perB_max = score_sel.max(dim=-1).values.cpu().numpy()
    print(f"  per-sample MAX selected residual:  "
          f"mean {perB_max.mean():.3f}  med {np.median(perB_max):.3f}  "
          f"p95 {np.percentile(perB_max, 95):.3f}  max {perB_max.max():.3f} px")

    # viz: sample 0 image, mark selected vs unselected
    img0 = imgs[0].permute(1, 2, 0).cpu().numpy().astype('uint8')
    Spx = img0.shape[0]; cs0 = float(cs[0].item()); o2l = float(Spx) / cs0
    valid0 = valid[0].cpu().numpy()
    sel0 = sel[0].cpu().numpy()
    dist_uv0 = dist_uvd[0, :, :2].cpu().numpy()
    duv_oracle0 = duv_orig[0].cpu().numpy()
    duv_pred0 = duv_pred_orig[0].cpu().numpy()
    target_uv = dist_uv0 + duv_oracle0 * o2l
    pred_uv = dist_uv0 + duv_pred0 * o2l
    resid0 = np.linalg.norm(duv_pred0 - duv_oracle0, axis=-1)

    fig, ax = plt.subplots(figsize=(7, 7))
    ax.imshow(img0)
    others = valid0 & ~sel0
    ax.scatter(dist_uv0[others, 0], dist_uv0[others, 1],
                s=10, c='gray', alpha=0.4, zorder=3, label='unselected')
    sc = ax.scatter(dist_uv0[sel0, 0], dist_uv0[sel0, 1],
                     c=resid0[sel0], cmap='plasma', vmin=0, vmax=2,
                     s=70, edgecolors='white', linewidths=1.0, zorder=6,
                     label=f'selected K={K}')
    ax.scatter(target_uv[sel0, 0], target_uv[sel0, 1],
                marker='x', c='yellow', s=40, linewidths=1.2, zorder=7,
                label='target (oracle)')
    ax.scatter(pred_uv[sel0, 0], pred_uv[sel0, 1],
                marker='o', facecolors='none', edgecolors='lime',
                s=40, linewidths=1.2, zorder=7, label='pred (uv0+Δuv_pred)')
    cb = fig.colorbar(sc, ax=ax, fraction=0.045)
    cb.set_label('|Δuv_pred − Δuv_oracle| (orig px)')
    ax.set_xlim(0, Spx); ax.set_ylim(Spx, 0); ax.axis('off')
    ax.legend(loc='lower right', fontsize=8)
    ax.set_title(
        f'CHEAT top-{K} on sample 0 — picked points are highest-quality predictions\n'
        f'(per-pt sub-px on selected; check spatial spread for aperture)',
        fontsize=10)
    fig.tight_layout()
    out = REPO / 'scripts' / '_debug' / '_outputs' / 'cheat_top20_sample0.png'
    fig.savefig(out, dpi=120, bbox_inches='tight')
    plt.close(fig)
    print(f"wrote → {out}")


if __name__ == '__main__':
    main()
