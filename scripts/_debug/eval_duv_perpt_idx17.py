"""Per-point Δuv prediction error (orig px) on idx=17.

This is the *upstream* number — what the σ-ellipse panel was implicitly
showing as small green ellipses on the truck. We want to confirm that on
the truck region the network's per-point Δuv is sub-pixel, even though
the solver-pose-then-reproject distance is several px (because the solver
mixes good points with bad ones)."""
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

IDX = 17
ROT = 0.30
T_M = 0.05
N_EVAL = 200
SEED = 7 + 1000


def _load_cfg():
    src = EXP_CFG_PATH.read_text(); ns = {}; exec(src, ns, ns); return ns['CFG']


def _build(cfg):
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

    model = _build(cfg).to(DEVICE)
    sd = torch.load(CKPT, map_location='cpu', weights_only=False)
    if isinstance(sd, dict) and ('state_dict' in sd or 'model' in sd):
        sd = sd.get('state_dict') or sd.get('model')
    model.load_state_dict(sd, strict=False)
    model.eval()

    point_in = torch.cat([dist_uvd[..., :3], dist_uvd[..., 4:5]], dim=-1)
    img_norm = imgs.float().div(255.0)
    with torch.no_grad():
        out = model(img_norm, point_in, key_padding_mask=pad_mask, vfp=vfp,
                    bucket_uvd=bucket_uvd, bucket_valid=bucket_valid)
    per_pt = out[0]
    duv_pred_local = per_pt[..., :2].detach()              # (B,N,2) local px

    S = float(cfg['img_size'])
    scale_l2o = (cs / S).reshape(-1, 1, 1)                  # (B,1,1)
    duv_pred_orig = duv_pred_local * scale_l2o              # (B,N,2) orig px
    duv_oracle_orig = duv_orig                              # already orig px

    # Per-point residual (orig px)
    resid = torch.linalg.vector_norm(
        duv_pred_orig - duv_oracle_orig, dim=-1)            # (B,N)
    resid_v = resid[valid].cpu().numpy()
    print(f"[duv-perpt] valid pts: {resid_v.size}  "
          f"(B={resid.shape[0]}, N={resid.shape[1]})")
    print(f"  mean   {resid_v.mean():.3f} px")
    print(f"  median {np.median(resid_v):.3f} px")
    print(f"  p25    {np.percentile(resid_v, 25):.3f} px")
    print(f"  p75    {np.percentile(resid_v, 75):.3f} px")
    print(f"  p90    {np.percentile(resid_v, 90):.3f} px")
    print(f"  p95    {np.percentile(resid_v, 95):.3f} px")
    print(f"  max    {resid_v.max():.3f} px")
    frac_subpx = (resid_v < 1.0).mean()
    frac_2px   = (resid_v < 2.0).mean()
    print(f"  frac <1px {frac_subpx*100:.1f}%   <2px {frac_2px*100:.1f}%")

    # Mean Δuv-magnitude (oracle) — a sense of "what does the perturbation push"
    mag_oracle = torch.linalg.vector_norm(
        duv_oracle_orig, dim=-1)[valid].cpu().numpy()
    print(f"[oracle] mean |Δuv| {mag_oracle.mean():.2f} px  "
          f"median {np.median(mag_oracle):.2f} px  "
          f"max {mag_oracle.max():.2f} px")

    # Viz on sample 0 — colour each LiDAR point by its Δuv residual (orig px)
    img0 = imgs[0].permute(1, 2, 0).cpu().numpy().astype('uint8')
    Spx = img0.shape[0]
    cs0 = float(cs[0].item())
    o2l = float(Spx) / cs0
    valid0 = valid[0].cpu().numpy()
    duv_oracle0 = duv_oracle_orig[0].cpu().numpy()
    duv_pred0 = duv_pred_orig[0].cpu().numpy()
    dist_uv0_local = dist_uvd[0, :, :2].cpu().numpy()  # local px (image coords)
    pred_uv_local = dist_uv0_local + (duv_pred0 * o2l)
    true_uv_local = dist_uv0_local + (duv_oracle0 * o2l)
    resid0 = np.linalg.norm(duv_pred0 - duv_oracle0, axis=-1)  # orig px

    fig, axes = plt.subplots(1, 2, figsize=(13, 6.5))
    ax = axes[0]
    ax.imshow(img0)
    sc = ax.scatter(dist_uv0_local[valid0, 0], dist_uv0_local[valid0, 1],
                     c=resid0[valid0], cmap='viridis_r', vmin=0.0, vmax=3.0,
                     s=22, edgecolors='black', linewidths=0.3, zorder=5)
    cb = fig.colorbar(sc, ax=ax, fraction=0.045)
    cb.set_label('|Δuv_pred − Δuv_oracle|  (orig px)', fontsize=9)
    ax.set_xlim(0, Spx); ax.set_ylim(Spx, 0); ax.axis('off')
    ax.set_title('per-point Δuv residual on sample 0\n'
                 '(yellow → green = sub-px, viridis_r so dark = bad)',
                 fontsize=10)

    ax = axes[1]
    ax.imshow(img0)
    for k in np.where(valid0)[0]:
        ax.plot([true_uv_local[k, 0], pred_uv_local[k, 0]],
                [true_uv_local[k, 1], pred_uv_local[k, 1]],
                color='cyan', lw=0.4, alpha=0.6, zorder=2)
    ax.scatter(true_uv_local[valid0, 0], true_uv_local[valid0, 1],
                s=22, c='yellow', marker='x', linewidths=1.0, zorder=6,
                label='target = uv0+Δuv_oracle')
    ax.scatter(pred_uv_local[valid0, 0], pred_uv_local[valid0, 1],
                s=14, facecolors='none', edgecolors='lime', linewidths=0.9,
                zorder=7, label='uv_pred (network Δuv)')
    ax.set_xlim(0, Spx); ax.set_ylim(Spx, 0); ax.axis('off')
    ax.set_title('per-point pred vs target  (cyan = residual line)',
                 fontsize=10)
    ax.legend(loc='upper right', fontsize=8)

    fig.suptitle(
        f'Per-point Δuv residual on val tile {IDX} ({ds.fnames[IDX]})  — '
        f'frozen baseline ckpt, no info_head used  '
        f'(rot=±{ROT}°, t=±{T_M}m, N={N_EVAL})',
        y=1.0, fontsize=11,
    )
    fig.tight_layout()
    out = REPO / 'scripts' / '_debug' / '_outputs' / 'duv_perpt_idx17.png'
    fig.savefig(out, dpi=120, bbox_inches='tight')
    plt.close(fig)
    print(f"[duv-perpt] wrote → {out}")


if __name__ == '__main__':
    main()
