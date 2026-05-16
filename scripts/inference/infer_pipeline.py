"""SINGLE source of truth for model inference. ALL viz / eval / BA scripts MUST
go through this module so the input to the model is BYTE-IDENTICAL to training.

The training data path is `PandaSetCalibDatasetFull.__getitem__(idx)`. This
module wraps that and the forward pass — nothing more. Do not re-implement
the crop/perturbation/bucketing logic anywhere else.

Public API:
    ds, c     = make_ds(exp, cache, split='train'|'val', oversample=1)
    model     = load_model(exp)                       # load_calib_model
    result    = infer_one(model, ds, idx)             # one sample, training-identical
    render_red_to_green(result, out_path, top_k=100)  # red→green top-K low σ viz

CLI smoke test:
    python -m scripts.inference.infer_pipeline \
        --exp zod_20260513_clean8k_pixonly \
        --cache /mnt/nvme6t/e2e_calib_cache/zod_v3_tiled_clean \
        --split train --n 1
"""
from __future__ import annotations
import sys, importlib.util
from pathlib import Path

import numpy as np
import torch

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))

from datasets.pandaset_full import PandaSetCalibDatasetFull
from scripts.inference.infer_calib import load_calib_model as load_model


def _load_cfg(exp: str) -> dict:
    cfg_path = REPO_ROOT / 'experiments' / exp / 'config.py'
    spec = importlib.util.spec_from_file_location('_cfg', cfg_path)
    mod = importlib.util.module_from_spec(spec); spec.loader.exec_module(mod)
    return mod.CFG


def make_ds(exp: str, cache: str, split: str = 'train', oversample: int = 1):
    """Build PandaSetCalibDatasetFull with the EXACT hparams the experiment trained
    on. Always use this — never hand-construct the dataset elsewhere."""
    c = _load_cfg(exp)
    ds = PandaSetCalibDatasetFull(
        cache, split=split,
        img_size      = c['img_size'],
        min_crop_px   = c.get('min_crop_px', 128),
        max_crop_px   = c.get('max_crop_px', 384),
        max_rot_deg   = c.get('max_rot_deg', 1.5),
        max_offset_m  = c.get('max_offset_m', 0.6),
        max_fx_pct    = c.get('max_fx_pct', 0.0),
        max_fy_pct    = c.get('max_fy_pct', 0.0),
        pose_frame    = c.get('pose_frame', 'orig'),
        grid_n        = c.get('grid_n', 16),
        n_full        = c.get('n_full', 1024),
        k_per_cell    = c.get('k_per_cell', 8),
        oversample    = int(oversample),
    )
    return ds, c


@torch.no_grad()
def infer_one(model, ds, idx: int, device: str = 'cuda', seed: int | None = None) -> dict:
    """Run model on ONE training-identical sample via ds.__getitem__(idx).

    Returns:
        idx           : int
        img           : (S, S, 3) uint8 numpy  — the crop the model saw
        hyp_uv        : (Nmax, 2)   — input hypothesis uv in crop-local px
        true_uv       : (Nmax, 2)   — ground-truth uv in crop-local px
        pred_uv       : (Nmax, 2)   — hyp_uv + Δ
        delta         : (Nmax, 2)   — model's Δuv prediction in crop-local px
        sigma_x,y     : (Nmax,)     — model σx, σy per pt (crop-local px)
        rho           : (Nmax,)     — model ρ per pt
        sigma_scalar  : (Nmax,)     — sqrt(sqrt(det Σ)), "radius-equivalent" σ
        is_obj        : (Nmax,) bool — from true_uvd[:, 3] > 0.5
        valid         : (Nmax,) bool — padding mask (true position is real)
    """
    if seed is not None:
        np.random.seed(seed)
    sample = ds[idx]
    img, true_uvd, dist_uvd, vfp, bucket_uvd, bucket_valid = sample[:6]

    img_in = img.unsqueeze(0).to(device).float().div_(255.0)
    Nmax = dist_uvd.shape[0]
    pad = torch.zeros(1, Nmax, dtype=torch.bool, device=device)
    use_intensity = bool(getattr(model, 'use_intensity', False))
    if use_intensity and dist_uvd.shape[-1] >= 5:
        point_in = torch.cat([dist_uvd[..., :3], dist_uvd[..., 4:5]], dim=-1)
    else:
        point_in = dist_uvd[..., :3]
    with torch.autocast(device_type='cuda', dtype=torch.bfloat16):
        out = model(img_in,
                    point_in.unsqueeze(0).to(device),
                    key_padding_mask=pad,
                    vfp=vfp.view(1).to(device),
                    bucket_uvd=bucket_uvd.unsqueeze(0).to(device),
                    bucket_valid=bucket_valid.unsqueeze(0).to(device))
    per_pt = out[0] if isinstance(out, tuple) else out
    p = per_pt[0].float().cpu().numpy()  # (Nmax, 5)

    hyp_uv  = dist_uvd[:, :2].numpy().astype(np.float32)
    true_uv = true_uvd[:, :2].numpy().astype(np.float32)
    delta   = p[:, :2].astype(np.float32)
    pred_uv = hyp_uv + delta
    sx = np.exp(p[:, 2]); sy = np.exp(p[:, 3]); rho = np.tanh(p[:, 4])
    det = (sx * sy) ** 2 * np.maximum(1.0 - rho ** 2, 1e-6)
    sigma_scalar = np.sqrt(np.sqrt(det)).astype(np.float32)
    is_obj = (true_uvd[:, 3].numpy() > 0.5)
    # padded slots: training fills unused with zeros — detect via hyp_uv == (0,0)
    valid  = ~((hyp_uv[:, 0] == 0) & (hyp_uv[:, 1] == 0))

    img_np = img.permute(1, 2, 0).cpu().numpy().astype(np.uint8)
    # Re-extract the ORIGINAL (pre-resize) crop for wider-FOV viz.
    # Dataset stores last crop position in _last_crop after __getitem__.
    crop = getattr(ds, '_last_crop', None)
    img_orig = None
    scale = 1.0
    if crop is not None:
        u0, v0, cs = int(crop['u0']), int(crop['v0']), int(crop['cs'])
        scale = float(cs) / float(img.shape[-1])
        inst = ds._load_inst(idx)
        if 'jpg_bytes' in inst:
            import io
            from PIL import Image as _PIL
            full = np.asarray(_PIL.open(io.BytesIO(bytes(inst['jpg_bytes']))).convert('RGB'))
        else:
            full = inst['img'].permute(1, 2, 0).numpy().astype(np.uint8)
        img_orig = full[v0:v0 + cs, u0:u0 + cs].copy()
    # depth (D_norm × 100 m) for BA Jacobians. dist_uvd col 2 is normalized
    # depth (z / 100), put it back into metres.
    z_m = (dist_uvd[:, 2].numpy().astype(np.float32) * 100.0)
    return dict(
        idx=int(idx), img=img_np, img_orig=img_orig, crop=crop, scale=scale,
        hyp_uv=hyp_uv, true_uv=true_uv, pred_uv=pred_uv, delta=delta,
        sigma_x=sx.astype(np.float32), sigma_y=sy.astype(np.float32), rho=rho.astype(np.float32),
        sigma_scalar=sigma_scalar, is_obj=is_obj, valid=valid, z=z_m,
    )


def render_red_to_green(result: dict, out_path: str | Path, top_k: int | None = 100,
                         title_extra: str = '') -> Path:
    """Correction viz on the training-identical crop.

    Draws (hyp, pred, GT) as a connected triplet per point so the correspondence
    is unambiguous:
       red ○       = hyp_uv (perturbed input)
       green ○     = pred_uv (= hyp + Δ)
       yellow ✗    = true GT
       red→green:    correction Δ (yellow arrow)
       green→GT:     residual after correction (cyan SOLID line, opaque)

    top_k:  None or -1  → show ALL valid points (per-tile mode).
            int         → show only the top-K lowest-σ points.
    """
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    img = result['img']; S = img.shape[0]
    valid = result['valid']; sigs = result['sigma_scalar']
    valid_idx = np.where(valid)[0]
    if len(valid_idx) == 0:
        return None
    if top_k is None or top_k < 0:
        keep = valid_idx
        mode_tag = 'ALL'
    else:
        order = valid_idx[np.argsort(sigs[valid_idx])]
        keep = order[:min(top_k, len(order))]
        mode_tag = f'top{len(keep)}'
    hyp = result['hyp_uv'][keep]; pred = result['pred_uv'][keep]
    true_= result['true_uv'][keep]; sigs_k = sigs[keep]
    err_pred_true = np.linalg.norm(pred - true_, axis=1)

    from matplotlib.patches import Ellipse
    # Use the 384-px original crop as background if available (much wider FOV),
    # otherwise fall back to the 128-px model input.
    img_bg = result.get('img_orig') if result.get('img_orig') is not None else img
    s_factor = result.get('scale', 1.0)
    S_render = img_bg.shape[0]
    # Scale all coords from model-local (128) to render space (e.g. 384)
    hyp_r  = hyp  * s_factor
    pred_r = pred * s_factor
    true_r = true_ * s_factor
    sx_k = result['sigma_x'][keep] * s_factor
    sy_k = result['sigma_y'][keep] * s_factor
    rho_k = result['rho'][keep]
    is_obj_k = result['is_obj'][keep]

    fig, ax = plt.subplots(1, 1, figsize=(11, 11), dpi=120)
    ax.imshow(img_bg)
    # 1) correction arrow: hyp → pred
    for (u0, v0), (u1, v1), io_flag in zip(hyp_r, pred_r, is_obj_k):
        col = 'orange' if io_flag else 'yellow'
        ax.annotate('', xy=(u1, v1), xytext=(u0, v0),
                     arrowprops=dict(arrowstyle='->', color=col,
                                      lw=0.8 if io_flag else 0.5,
                                      alpha=0.85 if io_flag else 0.55), zorder=2)
    # 2) residual SEGMENT pred ── GT (cyan / magenta for obj)
    for (u0, v0), (u1, v1), io_flag in zip(pred_r, true_r, is_obj_k):
        col = 'magenta' if io_flag else 'cyan'
        ax.plot([u0, u1], [v0, v1], color=col,
                lw=1.0 if io_flag else 0.7, alpha=0.95, zorder=3)
    # 3) σ ellipse around each pred
    for (cu, cv), sx, sy, rho, io_flag in zip(pred_r, sx_k, sy_k, rho_k, is_obj_k):
        cov = np.array([[sx*sx, rho*sx*sy], [rho*sx*sy, sy*sy]])
        w, V = np.linalg.eigh(cov)
        ang = np.degrees(np.arctan2(V[1, 1], V[0, 1]))
        e = Ellipse((cu, cv), 2*np.sqrt(max(w[1], 1e-6)), 2*np.sqrt(max(w[0], 1e-6)),
                    angle=ang, facecolor='none',
                    edgecolor='lime' if io_flag else 'lime',
                    lw=0.5 if io_flag else 0.3, alpha=0.55 if io_flag else 0.35, zorder=4)
        ax.add_patch(e)
    # markers: differentiate obj/bg
    obj_m = is_obj_k.astype(bool)
    bg_m  = ~obj_m
    if obj_m.any():
        ax.scatter(hyp_r[obj_m, 0], hyp_r[obj_m, 1], s=44, facecolors='none',
                    edgecolors='red', linewidths=1.6, zorder=5,
                    label=f'hyp obj ({obj_m.sum()})')
        ax.scatter(pred_r[obj_m, 0], pred_r[obj_m, 1], s=44, facecolors='none',
                    edgecolors='lime', linewidths=1.6, zorder=6,
                    label=f'pred obj')
        ax.scatter(true_r[obj_m, 0], true_r[obj_m, 1], s=64, c='magenta', marker='x',
                    linewidths=2.0, zorder=7, label='GT obj')
    if bg_m.any():
        ax.scatter(hyp_r[bg_m, 0], hyp_r[bg_m, 1], s=18, facecolors='none',
                    edgecolors='red', linewidths=0.7, alpha=0.7, zorder=5,
                    label=f'hyp bg ({bg_m.sum()})')
        ax.scatter(pred_r[bg_m, 0], pred_r[bg_m, 1], s=18, facecolors='none',
                    edgecolors='lime', linewidths=0.7, alpha=0.7, zorder=6,
                    label=f'pred bg')
        ax.scatter(true_r[bg_m, 0], true_r[bg_m, 1], s=28, c='yellow', marker='x',
                    linewidths=1.0, alpha=0.85, zorder=7, label='GT bg')
    ax.set_xlim(0, S_render); ax.set_ylim(S_render, 0); ax.axis('off')
    ax.set_title(f"idx={result['idx']}  {mode_tag} of {len(valid_idx)} valid pts  "
                  f"σ {sigs_k.min():.2f}-{sigs_k.max():.2f}px  "
                  f"|pred-GT| {err_pred_true.mean():.2f}±{err_pred_true.std():.2f}px"
                  f"{title_extra}",
                  fontsize=9)
    ax.legend(loc='lower right', fontsize=8, framealpha=0.85)
    plt.tight_layout(pad=0.2)
    out_path = Path(out_path)
    fig.savefig(out_path, dpi=96)
    plt.close(fig)
    return out_path


def main():
    import argparse, json
    ap = argparse.ArgumentParser()
    ap.add_argument('--exp', required=True)
    ap.add_argument('--cache', required=True)
    ap.add_argument('--split', choices=('train', 'val'), default='train')
    ap.add_argument('--n', type=int, default=1)
    ap.add_argument('--top-k', type=int, default=100,
                    help='top-K lowest-σ points to draw. -1 = ALL (per-tile mode)')
    ap.add_argument('--idxs', default=None)
    ap.add_argument('--seed', type=int, default=0)
    ap.add_argument('--out', default=None)
    args = ap.parse_args()

    print(f'load model: {args.exp}')
    model = load_model(args.exp)
    model.eval()
    print(f'build dataset: {args.cache} ({args.split})')
    ds, c = make_ds(args.exp, args.cache, args.split, oversample=1)
    print(f'  len(ds)={len(ds)}, img_size={c["img_size"]}')

    if args.idxs:
        idxs = [int(x) for x in args.idxs.split(',')]
    else:
        rng = np.random.default_rng(args.seed)
        idxs = list(rng.choice(len(ds), size=min(args.n, len(ds)), replace=False))

    out = Path(args.out) if args.out else (REPO_ROOT / 'experiments' / args.exp
                                            / f'vis_{args.split}_redgreen_top{args.top_k}')
    out.mkdir(parents=True, exist_ok=True)
    for old in out.glob('*.png'): old.unlink()

    for idx in idxs:
        r = infer_one(model, ds, int(idx), seed=int(idx))
        n_valid = int(r['valid'].sum())
        sigs_valid = r['sigma_scalar'][r['valid']]
        # pixel error before/after correction (true vs hyp / pred)
        err_b = float(np.linalg.norm(r['hyp_uv'][r['valid']]  - r['true_uv'][r['valid']], axis=1).mean())
        err_a = float(np.linalg.norm(r['pred_uv'][r['valid']] - r['true_uv'][r['valid']], axis=1).mean())
        print(f'  idx={idx}: N_valid={n_valid}  '
              f'σ range {sigs_valid.min():.2f}-{sigs_valid.max():.2f}px  '
              f'err hyp→true={err_b:.2f}  pred→true={err_a:.2f}px')
        out_path = out / f'idx{int(idx):06d}.png'
        render_red_to_green(r, out_path, top_k=args.top_k,
                             title_extra=f'  err b→a: {err_b:.2f}→{err_a:.2f}px')
    print(f'done → {out}')


if __name__ == '__main__':
    main()
