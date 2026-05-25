"""Side-by-side σ-ellipse comparison on one anchor tile:

  LEFT  panel:  Σ_σ = (W_σ)^{-1}     ← per-point NLL trained covariance
                                       (this is the OG green ellipse the
                                       network was always producing)

  RIGHT panel:  Σ_learned = (W_learned)^{-1}
                                       ← cov implied by the new pose-trust
                                       info head (no direct supervision)

Both panels share the same scene + arrow overlay (yellow X = GT, red O =
dist input, green O = pred, cyan arrow = network Δuv) so the only
difference is the ellipse: σ-head vs learned-head.

Loads a saved info_head.pt from a prior stream run.
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
from matplotlib.patches import Ellipse

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))

from datasets.pandaset_full import PandaSetCalibDatasetFull, collate_full
from models.model_depth import CalibNetDepth
from scripts.ba.ba_torch import (
    make_info_from_sigma_rho, solve_kb_xyz, project_kb,
    _apply_extrinsic, _K_with_delta, _split_delta,
)

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


def _draw_panel(ax, img, true_uv, dist_uv, pred_uv, Sigma2x2, valid,
                 title, ellipse_color='lime', ellipse_scale=1.0,
                 axis_drop_px=None, axis_fade_px=None,
                 info_top_pct=None, solve_uv=None):
    """If info_top_pct is set (e.g. 0.30), only the points whose major-axis
    is in the smallest top_pct fraction are drawn — i.e. the most-informative
    points. Overrides axis_drop/fade."""
    ax.imshow(img)
    S = img.shape[0]
    # pre-compute per-point major-axis (in display units, i.e. with
    # ellipse_scale applied) so we can pick a percentile threshold.
    majors = np.full(len(valid), np.nan, dtype=np.float64)
    for k in np.where(valid)[0]:
        cov = Sigma2x2[k]
        try:
            w = np.linalg.eigvalsh(cov)
        except np.linalg.LinAlgError:
            continue
        if w[1] <= 0:
            continue
        majors[k] = np.sqrt(w[1]) * ellipse_scale
    if info_top_pct is not None and np.isfinite(majors).any():
        thr = np.nanpercentile(majors, 100.0 * float(info_top_pct))
    else:
        thr = None
    if valid.any():
        # arrows: orange = GT->dist (perturbation), cyan = dist->pred (Δuv)
        for k in np.where(valid)[0]:
            ax.annotate('', xy=(dist_uv[k, 0], dist_uv[k, 1]),
                         xytext=(true_uv[k, 0], true_uv[k, 1]),
                         arrowprops=dict(arrowstyle='->', color='orange',
                                          lw=0.5, alpha=0.7), zorder=2)
            ax.annotate('', xy=(pred_uv[k, 0], pred_uv[k, 1]),
                         xytext=(dist_uv[k, 0], dist_uv[k, 1]),
                         arrowprops=dict(arrowstyle='->', color='cyan',
                                          lw=0.5, alpha=0.85), zorder=3)
        ax.scatter(true_uv[valid, 0], true_uv[valid, 1], s=22, c='yellow',
                    marker='x', linewidths=1.0, zorder=5)
        ax.scatter(dist_uv[valid, 0], dist_uv[valid, 1], s=18,
                    facecolors='none', edgecolors='red', linewidths=0.9,
                    zorder=6)
        ax.scatter(pred_uv[valid, 0], pred_uv[valid, 1], s=18,
                    facecolors='none', edgecolors='lime', linewidths=0.9,
                    zorder=7)
        if solve_uv is not None:
            ax.scatter(solve_uv[valid, 0], solve_uv[valid, 1], s=55,
                        facecolors='none', edgecolors='blue',
                        linewidths=1.6, marker='D', zorder=12)
        # ellipse from Σ at pred — drawn LAST so small ellipses aren't
        # hidden under the green pred markers
        for k in np.where(valid)[0]:
            cu, cv = pred_uv[k]
            cov = Sigma2x2[k]
            try:
                w, V = np.linalg.eigh(cov)
            except np.linalg.LinAlgError:
                continue
            if w[1] <= 0 or w[0] <= 0:
                continue
            ang = np.degrees(np.arctan2(V[1, 1], V[0, 1]))
            major = np.sqrt(w[1]) * ellipse_scale
            minor = np.sqrt(w[0]) * ellipse_scale
            if thr is not None:
                # percentile mode: only draw the "most informative" points
                if not np.isfinite(majors[k]) or majors[k] > thr:
                    continue
                alpha = 0.85
            else:
                if axis_drop_px is not None and major > float(axis_drop_px):
                    continue
                alpha = 0.85
                if axis_fade_px is not None and major > float(axis_fade_px):
                    hi = float(axis_drop_px) if axis_drop_px is not None \
                        else 4.0 * float(axis_fade_px)
                    t = np.clip((major - float(axis_fade_px))
                                 / max(hi - float(axis_fade_px), 1e-6),
                                 0.0, 1.0)
                    alpha = 0.85 * (1.0 - t) + 0.05 * t
            e = Ellipse((cu, cv), 2 * major, 2 * minor,
                         angle=ang, facecolor='none',
                         edgecolor=ellipse_color, lw=0.7, alpha=alpha, zorder=10)
            ax.add_patch(e)
    ax.set_xlim(0, S); ax.set_ylim(S, 0)
    ax.axis('off')
    ax.set_title(title, fontsize=11)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--idx', type=int, required=True)
    ap.add_argument('--head-pt', type=Path, required=True)
    ap.add_argument('--model-pt', type=Path, default=None,
                    help='full model.pt from an unlock run (overrides backbone). '
                         'If omitted uses frozen baseline ckpt + head-pt overlay.')
    ap.add_argument('--ox', type=float, default=0.30)
    ap.add_argument('--oy', type=float, default=0.30)
    ap.add_argument('--out', type=Path, required=True)
    ap.add_argument('--ellipse-scale-learned', type=float, default=1.0,
                    help='extra scale on learned ellipse (default 1.0)')
    ap.add_argument('--axis-drop-px', type=float, default=80.0,
                    help='hide ellipse if major axis exceeds this many px '
                         '(default 80; set 0 to disable)')
    ap.add_argument('--axis-fade-px', type=float, default=25.0,
                    help='start fading ellipse alpha when major axis exceeds '
                         'this (default 25)')
    ap.add_argument('--info-top-pct', type=float, default=None,
                    help='if set (e.g. 0.30), only draw the smallest-major-axis '
                         '(=most informative) top fraction of points per panel. '
                         'Overrides --axis-drop-px / --axis-fade-px.')
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
    print(f"[ell] tile {args.idx} ({ds.fnames[args.idx]})")

    batch = collate_full([win])
    moved = [t.to(DEVICE) if torch.is_tensor(t) else t for t in batch]
    imgs, true_uvd, dist_uvd, pad_mask, vfp, bucket_uvd, bucket_valid = moved[:7]

    model = _build_model(cfg).to(DEVICE)
    if args.model_pt is not None:
        sd = torch.load(args.model_pt, map_location='cpu', weights_only=False)
        if isinstance(sd, dict) and ('state_dict' in sd or 'model' in sd):
            sd = sd.get('state_dict') or sd.get('model')
        model.load_state_dict(sd, strict=False)
        print(f"[ell] loaded FULL model from {args.model_pt}")
    else:
        sd = torch.load(CKPT, map_location='cpu', weights_only=False)
        if isinstance(sd, dict) and ('state_dict' in sd or 'model' in sd):
            sd = sd.get('state_dict') or sd.get('model')
        model.load_state_dict(sd, strict=False)
        head_sd = torch.load(args.head_pt, map_location=DEVICE, weights_only=False)
        model.info_head.load_state_dict(head_sd)
        print(f"[ell] frozen baseline backbone + info_head from {args.head_pt}")
    model.eval()

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

    sx = per_pt[..., 2].exp(); sy = per_pt[..., 3].exp(); rho = per_pt[..., 4]
    W_sigma_full = make_info_from_sigma_rho(sx, sy, rho).detach()  # (B, N, 2, 2)
    with torch.no_grad():
        W_learn_full = model.info_head(q).detach()                 # (B, N, 2, 2)
    W_sigma = W_sigma_full[0]
    W_learn = W_learn_full[0]

    # Σ = W^{-1}
    eye2 = torch.eye(2, device=W_learn.device)[None].expand_as(W_learn)
    W_learn_reg = W_learn + 1e-6 * eye2
    W_sigma_reg = W_sigma + 1e-6 * eye2
    Sig_learn = torch.linalg.inv(W_learn_reg).cpu().numpy()
    Sig_sigma = torch.linalg.inv(W_sigma_reg).cpu().numpy()

    duv = per_pt[..., :2].cpu().numpy()[0]
    true_uv = true_uvd[0, :, :2].cpu().numpy()
    dist_uv = dist_uvd[0, :, :2].cpu().numpy()
    pred_uv = dist_uv + duv
    valid = ~((dist_uv[:, 0] == 0) & (dist_uv[:, 1] == 0))

    # ─── BA solve on this single tile, both with W=σ and W=learned ────
    duv_pred_local_t = per_pt[..., :2].detach()                   # (1,N,2)
    pts_cam_orig = moved[8]; duv_orig = moved[9]
    K_orig = moved[10];      cs = moved[11]
    pad_full_t = pad_mask
    valid_t = ~pad_full_t
    P0_orig = pts_cam_orig.clone()
    if pad_full_t.any():
        duv_pred_local_t = duv_pred_local_t.clone()
        duv_pred_local_t[pad_full_t] = 0.0
        P0_orig[pad_full_t] = torch.tensor(
            [0., 0., 1.], dtype=P0_orig.dtype, device=P0_orig.device)
    inst_idx = ds._load_inst(int(args.idx))
    dist_kb = inst_idx['distortion'].clone().detach().to(
        torch.float32).reshape(1, 4).to(DEVICE).expand(1, 4).contiguous()
    S_local = float(cfg['img_size'])
    scale_l2o = (cs / S_local).reshape(-1, 1, 1)               # (1,1,1)
    inv_l2o = (1.0 / scale_l2o).reshape(-1, 1, 1, 1)
    duv_pred_orig_t = duv_pred_local_t * scale_l2o
    W_sigma_orig = W_sigma_full * inv_l2o.pow(2)
    W_learn_orig = W_learn_full * inv_l2o.pow(2)
    DOFS = ['omega_x', 'omega_y', 'omega_z', 'tx', 'ty', 'tz']
    with torch.no_grad():
        d_sig, _ = solve_kb_xyz(P0_orig, duv_pred_orig_t, W_sigma_orig,
                                 K_orig, dist_kb, DOFS,
                                 valid=valid_t, n_iter=6, damping=1e-3)
        d_lrn, _ = solve_kb_xyz(P0_orig, duv_pred_orig_t, W_learn_orig,
                                 K_orig, dist_kb, DOFS,
                                 valid=valid_t, n_iter=6, damping=1e-3)
        def _proj_after(delta):
            d = _split_delta(delta, DOFS)
            omega = torch.stack([d['omega_x'], d['omega_y'], d['omega_z']], dim=-1)
            t_v   = torch.stack([d['tx'], d['ty'], d['tz']], dim=-1)
            P = _apply_extrinsic(P0_orig, omega, t_v)
            Kn = _K_with_delta(K_orig, d['dfx'], d['dfy'], d['dcx'], d['dcy'])
            return project_kb(P, Kn, dist_kb)                          # (1,N,2) orig-px
        uv_sig_orig = _proj_after(d_sig)
        uv_lrn_orig = _proj_after(d_lrn)
        uv0_orig = project_kb(P0_orig, K_orig, dist_kb)
        uv_target_orig = uv0_orig + duv_orig
    # convert orig-px → local-px (crop origin offset + scaling)
    # discover offset from uv0: dist_uv == uv0_orig*o2l - origin
    o2l_s = float(S_local / float(cs[0].item()))
    uv0_orig_np = uv0_orig[0].cpu().numpy()
    dist_uv_np = dist_uvd[0, :, :2].cpu().numpy()
    valid_np = ~((dist_uv_np[:, 0] == 0) & (dist_uv_np[:, 1] == 0))
    origin_local = (uv0_orig_np * o2l_s - dist_uv_np)[valid_np].mean(axis=0)
    def _o2l(uv_orig_np):
        return uv_orig_np * o2l_s - origin_local
    solve_uv_sigma = _o2l(uv_sig_orig[0].cpu().numpy())
    solve_uv_learn = _o2l(uv_lrn_orig[0].cpu().numpy())
    target_uv_local = _o2l(uv_target_orig[0].cpu().numpy())
    # quick sanity print
    print(f"[ell] cs[0]={float(cs[0].item()):.1f}, S={S_local}, "
          f"o2l = S/cs = {S_local / float(cs[0].item()):.4f}")
    sv = solve_uv_sigma[valid]
    print(f"[ell] solve_uv_sigma range (should be in [0,128]):  "
          f"u [{sv[:,0].min():.1f}, {sv[:,0].max():.1f}]  "
          f"v [{sv[:,1].min():.1f}, {sv[:,1].max():.1f}]")
    sl = solve_uv_learn[valid]
    print(f"[ell] solve_uv_learn range (should be in [0,128]):  "
          f"u [{sl[:,0].min():.1f}, {sl[:,0].max():.1f}]  "
          f"v [{sl[:,1].min():.1f}, {sl[:,1].max():.1f}]")
    pu = pred_uv[valid]
    print(f"[ell] pred_uv (green) range:       "
          f"u [{pu[:,0].min():.1f}, {pu[:,0].max():.1f}]  "
          f"v [{pu[:,1].min():.1f}, {pu[:,1].max():.1f}]")
    # uv0_orig * o2l vs dist_uv (local) to discover crop origin
    uv0_orig_np = uv0_orig[0].cpu().numpy()
    uv0_o2l_np = uv0_orig_np * (S_local / float(cs[0].item()))
    diff_u = (uv0_o2l_np - dist_uv)[valid]
    print(f"[ell] uv0_orig*o2l - dist_uv:  "
          f"mean ({diff_u[:,0].mean():.2f}, {diff_u[:,1].mean():.2f})  "
          f"std ({diff_u[:,0].std():.2f}, {diff_u[:,1].std():.2f})")
    err_sig = np.linalg.norm((solve_uv_sigma - target_uv_local)[valid] *
                              (float(cs[0].item()) / S_local), axis=-1)
    err_lrn = np.linalg.norm((solve_uv_learn - target_uv_local)[valid] *
                              (float(cs[0].item()) / S_local), axis=-1)
    do_nothing = np.linalg.norm(duv_orig[0].cpu().numpy()[valid], axis=-1)
    # green↔blue (pred ↔ solve_uv) — should be how much solver disagrees with the per-point pred
    gb_sig = np.linalg.norm((pred_uv - solve_uv_sigma)[valid] *
                             (float(cs[0].item()) / S_local), axis=-1)
    gb_lrn = np.linalg.norm((pred_uv - solve_uv_learn)[valid] *
                             (float(cs[0].item()) / S_local), axis=-1)
    # green↔yellow (pred ↔ target) — per-point Δuv error (the user's intuition)
    gy = np.linalg.norm((pred_uv - target_uv_local)[valid] *
                         (float(cs[0].item()) / S_local), axis=-1)
    print(f"[ell] sample 0 reproj (orig px):  do-nothing mean {do_nothing.mean():.2f}  "
          f"W=σ mean {err_sig.mean():.2f}  W=learned mean {err_lrn.mean():.2f}")
    print(f"[ell] green↔yellow (per-pt Δuv err):     "
          f"mean {gy.mean():.2f}  med {np.median(gy):.2f}  "
          f"p95 {np.percentile(gy,95):.2f}  max {gy.max():.2f}")
    print(f"[ell] green↔blue   (W=σ disagreement):   "
          f"mean {gb_sig.mean():.2f}  med {np.median(gb_sig):.2f}  "
          f"p95 {np.percentile(gb_sig,95):.2f}  max {gb_sig.max():.2f}")
    print(f"[ell] green↔blue   (W=learn disagreement):"
          f"mean {gb_lrn.mean():.2f}  med {np.median(gb_lrn):.2f}  "
          f"p95 {np.percentile(gb_lrn,95):.2f}  max {gb_lrn.max():.2f}")

    img_one = imgs[0].permute(1, 2, 0).cpu().numpy().astype('uint8')

    # rough size summary so we know the scale gap
    def _rad(Sig):
        ws = np.linalg.eigvalsh(Sig[valid])
        ws = np.clip(ws, 0, None)
        return np.sqrt(ws[:, 1]).mean()
    r_s = _rad(Sig_sigma); r_l = _rad(Sig_learn)
    print(f"[ell] mean major-axis  σ:{r_s:.3f}px  learned:{r_l:.3f}px  "
          f"(scale ratio learned/σ = {r_l/max(r_s,1e-6):.3f})")

    drop_px = float(args.axis_drop_px) if args.axis_drop_px > 0 else None
    fade_px = float(args.axis_fade_px) if args.axis_fade_px > 0 else None
    info_top = float(args.info_top_pct) if args.info_top_pct else None
    fig, axes = plt.subplots(1, 2, figsize=(14, 7.4))
    _draw_panel(axes[0], img_one, true_uv, dist_uv, pred_uv,
                Sig_sigma, valid,
                title=(f'Σ_σ — solve W=σ → reproj '
                       f'mean {err_sig.mean():.2f} px (do-nothing {do_nothing.mean():.2f})'),
                ellipse_color='lime',
                axis_drop_px=drop_px, axis_fade_px=fade_px,
                info_top_pct=info_top, solve_uv=solve_uv_sigma)
    _draw_panel(axes[1], img_one, true_uv, dist_uv, pred_uv,
                Sig_learn, valid,
                title=(f'Σ_learned — solve W=learned → reproj '
                       f'mean {err_lrn.mean():.2f} px'),
                ellipse_color='magenta',
                ellipse_scale=float(args.ellipse_scale_learned),
                axis_drop_px=drop_px, axis_fade_px=fade_px,
                info_top_pct=info_top, solve_uv=solve_uv_learn)
    fig.suptitle(
        f'σ-head vs learned-head — covariance ellipses on val tile {args.idx} '
        f'({ds.fnames[args.idx]})\n'
        f'large ellipse = "this point is uncertain / down-weighted in the BA solver"',
        y=0.99, fontsize=12,
    )
    fig.tight_layout()
    args.out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.out, dpi=120, bbox_inches='tight')
    plt.close(fig)
    print(f"[ell] wrote → {args.out}")


if __name__ == '__main__':
    main()
