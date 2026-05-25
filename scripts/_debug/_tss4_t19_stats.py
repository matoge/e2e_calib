"""Aggregate per-cell duv_pred statistics across ALL t19 tiles (right-edge,
u0=3328 v0=677) of TSS4 v3.

Why: 1 frame の arrows だけだと "たまたま" を否定できない。同じタイル位置で
1426 frame 集めて、親タイル 512×512 を 8×8 cell (=64 px) に bin して、
per-cell の (du, dv) 平均/分散を取る。KB4 calib drift が真に periphery で
一貫していれば、cell ごとに (du, dv) は frame を跨いで似た方向になり、
mean ベクトルは長く・std/|mean| は小さくなる。

出力: quiver (per-cell mean、color = consistency) + stats.csv。

Usage:
  python scripts/_debug/_tss4_t19_stats.py --ckpt-run \
      km_wv_wm_n4_img128_ml_dense_pe_dgx2_200ep_v2 --max-tiles 200
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
import scripts.eval.eval_shared_256x800 as _ev
from scripts.eval.eval_shared_256x800 import (
    _build_model, _build_subwin, DEVICE,
)
from scripts.ba.ba_torch import make_info_from_sigma_rho
from scripts._debug._calib_apply import make_warp_closure


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--cache', type=Path,
                    default=Path('/mnt/fsx/tmp/hfunaya/e2e_calib_cache/tss4_v3_tiled'))
    ap.add_argument('--cs', type=int, default=256, choices=(128, 256, 512))
    ap.add_argument('--ckpt-run', required=True)
    ap.add_argument('--tile-suffix', default='_t19.pt',
                    help='only tiles ending with this are aggregated')
    ap.add_argument('--max-tiles', type=int, default=None,
                    help='cap (debug); None = all')
    ap.add_argument('--batch', type=int, default=8,
                    help='# of TILES per forward (each tile = 4 sub-crops)')
    ap.add_argument('--cell-px', type=int, default=64,
                    help='parent-tile bin size (px). 512/64=8 → 8×8 cells')
    ap.add_argument('--stride', type=int, default=None,
                    help='sliding sub-crop stride in parent px. '
                         'Default: cs/2 (3×3 for cs=256). 64 → 5×5 for cs=256.')
    ap.add_argument('--out', type=Path,
                    default=REPO / 'scripts' / '_debug' / '_outputs')
    ap.add_argument('--apply-fit-json', type=Path, default=None,
                    help='warp every inst K/D/T_gt with this fit json BEFORE '
                         'model inference (= iterate the fit). pts (rear-axle '
                         'lidar) is NOT touched.')
    ap.add_argument('--out-suffix', type=str, default='',
                    help='extra filename suffix (e.g. _iter2)')
    args = ap.parse_args()

    run_dir = REPO / 'experiments' / args.ckpt_run
    _ev.CKPT = run_dir / 'best_model.pt'
    _ev.EXP_CFG_PATH = run_dir / 'config.py'
    cfg = _ev._load_cfg()
    print(f'[stats] ckpt={args.ckpt_run}  img_size={cfg["img_size"]}  '
          f'cs={args.cs}  cell={args.cell_px}px')

    ds_train = PandaSetCalibDatasetFull(
        cache_dir=args.cache, split='train',
        img_size=cfg['img_size'],
        min_crop_px=cfg['min_crop_px'], max_crop_px=cfg['max_crop_px'],
        max_offset_m=0.0, max_rot_deg=0.0,
        oversample=1, grid_n=cfg.get('grid_n', 16),
        center_band=0.0, preload=False,
    )
    ds_val = PandaSetCalibDatasetFull(
        cache_dir=args.cache, split='val',
        img_size=cfg['img_size'],
        min_crop_px=cfg['min_crop_px'], max_crop_px=cfg['max_crop_px'],
        max_offset_m=0.0, max_rot_deg=0.0,
        oversample=1, grid_n=cfg.get('grid_n', 16),
        center_band=0.0, preload=False,
    )
    # ---- optional: load fit json + build warp closure (shared helper) ----
    warp_inst, _fit = make_warp_closure(args.apply_fit_json, log_tag='stats')

    targets = []
    for ds in (ds_train, ds_val):
        for f in ds.fnames:
            if f.endswith(args.tile_suffix):
                targets.append((ds, f))
    if args.max_tiles is not None:
        targets = targets[:args.max_tiles]
    print(f'[stats] tiles to process: {len(targets)}')


    # Sliding sub-crop within parent tile (512×512).
    PT = 512
    stride = args.stride if args.stride is not None else max(1, args.cs // 2)
    last_off = PT - args.cs  # max valid u0/v0 so sub-crop fits
    if last_off < 0:
        raise SystemExit(f'cs={args.cs} larger than parent tile {PT}')
    offs = list(range(0, last_off + 1, stride))
    if offs[-1] != last_off:
        offs.append(last_off)
    u0v0_list = [(u, v) for v in offs for u in offs]
    print(f'[stats] sliding cs={args.cs} stride={stride} → '
          f'{len(offs)}×{len(offs)} = {len(u0v0_list)} sub-crops/tile  '
          f'(offs={offs})')
    # printing duplicate-removed list (remove duplicates while preserving order)
    seen = set(); u0v0_list_dedup = []
    for u0v0 in u0v0_list:
        if u0v0 not in seen:
            seen.add(u0v0); u0v0_list_dedup.append(u0v0)
    u0v0_list = u0v0_list_dedup
    print(f'[stats] sliding sub-crops/tile: {len(u0v0_list)}')

    model = _build_model(cfg).to(DEVICE)
    sd = torch.load(_ev.CKPT, map_location='cpu', weights_only=False)
    if isinstance(sd, dict) and ('state_dict' in sd or 'model' in sd):
        sd = sd.get('state_dict') or sd.get('model')
    model.load_state_dict(sd, strict=False)
    model.eval()
    for p in model.parameters():
        p.requires_grad_(False)
    use_intensity = getattr(model, 'use_intensity', True)

    S = float(cfg['img_size'])
    scale = float(args.cs) / S  # local→parent px per sub-crop

    n_cells = 512 // args.cell_px
    sum_du   = np.zeros((n_cells, n_cells), dtype=np.float64)
    sum_dv   = np.zeros((n_cells, n_cells), dtype=np.float64)
    sum_du2  = np.zeros((n_cells, n_cells), dtype=np.float64)
    sum_dv2  = np.zeros((n_cells, n_cells), dtype=np.float64)
    sum_n    = np.zeros((n_cells, n_cells), dtype=np.int64)
    n_frames_with_cell = np.zeros((n_cells, n_cells), dtype=np.int64)
    # info-matrix weighting: per-cell Σ W (2x2) and Σ W·d (2)
    sum_W   = np.zeros((n_cells, n_cells, 2, 2), dtype=np.float64)
    sum_Wd  = np.zeros((n_cells, n_cells, 2), dtype=np.float64)

    n_done = 0
    t0 = __import__('time').time()
    while n_done < len(targets):
        chunk = targets[n_done:n_done + args.batch]
        wins = []
        u0v0_used = []  # (u0, v0) per sub-crop in the flat batch
        tile_id_per_w = []  # which tile this win belongs to
        for tile_i, (ds, fname) in enumerate(chunk):
            idx = ds.fnames.index(fname)
            inst = ds._load_inst(idx)
            if warp_inst is not None:
                inst = warp_inst(inst)
            for (u0, v0) in u0v0_list:
                w = _build_subwin(ds, inst, np.zeros(3), np.zeros(3),
                                   u0=u0, v0=v0, cs=args.cs)
                if w is not None:
                    wins.append(w)
                    u0v0_used.append((u0, v0))
                    tile_id_per_w.append(tile_i)
        if not wins:
            n_done += len(chunk)
            continue
        moved = [t.to(DEVICE) if torch.is_tensor(t) else t
                 for t in collate_full(wins)]
        (imgs, true_uvd, dist_uvd, pad_mask, vfp,
         bucket_uvd, bucket_valid, _,
         pts_cam_orig, duv_orig, K_orig, cs_b, _delta1) = moved
        valid = ~pad_mask
        if use_intensity:
            point_in = torch.cat([dist_uvd[..., :3], dist_uvd[..., 4:5]], dim=-1)
        else:
            point_in = dist_uvd[..., :3]
        img_norm = imgs.float().div(255.0)
        with torch.no_grad():
            out = model(img_norm, point_in, key_padding_mask=pad_mask, vfp=vfp,
                        bucket_uvd=bucket_uvd, bucket_valid=bucket_valid)
        per_pt = out[0] if isinstance(out, tuple) else out
        duv_pred_local = per_pt[..., :2].detach()
        true_local = true_uvd[..., :2].detach()

        # σ-head → W (2x2) info matrix in LOCAL (S-px) units, then convert to
        # parent-px units. Σ_local has px² unit at S-scale; multiplying by
        # scale² gives Σ_parent. W = Σ⁻¹, so W_parent = W_local / scale².
        sx = per_pt[..., 2].exp()
        sy = per_pt[..., 3].exp()
        rho = per_pt[..., 4].clamp(-0.99, 0.99)
        W_local = make_info_from_sigma_rho(sx, sy, rho).detach()  # (B,N,2,2)
        W_parent = W_local / (scale * scale)
        W_np = W_parent.cpu().numpy()

        true_np = true_local.cpu().numpy()       # (B, N, 2)
        duv_np  = duv_pred_local.cpu().numpy()   # (B, N, 2)
        valid_np = valid.cpu().numpy()           # (B, N)

        # Per tile, accumulate cell stats
        per_tile_seen_cells = {}
        for b in range(true_np.shape[0]):
            tile_i = tile_id_per_w[b]
            u0s, v0s = u0v0_used[b]
            m = valid_np[b]
            u_in = true_np[b, m, 0] * scale + u0s
            v_in = true_np[b, m, 1] * scale + v0s
            du   = duv_np[b, m, 0]  * scale
            dv   = duv_np[b, m, 1]  * scale
            W_b = W_np[b, m]  # (M, 2, 2)
            cu = (u_in // args.cell_px).astype(np.int64)
            cv = (v_in // args.cell_px).astype(np.int64)
            ok = (cu >= 0) & (cu < n_cells) & (cv >= 0) & (cv < n_cells)
            cu = cu[ok]; cv = cv[ok]; du = du[ok]; dv = dv[ok]
            W_b = W_b[ok]
            if cu.size == 0:
                continue
            np.add.at(sum_du,  (cv, cu), du)
            np.add.at(sum_dv,  (cv, cu), dv)
            np.add.at(sum_du2, (cv, cu), du * du)
            np.add.at(sum_dv2, (cv, cu), dv * dv)
            np.add.at(sum_n,   (cv, cu), 1)
            # info-matrix accumulation: Σ W and Σ W·d per cell
            d_b = np.stack([du, dv], axis=-1)  # (M, 2)
            Wd_b = np.einsum('mij,mj->mi', W_b, d_b)  # (M, 2)
            np.add.at(sum_W,  (cv, cu), W_b)
            np.add.at(sum_Wd, (cv, cu), Wd_b)
            seen = per_tile_seen_cells.setdefault(tile_i, set())
            for cvi, cui in zip(cv, cu):
                seen.add((int(cvi), int(cui)))
        for tile_i, cells in per_tile_seen_cells.items():
            for (cvi, cui) in cells:
                n_frames_with_cell[cvi, cui] += 1

        n_done += len(chunk)
        if n_done % (args.batch * 5) == 0 or n_done >= len(targets):
            dt = __import__('time').time() - t0
            print(f'  [{n_done}/{len(targets)}]  ({dt:.0f}s)')

    # Per-cell statistics
    n = sum_n.astype(np.float64)
    n_safe = np.where(n > 0, n, 1.0)
    mean_du = sum_du / n_safe
    mean_dv = sum_dv / n_safe
    var_du  = sum_du2 / n_safe - mean_du ** 2
    var_dv  = sum_dv2 / n_safe - mean_dv ** 2
    var_du  = np.maximum(var_du, 0.0)
    var_dv  = np.maximum(var_dv, 0.0)
    std_du  = np.sqrt(var_du)
    std_dv  = np.sqrt(var_dv)
    mean_norm = np.sqrt(mean_du ** 2 + mean_dv ** 2)
    std_combined = np.sqrt(var_du + var_dv)
    consistency = mean_norm / np.where(std_combined > 1e-3, std_combined, 1e-3)

    # Info-matrix weighted per-cell mean: (Σ W)⁻¹ · Σ W·d
    info_mean_du = np.zeros_like(mean_du)
    info_mean_dv = np.zeros_like(mean_dv)
    info_post_var_du = np.zeros_like(mean_du)
    info_post_var_dv = np.zeros_like(mean_dv)
    info_post_cov_uv = np.zeros_like(mean_du)
    for cv in range(n_cells):
        for cu in range(n_cells):
            if sum_n[cv, cu] < 1:
                continue
            W = sum_W[cv, cu]
            try:
                C = np.linalg.inv(W)
            except np.linalg.LinAlgError:
                continue
            d = C @ sum_Wd[cv, cu]
            info_mean_du[cv, cu] = d[0]
            info_mean_dv[cv, cu] = d[1]
            info_post_var_du[cv, cu] = C[0, 0]
            info_post_var_dv[cv, cu] = C[1, 1]
            info_post_cov_uv[cv, cu] = C[0, 1]
    info_mean_norm = np.sqrt(info_mean_du ** 2 + info_mean_dv ** 2)
    info_post_std = np.sqrt(np.maximum(info_post_var_du + info_post_var_dv, 0.0))
    info_consistency = info_mean_norm / np.where(info_post_std > 1e-3, info_post_std, 1e-3)

    args.out.mkdir(parents=True, exist_ok=True)
    tile_tag = args.tile_suffix.replace('.pt', '').lstrip('_')
    csv_path = args.out / (
        f'tss4_{tile_tag}_stats_{args.ckpt_run}_cs{args.cs}{args.out_suffix}.csv')
    with csv_path.open('w') as f:
        f.write('cv,cu,n_pts,n_frames,mean_du,mean_dv,std_du,std_dv,'
                'mean_norm,std_combined,consistency\n')
        for cv in range(n_cells):
            for cu in range(n_cells):
                f.write(f'{cv},{cu},{int(sum_n[cv,cu])},'
                        f'{int(n_frames_with_cell[cv,cu])},'
                        f'{mean_du[cv,cu]:.4f},{mean_dv[cv,cu]:.4f},'
                        f'{std_du[cv,cu]:.4f},{std_dv[cv,cu]:.4f},'
                        f'{mean_norm[cv,cu]:.4f},'
                        f'{std_combined[cv,cu]:.4f},'
                        f'{consistency[cv,cu]:.4f}\n')
    print(f'[stats] wrote {csv_path}')

    # Quiver figure
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(1, 3, figsize=(21, 7), dpi=130)
    cells_centers_u = (np.arange(n_cells) + 0.5) * args.cell_px
    cells_centers_v = (np.arange(n_cells) + 0.5) * args.cell_px
    UU, VV = np.meshgrid(cells_centers_u, cells_centers_v)

    ok = sum_n >= 5
    ax = axes[0]
    ax.set_facecolor('#e8e8e8')
    ax.set_xlim(0, 512); ax.set_ylim(512, 0)
    ax.set_aspect('equal')
    ax.set_title(f'(a) hard-bin mean duv_pred  cell={args.cell_px}px, '
                 f'{len(targets)} frames\ncolor=|mean|/std',
                 fontsize=10)
    if ok.any():
        sc = ax.quiver(UU[ok], VV[ok], mean_du[ok], mean_dv[ok],
                       consistency[ok],
                       cmap='viridis',
                       angles='xy', scale_units='xy', scale=1.0,
                       width=0.004, headwidth=4, headlength=5)
        cb = fig.colorbar(sc, ax=ax, fraction=0.04, pad=0.02)
        cb.set_label('|mean|/std', fontsize=9)
    for cv in range(n_cells + 1):
        ax.axhline(cv * args.cell_px, color='white', lw=0.4, alpha=0.6)
    for cu in range(n_cells + 1):
        ax.axvline(cu * args.cell_px, color='white', lw=0.4, alpha=0.6)

    ax = axes[1]
    ax.set_facecolor('#e8e8e8')
    ax.set_xlim(0, 512); ax.set_ylim(512, 0)
    ax.set_aspect('equal')
    ax.set_title('(b) info-weighted mean duv_pred  '
                 '(Σ W)⁻¹·Σ W·d\ncolor=|mean|/√tr(Σ_post)',
                 fontsize=10)
    if ok.any():
        sc = ax.quiver(UU[ok], VV[ok],
                       info_mean_du[ok], info_mean_dv[ok],
                       info_consistency[ok],
                       cmap='viridis',
                       angles='xy', scale_units='xy', scale=1.0,
                       width=0.004, headwidth=4, headlength=5)
        cb = fig.colorbar(sc, ax=ax, fraction=0.04, pad=0.02)
        cb.set_label('|mean|/√tr(Σ_post)', fontsize=9)
    for cv in range(n_cells + 1):
        ax.axhline(cv * args.cell_px, color='white', lw=0.4, alpha=0.6)
    for cu in range(n_cells + 1):
        ax.axvline(cu * args.cell_px, color='white', lw=0.4, alpha=0.6)

    ax = axes[2]
    im = ax.imshow(info_mean_norm, origin='upper', cmap='magma',
                   extent=(0, 512, 512, 0))
    ax.set_title('(c) info-weighted |mean duv_pred| per cell', fontsize=10)
    cb = fig.colorbar(im, ax=ax, fraction=0.04, pad=0.02)
    cb.set_label('|mean| (px)', fontsize=9)
    fig.suptitle(f'TSS4 {tile_tag}  ckpt={args.ckpt_run}  '
                 f'cs={args.cs}  N_tiles={len(targets)}', fontsize=10)
    fig.tight_layout()
    png_path = args.out / (
        f'tss4_{tile_tag}_stats_{args.ckpt_run}_cs{args.cs}{args.out_suffix}.png')
    fig.savefig(png_path, dpi=140, bbox_inches='tight')
    plt.close(fig)
    print(f'[stats] wrote {png_path}')

    # Compact terminal summary
    flat = []
    for cv in range(n_cells):
        for cu in range(n_cells):
            if sum_n[cv, cu] >= 5:
                flat.append((cv, cu,
                             mean_du[cv, cu], mean_dv[cv, cu],
                             mean_norm[cv, cu], std_combined[cv, cu],
                             consistency[cv, cu], int(sum_n[cv, cu])))
    flat.sort(key=lambda r: -r[4])
    print(f'\n[stats] top-12 cells by |mean| (parent-px), {len(flat)} active cells:')
    print(f'  cv  cu   mean_du   mean_dv  |mean|    std  cons   n_pts')
    for cv, cu, mu, mv, mn, sd, co, np_ in flat[:12]:
        print(f'  {cv:2d}  {cu:2d}  {mu:+7.2f}  {mv:+7.2f}  {mn:6.2f}  '
              f'{sd:5.2f}  {co:4.2f}   {np_}')

    # Column-wise summary: edge bands vs center
    edge_w = max(1, n_cells // 8)  # ~12.5% on each side
    left  = list(range(0, edge_w))
    right = list(range(n_cells - edge_w, n_cells))
    center = list(range(edge_w, n_cells - edge_w))
    print(f'\n[stats] column-band weighted means (n_cells={n_cells}):')
    for label, cus in [(f'left   (cu={left[0]}..{left[-1]})',  left),
                        (f'center (cu={center[0]}..{center[-1]})', center),
                        (f'right  (cu={right[0]}..{right[-1]})', right)]:
        nm = sum_n[:, cus]
        if nm.sum() == 0:
            continue
        wmu = (sum_du[:, cus]).sum() / nm.sum()
        wmv = (sum_dv[:, cus]).sum() / nm.sum()
        # info-weighted column band: solve (Σ W_band) · μ = Σ Wd_band
        Wb  = sum_W[:, cus].sum(axis=(0, 1))   # (2,2)
        Wdb = sum_Wd[:, cus].sum(axis=(0, 1))  # (2,)
        try:
            ib = np.linalg.solve(Wb, Wdb)
            iwmu, iwmv = float(ib[0]), float(ib[1])
        except np.linalg.LinAlgError:
            iwmu = iwmv = float('nan')
        print(f'  {label}  hard du={wmu:+.3f} dv={wmv:+.3f}   '
              f'info du={iwmu:+.3f} dv={iwmv:+.3f}   '
              f'(n_pts={int(nm.sum())})')


if __name__ == '__main__':
    main()
