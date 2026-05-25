"""TSS4 全画像 (parent 3840x1952) で per-cell duv_pred を集計 — sliding
sub-crop 9× per parent-tile × 30 parent-tile positions × 1426 frames を
info-matrix weighting で aggregate する。t19 単独の延長: 全画像版。

Output:
  - <out>/tss4_full_frame_<ckpt>_cs<cs>.npz : sum_n / sum_du / sum_dv /
    sum_du2 / sum_dv2 / sum_W / sum_Wd (parent-grid cell)
  - <out>/tss4_full_frame_<ckpt>_cs<cs>.png : 1×3 quiver (hard / info /
    info-heatmap) on parent canvas
"""
from __future__ import annotations
import argparse
import sys
import time
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))

from datasets.pandaset_full import PandaSetCalibDatasetFull, collate_full
import scripts.eval.eval_shared_256x800 as _ev
from scripts.eval.eval_shared_256x800 import _build_model, _build_subwin, DEVICE
from scripts.ba.ba_torch import make_info_from_sigma_rho
from scripts._debug._calib_apply import make_warp_closure


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--cache', type=Path,
                    default=Path('/mnt/fsx/tmp/hfunaya/e2e_calib_cache/tss4_v3_tiled'))
    ap.add_argument('--cs', type=int, default=256, choices=(128, 256, 512))
    ap.add_argument('--ckpt-run', required=True)
    ap.add_argument('--cell-px', type=int, default=32,
                    help='parent-IMAGE bin (px). 3840/32=120 cells wide')
    ap.add_argument('--max-frames-per-tile', type=int, default=None,
                    help='cap (debug)')
    ap.add_argument('--batch', type=int, default=8,
                    help='# of PARENT TILES per forward (each = 9 sub-crops)')
    ap.add_argument('--prefetch-workers', type=int, default=8,
                    help='CPU threads for LMDB read + sub-crop carving')
    ap.add_argument('--out', type=Path,
                    default=REPO / 'scripts' / '_debug' / '_outputs')
    ap.add_argument('--apply-fit-json', type=Path, default=None,
                    help='warp every inst K/D/T_gt with this 13-DoF fit json '
                         'BEFORE model inference (= iterate the fit). '
                         'pts (rear-axle lidar) is NOT touched.')
    ap.add_argument('--only-tid', type=str, default=None,
                    help='restrict to one parent-tile id (e.g. "19" for t19). '
                         'Used for cheap iter sanity.')
    ap.add_argument('--out-suffix', type=str, default='',
                    help='extra filename suffix (e.g. _t19_iter2)')
    args = ap.parse_args()

    run_dir = REPO / 'experiments' / args.ckpt_run
    _ev.CKPT = run_dir / 'best_model.pt'
    _ev.EXP_CFG_PATH = run_dir / 'config.py'
    cfg = _ev._load_cfg()
    print(f'[full] ckpt={args.ckpt_run}  img_size={cfg["img_size"]}  '
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
    warp_inst, _fit = make_warp_closure(args.apply_fit_json, log_tag='full')

    # Group by parent-tile id ("_tNN") AND collect parent-tile origins.
    tile_groups = defaultdict(list)
    tile_origin = {}
    for ds in (ds_train, ds_val):
        for f in ds.fnames:
            tid = f.split('_t')[-1].replace('.pt', '')
            tile_groups[tid].append((ds, f))
    if args.only_tid is not None:
        keep_tid = args.only_tid.lstrip('t')
        if keep_tid not in tile_groups:
            raise SystemExit(f'tid {keep_tid!r} not in {sorted(tile_groups)}')
        tile_groups = {keep_tid: tile_groups[keep_tid]}
        print(f'[full] only-tid={keep_tid}  ({len(tile_groups[keep_tid])} frames)')
    # learn origins from first frame of each tile group
    for tid, members in tile_groups.items():
        ds0, f0 = members[0]
        idx0 = ds0.fnames.index(f0)
        inst0 = ds0._load_inst(idx0)
        tile_origin[tid] = (int(inst0['tile_u0']), int(inst0['tile_v0']))
    # parent IW, IH = max tile origin + 512 (each cached tile is 512x512)
    IW = max(u for (u, v) in tile_origin.values()) + 512
    IH = max(v for (u, v) in tile_origin.values()) + 512
    print(f'[full] parent IW×IH ≈ {IW}×{IH}  (from {len(tile_origin)} tile origins)')
    print(f'[full] tile positions: {sorted(tile_origin.keys(), key=int)}')

    # Sliding sub-crop within each 512×512 parent tile, stride=128.
    if args.cs == 256:
        offs = [0, 128, 256]              # 3×3 = 9 sub-crops/tile
    elif args.cs == 128:
        offs = [0, 128, 256, 384]         # 4×4 = 16
    else:
        offs = [0]
    u0v0_list = [(u, v) for v in offs for u in offs]
    print(f'[full] sliding sub-crops/parent-tile: {len(u0v0_list)} (cs={args.cs})')

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
    scale = float(args.cs) / S

    nW = (IW + args.cell_px - 1) // args.cell_px
    nH = (IH + args.cell_px - 1) // args.cell_px
    print(f'[full] parent grid: {nW}×{nH} cells (cell={args.cell_px}px)')
    sum_du   = np.zeros((nH, nW), dtype=np.float64)
    sum_dv   = np.zeros((nH, nW), dtype=np.float64)
    sum_du2  = np.zeros((nH, nW), dtype=np.float64)
    sum_dv2  = np.zeros((nH, nW), dtype=np.float64)
    sum_n    = np.zeros((nH, nW), dtype=np.int64)
    sum_W    = np.zeros((nH, nW, 2, 2), dtype=np.float64)
    sum_Wd   = np.zeros((nH, nW, 2), dtype=np.float64)
    sum_X    = np.zeros((nH, nW), dtype=np.float64)  # per-cell hard sum of X (m)
    sum_Y    = np.zeros((nH, nW), dtype=np.float64)
    sum_Z    = np.zeros((nH, nW), dtype=np.float64)

    # Build flat list of (ds, fname, tile_u0, tile_v0, fname_idx).
    # Pre-cache fnames.index() lookup since it's O(N) per call.
    fname_idx_cache = {}
    for ds in (ds_train, ds_val):
        for i, f in enumerate(ds.fnames):
            fname_idx_cache[(id(ds), f)] = i
    flat = []
    for tid, members in tile_groups.items():
        u0t, v0t = tile_origin[tid]
        if args.max_frames_per_tile is not None:
            members = members[:args.max_frames_per_tile]
        for (ds, f) in members:
            idx = fname_idx_cache[(id(ds), f)]
            flat.append((ds, f, u0t, v0t, idx))
    print(f'[full] total parent-tile loads: {len(flat)}  '
          f'prefetch_workers={args.prefetch_workers}')

    # Background prefetch: build sub-crop wins for each chunk on a thread
    # pool while the GPU forward is running.
    from concurrent.futures import ThreadPoolExecutor
    pool = ThreadPoolExecutor(max_workers=args.prefetch_workers)

    def build_chunk(chunk):
        wins, u0v0_used, tile_uv_used = [], [], []
        for (ds, f, u0t, v0t, idx) in chunk:
            inst = ds._load_inst(idx)
            if warp_inst is not None:
                inst = warp_inst(inst)
            for (u0, v0) in u0v0_list:
                w = _build_subwin(ds, inst, np.zeros(3), np.zeros(3),
                                   u0=u0, v0=v0, cs=args.cs)
                if w is not None:
                    wins.append(w)
                    u0v0_used.append((u0, v0))
                    tile_uv_used.append((u0t, v0t))
        return wins, u0v0_used, tile_uv_used

    chunks = [flat[i:i + args.batch] for i in range(0, len(flat), args.batch)]
    # prime pipeline: kick off first N futures up front so the pool is warm
    inflight_n = max(2, args.prefetch_workers)
    futures = []
    for ck in chunks[:inflight_n]:
        futures.append(pool.submit(build_chunk, ck))
    next_submit = inflight_n

    t0 = time.time()
    n_done = 0
    for chunk_i, ck in enumerate(chunks):
        fut = futures[chunk_i]
        wins, u0v0_used, tile_uv_used = fut.result()
        if next_submit < len(chunks):
            futures.append(pool.submit(build_chunk, chunks[next_submit]))
            next_submit += 1
        if not wins:
            n_done += len(ck); continue
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
        sx = per_pt[..., 2].exp()
        sy = per_pt[..., 3].exp()
        rho = per_pt[..., 4].clamp(-0.99, 0.99)
        W_local = make_info_from_sigma_rho(sx, sy, rho).detach()  # (B,N,2,2)
        W_parent = W_local / (scale * scale)

        true_np = true_uvd[..., :2].detach().cpu().numpy()
        duv_np  = duv_pred_local.cpu().numpy()
        valid_np = valid.cpu().numpy()
        W_np    = W_parent.cpu().numpy()
        pts_np  = pts_cam_orig.detach().cpu().numpy()  # (B, N, 3) in metres (orig cam frame)

        for b in range(true_np.shape[0]):
            u0s, v0s = u0v0_used[b]
            u0t, v0t = tile_uv_used[b]
            m = valid_np[b]
            u_par = true_np[b, m, 0] * scale + u0s + u0t
            v_par = true_np[b, m, 1] * scale + v0s + v0t
            du    = duv_np[b, m, 0] * scale
            dv    = duv_np[b, m, 1] * scale
            W_b   = W_np[b, m]
            X_b = pts_np[b, m, 0]
            Y_b = pts_np[b, m, 1]
            Z_b = pts_np[b, m, 2]
            cu = (u_par // args.cell_px).astype(np.int64)
            cv = (v_par // args.cell_px).astype(np.int64)
            ok = (cu >= 0) & (cu < nW) & (cv >= 0) & (cv < nH)
            cu = cu[ok]; cv = cv[ok]; du = du[ok]; dv = dv[ok]; W_b = W_b[ok]
            X_b = X_b[ok]; Y_b = Y_b[ok]; Z_b = Z_b[ok]
            if cu.size == 0:
                continue
            np.add.at(sum_du,  (cv, cu), du)
            np.add.at(sum_dv,  (cv, cu), dv)
            np.add.at(sum_du2, (cv, cu), du * du)
            np.add.at(sum_dv2, (cv, cu), dv * dv)
            np.add.at(sum_n,   (cv, cu), 1)
            np.add.at(sum_X,   (cv, cu), X_b)
            np.add.at(sum_Y,   (cv, cu), Y_b)
            np.add.at(sum_Z,   (cv, cu), Z_b)
            d_b = np.stack([du, dv], axis=-1)
            Wd_b = np.einsum('mij,mj->mi', W_b, d_b)
            np.add.at(sum_W,  (cv, cu), W_b)
            np.add.at(sum_Wd, (cv, cu), Wd_b)

        n_done += len(ck)
        if n_done % (args.batch * 10) == 0 or n_done >= len(flat):
            dt = time.time() - t0
            print(f'  [{n_done}/{len(flat)}]  ({dt:.0f}s)')

    pool.shutdown(wait=False)

    # Per-cell statistics
    n = sum_n.astype(np.float64)
    n_safe = np.where(n > 0, n, 1.0)
    mean_du = sum_du / n_safe
    mean_dv = sum_dv / n_safe
    mean_norm = np.sqrt(mean_du ** 2 + mean_dv ** 2)
    var_du  = np.maximum(sum_du2 / n_safe - mean_du ** 2, 0.0)
    var_dv  = np.maximum(sum_dv2 / n_safe - mean_dv ** 2, 0.0)
    std_combined = np.sqrt(var_du + var_dv)
    cons = mean_norm / np.where(std_combined > 1e-3, std_combined, 1e-3)

    info_mean_du = np.zeros_like(mean_du)
    info_mean_dv = np.zeros_like(mean_dv)
    post_var_uu  = np.zeros_like(mean_du)
    post_var_vv  = np.zeros_like(mean_du)
    for cv in range(nH):
        for cu in range(nW):
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
            post_var_uu[cv, cu] = C[0, 0]
            post_var_vv[cv, cu] = C[1, 1]
    info_mean_norm = np.sqrt(info_mean_du ** 2 + info_mean_dv ** 2)
    info_post_std  = np.sqrt(np.maximum(post_var_uu + post_var_vv, 0.0))
    info_cons = info_mean_norm / np.where(info_post_std > 1e-3, info_post_std, 1e-3)

    args.out.mkdir(parents=True, exist_ok=True)
    npz_path = args.out / (
        f'tss4_full_frame_{args.ckpt_run}_cs{args.cs}{args.out_suffix}.npz')
    np.savez(npz_path,
             sum_du=sum_du, sum_dv=sum_dv, sum_du2=sum_du2, sum_dv2=sum_dv2,
             sum_n=sum_n, sum_W=sum_W, sum_Wd=sum_Wd,
             sum_X=sum_X, sum_Y=sum_Y, sum_Z=sum_Z,
             cell_px=args.cell_px, IW=IW, IH=IH)
    print(f'[full] wrote {npz_path}')

    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    cells_u = (np.arange(nW) + 0.5) * args.cell_px
    cells_v = (np.arange(nH) + 0.5) * args.cell_px
    UU, VV = np.meshgrid(cells_u, cells_v)
    ok = sum_n >= 30  # demand at least 30 pts per cell

    aspect = IH / IW
    fig_w = 21.0
    fig_h = (fig_w / 3) * aspect + 1.0
    fig, axes = plt.subplots(1, 3, figsize=(fig_w, fig_h), dpi=130)
    # (a) hard quiver
    ax = axes[0]
    ax.set_facecolor('#0d0d0d')
    ax.set_xlim(0, IW); ax.set_ylim(IH, 0)
    ax.set_aspect('equal')
    if ok.any():
        sc = ax.quiver(UU[ok], VV[ok], mean_du[ok], mean_dv[ok], cons[ok],
                       cmap='turbo', angles='xy', scale_units='xy',
                       scale=1.0, width=0.0015, headwidth=4, headlength=5)
        cb = fig.colorbar(sc, ax=ax, fraction=0.025, pad=0.01)
        cb.set_label('|mean|/std', fontsize=8)
    ax.set_title(f'(a) hard-bin mean duv_pred  cell={args.cell_px}px',
                 fontsize=9)
    # (b) info quiver
    ax = axes[1]
    ax.set_facecolor('#0d0d0d')
    ax.set_xlim(0, IW); ax.set_ylim(IH, 0)
    ax.set_aspect('equal')
    if ok.any():
        sc = ax.quiver(UU[ok], VV[ok],
                       info_mean_du[ok], info_mean_dv[ok], info_cons[ok],
                       cmap='turbo', angles='xy', scale_units='xy',
                       scale=1.0, width=0.0015, headwidth=4, headlength=5)
        cb = fig.colorbar(sc, ax=ax, fraction=0.025, pad=0.01)
        cb.set_label('|mean|/√tr(Σ_post)', fontsize=8)
    ax.set_title('(b) info-weighted (Σ W)⁻¹·Σ W·d', fontsize=9)
    # (c) info |mean| heatmap
    ax = axes[2]
    masked = np.ma.masked_where(~ok, info_mean_norm)
    im = ax.imshow(masked, origin='upper', cmap='magma',
                    extent=(0, IW, IH, 0))
    cb = fig.colorbar(im, ax=ax, fraction=0.025, pad=0.01)
    cb.set_label('|info-mean| (px)', fontsize=8)
    ax.set_title('(c) info-weighted |mean|', fontsize=9)
    fig.suptitle(f'TSS4 full-frame  ckpt={args.ckpt_run}  cs={args.cs}  '
                 f'cell={args.cell_px}px  parent={IW}×{IH}', fontsize=10)
    fig.tight_layout()
    png_path = args.out / (
        f'tss4_full_frame_{args.ckpt_run}_cs{args.cs}{args.out_suffix}.png')
    fig.savefig(png_path, dpi=140, bbox_inches='tight')
    plt.close(fig)
    print(f'[full] wrote {png_path}')

    # band summary in image-cu coords
    edge_w = max(1, nW // 8)
    print(f'\n[full] image-column band weighted means (nW={nW} cells):')
    for label, cus in [(f'left   (cu=0..{edge_w-1})', list(range(0, edge_w))),
                        (f'mid    (cu={edge_w}..{nW-edge_w-1})',
                            list(range(edge_w, nW - edge_w))),
                        (f'right  (cu={nW-edge_w}..{nW-1})',
                            list(range(nW - edge_w, nW)))]:
        nm = sum_n[:, cus]
        if nm.sum() == 0:
            continue
        wmu = (sum_du[:, cus]).sum() / nm.sum()
        wmv = (sum_dv[:, cus]).sum() / nm.sum()
        Wb  = sum_W[:, cus].sum(axis=(0, 1))
        Wdb = sum_Wd[:, cus].sum(axis=(0, 1))
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
