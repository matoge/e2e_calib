"""Sliding-window inference + render top-N lowest-σ correction points per frame.

For each selected frame:
  1. Apply a random extrinsic perturbation (configurable per-axis rotation, trans).
  2. Generate overlapping sliding-window crops (window/stride configurable).
  3. Run the trained model on every crop, collect per-point (uv_ref, Δuv, Σ).
  4. Pool ALL points across crops in the frame, sort by sqrt(det Σ),
     take the top-N most confident.
  5. Render the full-image PNG with:
       red ○   = uv_ref (hypothesis = perturbed projection)
       green ○ = uv_ref + Δuv (model-corrected position)
       yellow arrow connects them.

Reuses load_model / project / grid_windows / infer_crop from ba_eval_v3.

Default mode: train split, full training-time perturbation σ.
"High-yaw" mode: bump --max-rot-deg to 3-5 deg, see how the model handles big rotations.

Usage:
  python -m scripts.visualization.vis_sliding_low_sigma \
      --exp zod_20260513_clean8k_pixonly \
      --cache /mnt/nvme6t/e2e_calib_cache/zod_v3_tiled_clean \
      --split train --n-frames 10 --n-top 100 \
      --max-rot-deg 1.5 --max-t-m 0.6 \
      --out experiments/zod_20260513_clean8k_pixonly/vis_train_lowsigma_top100
"""
from __future__ import annotations
import sys, argparse, json, random, time
from pathlib import Path

import numpy as np
import torch
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))

import torch.nn.functional as F
from datasets.pandaset_full import PandaSetCalibDatasetFull
from scripts.ba.ba_eval_v3 import (
    load_model, decode_jpg, project, grid_windows, ypr_to_R,
)


def _bucket_uvd(uvd_full_raw: np.ndarray, S: int, G: int = 16, K_per_cell: int = 8):
    """Replicate PandaSetCalibDatasetFull bucketing (lines 618-646).
    Returns (bucket_uvd (G²,K,3), bucket_valid (G²,K))."""
    cell_S = float(S) / G
    cu = np.clip((uvd_full_raw[:, 0] / cell_S).astype(np.int32), 0, G - 1)
    cv = np.clip((uvd_full_raw[:, 1] / cell_S).astype(np.int32), 0, G - 1)
    cell_id = cv * G + cu
    n_raw = uvd_full_raw.shape[0]
    shuf = np.random.permutation(n_raw)
    sorted_idx = shuf[np.argsort(cell_id[shuf], kind='stable')]
    sorted_uvd = uvd_full_raw[sorted_idx]
    sorted_cid = cell_id[sorted_idx]
    counts = np.bincount(sorted_cid, minlength=G * G)
    cell_starts = np.zeros(G * G + 1, dtype=np.int64)
    cell_starts[1:] = counts.cumsum()
    intra = np.arange(n_raw, dtype=np.int64) - cell_starts[sorted_cid]
    keep_mask = intra < K_per_cell
    slots = intra[keep_mask]
    cells = sorted_cid[keep_mask]
    bucket_uvd  = np.zeros((G * G, K_per_cell, 3), dtype=np.float32)
    bucket_valid = np.zeros((G * G, K_per_cell), dtype=bool)
    bucket_uvd[cells, slots]  = sorted_uvd[keep_mask]
    bucket_valid[cells, slots] = True
    return bucket_uvd, bucket_valid


@torch.no_grad()
def infer_crop_with_bucket(model, img_full, K, pts_world, R_off, t_off,
                            u0, v0, cs, S, min_pts=8):
    """infer_crop variant that also builds bucket_uvd / bucket_valid for use_frustum=True
    models. Returns same dict shape as ba_eval_v3.infer_crop."""
    DEV = next(model.parameters()).device
    IH, IW = img_full.shape[:2]
    if u0 < 0 or v0 < 0 or u0 + cs > IW or v0 + cs > IH:
        return None
    uv_off, z_off, pts_cam_off = project(pts_world, R_off.astype(np.float32),
                                          t_off.astype(np.float32), K)
    in_crop = ((uv_off[:, 0] >= u0) & (uv_off[:, 0] < u0 + cs) &
               (uv_off[:, 1] >= v0) & (uv_off[:, 1] < v0 + cs) &
               (z_off > 0.5))
    if in_crop.sum() < min_pts:
        return None
    sel = np.where(in_crop)[0]
    if len(sel) > 256:
        sel = np.random.default_rng(int(u0 * 1e3 + v0)).choice(sel, 256, replace=False)
    scale = float(S) / float(cs)
    uv_loc = np.stack([(uv_off[sel, 0] - u0) * scale,
                       (uv_off[sel, 1] - v0) * scale], axis=-1)
    z_loc = z_off[sel]
    dist_uvd = np.concatenate([uv_loc, z_loc[:, None]], axis=1).astype(np.float32)
    d_loc = (z_loc / 100.0).astype(np.float32)
    uvd_full_raw = np.concatenate([uv_loc, d_loc[:, None]], axis=1)
    bucket_uvd, bucket_valid = _bucket_uvd(uvd_full_raw, S=S, G=16, K_per_cell=8)

    crop = img_full[v0:v0 + cs, u0:u0 + cs]
    img_t = torch.from_numpy(crop).permute(2, 0, 1).float().unsqueeze(0)
    img_t = F.interpolate(img_t, size=(S, S), mode='bilinear', align_corners=False)
    img_t = (img_t / 255.0).to(DEV)
    Nq = dist_uvd.shape[0]
    pad = torch.zeros(1, Nq, dtype=torch.bool, device=DEV)
    vfp = torch.tensor([float(K[0, 0]) * S / cs], dtype=torch.float32, device=DEV)
    bucket_uvd_t = torch.from_numpy(bucket_uvd).unsqueeze(0).to(DEV)
    bucket_valid_t = torch.from_numpy(bucket_valid).unsqueeze(0).to(DEV)
    with torch.autocast(device_type='cuda', dtype=torch.bfloat16):
        _out = model(img_t,
                      torch.from_numpy(dist_uvd[None]).to(DEV),
                      key_padding_mask=pad,
                      vfp=vfp,
                      bucket_uvd=bucket_uvd_t,
                      bucket_valid=bucket_valid_t)
        per_pt = _out[0] if isinstance(_out, tuple) else _out
        out = per_pt[0].float().cpu().numpy()
    du_loc = out[:, 0]; dv_loc = out[:, 1]
    sx_loc = np.exp(out[:, 2]); sy_loc = np.exp(out[:, 3])
    rho    = np.tanh(out[:, 4])
    du_full = du_loc / scale; dv_full = dv_loc / scale
    sx_full = sx_loc / scale; sy_full = sy_loc / scale
    Sigma = np.zeros((Nq, 2, 2), dtype=np.float32)
    Sigma[:, 0, 0] = sx_full * sx_full
    Sigma[:, 1, 1] = sy_full * sy_full
    Sigma[:, 0, 1] = Sigma[:, 1, 0] = rho * sx_full * sy_full
    return dict(P=pts_cam_off[sel],
                uv_ref=uv_off[sel],
                d=np.stack([du_full, dv_full], axis=1).astype(np.float32),
                Σ=Sigma)


def perturb(rng: np.random.Generator, max_rot_deg: float, max_t_m: float,
            yaw_only: bool = False):
    """Return (ypr_deg [3], t_world [3]) drawn uniformly in the configured half-range."""
    if yaw_only:
        ypr = np.array([rng.uniform(-max_rot_deg, max_rot_deg), 0.0, 0.0], dtype=np.float32)
    else:
        ypr = rng.uniform(-max_rot_deg, max_rot_deg, size=3).astype(np.float32)
    t = rng.uniform(-max_t_m, max_t_m, size=3).astype(np.float32)
    return ypr, t


def sigma_scalar(S: np.ndarray) -> np.ndarray:
    """sqrt(det Σ) per point — area of the 1-σ ellipse (pixels²)^0.5."""
    det = S[:, 0, 0] * S[:, 1, 1] - S[:, 0, 1] * S[:, 1, 0]
    det = np.maximum(det, 1e-12)
    return np.sqrt(np.sqrt(det))  # double sqrt → pixel-scale "radius-equivalent"


def render_frame(out_path: Path, img, points_top, frame_id: str, n_total: int):
    """Render the full image with red/green correction markers + arrows."""
    IH, IW = img.shape[:2]
    dpi = 100
    fig, ax = plt.subplots(1, 1, figsize=(IW / dpi, IH / dpi), dpi=dpi)
    ax.imshow(img)

    if len(points_top) > 0:
        u_ref = points_top[:, 0]; v_ref = points_top[:, 1]
        u_prd = points_top[:, 2]; v_prd = points_top[:, 3]
        sigs  = points_top[:, 4]
        # arrows red→green
        for u0, v0, u1, v1 in zip(u_ref, v_ref, u_prd, v_prd):
            ax.annotate('', xy=(u1, v1), xytext=(u0, v0),
                         arrowprops=dict(arrowstyle='->', color='yellow',
                                          lw=0.6, alpha=0.7), zorder=2)
        # red = hypothesis
        ax.scatter(u_ref, v_ref, s=18, facecolors='none', edgecolors='red',
                    linewidths=0.8, zorder=3, label='hypothesis (perturbed)')
        # green = corrected
        ax.scatter(u_prd, v_prd, s=18, facecolors='none', edgecolors='lime',
                    linewidths=0.8, zorder=4, label='corrected (pred)')

    ax.set_xlim(0, IW); ax.set_ylim(IH, 0); ax.axis('off')
    title = (f'{frame_id}  top-{len(points_top)} of {n_total} pts (lowest σ)  '
              f'σ range: {points_top[:,4].min():.2f} → {points_top[:,4].max():.2f} px'
              if len(points_top) > 0 else f'{frame_id} — no points')
    ax.set_title(title, fontsize=9)
    ax.legend(loc='lower right', fontsize=7, framealpha=0.85)
    plt.tight_layout(pad=0.2)
    fig.savefig(out_path, dpi=dpi)
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--exp', required=True,
                    help='Experiment name under experiments/ (loads config.py + best_model.pt)')
    ap.add_argument('--cache', required=True, help='LMDB-packed cache dir')
    ap.add_argument('--split', choices=('train', 'val'), default='train')
    ap.add_argument('--n-frames', type=int, default=10)
    ap.add_argument('--n-top', type=int, default=100)
    ap.add_argument('--window', type=int, default=384)
    ap.add_argument('--stride', type=int, default=192)
    ap.add_argument('--max-rot-deg', type=float, default=1.5,
                    help='per-axis rotation half-range, degrees')
    ap.add_argument('--max-t-m',    type=float, default=0.6,
                    help='per-axis translation half-range, meters')
    ap.add_argument('--yaw-only', action='store_true',
                    help='if set, only perturb yaw (roll=pitch=0)')
    ap.add_argument('--seed', type=int, default=42)
    ap.add_argument('--idxs', type=str, default=None,
                    help='optional comma-separated explicit instance idxs (overrides random)')
    ap.add_argument('--out', default=None,
                    help='output dir (default: experiments/<exp>/vis_<split>_lowsigma_top<N>)')
    args = ap.parse_args()

    log = print
    log(f'load model: {args.exp}')
    exp_dir = REPO_ROOT / 'experiments' / args.exp
    model, c = load_model(exp_dir)
    S = int(c['img_size'])
    cs = int(args.window)
    stride = int(args.stride)

    log(f'load cache: {args.cache} ({args.split})')
    ds = PandaSetCalibDatasetFull(
        args.cache, split=args.split,
        img_size=S, oversample=1,
    )
    log(f'  {len(ds)} samples')

    if args.idxs:
        idxs = [int(x) for x in args.idxs.split(',')]
    else:
        rng = np.random.default_rng(args.seed)
        idxs = list(rng.choice(len(ds), size=min(args.n_frames, len(ds)), replace=False))
    log(f'  selected idxs: {idxs}')

    out = Path(args.out) if args.out else (exp_dir / f'vis_{args.split}_lowsigma_top{args.n_top}')
    out.mkdir(parents=True, exist_ok=True)
    for old in out.glob('*.png'): old.unlink()
    (out / 'meta.json').write_text(json.dumps({
        'exp': args.exp, 'cache': args.cache, 'split': args.split,
        'window': cs, 'stride': stride, 'S': S,
        'max_rot_deg': args.max_rot_deg, 'max_t_m': args.max_t_m,
        'yaw_only': bool(args.yaw_only), 'seed': args.seed,
        'idxs': [int(i) for i in idxs], 'n_top': args.n_top,
    }, indent=2))

    rng = np.random.default_rng(args.seed)
    summary = []
    for idx in idxs:
        inst = ds._load_inst(int(idx))
        if 'jpg_bytes' not in inst:
            log(f'  skip idx={idx}: no jpg_bytes (legacy cache?)'); continue
        img = decode_jpg(bytes(inst['jpg_bytes']))
        IH, IW = img.shape[:2]
        K = inst['K_full'].numpy().copy()
        # Tile cache: K_full is for the FULL image; shift cx,cy to be tile-local
        # so projection lands in [0, tile_W] / [0, tile_H] matching the stored tile.
        tu0 = int(inst.get('tile_u0', 0)); tv0 = int(inst.get('tile_v0', 0))
        K[0, 2] -= tu0; K[1, 2] -= tv0
        pts_w = inst['pts'].numpy()
        cam_pos = inst['cam_pos'].numpy()
        R_gt = inst['R_gt'].numpy()

        ypr_d, t_d = perturb(rng, args.max_rot_deg, args.max_t_m, yaw_only=args.yaw_only)
        R_off = R_gt @ ypr_to_R(ypr_d)
        t_off = cam_pos + t_d

        windows = grid_windows(IW, IH, cs, stride)
        all_uv_ref = []; all_d = []; all_S = []
        t0 = time.time()
        for (u0, v0) in windows:
            o = infer_crop_with_bucket(model, img, K, pts_w, R_off, t_off, u0, v0, cs, S)
            if o is None: continue
            all_uv_ref.append(o['uv_ref'])
            all_d.append(o['d'])
            all_S.append(o['Σ'])
        if not all_uv_ref:
            log(f'  idx={idx}: no valid windows'); continue
        uv_ref = np.concatenate(all_uv_ref, axis=0)
        d_pred = np.concatenate(all_d, axis=0)
        Sigma = np.concatenate(all_S, axis=0)
        n_total = uv_ref.shape[0]

        sigs = sigma_scalar(Sigma)
        order = np.argsort(sigs)
        keep = order[:min(args.n_top, n_total)]
        uv_ref_k = uv_ref[keep]
        d_k      = d_pred[keep]
        uv_prd_k = uv_ref_k + d_k
        sigs_k   = sigs[keep]
        points_top = np.stack([uv_ref_k[:, 0], uv_ref_k[:, 1],
                                uv_prd_k[:, 0], uv_prd_k[:, 1], sigs_k], axis=-1)

        scene = inst.get('scene', '?'); frame = inst.get('frame', '?')
        frame_id = f'idx{int(idx):06d}_scene{scene}_f{frame}'
        out_path = out / f'{frame_id}.png'
        render_frame(out_path, img, points_top, frame_id, n_total)
        el = time.time() - t0
        log(f'  idx={idx}: {len(windows)} windows, {n_total} pts → kept {len(keep)}, '
             f'σ {sigs_k.min():.2f}-{sigs_k.max():.2f}px, {el:.1f}s → {out_path.name}')
        summary.append(dict(
            idx=int(idx), scene=str(scene), frame=str(frame),
            n_total=int(n_total), n_kept=int(len(keep)),
            sigma_min=float(sigs_k.min()), sigma_max=float(sigs_k.max()),
            ypr=ypr_d.tolist(), t=t_d.tolist(),
        ))

    (out / 'summary.json').write_text(json.dumps(summary, indent=2))
    log(f'done → {out}')


if __name__ == '__main__':
    main()
