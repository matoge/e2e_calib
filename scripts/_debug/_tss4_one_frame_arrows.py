"""1フレームの t19/t0 タイルで model duv_pred を画像にオーバーレイ。

Each sub-crop (cs=256, stride=128 → 3×3 = 9 sub-crops/tile) を model に流して
得た per-point duv_pred (orig-px) を、parent-tile (512×512) 座標に戻して
画像に直接 quiver で重ねる。スケールは scale=1.0 (実 px 矢印長)。
"""
from __future__ import annotations
import argparse
import sys
from pathlib import Path

import numpy as np
import torch
from PIL import Image

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))

from datasets.pandaset_full import PandaSetCalibDatasetFull, collate_full
import scripts.eval.eval_shared_256x800 as _ev
from scripts.eval.eval_shared_256x800 import _build_model, _build_subwin, DEVICE


def render_one(ds, fname, ckpt_run, cs, stride, frame_label, out_path, args):
    cfg = _ev._load_cfg()

    idx = ds.fnames.index(fname)
    inst = ds._load_inst(idx)
    tile_u0 = int(inst.get('tile_u0', 0))
    tile_v0 = int(inst.get('tile_v0', 0))

    if 'jpg_bytes' in inst:
        import io
        img_tile = np.asarray(Image.open(io.BytesIO(inst['jpg_bytes'])).convert('RGB'))
    else:
        img_tile = inst['img'].permute(1, 2, 0).numpy()

    PT = 512
    last_off = PT - cs
    offs = list(range(0, last_off + 1, stride))
    if offs[-1] != last_off:
        offs.append(last_off)
    seen = set(); deduped = []
    for v in offs:
        for u in offs:
            if (u, v) not in seen:
                seen.add((u, v)); deduped.append((u, v))
    u0v0_list = deduped
    print(f'[arrows] {frame_label}: {len(u0v0_list)} sub-crops (cs={cs}, stride={stride})')

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
    scale = float(cs) / S

    wins = []
    u0v0_used = []
    for (u0, v0) in u0v0_list:
        w = _build_subwin(ds, inst, np.zeros(3), np.zeros(3), u0=u0, v0=v0, cs=cs)
        if w is not None:
            wins.append(w); u0v0_used.append((u0, v0))
    if not wins:
        print(f'[arrows] no valid sub-crops for {frame_label}')
        return

    moved = [t.to(DEVICE) if torch.is_tensor(t) else t for t in collate_full(wins)]
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

    true_np = true_uvd[..., :2].cpu().numpy()
    duv_np  = duv_pred_local.cpu().numpy()
    valid_np = valid.cpu().numpy()

    all_u, all_v, all_du, all_dv = [], [], [], []
    for b in range(true_np.shape[0]):
        u0s, v0s = u0v0_used[b]
        m = valid_np[b]
        u_par = true_np[b, m, 0] * scale + u0s
        v_par = true_np[b, m, 1] * scale + v0s
        du    = duv_np[b, m, 0] * scale
        dv    = duv_np[b, m, 1] * scale
        all_u.append(u_par); all_v.append(v_par)
        all_du.append(du); all_dv.append(dv)
    u = np.concatenate(all_u); v = np.concatenate(all_v)
    du = np.concatenate(all_du); dv = np.concatenate(all_dv)
    mag = np.sqrt(du*du + dv*dv)
    print(f'[arrows] {frame_label}: {u.size} pts, |duv| mean={mag.mean():.2f}px '
          f'p50={np.median(mag):.2f}px p95={np.percentile(mag, 95):.2f}px '
          f'max={mag.max():.2f}px')

    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    H, W = img_tile.shape[:2]
    fig, ax = plt.subplots(figsize=(W/96.0, H/96.0), dpi=160)
    ax.imshow(img_tile)
    ax.set_xlim(0, W); ax.set_ylim(H, 0)
    ax.set_aspect('equal')
    sc = ax.quiver(u, v, du, dv, mag,
                   cmap='turbo',
                   angles='xy', scale_units='xy', scale=1.0,
                   width=0.0018, headwidth=4, headlength=5)
    cb = fig.colorbar(sc, ax=ax, fraction=0.025, pad=0.01)
    cb.set_label('|duv_pred| (orig px, real-scale arrows)', fontsize=8)
    if args.grid > 0:
        for gx in range(0, W + 1, args.grid):
            major = (args.major_every > 0 and gx % args.major_every == 0)
            ax.axvline(gx, color='cyan', lw=0.4 if not major else 0.8,
                       alpha=0.35 if not major else 0.55)
        for gy in range(0, H + 1, args.grid):
            major = (args.major_every > 0 and gy % args.major_every == 0)
            ax.axhline(gy, color='cyan', lw=0.4 if not major else 0.8,
                       alpha=0.35 if not major else 0.55)
    ax.set_title(f'{frame_label}: 1-frame model duv_pred (cs={cs}, stride={stride}, '
                 f'9 sub-crops, parent-tile {W}×{H}, tile_u0={tile_u0} v0={tile_v0})',
                 fontsize=9)
    ax.axis('off')
    fig.tight_layout()
    fig.savefig(out_path, dpi=160, bbox_inches='tight')
    plt.close(fig)
    print(f'[arrows] wrote {out_path}')


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--cache', type=Path,
        default=Path('/mnt/fsx/tmp/hfunaya/e2e_calib_cache/tss4_v3_tiled'))
    ap.add_argument('--ckpt-run', required=True)
    ap.add_argument('--cs', type=int, default=256, choices=(128, 256, 512))
    ap.add_argument('--stride', type=int, default=128)
    ap.add_argument('--tile-suffixes', default='_t0.pt,_t19.pt')
    ap.add_argument('--frame-idx', type=int, default=0,
                    help='which frame to pick within each tile group (sorted)')
    ap.add_argument('--split', default='train', choices=('train', 'val'))
    ap.add_argument('--fname-prefix', default=None,
                    help='if set, restrict tile picks to fnames starting with this '
                         '(used to lock to a specific gid / scene+frame)')
    ap.add_argument('--grid', type=int, default=32,
                    help='grid line spacing in px (0 = off)')
    ap.add_argument('--major-every', type=int, default=128)
    ap.add_argument('--out', type=Path,
        default=REPO / 'scripts/_debug/_outputs/episode_iter0_22dof')
    ap.add_argument('--out-suffix', type=str, default='')
    args = ap.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)

    run_dir = REPO / 'experiments' / args.ckpt_run
    _ev.CKPT = run_dir / 'best_model.pt'
    _ev.EXP_CFG_PATH = run_dir / 'config.py'
    cfg = _ev._load_cfg()

    ds = PandaSetCalibDatasetFull(
        cache_dir=args.cache, split=args.split, img_size=cfg['img_size'],
        min_crop_px=cfg['min_crop_px'], max_crop_px=cfg['max_crop_px'],
        max_offset_m=0.0, max_rot_deg=0.0, oversample=1,
        grid_n=cfg.get('grid_n', 16), center_band=0.0, preload=False)

    suffixes = [s.strip() for s in args.tile_suffixes.split(',') if s.strip()]
    for suf in suffixes:
        members = [f for f in ds.fnames if f.endswith(suf)]
        if args.fname_prefix:
            members = [f for f in members if f.startswith(args.fname_prefix)]
        if not members:
            print(f'[arrows] no tile {suf} in train; skip'); continue
        members = sorted(members)
        if args.frame_idx >= len(members):
            print(f'[arrows] frame_idx={args.frame_idx} >= {len(members)} for {suf}'); continue
        fname = members[args.frame_idx]
        tag = suf.replace('.pt', '').lstrip('_')
        out_path = args.out / (
            f'arrows_{tag}_frame{args.frame_idx}{args.out_suffix}.jpg')
        render_one(ds, fname, args.ckpt_run, args.cs, args.stride,
                   frame_label=f'{tag} frame={args.frame_idx}',
                   out_path=out_path, args=args)


if __name__ == '__main__':
    main()
