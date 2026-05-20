"""Mirror demo_app: cache inst → DS.build_window(u0, v0, cs) → model
forward → render hyp(red) / pred(green) / GT(yellow) on the cropped
tile. Saves PNGs you can scp/cat. No browser needed.

Usage:
    docker exec caaas python3 /workspace/scripts/_debug/viz_build_window.py \\
        --exp km_wv_wm_dgx1_n4_v4_resume --n 5 --cs 384 --ox 0.5 --tx 0.1
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import torch
from scipy.spatial.transform import Rotation

from datasets.pandaset_full import PandaSetCalibDatasetFull
from scripts.inference.infer_calib import load_calib_model


def _build_sample_for(DS, idx, cs, ox, oy, tx, ty):
    inst = DS._load_inst(idx)
    if inst['pts'].shape[0] < DS.min_pts:
        return None
    K = inst['K_full'].numpy()
    pts = inst['pts'].numpy()
    cp = inst['cam_pos'].numpy()
    R_gt = inst['R_gt'].numpy()
    intensity = inst['intensity'].numpy()
    uv_full = inst['uv_full'].numpy()
    is_obj_full = (inst['is_obj'].numpy().astype(bool) if 'is_obj' in inst
                    else np.zeros(len(pts), dtype=bool))
    tu0 = int(inst.get('tile_u0', 0)); tv0 = int(inst.get('tile_v0', 0))
    if tu0 or tv0:
        uv_full = uv_full - np.array([tu0, tv0], dtype=np.float32)
    IH, IW = int(inst['IH']), int(inst['IW'])
    cs = min(cs, IW, IH)
    u0 = max(0, (IW - cs) // 2); v0 = max(0, (IH - cs) // 2)

    ypr = np.array([0.0, oy, ox], dtype=np.float64)
    R_pert = Rotation.from_euler('zyx', ypr, degrees=True).as_matrix()
    R_off = R_gt @ R_pert
    t_delta_world = R_gt @ np.array([tx, ty, 0.0], dtype=np.float64)
    cp_off = cp + t_delta_world
    K_pert = K.copy()
    pert_vec = np.array([t_delta_world[0], t_delta_world[1], t_delta_world[2],
                          ypr[0], ypr[1], ypr[2], 0.0, 0.0], dtype=np.float32)
    cand_idx = np.arange(len(pts), dtype=np.int64)
    built = DS.build_window(
        inst, pts[cand_idx], intensity[cand_idx], uv_full[cand_idx],
        cand_idx, is_obj_full, u0, v0, cs, K,
        R_off, cp_off, K_pert, cp, pert_vec, tu0, tv0, None, IW, IH)
    if built is None:
        return None
    img_t, true_uvd, dist_uvd, vfp, buc, buc_v, _ = built
    return dict(inst=inst, img=img_t, true_uvd=true_uvd, dist_uvd=dist_uvd,
                 vfp=vfp, bucket_uvd=buc, bucket_valid=buc_v,
                 cs=cs, u0=u0, v0=v0)


def _run_model(model, pkt, device):
    img = pkt['img'].unsqueeze(0).to(device).float().div_(255.0)
    dist = pkt['dist_uvd'].unsqueeze(0).to(device)
    use_intensity = bool(getattr(model, 'use_intensity', False))
    pin = (torch.cat([dist[..., :3], dist[..., 4:5]], -1)
            if use_intensity else dist[..., :3])
    pad = torch.zeros(1, dist.shape[1], dtype=torch.bool, device=device)
    with torch.no_grad(), torch.autocast(device_type='cuda', dtype=torch.float16):
        out = model(img, pin, key_padding_mask=pad,
                    vfp=pkt['vfp'].view(1).to(device),
                    bucket_uvd=pkt['bucket_uvd'].unsqueeze(0).to(device),
                    bucket_valid=pkt['bucket_valid'].unsqueeze(0).to(device))
    return (out[0] if isinstance(out, tuple) else out)[0].float().cpu().numpy()


def _render(pkt, per_pt, out_path: Path, title=''):
    img = pkt['img'].permute(1, 2, 0).numpy().astype(np.uint8)
    S = img.shape[0]
    true_uv = pkt['true_uvd'][:, :2].numpy()
    hyp_uv = pkt['dist_uvd'][:, :2].numpy()
    valid = ~((hyp_uv[:, 0] == 0) & (hyp_uv[:, 1] == 0))
    pred_uv = hyp_uv + per_pt[:, :2]
    err_pre = np.linalg.norm(true_uv[valid] - hyp_uv[valid], axis=1).mean()
    err_post = np.linalg.norm(pred_uv[valid] - true_uv[valid], axis=1).mean()

    fig, ax = plt.subplots(1, 1, figsize=(7, 7), dpi=110)
    ax.imshow(img)
    ax.scatter(hyp_uv[valid, 0], hyp_uv[valid, 1], s=18, facecolors='none',
                edgecolors='red', linewidths=0.9, label='input (hyp)')
    ax.scatter(pred_uv[valid, 0], pred_uv[valid, 1], s=18, facecolors='none',
                edgecolors='lime', linewidths=0.9, label='pred')
    ax.scatter(true_uv[valid, 0], true_uv[valid, 1], s=24, c='yellow',
                marker='x', linewidths=1.0, label='GT')
    for (u0, v0), (u1, v1) in zip(hyp_uv[valid], pred_uv[valid]):
        ax.annotate('', xy=(u1, v1), xytext=(u0, v0),
                     arrowprops=dict(arrowstyle='->', color='orange',
                                      lw=0.6, alpha=0.7))
    ax.set_xlim(0, S); ax.set_ylim(S, 0); ax.axis('off')
    ax.set_title(f'{title}  N={int(valid.sum())}  '
                  f'err pre={err_pre:.2f}→post={err_post:.2f}px',
                  fontsize=8)
    ax.legend(loc='lower right', fontsize=7, framealpha=0.85)
    plt.tight_layout(pad=0.2)
    fig.savefig(out_path, dpi=96)
    plt.close(fig)
    return err_pre, err_post


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--exp', default='km_wv_wm_dgx1_n4_v4_resume')
    ap.add_argument('--cache', default='/cache/kamikado_v3_tiled')
    ap.add_argument('--n', type=int, default=5)
    ap.add_argument('--cs', type=int, default=384)
    ap.add_argument('--ox', type=float, default=0.0)
    ap.add_argument('--oy', type=float, default=0.0)
    ap.add_argument('--tx', type=float, default=0.0)
    ap.add_argument('--ty', type=float, default=0.0)
    ap.add_argument('--out', default=None)
    ap.add_argument('--seed', type=int, default=42,
                    help='random seed for picking idxs')
    args = ap.parse_args()

    out = Path(args.out) if args.out else (REPO_ROOT / 'experiments' / args.exp /
                                              '_vis_buildwin')
    out.mkdir(parents=True, exist_ok=True)
    for old in out.glob('*.png'): old.unlink()

    # Pool train + val so we see ALL scenes in the cache (kamikado has
    # only 1 scene in val).
    DS_t = PandaSetCalibDatasetFull(args.cache, split='train',
                                      max_offset_m=0.20, max_rot_deg=0.5,
                                      min_crop_px=128, max_crop_px=384,
                                      oversample=1)
    DS_v = PandaSetCalibDatasetFull(args.cache, split='val',
                                      max_offset_m=0.20, max_rot_deg=0.5,
                                      min_crop_px=128, max_crop_px=384,
                                      oversample=1)
    DS = DS_t  # build_sample_for uses fnames from this; we'll override
    DS.fnames = list(DS_t.fnames) + list(DS_v.fnames)
    model = load_calib_model(args.exp).eval()
    device = torch.device('cuda')

    # Group cache instances by their REAL scene string (inst['scene']),
    # not the gid hash prefix. Each scene = a separate driving sequence.
    from collections import defaultdict
    by_scene = defaultdict(list)
    print(f'  scanning {len(DS.fnames)} insts to group by scene ...')
    for i in range(len(DS.fnames)):
        try:
            scene = str(DS._load_inst(i).get('scene', ''))
        except Exception:
            continue
        by_scene[scene].append(i)
    scenes = sorted(by_scene.keys())
    print(f'  found {len(scenes)} scenes: {scenes[:6]}{" ..." if len(scenes) > 6 else ""}')
    rng = np.random.RandomState(args.seed)
    rng.shuffle(scenes)
    idxs = []
    for scene in scenes[:args.n]:
        idxs.append(int(rng.choice(by_scene[scene])))
    if len(idxs) < args.n:
        # fewer scenes than requested — pad with random extras across all scenes
        extras = rng.choice([i for s in scenes for i in by_scene[s]],
                              size=args.n - len(idxs), replace=False)
        idxs.extend(int(x) for x in extras)
    print(f'  picked {len(idxs)} idxs from {len(set(scenes[:args.n]))} distinct scenes')

    errs = []
    # Tag each pick with which scene it came from so file names self-document.
    idx_to_scene = {}
    for sc, lst in by_scene.items():
        for j in lst:
            idx_to_scene[j] = sc
    for k, idx in enumerate(idxs):
        pkt = _build_sample_for(DS, idx, args.cs, args.ox, args.oy,
                                  args.tx, args.ty)
        if pkt is None:
            print(f'  idx={idx}: skip (too few pts)'); continue
        per_pt = _run_model(model, pkt, device)
        scene_short = idx_to_scene.get(idx, 'unknown')[-30:]
        png = out / f'scene_{k:02d}_{scene_short}_idx{idx:05d}.png'
        e_pre, e_post = _render(pkt, per_pt, png,
                                  title=f'{scene_short}  cs={args.cs}  '
                                  f'pert=ox{args.ox}/oy{args.oy}/tx{args.tx}/ty{args.ty}')
        errs.append((e_pre, e_post))
        print(f'  saved {png.name}  err_pre={e_pre:.2f} err_post={e_post:.2f}')

    if errs:
        a = np.asarray(errs)
        print(f'\n{len(errs)} frames: pre mean={a[:,0].mean():.2f}±{a[:,0].std():.2f}  '
              f'post mean={a[:,1].mean():.2f}±{a[:,1].std():.2f}')
    print(f'\nout: {out}')


if __name__ == '__main__':
    main()
