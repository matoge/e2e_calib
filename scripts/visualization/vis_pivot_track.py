"""Show the SAME 3D pivot point as seen from increasing baselines.

For a chosen val scene and starting frame A, pick pivot points that remain
visible through a long frame sequence (frames A, A+5, A+10, A+20, A+40, A+70).
For each, crop a patch centered on the pivot's projection in that frame.

Layout: 1 row per pivot × 6 columns = one baseline per column.
As baseline grows, the same physical object (traffic light, pole, building
corner) moves and zooms within the image, but the pivot ★ stays at patch
center.

Output: experiments/cross_frame_PIVOT_TRACK/pivot_track.png
"""
import sys, pathlib
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[2]))

import argparse, random
from pathlib import Path
import numpy as np
import torch
import torch.nn.functional as F
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from datasets.pandaset_pair import _SceneData, _project, _invert_mat


def pad_crop(img, u0, v0, CROP, IW, IH):
    u0i = int(np.floor(u0)); v0i = int(np.floor(v0))
    u1i = u0i + CROP;         v1i = v0i + CROP
    pad_left = max(0, -u0i); pad_top = max(0, -v0i)
    src_u0 = max(0, u0i); src_v0 = max(0, v0i)
    src_u1 = min(IW, u1i); src_v1 = min(IH, v1i)
    out = np.zeros((CROP, CROP, 3), dtype=img.dtype)
    cw = src_u1 - src_u0; ch = src_v1 - src_v0
    if cw > 0 and ch > 0:
        out[pad_top:pad_top + ch, pad_left:pad_left + cw] = img[src_v0:src_v1, src_u0:src_u1]
    return out


def project_world_to_cam(P_w, T_w2c, K):
    P_h = np.append(P_w, 1.0)
    P_c = (T_w2c @ P_h)[:3]
    if P_c[2] < 1.0:
        return None, None
    uv = (K @ P_c)[:2] / P_c[2]
    return uv.astype(np.float32), float(P_c[2])


def find_tracks(scn, fi_A, baselines, IW, IH, K, target_depth_range=(15, 80),
                 height_min=None, n_tracks=8, rng=None):
    """Find pivot 3D points in frame A that project inside the image in ALL
    baseline target frames.

    height_min (world z): if set, restrict to elevated points — typically
    traffic lights / signs / pole tops are 3-8m off the ground so this is
    a cheap proxy for "traffic-light-like" pivots when PandaSet lacks the
    explicit label.
    """
    pts_w_A, uv_Af, z_Af, in_A = scn.frame_data(fi_A)
    d_ok = (z_Af >= target_depth_range[0]) & (z_Af <= target_depth_range[1])
    candidates_mask = in_A & d_ok
    if height_min is not None:
        candidates_mask &= pts_w_A[:, 2] >= height_min
    candidates = np.where(candidates_mask)[0]
    if rng is not None:
        rng.shuffle(candidates)
    picked = []
    for ci in candidates:
        P_w = pts_w_A[ci]
        uvs = []
        ok = True
        # frame A itself
        uvs.append((fi_A, uv_Af[ci], z_Af[ci]))
        for bl in baselines:
            fi_B = fi_A + bl
            if fi_B < 0 or fi_B >= scn.n_frames:
                ok = False; break
            uv, z = project_world_to_cam(P_w, scn.T_w2c[fi_B], K)
            if uv is None or not (0 <= uv[0] < IW and 0 <= uv[1] < IH):
                ok = False; break
            uvs.append((fi_B, uv, z))
        if ok:
            picked.append((P_w, uvs))
            if len(picked) >= n_tracks: break
    return picked


def main(args):
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    # val scene pool
    root = Path(args.scenes_root)
    names = sorted([p.name for p in root.iterdir() if p.is_dir() and p.name.isdigit()])
    shuffled = sorted([str(root / n) for n in names])
    random.Random(42).shuffle(shuffled)
    cutoff = int(len(shuffled) * args.train_frac)
    val_roots = shuffled[cutoff:]
    print(f'val scenes: {[Path(r).name for r in val_roots]}')

    # Use a single scene for continuous tracking
    picked_scene = args.scene or Path(val_roots[0]).name
    scn_path = root / picked_scene
    scn = _SceneData(scn_path)
    scn.precompute_all()

    baselines = args.baselines   # e.g. [5, 10, 20, 40, 70]
    columns_bl = [0] + list(baselines)   # include fi_A itself
    n_cols = len(columns_bl)

    rng = np.random.default_rng(args.seed)
    fi_A = args.fi_A
    print(f'anchor: scene {picked_scene}, fi_A={fi_A}')

    tracks = find_tracks(scn, fi_A, baselines, scn.IW, scn.IH, scn.K,
                         target_depth_range=(args.depth_min, args.depth_max),
                         height_min=args.height_min,
                         n_tracks=args.n_tracks, rng=rng)
    print(f'{len(tracks)} tracks (3D points visible in all frames)')
    if not tracks:
        print('no valid track — try a different --fi-A or --depth-range.')
        return

    n_rows = len(tracks)
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(2.4 * n_cols, 2.4 * n_rows), dpi=110)
    fig.patch.set_facecolor('#f6f4ed')
    if n_rows == 1: axes = axes[None, :]
    CROP = args.crop

    for ri, (P_w, uvs) in enumerate(tracks):
        for ci, bl in enumerate(columns_bl):
            fi, uv, z = uvs[ci]
            half = CROP / 2
            u0 = uv[0] - half; v0 = uv[1] - half
            img = scn.load_image(fi)
            patch = pad_crop(img, u0, v0, CROP, scn.IW, scn.IH)
            # resize to display size
            t = torch.from_numpy(patch).permute(2, 0, 1).float() / 255.0
            t = F.interpolate(t.unsqueeze(0), size=(args.img_size, args.img_size),
                               mode='bilinear', align_corners=False).squeeze(0)
            arr = (t.permute(1, 2, 0).numpy() * 255).astype(np.uint8)

            ax = axes[ri, ci]
            ax.imshow(arr)
            ax.set_xticks([]); ax.set_yticks([])
            ax.set_xlim(-0.5, args.img_size - 0.5)
            ax.set_ylim(args.img_size - 0.5, -0.5)
            # pivot at exact center
            ax.plot(args.img_size / 2, args.img_size / 2,
                     marker='*', color='#c13c14', markersize=22,
                     markeredgecolor='white', mew=1.5, zorder=10)
            if ri == 0:
                lbl = f'fi={fi}' if bl == 0 else f'fi+{bl}  (Δ{bl})'
                ax.set_title(lbl, fontsize=9, loc='left', pad=3)
            if ci == 0:
                ax.set_ylabel(f'd={z:.0f}m', fontsize=9, rotation=0,
                               ha='right', va='center', labelpad=20)

    plt.suptitle(f'Same 3D pivot across baselines — scene {picked_scene}, anchor fi={fi_A}\n'
                  f'(each row: one physical point; pivot ★ always at patch center regardless of baseline)',
                  fontsize=11, y=0.995)
    plt.tight_layout()
    outpath = out_dir / f'pivot_track_{picked_scene}_fi{fi_A}.png'
    plt.savefig(outpath, dpi=110, bbox_inches='tight', facecolor='#f6f4ed')
    print(f'saved → {outpath}')


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('--scenes-root', default='/mnt/mininas/datasets/pandaset')
    ap.add_argument('--train-frac', type=float, default=0.80)
    ap.add_argument('--out', default='experiments/cross_frame_PIVOT_TRACK')
    ap.add_argument('--scene', default=None, help='override scene name (e.g., 008)')
    ap.add_argument('--fi-A', type=int, default=0, help='anchor frame')
    ap.add_argument('--baselines', type=int, nargs='+', default=[5, 10, 20, 40, 70])
    ap.add_argument('--depth-min', type=float, default=15.0)
    ap.add_argument('--depth-max', type=float, default=80.0)
    ap.add_argument('--height-min', type=float, default=None,
                    help='world-z threshold, filter points ABOVE this height '
                         '(~3.0 m biases toward traffic lights / signs / pole tops)')
    ap.add_argument('--n-tracks', type=int, default=8)
    ap.add_argument('--crop', type=int, default=192)
    ap.add_argument('--img-size', type=int, default=128, help='display size')
    ap.add_argument('--seed', type=int, default=42)
    args = ap.parse_args()
    main(args)
