"""Multi-frame fusion on the real inference path: does fusing F frames help, and which rule?

Per frame, the same path as the API: perturb the true pose by δ (the SAME δ for every frame of a
draw — one rig, one calibration error), build_inst → make_dataset → predict_pose → (δ̂_f, H_f).
Then F frames are fused with scripts/eval/frame_fusion.fuse (sum | gate3 | huber | tukey) and the
corrected pose E(δ̄) @ T_in is compared with the true pose (per-axis yaw/pitch/roll, x/y/z, and
the geodesic angle / centre distance).

Frames: --n-frames evenly spaced over val; per draw, --n-groups random groups of F frames.
Grouping: --group scene  = F frames from the same scene (PandaSet: 80 frames per val scene)
          --group any    = F frames from any scenes (nuScenes val has only 4 frames per scene)

    python scripts/eval/multiframe_infer.py --exp nsps_s2_pad256 --cache <cache> --tag ps \
        --group scene --draws 3 --out experiments/multiframe
"""
import argparse, json, os, sys
from collections import defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
import numpy as np

from scripts.eval.frame_fusion import fuse
from scripts.inference.infer_calib import (load_model, build_inst, make_dataset, infer, perturb_T,
                                           apply_delta, pose_error, pose_error_axes, load_val_frame)
from datasets.pandaset_full import PandaSetCalibDatasetFull

AX = ('yaw_deg', 'pitch_deg', 'roll_deg', 'x_m', 'y_m', 'z_m')


def per_frame(model, c, cache, draws, seed, out_npz, n_frames, max_scenes=0):
    """(draw, frame) → δ̂ (6,), H (6,6). Cached in out_npz."""
    if os.path.exists(out_npz):
        z = np.load(out_npz, allow_pickle=True)
        return z['D'], z['H'], z['inj'], list(z['scene']), z['T'], z['Tin']
    src = PandaSetCalibDatasetFull(cache_dir=cache, split='val', img_size=256, grid_n=16,
                                   min_crop_px=256, max_crop_px=256, oversample=1)
    if max_scenes > 0:
        # シーンを等間隔に max_scenes 個選び、その全フレームを使う (シーケンス内の融合を測る用)
        sc = [src._load_inst(i)['scene'] for i in range(len(src))]
        names = sorted(set(sc))
        pick = set(names[k] for k in np.linspace(0, len(names) - 1, min(max_scenes, len(names))).astype(int))
        frames = np.array([i for i, q in enumerate(sc) if q in pick])
    else:
        frames = np.linspace(0, len(src) - 1, min(n_frames, len(src))).astype(int)   # 等間隔に n_frames 枚
    n = len(frames)
    g = np.random.default_rng(seed)
    inj = np.concatenate([(g.random((draws, 3)) * 2 - 1) * c['rot_deg'],
                          (g.random((draws, 3)) * 2 - 1) * c['t_m']], 1)   # (draws, 6) rot deg, t m
    D = np.zeros((draws, n, 6)); H = np.zeros((draws, n, 6, 6))
    T_all = np.zeros((n, 4, 4)); Tin = np.zeros((draws, n, 4, 4)); scene = []
    for i, fi in enumerate(frames):
        img, pts, K, T, label, _, dist, fe = load_val_frame(int(fi), cache)
        T_all[i] = T; scene.append(label.split('/')[0])
        for d in range(draws):
            Tp = perturb_T(T, rot_deg=inj[d, :3], t_m=inj[d, 3:])
            ds = make_dataset(c, [build_inst(img=img, pts=pts, K=K, T_cam_lidar=Tp, dist=dist, is_fisheye=fe)])
            D[d, i], H[d, i], _ = infer(model, ds)
            Tin[d, i] = Tp
        if i % 20 == 0:
            print(f'  frame {i}/{n} {label}', flush=True)
    np.savez(out_npz, D=D, H=H, inj=inj, scene=np.array(scene), T=T_all, Tin=Tin)
    return D, H, inj, scene, T_all, Tin


def groups_of(scene, F, mode, rng, n_groups):
    """F フレームの組を n_groups 個、ランダムに引く (組の中は重複なし、組どうしは重なってよい)。
    mode='scene' は同じシーンの中から (F 枚以上あるシーンだけ)。"""
    if mode == 'any':
        return [rng.choice(len(scene), F, replace=False) for _ in range(n_groups)]
    by = defaultdict(list)
    for i, s in enumerate(scene):
        by[s].append(i)
    ok = [ii for ii in by.values() if len(ii) >= F]
    if not ok:
        return []
    return [rng.choice(ok[rng.integers(len(ok))], F, replace=False) for _ in range(n_groups)]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--exp', required=True)
    ap.add_argument('--ckpt', default='best_model.pt')
    ap.add_argument('--cache', required=True)
    ap.add_argument('--tag', required=True)
    ap.add_argument('--group', choices=['scene', 'any'], default='scene')
    ap.add_argument('--F', default='1,2,4,8,16,32')
    ap.add_argument('--draws', type=int, default=2, help='how many injected δ (each applied to all frames)')
    ap.add_argument('--n-frames', type=int, default=200, help='frames evenly spaced over val')
    ap.add_argument('--n-groups', type=int, default=40, help='random groups of F frames per draw')
    ap.add_argument('--max-scenes', type=int, default=0,
                    help='>0: use every frame of this many scenes (evenly spaced by name) instead of --n-frames')
    ap.add_argument('--seed', type=int, default=20261008)
    ap.add_argument('--out', default='experiments/multiframe')
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    model, c = load_model(a.exp, ckpt=a.ckpt)
    D, H, inj, scene, T, Tin = per_frame(model, c, a.cache, a.draws, a.seed,
                                         os.path.join(a.out, f'perframe_{a.exp}_{a.ckpt.split(".")[0]}_{a.tag}_'
                                                      f'{f"sc{a.max_scenes}" if a.max_scenes else a.n_frames}.npz'),
                                         a.n_frames, a.max_scenes)
    rng = np.random.default_rng(a.seed)
    res = {}
    print(f'\n{a.exp} {a.tag} group={a.group}  frames {len(scene)}  draws {a.draws}')
    print(f'{"F":>3} {"rule":>6} {"n":>4} | {"rot geo med":>11} {"p90":>7} {"max":>7} | {"t med":>7} {"max":>7} |'
          f' {"|yaw| max":>9} {"|pitch| max":>11} {"|roll| max":>10} | {"|x| max":>7} {"|y| max":>7} {"|z| max":>7}')
    for F in [int(x) for x in a.F.split(',')]:
        G = groups_of(scene, F, a.group, rng, a.n_groups)
        if not G:
            continue
        for rule in (['single'] if F == 1 else ['sum', 'gate3', 'huber', 'tukey']):
            rows = []
            for d in range(a.draws):
                for gi in G:
                    if rule == 'single':
                        x = D[d, gi[0]]
                    else:
                        x = fuse(H[d, gi], D[d, gi], mode=rule)[0]
                    f0 = gi[0]                                  # same rig → same δ for every frame
                    Tc = apply_delta(Tin[d, f0], x)
                    geo = pose_error(Tc, T[f0]); ax = pose_error_axes(Tc, T[f0])
                    rows.append([geo[0], geo[1]] + [abs(ax[k]) for k in AX])
            r = np.array(rows)
            res[f'{F}_{rule}'] = dict(F=F, rule=rule, n=len(r), rot_med=float(np.median(r[:, 0])),
                                      rot_p90=float(np.percentile(r[:, 0], 90)), rot_max=float(r[:, 0].max()),
                                      t_med=float(np.median(r[:, 1])), t_max=float(r[:, 1].max()),
                                      axes_med=np.median(r[:, 2:], 0).tolist(), axes_max=r[:, 2:].max(0).tolist())
            mx = r[:, 2:].max(0)
            print(f'{F:>3} {rule:>6} {len(r):>4} | {np.median(r[:,0]):11.4f} {np.percentile(r[:,0],90):7.4f} '
                  f'{r[:,0].max():7.4f} | {np.median(r[:,1]):7.4f} {r[:,1].max():7.4f} | {mx[0]:9.4f} {mx[1]:11.4f} '
                  f'{mx[2]:10.4f} | {mx[3]:7.4f} {mx[4]:7.4f} {mx[5]:7.4f}', flush=True)
    with open(os.path.join(a.out, f'fusion_{a.exp}_{a.tag}_{a.group}.json'), 'w') as f:
        json.dump(res, f, indent=1)


if __name__ == '__main__':
    main()
