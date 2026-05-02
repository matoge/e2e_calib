"""Post-hoc inject `uv_full`, `z_cam`, `is_obj` into existing PandaSet v3 cache.

Old inst.pt lacked these keys → __getitem__ recomputed is_obj (13k pts × 160 cubs)
at 412ms/call, capping SPS to ~200 even with 90 workers. Pre-injection reduces
getitem to ~20ms → 4000+ sps target.

Usage:
    python inject_pandaset_is_obj.py --cache /dev/shm/pandaset_v3_full --workers 48
"""
import argparse, sys, time
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
import numpy as np
import torch
from concurrent.futures import ProcessPoolExecutor, as_completed

from datasets.pandaset_full import _is_obj_per_point


def _inject_one(args_tuple):
    path_str, overwrite = args_tuple
    path = Path(path_str)
    try:
        inst = torch.load(path, weights_only=False)
    except Exception as e:
        return path.name, False, f'load_err:{e}'

    # Skip if already done (unless --overwrite)
    if (not overwrite) and ('is_obj' in inst) and ('uv_full' in inst) and ('z_cam' in inst):
        return path.name, True, 'skip'

    pts_vis = inst['pts'].numpy()
    K = inst['K_full'].numpy()
    T_gt = inst['T_gt'].numpy()
    cubs = inst.get('cuboids', [])

    # Re-project: pts (world) → cam frame via T_gt, then K
    homo = np.column_stack([pts_vis, np.ones(len(pts_vis), dtype=np.float32)])
    pcam = (T_gt @ homo.T)[:3].T              # (N, 3)
    z_vis = pcam[:, 2].astype(np.float32)
    uv_vis = ((K @ pcam.T)[:2] /
              np.maximum(pcam[:, 2:].T, 1e-6)).T.astype(np.float32)
    is_obj_vis = _is_obj_per_point(pts_vis, cubs)                   # (N,) float32

    inst['uv_full'] = torch.from_numpy(uv_vis)
    inst['z_cam']   = torch.from_numpy(z_vis)
    inst['is_obj']  = torch.from_numpy(is_obj_vis)

    tmp = path.with_suffix(path.suffix + '.tmp')
    torch.save(inst, tmp)
    tmp.replace(path)
    return path.name, True, f'N={len(pts_vis)} obj={int(is_obj_vis.sum())} cubs={len(cubs)}'


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--cache',   default='/dev/shm/pandaset_v3_full')
    ap.add_argument('--workers', type=int, default=48)
    ap.add_argument('--overwrite', action='store_true',
                    help='re-inject even if keys already present')
    ap.add_argument('--limit', type=int, default=None,
                    help='only process first N files (smoke test)')
    args = ap.parse_args()

    inst_dir = Path(args.cache) / 'inst'
    files = sorted(inst_dir.glob('*.pt'))
    if args.limit:
        files = files[:args.limit]
    print(f'cache={args.cache} files={len(files)} workers={args.workers}', flush=True)

    tuples = [(str(p), args.overwrite) for p in files]
    t0 = time.time()
    n_ok = n_skip = n_err = 0
    if args.workers <= 1:
        for i, tup in enumerate(tuples):
            name, ok, info = _inject_one(tup)
            if not ok:
                n_err += 1
            elif info == 'skip':
                n_skip += 1
            else:
                n_ok += 1
            if (i + 1) % 100 == 0 or i == len(tuples) - 1:
                dt = time.time() - t0
                print(f'  [{i+1}/{len(files)}] ok={n_ok} skip={n_skip} err={n_err} '
                      f'{(i+1)/max(dt,1e-6):.1f}/s  last={name} {info}', flush=True)
    else:
        with ProcessPoolExecutor(max_workers=args.workers) as ex:
            futs = [ex.submit(_inject_one, t) for t in tuples]
            done = 0
            for fut in as_completed(futs):
                name, ok, info = fut.result()
                if not ok:
                    n_err += 1
                elif info == 'skip':
                    n_skip += 1
                else:
                    n_ok += 1
                done += 1
                if done % 100 == 0 or done == len(files):
                    dt = time.time() - t0
                    print(f'  [{done}/{len(files)}] ok={n_ok} skip={n_skip} err={n_err} '
                          f'{done/max(dt,1e-6):.1f}/s  last={name} {info}', flush=True)

    print(f'\ndone: ok={n_ok} skip={n_skip} err={n_err} in {(time.time()-t0):.1f}s', flush=True)


if __name__ == '__main__':
    main()
