"""For each tile cache, look at how many lidar points each instance has
in the cached uv_full. We don't care about per-crop windowing here —
just the raw count per cached tile.
"""
import sys
sys.path.insert(0, '/home/hfunaya/git/e2e_calib')

from datasets.pandaset_full import PandaSetCalibDatasetFull, _unpack_lmdb_inst
import numpy as np

CACHES = [
    '/home/hfunaya/cache_v4/kamikado_v3_tiled',
    '/home/hfunaya/cache_v4/woven_v3_tile',
    '/home/hfunaya/clearml/data/cache/waymo_v3_tiled_i',
]

THRESHOLDS = [8, 32, 64, 128, 256, 512]

for cp in CACHES:
    ds = PandaSetCalibDatasetFull(cp, split='train', img_size=128)
    N = min(2000, len(ds))
    counts = []
    rng = np.random.RandomState(0)
    sample_idx = rng.choice(len(ds), size=N, replace=False)
    for i in sample_idx:
        try:
            inst = ds._load_inst(int(i))
        except Exception:
            continue
        if 'uv_full' in inst:
            uv = inst['uv_full']
            n = uv.shape[0] if hasattr(uv, 'shape') else len(uv)
        elif 'pts' in inst:
            n = len(inst['pts'])
        else:
            n = 0
        counts.append(int(n))
    counts = np.array(counts)
    print(f'\n  {cp.split("/")[-1]}  N={len(counts)}  '
          f'mean={counts.mean():.0f} median={int(np.median(counts))} '
          f'p10={int(np.percentile(counts, 10))} p1={int(np.percentile(counts, 1))}')
    for t in THRESHOLDS:
        below = (counts < t).sum() / len(counts) * 100
        print(f'    < {t:>4d} pts:  {below:5.1f}% of tiles')
