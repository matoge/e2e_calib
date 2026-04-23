"""Build multicrop caches for PS, NS, WM (sequential, OOM-safe).

Outputs to /mnt/nvme6t/e2e_calib_cache/:
    pandaset_mc_cache.pt
    nuscenes_mc_cache.pt
    waymo_mc_cache.pt
"""
import sys, pathlib; sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[2]))
import sys, time
from pathlib import Path

CACHE_DIR = Path('/mnt/nvme6t/e2e_calib_cache')
CACHE_DIR.mkdir(parents=True, exist_ok=True)

PS_CACHE = CACHE_DIR / 'pandaset_mc_s64_cache.pt'
NS_CACHE = CACHE_DIR / 'nuscenes_mc_s64_cache.pt'
WM_CACHE = CACHE_DIR / 'waymo_mc_s64_cache.pt'


def build_ps():
    from dataset_pandaset import build_cache
    print(f"=== PS build ===  → {PS_CACHE}", flush=True)
    t = time.time()
    build_cache('/mnt/mininas/datasets/pandaset', str(PS_CACHE),
                val_fraction=0.15, seed=42, random_crops=True,
                bbox_scale=3.0, min_pts=8, frame_sample=0.3)
    print(f"=== PS done in {(time.time()-t)/60:.1f}min ===", flush=True)


def build_ns():
    from dataset_nuscenes import build_cache
    print(f"=== NS build ===  → {NS_CACHE}", flush=True)
    t = time.time()
    build_cache('/mnt/nvme6t/nuscenes/data', str(NS_CACHE),
                version='v1.0-trainval', val_fraction=0.15, seed=42,
                random_crops=True, bbox_scale=3.0, min_pts=8,
                frame_sample=0.15, max_per_scene=100, num_workers=12)
    print(f"=== NS done in {(time.time()-t)/60:.1f}min ===", flush=True)


def build_wm():
    from dataset_waymo import build_cache
    print(f"=== WM build ===  → {WM_CACHE}", flush=True)
    t = time.time()
    build_cache(str(WM_CACHE), max_segs=None, random_crops=True,
                frame_sample=0.15, num_workers=4)
    print(f"=== WM done in {(time.time()-t)/60:.1f}min ===", flush=True)


def main():
    which = sys.argv[1:] or ['ps', 'ns', 'wm']
    for tag in which:
        if   tag == 'ps': build_ps()
        elif tag == 'ns': build_ns()
        elif tag == 'wm': build_wm()
        else: raise SystemExit(f"unknown: {tag}")


if __name__ == '__main__':
    main()
