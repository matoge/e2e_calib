"""Rebuild a small NuScenes cache (~5 scenes) for density verification."""
import sys, pathlib; sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[2]))
from datasets.nuscenes import build_cache

build_cache(
    nuscenes_root='/mnt/nvme6t/nuscenes/data',
    cache_path='/tmp/nuscenes_small_cache.pt',
    version='v1.0-trainval',
    val_fraction=0.15,
    max_scenes=5,
    random_crops=True,
    bbox_scale=2.0,
    min_pts=8,
    num_workers=5,
)
