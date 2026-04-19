"""Full NuScenes trainval cache rebuild with fixed cache-side sampling
(margin=0.5, 2px grid 64×64 over 2×ROI)."""
from dataset_nuscenes import build_cache

build_cache(
    nuscenes_root='/mnt/nvme6t/nuscenes/data',
    cache_path='/tmp/nuscenes_static_cache.pt',
    version='v1.0-trainval',
    val_fraction=0.15,
    random_crops=True,
    bbox_scale=2.0,
    min_pts=8,
    num_workers=16,
)
