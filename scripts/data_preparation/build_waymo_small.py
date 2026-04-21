"""Small Waymo cache (5 segments, 10% frames) for density verification."""
from dataset_waymo import build_cache
build_cache('/tmp/waymo_small_cache.pt', max_segs=5, random_crops=True,
            frame_sample=0.1, num_workers=4)
