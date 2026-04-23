"""Small Waymo cache (5 segments, 10% frames) for density verification."""
import sys, pathlib; sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[2]))
from datasets.waymo import build_cache
build_cache('/tmp/waymo_small_cache.pt', max_segs=5, random_crops=True,
            frame_sample=0.1, num_workers=4)
