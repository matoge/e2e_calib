"""Full Waymo cache rebuild with fixed cache-side sampling
(margin=0.5, 2px grid 64×64 over 2×ROI)."""
import sys, pathlib; sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[2]))
from datasets.waymo import build_cache
build_cache('/tmp/waymo_v2_cache.pt', max_segs=None, random_crops=True,
            frame_sample=0.05, num_workers=8)
