"""Patch nerfstudio's full_images_lidar_datamanager.py to force eval
n_batches=1 (= no batched repeat in lidar_rasterization, avoids OOM on
high-density VLS128 frames).

Idempotent: if already patched, no-op.
"""
import re
from pathlib import Path

P = Path('/workspace/neurad-studio/nerfstudio/data/datamanagers/'
         'full_images_lidar_datamanager.py')
src = P.read_text()
marker = '# PATCH: force n_batches=1'
if marker in src:
    print('[patch] already applied')
else:
    old = '            max_points_per_tile = ELEV_CHANNELS_PER_TILE * AZIM_CHANNELS_PER_TILE'
    new = ('            max_points_per_tile = ELEV_CHANNELS_PER_TILE * '
           'AZIM_CHANNELS_PER_TILE * 1024  ' + marker)
    if old not in src:
        raise SystemExit('[patch] could not find target line')
    src = src.replace(old, new)
    P.write_text(src)
    print('[patch] applied: max_points_per_tile *= 1024')
