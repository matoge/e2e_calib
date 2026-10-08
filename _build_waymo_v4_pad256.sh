#!/bin/bash
# Waymo v2 → キャッシュ (build_waymo_v4.py)。nuScenes 6 カメラと同じくらいの大きさ:
# 1 Hz (--stride 10)、train 100 セグメント × 5 カメラ ≈ 10,000、val 10 セグメント ≈ 1,000。
set -euo pipefail
cd "$(dirname "$0")"
export PYTHONPATH=$PWD
PY=/home/hiro/miniconda3/envs/neurad/bin/python
OUT=/mnt/ssd2t/work/e2e_calib/cache/waymo_v4_5cam_pad256
$PY -u scripts/preprocessing/build_waymo_v4.py --src /mnt/ssd2t/work/waymo_v2 --out $OUT --stride 10 --val-max-segs 10 --workers 2
$PY -u scripts/preprocessing/convert_tile_cache_to_lmdb.py --cache $OUT --workers 4
echo DONE
