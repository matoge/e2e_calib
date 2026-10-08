#!/bin/bash
# intensity を [0,1] に正規化し、画像の外 256 px まで点を残したキャッシュを作り直す (2026-10-08)。
# 設定は既存の ns_850x4 (build_ns_850.sh) と pandaset_v3_full (既定値: stride 1, val-frac 0.15, seed 42) と同じ。
set -euo pipefail
cd "$(dirname "$0")"
export PYTHONPATH=$PWD
export LD_LIBRARY_PATH=/home/hiro/miniconda3/envs/bilateraldriving/lib:${LD_LIBRARY_PATH:-}
PY=/home/hiro/miniconda3/envs/neurad/bin/python
C=/mnt/ssd2t/work/e2e_calib/cache
SRC=/mnt/ssd2t/work/nuScenes/trainval
(
  $PY -u scripts/preprocessing/build_pandaset_full_v3.py --root /mnt/ssd2t/work/pandaset \
      --out $C/pandaset_pad256 --cam front_camera --stride 1 --val-frac 0.15 --seed 42 --workers 4
  $PY -u scripts/preprocessing/convert_tile_cache_to_lmdb.py --cache $C/pandaset_pad256 --workers 4
  echo DONE_PANDASET
) > $C/../build_pandaset_pad256.log 2>&1 &
(
  $PY -u scripts/preprocessing/build_nuscenes_v3.py --data-root $SRC --meta-dir $SRC/v1.0-trainval \
      --out $C/ns_850x4_pad256 --cams CAM_FRONT --stride 1 --frame-frac 0.1 --val-frac 0.1 --workers 2
  $PY -u scripts/preprocessing/convert_tile_cache_to_lmdb.py --cache $C/ns_850x4_pad256 --workers 4
  echo DONE_NS
) > $C/../build_ns_850x4_pad256.log 2>&1 &
wait
echo DONE_ALL
