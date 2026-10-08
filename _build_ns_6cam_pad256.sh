#!/bin/bash
# nuScenes 6 カメラ (ns_850x4 と同じ 850 シーン × 4 フレーム、--frame-frac 0.1) を今の builder で作る。
# intensity /255、画像の外 256 px まで点、LiDAR だけ。
set -euo pipefail
cd "$(dirname "$0")"
export PYTHONPATH=$PWD LD_LIBRARY_PATH=/home/hiro/miniconda3/envs/bilateraldriving/lib:${LD_LIBRARY_PATH:-}
PY=/home/hiro/miniconda3/envs/neurad/bin/python
SRC=/mnt/ssd2t/work/nuScenes/trainval
OUT=/mnt/ssd2t/work/e2e_calib/cache/ns_850x4_6cam_pad256
$PY -u scripts/preprocessing/build_nuscenes_v3.py --data-root $SRC --meta-dir $SRC/v1.0-trainval --out $OUT \
  --cams CAM_FRONT,CAM_FRONT_LEFT,CAM_FRONT_RIGHT,CAM_BACK,CAM_BACK_LEFT,CAM_BACK_RIGHT \
  --stride 1 --frame-frac 0.1 --val-frac 0.1 --workers 2
$PY -u scripts/preprocessing/convert_tile_cache_to_lmdb.py --cache $OUT --workers 4
echo DONE
