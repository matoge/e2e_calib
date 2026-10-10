#!/bin/bash
# Waymo v2 → キャッシュ (build_waymo_v4.py、Waymo 公式の投影 = cp 点と同じ連鎖)。nuScenes 6 カメラと同じくらいの大きさ:
# 1 Hz (--stride 10)、train 100 セグメント × 5 カメラ ≈ 10,000、val 10 セグメント ≈ 1,000。
# builder は waymo env (tensorflow + waymo-open-dataset + torch cpu)、1 セグメントで RAM ~13 GB なので 1 worker。
set -euo pipefail
cd "$(dirname "$0")"
export PYTHONPATH=$PWD TURBOJPEG_LIB=/home/hiro/miniconda3/envs/neurad/lib/libturbojpeg.so.0
OUT=/mnt/ssd2t/work/e2e_calib/cache/waymo_v5_5cam_z2_pad256
/home/hiro/miniconda3/envs/waymo/bin/python -u scripts/preprocessing/build_waymo_v4.py --src /mnt/ssd2t/work/waymo_v2 --out $OUT --stride 10 --val-max-segs 10 --workers 1 --min-depth 2.0
/home/hiro/miniconda3/envs/neurad/bin/python -u scripts/preprocessing/convert_tile_cache_to_lmdb.py --cache $OUT --workers 4
/home/hiro/miniconda3/envs/neurad/bin/python scripts/visualization/vis_cache_proj.py --cache $OUT --out docs/_figs/2026-10-09/waymo_v5_full_val.jpg --split val --n 10
echo DONE
