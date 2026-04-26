#!/bin/bash
# Per-dataset baseline using v42 recipe (best so far) — front cam only,
# stratified + frustum-full, img=64, crop=128-256, mining on.
# Sequential runs; each ~25-40 min.
set -u
ROOT=/home/hiro/git/e2e_calib
SUMMARY=/tmp/orch_perdataset.csv
cd "$ROOT"

log() { echo "[orch $(date +'%H:%M:%S')] $*"; }
parse_best_val()    { grep -oP 'best val err = \K[0-9.]+' "$1" | tail -1; }
parse_best_val_nll(){ grep -oP 'val_nll=\K[0-9.]+' "$1" | sort -n | head -1; }
parse_total_eps()   { grep -cE '^\[.*\] ep ' "$1"; }

[ -f "$SUMMARY" ] || echo "name,dataset,best_val_err,best_val_nll,n_eps,log" > "$SUMMARY"

run() {
    local name=$1 ds_root=$2 cam=$3 ds_label=$4
    local logf="/tmp/${name}.log"
    log "kicking $name ($ds_label, cam=$cam)"
    rm -rf "$ROOT/experiments/cross_frame_${name}" 2>/dev/null
    nohup python -u train_cross_frame.py --name "$name" --full \
        --scene "${ds_root%,*}/$(ls "${ds_root%,*}" | head -1)" \
        --scenes-root "$ds_root" \
        --train-frac 0.8 --cameras "$cam" \
        --img-size 64 --max-points 256 --batch-size 32 --lr 1e-3 \
        --n-overfit 64 --epochs 100 --log-every 1 \
        --baseline-min 1 --baseline-max 20 \
        --sigma-ypr 1.0 --sigma-t 0.2 \
        --crop-min 128 --crop-max 256 \
        --num-workers 16 --virtual-epoch 20000 \
        --deform-mode sl --n-cross-layers 2 --n-intra-layers 2 \
        --no-uvd --frustum-full \
        --mine-val --val-pool-size 1000 --migrate-k 500 \
        --overfit-patience 3 --overfit-metric nll \
        --rewind-back 3 --rewind-lr-reset \
        --sentinel-size 1000 --stop-no-improve-migrations 3 \
        > "$logf" 2>&1 &
    local pid=$!
    log "  pid=$pid waiting…"
    while kill -0 "$pid" 2>/dev/null; do sleep 30; done
    sleep 3
    local v=$(parse_best_val "$logf")
    local n=$(parse_best_val_nll "$logf")
    local e=$(parse_total_eps "$logf")
    echo "$name,$ds_label,$v,$n,$e,$logf" >> "$SUMMARY"
    log "  $name done: val_err=$v val_nll=$n eps=$e"
}

# PandaSet
run v44_panda_only      /mnt/nvme6t/pandaset    front_camera    pandaset
# Waymo (converted)
run v45_waymo_only      /mnt/nvme6t/waymo_ps    front_camera    waymo
# NuScenes (6cam-converted)
run v46_ns_only         /mnt/nvme6t/nuscenes_ps front_camera    nuscenes
# AV2 — front center (portrait)
run v47_av2_only        /mnt/nvme6t/av2_ps      ring_front_center  av2
# DDAD — CAMERA_01 is the narrow front
run v48_ddad_only       /mnt/nvme6t/ddad_ps     CAMERA_01       ddad

log "ALL DONE — summary at $SUMMARY"
column -t -s, "$SUMMARY"
