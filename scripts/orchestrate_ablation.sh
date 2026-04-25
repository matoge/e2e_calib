#!/bin/bash
# Sequential ablation orchestrator:
#   v41: stratified only      (--no-frustum-full)
#   v42: stratified + dense   (default frustum-full=True)
# Both run on PandaSet 103-scene front_camera, img=64, crop=128-256, 60 ep.
# Pass criteria evaluated by best_val_err parse.
# v43: combined production run, gated on v41/v42 pass.
#
# Logs:
#   /tmp/orch_summary.csv          — pass/fail table
#   /tmp/<name>.log                — per-run training log
set -u
ROOT=/home/hiro/git/e2e_calib
SUMMARY=/tmp/orch_summary.csv

PANDA_103=/mnt/nvme6t/pandaset
ALL_DATA=/mnt/nvme6t/pandaset,/mnt/nvme6t/waymo_ps,/mnt/nvme6t/nuscenes_ps,/mnt/nvme6t/av2_ps,/mnt/nvme6t/ddad_ps

cd "$ROOT"

# === helpers ====================================================================

log()  { echo "[orch $(date +'%H:%M:%S')] $*"; }
parse_best_val()   { grep -oP 'best val err = \K[0-9.]+' "$1" | tail -1; }
parse_best_val_nll() { grep -oP 'val_nll=\K[0-9.]+' "$1" | sort -n | head -1; }
parse_total_eps()  { grep -cE '^\[.*\] ep ' "$1"; }

run_and_wait() {
    local name=$1 ; shift
    local logf="/tmp/${name}.log"
    rm -rf "$ROOT/experiments/cross_frame_${name}" 2>/dev/null
    log "kicking $name → $logf"
    nohup python -u train_cross_frame.py --name "$name" "$@" > "$logf" 2>&1 &
    local pid=$!
    log "  pid=$pid  waiting…"
    while kill -0 "$pid" 2>/dev/null; do sleep 30; done
    log "  $name done"
    sleep 3   # let log flush
    local best_val=$(parse_best_val "$logf")
    local best_nll=$(parse_best_val_nll "$logf")
    local n_eps=$(parse_total_eps "$logf")
    log "  best_val_err=$best_val  best_val_nll=$best_nll  ep_seen=$n_eps"
    echo "$name,$best_val,$best_nll,$n_eps,$logf" >> "$SUMMARY"
    echo "$best_val"
}

float_le() { awk -v a="$1" -v b="$2" 'BEGIN { exit !(a+0 <= b+0) }'; }

# === init summary ===============================================================
[ -f "$SUMMARY" ] || echo "name,best_val_err,best_val_nll,n_eps,log" > "$SUMMARY"

# === v41: stratified-only baseline ==============================================
v41_val=$(run_and_wait v41_strat_panda103 --full \
    --scene "$PANDA_103/015" --scenes-root "$PANDA_103" \
    --train-frac 0.8 --cameras front_camera \
    --img-size 64 --max-points 256 --batch-size 32 --lr 1e-3 \
    --n-overfit 64 --epochs 100 --log-every 1 \
    --baseline-min 1 --baseline-max 20 \
    --sigma-ypr 1.0 --sigma-t 0.2 \
    --crop-min 128 --crop-max 256 \
    --num-workers 16 --virtual-epoch 20000 \
    --deform-mode sl --n-cross-layers 2 --n-intra-layers 2 \
    --no-uvd --no-frustum-full \
    --mine-val --val-pool-size 1000 --migrate-k 500 \
    --overfit-patience 3 --overfit-metric nll \
    --rewind-back 3 --rewind-lr-reset \
    --sentinel-size 1000 --stop-no-improve-migrations 3)

if ! float_le "$v41_val" 4.0; then
    log "v41 FAIL (val_err=$v41_val > 4.0); aborting."
    exit 1
fi
log "v41 PASS (val_err=$v41_val)"

# === v42: stratified + frustum-full ============================================
v42_val=$(run_and_wait v42_strat_dense_panda103 --full \
    --scene "$PANDA_103/015" --scenes-root "$PANDA_103" \
    --train-frac 0.8 --cameras front_camera \
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
    --sentinel-size 1000 --stop-no-improve-migrations 3)

# v42 must not be more than +20% worse than v41 (1.2× factor)
v42_threshold=$(awk -v a="$v41_val" 'BEGIN{print a*1.2}')
if ! float_le "$v42_val" "$v42_threshold"; then
    log "v42 FAIL (val_err=$v42_val > 1.2× v41 = $v42_threshold); aborting v43."
    exit 1
fi
log "v42 PASS (val_err=$v42_val ≤ $v42_threshold)"

# === v43: combined production run ==============================================
log "Phase 2 — combined run (img=128, batch=64, all datasets, val mining)"
run_and_wait v43_combined_full --full \
    --scene "$PANDA_103/015" --scenes-root "$ALL_DATA" \
    --train-frac 0.8 --cameras all \
    --img-size 128 --max-points 256 --batch-size 64 --lr 1e-3 \
    --n-overfit 64 --epochs 100 --log-every 1 \
    --baseline-min 1 --baseline-max 20 \
    --sigma-ypr 1.0 --sigma-t 0.2 \
    --crop-min 128 --crop-max 384 \
    --num-workers 24 --virtual-epoch 20000 \
    --deform-mode sl --n-cross-layers 2 --n-intra-layers 2 --uvd \
    --frustum-full \
    --mine-val --val-pool-size 4000 --migrate-k 500 \
    --overfit-patience 3 --overfit-metric nll \
    --rewind-back 3 --rewind-lr-reset \
    --sentinel-size 1000 --stop-no-improve-migrations 3

log "all done — summary at $SUMMARY"
column -t -s, "$SUMMARY"
