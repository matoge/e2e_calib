#!/bin/bash
# SplatAD SO3xR3 smoke on 4 PS scenes, then auto-render first+last frame
# vs GT for visual sanity. Output: a 3-row PNG per scene under
# /raid/home/hfunaya/splatad_ps_outputs_y1/_compare/.
#
# Run on 4 V100s (GPU 1-4). 30k iter ≈ 30 min/scene = ~30 min total wall.
set -euo pipefail

PS_ROOT=/mnt/fsx/tmp/hfunaya/pandaset
OUT_ROOT=/raid/home/hfunaya/splatad_ps_outputs_y1
IMAGE=splatad-v100:neurad
ITERS=30000
TS=$(date +%m%d_%H%M)

# Pick first 4 scenes that exist (deterministic).
SCENES=("$@")
if [ "${#SCENES[@]}" -eq 0 ]; then
    mapfile -t SCENES < <(ls "$PS_ROOT" | grep -E '^[0-9]{3}$' | sort | head -4)
fi
GPUS=(1 2 3 4)
echo "[info] $(date -Is) scenes=${SCENES[*]} | iters=$ITERS | out=$OUT_ROOT"
mkdir -p "$OUT_ROOT" "$OUT_ROOT/_compare"

run_scene() {
    local gpu=$1; local scene=$2
    local exp="ps${scene}_front_so3xr3_${TS}"
    local exp_dir="$OUT_ROOT/${exp}"
    mkdir -p "$exp_dir"

    echo "[gpu $gpu] === $scene → $exp ===" >&2

    # 1) Train
    docker run --rm --runtime=nvidia \
        -e NVIDIA_VISIBLE_DEVICES=$gpu \
        -e CUDA_VISIBLE_DEVICES=0 \
        -v "$PS_ROOT":/data/pandaset:ro \
        -v "$OUT_ROOT":/workspace/outputs \
        --shm-size=16g --ipc=host \
        --name "splatad_train_ps${scene}" \
        $IMAGE \
        bash -lc "
            ns-train splatad \
                --output-dir /workspace/outputs \
                --experiment-name '$exp' \
                --max-num-iterations $ITERS \
                --viewer.quit-on-train-completion True \
                --vis tensorboard \
                --pipeline.model.mcmc-cap-max 16000000 \
                --pipeline.model.camera-optimizer.mode SO3xR3 \
                pandaset-data \
                  --data /data/pandaset \
                  --sequence $scene \
                  --add-missing-points False \
                  --cameras front
        " > "$OUT_ROOT/${exp}_train.log" 2>&1
    train_rc=$?
    if [ $train_rc -ne 0 ]; then
        echo "[gpu $gpu] $scene TRAIN failed rc=$train_rc — see ${exp}_train.log" >&2
        return $train_rc
    fi

    # Find auto-generated config.yml (latest timestamp dir under exp/splatad/)
    CFG=$(ls -1dt "$OUT_ROOT/$exp/splatad"/*/config.yml 2>/dev/null | head -1)
    if [ -z "$CFG" ]; then
        echo "[gpu $gpu] $scene NO config.yml — skip render" >&2
        return 1
    fi
    # Translate host path → container path
    CFG_C="${CFG/$OUT_ROOT//workspace/outputs}"

    # 2) Patch config: turn use_camopt_in_eval on
    python3 -c "
import yaml, sys
with open('$CFG') as f: cfg = yaml.safe_load(f)
co = cfg.get('pipeline', {}).get('model', {}).get('camera_optimizer', {})
co['use_camopt_in_eval'] = True
with open('$CFG', 'w') as f: yaml.dump(cfg, f, sort_keys=False)
print('camopt_in_eval=on')
" >&2 || true

    # 3) Render train poses (= dataset frames with learned pose adj)
    docker run --rm --runtime=nvidia \
        -e NVIDIA_VISIBLE_DEVICES=$gpu \
        -e CUDA_VISIBLE_DEVICES=0 \
        -v "$PS_ROOT":/data/pandaset:ro \
        -v "$OUT_ROOT":/workspace/outputs \
        --shm-size=16g --ipc=host \
        --name "splatad_render_ps${scene}" \
        $IMAGE \
        bash -lc "
            ns-render dataset \
                --load-config '$CFG_C' \
                --output-path /workspace/outputs/${exp}/render_train \
                --pose-source train \
                --rendered-output-names rgb
        " > "$OUT_ROOT/${exp}_render.log" 2>&1
    render_rc=$?
    if [ $render_rc -ne 0 ]; then
        echo "[gpu $gpu] $scene RENDER failed rc=$render_rc" >&2
        return $render_rc
    fi

    echo "[gpu $gpu] $scene done." >&2
    return 0
}
export -f run_scene
export PS_ROOT OUT_ROOT IMAGE ITERS TS

# Launch 4 parallel
declare -A pid2scene
for i in "${!SCENES[@]}"; do
    gpu=${GPUS[$i]}
    scene=${SCENES[$i]}
    run_scene $gpu $scene &
    pid2scene[$!]=$scene
done

# Wait all
for pid in "${!pid2scene[@]}"; do
    wait $pid && echo "[ok] ${pid2scene[$pid]} (pid=$pid)" || echo "[FAIL] ${pid2scene[$pid]} (pid=$pid)"
done

# Build first/last + GT 3-up comparison images on host
echo "[info] generating comparison PNGs..."
/home/hfunaya/.pyenv/versions/3.10.4/bin/python /home/hfunaya/git/e2e_calib/scripts/splatad/compare_first_last_vs_gt.py \
    --out-root "$OUT_ROOT" \
    --ts "$TS" \
    --scenes "${SCENES[@]}" \
    --compare-dir "$OUT_ROOT/_compare"

echo "[done] $(date -Is) all done. compare PNGs → $OUT_ROOT/_compare/"
