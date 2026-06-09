#!/bin/bash
# SplatAD SO3xR3 sweep on DGX-3 (16 V100s):
#   - Each scene = 1 GPU
#   - 16M Gaussian cap; on OOM, automatic re-try with 10M
#   - ClearML task per scene (project=splatad/ps_so3xr3)
#   - First/last frame vs GT 3-up image posted to ClearML
#
# Usage:
#   bash scripts/splatad/run_all_ps_dgx3.sh [SCENES...]
#   (default: all 103 PandaSet scenes)
set -euo pipefail
PS_ROOT=${PS_ROOT:-/mnt/fsx/tmp/hfunaya/pandaset}
OUT_ROOT=${OUT_ROOT:-/raid/home/hfunaya/splatad_ps_outputs_y1}
IMAGE=${IMAGE:-splatad-v100:neurad}
ITERS=${ITERS:-30001}
N_PARALLEL=${N_PARALLEL:-16}     # 16 V100s on DGX-3
TS=$(date +%m%d_%H%M)

GPUS=()
for i in $(seq 0 $((N_PARALLEL - 1))); do GPUS+=($i); done

mapfile -t SCENES < <(if [ "$#" -gt 0 ]; then for s in "$@"; do echo "$s"; done; \
                       else ls "$PS_ROOT" | grep -E '^[0-9]{3}$' | sort; fi)
mkdir -p "$OUT_ROOT" "$OUT_ROOT/_compare" "$OUT_ROOT/_logs"
echo "[info] $(date -Is) scenes=${#SCENES[@]} parallel=$N_PARALLEL iters=$ITERS"

run_scene() {
    local gpu=$1; local scene=$2; local cap=$3
    local exp="ps${scene}_front_so3xr3_${TS}_cap${cap}"
    local exp_dir="$OUT_ROOT/${exp}"
    local log="$OUT_ROOT/_logs/${exp}_train.log"
    mkdir -p "$exp_dir"

    echo "[gpu $gpu] === $scene cap=${cap}M ===" >&2

    # ClearML task init + log streamer (host-side, follows train.log)
    /home/hfunaya/.pyenv/versions/3.10.4/bin/python - "$exp" "$scene" "$log" "$cap" <<'PY' &
import sys, time, re, os
exp, scene, log_path, cap = sys.argv[1:5]
from clearml import Task
t = Task.init(project_name='splatad/ps_so3xr3', task_name=exp,
              task_type=Task.TaskTypes.training, reuse_last_task_id=False,
              auto_connect_frameworks={'tensorboard': False, 'pytorch': False})
t.connect({'scene': scene, 'cap_M': int(cap), 'log_path': log_path})
log = t.get_logger()
while not os.path.exists(log_path): time.sleep(2)
re_iter = re.compile(r'Step \(\s*(\d+)\)')
re_kv = re.compile(r'(\w+)\s*=\s*([-\d\.eE+]+)')
with open(log_path) as f:
    f.seek(0, 2)
    while True:
        line = f.readline()
        if not line:
            time.sleep(1); continue
        m = re_iter.search(line)
        if m:
            it = int(m.group(1))
            for k, v in re_kv.findall(line):
                if k == 'Step': continue
                try: vf = float(v)
                except ValueError: continue
                log.report_scalar(title='train', series=k, value=vf, iteration=it)
PY
    local tailer_pid=$!

    docker run --rm --runtime=nvidia \
        -e NVIDIA_VISIBLE_DEVICES=$gpu \
        -e CUDA_VISIBLE_DEVICES=0 \
        -v "$PS_ROOT":/data/pandaset:ro \
        -v "$OUT_ROOT":/workspace/outputs \
        --shm-size=16g --ipc=host \
        --name "splatad_${exp}" \
        $IMAGE \
        bash -lc "
            ns-train splatad \
                --output-dir /workspace/outputs \
                --experiment-name '$exp' \
                --max-num-iterations $ITERS \
                --viewer.quit-on-train-completion True \
                --vis tensorboard \
                --pipeline.model.mcmc-cap-max ${cap}000000 \
                --pipeline.model.camera-optimizer.mode SO3xR3 \
                pandaset-data --data /data/pandaset --sequence $scene \
                              --add-missing-points False --cameras front \
            && CFG=\$(ls -1t /workspace/outputs/$exp/splatad/*/config.yml | head -1) \
            && python3.10 -c \"
import yaml
with open(r'\$CFG') as f: cfg = yaml.safe_load(f)
co = cfg.get('pipeline',{}).get('model',{}).get('camera_optimizer',{})
co['use_camopt_in_eval'] = True
with open(r'\$CFG','w') as f: yaml.dump(cfg, f, sort_keys=False)
\" \
            && ns-render dataset --load-config \$CFG \
                --output-path /workspace/outputs/${exp}/render_train \
                --pose-source train --rendered-output-names rgb
        " > "$log" 2>&1
    local rc=$?
    kill $tailer_pid 2>/dev/null || true

    # Detect OOM and retry with 10M cap (only if we tried 16M)
    if [ $rc -ne 0 ] && [ "$cap" = "16" ] && grep -q "OutOfMemoryError" "$log" 2>/dev/null; then
        echo "[gpu $gpu] $scene OOM at 16M → retry with 10M" >&2
        rm -rf "$exp_dir"
        run_scene $gpu $scene 10
        return $?
    fi

    if [ $rc -eq 0 ]; then
        # Build first/last vs GT compare image + post to ClearML
        /home/hfunaya/.pyenv/versions/3.10.4/bin/python - "$exp" "$scene" "$OUT_ROOT" <<'PY'
import sys, os
from pathlib import Path
exp, scene, out_root = sys.argv[1:4]
out = Path(out_root) / exp
render_dir = None
for c in (out / 'render_train').rglob('*'):
    if c.is_file() and c.suffix.lower() in ('.png', '.jpg'):
        render_dir = c.parent; break
if render_dir is None:
    print(f'no rendered frames for {exp}'); sys.exit(0)
files = sorted(p for p in render_dir.iterdir() if p.suffix.lower() in ('.png', '.jpg'))
if len(files) < 2: sys.exit(0)
import numpy as np
from PIL import Image, ImageDraw, ImageFont
from clearml import Task
def gt_for(rp, scene):
    try: idx = int(rp.stem)
    except: return None
    gt_dir = Path(f'/mnt/fsx/tmp/hfunaya/pandaset/{scene}/camera/front_camera')
    for p in (gt_dir / f'{idx:02d}.jpg', gt_dir / f'{idx:03d}.jpg'):
        if p.exists(): return p
    if gt_dir.is_dir():
        fl = sorted(gt_dir.glob('*.jpg'))
        if 0 <= idx < len(fl): return fl[idx]
    return None
def make_3up(r_p, g_p, title):
    r = np.asarray(Image.open(r_p).convert('RGB'))
    g = np.asarray(Image.open(g_p).convert('RGB'))
    if r.shape != g.shape:
        g = np.asarray(Image.open(g_p).convert('RGB').resize((r.shape[1], r.shape[0])))
    diff = np.abs(r.astype(np.int16) - g.astype(np.int16)).clip(0, 255).astype(np.uint8)
    panel = np.concatenate([r, g, diff], axis=1)
    img = Image.fromarray(panel)
    d = ImageDraw.Draw(img)
    try: f = ImageFont.truetype('/usr/share/fonts/truetype/dejavu/DejaVuSansMono-Bold.ttf', 22)
    except: f = ImageFont.load_default()
    d.text((10, 10), title, fill='yellow', font=f)
    mse = ((r.astype(np.float32) - g.astype(np.float32))**2).mean()
    psnr = 99.0 if mse < 1e-9 else 10.0 * np.log10(255.0**2 / mse)
    d.text((10, 38), f'PSNR={psnr:.2f} dB', fill='lime', font=f)
    return img, psnr
t = None
try:
    ts = Task.get_tasks(task_name=exp, project_name='splatad/ps_so3xr3')
    if ts: t = ts[0]
except: pass
logger = t.get_logger() if t is not None else None
for tag, rp in (('first', files[0]), ('last', files[-1])):
    gp = gt_for(rp, scene)
    if gp is None: print(f'{tag}: GT not found'); continue
    img, psnr = make_3up(rp, gp, f'{exp} {tag} {rp.stem}  L=render  C=GT  R=|diff|')
    out_p = out / f'compare_{tag}.png'
    img.save(out_p, quality=92)
    print(f'{exp} {tag}: PSNR={psnr:.2f} → {out_p}')
    if logger is not None:
        logger.report_image(title='compare', series=tag, iteration=0, local_path=str(out_p))
        logger.report_scalar(title='final_psnr', series=tag, value=float(psnr), iteration=0)
PY
    fi
    return $rc
}

declare -A pid2info
for i in "${!SCENES[@]}"; do
    while [ "${#pid2info[@]}" -ge "$N_PARALLEL" ]; do
        for pid in "${!pid2info[@]}"; do
            if ! kill -0 "$pid" 2>/dev/null; then
                wait "$pid" 2>/dev/null && echo "[ok] ${pid2info[$pid]}" || echo "[FAIL] ${pid2info[$pid]}"
                unset pid2info[$pid]
                break
            fi
        done
        sleep 5
    done
    gpu=${GPUS[$((i % ${#GPUS[@]}))]}
    scene=${SCENES[$i]}
    run_scene $gpu $scene 16 &
    pid=$!
    pid2info[$pid]="gpu=$gpu scene=$scene"
    echo "[$((i + 1))/${#SCENES[@]}] kicked $scene gpu=$gpu (16M cap)"
done

echo "[info] waiting for ${#pid2info[@]} stragglers"
for pid in "${!pid2info[@]}"; do
    wait $pid 2>/dev/null && echo "[ok-final] ${pid2info[$pid]}" || echo "[FAIL-final] ${pid2info[$pid]}"
done

echo "[done] $(date -Is) all $((${#SCENES[@]})) scenes processed → $OUT_ROOT/_compare/"
