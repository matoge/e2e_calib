#!/usr/bin/env bash
# Kick SplatAD on one Woven sequence.
#
# Usage:
#   ./_kick_splatad_woven.sh <SEQ_DIR> <VEHICLE> [GPU=4] [MAX_STEPS=20000]
#
# Mounts host woven_dataparser.py into the splatad-v100 container, registers
# it via a tiny shim into nerfstudio.configs.dataparser_configs, and runs
# ns-train splatad woven-data.

set -euo pipefail
SEQ_DIR=${1:?seq_dir}
VEHICLE=${2:?vehicle}
GPU=${3:-4}
MAX_STEPS=${4:-20000}

SCRIPTS=/home/hfunaya/git/e2e_calib/scripts/splatad_kb
NAME=splatad_$(basename "$SEQ_DIR" | sed 's/sequence=\([a-z0-9]*\).*/\1/')_$(date +%H%M)
OUT_PARENT=/raid/home/hfunaya/splatad_woven

# Docker bind:
#   /host_woven       -> $SCRIPTS                          (woven_dataparser.py + shim)
#   /seq              -> $SEQ_DIR
#   /workspace/outputs-> $OUT_PARENT/$NAME (mkdir inside container; root can write
#                                           even if $OUT_PARENT is root-owned)
docker run --rm --runtime=nvidia --shm-size=16g \
  --name "$NAME" \
  -e NVIDIA_VISIBLE_DEVICES=$GPU -e CUDA_VISIBLE_DEVICES=0 -e TORCH_CUDA_ARCH_LIST=7.0 \
  -e PYTHONPATH=/host_woven:/workspace/neurad-studio:/workspace/splatad \
  -e EXP_NAME="$NAME" \
  -v "$SCRIPTS":/host_woven:ro \
  -v "$SEQ_DIR":/seq:ro \
  -v "$OUT_PARENT":/out_parent \
  -v /home/hfunaya/clearml-dgx2.conf:/root/clearml.conf:ro \
  -e CLEARML_PROJECT=e2e_calib/splat_ad \
  -e CLEARML_TASK_NAME="$NAME" \
  -w /workspace/neurad-studio \
  splatad-v100:neurad \
  bash -lc "
set -euo pipefail
# 0) make output dir inside container (root can mkdir under $OUT_PARENT
#    even if it is root-owned on the host).
mkdir -p /out_parent/\$EXP_NAME
# 0b) clearml is not in splatad-v100:neurad; install without deps so the
#     pre-baked numpy/pandas/torch stack stays binary-compatible.
pip install --quiet --no-deps --no-input clearml >/dev/null 2>&1 || pip install --no-deps clearml
# clearml's runtime deps that are NOT already in the image:
pip install --quiet --no-deps --no-input attrs furl jsonschema pathlib2 \
    psutil pyjwt pyparsing python-dateutil orderedmultidict requests \
    six urllib3 pillow >/dev/null 2>&1 || true
# 1) inject WovenDataParserConfig into dataparser_configs.dataparsers
python /host_woven/_register_woven_dataparser.py
# 2) clearml auto-bind via tensorboard
cat > /tmp/wrap_clearml.py <<'PY'
import os, sys
from clearml import Task
Task.init(project_name=os.environ['CLEARML_PROJECT'],
          task_name=os.environ['CLEARML_TASK_NAME'],
          auto_connect_frameworks={'tensorboard': True, 'pytorch': False})
print('[clearml] init', flush=True)
sys.argv = ['ns-train'] + sys.argv[1:]
import nerfstudio.scripts.train as _t
_t.entrypoint()
PY
# 3) train
python /tmp/wrap_clearml.py splatad \
  --output-dir /out_parent/\$EXP_NAME --experiment-name '$NAME' \
  --max-num-iterations $MAX_STEPS \
  --vis tensorboard \
  woven-data --data /seq --vehicle '$VEHICLE'
"
