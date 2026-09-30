#!/bin/bash
# Container entrypoint for the auto-calibration serving image.
#
# Forwards env vars to auto_calib_server.py CLI flags. Every knob has a
# sensible default so `docker run <image>` on any host with a GPU works
# out-of-the-box:
#
#   PORT              (default 8501)
#   HOST              (default 0.0.0.0)
#   CKPT_DIR          (default /app/weights/kmwv_s3_ba40_512r256_0901_1344 — baked in)
#   HOOD_MASK_ROOT    (default /app/hood_masks — baked in if provided, else disabled)
#   DEVICE            (default cuda if visible, else cpu)
set -eu

PORT="${PORT:-8501}"
HOST="${HOST:-0.0.0.0}"
CKPT_DIR="${CKPT_DIR:-/app/weights/kmwv_s3_ba40_512r256_0901_1344}"
HOOD_MASK_ROOT="${HOOD_MASK_ROOT:-/app/hood_masks}"

# Auto-detect device unless explicitly overridden.
if [ -z "${DEVICE:-}" ]; then
    if python -c "import torch, sys; sys.exit(0 if torch.cuda.is_available() else 1)" \
             > /dev/null 2>&1; then
        DEVICE=cuda
    else
        DEVICE=cpu
    fi
fi
echo "[entrypoint] PORT=${PORT} HOST=${HOST} DEVICE=${DEVICE}"
echo "[entrypoint] CKPT_DIR=${CKPT_DIR}"
echo "[entrypoint] HOOD_MASK_ROOT=${HOOD_MASK_ROOT}"

exec python /app/scripts/serving/auto_calib_server.py \
    --port "${PORT}" \
    --host "${HOST}" \
    --device "${DEVICE}" \
    --ckpt "${CKPT_DIR}" \
    --hood-mask-root "${HOOD_MASK_ROOT}"
