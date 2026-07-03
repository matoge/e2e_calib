#!/usr/bin/env bash
# All-in-one preprocessing for one woven_sequence clip:
#   1. extract tar.gz (skip if already extracted)
#   2. lidar_deskew (sensor → vls128_rear_axle)  [skip if exists]
#   3. SAM3 dynamic mask on tss4_fcm jpgs (fisheye full-res)
#   4. bake masks (SAM3 + dashboard polygon, fisheye-native, no undistort)
#
# Output: SEQ_OUT_DIR with vls128_rear_axle/, masks_baked/
# Optional: PandaSet variant disabled by default (use woven_parser.py fisheye-native)
#
# Usage:
#   preprocess_woven_seq.sh <tar.gz path> [SEQ_OUT_ROOT] [VEHICLE]
#
# Example:
#   preprocess_woven_seq.sh \
#     /mnt/ecp-perception/woven_sequence/tss4_calib_raw_02/20230608_220033/sequence=248_20230608_220033_1686263612052-1686263631966.tar.gz
#
set -euo pipefail
TAR=${1:?tar.gz path required}
SEQ_ROOT=${2:-/raid/home/hfunaya/woven_sequence_extracted_raw02}
VEHICLE=${3:-248}

REPO=/home/hfunaya/git/e2e_calib
PY=/home/hfunaya/.pyenv/versions/3.10.4/bin/python

[[ -f "$TAR" ]] || { echo "tar not found: $TAR"; exit 1; }
SEQ_NAME=$(basename "$TAR" .tar.gz)
SEQ_DIR="$SEQ_ROOT/$SEQ_NAME"
mkdir -p "$SEQ_ROOT"

# ---- 1. extract ---------------------------------------------------------
if [[ ! -d "$SEQ_DIR/tss4_fcm" ]]; then
  echo "[1/4] extracting → $SEQ_DIR"
  tar -xzf "$TAR" -C "$SEQ_ROOT"
else
  echo "[1/4] already extracted: $SEQ_DIR"
fi
N_FCM=$(ls "$SEQ_DIR/tss4_fcm" | wc -l)
N_VLS=$(ls "$SEQ_DIR/vls128"   | wc -l)
echo "      n_fcm=$N_FCM n_vls=$N_VLS"

# ---- 2. lidar deskew ----------------------------------------------------
DESKEW_OUT="$SEQ_DIR/vls128_rear_axle"
if [[ ! -d "$DESKEW_OUT" ]] || [[ -z $(ls "$DESKEW_OUT" 2>/dev/null) ]]; then
  echo "[2/4] running lidar_deskew"
  $PY "$REPO/scripts/webui_kb_fit/lidar_deskew.py" \
      --raw-seq-dir "$SEQ_DIR" --out-seq-dir "$SEQ_DIR"
  # script writes to vls128_rear_axle_deskew/ -- rename if needed
  if [[ -d "$SEQ_DIR/vls128_rear_axle_deskew" ]] && [[ ! -d "$DESKEW_OUT" ]]; then
    mv "$SEQ_DIR/vls128_rear_axle_deskew" "$DESKEW_OUT"
  fi
else
  echo "[2/4] vls128_rear_axle already populated"
fi

# ---- 3. SAM3 dynamic mask (host pyenv 3.10.4 with transformers 5.6.0.dev0) ----
SAM3_OUT="$SEQ_DIR/_sam3"
if [[ ! -d "$SAM3_OUT/inst" ]] || [[ -z $(ls "$SAM3_OUT/inst" 2>/dev/null) ]]; then
  echo "[3/4] running SAM3 (host pyenv)"
  CUDA_VISIBLE_DEVICES=${CUDA:-4} $PY "$REPO/scripts/webui_kb_fit/sam3_dynamic_mask.py" \
      --seq-dir "$SEQ_DIR" \
      --out-dir "$SAM3_OUT"
else
  echo "[3/4] SAM3 inst already populated"
fi

# ---- 4. bake masks (SAM3 + dashboard, fisheye-native) ------------------
BAKED_OUT="$SEQ_DIR/_masks_baked"
if [[ ! -d "$BAKED_OUT" ]] || [[ -z $(ls "$BAKED_OUT" 2>/dev/null) ]]; then
  echo "[4/4] baking masks (fisheye, no undistort)"
  $PY "$REPO/scripts/splatad_kb/bake_masks.py" \
      --seq-dir "$SEQ_DIR" \
      --out-dir "$BAKED_OUT" \
      --inst-dir "$SAM3_OUT/inst" \
      --vehicle "$VEHICLE"
else
  echo "[4/4] _masks_baked already populated"
fi

echo
echo "=== preprocessing done ==="
echo "  SEQ_DIR     = $SEQ_DIR"
echo "  vls128_rear_axle = $DESKEW_OUT  (deskewed LiDAR)"
echo "  _masks_baked = $BAKED_OUT  (final masks, fisheye full-res)"
echo
echo "next: woven_parser.py (fisheye-native) for GS training"
