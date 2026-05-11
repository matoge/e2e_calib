#!/bin/bash
# Download ZOD Frames core lidar tarballs from Dropbox shared link.
#
# Requirements:
#   - DBX_TOKEN env var (Dropbox short-lived access token, sl. prefix)
#     → get from https://www.dropbox.com/developers/apps (4h validity)
#   - destination ~/zod/ or /mnt/.../zod/ with ~300GB free
#   - jq is NOT required; uses printf for JSON
#
# Usage:
#   export DBX_TOKEN=sl.u.AGc...
#   bash scripts/preprocessing/download_zod_dropbox.sh /mnt/your/zod
#
# Downloads 10 core lidar tarballs (frame ranges 000000-099999, ~37 GB each,
# total ~360 GB). Each shards 10K frames. Parallel 7 connections ≈ 100 MB/s
# aggregate on home gigabit, ≈ 200 MB/s aggregate on 2 Gbps fiber.

set -u
OUT_DIR="${1:-/mnt/nvme6t/zod}"
SHARED_URL="https://www.dropbox.com/scl/fo/q81qqpiqygaeys7mppgoe/ABMW5G9RLSH6wqncsW8zY34/single_frames?rlkey=ocr9n0gq3u083zj8sn1yo1ak6"

if [ -z "${DBX_TOKEN:-}" ]; then
    echo "ERR: set DBX_TOKEN env var (from .bashrc or shell). See header." >&2
    exit 1
fi
mkdir -p "$OUT_DIR"

# Sanity check token
chk=$(curl -sX POST https://api.dropboxapi.com/2/check/user \
    -H "Authorization: Bearer $DBX_TOKEN" \
    -H "Content-Type: application/json" -d '{"query":"hi"}')
if [[ "$chk" != *'"result":"hi"'* ]]; then
    echo "ERR: DBX_TOKEN invalid or expired:" >&2
    echo "  response: $chk" >&2
    exit 2
fi
echo "$(date +%H:%M:%S) token OK"

# DL one tarball
dl_one() {
    local chunk="$1"
    local name="lidar_velodyne_core_${chunk}.tar.gz"
    local out="$OUT_DIR/$name"
    if [ -f "$out" ] && [ $(stat -c%s "$out") -gt 1000000000 ]; then
        echo "$(date +%H:%M:%S) [skip] $name exists"
        return 0
    fi
    echo "$(date +%H:%M:%S) [DL] $name"
    local api_arg
    api_arg=$(printf '{"url":"%s","path":"/%s"}' "$SHARED_URL" "$name")
    local t0=$(date +%s)
    curl --silent --show-error -X POST \
        https://content.dropboxapi.com/2/sharing/get_shared_link_file \
        -H "Authorization: Bearer $DBX_TOKEN" \
        -H "Dropbox-API-Arg: $api_arg" \
        --output "$out.part"
    local sz=$(stat -c%s "$out.part" 2>/dev/null || echo 0)
    if [ "$sz" -lt 1000000000 ]; then
        echo "$(date +%H:%M:%S) [FAIL] $name only $sz bytes:" >&2
        head -c 500 "$out.part" >&2
        rm -f "$out.part"
        return 1
    fi
    mv "$out.part" "$out"
    local t1=$(date +%s)
    local el=$((t1 - t0)); [ "$el" -lt 1 ] && el=1
    echo "$(date +%H:%M:%S) [done] $name $((sz / 1024 / 1024 / 1024)) GB in ${el}s = $((sz / 1024 / 1024 / el)) MB/s"
}
export -f dl_one
export OUT_DIR DBX_TOKEN SHARED_URL

# 10 chunks of 10K frames each = 100K total
CHUNKS="000000_009999 010000_019999 020000_029999 030000_039999 \
        040000_049999 050000_059999 060000_069999 070000_079999 \
        080000_089999 090000_099999"

# 7 parallel max (Dropbox per-link aggregate cap ~1 Gbps)
echo "$(date +%H:%M:%S) starting parallel DL (7 max)..."
echo "$CHUNKS" | tr ' ' '\n' | xargs -P7 -I{} bash -c 'dl_one "$@"' _ {}
echo "$(date +%H:%M:%S) all DLs done"
ls -lh "$OUT_DIR"/lidar_velodyne_core_*.tar.gz
