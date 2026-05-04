#!/usr/bin/env bash
# infra/build_images.sh — build SM70 (V100/Volta) and SM120 (Blackwell/RTX 5080)
# variants of e2e-calib-train from the same Dockerfile.
#
# usage:
#   ./infra/build_images.sh sm70     # build only SM70 (V100, dgx2/sakurai2)
#   ./infra/build_images.sh sm120    # build only SM120 (RTX 5080, yokohama1)
#   ./infra/build_images.sh both     # both (default)
set -e
cd "$(dirname "$0")/.."

TARGET="${1:-both}"

# nvcr.io/nvidia/pytorch tag chosen per arch:
#   24.02-py3   torch 2.2  CUDA 12.3  → SM70 OK, SM120 NO
#   25.06-py3   torch 2.8  CUDA 12.9  → SM70 OK, SM120 OK (use this for Blackwell)
declare -A BASE
BASE[sm70]="nvcr.io/nvidia/pytorch:24.02-py3"
BASE[sm120]="nvcr.io/nvidia/pytorch:25.06-py3"

build_one() {
  local arch="$1"
  local base="${BASE[$arch]}"
  echo "==> building e2e-calib-train:${arch}  (base: ${base})"
  docker build \
    --build-arg "BASE_IMAGE=${base}" \
    -f infra/Dockerfile.train \
    -t "e2e-calib-train:${arch}" \
    .
}

case "$TARGET" in
  sm70)  build_one sm70 ;;
  sm120) build_one sm120 ;;
  both)  build_one sm70 ; build_one sm120 ;;
  *)     echo "usage: $0 {sm70|sm120|both}" ; exit 2 ;;
esac

echo
echo "built:"
docker images e2e-calib-train --format 'table {{.Repository}}:{{.Tag}}\t{{.Size}}\t{{.CreatedSince}}'
