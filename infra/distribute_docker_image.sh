#!/usr/bin/env bash
# Distribute a locally-built Docker image to all DGX nodes via Lustre.
#
# Workflow:
#   1. Build `e2e-calib-train:local` on one node (typically dgx2).
#   2. This script `docker save` → /mnt/fsx/tmp/hfunaya/images/<tag>.tar
#      then ssh-es into the other nodes and `docker load` from that tar.
#
# No docker registry, no scp; Lustre is the transport (all 4 DGX see /mnt/fsx).
#
# Usage:
#   # from the node that already has the image loaded:
#   ./infra/distribute_docker_image.sh                         # defaults to e2e-calib-train:local → all 4
#   ./infra/distribute_docker_image.sh --image foo:bar
#   ./infra/distribute_docker_image.sh --hosts "dgx1 dgx3 dgx4"    # skip dgx2
#   ./infra/distribute_docker_image.sh --skip-save                  # reuse existing tar
#
# Assumptions:
#   - Source host = the host running this script (runs `docker save` locally)
#   - Target hosts are reachable via `ssh <host>` (ssh alias resolved)
#   - All hosts mount /mnt/fsx at the same path (Lustre)
#   - docker is available on all hosts and the user is in the `docker` group
set -euo pipefail

IMAGE="e2e-calib-train:local"
HOSTS="dgx1 dgx3 dgx4"
TAR_DIR="/mnt/fsx/tmp/hfunaya/images"
SKIP_SAVE=0

usage() {
  cat <<EOF
Usage: $0 [options]
  --image TAG       docker image tag to distribute (default: $IMAGE)
  --hosts "A B C"   space-separated list of target hosts (default: "$HOSTS")
  --tar-dir PATH    directory under /mnt/fsx for the tar (default: $TAR_DIR)
  --skip-save       reuse existing tar, only run `docker load` on targets
  -h, --help        this help
EOF
  exit 2
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --image)     IMAGE="$2"; shift 2 ;;
    --hosts)     HOSTS="$2"; shift 2 ;;
    --tar-dir)   TAR_DIR="$2"; shift 2 ;;
    --skip-save) SKIP_SAVE=1; shift 1 ;;
    -h|--help)   usage ;;
    *) echo "unknown arg: $1" >&2; usage ;;
  esac
done

TAR_NAME="$(echo "$IMAGE" | tr '/:' '__').tar"
TAR_PATH="$TAR_DIR/$TAR_NAME"

mkdir -p "$TAR_DIR"

if [[ "$SKIP_SAVE" -eq 0 ]]; then
  echo "[save] $IMAGE → $TAR_PATH"
  if ! docker image inspect "$IMAGE" >/dev/null 2>&1; then
    echo "[err] image not found locally: $IMAGE" >&2
    echo "      build it first, e.g.:" >&2
    echo "      docker build -f infra/Dockerfile.train -t $IMAGE ." >&2
    exit 1
  fi
  docker save "$IMAGE" -o "$TAR_PATH"
  ls -lh "$TAR_PATH"
else
  echo "[skip-save] reusing $TAR_PATH"
  [[ -f "$TAR_PATH" ]] || { echo "[err] tar missing: $TAR_PATH" >&2; exit 1; }
fi

for H in $HOSTS; do
  echo "[load on $H] docker load -i $TAR_PATH"
  # Some shells may not propagate PATH; use full ssh command line.
  if ! ssh -o BatchMode=yes -o ConnectTimeout=10 "$H" "docker load -i '$TAR_PATH'" </dev/null; then
    echo "[warn] load failed on $H (maybe docker group / ssh issue); continue" >&2
    continue
  fi
  ssh -o BatchMode=yes "$H" "docker image inspect '$IMAGE' --format '{{.Id}} {{.RepoTags}}'" </dev/null || true
done

echo
echo "Done. Verify with:"
echo "  for h in $HOSTS; do ssh \$h 'docker images $IMAGE'; done"
