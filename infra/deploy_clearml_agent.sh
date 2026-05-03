#!/usr/bin/env bash
# Deploy clearml-agent (docker-compose mode) to a DGX node.
#
# Usage:
#   ./infra/deploy_clearml_agent.sh dgx3   # → dgx3-gpu queue
#   ./infra/deploy_clearml_agent.sh dgx4   # → dgx4-gpu queue
#
# Prerequisite:
#   - ssh alias `dgx3` / `dgx4` resolves
#   - remote has docker, /mnt/fsx mounted, ~/clearml.conf present
#
# What this does:
#   1. scp  infra/clearml-agent.compose.yml to ~/mcp_hub/ on remote
#   2. sed  in the right queue name + worker id
#   3. docker compose up -d
#   4. wait & show `docker logs`
set -euo pipefail

HOST="${1:?usage: $0 <dgx-host>}"
QUEUE="${HOST}-gpu"            # dgx3 -> dgx3-gpu, etc.

COMPOSE_SRC="$(dirname "$0")/clearml-agent.compose.yml"
[ -f "$COMPOSE_SRC" ] || { echo "missing $COMPOSE_SRC"; exit 1; }

echo "[1/5] scp compose to $HOST:~/mcp_hub/"
ssh "$HOST" "mkdir -p ~/mcp_hub"
scp "$COMPOSE_SRC" "$HOST":~/mcp_hub/docker-compose.clearml-agent.yml

echo "[2/5] patch queue name to $QUEUE + worker id $HOST-gpu"
ssh "$HOST" "sed -i \
  -e 's|CLEARML_WORKER_ID:.*|CLEARML_WORKER_ID: ${HOST}-gpu|' \
  -e 's|CLEARML_AGENT_EXTRA_ARGS:.*|CLEARML_AGENT_EXTRA_ARGS: \"--queue ${QUEUE} --gpus all --foreground\"|' \
  -e 's|container_name:.*|container_name: clearml-agent-${HOST}|' \
  ~/mcp_hub/docker-compose.clearml-agent.yml"

echo "[3/5] verify clearml.conf on $HOST"
ssh "$HOST" 'test -f ~/clearml.conf || { echo "MISSING ~/clearml.conf on '"$HOST"'"; exit 2; }'

echo "[4/5] docker compose up -d"
ssh "$HOST" "cd ~/mcp_hub && docker compose -f docker-compose.clearml-agent.yml up -d"

sleep 5
echo "[5/5] last 20 log lines from clearml-agent-$HOST:"
ssh "$HOST" "docker logs --tail 20 clearml-agent-$HOST"

echo
echo "Done. Verify on https://clearml.budda.site/workers-and-queues"
echo "Submit a test task with:"
echo "  ./infra/submit_clearml_task.sh --queue $QUEUE --name smoke_${HOST} --script scripts/training/train_ps_v3_ddp.py --args ..."
