#!/usr/bin/env bash
# infra/entrypoint.sh — dual-mode entrypoint for e2e-calib-train image.
#
# Mode A: ClearML agent daemon (CLEARML_AGENT_DAEMON=1)
#   - Required: CLEARML_AGENT_QUEUE
#   - Required mount: $HOME/clearml.conf → /root/clearml.conf:ro
#   - Runs `clearml-agent daemon --foreground --queue $QUEUE`
#
# Mode B: pass-through (default)
#   - exec "$@" verbatim. Used for direct training runs, shells, etc.
set -e

if [[ "${CLEARML_AGENT_DAEMON:-0}" == "1" ]]; then
  if [[ -z "${CLEARML_AGENT_QUEUE:-}" ]]; then
    echo "[entrypoint] CLEARML_AGENT_DAEMON=1 but CLEARML_AGENT_QUEUE is empty" >&2
    exit 2
  fi
  if [[ ! -f /root/clearml.conf ]]; then
    echo "[entrypoint] expected /root/clearml.conf — mount your host clearml.conf:" >&2
    echo "             docker run -v \$HOME/clearml.conf:/root/clearml.conf:ro ..." >&2
    exit 2
  fi
  echo "[entrypoint] starting clearml-agent on queue=${CLEARML_AGENT_QUEUE}"
  exec clearml-agent daemon --foreground --queue "${CLEARML_AGENT_QUEUE}"
fi

exec "$@"
