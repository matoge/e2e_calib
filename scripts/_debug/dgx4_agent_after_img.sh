#!/bin/bash
set -euo pipefail
echo "[$(date)] waiting for image transfer to dgx4..."
while ps -ef | grep -E "[d]ocker save.*dgx4" | grep -q .; do
  sleep 30
done
sleep 5
ssh dgx4 "docker images | grep np2 | head -1"
echo "[$(date)] starting dgx4 agent..."
ssh dgx4 "CLEARML_CONFIG_FILE=/home/hfunaya/clearml-dgx2.conf nohup ~/venv_clearml/bin/clearml-agent daemon --queue dgx4 --docker e2e-calib-train:np2 --gpus all --detached > /home/hfunaya/clearml_agent_dgx4.log 2>&1"
sleep 8
ssh dgx4 "ps -ef | grep [c]learml-agent | head"
echo "[$(date)] done."
