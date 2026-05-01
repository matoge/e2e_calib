#!/bin/bash
# Sigma sweep on PD 192² cache (v9-equivalent recipe). Find cliff between
# linear scaling and divergence. Each run ~30 min. Total ~2.5 h.
set -e
cd "$(dirname "$0")/../.."

# wait for any running v25
while pgrep -f train_ps_v25 >/dev/null; do sleep 30; done

for SIG in 1.0 2.0 2.5 3.0 4.0; do
    NAME="ps_v26_sigma${SIG/./_}_sweep"
    [ -d "experiments/$NAME" ] && rm -rf "experiments/$NAME"

    cat > /tmp/run_sweep_$SIG.py <<PY
import sys, pathlib; sys.path.insert(0, str(pathlib.Path("$PWD")))
import scripts.training.train_ps_v25_panda_sigma05 as v25
v25.CFG.update(name='${NAME}', sigma_ypr=${SIG}, sigma_t=$(python -c "print(0.2 * ${SIG} / 0.5)"))
v25.main()
PY
    echo "=== σ=${SIG} → $NAME ==="
    python -u /tmp/run_sweep_$SIG.py >/tmp/$NAME.log 2>&1
    grep "Best val NLL" experiments/$NAME/train.log | tail -1
done
echo "sweep done"
