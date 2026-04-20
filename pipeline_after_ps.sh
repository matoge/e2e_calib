#!/usr/bin/env bash
set -e
cd /home/hiro/git/e2e_calib

# Wait for PS build to finish (marker line in its log)
echo "[pipeline] waiting for PS build to finish..."
until grep -q "^Saved" logs/build_ps.log 2>/dev/null; do
    sleep 10
done
echo "[pipeline] PS done, starting NS"
python build_all_mc_caches.py ns > logs/build_ns.log 2>&1
echo "[pipeline] NS done, starting WM"
python build_all_mc_caches.py wm > logs/build_wm.log 2>&1
echo "[pipeline] WM done, starting training"
python train_all_v2_mc.py > logs/train_all_v2_mc.log 2>&1
echo "[pipeline] training finished"
