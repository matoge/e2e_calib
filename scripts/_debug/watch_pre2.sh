#!/bin/bash
for i in $(seq 1 80); do
  sleep 30
  d2_pre=$(docker logs km_wv_wm_dgx2_n2 2>&1 | grep -c 'preflight OK')
  d1_pre=$(ssh dgx1 "docker logs km_wv_wm_dgx1_n4 2>&1 | grep -c 'preflight OK'")
  d2_ep=$(docker logs km_wv_wm_dgx2_n2 2>&1 | grep -oE '\[ *[0-9]+/50\]' | tail -1)
  d1_ep=$(ssh dgx1 "docker logs km_wv_wm_dgx1_n4 2>&1 | grep -oE '\[ *[0-9]+/50\]' | tail -1")
  d2_dead=$(docker ps -a --filter name=km_wv_wm_dgx2_n2 --format '{{.Status}}' | grep -c Exited)
  d1_dead=$(ssh dgx1 "docker ps -a --filter name=km_wv_wm_dgx1_n4 --format '{{.Status}}' | grep -c Exited")
  echo "[$((i*30))s] DGX2:pre=$d2_pre ep=$d2_ep dead=$d2_dead | DGX1:pre=$d1_pre ep=$d1_ep dead=$d1_dead"
  if [ "$d2_dead" -ge 1 ] || [ "$d1_dead" -ge 1 ]; then
    echo "DEAD"
    break
  fi
  if [ "$d2_pre" -ge 1 ] && [ "$d1_pre" -ge 1 ]; then echo "BOTH preflight PASS"; break; fi
done
