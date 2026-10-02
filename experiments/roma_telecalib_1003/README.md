# roma_telecalib_1003

Dense feature-matching based TELE/FCM extrinsic calibration (woven_sequence).

Layout
```
roma_telecalib_1003/
  src/
    RoMa/                 # cloned github.com/Parskatt/RoMa (romatch pkg)
    tele_fcm_roma.py      # smoke script (one frame → correspondences)
    weights/              # Roma model weights (auto-dl on first roma_outdoor)
  data/                   # (symlinks / cached test pairs)
  results/                # per-frame R, t_dir, correspondences counts
  logs/
```

Run (smoke)
```bash
# in container with romatch installed
cd /workspace/experiments/roma_telecalib_1003
python src/tele_fcm_roma.py \
    --seq /home/hfunaya/git/loom/backend/assets/woven_sequence/llinking_27/tf_long2/sequence=ip654_1337941440921107425_... \
    --frame 10 \
    --out results/frame0010.json
```

Plan
1. Pick 10 random frames from a CAL OK sequence (fcm+tele both present)
2. For each: RoMa dense → essential matrix → R, t_dir
3. Fix t = rig mechanical baseline (from setting-*.json tele.pos - fcm.pos)
4. Aggregate R across frames via quaternion median → final R_tele_from_fcm
5. Compare to setting-*.json's initial R → delta rotation
