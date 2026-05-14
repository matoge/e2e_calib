# zod_tss4_20260514_joint_intensity_4ch_os8_100ep_repnearest

Joint ZOD (clean8k intensity) + TSS4 training, identical to the parallel run
`zod_tss4_20260514_joint_intensity_4ch_os8_100ep` except for one knob:

| | legacy (`cell_center`) | this run (`nearest_cam`) |
|---|---|---|
| Rep selection per occupied cell | point with min uv-distance to the cell's geometric center | point with **min depth (z)** in the cell |
| Stability under small motion | the chosen rep flips foreground↔background when a cell straddles an object edge | foreground point always wins — pole / sign stays the rep even when the cell content shifts a few px |

## Why

In the slider demo, the green X (rep) on a cell containing a roadside pole
was *not* on the pole — it landed on a road point that happened to sit
closer to the cell's center. Under small viewpoint changes the rep would
visibly jump between the pole and the road, which is exactly the kind of
instability BA punishes.

Selecting `argmin(z)` per cell pins the rep to the closest object in that
cell. Since the camera always faces forward, "closest in z" ≈ "closest in
3D" for the surfaces visible in the frame; replacing the `‖xyz‖` minimum
with `z` only saves a sqrt per point (no semantic change for the cases that
matter).

## Implementation

Single flag plumbed end-to-end:

* `datasets/pandaset_full.py` — `PandaSetCalibDatasetFull(... rep_strategy=...)`
  with `cell_center` (legacy default) and `nearest_cam` options. The switch
  is one ternary inside `build_window` that picks `score` for the
  per-cell `lexsort`.
* `scripts/training/train_ps_v3.py` — `--rep-strategy` CLI flag, logged in
  `train.log` and persisted into `config.py`.

Old `cell_center` checkpoints stay 100% valid: the legacy code path is
default-on.

## Baselines (same config except rep_strategy)

| run | rep_strategy | best val_nll |
|---|---|---|
| `zod_tss4_20260514_joint_intensity_4ch_os8_100ep` | cell_center | 3.93 @ ep 5 (in flight when killed) |
| **this run** | nearest_cam | (training) |

Target: beat 3.93 by epoch 5; if comparable, the value of `nearest_cam`
shows up as **demo / inference stability**, not val NLL per se — the cell
rep is no longer position-dependent so the per-tile BA estimate stays
locked when the user nudges sliders or the camera moves.

## CLI

```
python -u scripts/training/train_ps_v3.py \
  --name zod_tss4_20260514_joint_intensity_4ch_os8_100ep_repnearest \
  --cache /mnt/nvme6t/e2e_calib_cache/zod_v3_tiled_clean_i \
          /mnt/nvme6t/e2e_calib_cache/tss4_v3_tiled \
  --epochs 100 --batch-size 256 --workers 16 --val-every 5 --oversample 8 \
  --min-crop-px 128 --max-crop-px 512 --use-intensity \
  --rep-strategy nearest_cam
```
