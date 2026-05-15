# LiDAR intensity quirks across the V3-i tile caches

The 4-channel `(u, v, d, intensity)` layout is shared across datasets, but the **raw intensity scale is dataset-dependent**. We normalize at load time in `datasets/pandaset_full.py:_load_inst()` so the model sees a comparable [0, 1] band on every batch.

## Observed raw ranges

| dataset | sample of `inst['intensity']` (5 random tiles) | shape of distribution | normalization |
|---|---|---|---|
| **Waymo** (`waymo_v3_tiled_i`) | min=0.001, max=0.367, mean=0.040<br>min=0.001, max=**16.250**, mean=0.068 (rare spike) | mostly < 0.4, very rare outliers ≥ 16 | `clip(x, 0, 1)` |
| **kamikado** (`kamikado_v3_tiled`) | min=0, max=96, mean=5.3 | uint8 / 128 quant (ip664 sensor) | `clip(x / 128, 0, 1)` |
| **woven** (`woven_v3_tile`) | min=0, max=63, p99=46 | same ip664 sensor family | `clip(x / 128, 0, 1)` |
| **PandaSet / NuScenes / others** | typical 0..255 | uint8 | `clip(x / 255, 0, 1)` |

## Why this matters

Without normalization, the rare Waymo `intensity = 16.25` spike enters

* `dist_uvd[..., 4]` → `point_mlp` (`Linear(4, 64)` in `models/model_depth.py`) and
* `bucket_uvd[..., 3]` → `kv_proj` (`Linear(4, 2*d_local)` in `FrustumLocalEncoder`)

at fp16. After 2-3 layers of attention/FFN the activations overflow → softmax → **`loss=+nan` from the very first step** (we hit this on `km_wv_wm_dgx2_8gpu_200ep_os4_n2`).

## Code

`datasets/pandaset_full.py:_load_inst()`:

```python
intensity = np.asarray(intensity, dtype=np.float32)
if intensity.size:
    name = self.cache_dir.name.lower()
    if 'waymo' in name:
        intensity = np.clip(intensity, 0.0, 1.0)            # spec already [0,1] + outlier clip
    elif 'kamikado' in name or 'woven' in name:
        intensity = np.clip(intensity / 128.0, 0.0, 1.0)    # ip664 / 128 quant
    else:
        intensity = np.clip(intensity / 255.0, 0.0, 1.0)    # uint8 default
```

The dispatch is by `cache_dir.name`, so dropping a new cache directory whose name doesn't contain `waymo`/`kamikado`/`woven` falls through to the uint8 default. Add a branch when you onboard a new sensor.

## Verification

After the fix, `dist_uvd[:, 4]` and `bucket_uvd[..., 3]` stay in `[0, ~0.8]` across all three caches (verified 2026-05-16 on `kamikado_v3_tiled`, `woven_v3_tile`, `waymo_v3_tiled_i`). Train loss returns to a finite value from step 1.

## When to bypass

If you need to A/B test "no normalization" or feed raw counts, the cleanest path is to override per-instance after construction:

```python
ds = PandaSetCalibDatasetFull(...)
ds._normalize_intensity = False  # not implemented yet — file an issue first
```

For now there is no flag — edit the helper if you really need raw values.
