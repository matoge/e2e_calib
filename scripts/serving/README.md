# Calibration server (`scripts/serving/`)

REST endpoint that takes one frame (image + LiDAR + declared K & T_cw) and
returns the per-cam calibration correction the model + BA solver infer.
Pipeline reuses the canonical `scripts.ba.ba_multicam_corr` code (no fork).

```
client ──POST /infer──▶ calib_server.py ──▶ pipeline.py ──▶ infer_tiles + solve_dofs
                            │                  │
                            │                  └── DOF_JAC library
                            │
                            └── model_loader.py ──┬── local ckpt
                                                  └── ClearML Model Registry
```

## Quick start

### 1. Local checkpoint (dev)

```bash
export CALIB_MODEL_SOURCE=local
export CALIB_MODEL_CKPT=$PWD/experiments/ps_20260513_pixonly/best_model.pt
export CALIB_MODEL_USE_INTENSITY=0           # match training config
export CALIB_HOST=0.0.0.0  CALIB_PORT=8080
python -m scripts.serving.calib_server
```

### 2. ClearML model registry (prod)

```bash
# training side — publish your best_model.pt with tags
python -m scripts.serving.tag_model \
    --ckpt experiments/ps_20260513_pixonly/best_model.pt \
    --name calib_best \
    --tag latest --tag prod \
    --comment 'PS front pixel-only, val_nll 2.24'

# server side — auto-pulls + auto-reloads
export CALIB_MODEL_SOURCE=clearml
export CALIB_MODEL_PROJECT=e2e_calib/calib
export CALIB_MODEL_NAME=calib_best
export CALIB_MODEL_TAG=prod
export CALIB_RELOAD_INTERVAL_SEC=60
python -m scripts.serving.calib_server
```

The server polls ClearML every `CALIB_RELOAD_INTERVAL_SEC` and atomically
swaps the pipeline when a newer model under `CALIB_MODEL_TAG` appears.
Mid-flight requests finish on the old model; subsequent requests see the
new version. Manual swap also available via `POST /reload`.

## Endpoints

| method | path     | purpose |
|--------|----------|---------|
| POST   | /infer   | run inference on one frame; see [schemas.py](schemas.py) |
| GET    | /healthz | liveness + active model_version |
| POST   | /reload  | force-pull and rebuild the pipeline |

### Request body (POST /infer)

```json
{
  "image_b64": "<base64 JPEG/PNG>",
  "pts_world": [x0,y0,z0, x1,y1,z1, ...],
  "intensity": [i0,i1,...],          // optional, required iff model.use_intensity
  "K": [fx,0,cx, 0,fy,cy, 0,0,1],
  "T_cw": [r11,r12,r13,tx, ...],     // row-major 4x4 world→cam
  "dof": ["omega_x","omega_y","tx","ty","df_common"]  // optional override
}
```

### Response

```json
{
  "correction": {
    "omega_x_deg": -0.31, "omega_y_deg": +0.22,
    "tx_m": -0.024, "ty_m": +0.118,
    "df_common_px": 5.3, ...
  },
  "n_pts_pooled": 8421,
  "n_tiles": 36,
  "elapsed_ms": 42.7,
  "model_version": "e2e_calib/calib/calib_best#abc123..."
}
```

## Versioning model

The intended hand-off:

1. **train** → produces `best_model.pt` inside `experiments/<name>/`.
2. **tag**   → `scripts/serving/tag_model.py --tag latest [--tag prod]`
              registers the file as a versioned `Model` in ClearML's
              registry under `(project, name)`.
3. **deploy** → server in ClearML mode polls + auto-swaps. To roll back,
   re-tag a previous Model id with `prod` (use `--demote` to strip the
   tag from the current one cleanly).

Production discipline (we don't get atomic guarantees from the registry):

- Exactly one model carries `tag=prod` at a time. Use `--demote` when
  promoting. The server's poller is read-mostly so a brief overlap is
  fine — the *latest* matching model wins.
- `tag=canary` for shadow deploys (run a second server instance with
  `CALIB_MODEL_TAG=canary`).
- `tag=latest` is `prod` plus any newer just-tagged candidates — useful
  for staging environments.

## TensorRT / optimization hooks (future)

`pipeline.py` currently runs the canonical PyTorch path with bf16
autocast on CUDA. Two known speedups, neither blocking:

1. **TensorRT engine** for the backbone + cross-attn. The custom
   MSDeformAttn op needs a TRT plugin (mmcv provides one; vendor +
   register in `pipeline.py`). Drop a prebuilt `engine.plan` at
   `$CALIB_TRT_ENGINE` and gate the build on its presence; PyTorch path
   stays as fallback for any version mismatch.

2. **Top-K pivot selection** post-inference. Current BA pools every
   valid pivot; selecting the top-K by model confidence (1 / √(σuσv))
   drops 5-10× of the per-pt H/b accumulator work with negligible
   accuracy loss (per session notes: top-100/tile already deterministic
   in `infer_tiles` after a recent patch). Wire as a `--top-k N` per-
   request override in `schemas.CalibRequest.ba`.

## Operational notes

- **Pre/post-processing**: image decode (TurboJPEG ~5ms) + projection
  (1ms) + tile slicing + the 5-DoF solve (numpy 5x5 → microseconds).
  Total wall is dominated by the model forward; batching multiple
  requests at the queue level would amortize CUDA launch overhead.
- **Worker count**: keep uvicorn at `--workers 1`. The GPU model is one
  per process; scale horizontally with container replicas + a load
  balancer, not threads.
- **Logging**: optional. Add `clearml.Logger` calls in `infer()` if you
  want per-request scalars (latency, n_pts) on the dashboard; we kept
  the hot path free of network calls by default.
