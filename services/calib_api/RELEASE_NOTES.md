# Calibration API — Release Notes

## v0.1.0 — 2026-05-22

First versioned release of the self-service calibration page.

**URL**: http://172.16.200.185:8082/calibrate
**Backend**: FastAPI sidecar (`calib-api`) on the `clearml_backend` Docker network, GPU 15 resident.
**Checkpoint**: `experiments/km_wv_wm_dgx2_n4_img128_8gpu_HEAD/best_model.pt` (img_size=128).

### What it does

Push the rig out of alignment with the perturbation sliders (roll/pitch/yaw ±1°, tx/ty/tz ±0.1 m), hit **Run calibration**, and the frozen ~8 MB σ-head emits per-point Fisher info on every sub-tile of one camera frame. One closed-form Gauss-Newton step in original-camera coordinates recovers the rig pose. Output is a 3-panel overlay (GT / perturbed / corrected) plus the residual reproj-px before and after.

### Headline numbers (default fisheye sample, idx=17)

| mode | sub-crops batched | reproj px (pre → post) | wall time |
|---|---:|---:|---:|
| `tile_only` cs=512 | 1 | ~25 → ~3 | < 0.3 s |
| `tile_only` cs=256 | 4 | ~25 → ~2 | < 0.3 s |
| `whole_frame` cs=256 | 117 | **19.4 → 0.82** | 1.4 s |
| `random_val` cs=256 | ~800 | ~25 → ~0.45 | 4 s |

→ a single fisheye frame is enough to drive residuals **sub-pixel**.

### Endpoints

- `GET  /calibrate`            — drag-drop UI
- `GET  /api/default`          — default sample metadata (now returns `api_version` + `git_rev`)
- `GET  /api/frames`           — catalog of fisheye val frames (98 of them) with sibling-tile counts
- `GET  /api/frame/{first_idx}`— per-frame parent-tile PNG + size
- `POST /api/calibrate`        — JSON body, returns δ̂, pre/post reproj px, overlay PNG URL

### UI features

- Perturbation sliders (roll/pitch/yaw, tx/ty/tz)
- Aggregation preset dropdown (`tile_only` / `whole_frame` / `random_val`)
- Frame picker — dropdown of all 98 fisheye frames + 🎲 random button
- Inline API reference at the bottom of the page (curl + Python examples)

### Known limitations

- **Tiled-data only**: requests must reference a `first_idx` from the val-set LMDB. There is no raw-image upload endpoint yet — providing your own image + LiDAR + intrinsics and getting it auto-tiled is **scoped for v0.2**.
- **Overlay shows the anchor tile only**, not the full whole-frame composite — BA already uses every sibling tile, but visualization is single-tile (planned fix).
- `random_val` is paper-headline mode and pulls from the full val split, not from one frame — useful as a benchmark, less useful as a per-frame demo.

### Infrastructure

- Sidecar container `calib-api` on `clearml_backend` (GPU 15, `--gpus '"device=15"'`).
- nginx routes (`services/calib_api/nginx/calibrate.conf`) installed in `clearml-webserver` at `/etc/nginx/default.d/calibrate.conf`.
- Static UI under `services/calib_api/static/`, results cached under `services/calib_api/results/`.
