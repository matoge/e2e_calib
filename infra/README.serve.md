# e2e_calib serving image — build, run, and deploy to Amazon ECS

Self-contained Docker image that runs the LiDAR–camera auto-calibration API
(CalibNet2 + 40-tile fused Bundle Adjustment + inverse-variance frame pool)
behind FastAPI on port **8501**.

The image ships with weights baked in
(`experiments/kmwv_s3_ba40_512r256_0901_1344/best_model.pt`, 4.5 MB) so the
running container has no external state.  Deployment target is ECS
(Fargate for CPU workloads, EC2 with `g4dn.xlarge` or better for GPU).

For architectural background see:
- [docs/2026-08-31_nuscenes-calibration.md](../docs/2026-08-31_nuscenes-calibration.md) — training recipe
- [docs/2026-09-30_cam6_heldout/](../docs/assets/2026-09-30_cam6_heldout/) — held-out cam6 evaluation
- [docs/2026-09-30_unilab001_apply/](../docs/assets/2026-09-30_unilab001_apply/) — production apply-time viz

## API surface

| Endpoint | Purpose |
|---|---|
| `GET  /health` | Liveness for ECS health-checks. Returns `{status:"ok", model:..., device:...}` once weights load. |
| `POST /calibrate/generic/frame` | **Recommended.** Single frame — client sends image bytes + LiDAR + intrinsics + extrinsics inline. No filesystem dependency. |
| `POST /calibrate/generic/sequence` | Multi-frame joint pool — same payload style, `frames` is a list. Returns pooled δ, inverse-variance σ, χ²-gate k dispersion. |
| `POST /calibrate/frame` | Legacy: reads `seq_dir` from local filesystem (WovenSequence layout). Kept for LOOM's existing integration. |
| `POST /calibrate/sequence` | Legacy sequence variant. |
| `GET  /docs`, `/redoc`, `/openapi.json` | Auto-generated Swagger UI + machine-readable schema. |

**Client contract for `/calibrate/generic/frame`** (see `GET /openapi.json`
for exact schema):

```json
{
  "image_b64": "<base64 JPEG/PNG bytes>",
  "points_xyzi_b64": "<base64 float32 (N,4) [x,y,z,intensity] in LiDAR frame>",
  "camera": {
    "resolution": [W, H],
    "fc": [fx, fy],
    "cc": [cx, cy],
    "dist_kb4": [k1, k2, k3, k4],
    "T_cam_lidar": [[..],[..],[..],[..]],
    "setting_rot_rad": [roll, pitch, yaw],
    "setting_mp_m":  [x, y, z]
  },
  "hood_polygon_uv": [[u,v], ...],
  "frame_id": "..."
}
```

Everything except `image_b64`, `points_xyzi_b64`, and the two intrinsic
fields (`fc`, `cc`) is optional — omit `setting_rot_rad`/`setting_mp_m` and
the response drops the `deltas`/`sd` fields but still gives you the raw
camera-FRD δ and 6×6 information matrix.

**Client contract for `/calibrate/generic/sequence`**: same payload
wrapped as `{frames: [frame_1, frame_2, ...], use_gate3: true, k_correct:
true}`.

## Build

Two flavours, one Dockerfile, base image selected at build time:

```bash
# GPU (CUDA 12.1, expects nvidia-container-runtime on the host / ECS instance)
docker build -f infra/Dockerfile.serve \
    --build-arg BASE_IMAGE=pytorch/pytorch:2.4.1-cuda12.1-cudnn9-runtime \
    -t e2e-calib-serve:gpu .

# CPU-only (Fargate-friendly; ~2 GB image vs ~6 GB for GPU)
docker build -f infra/Dockerfile.serve \
    --build-arg BASE_IMAGE=pytorch/pytorch:2.4.1-cpu \
    -t e2e-calib-serve:cpu .
```

On CPU a single-frame call takes ~5 s (vs ~150 ms on a T4). A 10-frame
sequence pool on CPU is ~50 s. If your workflow needs interactive latency
(LOOM slider drag → auto-calibrate loop) use the GPU flavour.

## Run locally (smoke test)

```bash
docker run --rm --gpus all -p 8501:8501 e2e-calib-serve:gpu

# In another terminal:
curl -s http://localhost:8501/health | jq .
# → {"status":"ok","model":".../kmwv_s3_ba40_512r256_0901_1344","device":"cuda", ...}

python scripts/_debug/smoke_generic_endpoint.py   # from a checkout of this repo
# → "[smoke] ✅ PASS: generic path matches reference path."
```

## Configuration (environment variables)

| Var | Default | Purpose |
|---|---|---|
| `PORT` | `8501` | Uvicorn bind port. |
| `HOST` | `0.0.0.0` | Uvicorn bind host. |
| `CKPT_DIR` | `/app/weights/kmwv_s3_ba40_512r256_0901_1344` | Which baked-in checkpoint to load. Ship additional experiments by extending `COPY experiments/...` in the Dockerfile. |
| `HOOD_MASK_ROOT` | `/app/hood_masks` | Directory of per-car hood polygons for the legacy WovenSequence endpoints. Ignored by `/calibrate/generic/*` (client sends `hood_polygon_uv` inline). |
| `DEVICE` | auto | `cuda` if a GPU is visible via `torch.cuda.is_available()`, else `cpu`. Override to force CPU inference on a GPU host. |

## Deploy to Amazon ECS

The image is stateless, has a HEALTHCHECK, drains SIGTERM cleanly via
`tini`, and runs as an unprivileged UID (1000). It ticks all the ECS
best-practice boxes.

### 1. Push to ECR

```bash
export AWS_REGION=ap-northeast-1                    # adjust
export ACCOUNT_ID=$(aws sts get-caller-identity --query Account --output text)
export ECR_REPO=e2e-calib-serve

aws ecr create-repository --repository-name ${ECR_REPO} --region ${AWS_REGION}
aws ecr get-login-password --region ${AWS_REGION} \
    | docker login --username AWS --password-stdin \
                    ${ACCOUNT_ID}.dkr.ecr.${AWS_REGION}.amazonaws.com

docker tag e2e-calib-serve:gpu \
    ${ACCOUNT_ID}.dkr.ecr.${AWS_REGION}.amazonaws.com/${ECR_REPO}:gpu-$(date +%Y%m%d)
docker push \
    ${ACCOUNT_ID}.dkr.ecr.${AWS_REGION}.amazonaws.com/${ECR_REPO}:gpu-$(date +%Y%m%d)
```

### 2. Task definition (GPU on EC2 launch type)

Minimal skeleton — plug in the pushed image URI and log group name:

```json
{
  "family": "e2e-calib-serve",
  "requiresCompatibilities": ["EC2"],
  "networkMode": "awsvpc",
  "cpu": "4096",
  "memory": "16384",
  "containerDefinitions": [{
    "name": "serve",
    "image": "<account>.dkr.ecr.<region>.amazonaws.com/e2e-calib-serve:gpu-YYYYMMDD",
    "essential": true,
    "portMappings": [{"containerPort": 8501, "protocol": "tcp"}],
    "resourceRequirements": [
      {"type": "GPU", "value": "1"}
    ],
    "healthCheck": {
      "command": ["CMD-SHELL",
                  "curl -sf http://localhost:8501/health | grep -q '\"status\":\"ok\"' || exit 1"],
      "interval": 30, "timeout": 5, "retries": 3, "startPeriod": 60
    },
    "environment": [
      {"name": "DEVICE", "value": "cuda"}
    ],
    "logConfiguration": {
      "logDriver": "awslogs",
      "options": {
        "awslogs-group": "/ecs/e2e-calib-serve",
        "awslogs-region": "<region>",
        "awslogs-stream-prefix": "serve"
      }
    }
  }]
}
```

### 3. Task definition (CPU on Fargate)

Drop the `resourceRequirements`, switch `requiresCompatibilities` to
`FARGATE`, use the CPU image tag. Set `DEVICE=cpu` explicitly.

### 4. Service + ALB

Standard pattern — put the task behind an internal ALB, target group on
port 8501, health check path `/health`, threshold status code 200,
healthy-threshold 2 / unhealthy-threshold 3. Model load takes 10-20 s so
set `HealthCheckGracePeriodSeconds` on the service to ≥ 60 s to avoid
the initial task getting killed before it's ready.

## Updating the checkpoint

The checkpoint dir is one of the last COPY layers, so replacing it is a
cheap rebuild. Two options:

1. **Bake a new default** — swap the path in `Dockerfile.serve` COPY and
   `entrypoint.serve.sh`, rebuild, push a new tag.
2. **Multi-checkpoint image** — add more `COPY experiments/<name>/...`
   lines, then choose at run-time via `CKPT_DIR` env var.

## Rebuilding after code changes

Only need to re-COPY the changed source layer — Docker caches the `pip
install` layer. Typical cycle:

```bash
docker build -f infra/Dockerfile.serve \
    --build-arg BASE_IMAGE=pytorch/pytorch:2.4.1-cuda12.1-cudnn9-runtime \
    -t e2e-calib-serve:gpu .
```

Takes ~30 s if only `scripts/` changed, ~5 min for a full rebuild.

## Known limitations

- **Base ECR tags aren't publicly cached.** Both `pytorch/pytorch:2.4.1-*`
  base images pull from Docker Hub. If your ECS network can't reach Docker
  Hub, mirror them to ECR yourself and adjust `BASE_IMAGE`.
- **CalibNet2 is trained on Woven/kamikado fisheye rigs.** Pinhole cameras
  work (leave `dist_kb4` at zeros) but haven't been quantitatively
  evaluated against WovenSequence baselines. Verify on your own held-out
  set before shipping.
- **The legacy `/calibrate/{frame,sequence}` endpoints assume the
  WovenSequence directory layout.** They exist for LOOM's transition
  period only. New integrations should go through
  `/calibrate/generic/{frame,sequence}`.
