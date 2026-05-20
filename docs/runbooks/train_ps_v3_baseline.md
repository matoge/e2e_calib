# Runbook: Production training (`train_ps_v3_ddp.py`) on DGX2

This is the *exact* recipe that was working on 2026-05-18 (the day before this
file was written) for the `km_wv_wm_*` baselines. **It does not use
`clearml-task`** — `clearml-task` insists on a `requirements.txt` that this
repo does not have, so the recipe instead launches `accelerate launch`
directly inside the prebuilt training container and reports to ClearML from
inside the trainer (via `--clearml`).

If you ever find yourself trying to "reproduce yesterday's training" and you
are reaching for `clearml-task`, **stop** — read this file. The recipe is
just `docker run -d ... bash -c 'accelerate launch ...'`.

## Hosts and environment

- This DGX2 is **the** ClearML server (`172.16.200.185`, ports 8082/3/4).
  Conf file is at `/home/hfunaya/clearml-dgx2.conf`. Bind it into the
  container at `/root/clearml.conf`.
- Training image: `e2e-calib-train:np2` (10.8 GB; built 2026-05-16,
  `numpy>=2`, torch, accelerate 0.34, clearml 2.1.7, lmdb).
  - `e2e-calib-train:sm70` is the older 22.3 GB ancestor — do not use.
- Caches on this host:
  - `/home/hfunaya/cache/kamikado_v3_tiled` (LMDB, with `data.lmdb`)
  - `/home/hfunaya/cache/woven_v3_tile`
  - `/home/hfunaya/clearml/data/cache/waymo_v3_tiled_i`  ← **note: not under `~/cache/`**

## Recipe

```bash
docker stop km_wv_wm 2>/dev/null
docker rm   km_wv_wm 2>/dev/null

docker run -d --name km_wv_wm \
  --gpus '"device=8,9,10,11,12,13,14,15"' --shm-size=128g \
  --ulimit nofile=1048576:1048576 \
  -v /home/hfunaya/git/e2e_calib:/workspace \
  -v /home/hfunaya/cache/kamikado_v3_tiled:/cache/kamikado_v3_tiled:ro \
  -v /home/hfunaya/cache/woven_v3_tile:/cache/woven_v3_tile:ro \
  -v /home/hfunaya/clearml/data/cache/waymo_v3_tiled_i:/cache/waymo_v3_tiled_i:ro \
  -v /home/hfunaya/clearml-dgx2.conf:/root/clearml.conf:ro \
  -e CLEARML_CONFIG_FILE=/root/clearml.conf \
  e2e-calib-train:np2 \
  bash -c 'cd /workspace && accelerate launch \
      --num_processes=8 --mixed_precision=fp16 \
      scripts/training/train_ps_v3_ddp.py \
        --name km_wv_wm_dgx2_n4_img128_8gpu \
        --cache /cache/kamikado_v3_tiled,/cache/woven_v3_tile,/cache/waymo_v3_tiled_i \
        --epochs 200 --batch-size 128 --workers 8 --oversample 4 \
        --n-layers 4 --val-size 800 \
        --min-crop-px 256 --max-crop-px 512 \
        --convnext --deform-mode sl \
        --clearml --why baseline_n4_img128_8gpu_dgx2'
```

## Known bugs fixed on 2026-05-20 to make this work today

- `datasets/pandaset_full.py` had two issues that meant val splits could
  produce zero pivot candidates and then **recurse forever** in
  `__getitem__`'s `self[random_idx]` fallback (RecursionError, ~978 deep).
  - `center_band > 0` mixed parent-image `K[1, 2]` (cy) with tile-local `IH`
    on tile caches, so the band sat off-frame and the AND of pivot masks
    was empty. Subtract `tile_v0` from cy in the pivot-band code.
  - The "max_tries failed" fallback was `return self[random.randint(...)]`,
    which recurses. Replaced by an iterative re-roll that tries up to 1024
    distinct indices, then raises a clear `RuntimeError`.

## Pre-flight (the things that have bitten us)

1. **GPU 0–3 may be held by `gallant_snyder` etc.** — pin to 8–15 with
   `--gpus '"device=8,9,10,11,12,13,14,15"'`. `--gpus all` will collide.
2. **No `requirements.txt` ⇒ no `clearml-task`.** Don't try.
3. **System python's `clearml-task` is broken** (OpenSSL ABI mismatch). If
   for some reason you must invoke it, use `PYENV_VERSION=3.10.4`. But you
   shouldn't — see (2).
4. **Bare-metal `python train_ps_v3.py` ENOMEMs** with `workers>0` in this
   shell because `vm.max_map_count` is the kernel default and we don't have
   sudo. Inside the container with `--shm-size=128g
   --ulimit nofile=1048576:1048576` it's fine. **Always use docker.**
5. The trainer reports to ClearML via `--clearml`. The container needs
   `/root/clearml.conf` mounted ro.
6. `--num_processes` must equal the number of GPUs you pinned with
   `--gpus`. Mismatch ⇒ NCCL hang.

## Verifying it's actually training

`docker logs -f km_wv_wm` should within ~90 s show the cache-load lines
followed by `step 100 loss=... sps=...`. If you don't see SPS, it's not
training — re-check shm and gpu pinning.

## Reference numbers (so you know what "working" looks like)

- 8-GPU, n4, img128, bs=128, os=4, ConvNeXt, fp16:
  global SPS ≈ 5000 (per-GPU ~625), ep1 val ≈ 3.2.
- Single-GPU bare-metal smoke (workers=0, bs=16): ~132 sps. **Slow path
  — only useful as a "does the loop run" check, not for convergence.**
