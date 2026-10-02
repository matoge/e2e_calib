# e2e-calib-roma:np2 — RoMaV2 + DINOv3 inference image.
# Base = training image. Upgrade torch to 2.5.1+cu121 (DINOv3 needs
# torch.amp.custom_fwd introduced in 2.4, and cu121 wheels have kernels
# for DGX-2 compute caps). Install romav2 and its runtime deps.
FROM e2e-calib-train:np2

RUN pip install --no-cache-dir --upgrade \
      torch==2.5.1 torchvision==0.20.1 \
      --index-url https://download.pytorch.org/whl/cu121

RUN pip install --no-cache-dir --no-deps romav2==2.0.1

# romav2 runtime deps (torch pinned above, don't let romav2 drag a newer one)
RUN pip install --no-cache-dir \
      einops>=0.8.1 \
      pillow>=10.0.0 \
      rich>=13.0.0 \
      tqdm>=4.67.1

# fused-local-corr: optional speed-up. --no-deps keeps our torch 2.5.1+cu121
# pin intact (newer fused-local-corr wheels drag torch 2.11+cu13 → no DGX2
# kernel image). Pure-Python path works if the compile hook is missing.
RUN pip install --no-cache-dir --no-deps fused-local-corr || \
    echo "[info] fused-local-corr skipped"

# Keep the base image's working cv2 (no typing/__init__.py). Only add scipy.
RUN pip install --no-cache-dir scipy
