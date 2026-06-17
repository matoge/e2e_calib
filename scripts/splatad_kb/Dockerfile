# Minimal image for woven KB-fisheye Gaussian-Splatting trainer.
# gsplat 1.5.3 (KB4 fisheye CUDA kernel) + torch 2.4 + cv2 + clearml
FROM pytorch/pytorch:2.4.0-cuda11.8-cudnn9-devel

ENV DEBIAN_FRONTEND=noninteractive
RUN apt-get update && apt-get install -y --no-install-recommends \
        git ffmpeg libsm6 libxext6 libgl1 \
        && rm -rf /var/lib/apt/lists/*

RUN pip install --no-cache-dir \
        gsplat==1.5.3 \
        opencv-python-headless==4.10.0.84 \
        scipy==1.13.1 \
        numpy==1.26.4 \
        pandas==2.2.2 \
        clearml==1.16.4 \
        pyarrow==17.0.0

# Pre-build gsplat CUDA kernels (Volta+, V100=7.0; A100=8.0)
ENV TORCH_CUDA_ARCH_LIST="7.0;7.5;8.0;8.6"
RUN python -c "import gsplat; print(gsplat.__version__)" \
    && python -c "from gsplat.cuda._wrapper import _make_lazy_cuda_func"

WORKDIR /workspace
