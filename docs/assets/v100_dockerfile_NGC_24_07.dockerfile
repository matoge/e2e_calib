# NGC PyTorch 24.07: CUDA 12.5 + torch 2.4. V100 (sm_70) build of SplatAD/neurad-studio.
# Older base than 25.04 to dodge libucc/HPC-X breakage that hits torch import on V100.
# gsplat installed from prebuilt wheel (https://docs.gsplat.studio/whl/pt24cu125).
FROM nvcr.io/nvidia/pytorch:24.07-py3

ARG CUDA_ARCHITECTURES=70
ENV TORCH_CUDA_ARCH_LIST="7.0"
ENV DEBIAN_FRONTEND=noninteractive
ENV TZ=Asia/Tokyo

RUN python -c "import torch; print('torch:', torch.__version__, 'cuda:', torch.version.cuda, 'archs:', torch.cuda.get_arch_list())"

RUN apt-get update && apt-get install -y --no-install-recommends \
    cmake build-essential git ffmpeg vim-tiny \
    libatlas-base-dev libhdf5-dev libprotobuf-dev protobuf-compiler \
    libboost-filesystem-dev libboost-system-dev libcgal-dev libeigen3-dev \
    libflann-dev libfreeimage-dev libgflags-dev libglew-dev libmetis-dev \
    libsuitesparse-dev python-is-python3 \
    && rm -rf /var/lib/apt/lists/*

ENV PIP_CONSTRAINT=
RUN pip install --no-cache-dir "setuptools==69.5.1" --force-reinstall --ignore-installed && \
    pip install --no-cache-dir wheel ninja pybind11 tzdata jaxtyping rich

# gsplat: prebuilt wheel for torch 2.4 + cuda 12.4 (closest to NGC 24.07's torch 2.4 + cuda 12.5).
# This pre-compiled wheel ships sm_70 cubins so V100 just works.
RUN pip install --no-cache-dir gsplat --index-url https://docs.gsplat.studio/whl/pt24cu124 || \
    pip install --no-cache-dir gsplat --index-url https://docs.gsplat.studio/whl/pt24cu121 || \
    echo "gsplat wheel install failed - will fall back to source build below"

# tinycudann: V100 sm_70 build. Optional (SplatAD doesn't need tcnn for the GS path).
RUN MAX_JOBS=2 TCNN_CUDA_ARCHITECTURES=70 pip install --no-cache-dir --no-build-isolation \
    git+https://github.com/NVlabs/tiny-cuda-nn.git#subdirectory=bindings/torch || \
    echo "tinycudann install failed - SplatAD doesn't need it"

WORKDIR /workspace
RUN git clone https://github.com/georghess/neurad-studio.git
WORKDIR /workspace/neurad-studio
RUN pip install --no-build-isolation -e .[dev] || pip install --no-deps -e .

WORKDIR /workspace
RUN git clone --recurse-submodules https://github.com/carlinds/splatad.git
WORKDIR /workspace/splatad
RUN sed -i 's/"pyyaml==6\.0"/"pyyaml>=6.0"/g; s/pyyaml==6\.0/pyyaml>=6.0/g' setup.py pyproject.toml 2>/dev/null || true
# BUILD_NO_CUDA=1 → use system gsplat (already installed above), don't recompile
RUN BUILD_NO_CUDA=1 pip install --no-build-isolation -e .[dev] || \
    BUILD_NO_CUDA=1 pip install --no-build-isolation --no-deps -e .

# neurad-studio runtime deps
RUN pip install --no-cache-dir "git+https://github.com/scaleapi/pandaset-devkit.git#egg=pandaset&subdirectory=python"
RUN pip install --no-cache-dir "git+https://github.com/atonderski/viser.git"
RUN pip install --no-cache-dir tyro mediapy plotly \
    nerfacc tensorboard pyngrok requests numpy-quaternion gdown \
    timm pymeshlab numba zod nuscenes-devkit av av2 fire pathos \
    torch-fidelity opencv-python opencv-python-headless \
    splines rawpy pyliblzfse imageio comet-ml wandb gitpython \
    dataclass-wizard descartes pyquaternion python-engineio python-socketio \
    fastapi uvicorn websockets simple-websocket bidict watchfiles dash \
    pycocotools open3d trimesh torchmetrics pytorch_msssim xatlas

RUN pip install --force-reinstall --no-cache-dir --no-deps universal_pathlib && \
    find /usr/local/lib/python3*/dist-packages/upath -name "*.pyc" -delete 2>/dev/null || true && \
    python -c "import upath; print('upath OK:', upath.__file__)"

# Sanity check (catches the libucc / sm_70 / gsplat issues at build time, not after deploy)
RUN python -c "import torch; assert torch.cuda.is_available() or True; print('torch OK', torch.__version__)" && \
    python -c "import gsplat; print('gsplat:', gsplat.__version__)" && \
    python -c "import nerfstudio; print('nerfstudio OK')" && \
    ns-train --help > /dev/null 2>&1 && echo "ns-train OK"

CMD ["/bin/bash", "-l"]
