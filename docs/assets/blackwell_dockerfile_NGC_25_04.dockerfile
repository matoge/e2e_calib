# NGC PyTorch 25.04: CUDA 12.8 + torch 2.7 with Blackwell sm_120 prebuilt + cudafe++ that handles new arches
FROM nvcr.io/nvidia/pytorch:25.04-py3

ARG CUDA_ARCHITECTURES=120
ENV TORCH_CUDA_ARCH_LIST="12.0"
ENV DEBIAN_FRONTEND=noninteractive
ENV TZ=Europe/Berlin

# Verify base sanity
RUN python -c "import torch; print('torch:', torch.__version__, 'cuda:', torch.version.cuda, 'archs:', torch.cuda.get_arch_list())"

# Minimal apt deps for COLMAP-style builds (neurad-studio expects these)
RUN apt-get update && apt-get install -y --no-install-recommends \
    cmake build-essential git ffmpeg vim-tiny \
    libatlas-base-dev libhdf5-dev libprotobuf-dev protobuf-compiler \
    libboost-filesystem-dev libboost-system-dev libcgal-dev libeigen3-dev \
    libflann-dev libfreeimage-dev libgflags-dev libglew-dev libmetis-dev \
    libsuitesparse-dev python-is-python3 \
    && rm -rf /var/lib/apt/lists/*

# NGC pip constraint file pins setuptools==78, neurad-studio pyproject needs <70.
# Bypass constraint by unsetting PIP_CONSTRAINT.
ENV PIP_CONSTRAINT=
RUN pip install --no-cache-dir "setuptools==69.5.1" --force-reinstall --ignore-installed && \
    pip install --no-cache-dir wheel ninja pybind11 tzdata

# tinycudann: try sm_120 with MAX_JOBS=2 to avoid OOM on cicc
RUN MAX_JOBS=2 TCNN_CUDA_ARCHITECTURES=120 pip install --no-cache-dir --no-build-isolation \
    git+https://github.com/NVlabs/tiny-cuda-nn.git#subdirectory=bindings/torch || \
    echo "tinycudann install failed - SplatAD doesn't need it"

WORKDIR /workspace

RUN git clone https://github.com/georghess/neurad-studio.git
WORKDIR /workspace/neurad-studio
RUN pip install --no-build-isolation -e .[dev] || pip install --no-deps -e .

WORKDIR /workspace
RUN git clone --recurse-submodules https://github.com/carlinds/splatad.git
WORKDIR /workspace/splatad
# Loosen PyYAML==6.0 pin (legacy, NGC has PyYAML 6.0.2 which works fine)
RUN sed -i 's/"pyyaml==6\.0"/"pyyaml>=6.0"/g; s/pyyaml==6\.0/pyyaml>=6.0/g' setup.py pyproject.toml 2>/dev/null || true
RUN BUILD_NO_CUDA=1 pip install --no-build-isolation -e .[dev] || \
    BUILD_NO_CUDA=1 pip install --no-build-isolation --no-deps -e .

# Explicit install of all neurad-studio runtime deps (from-git custom forks first)
RUN pip install --no-cache-dir "git+https://github.com/scaleapi/pandaset-devkit.git#egg=pandaset&subdirectory=python"
RUN pip install --no-cache-dir "git+https://github.com/atonderski/viser.git"
RUN pip install --no-cache-dir tyro mediapy plotly jaxtyping \
    nerfacc rich tensorboard pyngrok requests numpy-quaternion gdown \
    timm pymeshlab numba zod nuscenes-devkit av av2 fire pathos \
    torch-fidelity opencv-python opencv-python-headless \
    splines rawpy pyliblzfse imageio comet-ml wandb gitpython \
    dataclass-wizard descartes pyquaternion python-engineio python-socketio \
    fastapi uvicorn websockets simple-websocket bidict watchfiles dash plotly \
    pycocotools open3d trimesh torchmetrics

# Fix upath bytecode mismatch on Python 3.12 in NGC container — reinstall universal_pathlib
# (the upath module comes from universal-pathlib package)
RUN pip install --force-reinstall --no-cache-dir --no-deps universal_pathlib && \
    find /usr/local/lib/python3.12/dist-packages/upath -name "*.pyc" -delete 2>/dev/null || true && \
    python -c "import upath; print('upath OK:', upath.__file__)"

RUN python -c "import viser; viser.ViserServer()" || true
RUN ns-install-cli --mode install || true

CMD /bin/bash -l
