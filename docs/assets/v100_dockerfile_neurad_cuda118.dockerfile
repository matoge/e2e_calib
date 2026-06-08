# V100 (sm_70) build of neurad-studio + SplatAD, based on the upstream
# georghess/neurad-studio Dockerfile but with two surgical fixes:
#   1. CUDA_ARCHITECTURES default = "70" (V100 only).
#   2. The upstream `awk '$0 > 70 ...'` strict-greater filter is changed to
#      `$0 >= 70` so sm_70 actually reaches TORCH_CUDA_ARCH_LIST when
#      gsplat/splatad/neurad-studio compile.
#
# Base: nvidia/cuda:11.8.0-devel-ubuntu22.04 — no /opt/hpcx, no libucc.so.1
# undefined-symbol bug. PyTorch 2.0.1+cu118 wheels are installed via pip.
ARG CUDA_VERSION=11.8.0
ARG OS_VERSION=22.04
FROM nvidia/cuda:${CUDA_VERSION}-devel-ubuntu${OS_VERSION}
ARG CUDA_VERSION
ARG OS_VERSION

LABEL org.opencontainers.image.licenses="Apache License 2.0"
LABEL org.opencontainers.image.base.name="docker.io/library/nvidia/cuda:${CUDA_VERSION}-devel-ubuntu${OS_VERSION}"

# V100 sm_70 only — speeds up build and is what DGX-2 actually has.
ARG CUDA_ARCHITECTURES=70

ENV DEBIAN_FRONTEND=noninteractive
ENV TZ=Asia/Tokyo
ENV CUDA_HOME="/usr/local/cuda"

RUN apt-get update && \
    apt-get install -y --no-install-recommends \
    build-essential cmake curl wget ffmpeg git vim-tiny \
    libatlas-base-dev libhdf5-dev libprotobuf-dev protobuf-compiler \
    libboost-filesystem-dev libboost-graph-dev libboost-program-options-dev \
    libboost-system-dev libboost-test-dev libcgal-dev libeigen3-dev \
    libflann-dev libfreeimage-dev libgflags-dev libglew-dev libmetis-dev \
    libqt5opengl5-dev libsuitesparse-dev \
    python-is-python3 python3.10-dev python3-pip \
    qtbase5-dev \
    && rm -rf /var/lib/apt/lists/*

RUN python3.10 -m pip install --no-cache-dir --upgrade pip "setuptools<70.0" \
    pathtools promise pybind11
SHELL ["/bin/bash", "-c"]
RUN python3.10 -m pip install --no-cache-dir \
    torch==2.0.1+cu118 torchvision==0.15.2+cu118 \
    --extra-index-url https://download.pytorch.org/whl/cu118
RUN TCNN_CUDA_ARCHITECTURES=${CUDA_ARCHITECTURES} python3.10 -m pip install --no-cache-dir \
    git+https://github.com/NVlabs/tiny-cuda-nn.git#subdirectory=bindings/torch

# waymo dataset reader (used by neurad-studio)
RUN python3.10 -m pip install --no-cache-dir waymo-open-dataset-tf-2-11-0==1.6.1
RUN python3.10 -m pip install --no-cache-dir tzdata

WORKDIR /workspace

RUN git clone https://github.com/georghess/neurad-studio.git
WORKDIR /workspace/neurad-studio
# Patch: $0 > 70 → $0 >= 70 so sm_70 (V100) actually reaches the list.
RUN export TORCH_CUDA_ARCH_LIST="$(echo "$CUDA_ARCHITECTURES" | tr ';' '\n' | awk '$0 >= 70 {print substr($0,1,1)"."substr($0,2)}' | tr '\n' ' ' | sed 's/ $//')" && \
    echo "TORCH_CUDA_ARCH_LIST=$TORCH_CUDA_ARCH_LIST" && \
    python3.10 -m pip install -e .[dev]
WORKDIR /workspace

RUN git clone --recurse-submodules https://github.com/carlinds/splatad.git
WORKDIR /workspace/splatad
RUN export TORCH_CUDA_ARCH_LIST="$(echo "$CUDA_ARCHITECTURES" | tr ';' '\n' | awk '$0 >= 70 {print substr($0,1,1)"."substr($0,2)}' | tr '\n' ' ' | sed 's/ $//')" && \
    echo "TORCH_CUDA_ARCH_LIST=$TORCH_CUDA_ARCH_LIST" && \
    BUILD_NO_CUDA=1 python3.10 -m pip install -e .[dev]

# Sanity at build time so we catch sm_70 / libucc / driver issues early.
RUN python3.10 -c "import torch; print('torch', torch.__version__, 'cuda', torch.version.cuda); print('archs', torch.cuda.get_arch_list())" && \
    python3.10 -c "import gsplat; print('gsplat', gsplat.__version__)" && \
    python3.10 -c "import nerfstudio; print('nerfstudio OK')"

RUN python3.10 -c "import viser; viser.ViserServer()" || true
RUN ns-install-cli --mode install || true

CMD ["/bin/bash", "-l"]
