# e2e_calib runtime image — matches yokohama1's Python 3.13 + torch 2.11+cu130 stack.
# Used by ClearML agent in --docker mode so we sidestep pip/conda version-resolution
# whack-a-mole when the host (e.g. sakurai2) doesn't have matching Python/CUDA.
#
# Build:
#   docker build -t e2e-calib:py313-cu130 -f docker/Dockerfile .
# Save + transfer (no registry):
#   docker save e2e-calib:py313-cu130 | zstd | ssh sakurai2 'zstd -d | docker load'
# Use:
#   clearml-agent daemon --queue sakurai2 --gpus 0 --docker e2e-calib:py313-cu130

FROM nvidia/cuda:13.0.0-cudnn-runtime-ubuntu24.04

ENV DEBIAN_FRONTEND=noninteractive \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1

# System python 3.12 ships with Ubuntu 24.04 — add deadsnakes for 3.13 to match host.
RUN apt-get update && apt-get install -y --no-install-recommends \
        software-properties-common ca-certificates curl git wget \
        build-essential \
    && add-apt-repository -y ppa:deadsnakes/ppa \
    && apt-get update \
    && apt-get install -y --no-install-recommends \
        python3.13 python3.13-dev python3.13-venv \
        libgl1 libglib2.0-0 libturbojpeg0-dev \
    && rm -rf /var/lib/apt/lists/*

# pip via ensurepip (the deadsnakes packages don't ship pip).
# Pre-install cryptography + cffi for Python 3.13 so clearml-agent (which
# bootstrap-installs itself inside the container) doesn't fall back to the apt
# `cryptography` built for Python 3.12 → `_cffi_backend` import crash.
RUN python3.13 -m ensurepip --upgrade \
    && python3.13 -m pip install --upgrade 'pip>=23,<26' setuptools wheel \
    && python3.13 -m pip install --upgrade --ignore-installed cffi cryptography pyjwt \
    && ln -sf /usr/bin/python3.13 /usr/local/bin/python \
    && ln -sf /usr/bin/python3.13 /usr/local/bin/python3

# Heavy GPU deps first (better layer cache).
RUN python -m pip install --extra-index-url https://download.pytorch.org/whl/cu130 \
        'torch==2.11.0'

# App deps from snapshot. nuscenes-devkit is installed separately --no-deps
# because its `numpy<2.0` pin conflicts with the rest (yokohama runs the same
# numpy 2.x combo without issues at runtime).
COPY docker/requirements.txt /tmp/requirements.txt
RUN python -m pip install -r /tmp/requirements.txt
RUN python -m pip install --no-deps nuscenes-devkit==1.2.0

WORKDIR /workspace
