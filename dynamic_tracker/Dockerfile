FROM nvidia/cuda:11.8.0-cudnn8-devel-ubuntu22.04

ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    PIP_DEFAULT_TIMEOUT=300 \
    NVIDIA_VISIBLE_DEVICES=all \
    NVIDIA_DRIVER_CAPABILITIES=compute,utility \
    HF_HOME=/root/.cache/huggingface \
    COTRACKER_CHECKPOINT=/root/.cache/cotracker/scaled_online.pth \
    VIRTUAL_ENV=/opt/venv \
    PATH=/opt/venv/bin:$PATH \
    PYTHONPATH=/workspace/dynamic_tracker

WORKDIR /workspace

RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    ca-certificates \
    ffmpeg \
    git \
    libgl1 \
    libglib2.0-0 \
    libgomp1 \
    libsm6 \
    libxext6 \
    libxrender1 \
    python3 \
    python3-dev \
    python3-pip \
    python3-venv \
    wget \
    && rm -rf /var/lib/apt/lists/*

RUN update-alternatives --install /usr/bin/python python /usr/bin/python3 1 && \
    python -m venv "$VIRTUAL_ENV" && \
    python -m pip install --upgrade pip setuptools wheel

RUN python -m pip install \
    --retries 20 \
    --timeout 300 \
    --index-url https://download.pytorch.org/whl/cu118 \
    --trusted-host download.pytorch.org \
    torch==2.7.0 \
    torchvision==0.22.0

RUN python -m pip install \
    --retries 10 \
    --timeout 300 \
    numpy==1.26.4 \
    opencv-python-headless==4.11.0.86 \
    open3d==0.19.0 \
    scipy==1.15.3 \
    scikit-image==0.25.2 \
    scikit-learn==1.7.2 \
    sentence-transformers \
    transformers==4.57.6 \
    timm==1.0.22 \
    tqdm==4.67.1 \
    pillow==11.3.0 \
    matplotlib==3.10.7 \
    distinctipy==1.3.4

RUN python -m pip install \
    --retries 10 \
    --timeout 300 \
    "git+https://github.com/facebookresearch/co-tracker.git"

CMD ["bash"]
