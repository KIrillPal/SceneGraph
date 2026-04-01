FROM nvidia/cuda:11.8.0-cudnn8-devel-ubuntu22.04

ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    PIP_DEFAULT_TIMEOUT=300 \
    PIP_INDEX_URL=https://pypi.tuna.tsinghua.edu.cn/simple/ \
    PIP_TRUSTED_HOST=pypi.tuna.tsinghua.edu.cn \
    NVIDIA_VISIBLE_DEVICES=all \
    NVIDIA_DRIVER_CAPABILITIES=compute,utility \
    TORCH_CUDA_ARCH_LIST="7.0;8.0" \
    HF_HOME=/root/.cache/huggingface \
    VIRTUAL_ENV=/opt/venv \
    PATH=/opt/venv/bin:$PATH

WORKDIR /workspace/sam3

RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    ca-certificates \
    ffmpeg \
    git \
    libgl1 \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender1 \
    python3 \
    python3-pip \
    python3-venv \
    && rm -rf /var/lib/apt/lists/*

RUN update-alternatives --install /usr/bin/python python /usr/bin/python3 1 && \
    python -m venv "$VIRTUAL_ENV"

COPY pyproject.toml README.md LICENSE MANIFEST.in ./
COPY sam3 ./sam3

RUN python -m pip install --upgrade pip setuptools wheel

RUN python -m pip install \
    --retries 20 \
    --timeout 300 \
    --index-url https://download.pytorch.org/whl/cu118 \
    --trusted-host download.pytorch.org \
    torch==2.7.0 \
    torchvision==0.22.0 \
    torchaudio==2.7.0

RUN python -m pip install \
    --retries 10 \
    --timeout 300 \
    --index-url "$PIP_INDEX_URL" \
    --trusted-host "$PIP_TRUSTED_HOST" \
    decord

RUN python -m pip install \
    --retries 10 \
    --timeout 300 \
    --index-url "$PIP_INDEX_URL" \
    --trusted-host "$PIP_TRUSTED_HOST" \
    opencv-python-headless

RUN python -m pip install \
    --retries 10 \
    --timeout 300 \
    --index-url "$PIP_INDEX_URL" \
    --trusted-host "$PIP_TRUSTED_HOST" \
    scikit-image

RUN python -m pip install \
    --retries 10 \
    --timeout 300 \
    --index-url "$PIP_INDEX_URL" \
    --trusted-host "$PIP_TRUSTED_HOST" \
    .

RUN python -m pip install \
    --retries 10 \
    --timeout 300 \
    --index-url "$PIP_INDEX_URL" \
    --trusted-host "$PIP_TRUSTED_HOST" \
    --force-reinstall "setuptools<82"

RUN python -m pip install \
    --retries 10 \
    --timeout 300 \
    --index-url "$PIP_INDEX_URL" \
    --trusted-host "$PIP_TRUSTED_HOST" \
    einops

RUN python -m pip install \
    --retries 10 \
    --timeout 300 \
    --index-url "$PIP_INDEX_URL" \
    --trusted-host "$PIP_TRUSTED_HOST" \
    pycocotools

RUN python -m pip install \
    --retries 10 \
    --timeout 300 \
    --index-url "$PIP_INDEX_URL" \
    --trusted-host "$PIP_TRUSTED_HOST" \
    psutil

RUN apt-get update && apt-get install -y --no-install-recommends \
    python3-dev \
    && rm -rf /var/lib/apt/lists/*

CMD ["bash"]
