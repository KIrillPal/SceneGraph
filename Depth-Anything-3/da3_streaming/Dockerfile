FROM pytorch/pytorch:2.5.1-cuda12.4-cudnn9-devel

ENV DEBIAN_FRONTEND=noninteractive \
    PIP_NO_CACHE_DIR=1 \
    PYTHONUNBUFFERED=1 \
    TORCH_HOME=/workspace/.cache/torch \
    CUDA_HOME=/usr/local/cuda

WORKDIR /workspace

RUN apt-get update && apt-get install -y --no-install-recommends \
    ca-certificates \
    curl \
    git \
    libgl1 \
    libglib2.0-0 \
    libgomp1 \
 && rm -rf /var/lib/apt/lists/*

COPY pyproject.toml README.md requirements.txt /workspace/
COPY src /workspace/src
COPY da3_streaming/requirements.txt /workspace/da3_streaming/requirements.txt

RUN python -m pip install --upgrade pip

RUN python -m pip install --index-url https://download.pytorch.org/whl/cu124 \
    torch==2.5.1 torchvision==0.20.1 torchaudio==2.5.1 xformers==0.0.28.post3

RUN grep -Ev '^(torch|torchvision|torchaudio|xformers)([<>=!~].*)?$' /workspace/requirements.txt > /tmp/requirements_no_torch.txt

RUN python -m pip install -r /tmp/requirements_no_torch.txt

RUN python -m pip install -e . --no-deps

RUN python -m pip install -e ".[app]" --no-deps

RUN grep -v '^faiss-gpu$' /workspace/da3_streaming/requirements.txt > /tmp/da3_streaming_requirements.txt

RUN python -m pip install -r /tmp/da3_streaming_requirements.txt

RUN python -m pip install faiss-cpu

CMD ["bash"]
