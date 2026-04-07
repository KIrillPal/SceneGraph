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

RUN python -m pip install xformers "torch>=2" torchvision

RUN python -m pip install -e .

RUN python -m pip install -e ".[app]"

RUN grep -v '^faiss-gpu$' /workspace/da3_streaming/requirements.txt > /tmp/da3_streaming_requirements.txt

RUN python -m pip install -r /tmp/da3_streaming_requirements.txt

RUN python -m pip install faiss-cpu

CMD ["bash"]
