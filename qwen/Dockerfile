FROM pytorch/pytorch:2.8.0-cuda12.8-cudnn9-runtime

ENV DEBIAN_FRONTEND=noninteractive \
    PIP_NO_CACHE_DIR=1 \
    PYTHONUNBUFFERED=1 \
    HF_HOME=/root/.cache/huggingface \
    PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

RUN apt-get update && \
    apt-get install -y --no-install-recommends git curl && \
    rm -rf /var/lib/apt/lists/*

RUN pip install --upgrade pip && \
    pip install --index-url https://download.pytorch.org/whl/cu128 torchvision && \
    pip install "transformers[serving] @ git+https://github.com/huggingface/transformers.git@main" accelerate pillow sentencepiece openai bitsandbytes

WORKDIR /workspace

EXPOSE 8000
