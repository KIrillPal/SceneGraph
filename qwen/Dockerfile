FROM vllm/vllm-openai:latest

ARG TRANSFORMERS_REF=main

RUN python3 -m pip install --no-cache-dir --upgrade pip && \
    python3 -m pip install --no-cache-dir --upgrade \
      "git+https://github.com/huggingface/transformers.git@${TRANSFORMERS_REF}"

ENV PYTHONUNBUFFERED=1 \
    HF_HOME=/root/.cache/huggingface

WORKDIR /workspace

EXPOSE 8000
