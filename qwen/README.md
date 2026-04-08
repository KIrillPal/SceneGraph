# Qwen Docker Setup

This folder contains a simple Docker image for running supported `Qwen` models with the Hugging Face `transformers` library.

It does not auto-start anything. You enter the container in interactive mode and run commands manually.

Important:

- on smaller GPUs like a `RTX 4070`, use quantization instead of relying on automatic offload
- this image is generic: choose the Qwen model you want when running `transformers serve`

## Files

- `Dockerfile`: installs dependencies only
- `build.sh`: builds the image
- `start.sh`: opens an interactive shell on GPU `0`

## Build

```bash
./build.sh
```

If you built the image before the PyTorch update, rebuild it so the container no longer uses the old torch stack:

```bash
docker rmi qwen-transformers || true
./build.sh
```

## Start

```bash
./start.sh
```

This starts a container with:

- only GPU `0`
- port `8000` mapped to the host
- Hugging Face cache mounted to `/root/.cache/huggingface`
- the repo mounted to `/workspace`

## Environment Variables

You can override these when starting the container:

```bash
IMAGE_NAME=qwen-transformers
CONTAINER_NAME=qwen-transformers
PORT=8000
HF_CACHE_DIR=$HOME/.cache/huggingface
```

Example:

```bash
PORT=8010 CONTAINER_NAME=qwen35 ./start.sh
```

## Run The Server Manually

Inside the container, use the model name as a positional argument.

General form:

```bash
transformers serve <MODEL_ID> \
  --host 0.0.0.0 \
  --port 8000 \
  --dtype <DTYPE> \
  --attn-implementation sdpa \
  --continuous-batching
```

Examples of `MODEL_ID` you could use:

- `Qwen/Qwen3.5-9B`
- `Qwen/Qwen3.5-27B`
- `Qwen/Qwen3.5-35B-A3B`
- `Qwen/Qwen2.5-VL-7B-Instruct`

## Qwen3.5-9B Examples

These concrete examples are kept here because they are useful reference points for different cards.

For `A100`:

```bash
transformers serve Qwen/Qwen3.5-9B \
  --host 0.0.0.0 \
  --port 8000 \
  --dtype bfloat16 \
  --attn-implementation sdpa \
  --continuous-batching
```

For `V100` or GPUs without `bfloat16`:

```bash
transformers serve Qwen/Qwen3.5-9B \
  --host 0.0.0.0 \
  --port 8000 \
  --dtype float16 \
  --attn-implementation sdpa \
  --continuous-batching
```

For a smaller GPU such as `RTX 4070`, try 4-bit quantization:

```bash
transformers serve Qwen/Qwen3.5-9B \
  --host 0.0.0.0 \
  --port 8000 \
  --dtype float16 \
  --attn-implementation sdpa \
  --quantization bnb-4bit \
  --continuous-batching
```

The error you hit happened because this `transformers serve` version does not support `--force-model`. The model must be passed directly after `serve`.

The later error:

```text
The current `device_map` had weights offloaded to the disk ... Please provide an `offload_folder`
```

means the model did try to load on your current GPU, but it did not have enough VRAM in the chosen precision, so Transformers attempted an automatic offload path that this model does not support cleanly in this CLI flow.

If you hit a `set_submodule` error while using `bnb-4bit`, that points to an older PyTorch stack inside the container. Rebuild the image so it uses the newer Docker base in this folder.

In practice:

- `A100`: run `bfloat16`
- `V100`: run `float16`
- `RTX 4070` laptop: try `bnb-4bit`
- if `bnb-4bit` still fails, use a smaller model

For larger Qwen models, the same dtype guidance applies, but VRAM requirements go up significantly.

## Multi-Image Request Example

```bash
curl http://localhost:8000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "Qwen/Qwen3.5-9B",
    "messages": [
      {
        "role": "user",
        "content": [
          {"type": "text", "text": "Compare these images."},
          {"type": "image_url", "image_url": {"url": "https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/diffusers/astronaut.jpg"}},
          {"type": "image_url", "image_url": {"url": "https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/transformers/tasks/car.jpg"}}
        ]
      }
    ],
    "max_tokens": 512
  }'
```

## Notes

- `start.sh` always uses `--gpus "device=0"`
- `build.sh` and `start.sh` are model-agnostic; the model is chosen when you run `transformers serve`
- the `flash-linear-attention` / `causal-conv1d` message is a performance warning, not the main failure
- for large multi-image prompts, you may still need to reduce image size or output length to avoid OOM
