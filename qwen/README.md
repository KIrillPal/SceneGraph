# Qwen vLLM Setup

This folder contains a small Docker image for serving `Qwen` models with `vLLM`.

The image does not auto-start a model server. `start.sh` opens an interactive shell inside the container, and you run `vllm serve ...` manually.

## Files

- `Dockerfile`: thin wrapper over `vllm/vllm-openai`
- `build.sh`: builds the local image
- `start.sh`: opens an interactive shell on GPU `0`
- `extract_relationships.py`: sends selected frames to the local OpenAI-compatible server

## Build

```bash
./build.sh
```

If you want to rebuild from scratch:

```bash
docker rmi qwen-vllm || true
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
IMAGE_NAME=qwen-vllm
CONTAINER_NAME=qwen-vllm
PORT=8000
HF_CACHE_DIR=$HOME/.cache/huggingface
```

Example:

```bash
PORT=8010 CONTAINER_NAME=qwen27b ./start.sh
```

## Run The Server Manually

Inside the container:

```bash
vllm serve Qwen/Qwen3.5-27B \
  --host 0.0.0.0 \
  --port 8000 \
  --dtype bfloat16 \
  --api-key dummy \
  --generation-config vllm
```

Notes:

- `--generation-config vllm` avoids inheriting model repo generation defaults that can be surprising
- keep the repo mounted at `/workspace` if you want to send image paths directly from `extract_relationships.py`
- if Hugging Face rate limits you, login and pre-download the model to a local path, then serve that local path instead of the HF model id

Examples of model ids you can use:

- `Qwen/Qwen3.5-9B`
- `Qwen/Qwen3.5-27B`
- `Qwen/Qwen3.5-35B-A3B`
- `Qwen/Qwen2.5-VL-7B-Instruct`

## Thinking Control

Qwen 3.x models support thinking and non-thinking mode.

For this repo, the default client request in `extract_relationships.py` does both:

- adds `/no_think` in the first user text block
- sends `chat_template_kwargs: {"enable_thinking": false}`

That requires a backend like `vLLM` that supports `chat_template_kwargs`.

## Structured JSON Output

`extract_relationships.py` also sends:

```json
"response_format": {"type": "json_object"}
```

so the server is asked to return JSON instead of free-form reasoning text.

## Relationship Extraction Script

Run from the repo root on the host machine:

```bash
python qwen/extract_relationships.py \
  --selected-dir data/0/selected_frames \
  --endpoint http://localhost:8000/v1/chat/completions \
  --model Qwen/Qwen3.5-27B \
  --api-key dummy \
  --save-response-json
```

Expected selected-frame input:

- `data/0/selected_frames/frames.json`
- `data/0/selected_frames/unmarked_frames/`

By default it saves:

- raw assistant text to `selected_frames/qwen_relationships_raw.json`
- optional full API response to `selected_frames/qwen_relationships_raw_response.json`

## Direct Multi-Image Request Example

```bash
curl http://localhost:8000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -H "Authorization: Bearer dummy" \
  -d '{
    "model": "Qwen/Qwen3.5-27B",
    "messages": [
      {
        "role": "user",
        "content": [
          {"type": "text", "text": "/no_think\nDescribe this image. Output only JSON."},
          {"type": "image_url", "image_url": {"url": "/workspace/data/0/images/frame_001.jpg"}}
        ]
      }
    ],
    "chat_template_kwargs": {"enable_thinking": false},
    "response_format": {"type": "json_object"},
    "temperature": 0,
    "max_tokens": 1024
  }'
```

## Notes

- `start.sh` always uses `--gpus "device=0"`
- `build.sh` and `start.sh` are model-agnostic; the model is chosen when you run `vllm serve`
- for large multi-image prompts, you may still need to reduce image count or increase `max_tokens`
