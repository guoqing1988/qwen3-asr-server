<div align="center">

<h1>Qwen3-ASR</h1>
<h3>Ready-to-use Local Speech Recognition API Service</h3>

Speech recognition API service centered on [Qwen3-ASR](https://github.com/QwenLM/Qwen3-ASR), with CUDA vLLM and CPU Rust backends, OpenAI API compatibility, Alibaba Cloud Speech API compatibility, and a Paraformer realtime websocket capability.

[简体中文](./README.md)

---

![Static Badge](https://img.shields.io/badge/Python-3.10+-blue?logo=python)
![Static Badge](https://img.shields.io/badge/Torch-2.10.0-%23EE4C2C?logo=pytorch&logoColor=white)
![Static Badge](https://img.shields.io/badge/CUDA-12.8_default-%2376B900?logo=nvidia&logoColor=white)

</div>

## Features

- **Hybrid Runtime Stack** - Uses auto-selected Qwen3-ASR for offline inference and Paraformer realtime for websocket streaming
- **Multilingual Recognition** - Qwen3-ASR supports 52 languages and dialects with automatic detection (zh/en/ja/ko/...)
- **Speaker Diarization** - Automatic multi-speaker identification using CAM++ model
- **Voiceprint Registration & Naming** - Register voiceprint samples per speaker; matched `speaker_id` values are replaced with display names
- **OpenAI API Compatible** - Supports `/v1/audio/transcriptions` endpoint, works with OpenAI SDK
- **Alibaba Cloud API Compatible** - Supports Alibaba Cloud Speech RESTful API and WebSocket streaming protocol
- **WebSocket Real-time Streaming** - True streaming recognition with incremental partial results
- **GPU Memory Control** - Configurable vLLM memory utilization (`QWEN_GPU_MEMORY_UTILIZATION`)
- **Lazy Model Loading** - Auto-unloads models after idle timeout (default 5 min) to release VRAM; next request triggers automatic lazy reload
- **Smart Far-Field Filtering** - Automatically filters far-field sounds and ambient noise in streaming ASR
- **Intelligent Audio Segmentation** - VAD-based greedy merge algorithm for automatic long audio splitting
- **GPU Batch Processing** - Batch inference support, 2-3x faster than sequential processing
- **Resource-Aware Runtime** - Auto-selects the appropriate Qwen3-ASR model for the current machine

## Acknowledgements

- [Qwen3-ASR](https://github.com/QwenLM/Qwen3-ASR) provides the official model family and multimodal/vLLM usage guidance
- [QwenASR](https://github.com/huanglizhuo/QwenASR) provides the CPU Rust backend vendored by this project
- [Qwen3-ASR Server](https://github.com/Quantatirsk/qwen3-asr) upstream reference implementation of this project

## Quick Deployment

### 1. Local systemd Deployment (Recommended, no Docker)

For Linux servers with NVIDIA GPU:

```bash
# Install uv package manager
curl -LsSf https://astral.sh/uv/install.sh | sh

# Clone and install dependencies
git clone <your-repo> && cd qwen3-asr-server
uv venv --python 3.12
uv sync

# Configure (port, model paths, etc. are all driven by .env, never hard-coded)
cp .env.example .env
# Models live under /data1/models (outside the project directory), e.g.:
#   ASR model        QWEN3_ASR_MODEL_PATH=/data1/models/Confucius4-R2T2
#   Forced aligner   QWEN3_FORCED_ALIGNER_PATH=/data1/models/Qwen3-ForcedAligner-0.6B
#   ModelScope       MODELSCOPE_CACHE=/data1/models/modelscope/hub

# Register and start service (symlinked: editing the repo unit file + daemon-reload is enough)
sudo ln -sfn "$(pwd)/qwen3-asr.service" /etc/systemd/system/qwen3-asr.service
sudo systemctl daemon-reload
sudo systemctl enable --now qwen3-asr
```

Service URLs (the port is driven by `PORT` in `.env`):
- **API Endpoint**: `http://localhost:9101`
- **API Docs**: `http://localhost:9101/docs`

> See [Deployment Guide → Local Deployment](./docs/deployment.md#linux-本地部署systemd无需-docker) for details.

### 2. Docker Deployment

```bash
# Copy and edit configuration
cp .env.example .env
# Edit .env to set API_KEY (optional)

# Start service (GPU version)
docker compose up -d

# Or CPU version
docker compose -f docker-compose-cpu.yml up -d

# Multi-GPU auto mode (one instance per visible GPU)
CUDA_VISIBLE_DEVICES=0,1,2,3 docker compose up -d
```

Service URLs:
- **API Endpoint**: `http://localhost:9101`
- **API Docs**: `http://localhost:9101/docs`

**docker run (alternative):**

```bash
# GPU version
docker run -d --name qwen3-asr \
  --gpus all \
  -p 9101:8000 \
  -e CUDA_VISIBLE_DEVICES=0,1,2,3 \
  -e API_KEY=your_api_key \
  -v ./models/modelscope:/root/.cache/modelscope \
  -v ./models/huggingface:/root/.cache/huggingface \
  -v ./data:/app/data \
  quantatrisk/qwen3-asr:gpu-latest

# CPU version
docker run -d --name qwen3-asr \
  -p 9101:8000 \
  -v ./models/modelscope:/root/.cache/modelscope \
  -v ./models/huggingface:/root/.cache/huggingface \
  -v ./data:/app/data \
  quantatrisk/qwen3-asr:cpu-latest
```

> **Note**: GPU images default to CUDA 12.8/cu128 for Blackwell-capable GPUs.
> Developers can rebuild `Dockerfile.gpu` for CUDA 12.6, CUDA 13.0, or another backend by overriding Docker build args.
> CPU images now support `qwen3-asr-0.6b` via the bundled QwenASR Rust backend. The default CPU image uses a portable Rust target; set `QWENASR_RUST_TARGET_CPU=native` only for self-built, host-specific images.
> On CUDA vLLM and CPU Rust, `word_timestamps=true` now triggers the forced aligner automatically.
> On macOS / Apple Silicon, Qwen3-ASR now runs through the Rust CPU backend.

**Custom GPU backend builds:**

```bash
# Default GPU build: CUDA 12.8 / PyTorch cu128
docker build -t qwen3-asr:gpu-cu128 -f Dockerfile.gpu .

# CUDA 12.6 build for older deployments
docker build -t qwen3-asr:gpu-cu126 -f Dockerfile.gpu \
  --build-arg PYTORCH_BASE_IMAGE=pytorch/pytorch:2.10.0-cuda12.6-cudnn9-runtime \
  --build-arg PYTORCH_CUDA_INDEX=https://download.pytorch.org/whl/cu126 \
  --build-arg CUDA_NVCC_PACKAGE=cuda-nvcc-12-6 \
  --build-arg TORCH_CUDA_ARCH_LIST="8.0;8.6;8.9" \
  .

# CUDA 13.0 build when your driver/toolchain requires it
docker build -t qwen3-asr:gpu-cu130 -f Dockerfile.gpu \
  --build-arg PYTORCH_BASE_IMAGE=pytorch/pytorch:2.10.0-cuda13.0-cudnn9-runtime \
  --build-arg PYTORCH_CUDA_INDEX=https://download.pytorch.org/whl/cu130 \
  --build-arg CUDA_NVCC_PACKAGE=cuda-nvcc-13-0 \
  --build-arg TORCH_CUDA_ARCH_LIST="12.0+PTX" \
  .
```

**Offline Deployment**: Use the helper script to prepare the current runtime model package, then copy to the offline machine:

```bash
# 1. Prepare models
./scripts/prepare-models.sh

# 2. Copy the package to offline server
scp qwen3-asr-models-*.tar.gz user@server:/opt/qwen3-asr/

# 3. On offline server, extract and start
tar -xzvf qwen3-asr-models-*.tar.gz
docker compose up -d
```

> Detailed deployment instructions: [Deployment Guide](./docs/deployment.md)

### Local Development

**System Requirements:**

- Python 3.10+
- CUDA 12.8+ for the default GPU image; CUDA 12.6 / 13.0 can be built with Docker args
- FFmpeg (audio format conversion)

**Installation:**

Runtime dependency locks now default to the GPU stack at the repo root, with CPU kept as a specialized environment:

| Mode | Command | Notes |
|------|---------|-------|
| GPU (default) | `uv sync` | Syncs the root [pyproject.toml](/opt/qwen3-asr/pyproject.toml) and [uv.lock](/opt/qwen3-asr/uv.lock) into `.venv`, including CUDA 12.8/cu128 `torch/torchaudio/torchvision` |
| CPU (specialized) | `./scripts/sync_cpu_env.sh` | Syncs the dedicated CPU lock in [environments/cpu/pyproject.toml](/opt/qwen3-asr/environments/cpu/pyproject.toml) into `.venv` |

```bash
# Clone project
cd qwen3-asr

# Install dependencies (Linux/CUDA)
uv sync

# Start service
source .venv/bin/activate
python start.py
```

macOS / Apple Silicon local development:

```bash
./scripts/sync_cpu_env.sh
source .venv/bin/activate
python start.py
```

Interactive local terminals use the startup UI automatically. Containers and
multi-worker deployments use normal logs.

## Runtime Defaults

Current runtime behavior on the mainline codebase:

- `DEVICE=auto` resolves to `cuda:0` when CUDA is available, otherwise `cpu`
- `DEVICE=mps` is normalized to `cpu`
- `Linux + CUDA` uses official `vLLM`
- `Linux + CPU` uses vendored `QwenASR` Rust
- `macOS / Apple Silicon` also uses vendored `QwenASR` Rust
- macOS / Apple Silicon defaults to `qwen3-asr-0.6b`
- `qwen3-asr-1.7b` on macOS is only used when `QWEN3_ASR_MODEL=qwen3-asr-1.7b`
- The default offline model is `qwen3-asr-confucius4-r2t2` (Confucius4-R2T2, CUDA vLLM path)
- `word_timestamps=true` works on the current offline CUDA and CPU Rust paths
- WebSocket streaming does not currently return word-level timestamps
- CAM++ speaker diarization remains required and still follows `DEVICE`; on CPU its main hotspot is speaker verification embedding

## API Endpoints

### Authentication

When `API_KEY` is configured in `.env`, every API endpoint requires the key.
Different endpoint groups accept different headers:

| Endpoint group | `X-NLS-Token` | `Authorization: Bearer` |
|----------------|:---:|:---:|
| `/api/v1/voiceprint-*` (voiceprint management) | ✅ | ✅ |
| `/stream/*` (Alibaba Cloud compatible REST) | ✅ | ✅ |
| `/v1/*` (OpenAI compatible) | ✅ | ✅ |
| `/ws/v1/asr/*` (WebSocket) | header or query param `token` / `x_nls_token` | ❌ |

Notes:
- All HTTP endpoint groups accept both `X-NLS-Token` and
  `Authorization: Bearer` (`X-NLS-Token` takes precedence).
- WebSocket connections cannot send custom headers from browsers, so tokens
  are accepted via query parameters (`token` / `x_nls_token`).
- `/docs` (Swagger), `/redoc` and `/` (root info) are public and need no auth.
- When `API_KEY` is empty, authentication is skipped for all endpoints.

### HTTP Transcription Endpoints Compared

Both offline transcription endpoints are **synchronous** (submit the full
audio, wait, and get the complete result in one response). Both use
qwen3-asr-confucius4-r2t2 and support speaker diarization, voiceprint naming and
word-level timestamps. True streaming lives on the WebSocket endpoints below:

| Aspect | `/stream/v1/asr` (Alibaba Cloud compatible) | `/v1/audio/transcriptions` (OpenAI compatible) |
|--------|---------------------------------------------|-------------------------------------------------|
| Input | Raw audio body (octet-stream) or `audio_address` URL; no multipart upload | `file` upload or `audio_address` URL |
| Hotwords | ✅ `vocabulary_id` | ✅ `prompt` (merged with preset) |
| Language | Auto-detected | ✅ `language` parameter |
| Response | Fixed Alibaba Cloud JSON (`task_id`/`status`/`result`/`processing_time`) | `json` / `text` / `srt` / `vtt` / `verbose_json` |
| Best for | Migrating Alibaba NLS clients (hotwords, same protocol) | OpenAI ecosystem (SDK, subtitle output) |

### OpenAI Compatible API

| Endpoint | Method | Function |
|----------|--------|----------|
| `/v1/audio/transcriptions` | POST | Audio transcription (OpenAI compatible) |
| `/v1/models` | GET | Offline model list |

**Request Parameters:**

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `file` | file | Preferred when provided | Audio/video file |
| `audio_address` | string | Optional | Audio/video URL (HTTP/HTTPS). Ignored when `file` is also provided |
| `language` | string | Auto-detect | Language code (zh/en/ja) |
| `enable_speaker_diarization` | bool | `true` | Enable speaker diarization |
| `word_timestamps` | bool | `false` | Return word-level timestamps when the backend supports them. Qwen CUDA vLLM and CPU Rust automatically use the forced aligner when enabled. |
| `response_format` | string | `verbose_json` | Output format |
| `prompt` | string | optional | Hotword context (comma-separated, ≤512 chars), merged with the `ASR_DEFAULT_HOTWORDS` preset before being injected into the recognition context |
| `temperature` | float | `0` | Sampling temperature (reserved) |

**Audio / Video Input Methods:**
- **File Upload**: Use `file` parameter to upload an audio file or a video container with an audio track
- **URL Download**: Use `audio_address` parameter to provide an audio/video URL, service will download automatically
- **Precedence**: If both `file` and `audio_address` are provided, the service uses `file` and ignores `audio_address`

**Usage Examples:**

```python
# Using OpenAI SDK
from openai import OpenAI

client = OpenAI(base_url="http://localhost:9101/v1", api_key="your_api_key")

with open("audio.wav", "rb") as f:
    transcript = client.audio.transcriptions.create(
        file=f,
        response_format="verbose_json"  # Get segments and speaker info
    )
print(transcript.text)
```

```bash
# Using curl
curl -X POST "http://localhost:9101/v1/audio/transcriptions" \
  -H "Authorization: Bearer your_api_key" \
  -F "file=@audio.wav" \
  -F "model=qwen3-asr-0.6b" \
  -F "response_format=verbose_json" \
  -F "enable_speaker_diarization=true"
```

**Supported Response Formats:** `json`, `text`, `srt`, `vtt`, `verbose_json`

### Alibaba Cloud Compatible API

| Endpoint | Method | Function |
|----------|--------|----------|
| `/stream/v1/asr` | POST | Speech recognition (long audio support) |
| `/stream/v1/asr/models` | GET | Declared model/capability entries |
| `/stream/v1/asr/health` | GET | Health check |
| `/ws/v1/asr` | WebSocket | Streaming ASR (Alibaba Cloud protocol compatible) |
| `/ws/v1/asr/funasr` | WebSocket | FunASR streaming (backward compatible) |
| `/ws/v1/asr/qwen` | WebSocket | Qwen3-ASR streaming |

**Request Parameters:**

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `audio_address` | string | `https://media.cdn.vect.one/podcast_demo.mp4` (docs example) | Audio/video URL (optional; ignored when body content is uploaded) |
| `sample_rate` | int | `16000` | Sample rate |
| `enable_speaker_diarization` | bool | `true` | Enable speaker diarization |
| `word_timestamps` | bool | `false` | Return word-level timestamps when the backend supports them. Qwen CUDA vLLM and CPU Rust automatically use the forced aligner when enabled. |
| `vocabulary_id` | string | - | Hotword context (comma-separated, for example: `word1,word2,Bank of China`), merged with the `ASR_DEFAULT_HOTWORDS` preset; falls back to the preset when omitted. **Deprecated:** numeric weights are unsupported and ignored. |

**Usage Examples:**

```bash
# Basic usage
curl -X POST "http://localhost:9101/stream/v1/asr" \
  -H "Content-Type: application/octet-stream" \
  --data-binary @audio.wav

# With parameters
curl -X POST "http://localhost:9101/stream/v1/asr?enable_speaker_diarization=true" \
  -H "Content-Type: application/octet-stream" \
  --data-binary @audio.wav
```

**Response Example:**

```json
{
  "task_id": "xxx",
  "status": 200,
  "message": "SUCCESS",
  "result": "Speaker1 content...\nSpeaker2 content...",
  "duration": 60.5,
  "processing_time": 1.234,
  "segments": [
    {
      "text": "Today is a nice day.",
      "start_time": 0.0,
      "end_time": 2.5,
      "speaker_id": "Speaker1",
      "word_tokens": [
        {"text": "Today", "start_time": 0.0, "end_time": 0.5},
        {"text": "is", "start_time": 0.5, "end_time": 0.7},
        {"text": "a nice day", "start_time": 0.7, "end_time": 1.5}
      ]
    }
  ]
}
```

### WebSocket Streaming API

Three WebSocket endpoints are available under `/ws/v1/asr`:

| Endpoint | Backend | Notes |
|----------|---------|-------|
| `/ws/v1/asr` | FunASR realtime (Paraformer) | Alibaba Cloud protocol compatible, Chinese streaming with realtime punctuation |
| `/ws/v1/asr/funasr` | FunASR realtime | Backward-compatible alias of the above |
| `/ws/v1/asr/qwen` | Qwen3-ASR (vLLM) | Qwen3 streaming, multilingual (zh/en/ja/...) |

All three endpoints are **real-time speech transcription** — audio is sent
incrementally and partial results stream back while speaking. This is the
fundamental difference from the offline HTTP endpoints:

| | WebSocket (real-time) | HTTP offline (`/v1/*`, `/stream/v1/asr`) |
|---|---|---|
| Interaction | Long-lived connection, send audio while receiving results | Submit the full audio once, wait synchronously |
| Results | Partial increments → `segment_end` confirmation | Full result (segments + timestamps) |
| Speaker diarization | ❌ | ✅ |
| Voiceprint naming | ❌ | ✅ |
| Word-level timestamps | ❌ | ✅ |
| Best for | Live captions, voice assistants, intercom | Recording transcription, meeting minutes, post-analysis |

For "real-time + speakers", use the dual-request pattern: WebSocket for live
captions, then re-send the full recording to the offline endpoint for
diarized/voiceprint-named segments.

Authentication: WebSocket connections cannot send custom headers from
browsers, so pass the API key as a query parameter (`?token=YOUR_KEY` or
`x_nls_token`). Skipped when `API_KEY` is not configured.

**Qwen3 streaming protocol** (`/ws/v1/asr/qwen`) - JSON control messages plus
binary audio frames:

1. Connect, then send `start`:
   ```json
   {"type": "start", "payload": {
     "format": "pcm",              // "pcm" or "wav"
     "sample_rate": 16000,
     "language": "zh",             // optional, auto-detected
     "context": "",                // hotword context
     "enable_inverse_text_normalization": true,
     "chunk_size_sec": 2.0,
     "unfixed_chunk_num": 2,
     "unfixed_token_num": 5
   }}
   ```
2. Server replies `started` with the applied params.
3. Send binary audio frames continuously (16kHz int16 PCM, or WAV bytes).
4. Server emits `segment_start` / `segment_end` incrementally; `segment_end`
   carries the full transcript so far (`result.text`) and the confirmed list.
5. Send `{"type": "stop"}` to finish; errors arrive as `{"type": "error", ...}`.

Python example:

```python
import asyncio, json, websockets

async def main():
    async with websockets.connect(
        "ws://localhost:9101/ws/v1/asr/qwen?token=YOUR_API_KEY"
    ) as ws:
        await ws.send(json.dumps({"type": "start", "payload": {"format": "pcm"}}))
        with open("audio.wav", "rb") as f:
            f.read(44)  # skip WAV header
            while chunk := f.read(3200):  # 0.1s frames @16kHz int16
                await ws.send(chunk)
        await ws.send(json.dumps({"type": "stop"}))
        async for raw in ws:
            msg = json.loads(raw)
            if msg.get("type") == "segment_end":
                print(msg["result"]["text"])

asyncio.run(main())
```

**FunASR streaming protocol** (`/ws/v1/asr` and `/ws/v1/asr/funasr`) -
Alibaba Cloud NLS style: send JSON frames
(`{"header": {"message_id": "...", "task": "start"}, "payload": {...}}`)
followed by binary audio, then `task: "stop"`. The server emits events:
`TranscriptionStarted` → `SentenceBegin` → `TranscriptionResultChanged`
(intermediate results) → `SentenceEnd` → `TranscriptionCompleted`. Payload
options include `format`, `sample_rate`, `enable_intermediate_result`,
`enable_punctuation`, `enable_itn`.

**Streaming behavior** (Qwen3 path): recognition is genuinely real-time. Each
audio chunk (default 2s) is decoded incrementally and pushed as
`type: "result"` with `is_partial: true` — clients can render live captions.
When a sentence finishes (3s silence or the per-segment cap) a `segment_end`
is emitted and the text is confirmed. Latency tuning:

| Param | Default | Effect |
|-------|---------|--------|
| `chunk_size_sec` | 2.0 | First-char latency; lower (e.g. 1.0) for more real-time feel |
| `unfixed_token_num` | 5 | Tail rollback tokens for boundary correction |
| `unfixed_chunk_num` | 2 | Warm-up chunks before truncation is allowed |
| `max_new_tokens` | 2048 | Per-chunk generation cap (server default, configurable) |
| `max_new_tokens_initial` | 32 | Per-chunk generation cap during warm-up |
| `warmup_chunks` | 8 | Number of warm-up chunks (first N use the initial cap) |

> **About `max_new_tokens` (measured on Confucius4)**: this project re-runs the
> **full accumulated audio** every chunk (no official KV-incremental decoding),
> so the value is the generation cap for the whole accumulated text and must be
> large enough; setting it to `2` truncates the text each round until only
> punctuation is left. It is a **cap, not a cost** (generation stops at EOS), so
> raising it does not add latency — low latency comes from `chunk_size_sec`.
> Server defaults are configurable via the `QWEN_STREAM_*` environment variables.

**Streaming limitations**:
- Both paths do **not** return word-level timestamps.
- The FunASR path keeps at most ~10 seconds of pending audio and applies
  backpressure instead of dropping audio.
- **Neither path supports speaker diarization or voiceprint naming.** Voiceprint
  matching requires complete per-speaker segments, so it only runs on the
  offline endpoints. Workaround: transcribe live with WebSocket, then re-send
  the full recording to `/v1/audio/transcriptions` (or `/stream/v1/asr`) to get
  diarized segments with registered voiceprint names.

## Speaker Diarization

Multi-speaker automatic identification based on CAM++ model:

- **Enabled by Default** - `enable_speaker_diarization=true`
- **Automatic Detection** - No preset speaker count needed, model auto-detects
- **Speaker Labels** - Response includes `speaker_id` field (e.g., "Speaker1", "Speaker2")
- **Smart Merging** - Two-layer merge strategy to avoid isolated short segments:
  - Layer 1: Accumulate merge same-speaker segments < 10 seconds
  - Layer 2: Accumulate merge continuous segments up to 60 seconds
- **Subtitle Support** - SRT/VTT output includes speaker labels `[Speaker1] text content`

Disable speaker diarization:

```bash
# OpenAI API
-F "enable_speaker_diarization=false"

# Alibaba Cloud API
?enable_speaker_diarization=false
```

## Audio Processing

### Intelligent Segmentation Strategy

Automatic long audio segmentation:

1. **VAD Voice Detection** - Detect voice boundaries, filter silence
2. **Greedy Merge** - Accumulate voice segments, ensure each segment does not exceed `MAX_SEGMENT_SEC` (default 60s)
3. **Silence Split** - Force split when silence between voice segments exceeds 3 seconds
4. **Batch Inference** - Multi-segment parallel processing, 2-3x performance improvement in GPU mode

### WebSocket Streaming Limitations

**FunASR Model Limitations** (using `/ws/v1/asr` or `/ws/v1/asr/funasr`):
- ✅ Real-time speech recognition, low latency
- ✅ Sentence-level timestamps
- ❌ **Word-level timestamps** (not implemented on the FunASR realtime path)
- ❌ **Confidence scores** (not implemented)
- Audio ingress is bounded to 10 seconds and applies WebSocket backpressure instead of dropping audio. Send incrementally and continue reading recognition events.
- This path uses Paraformer, while offline endpoints use Qwen3-ASR. Compare latency within the same path, not their total throughput.

**Qwen3-ASR Streaming** (using `/ws/v1/asr/qwen`):
- ✅ Multi-language real-time recognition
- ✅ CUDA vLLM and CPU Rust both support the current streaming path
- ❌ Word-level timestamps are not available in the current streaming path

### Qwen3 Runtime Matrix

| Runtime | Backend | Offline | WebSocket Streaming | Word Timestamps Offline | Word Timestamps Streaming | Maturity |
|---------|---------|---------|---------------------|-------------------------|---------------------------|----------|
| Linux + NVIDIA GPU | Official vLLM 0.19.0 | ✅ | ✅ | ✅ | ❌ | Production-oriented |
| CPU / macOS | QwenASR Rust | ✅ | ✅ | ✅ (forced aligner) | ❌ | Recommended local fallback |

## Offline-Capable Models

| Model ID | Name | Description | Features |
|----------|------|-------------|----------|
| `qwen3-asr-confucius4-r2t2` | Confucius4-R2T2 | **Default model**. NetEase Youdao realtime ASR (`default: true` in `models.json`); CUDA uses vLLM | Offline/Realtime |
| `qwen3-asr-1.7b` | Qwen3-ASR 1.7B | High-performance multilingual ASR, 52 languages + dialects; CUDA uses vLLM | Offline/Realtime |
| `qwen3-asr-0.6b` | Qwen3-ASR 0.6B | Lightweight multilingual ASR; CUDA uses vLLM, CPU/macOS uses Rust backend | Offline/Realtime |

## Realtime-Only Capability

| Capability ID | Runtime | Description |
|---------------|---------|-------------|
| `paraformer-large` | FunASR realtime | Chinese websocket realtime stack with realtime punctuation |

**Runtime selection:**
- **Default**: `qwen3-asr-confucius4-r2t2` (low VRAM: ~6.0GB for the ASR engine alone, ~8.7GB for the full stack with VAD/voiceprint/Paraformer/punctuation on a 24GB card)
- **Tight VRAM / CPU-only / macOS**: Select the vendored Rust-backed `qwen3-asr-0.6b`
- **Environment override**: Set `QWEN3_ASR_MODEL=qwen3-asr-confucius4-r2t2`, `qwen3-asr-1.7b` or `qwen3-asr-0.6b` to bypass automatic selection
- **Local paths**: override with `QWEN3_ASR_MODEL_PATH` / `QWEN3_FORCED_ALIGNER_PATH` (`models.json` supports `${VAR:-default}` placeholders)
- `paraformer-large` realtime capability is always prepared for websocket streaming

At startup the service checks the current runtime model plan and downloads missing models by default. Set `HF_HUB_OFFLINE=1` only for strictly offline deployments with a prepared cache.

## Model Inventory

All runtime models live under `/data1/models/` (paths are configurable via
`MODELSCOPE_CACHE`, `HF_HOME`, `QWEN3_ASR_MODEL_PATH`, `QWEN3_FORCED_ALIGNER_PATH`),
split across two channels:

| Model | Size | Channel & Path | Used By |
|-------|------|----------------|---------|
| Confucius4-R2T2 | ~3.9GB | Local: `/data1/models/Confucius4-R2T2` | **Default** offline transcription, `/ws/v1/asr/qwen` streaming |
| Qwen3-ForcedAligner 0.6B | ~1.8GB | Local: `/data1/models/Qwen3-ForcedAligner-0.6B` | Word-level timestamps (`word_timestamps=true`, offline, lazy-loaded) |
| Paraformer Large | ~849MB | MS: `/data1/models/modelscope/hub/models/iic/speech_paraformer-large_...` | `/ws/v1/asr` + `/ws/v1/asr/funasr` (Chinese realtime streaming) |
| Punctuation (offline) | ~300MB | MS: `/data1/models/modelscope/hub/models/iic/punc_ct-transformer_zh-cn-common-vocab272727-pytorch` | Punctuation for offline/streaming ASR |
| Punctuation (realtime) | ~300MB | MS: `/data1/models/modelscope/hub/models/iic/punc_ct-transformer_..._vad_realtime-...` | Realtime punctuation in the FunASR websocket path |
| CAM++ diarization | ~300MB | MS: `/data1/models/modelscope/hub/models/iic/speech_campplus_speaker-diarization_common` | Speaker diarization (offline) |
| CAM++ speaker verification | ~90MB | MS: `/data1/models/modelscope/hub/models/damo/speech_campplus_sv_zh-cn_16k-common` | Speaker clustering + voiceprint matching |
| CAM++ SCL | ~90MB | MS: `/data1/models/modelscope/hub/models/damo/speech_campplus-transformer_scl_zh-cn_16k-common` | Speaker clustering |
| VAD (FSMN) | ~69MB | MS: `/data1/models/modelscope/hub/models/damo/speech_fsmn_vad_zh-cn-16k-common-pytorch` | Voice activity detection for audio splitting |

**Scenario guidance**:

- **Chinese realtime streaming** (low latency, realtime punctuation): FunASR
  path uses Paraformer Large + realtime punctuation (ModelScope).
- **Multilingual / offline / word timestamps**: Confucius4-R2T2 + ForcedAligner.
- **Speaker diarization & voiceprints** (offline only): CAM++ models (ModelScope).
- **Download behavior**: missing models are auto-downloaded at startup
  (ModelScope models from `modelscope.cn`, Qwen models from HuggingFace, or the
  `HF_ENDPOINT` mirror). For fully offline deployments, place the models under
  `/data1/models` and configure the path variables above.

**Endpoint × model matrix**:

| Endpoint | Model |
|----------|-------|
| `POST /v1/audio/transcriptions` | qwen3-asr-confucius4-r2t2 |
| `POST /stream/v1/asr` | qwen3-asr-confucius4-r2t2 |
| `GET /v1/models`, `health`, `/stream/v1/asr/models` | read-only, no inference |
| `/ws/v1/asr/qwen` | qwen3-asr-confucius4-r2t2 |
| `/ws/v1/asr`, `/ws/v1/asr/funasr` | paraformer-large (Chinese realtime) |

All HTTP transcription endpoints share the default offline model
(`qwen3-asr-confucius4-r2t2`; override with `QWEN3_ASR_MODEL=qwen3-asr-0.6b`).
Paraformer serves only the two FunASR-compatible WebSocket endpoints.

## Voiceprint Database

Persistent speaker identity matching: when a voiceprint is registered for a
speaker, matched segments in ASR results replace the `speaker_id` value with the
registered display name. The ASR response schema is unchanged; uncertain matches
keep the local diarization label (`说话人1`, `Speaker1`, ...).

- **Enabled by Deployment** - Controlled by `VOICEPRINT_ENABLED`, not request parameters
- **Local Vector Store** - Uses SQLite plus `sqlite-vec`; no external PostgreSQL service is required
- **Multiple Samples per Speaker** - Register several single-speaker clips for the same identity
- **Conservative Matching** - Internal score uses `max_score * 0.7 + top3_mean_score * 0.3`

Create a speaker with one or more voiceprint samples:

```bash
curl -X POST 'http://localhost:9101/api/v1/voiceprint-speakers' \
  -F 'display_name=Alice' \
  -F 'file=@speaker_reference.wav'
```

Add more samples to an existing speaker:

```bash
curl -X POST 'http://localhost:9101/api/v1/voiceprint-speakers/{speaker_id}/samples' \
  -F 'file=@another_reference.wav'
```

List registered speakers:

```bash
curl 'http://localhost:9101/api/v1/voiceprint-speakers'
```

Soft-delete a speaker:

```bash
curl -X DELETE 'http://localhost:9101/api/v1/voiceprint-speakers/{speaker_id}'
```

The voiceprint database is persisted under `./data/voiceprints.sqlite3` (mounted
by Docker Compose). See [docs/voiceprint-architecture.md](docs/voiceprint-architecture.md)
for storage and matching design details.

## Environment Variables

Settings in `.env.example`:

| Variable | Default | Description |
|----------|---------|-------------|
| `PORT` | `8000` | Service listen port (set as needed for local deployment; freely changeable) |
| `DEVICE` | `auto` | Inference device (`cuda:0`, `cpu`) |
| `NGINX_PORT` | `9101` | Host port exposed by Docker Compose |
| `API_KEY` | - | API authentication key (optional, unauthenticated if not set) |
| `CUDA_VISIBLE_DEVICES` | `0` | Visible GPU list; one backend instance is started per visible GPU |
| `QWEN3_ASR_MODEL` | auto | Force `qwen3-asr-confucius4-r2t2`, `qwen3-asr-1.7b` or `qwen3-asr-0.6b` |
| `QWEN3_ASR_MODEL_PATH` | `netease-youdao/Confucius4-R2T2` | ASR model path (local absolute path or HF repo id; referenced via `${VAR:-default}` in `models.json`) |
| `QWEN3_FORCED_ALIGNER_PATH` | `Qwen/Qwen3-ForcedAligner-0.6B` | Word-level timestamp aligner path |
| `HF_HUB_OFFLINE` | unset | Set to `1` only after preparing models for offline deployment |
| `HF_ENDPOINT` | unset | Online Hugging Face mirror endpoint, for example `https://hf-mirror.com` |
| `HF_HOME` | `~/.cache/huggingface` | HuggingFace cache directory (this project uses `/data1/models/huggingface`) |
| `MODELSCOPE_CACHE` | `~/.cache/modelscope/hub/models` | ModelScope cache root (this project uses `/data1/models/modelscope/hub`; the library appends `/models/<id>`) |
| `QWEN_GPU_MEMORY_UTILIZATION` | auto | vLLM GPU memory utilization (0.0-1.0) for the main ASR model. Operational parameters live in `.env` only (no longer hard-coded in `models.json`); explicit values win over auto-estimation |
| `QWEN_KV_CACHE_MEMORY_BYTES` | - | Exact KV cache size in bytes (e.g. `1610612736` = 1.5GiB). Once set, vLLM no longer fills VRAM by utilization |
| `QWEN_MAX_NUM_SEQS` | vLLM default | Max concurrent sequences (this project uses `5`, i.e. 5 concurrent requests) |
| `QWEN_MAX_NUM_BATCHED_TOKENS` | vLLM default | Max tokens per batch (also the encoder cache budget; this project uses `2048`) |
| `QWEN_MAX_MODEL_LEN` | `4096` | Context length limit |
| `QWEN_ENFORCE_EAGER` | `false` | Disable CUDA graph / torch.compile, saving ~0.5GB VRAM (this project uses `true`) |
| `QWEN_FORCE_ALIGNER_GPU_MEMORY_UTILIZATION` | inherits | GPU memory utilization for the forced aligner vLLM instance (0.0-1.0) |
| `QWEN_FORCE_ALIGNER_MAX_MODEL_LEN` | - | `max_model_len` for the forced aligner (this project uses `2048`) |
| `QWEN_FORCED_ALIGNER_ENABLED` | `true` | Enable the forced aligner (`false` disables word-level timestamps) |
| `QWEN_WARMUP_FORCED_ALIGNER` | `false` | Warm up the aligner at startup. When `false`, it is lazily loaded on the first `word_timestamps=true` request (key to cutting startup VRAM from 16GB to 6GB) |
| `QWEN_STREAM_MAX_NEW_TOKENS` | `2048` | Per-chunk generation cap for streaming. Each round re-runs the full accumulated audio (no KV-incremental decoding), so it must cover the whole accumulated text; too small truncates the text. **It is a cap, not a cost** (stops at EOS), so raising it does not add latency |
| `QWEN_STREAM_MAX_NEW_TOKENS_INITIAL` | `32` | Generation cap during warm-up (first `QWEN_STREAM_WARMUP_CHUNKS` chunks) |
| `QWEN_STREAM_WARMUP_CHUNKS` | `8` | Number of streaming warm-up chunks |
| `QWEN_STREAM_SEGMENT_MAX_SEC` | `12` | **Sliding window**: max segment duration in seconds. On expiry the segment is finalized and a new one starts, keeping each decoding window constant regardless of session length (suited to hour/day-long realtime tasks) |
| `QWEN_STREAM_SILENCE_SEC` | `2.0` | Silence threshold in seconds; accumulated silence beyond it finalizes the segment (natural sentence boundary preferred) |
| `QWEN_STREAM_OVERLAP_SEC` | `1.0` | Inter-segment audio overlap in seconds: the new segment is primed with the previous segment's tail to reduce word loss on forced cuts; overlapping text is deduplicated (case-insensitive) |
| `QWEN_IDLE_UNLOAD_TIMEOUT` | `15` | First tier: sleep the vLLM engines (weights offloaded to CPU, processes stay resident, wake-up <1s) after this many seconds without requests; `0` disables (this project uses `15`) |
| `QWEN_SHUTDOWN_AFTER_S` | `600` | Second tier: after this many more idle seconds the engines are fully shut down, returning the CUDA context that sleep keeps reserved; `0` disables; the next request needs a full reload (11-21s) |
| `QWEN_LOW_VRAM_THRESHOLD_GB` | `6.0` | Trigger VRAM-pressure sleep when free VRAM drops below this value while engines are idle |
| `QWEN_VRAM_PRESSURE_COOLDOWN_S` | `60` | Idle seconds required before a VRAM-pressure sleep |
| `ASR_DEFAULT_HOTWORDS` | (empty) | Server-side preset hotwords (comma-separated, recommended ≤512 chars), merged with request hotwords (OpenAI `prompt` / native `vocabulary_id`) before being injected into the recognition context; hotwords are adopted preferentially, not forced |
| `VOICEPRINT_ENABLED` | `true` | Enable voiceprint identity matching in ASR results |
| `VOICEPRINT_DB_PATH` | `./data/voiceprints.sqlite3` | SQLite + sqlite-vec voiceprint database path |
| `VOICEPRINT_MATCH_THRESHOLD` | `0.70` | Speaker identity match threshold |
| `VOICEPRINT_DEVICE` | `auto` | Device for voiceprint inference. Defaults to `auto`: GPU when CUDA is available (per-segment input is capped at 30s, peak ~840MiB), CPU fallback otherwise (5-8x slower) |

### GPU Memory Control and Idle Unload

The CUDA vLLM runtime can be squeezed with these `.env` knobs:

```dotenv
QWEN_GPU_MEMORY_UTILIZATION=0.35
QWEN_KV_CACHE_MEMORY_BYTES=1610612736   # 1.5GiB, precise KV cache control
QWEN_MAX_NUM_SEQS=5
QWEN_MAX_NUM_BATCHED_TOKENS=2048
QWEN_MAX_MODEL_LEN=4096
QWEN_ENFORCE_EAGER=true
QWEN_WARMUP_FORCED_ALIGNER=false        # lazy-load the aligner
```

**Lazy forced aligner**: the aligner is a separate vLLM instance (~3GB). With
`QWEN_WARMUP_FORCED_ALIGNER=false` it is not warmed up at startup and only loads
when a `word_timestamps=true` request actually arrives — the key change that
brings normal startup VRAM down from 16GB to ~6GB.

**Measured VRAM** (RTX 4090 24GB, `CUDA_VISIBLE_DEVICES=7`):

| Scenario | VRAM |
|----------|------|
| Baseline (idle) | 0.86GB |
| Sleeping (idle) | ~2.8GB (main process 1.9GB + engine residue 0.95GB) |
| Awake: ASR engine only (Confucius4-R2T2) | ~8.8GB |
| Awake: + forced aligner (word-level timestamps) | ~6.0GB more (peak total ~17GB) |

**Idle sleep**: with `QWEN_IDLE_UNLOAD_TIMEOUT` set, a background monitor puts the
vLLM engines to `sleep` after the configured time without any request — weights
are offloaded to CPU and the KV cache is dropped, but the engine processes stay
resident, so the next request just `wake_up`s them (measured 0.1-0.4s; a full
reload would take 11-21s). Requests that are running or waiting on the engine are
never slept mid-flight, and the forced aligner keeps its own activity counter and
idle timer (it is never slept while busy). Set `0` to keep models awake
permanently; this project uses `15` (15 seconds). Setting
`QWEN_ENABLE_SLEEP_MODE=false` restores the old behaviour (EngineCore processes
are terminated and lazily reloaded).

**VRAM-pressure sleep**: when free VRAM drops below `QWEN_LOW_VRAM_THRESHOLD_GB`
and an engine has been idle longer than `QWEN_VRAM_PRESSURE_COOLDOWN_S`, it yields
the GPU to other processes automatically.

**Second-tier full release**: after `QWEN_SHUTDOWN_AFTER_S` (default `600`, i.e. 10
minutes) of continued idleness the engine processes are shut down, returning the
CUDA context that sleep mode keeps reserved (~1.9GB for both engines). The next
request then triggers a lazy reload (11-21s).

Note: if you rely on utilization alone (without `QWEN_KV_CACHE_MEMORY_BYTES`),
going too low fails startup — the main engine needs at least ~0.16 and the forced
aligner ~0.12 (vLLM 0.19 also profiles CUDA graph memory), otherwise there is no
room left for the KV cache.

## API Documentation

After starting the service:

- Swagger UI: `http://localhost:9101/docs`
- ReDoc: `http://localhost:9101/redoc`

## Links

- **Deployment Guide**: [Detailed Docs](./docs/deployment.md)
- **Qwen3-ASR**: [Qwen3-ASR GitHub](https://github.com/QwenLM/Qwen3-ASR)
- **FunASR**: [FunASR GitHub](https://github.com/alibaba-damo-academy/FunASR)
- **Chinese README**: [中文文档](./README.md)

## License

This project uses the MIT License - see [LICENSE](LICENSE) file for details.

## Star History

[![Star History Chart](https://api.star-history.com/svg?repos=Quantatirsk/qwen3-asr&type=Date)](https://star-history.com/#Quantatirsk/qwen3-asr&Date)

## Contributing

Issues and Pull Requests are welcome to improve the project!
