<div align="center">

<h1>Qwen3-ASR</h1>
<h3>开箱即用的本地私有化部署语音识别服务</h3>

以 [Qwen3-ASR](https://github.com/QwenLM/Qwen3-ASR) 为核心的语音识别 API 服务，提供 CUDA vLLM（默认）、CUDA llama.cpp Q8（低显存，可选）与 CPU Rust 后端，兼容阿里云语音 API 和 OpenAI Audio API，并保留 Paraformer realtime WebSocket 能力。

[English](./README_en.md)

---

![Static Badge](https://img.shields.io/badge/Python-3.10+-blue?logo=python)
![Static Badge](https://img.shields.io/badge/Torch-2.10.0-%23EE4C2C?logo=pytorch&logoColor=white)
![Static Badge](https://img.shields.io/badge/CUDA-12.8_default-%2376B900?logo=nvidia&logoColor=white)

</div>

## 主要特性

- **混合运行时栈** - 离线推理由自动选择的 Qwen3-ASR 提供，WebSocket 流式由 Paraformer realtime 能力提供
- **多语言识别** - Qwen3-ASR 支持 52 种语言与方言，中英日韩等自动检测
- **说话人分离** - 基于 CAM++ 模型自动识别多说话人，返回说话人标记
- **声纹注册与说话人命名** - 为说话人注册声纹样本后，转写结果的 `speaker_id` 自动替换为显示名
- **OpenAI API 兼容** - 支持 `/v1/audio/transcriptions` 端点，可直接使用 OpenAI SDK
- **阿里云 API 兼容** - 支持阿里云语音识别 RESTful API 和 WebSocket 流式协议
- **WebSocket 实时流式** - 边说边出字的实时语音识别，支持增量 partial 结果
- **GPU 显存控制** - 可调 vLLM 显存利用率（`QWEN_GPU_MEMORY_UTILIZATION`）
- **懒加载模型** - 空闲超时（默认 5 分钟）自动卸载模型释放显存，下次请求自动懒加载恢复
- **智能远场过滤** - 流式 ASR 自动过滤远场声音和环境音，减少误触发
- **智能音频分段** - 基于 VAD 的贪婪合并算法，自动切分长音频，避免包含过长静音
- **GPU 批处理加速** - 支持批量推理，比逐个处理快 2-3 倍
- **资源感知运行时** - 根据当前机器资源自动选择合适的 Qwen3-ASR 模型

## 致谢

- [Qwen3-ASR](https://github.com/QwenLM/Qwen3-ASR) 提供官方模型与多模态 / vLLM 使用方式
- [QwenASR](https://github.com/huanglizhuo/QwenASR) 提供本项目 vendored 的 CPU Rust backend
- [Qwen3-ASR Server](https://github.com/Quantatirsk/qwen3-asr) 本项目的上游参考实现

## 快速部署

### 1. 本地 systemd 部署（推荐，无需 Docker）

适用于已安装 NVIDIA GPU 的 Linux 服务器：

```bash
# 安装 uv 包管理器
curl -LsSf https://astral.sh/uv/install.sh | sh

# 克隆并安装依赖
git clone <your-repo> && cd qwen3-asr-server
uv venv --python 3.12
uv sync

# 配置环境变量（端口、模型路径等全部由 .env 驱动，不写死在代码里）
cp .env.example .env
# 模型统一放在 /data1/models（不在项目目录内），例如：
#   ASR 主模型       QWEN3_ASR_MODEL_PATH=/data1/models/Confucius4-R2T2
#   词级时间戳对齐器  QWEN3_FORCED_ALIGNER_PATH=/data1/models/Qwen3-ForcedAligner-0.6B
#   ModelScope 模型  MODELSCOPE_CACHE=/data1/models/modelscope/hub

# 注册并启动服务（软链接方式：改仓库里的 unit 文件后 daemon-reload 即生效）
sudo ln -sfn "$(pwd)/qwen3-asr.service" /etc/systemd/system/qwen3-asr.service
sudo systemctl daemon-reload
sudo systemctl enable --now qwen3-asr
```

服务访问地址（端口由 `.env` 的 `PORT` 决定）：
- **API 端点**: `http://localhost:9101`
- **API 文档**: `http://localhost:9101/docs`

> 详细步骤见 [部署指南 → Linux 本地部署](./docs/deployment.md#linux-本地部署systemd无需-docker)

### 2. Docker 部署

```bash
# 复制并编辑配置
cp .env.example .env
# 编辑 .env 设置 API_KEY（可选）

# 启动服务（GPU 版本）
docker compose up -d

# 或 CPU 版本
docker compose -f docker-compose-cpu.yml up -d

# 多卡自动模式（每张可见卡自动拉起 1 个实例）
CUDA_VISIBLE_DEVICES=0,1,2,3 docker compose up -d
```

服务访问地址：
- **API 端点**: `http://localhost:9101`
- **API 文档**: `http://localhost:9101/docs`

**docker run 方式（替代）:**

```bash
# GPU 版本
docker run -d --name qwen3-asr \
  --gpus all \
  -p 9101:8000 \
  -e CUDA_VISIBLE_DEVICES=0,1,2,3 \
  -e API_KEY=your_api_key \
  -v ./models/modelscope:/root/.cache/modelscope \
  -v ./models/huggingface:/root/.cache/huggingface \
  -v ./data:/app/data \
  quantatrisk/qwen3-asr:gpu-latest

# CPU 版本
docker run -d --name qwen3-asr \
  -p 9101:8000 \
  -v ./models/modelscope:/root/.cache/modelscope \
  -v ./models/huggingface:/root/.cache/huggingface \
  -v ./data:/app/data \
  quantatrisk/qwen3-asr:cpu-latest
```

> **注意**: GPU 镜像默认使用 CUDA 12.8/cu128，以覆盖 Blackwell 等新架构 GPU。
> 开发者可通过 Docker build args 自行构建 CUDA 12.6、CUDA 13.0 或其他后端组合。
> 当前 CPU 镜像已通过内置 QwenASR Rust backend 支持 `qwen3-asr-0.6b`。默认 CPU 镜像使用可分发 Rust 构建目标；只有自建且构建机/部署机 CPU 同构时才建议设置 `QWENASR_RUST_TARGET_CPU=native`。
> CUDA vLLM 与 CPU Rust 路径下，`word_timestamps=true` 都会自动调用 forced aligner；当前实际后端为 `CUDA -> vLLM`、`CPU/macOS -> vendored QwenASR Rust`。
> Apple Silicon 上的 Qwen3-ASR 现已统一走 Rust CPU backend。

**自定义 GPU 后端构建：**

```bash
# 默认 GPU 构建：CUDA 12.8 / PyTorch cu128
docker build -t qwen3-asr:gpu-cu128 -f Dockerfile.gpu .

# CUDA 12.6 构建，用于旧部署环境
docker build -t qwen3-asr:gpu-cu126 -f Dockerfile.gpu \
  --build-arg PYTORCH_BASE_IMAGE=pytorch/pytorch:2.10.0-cuda12.6-cudnn9-runtime \
  --build-arg PYTORCH_CUDA_INDEX=https://download.pytorch.org/whl/cu126 \
  --build-arg CUDA_NVCC_PACKAGE=cuda-nvcc-12-6 \
  --build-arg TORCH_CUDA_ARCH_LIST="8.0;8.6;8.9" \
  .

# CUDA 13.0 构建，用于需要 CUDA 13 工具链的环境
docker build -t qwen3-asr:gpu-cu130 -f Dockerfile.gpu \
  --build-arg PYTORCH_BASE_IMAGE=pytorch/pytorch:2.10.0-cuda13.0-cudnn9-runtime \
  --build-arg PYTORCH_CUDA_INDEX=https://download.pytorch.org/whl/cu130 \
  --build-arg CUDA_NVCC_PACKAGE=cuda-nvcc-13-0 \
  --build-arg TORCH_CUDA_ARCH_LIST="12.0+PTX" \
  .
```

**内网部署**：使用辅助脚本准备当前运行计划所需模型，然后复制到内网机器：

```bash
# 1. 准备模型
./scripts/prepare-models.sh

# 2. 复制到内网服务器
scp qwen3-asr-models-*.tar.gz user@server:/opt/qwen3-asr/

# 3. 在内网服务器解压并启动
tar -xzvf qwen3-asr-models-*.tar.gz
docker compose up -d
```

> 详细部署说明请查看 [部署指南](./docs/deployment.md)

### 本地开发

**系统要求:**

- Python 3.10+
- 默认 GPU 镜像要求 CUDA 12.8+；CUDA 12.6 / 13.0 可通过 Docker build args 自行构建
- FFmpeg (音频格式转换)

**安装步骤:**

运行时依赖现在改成“根目录默认 GPU，CPU 单独特化环境”：

| 模式 | 命令 | 说明 |
|------|------|------|
| GPU（默认） | `uv sync` | 同步根目录 [pyproject.toml](pyproject.toml) 和 [uv.lock](uv.lock) 到 `.venv`，包含 CUDA 12.8/cu128 `torch/torchaudio/torchvision` |
| CPU（特化） | `./scripts/sync_cpu_env.sh` | 同步 [environments/cpu/pyproject.toml](environments/cpu/pyproject.toml) 对应的 CPU lock 到 `.venv` |

```bash
# 克隆项目
cd qwen3-asr

# 安装依赖（Linux/CUDA）
uv sync

# 启动服务
source .venv/bin/activate
python start.py
```

macOS / Apple Silicon 本地开发：

```bash
./scripts/sync_cpu_env.sh
source .venv/bin/activate
python start.py
```

交互式本地终端默认显示启动界面；容器和多 worker 部署使用普通日志输出。

## 当前运行时默认值

当前主线代码的运行时行为如下：

- `DEVICE=auto`
  - 有 CUDA 时解析为 `cuda:0`
  - 否则解析为 `cpu`
- `DEVICE=mps` 会直接归一化为 `cpu`
- `Linux + CUDA` 默认使用官方 `vLLM`；`ASR_ENGINE=llamacpp` 可切换到官方 Q8 GGUF（llama-server，显存约 4GB）
  - Q8 流式默认走 `token` 级 append-only（`QWEN_LLAMACPP_STREAM_MODE=append`）：partial 只增不减、不回撤；`redecode` 可回退为每秒全量重解码
- `Linux + CPU` 使用 vendored `QwenASR` Rust
- `macOS / Apple Silicon` 也使用 vendored `QwenASR` Rust
- macOS / Apple Silicon 默认总是 `qwen3-asr-0.6b`
- 在 macOS 上，只有设置 `QWEN3_ASR_MODEL=qwen3-asr-1.7b` 时才会使用 `qwen3-asr-1.7b`
- 默认离线模型为 `qwen3-asr-confucius4-r2t2`（Confucius4-R2T2，CUDA vLLM 路径）
- `word_timestamps=true` 在当前离线 CUDA 与 CPU Rust 路径下可用
- WebSocket 流式路径当前不返回词级时间戳
- CAM++ 说话人分离仍然必须保留，并继续跟随 `DEVICE`；在 CPU 上的主要热点仍是 speaker verification embedding

## API 接口

### 鉴权

在 `.env` 中配置 `API_KEY` 后，所有接口都要求认证。不同接口组接受的
请求头不同：

| 接口组 | `X-NLS-Token` | `Authorization: Bearer` |
|--------|:---:|:---:|
| `/api/v1/voiceprint-*`（声纹管理） | ✅ | ✅ |
| `/stream/*`（阿里云兼容 REST） | ✅ | ✅ |
| `/v1/*`（OpenAI 兼容） | ✅ | ✅ |
| `/ws/v1/asr/*`（WebSocket） | 请求头或查询参数 `token` / `x_nls_token` | ❌ |

说明：
- 所有 HTTP 接口组同时接受 `X-NLS-Token` 与 `Authorization: Bearer`
  （`X-NLS-Token` 优先）
- WebSocket 浏览器端无法发送自定义请求头，因此支持查询参数传 token
- `/docs`（Swagger）、`/redoc` 和 `/`（根路径信息）为公开页面，无需认证
- `API_KEY` 未配置时，所有接口跳过认证

### HTTP 转写接口对比

两个离线转写端点都是**同步**接口（提交完整音频后等待一次返回完整结果），
均走默认离线模型（`qwen3-asr-confucius4-r2t2`），支持说话人分离、声纹命名与词级时间戳。真正的流式是
WebSocket 端点（见下文）：

| 维度 | `/stream/v1/asr`（阿里云兼容） | `/v1/audio/transcriptions`（OpenAI 兼容） |
|------|------------------------------|------------------------------------------|
| 输入方式 | 请求体原始音频（octet-stream）或 `audio_address` URL；不支持 multipart 上传 | `file` 文件上传 或 `audio_address` URL |
| 热词 | ✅ `vocabulary_id` | ✅ `prompt`（与预设热词合并） |
| 语言指定 | 自动检测 | ✅ `language` 参数 |
| 响应格式 | 固定阿里云 JSON（`task_id`/`status`/`result`/`processing_time`） | `json` / `text` / `srt` / `vtt` / `verbose_json` |
| 适合场景 | 阿里云 NLS 老客户端迁移（热词、协议一致） | OpenAI 生态（OpenAI SDK、字幕输出） |

### OpenAI 兼容接口

| 端点                         | 方法 | 功能                    |
| ---------------------------- | ---- | ----------------------- |
| `/v1/audio/transcriptions` | POST | 音频转写（OpenAI 兼容） |
| `/v1/models`               | GET  | 离线模型列表                |

**请求参数:**

| 参数                           | 类型   | 默认值                | 说明                                  |
| ------------------------------ | ------ | --------------------- | ------------------------------------- |
| `file`                       | file   | 提供时优先使用         | 音频/视频文件                          |
| `audio_address`              | string | 可选                  | 音频/视频文件 URL（HTTP/HTTPS）；若同时提供 `file`，则忽略 |
| `language`                   | string | 自动检测              | 语言代码 (zh/en/ja)                   |
| `enable_speaker_diarization` | bool   | `true`              | 启用说话人分离                        |
| `word_timestamps`            | bool   | `false`             | 返回后端支持的字词级时间戳；Qwen CUDA vLLM 与 CPU Rust 在启用时会自动调用 forced aligner |
| `response_format`            | string | `verbose_json`      | 输出格式                              |
| `prompt`                     | string | 可选                  | 热词/上下文提示（英文逗号分隔，≤512 字符），与 `ASR_DEFAULT_HOTWORDS` 预设热词合并后注入识别上下文 |
| `temperature`                | float  | `0`                   | 采样温度（保留兼容）                  |

**音频/视频输入方式:**
- **文件上传**: 使用 `file` 参数上传音频文件或带音轨的视频容器
- **URL 下载**: 使用 `audio_address` 参数提供音频/视频 URL，服务将自动下载
- **优先级**: 如果同时提供 `file` 和 `audio_address`，服务会优先使用 `file`，并忽略 `audio_address`

**使用示例:**

```python
# 使用 OpenAI SDK
from openai import OpenAI

client = OpenAI(base_url="http://localhost:9101/v1", api_key="your_api_key")

with open("audio.wav", "rb") as f:
    transcript = client.audio.transcriptions.create(
        file=f,
        response_format="verbose_json"  # 获取分段和说话人信息
    )
print(transcript.text)
```

```bash
# 使用 curl
curl -X POST "http://localhost:9101/v1/audio/transcriptions" \
  -H "Authorization: Bearer your_api_key" \
  -F "file=@audio.wav" \
  -F "model=qwen3-asr-0.6b" \
  -F "response_format=verbose_json" \
  -F "enable_speaker_diarization=true"
```

**支持的响应格式:** `json`, `text`, `srt`, `vtt`, `verbose_json`

### 阿里云兼容接口

| 端点                      | 方法      | 功能                   |
| ------------------------- | --------- | ---------------------- |
| `/stream/v1/asr`        | POST      | 语音识别（支持长音频） |
| `/stream/v1/asr/models` | GET       | 声明条目列表               |
| `/stream/v1/asr/health` | GET       | 健康检查               |
| `/ws/v1/asr`            | WebSocket | 流式语音识别（阿里云协议兼容） |
| `/ws/v1/asr/funasr`     | WebSocket | FunASR 流式识别（向后兼容）   |
| `/ws/v1/asr/qwen`       | WebSocket | **已废弃**（返回 `endpoint_deprecated`）|
| `/v1/realtime`          | WebSocket | 标准 OpenAI Realtime 转录（`intent=transcription`）|

**请求参数:**

| 参数                           | 类型   | 默认值             | 说明                                  |
| ------------------------------ | ------ | ------------------ | ------------------------------------- |
| `audio_address`              | string | `https://media.cdn.vect.one/podcast_demo.mp4`（文档示例） | 音频/视频 URL（可选；若同时上传内容则忽略） |
| `sample_rate`                | int    | `16000`          | 采样率                                |
| `enable_speaker_diarization` | bool   | `true`           | 启用说话人分离                        |
| `word_timestamps`            | bool   | `false`          | 返回后端支持的字词级时间戳；Qwen CUDA vLLM 与 CPU Rust 在启用时会自动调用 forced aligner |
| `vocabulary_id`              | string | -                  | 无权重热词上下文（英文逗号分隔，如：`词1,词2,Bank of China`），与 `ASR_DEFAULT_HOTWORDS` 预设热词合并生效；未传时用预设热词兜底。**Deprecated：** 数字权重不受支持，传入时会被忽略。 |

**使用示例:**

```bash
# 基本用法
curl -X POST "http://localhost:9101/stream/v1/asr" \
  -H "Content-Type: application/octet-stream" \
  --data-binary @audio.wav

# 带参数
curl -X POST "http://localhost:9101/stream/v1/asr?enable_speaker_diarization=true" \
  -H "Content-Type: application/octet-stream" \
  --data-binary @audio.wav
```

**响应示例:**

```json
{
  "task_id": "xxx",
  "status": 200,
  "message": "SUCCESS",
  "result": "说话人1的内容...\n说话人2的内容...",
  "duration": 60.5,
  "processing_time": 1.234,
  "segments": [
    {
      "text": "今天天气不错。",
      "start_time": 0.0,
      "end_time": 2.5,
      "speaker_id": "说话人1",
      "word_tokens": [
        {"text": "今天", "start_time": 0.0, "end_time": 0.5},
        {"text": "天气", "start_time": 0.5, "end_time": 0.9},
        {"text": "不错", "start_time": 0.9, "end_time": 1.3}
      ]
    }
  ]
}
```

### WebSocket 流式接口

提供两类 WebSocket 端点：

| 端点 | 后端 | 说明 |
|------|------|------|
| `/v1/realtime` | Qwen3-ASR（R2T2，vLLM / llama.cpp Q8） | **推荐**：标准 OpenAI Realtime transcription 协议，多语言 |
| `/ws/v1/asr` | FunASR realtime（Paraformer） | 阿里云协议兼容，中文流式 + 实时标点 |
| `/ws/v1/asr/funasr` | FunASR realtime | 上述端点的向后兼容别名 |
| `/ws/v1/asr/qwen` | — | **已废弃**：连接后返回 `endpoint_deprecated` 并关闭（close code 1008），请改用 `/v1/realtime` |

这些端点都是**实时语音转录**——音频增量发送，说话的同时返回增量结果。
这与离线 HTTP 接口有本质区别：

| | WebSocket（实时） | HTTP 离线（`/v1/*`、`/stream/v1/asr`） |
|---|---|---|
| 交互方式 | 长连接，边传音频边收结果 | 一次性提交完整音频，同步等待 |
| 结果 | 增量 partial → `segment_end` 确认 | 完整结果（分段 + 时间戳） |
| 说话人分离 | ❌ | ✅ |
| 声纹命名 | ❌ | ✅ |
| 词级时间戳 | ❌ | ✅ |
| 适合场景 | 实时字幕、语音助手、对讲 | 录音转写、会议纪要、事后分析 |

需要"实时 + 说话人"时使用双请求模式：WebSocket 实时出字幕，结束后把
完整录音再发离线接口，获得带说话人分离/声纹名的分段结果。

鉴权：浏览器端 WebSocket 无法发送自定义请求头，通过查询参数传 API key
（`?token=YOUR_KEY` 或 `x_nls_token`）；未配置 `API_KEY` 时免鉴权。

**OpenAI Realtime 转录协议**（`/v1/realtime`）——标准 OpenAI Realtime
WebSocket 子集（`intent=transcription`），识别复用 Qwen3-ASR 流式引擎。

鉴权（三选一）：`Authorization: Bearer <API_KEY>`、查询参数 `?token=<API_KEY>`、
或子协议 `openai-insecure-api-key.<API_KEY>`；未配置 `API_KEY` 时免鉴权。
握手成功回 `Sec-WebSocket-Protocol: realtime`。

客户端事件：

| 事件 | 说明 |
|------|------|
| `session.update` | 设置 `audio.input.format.rate`（16000/24000）、`audio.input.transcription.{model,prompt,language,keywords}`、`audio.input.turn_detection`（`null` 或 `{"type":"server_vad"}`） |
| `input_audio_buffer.append` | `audio` 为 base64 的 PCM16 little-endian；单次 ≤10s |
| `input_audio_buffer.commit` | 手动落定当前段（`turn_detection: null` 时必须） |
| `input_audio_buffer.clear` | 丢弃未提交音频 |

服务端事件：`session.created` / `session.updated`、
`conversation.item.input_audio_transcription.delta`（增量文本，拼接即全文）、
`.completed`（`transcript` 为定稿全文，含标点/ITN 与 two-pass 重转修正）、
`.failed`、`input_audio_buffer.committed`、`input_audio_buffer.cleared`、`error`。

`turn_detection: {"type":"server_vad"}` 时由 FireRedVAD 在句尾静音自动落定
（连续讲话按 `QWEN_STREAM_SEGMENT_MAX_SEC`，默认 60s 兜底），无需客户端
commit；`turn_detection: null` 时整段由客户端 commit 触发落定。
采样率、chunk 粒度、热词等解码参数由服务端 `.env`（`QWEN_STREAM_*`）控制。

Python 示例：

```python
import asyncio, base64, json, websockets

async def main():
    uri = "ws://localhost:9101/v1/realtime?token=YOUR_API_KEY&intent=transcription"
    async with websockets.connect(uri, subprotocols=["realtime"]) as ws:
        print(json.loads(await ws.recv()))  # session.created
        await ws.send(json.dumps({
            "type": "session.update",
            "session": {"audio": {"input": {
                "format": {"type": "audio/pcm", "rate": 16000},
                "transcription": {"model": "qwen3-asr", "prompt": "Bloomberg, 周树门"},
                "turn_detection": {"type": "server_vad"},
            }}}
        }))
        with open("audio.wav", "rb") as f:
            f.read(44)  # 跳过 WAV 头
            while chunk := f.read(3200):  # 16kHz int16，0.1s
                await ws.send(json.dumps({
                    "type": "input_audio_buffer.append",
                    "audio": base64.b64encode(chunk).decode(),
                }))
                await asyncio.sleep(0.1)
                while True:  # 持续读取事件（否则服务端会积压）
                    try:
                        event = json.loads(await asyncio.wait_for(ws.recv(), 0.001))
                    except asyncio.TimeoutError:
                        break
                    if event["type"].endswith(".delta"):
                        print(event["delta"], end="", flush=True)
                    elif event["type"].endswith(".completed"):
                        print("\n[segment]", event["transcript"])

asyncio.run(main())
```

**FunASR 流式协议**（`/ws/v1/asr` 与 `/ws/v1/asr/funasr`）——阿里云 NLS
风格：先发 JSON 帧（`{"header": {"message_id": "...", "task": "start"},
"payload": {...}}`），随后发二进制音频，最后发 `task: "stop"`。服务器事件
顺序：`TranscriptionStarted` → `SentenceBegin` → `TranscriptionResultChanged`
（中间结果）→ `SentenceEnd` → `TranscriptionCompleted`。payload 支持
`format`、`sample_rate`、`enable_intermediate_result`、`enable_punctuation`、
`enable_itn` 等参数。

**实时行为**（Qwen3 路径）：真正的实时识别。每个音频 chunk（默认 160ms）
增量解码后立即推送 `conversation.item.input_audio_transcription.delta`，
客户端拼接 delta 即得实时字幕；FireRedVAD 检测到句尾静音（或单段达到
`QWEN_STREAM_SEGMENT_MAX_SEC` 兜底）时推送 `.completed` 定稿文本。段间无重叠、
无字符去重，文本自然拼接。以下解码参数由服务端 `.env` 控制（客户端不可覆盖），
延迟调优：

| 环境变量 | 默认 | 作用 |
|------|------|------|
| `QWEN_STREAM_CHUNK_SEC` | 0.16 | 解码粒度/出字延迟，官方推荐 160ms（80ms~2s） |
| `QWEN_STREAM_LOOKAHEAD_SEC` | 0.16 | 首 chunk 额外音频，帮助首段判断语言/专名 |
| `QWEN_STREAM_UNFIXED_TOKEN_NUM` | 1 | 句尾回滚修正 token 数（官方默认 1） |
| `QWEN_STREAM_UNFIXED_CHUNK_NUM` | 0 | 前 N 个 chunk 不接历史 prefix（官方默认 0） |
| `QWEN_STREAM_MAX_NEW_TOKENS` | 512 | 生成**上限**，实际按 chunk 时长动态取小值；0 表示不设上限 |
| `QWEN_STREAM_SEGMENT_MAX_SEC` | 60 | 连续讲话无静音时的强制落定上限 |
| `QWEN_STREAM_FINAL_PASS` | true | 段定稿时用离线路径重转校正专名 |

> **`max_new_tokens` 说明**：采用官方 R2T2 算法，每轮生成量约为
> `(chunk + lookahead) / 1280 * 2`（160ms chunk 约 8 个 token），
> 配置值仅作硬上限，不再是「整段文本上限」。低延迟靠 `chunk_size_sec` 与
> 动态上限共同保证。
>
> **热词/上下文**：`session.update` 的 `transcription.prompt` / `keywords`
> 与 `ASR_DEFAULT_HOTWORDS` 合并后按官方约定直接作为 system prompt（不再包
> "Transcribe the speech..." 指令），离线与流式共用；流式为 append-only，
> 首段已输出文本无法回改，段定稿时可用 `QWEN_STREAM_FINAL_PASS=true`
> 走离线路径重转校正专名。
>
> **delta 单调性**：流式 partial 关闭 ITN（避免 `一`→`1` 这类改写破坏
> 增量拼接），`.completed` 的 `transcript` 才做 ITN 与标点规范化。

**流式限制**：
- 两条路径都不返回词级时间戳
- FunASR 路径最多保留约 10 秒待处理音频，队列满时对发送端施加背压而非
  丢弃音频
- **两条路径都不支持说话人分离与声纹命名**——声纹匹配需要完整的单人
  片段，因此只运行在离线接口。变通方案：先用 WebSocket 实时出字幕，结束
  后把完整录音再发一次 `/v1/audio/transcriptions`（或 `/stream/v1/asr`），
  获得带说话人分离与声纹名的分段结果（双请求模式）

## 说话人分离

基于 CAM++ 模型实现多说话人自动识别：

- **默认开启** - `enable_speaker_diarization=true`
- **自动识别** - 无需预设说话人数量，模型自动检测
- **说话人标记** - 响应中包含 `speaker_id` 字段（如 "说话人1"、"说话人2"）
- **智能合并** - 两层合并策略避免孤立短片段：
  - 第一层：小于10秒的同说话人片段累积合并
  - 第二层：连续片段累积合并至60秒上限
- **字幕支持** - SRT/VTT 格式输出包含说话人标记 `[说话人1] 文本内容`

关闭说话人分离：

```bash
# OpenAI API
-F "enable_speaker_diarization=false"

# 阿里云 API
?enable_speaker_diarization=false
```

## 音频处理

### 智能分段策略

长音频自动分段处理：

1. **VAD 语音检测** - 检测语音边界，过滤静音
2. **贪婪合并** - 短片段与相邻段合并，同段不超过 `MAX_SEGMENT_SEC`（默认60秒）
3. **有界切分不丢尾音** - 超长段按低能量点或固定时长切分，末尾残段并入前段（不会因不足最小时长被丢弃）
4. **句末标点补齐** - R2T2 在段尾被截断时通常不给标点，离线结果会用 FunASR 标点模型补一个句末标点（`ASR_RESTORE_SENTENCE_ENDING=true`，只补标点、不改写正文）
5. **批处理推理** - 多片段并行处理，GPU 模式下性能提升 2-3 倍

### WebSocket 流式识别限制

**FunASR 模型限制**（使用 `/ws/v1/asr` 或 `/ws/v1/asr/funasr`）：
- ✅ 实时语音识别、低延迟
- ✅ 字句级时间戳
- ❌ **词级时间戳**（FunASR realtime 路径未实现）
- ❌ **置信度分数**（未实现）
- 音频入口固定保留最多 10 秒待处理数据；队列满时会对 WebSocket 发送端施加背压，不会静默丢弃音频。客户端应持续读取识别事件并增量发送。
- 该路径使用 Paraformer，离线接口使用 Qwen3-ASR；应在同一路径内比较延迟，不能直接比较两者总吞吐。

**Qwen3-ASR 流式**（使用 `/v1/realtime`，标准 OpenAI Realtime 协议）：
- ✅ 支持多语言实时识别
- ✅ 当前支持 CUDA vLLM 与 CPU Rust 两条流式路径
- ❌ 当前流式路径不返回词级时间戳

### Qwen3 运行时矩阵

| 运行环境 | 后端 | 离线转写 | WebSocket 流式 | 离线词级时间戳 | 流式词级时间戳 | 成熟度 |
|---------|------|---------|----------------|----------------|----------------|--------|
| Linux + NVIDIA GPU | 官方 vLLM 0.19.0 | ✅ | ✅ | ✅ | ❌ | 面向生产 |
| CPU / macOS | QwenASR Rust | ✅ | ✅ | ✅（forced aligner） | ❌ | 推荐本地后端 |

## 支持离线的模型

| 模型 ID              | 名称              | 说明                                     | 特性      |
| -------------------- | ----------------- | ---------------------------------------- | --------- |
| `qwen3-asr-confucius4-r2t2` | Confucius4-R2T2 | **默认模型**。网易有道实时 ASR（`models.json` 中 `default: true`）；CUDA 使用 vLLM | 离线/实时 |
| `qwen3-asr-1.7b`   | Qwen3-ASR 1.7B    | 高性能多语言 ASR；CUDA 使用 vLLM | 离线/实时 |
| `qwen3-asr-0.6b`   | Qwen3-ASR 0.6B    | 轻量版多语言 ASR；CUDA 使用 vLLM，CPU/macOS 使用 Rust backend | 离线/实时 |

## 仅实时能力

| 能力 ID | 运行时 | 说明 |
| ------- | ------ | ---- |
| `paraformer-large` | FunASR realtime | 中文 WebSocket 实时识别栈，包含实时标点链路 |

**运行时选择:**
- **默认**：`qwen3-asr-confucius4-r2t2`（显存需求低：24GB 卡实测仅 ASR 引擎约 6.0GB，含 VAD/声纹/Paraformer/标点完整栈约 8.7GB）
- **显存紧张 / 纯 CPU / macOS**: 选择基于 vendored Rust 的 `qwen3-asr-0.6b`
- **环境变量覆盖**: 设置 `QWEN3_ASR_MODEL=qwen3-asr-confucius4-r2t2`、`qwen3-asr-1.7b` 或 `qwen3-asr-0.6b` 可跳过自动选择
- 模型本地路径可用 `QWEN3_ASR_MODEL_PATH` / `QWEN3_FORCED_ALIGNER_PATH` 覆盖（`models.json` 支持 `${VAR:-default}` 占位符）
- `paraformer-large` 实时能力始终为 WebSocket 流式准备

启动时会先检测当前运行计划所需模型；如果本地缓存缺失，会自动下载。离线部署可显式设置 `HF_HUB_OFFLINE=1` 并提前准备模型缓存。

## 模型清单

所有运行时模型统一放在 `/data1/models/`（路径全部可通过 `MODELSCOPE_CACHE`、`HF_HOME`、`QWEN3_ASR_MODEL_PATH`、`QWEN3_FORCED_ALIGNER_PATH` 调整），分为两个渠道：

| 模型 | 大小 | 渠道与路径 | 用途 |
|------|------|-----------|------|
| Confucius4-R2T2 | ~3.9GB | 本地：`/data1/models/Confucius4-R2T2` | **默认**离线转写、`/v1/realtime` 流式（多语言） |
| Qwen3-ForcedAligner 0.6B | ~1.8GB | 本地：`/data1/models/Qwen3-ForcedAligner-0.6B` | 词级时间戳（离线 `word_timestamps=true`，懒加载） |
| Paraformer Large | ~849MB | MS：`/data1/models/modelscope/hub/models/iic/speech_paraformer-large_...` | `/ws/v1/asr` + `/ws/v1/asr/funasr`（中文实时流式） |
| 标点（离线） | ~300MB | MS：`/data1/models/modelscope/hub/models/iic/punc_ct-transformer_zh-cn-common-vocab272727-pytorch` | 离线/流式标点 |
| 标点（实时） | ~300MB | MS：`/data1/models/modelscope/hub/models/iic/punc_ct-transformer_..._vad_realtime-...` | FunASR 流式路径实时标点 |
| CAM++ 说话人分离 | ~300MB | MS：`/data1/models/modelscope/hub/models/iic/speech_campplus_speaker-diarization_common` | 说话人分离（离线） |
| CAM++ 声纹验证 | ~90MB | MS：`/data1/models/modelscope/hub/models/damo/speech_campplus_sv_zh-cn_16k-common` | 说话人聚类 + 声纹匹配 |
| CAM++ SCL | ~90MB | MS：`/data1/models/modelscope/hub/models/damo/speech_campplus-transformer_scl_zh-cn_16k-common` | 说话人聚类 |
| VAD（FSMN） | ~69MB | MS：`/data1/models/modelscope/hub/models/damo/speech_fsmn_vad_zh-cn-16k-common-pytorch` | 语音活动检测、音频切分 |

**使用场景指引**：

- **中文实时流式**（低延迟 + 实时标点）：FunASR 路径使用 Paraformer Large + 实时标点模型（ModelScope 渠道）
- **多语言 / 离线 / 词级时间戳**：Confucius4-R2T2 + ForcedAligner
- **说话人分离与声纹**（仅离线接口）：CAM++ 系列模型（ModelScope 渠道）
- **下载行为**：启动时自动补下缺失模型（ModelScope 模型走 `modelscope.cn`，Qwen 模型走 HuggingFace 或 `HF_ENDPOINT` 镜像）；完全离线部署请提前把模型放到 `/data1/models` 并配好上述路径变量

**接口 × 模型对应**：

| 接口 | 模型 |
|------|------|
| `POST /v1/audio/transcriptions` | qwen3-asr-confucius4-r2t2 |
| `POST /stream/v1/asr` | qwen3-asr-confucius4-r2t2 |
| `GET /v1/models`、`health`、`/stream/v1/asr/models` | 只读查询，不跑推理 |
| `/v1/realtime` | qwen3-asr-confucius4-r2t2 |
| `/ws/v1/asr`、`/ws/v1/asr/funasr` | paraformer-large（中文实时） |
| `/ws/v1/asr/qwen` | 已废弃，返回 `endpoint_deprecated` |

所有 HTTP 转写接口共用默认离线模型（`qwen3-asr-confucius4-r2t2`，可用
`QWEN3_ASR_MODEL=qwen3-asr-0.6b` 覆盖）；Paraformer 只服务于两个
FunASR 兼容的 WebSocket 端点。

## 声纹数据库

支持持久化说话人身份匹配：为说话人注册声纹后，ASR 结果中匹配到的分段会把
`speaker_id` 替换为注册的显示名。ASR 响应结构不变，匹配不确定时保留本地
说话人分离标签（`说话人1`、`Speaker1` 等）。

- **部署侧启用** - 由 `VOICEPRINT_ENABLED` 控制，不增加请求参数
- **本地向量库** - 使用 SQLite + `sqlite-vec`，无需额外 PostgreSQL 服务
- **同一说话人多样本** - 可为一个 speaker 注册多段单人音频
- **保守匹配策略** - 内部分数为 `max_score * 0.7 + top3_mean_score * 0.3`

创建说话人并注册一个或多个声纹样本：

```bash
curl -X POST 'http://localhost:9101/api/v1/voiceprint-speakers' \
  -F 'display_name=Alice' \
  -F 'file=@speaker_reference.wav'
```

给已有说话人追加样本：

```bash
curl -X POST 'http://localhost:9101/api/v1/voiceprint-speakers/{speaker_id}/samples' \
  -F 'file=@another_reference.wav'
```

查看已注册说话人：

```bash
curl 'http://localhost:9101/api/v1/voiceprint-speakers'
```

软删除说话人：

```bash
curl -X DELETE 'http://localhost:9101/api/v1/voiceprint-speakers/{speaker_id}'
```

声纹数据库持久化在 `./data/voiceprints.sqlite3`（Docker Compose 已挂载 `./data`）。
存储结构和匹配策略详见 [voiceprint-architecture.md](docs/voiceprint-architecture.md)。

## 环境变量

默认部署只需要关注 `.env.example` 中的少数配置：

| 变量                               | 默认值       | 说明                                            |
| ---------------------------------- | ------------ | ----------------------------------------------- |
| `PORT` | `8000` | 服务监听端口（本地部署按需设置，可随意修改） |
| `DEVICE` | `auto` | 推理设备（`cuda:0`、`cpu`） |
| `NGINX_PORT` | `9101` | Docker Compose 映射到宿主机的端口 |
| `API_KEY` | - | API 认证密钥（可选，未配置时无需认证） |
| `CUDA_VISIBLE_DEVICES` | `0` | GPU 设备列表，多卡用 `0,1,2,3` |
| `QWEN3_ASR_MODEL` | 自动选择 | 强制选择 `qwen3-asr-confucius4-r2t2`、`qwen3-asr-1.7b` 或 `qwen3-asr-0.6b` |
| `QWEN3_ASR_MODEL_PATH` | `netease-youdao/Confucius4-R2T2` | ASR 主模型路径（本地绝对路径或 HF 仓库 id；`models.json` 用 `${VAR:-default}` 引用） |
| `QWEN3_FORCED_ALIGNER_PATH` | `Qwen/Qwen3-ForcedAligner-0.6B` | 词级时间戳对齐器路径 |
| `HF_HUB_OFFLINE` | `0` | 离线部署且模型已准备好时设为 `1` |
| `HF_ENDPOINT` | - | 在线镜像站；离线部署不要设置 |
| `HF_HOME` | `~/.cache/huggingface` | HuggingFace 缓存目录（本项目设为 `/data1/models/huggingface`） |
| `MODELSCOPE_CACHE` | `~/.cache/modelscope/hub/models` | ModelScope 模型缓存根目录（本项目设为 `/data1/models/modelscope/hub`，库内部会再拼 `/models/<id>`） |
| `QWEN_GPU_MEMORY_UTILIZATION` | 自动计算 | 主 ASR 模型的 vLLM 显存利用率（0.0-1.0）。运维参数一律走 `.env`（不再写死在 `models.json`）；显式设置优先于自动估算 |
| `QWEN_KV_CACHE_MEMORY_BYTES` | - | KV cache 精确字节数（如 `1610612736` = 1.5GiB）。设置后 vLLM 不再按 utilization 吃满显存 |
| `QWEN_MAX_NUM_SEQS` | vLLM 默认 | 最大并发序列数（本项目设 `5`，即 5 并发） |
| `QWEN_MAX_NUM_BATCHED_TOKENS` | vLLM 默认 | 单批最大 token 数（同时是 encoder cache 预算，本项目设 `2048`） |
| `QWEN_MAX_MODEL_LEN` | `4096` | 上下文长度上限 |
| `QWEN_ENFORCE_EAGER` | `false` | 关闭 CUDA graph / torch.compile，约省 0.5GB 显存（本项目设 `true`） |
| `QWEN_FORCE_ALIGNER_GPU_MEMORY_UTILIZATION` | 继承主模型 | forced aligner（词级时间戳）vLLM 实例的显存利用率（0.0-1.0） |
| `QWEN_FORCE_ALIGNER_MAX_MODEL_LEN` | - | forced aligner 的 `max_model_len`（本项目设 `2048`） |
| `QWEN_FORCED_ALIGNER_ENABLED` | `true` | 是否启用 forced aligner（`false` 则词级时间戳不可用） |
| `QWEN_WARMUP_FORCED_ALIGNER` | `false` | 是否在启动时预热 aligner。`false` 时改为首次 `word_timestamps=true` 请求才懒加载（启动显存 16GB → 6GB 的关键） |
| `ASR_RESTORE_SENTENCE_ENDING` | `true` | 离线结果缺句末标点时用 FunASR CT-Transformer 补一个（仅补标点，不改写正文） |
| `QWEN_STREAM_CHUNK_SEC` | `0.16` | 流式解码粒度（秒），官方 R2T2 推荐 80ms~2s；决定出字延迟 |
| `QWEN_STREAM_LOOKAHEAD_SEC` | `0.16` | 首个 chunk 额外音频（秒），帮助首段判断语言/专名 |
| `QWEN_STREAM_UNFIXED_CHUNK_NUM` | `0` | 前 N 个 chunk 不接历史 prefix（官方默认 0） |
| `QWEN_STREAM_UNFIXED_TOKEN_NUM` | `1` | prefix 回滚 token 数（官方默认 1） |
| `QWEN_STREAM_MAX_NEW_TOKENS` | `512` | 单 chunk 生成**上限**；实际按 chunk 时长动态取小值（约 `(chunk+lookahead)/1280*2`），`0` 表示不设上限 |
| `QWEN_STREAM_SEGMENT_MAX_SEC` | `60` | 单段最大时长（秒），连续讲话无静音时的兜底切段 |
| `QWEN_STREAM_SILENCE_SEC` | `2.0` | VAD 不可用时回退的能量阈值静音切段（秒） |
| `QWEN_STREAM_OVERLAP_SEC` | `0` | 段间重叠音频（秒）；官方 append-only 不需要，保留仅为兼容 |
| `QWEN_STREAM_VAD_MODEL_PATH` | `/data1/models/FireRedVAD/Stream-VAD` | FireRedVAD 流式 VAD 模型目录；缺失时回退能量阈值 |
| `QWEN_STREAM_VAD_MIN_SILENCE_FRAMES` | `50` | 句尾静音帧数（10ms/帧，50 = 500ms）；越小断句越碎 |
| `QWEN_STREAM_FINAL_PASS` | `true` | 段定稿时用离线路径重转（two-pass），修正流式无法回改的热词/专名 |
| `QWEN_IDLE_UNLOAD_TIMEOUT` | `15` | 一级释放：无请求超过该秒数后让 vLLM 引擎休眠（sleep：权重 offload 到 CPU、进程常驻，唤醒 <1s）；`0` 禁用（本项目设 `15` = 15 秒） |
| `QWEN_SHUTDOWN_AFTER_S` | `600` | 二级释放：仍空闲超过该秒数则彻底 shutdown 子进程（连 sleep 残留的 CUDA context 一起归还）；`0` 禁用；代价是下次请求重载 11~21s |
| `QWEN_LOW_VRAM_THRESHOLD_GB` | `6.0` | 可用显存低于该值且引擎空闲时触发显存压力休眠 |
| `QWEN_VRAM_PRESSURE_COOLDOWN_S` | `60` | 显存压力休眠所需的引擎空闲秒数 |
| `ASR_DEFAULT_HOTWORDS` | （空） | 服务端预设热词（英文逗号分隔，建议 ≤512 字符），与请求热词（OpenAI `prompt` / 原生 `vocabulary_id` / 流式 `context`）合并后注入识别上下文；热词为倾向性采纳而非强制替换 |
| `VOICEPRINT_ENABLED` | `true` | 是否启用 ASR 结果声纹身份匹配 |
| `VOICEPRINT_DB_PATH` | `./data/voiceprints.sqlite3` | SQLite + sqlite-vec 声纹数据库路径 |
| `VOICEPRINT_MATCH_THRESHOLD` | `0.70` | 说话人身份匹配阈值 |
| `VOICEPRINT_DEVICE` | `auto` | 声纹推理设备。默认 `auto`：有 CUDA 用 GPU（分段已限长 ≤30s，峰值约 840MiB），无 CUDA 降级 CPU |

### 显存控制与空闲卸载

CUDA vLLM 运行时可用以下手段压缩常驻显存（全部由 `.env` 驱动）：

```dotenv
QWEN_GPU_MEMORY_UTILIZATION=0.35
QWEN_KV_CACHE_MEMORY_BYTES=1610612736   # 1.5GiB，精确控制 KV cache
QWEN_MAX_NUM_SEQS=5
QWEN_MAX_NUM_BATCHED_TOKENS=2048
QWEN_MAX_MODEL_LEN=4096
QWEN_ENFORCE_EAGER=true
QWEN_WARMUP_FORCED_ALIGNER=false        # aligner 懒加载
```

**forced aligner 懒加载**：aligner 是独立的 vLLM 实例（约 3GB）。`QWEN_WARMUP_FORCED_ALIGNER=false` 时不再于启动阶段预热，只在真正收到 `word_timestamps=true` 的请求时才加载——这是常规启动显存从 16GB 降到 6GB 的关键。

**实测显存**（RTX 4090 24GB，`CUDA_VISIBLE_DEVICES=7`）：

| 场景 | 显存 |
|------|------|
| 基线（空载） | 0.86GB |
| 仅 ASR 引擎（Confucius4-R2T2） | ~6.0GB |
| 完整服务（+ VAD / CAM++ / Paraformer / 标点） | ~8.7GB |
| 显式请求词级时间戳（+ aligner） | ~12GB（峰值） |

**空闲休眠**：设置 `QWEN_IDLE_UNLOAD_TIMEOUT` 后，后台监控会在连续无请求超过该时长时让 vLLM 引擎 `sleep`——权重 offload 到 CPU、丢弃 KV cache，**子进程常驻**，下次请求自动 `wake_up`（实测 0.1~0.4 秒）。正在执行或等待引擎的请求不会被休眠，forced aligner 有独立的活跃计数与空闲计时（忙时不睡）。设置 `0` 可让模型常驻不休眠；本项目生产设 `15`（15 秒）。关闭 `QWEN_ENABLE_SLEEP_MODE` 则回退为旧行为（销毁 EngineCore 子进程，重新加载需 11~21 秒）。

**显存压力休眠**：可用显存低于 `QWEN_LOW_VRAM_THRESHOLD_GB` 且引擎空闲超过 `QWEN_VRAM_PRESSURE_COOLDOWN_S` 时自动让位给其他进程。

**二级彻底释放**：仍空闲超过 `QWEN_SHUTDOWN_AFTER_S`（默认 `600` = 10 分钟）则 shutdown 子进程，把 sleep 残留的 CUDA context（两引擎合计约 1.9GB）也归还——空闲时显存回到最低，下次请求走懒加载（11~21s）。

> **注意**：若不用 `QWEN_KV_CACHE_MEMORY_BYTES` 而只靠 utilization，调得过低会导致 vLLM 启动失败（主引擎低于约 0.16、aligner 低于约 0.12 时，扣除权重后无剩余空间分配 KV cache）。

## API 文档

启动服务后访问：

- Swagger UI: `http://localhost:9101/docs`
- ReDoc: `http://localhost:9101/redoc`

## 相关链接

- **部署指南**: [详细文档](./docs/deployment.md)
- **Qwen3-ASR**: [Qwen3-ASR GitHub](https://github.com/QwenLM/Qwen3-ASR)
- **FunASR**: [FunASR GitHub](https://github.com/alibaba-damo-academy/FunASR)
- **QwenASR**: [QwenASR GitHub](https://github.com/huanglizhuo/QwenASR)

## 许可证

本项目采用 MIT 许可证 - 查看 [LICENSE](../LICENSE) 文件了解详情。

## Star 历史

[![Star History Chart](https://api.star-history.com/svg?repos=Quantatirsk/qwen3-asr&type=Date)](https://star-history.com/#Quantatirsk/qwen3-asr&Date)

## 贡献

欢迎提交 Issue 和 Pull Request 来改进项目!
