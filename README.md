<div align="center">

<h1>Qwen3-ASR</h1>
<h3>开箱即用的本地私有化部署语音识别服务</h3>

以 [Qwen3-ASR](https://github.com/QwenLM/Qwen3-ASR) 为核心的语音识别 API 服务，提供 CUDA vLLM 与 CPU Rust 两种后端，兼容阿里云语音 API 和 OpenAI Audio API，并保留 Paraformer realtime WebSocket 能力。

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

### 1. Docker 部署(推荐)

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
| GPU（默认） | `uv sync` | 同步根目录 [pyproject.toml](/opt/qwen3-asr/pyproject.toml) 和 [uv.lock](/opt/qwen3-asr/uv.lock) 到 `.venv`，包含 CUDA 12.8/cu128 `torch/torchaudio/torchvision` |
| CPU（特化） | `./scripts/sync_cpu_env.sh` | 同步 [environments/cpu/pyproject.toml](/opt/qwen3-asr/environments/cpu/pyproject.toml) 对应的 CPU lock 到 `.venv` |

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
- `Linux + CUDA` 使用官方 `vLLM`
- `Linux + CPU` 使用 vendored `QwenASR` Rust
- `macOS / Apple Silicon` 也使用 vendored `QwenASR` Rust
- macOS / Apple Silicon 默认总是 `qwen3-asr-0.6b`
- 在 macOS 上，只有设置 `QWEN3_ASR_MODEL=qwen3-asr-1.7b` 时才会使用 `qwen3-asr-1.7b`
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
| `prompt`                     | string | -                     | 提示文本（保留兼容）                  |
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
| `/ws/v1/asr/qwen`       | WebSocket | Qwen3-ASR 流式识别 |

**请求参数:**

| 参数                           | 类型   | 默认值             | 说明                                  |
| ------------------------------ | ------ | ------------------ | ------------------------------------- |
| `audio_address`              | string | `https://media.cdn.vect.one/podcast_demo.mp4`（文档示例） | 音频/视频 URL（可选；若同时上传内容则忽略） |
| `sample_rate`                | int    | `16000`          | 采样率                                |
| `enable_speaker_diarization` | bool   | `true`           | 启用说话人分离                        |
| `word_timestamps`            | bool   | `false`          | 返回后端支持的字词级时间戳；Qwen CUDA vLLM 与 CPU Rust 在启用时会自动调用 forced aligner |
| `vocabulary_id`              | string | -                  | 无权重热词上下文（如：`词1 词2`）。**Deprecated：** 数字权重不受支持，传入时会被忽略。 |

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

`/ws/v1/asr` 下提供三个 WebSocket 端点：

| 端点 | 后端 | 说明 |
|------|------|------|
| `/ws/v1/asr` | FunASR realtime（Paraformer） | 阿里云协议兼容，中文流式 + 实时标点 |
| `/ws/v1/asr/funasr` | FunASR realtime | 上述端点的向后兼容别名 |
| `/ws/v1/asr/qwen` | Qwen3-ASR（vLLM） | Qwen3 流式，多语言（中/英/日等） |

三个端点都是**实时语音转录**——音频增量发送，说话的同时返回增量结果。
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

**Qwen3 流式协议**（`/ws/v1/asr/qwen`）——JSON 控制消息 + 二进制音频帧：

1. 连接后发送 `start`：
   ```json
   {"type": "start", "payload": {
     "format": "pcm",              // "pcm" 或 "wav"
     "sample_rate": 16000,
     "language": "zh",             // 可选，自动检测
     "context": "",                // 热词上下文
     "enable_inverse_text_normalization": true,
     "chunk_size_sec": 2.0,
     "unfixed_chunk_num": 2,
     "unfixed_token_num": 5
   }}
   ```
2. 服务器回 `started`（带生效参数）
3. 持续发送二进制音频帧（16kHz int16 PCM 或 WAV 字节）
4. 服务器增量返回 `segment_start` / `segment_end`，`segment_end` 携带累计
   全文（`result.text`）和已确认文本列表
5. 发送 `{"type": "stop"}` 结束；错误以 `{"type": "error", ...}` 返回

Python 示例：

```python
import asyncio, json, websockets

async def main():
    async with websockets.connect(
        "ws://localhost:9101/ws/v1/asr/qwen?token=YOUR_API_KEY"
    ) as ws:
        await ws.send(json.dumps({"type": "start", "payload": {"format": "pcm"}}))
        with open("audio.wav", "rb") as f:
            f.read(44)  # 跳过 WAV 头
            while chunk := f.read(3200):  # 16kHz int16，0.1s 一帧
                await ws.send(chunk)
        await ws.send(json.dumps({"type": "stop"}))
        async for raw in ws:
            msg = json.loads(raw)
            if msg.get("type") == "segment_end":
                print(msg["result"]["text"])

asyncio.run(main())
```

**FunASR 流式协议**（`/ws/v1/asr` 与 `/ws/v1/asr/funasr`）——阿里云 NLS
风格：先发 JSON 帧（`{"header": {"message_id": "...", "task": "start"},
"payload": {...}}`），随后发二进制音频，最后发 `task: "stop"`。服务器事件
顺序：`TranscriptionStarted` → `SentenceBegin` → `TranscriptionResultChanged`
（中间结果）→ `SentenceEnd` → `TranscriptionCompleted`。payload 支持
`format`、`sample_rate`、`enable_intermediate_result`、`enable_punctuation`、
`enable_itn` 等参数。

**实时行为**（Qwen3 路径）：真正的实时识别。每个音频 chunk（默认 2 秒）
增量解码后立即推送 `type: "result"`（`is_partial: true`），客户端可渲染
实时字幕；一句话结束（静音 3 秒或单段上限）时推送 `segment_end` 确认文本。
延迟调优：

| 参数 | 默认 | 作用 |
|------|------|------|
| `chunk_size_sec` | 2.0 | 出字延迟；调小（如 1.0）更实时 |
| `unfixed_token_num` | 5 | 句尾回滚修正 token 数 |
| `unfixed_chunk_num` | 2 | 允许截断前的预热 chunk 数 |

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
2. **贪婪合并** - 累积语音段，确保每段不超过 `MAX_SEGMENT_SEC`（默认60秒）
3. **静音切分** - 语音段间静音超过3秒时强制切分，避免包含过长静音
4. **批处理推理** - 多片段并行处理，GPU 模式下性能提升 2-3 倍

### WebSocket 流式识别限制

**FunASR 模型限制**（使用 `/ws/v1/asr` 或 `/ws/v1/asr/funasr`）：
- ✅ 实时语音识别、低延迟
- ✅ 字句级时间戳
- ❌ **词级时间戳**（FunASR realtime 路径未实现）
- ❌ **置信度分数**（未实现）
- 音频入口固定保留最多 10 秒待处理数据；队列满时会对 WebSocket 发送端施加背压，不会静默丢弃音频。客户端应持续读取识别事件并增量发送。
- 该路径使用 Paraformer，离线接口使用 Qwen3-ASR；应在同一路径内比较延迟，不能直接比较两者总吞吐。

**Qwen3-ASR 流式**（使用 `/ws/v1/asr/qwen`）：
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
| `qwen3-asr-1.7b`   | Qwen3-ASR 1.7B    | 高性能多语言 ASR；CUDA 使用 vLLM | 离线/实时 |
| `qwen3-asr-0.6b`   | Qwen3-ASR 0.6B    | 轻量版多语言 ASR；CUDA 使用 vLLM，CPU/macOS 使用 Rust backend | 离线/实时 |

## 仅实时能力

| 能力 ID | 运行时 | 说明 |
| ------- | ------ | ---- |
| `paraformer-large` | FunASR realtime | 中文 WebSocket 实时识别栈，包含实时标点链路 |

**运行时选择:**
- **显存 >= 32GB**: 选择 `qwen3-asr-1.7b`
- **显存 < 32GB**: 选择 `qwen3-asr-0.6b`
- **无 CUDA**: 选择基于 vendored Rust 的 `qwen3-asr-0.6b`
- **macOS / Apple Silicon**: 无论内存大小多少，默认都选择 `qwen3-asr-0.6b`
- **环境变量覆盖**: 设置 `QWEN3_ASR_MODEL=qwen3-asr-1.7b` 或 `QWEN3_ASR_MODEL=qwen3-asr-0.6b` 可跳过自动选择
- `paraformer-large` 实时能力始终为 WebSocket 流式准备

启动时会先检测当前运行计划所需模型；如果本地缓存缺失，会自动下载。离线部署可显式设置 `HF_HUB_OFFLINE=1` 并提前准备模型缓存。

## 模型清单

所有运行时模型缓存在项目 `models/` 目录下并挂载进容器，分为两个渠道：

| 模型 | 大小 | 渠道与路径 | 用途 |
|------|------|-----------|------|
| Qwen3-ASR 1.7B | ~4.4GB | HF：`models/huggingface/hub/models--Qwen--Qwen3-ASR-1.7B` | 离线转写、`/ws/v1/asr/qwen` 流式（多语言） |
| Qwen3-ForcedAligner 0.6B | ~1.8GB | HF：`models/huggingface/hub/models--Qwen--Qwen3-ForcedAligner-0.6B` | 词级时间戳（离线 `word_timestamps=true`） |
| Paraformer Large | ~849MB | MS：`models/modelscope/hub/models/iic/speech_paraformer-large_...` | `/ws/v1/asr` + `/ws/v1/asr/funasr`（中文实时流式） |
| 标点（离线） | ~300MB | MS：`models/modelscope/hub/models/iic/punc_ct-transformer_zh-cn-common-vocab272727-pytorch` | 离线/流式标点 |
| 标点（实时） | ~300MB | MS：`models/modelscope/hub/models/iic/punc_ct-transformer_..._vad_realtime-...` | FunASR 流式路径实时标点 |
| CAM++ 说话人分离 | ~300MB | MS：`models/modelscope/hub/models/iic/speech_campplus_speaker-diarization_common` | 说话人分离（离线） |
| CAM++ 声纹验证 | ~90MB | MS：`models/modelscope/hub/models/damo/speech_campplus_sv_zh-cn_16k-common` | 说话人聚类 + 声纹匹配 |
| CAM++ SCL | ~90MB | MS：`models/modelscope/hub/models/damo/speech_campplus-transformer_scl_zh-cn_16k-common` | 说话人聚类 |
| VAD（FSMN） | ~69MB | MS：`models/modelscope/hub/models/damo/speech_fsmn_vad_zh-cn-16k-common-pytorch` | 语音活动检测、音频切分 |

**使用场景指引**：

- **中文实时流式**（低延迟 + 实时标点）：FunASR 路径使用 Paraformer Large + 实时标点模型（ModelScope 渠道）
- **多语言 / 离线 / 词级时间戳**：Qwen3-ASR 1.7B + ForcedAligner（HuggingFace 渠道）
- **说话人分离与声纹**（仅离线接口）：CAM++ 系列模型（ModelScope 渠道）
- **下载行为**：启动时自动补下缺失模型（ModelScope 模型走 `modelscope.cn`，Qwen 模型走 HuggingFace 或 `HF_ENDPOINT` 镜像）；完全离线部署先执行一次 `./scripts/prepare-models.sh` 并拷贝 `models/` 目录

**接口 × 模型对应**：

| 接口 | 模型 |
|------|------|
| `POST /v1/audio/transcriptions` | qwen3-asr-1.7b |
| `POST /stream/v1/asr` | qwen3-asr-1.7b |
| `GET /v1/models`、`health`、`/stream/v1/asr/models` | 只读查询，不跑推理 |
| `/ws/v1/asr/qwen` | qwen3-asr-1.7b |
| `/ws/v1/asr`、`/ws/v1/asr/funasr` | paraformer-large（中文实时） |

所有 HTTP 转写接口共用默认离线模型（`qwen3-asr-1.7b`，可用
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
| `NGINX_PORT`                    | `9101`     | Docker Compose 映射到宿主机的端口             |
| `API_KEY`                       | -           | API 认证密钥（可选，未配置时无需认证）        |
| `CUDA_VISIBLE_DEVICES`          | `0`         | GPU Compose 可见设备列表，多卡用 `0,1,2,3`   |
| `QWEN3_ASR_MODEL`               | 自动选择     | 强制选择 `qwen3-asr-1.7b` 或 `qwen3-asr-0.6b` |
| `HF_HUB_OFFLINE`                | `0`         | 离线部署且模型已准备好时设为 `1`              |
| `HF_ENDPOINT`                   | -           | 在线镜像站；离线部署不要设置                  |
| `QWEN_GPU_MEMORY_UTILIZATION`   | 自动计算     | 主 ASR 模型的 vLLM 显存利用率（0.0-1.0）。默认按 `12GB / 总显存` 自动计算；调低可减少常驻显存 |
| `QWEN_FORCE_ALIGNER_GPU_MEMORY_UTILIZATION` | 继承主模型 | forced aligner（词级时间戳）vLLM 实例的显存利用率（0.0-1.0） |
| `QWEN_IDLE_UNLOAD_TIMEOUT`      | `300`       | 无请求超过该秒数后卸载 vLLM 引擎释放显存；`0` 禁用空闲卸载 |
| `VOICEPRINT_ENABLED`            | `true`      | 是否启用 ASR 结果声纹身份匹配 |
| `VOICEPRINT_DB_PATH`            | `./data/voiceprints.sqlite3` | SQLite + sqlite-vec 声纹数据库路径 |
| `VOICEPRINT_MATCH_THRESHOLD`    | `0.70`      | 说话人身份匹配阈值 |

### 显存控制与空闲卸载

CUDA vLLM 运行时按显存利用率比例预分配显存（默认 `12GB / 总显存`，上限 0.95）。在 48GB 显卡上，主引擎约预留 12GB，forced aligner 还会再预留一份。要降低常驻显存，可在 `.env` 中调低比例：

```dotenv
QWEN_GPU_MEMORY_UTILIZATION=0.20
QWEN_FORCE_ALIGNER_GPU_MEMORY_UTILIZATION=0.15
```

**空闲卸载**：设置 `QWEN_IDLE_UNLOAD_TIMEOUT`（默认 `300` 秒）后，后台监控会在连续无请求超过该时长时卸载 vLLM 引擎（主模型 + forced aligner），终止其 EngineCore 子进程并将显存归还系统。下一次请求会自动触发懒加载重建，约需 30-60 秒。正在执行或排队等待引擎的请求不会被中途卸载，监控仅在引擎完全空闲时触发。设置 `QWEN_IDLE_UNLOAD_TIMEOUT=0` 可让模型常驻不卸载。

> **注意**：显存利用率调得过低会导致 vLLM 启动失败（例如 48GB 卡上主引擎低于约 0.16、aligner 低于约 0.12 时，扣除权重后无剩余空间分配 KV cache）。

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
