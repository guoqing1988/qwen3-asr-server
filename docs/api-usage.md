# 接口用法（API Usage）

本文按「怎么用」组织：每个端点给出用途、鉴权、参数、可直接复制的示例和响应样例。
所有示例基于当前生产模型 **`qwen3-asr-confucius4-r2t2`**（Confucius4-R2T2）。

> 交互式调试：服务启动后打开 `http://<host>:<PORT>/docs`（Swagger UI）或 `/redoc`。
> 下文示例端口用本机 systemd 部署的 `18109`（= `.env` 的 `PORT`）；Docker 部署时改为宿主机映射端口（默认 `9101`）。

## 目录

- [1. 端点总览](#1-端点总览)
- [2. 鉴权](#2-鉴权)
- [3. 离线转写 `POST /v1/audio/transcriptions`](#3-离线转写-post-v1audio-transcriptions)
- [4. 模型列表 `GET /v1/models`](#4-模型列表-get-v1models)
- [5. OpenAI Realtime `WS /v1/realtime`](#5-openai-realtime-ws-v1realtime)
- [6. 声纹注册 `/api/v1/voiceprint-speakers`](#6-声纹注册-apiv1voiceprint-speakers)
- [7. 模型 ID 与别名](#7-模型-id-与别名)
- [8. 错误码](#8-错误码)

## 1. 端点总览

| 端点 | 方法 | 用途 | 返回说话人 | 返回词级时间 |
|------|------|------|-----------|-------------|
| `/v1/audio/transcriptions` | POST | OpenAI Audio API 兼容，文件/URL 转写 | ✅ 分段 | ✅ 可选 |
| `/v1/models` | GET | 列出当前在线离线模型 | — | — |
| `/v1/realtime` | WS | OpenAI Realtime transcription 子集（R2T2） | ✅ 整段标签 | ❌ |
| `/api/v1/voiceprint-speakers` | GET/POST/DELETE | 声纹注册、列表、删除 | — | — |

选型建议：

- **离线录音 / 会议纪要** → `/v1/audio/transcriptions`（说话人分离 + 词级时间戳齐全）
- **已有 OpenAI 客户端** → 同一端点，改 `base_url` 即可
- **实时字幕 / 语音助手** → `/v1/realtime`（带实时说话人标签）
- **需要给说话人起名字** → 先声纹注册，再走离线转写

## 2. 鉴权

三种方式任选，值均为 `.env` 中的 `API_KEY`：

```bash
# 1) Bearer（推荐）
-H "Authorization: Bearer $API_KEY"
# 2) 查询参数（浏览器 WebSocket）
?token=$API_KEY
# 3) 兼容旧客户端的 X-NLS-Token 请求头
-H "X-NLS-Token: $API_KEY"
```

WebSocket 还支持 OpenAI 不安全子协议：`openai-insecure-api-key.<API_KEY>`（握手回 `subprotocol=realtime`）。
未配置 `API_KEY` 时全部放开。

## 3. 离线转写 `POST /v1/audio/transcriptions`

`multipart/form-data`。音频/视频（wav/mp3/m4a/flac/ogg/aac/amr/pcm/webm/mp4/mov/mkv/avi）自动转 16kHz。

### 参数

| 参数 | 类型 | 默认 | 说明 |
|------|------|------|------|
| `file` | 文件 | — | 上传音频/视频；与 `audio_address` 二选一，同时提供时以 `file` 优先 |
| `audio_address` | string | — | 音频/视频 URL（HTTP/HTTPS），`file` 为空时使用 |
| `model` | string | — | 兼容字段，**当前不参与选模型**（始终用已激活模型，见[第 9 节](#9-模型-id-与别名)） |
| `language` | string | 自动 | ISO-639-1，如 `zh`/`en`/`ja`；不传自动检测 |
| `enable_speaker_diarization` | bool | `true` | 说话人分离（Nemotron-3，最多 8 人）；开启时内部自动跑 forced aligner 做归属 |
| `word_timestamps` | bool | `false` | 返回词级时间戳；`ALIGNMENT_MODE=uniform` 时不加载 aligner、按字均分 |
| `response_format` | string | `verbose_json` | `json` / `text` / `srt` / `vtt` / `verbose_json` |
| `prompt` | string | — | 热词/上下文（英文逗号分隔，≤512 字符），与 `ASR_DEFAULT_HOTWORDS` 合并 |
| `temperature`、`timestamp_granularities[]` | — | — | 仅为 OpenAI 兼容保留，传入被忽略 |

### 示例

```bash
# 基本转写（verbose_json：分段 + 说话人 + 时间戳）
curl -X POST "http://127.0.0.1:18109/v1/audio/transcriptions" \
  -H "Authorization: Bearer $API_KEY" \
  -F "file=@meeting.mp3" \
  -F "model=qwen3-asr-confucius4-r2t2" \
  -F "response_format=verbose_json"

# 只要纯文本
curl -X POST "http://127.0.0.1:18109/v1/audio/transcriptions" \
  -H "Authorization: Bearer $API_KEY" \
  -F "file=@meeting.mp3" -F "response_format=text"

# 字幕（SRT，含 [说话人N] 前缀）
curl -X POST "http://127.0.0.1:18109/v1/audio/transcriptions" \
  -H "Authorization: Bearer $API_KEY" \
  -F "file=@meeting.mp3" -F "response_format=srt" -o meeting.srt

# 词级时间戳 + 热词
curl -X POST "http://127.0.0.1:18109/v1/audio/transcriptions" \
  -H "Authorization: Bearer $API_KEY" \
  -F "file=@meeting.mp3" \
  -F "word_timestamps=true" \
  -F "prompt=Bloomberg,周树门,商业周刊" \
  -F "response_format=verbose_json"

# 关闭说话人分离（更快）
curl -X POST "http://127.0.0.1:18109/v1/audio/transcriptions" \
  -H "Authorization: Bearer $API_KEY" \
  -F "file=@meeting.mp3" -F "enable_speaker_diarization=false"

# 转 URL 音频
curl -X POST "http://127.0.0.1:18109/v1/audio/transcriptions" \
  -H "Authorization: Bearer $API_KEY" \
  -F "audio_address=https://media.cdn.vect.one/podcast_demo.mp4" \
  -F "response_format=json"
```

Python（OpenAI SDK，改 `base_url` 即可复用）：

```python
from openai import OpenAI

client = OpenAI(base_url="http://127.0.0.1:18109/v1", api_key="<API_KEY>")

with open("meeting.mp3", "rb") as f:
    result = client.audio.transcriptions.create(
        model="qwen3-asr-confucius4-r2t2",
        file=f,
        response_format="verbose_json",
        extra_body={"word_timestamps": "true", "prompt": "Bloomberg,周树门"},
    )

for seg in result.segments:
    print(f"[{seg.start:6.2f}-{seg.end:6.2f}] {seg.speaker or '未知'}: {seg.text}")
```

### `verbose_json` 响应

```json
{
  "task": "transcribe",
  "language": "zh",
  "duration": 10.0,
  "text": "有几个事情呢，你你那个板给我拿过来，有几个。",
  "segments": [
    {
      "id": 0,
      "seek": 392,
      "start": 3.92,
      "end": 8.32,
      "text": "有几个事情呢，你你那个板给我拿过来，有几个。",
      "tokens": [],
      "temperature": 0.0,
      "avg_logprob": 0.0,
      "compression_ratio": 0.0,
      "no_speech_prob": 0.0,
      "speaker": "说话人1"
    }
  ],
  "words": [{"word": "有", "start": 3.92, "end": 4.13}],
  "word_timestamp_method": "forced_alignment"
}
```

| 字段 | 说明 |
|------|------|
| `segments[].speaker` | 说话人标签；命中已注册声纹时替换为注册的显示名（如 `张三`），关闭分离时不出现 |
| `words`（顶层） | 全局词级时间（相对整条音频），仅 `word_timestamps=true` 时出现 |
| `word_timestamp_method` | `forced_alignment`（声学对齐，默认）或 `uniform_fallback`（按字均分，`ALIGNMENT_MODE=uniform`） |
| `text` | 全文；开启分离时为多段拼接文本 |

> 注意：`/v1/audio/transcriptions` 用 OpenAI 风格的 `speaker`；
> `speaker_id`，且开启 `word_timestamps` 时词级时间放在 `segments[].word_tokens`。

`response_format=text` 直接返回纯文本（`Content-Type: text/plain`），`srt`/`vtt` 返回字幕文件内容。

### 段时长说明（两层上限）

| 层级 | 上限 | 说明 |
|------|------|------|
| 送 ASR 的推理段 | 硬上限 `MAX_SEGMENT_SEC`（默认 60s） | 切分强制生效，恒不超 |
| 响应 `segments[]`（说话人段落） | 60s 软上限 / 120s 硬上限，实际最长约 180s | 同一说话人的连续推理段会合并成段落 |

`segments[]` 中超过 60s 的段是正常现象：段落跨度 >120s 才无条件断开，跨度 >60s 时还要求上一段以
`。！？.!?` 结尾才断；模型在段尾常不给句末标点，于是同一人连续讲话（停顿 ≤1s）会继续合并。
`enable_speaker_diarization=false` 时不做这类合并，段时长更接近推理段（≤60s）。

## 4. 模型列表 `GET /v1/models`

```bash
curl -s http://127.0.0.1:18109/v1/models -H "Authorization: Bearer $API_KEY"
```

```json
{"object": "list", "data": [{"id": "qwen3-asr-confucius4-r2t2", "object": "model", "created": 1792635247, "owned_by": "qwen3-asr"}]}
```

## 5. OpenAI Realtime `WS /v1/realtime`

`ws://<host>:18109/v1/realtime?intent=transcription&token=<API_KEY>`

### 客户端 → 服务端

| 事件 | 说明 |
|------|------|
| `session.update` | 设置 `format.rate`（16000/24000）、`transcription.prompt/language/keywords`、`turn_detection`（`null` 手动 commit 或 `{"type":"server_vad"}`） |
| `input_audio_buffer.append` | `{"audio": "<base64 PCM16>"}`，单次 ≤10s |
| `input_audio_buffer.commit` | 手动模式下提交并定稿当前段 |
| `input_audio_buffer.clear` | 丢弃未提交音频 |

### 服务端 → 客户端

| 事件 | 说明 |
|------|------|
| `session.created` / `session.updated` | 含 `session.speaker_diarization`（是否启用了实时说话人标签） |
| `conversation.item.input_audio_transcription.delta` | 增量文本；流式阶段**关闭 ITN**，保证 delta 单调 |
| `conversation.item.input_audio_transcription.completed` | 定稿：`transcript`（已 ITN + 标点 + two-pass 重转）、`speaker`（`说话人N` 或 `null`） |
| `input_audio_buffer.committed` / `cleared` | 缓冲确认 |
| `error` | `{"error": {"code": ..., "message": ...}}` |

`server_vad` 模式：句尾静音自动落定并重建段；连续讲话按 `QWEN_STREAM_SEGMENT_MAX_SEC`（60s）兜底。
说话人标签与 ASR 并行推理，**滞后约 1s**；关闭 `REALTIME_SPEAKER_LABELS` 时 `speaker` 恒为 `null`。

### Python 示例（带说话人）

```python
import asyncio, base64, json
import numpy as np
import soundfile as sf
import websockets

KEY = "<API_KEY>"
URL = f"ws://127.0.0.1:18109/v1/realtime?intent=transcription&token={KEY}"

async def main():
    audio, sr = sf.read("speech.wav", dtype="float32")
    pcm = (np.clip(audio, -1, 1) * 32768).astype("<i2").tobytes()
    async with websockets.connect(URL, subprotocols=["realtime"]) as ws:
        print(json.loads(await ws.recv())["type"])  # session.created

        await ws.send(json.dumps({"type": "session.update", "session": {
            "audio": {
                "input": {
                    "format": {"type": "audio/pcm", "rate": 16000},
                    "transcription": {"model": "qwen3-asr-confucius4-r2t2",
                                      "prompt": "Bloomberg,周树门", "language": "zh"},
                    "turn_detection": {"type": "server_vad"},
                }
            }
        }}))

        async def reader():
            async for raw in ws:
                ev = json.loads(raw)
                if ev["type"].endswith("delta"):
                    print(ev["delta"], end="", flush=True)
                elif ev["type"].endswith("completed"):
                    print(f"\n[{ev.get('speaker')}] {ev['transcript']}")

        task = asyncio.create_task(reader())
        frame = 3200  # 200ms @16k
        for off in range(0, len(pcm), frame):
            await ws.send(json.dumps({
                "type": "input_audio_buffer.append",
                "audio": base64.b64encode(pcm[off:off + frame]).decode(),
            }))
            await asyncio.sleep(0.2)  # 按实时速率发送效果最好
        await ws.send(json.dumps({"type": "input_audio_buffer.commit"}))
        await asyncio.sleep(5)
        task.cancel()

asyncio.run(main())
```

> **发送速率**：建议按 1x 实时速率发送。一次性猛灌后立即停发时，末段说话人标签会用
> `flush` 把尾部音频按最后一块推理一次（精度略降），最大等待 `REALTIME_SPEAKER_LABEL_TIMEOUT`。

## 6. 声纹注册 `/api/v1/voiceprint-speakers`

注册后，离线转写中匹配到的说话人 `speaker_id` 会被替换为注册显示名（如 `张三`）。

```bash
# 注册（一个或多个单人样本，建议每段 ≥2s）
curl -X POST "http://127.0.0.1:18109/api/v1/voiceprint-speakers" \
  -H "Authorization: Bearer $API_KEY" \
  -F "display_name=张三" \
  -F "description=产品经理" \
  -F "file=@zhangsan_1.wav" -F "file=@zhangsan_2.wav"
# → {"speaker_id": "...", "display_name": "张三", "voiceprint_id": "...", "voiceprint_ids": [...], "voiceprint_count": 2}

# 追加样本
curl -X POST "http://127.0.0.1:18109/api/v1/voiceprint-speakers/<speaker_id>/samples" \
  -H "Authorization: Bearer $API_KEY" -F "file=@zhangsan_3.wav"

# 列表 / 删除
curl -s "http://127.0.0.1:18109/api/v1/voiceprint-speakers" -H "Authorization: Bearer $API_KEY"
curl -X DELETE "http://127.0.0.1:18109/api/v1/voiceprint-speakers/<speaker_id>" \
  -H "Authorization: Bearer $API_KEY"
```

匹配不确定时保留 `说话人N` 原标签；声纹库为 sqlite-vec 本地存储。

## 7. 模型 ID 与别名

`/v1/models` 当前返回 `qwen3-asr-confucius4-r2t2`（Confucius4-R2T2，vLLM bf16 或 llama.cpp Q8，由 `ASR_ENGINE` 决定）。

- **`/v1/audio/transcriptions`**：`model` 字段不参与路由，始终使用当前激活模型（传任意值都能跑，建议传 `/v1/models` 的 id）
- **`/v1/realtime`**：`transcription.model` 仅回显；服务端固定用当前激活的 Qwen3-ASR
- **`QWEN3_ASR_MODEL`**：切换/指定模型的唯一开关

接受的历史别名（归一化后指向同一模型）：

| 别名 | 归一化为 |
|------|---------|
| `qwen3-asr` | 当前激活的 Qwen3-ASR |
| `qwen3-asr-confucius4-r2t2`、`confucius4-r2t2`、`confucius4`、`r2t2`、`netease-youdao/confucius4-r2t2` | `qwen3-asr-confucius4-r2t2` |
| `qwen3-asr-0.6b`、`0.6b`、`0.6`、`qwen/qwen3-asr-0.6b` | `qwen3-asr-0.6b`（需该模型已部署） |

## 8. 错误码

统一 JSON：

```json
{"error_code": "INVALID_PARAMETER", "message": "Invalid audio file", "task_id": "...", "timestamp": "2026-10-20T10:00:00Z", "details": {}}
```

| HTTP | error_code | 场景 |
|------|-----------|------|
| 400 | `INVALID_PARAMETER` | 音频损坏/为空、参数非法、采样率不支持 |
| 400 | `AUTHENTICATION_FAILED` | 声纹接口鉴权失败 |
| 401 | `AUTHENTICATION_FAILED` | API Key 错误或缺失 |
| 500 | `DEFAULT_SERVER_ERROR` | 模型推理失败、说话人分离不可用等 |

WebSocket 侧错误以 `error` 事件返回；未授权/协议不符时直接 `close(1008)`。
