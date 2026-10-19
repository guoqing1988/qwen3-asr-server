# 标准 OpenAI Realtime 转录接口（WebSocket）

日期：2026-10-19

## 背景

上游 AsrServe 已把实时转录改为标准 OpenAI Realtime 协议（`ws /v1/realtime`，`intent=transcription`），
本项目此前用的是自定义协议 `ws /ws/v1/asr/qwen`（`start` / 二进制音频 / `stop`，返回自定义 `result` /
`segment_end` 事件）。目标：**与上游对齐，走标准 OpenAI Realtime 协议，废弃自定义 qwen 端点**，
识别引擎（vLLM R2T2 / llama.cpp Q8）与返回内容质量保持不变。

## 目标

1. 新增 `ws /v1/realtime`：OpenAI Realtime **transcription** 子集，纯 WebSocket（无 HTTP/SSE 版本）。
2. 复用现有流式引擎（`Qwen3ASREngine` 的 `init_streaming_state/streaming_transcribe/finish_streaming_transcribe`）
   与 FireRedVAD 断句，不复制识别逻辑。
3. 废弃 `ws /ws/v1/asr/qwen`：保留路由但返回 410 并提示新端点（避免老客户端静默失败），一个版本周期后删除。
4. 保留 `ws /ws/v1/asr`、`/ws/v1/asr/funasr`（FunASR 兼容）不动。

## 协议设计

### 鉴权

- `Authorization: Bearer <API_KEY>`（浏览器无法设头时用 query `?token=`）；
- 兼容 OpenAI 子协议：`openai-insecure-api-key.<key>`，服务端回 `Sec-WebSocket-Protocol: realtime`。

### 客户端事件

| type | 处理 |
|---|---|
| `session.update` | 解析 `session.audio.input.transcription`（model/prompt/language/languages/keywords）与 `turn_detection`，重建流式状态 |
| `input_audio_buffer.append` | base64 → PCM16 → 重采样到 16k（soxr，仅 rate≠16000 时）→ 进入流式管线 |
| `input_audio_buffer.commit` | 落定当前段，发 `conversation.item.input_audio_transcription.completed` |
| `input_audio_buffer.clear` | 丢弃未提交缓冲与当前段状态 |

### 服务端事件

| type | 触发 |
|---|---|
| `session.created` / `session.updated` | 连接建立 / `session.update` 生效 |
| `conversation.item.input_audio_transcription.delta` | 当前段文本变化（增量文本 = 最新全文，遵循 append-only） |
| `conversation.item.input_audio_transcription.completed` | 段定稿（VAD 静音自动提交或显式 commit） |
| `conversation.item.input_audio_transcription.failed` | 识别异常（该段丢弃） |
| `input_audio_buffer.committed` / `.cleared` | 对应客户端事件确认 |
| `error` | 协议/参数/服务错误（含 `type`、`code`、`message`） |

### turn_detection

- `null`（默认按上游）：客户端拥有段边界，仅 `commit` 落定；
- `{"type":"server_vad"}`：沿用现有 FireRedVAD 静音自动落定（保留当前实时体验）。

## 实现

| 文件 | 内容 |
|---|---|
| `app/services/asr/openai_realtime.py`（新） | 配置模型（pydantic，参考上游 `openai_protocol.py` 精简）+ `OpenAIRealtimeSession`（事件循环、重采样、段管理、事件发送） |
| `app/api/v1/openai_realtime.py`（新） | `@router.websocket("/v1/realtime")`，鉴权后交给 session |
| `app/main.py` | 注册新 router |
| `app/api/v1/websocket_asr.py` | qwen 端点改为 410 + 提示 |
| `app/services/qwen3_websocket_asr.py` | 抽取可复用的段处理（`_finalize_segment` 等）供网关调用，不改行为 |

复用点（不改语义）：`Qwen3ASREngine` 流式接口、`create_stream_vad()`、`offline two-pass`
（`QWEN_STREAM_FINAL_PASS`）、热词合并 `merge_hotwords`。

### 段状态

- 段内累积音频（two-pass 用）、`streaming_state`、VAD 静音计数、`segment_index`；
- 落定：`finish_streaming_transcribe` → 可选 two-pass 重转 → `completed`；
- `clear`：丢弃缓冲与状态，`segment_index` 不变（已提交段保留）。

## 测试

1. 单测 `tests/test_openai_realtime.py`：配置校验（pcm16/rate/language+keywords 冲突）、事件分派、
   pcm16 重采样、commit/clear、delta/completed 序列（fake engine）。
2. 集成：真实 WS 客户端（16k 与 24k PCM）推 10s 音频，校验 `session.created` → `delta*` →
   `completed`；显式 commit 与 `server_vad` 两种模式；`/ws/v1/asr/qwen` 返回 410。
3. 真实音频回归：用 `/home/ubuntu/realtime_10min.wav` 推 60s，确认文本与旧端点一致、0 回撤。

## 文档

- `README.md`：接口表新增 `/v1/realtime`，标注 `/ws/v1/asr/qwen` 废弃；
- `docs/realtime.md`（新或补充）：事件示例（Python/JS 客户端）；
- `CLAUDE.md`：流式入口说明；
- `docs/troubleshooting.md`：常见协议错误（未 base64、rate 不支持、未 session.update）。

## 风险与回滚

- 新端点独立，旧 qwen 端点保留（410 提示）→ 回滚只需恢复旧路由；
- 引擎/显存路径不变，Q8 与 vLLM 均适用；
- 24k 输入用 soxr 重采样，若不可用回退线性插值（记录告警）。

---

## 实施结果（2026-10-19 完成）

- 新增 `app/services/asr/openai_realtime.py`（会话/事件/重采样/段状态/VAD 落定/two-pass）
  与 `app/api/v1/realtime.py`（`ws /v1/realtime`，三种鉴权 + `intent` 校验），
  `app/api/v1/__init__.py` 注册。
- `/ws/v1/asr/qwen` 改为废弃：回 `endpoint_deprecated` 后 `close(1008)`；`/ws/v1/asr`、
  `/ws/v1/asr/funasr` 未动。
- 协议细节落地：16k 直通 / 24k 走 soxr 流式重采样；流式 partial 关闭 ITN 保证 delta 单调，
  `.completed` 做 ITN + two-pass；`server_vad` 静音自动落定 + `QWEN_STREAM_SEGMENT_MAX_SEC`
  强制落定；单次 append ≤10s 校验。
- 单测 `tests/test_openai_realtime.py` 15 项；全量 `pytest tests/ -q` → **167 passed**。
- 生产端到端（Q8，18109）：commit 模式 10s（16k/24k）与 `server_vad` 90s 真实音频均正常
  （90s → 6 段 completed / 198 delta）；错误 token → HTTP 403；子协议鉴权 → `realtime`；
  废弃端点回 `endpoint_deprecated` + 1008。
- 文档：README（端点表 + 协议示例 + `QWEN_STREAM_*` 参数表）、CLAUDE.md（实时接口段）、
  troubleshooting 第 14~17 条。
