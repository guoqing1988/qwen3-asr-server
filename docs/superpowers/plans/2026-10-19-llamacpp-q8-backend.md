# llama.cpp Q8 后端（ASR_ENGINE=vllm|llamacpp）实施计划

## 目标

在**不改变任何对外 API 输入/输出结构**的前提下，新增 llama.cpp Q8 推理后端：

- `ASR_ENGINE=vllm`（默认）：保持现状（vLLM bf16）。
- `ASR_ENGINE=llamacpp`：使用官方 Q8 GGUF（`/data1/models/Confucius4-R2T2-GGUF`）+ 本机 `llama-server`。
- 词级时间戳默认关闭（`word_timestamps=false`）；显式开启时仍返回相同的 `words` 结构（复用 `Qwen3-ForcedAligner-0.6B`）。
- 离线、长音频、说话人分离、声纹、WebSocket 流式、two-pass 全部保持可用。

## 实测依据

| 项目 | 结果 |
|---|---|
| Q8 显存（4090，含 mmproj） | 3.6~4.2GB（vLLM bf16 为 8.3GB） |
| 40s 英文离线 | 0.62s，`gpt` 正确 |
| 流式每轮（20s 累积音频 + 2 token） | 0.088s（160ms 预算内） |
| llama.cpp 单次音频上限 | 约 56~60s → 后端内部按 ≤25s 再切段 |
| vLLM 直接加载该 GGUF | 不支持（`architecture qwen3vl is not supported yet`） |

## 设计

### 新文件 `app/services/asr/qwen3_llamacpp.py`

`LlamaCppQ8Backend`，方法集与 `Qwen3VLLMBackend` 对齐：

- 进程管理：`llama-server` 子进程（`-m Q8.gguf --mmproj ... -ngl 99 -c 8192 --parallel N`），启动后轮询 `/health`，`shutdown()` 终止。
- 离线：`POST /v1/chat/completions`，音频以 base64 WAV（`input_audio`）传入；system=热词 context；强制语言用 assistant prefix `language X<asr_text>`。
- 长段保护：内部把 >25s 音频切片逐段识别后拼接（规避 ~60s 音频编码上限）。
- 流式：每 chunk 累积音频 + prefix 回滚（字符级）+ `max_tokens=2`（首轮 4），append-only 校验；`finish_stream` 用完整音频重解码收尾。
- 词级时间戳：Q8 后端内独立加载 `Qwen3-ForcedAligner-0.6B`（vLLM pooling 实例），返回与 vLLM 相同的 `WordToken`。
- `unload/shutdown`、aligner `sleep/wake`。

### 接入点

- `app/core/config.py`：`ASR_ENGINE`、`QWEN_LLAMACPP_*`。
- `app/services/asr/qwen3_engine.py`：
  - `_select_backend()` 增加 `llamacpp`；
  - `transcribe_file` / `transcribe_array` / `transcribe_file_with_vad` / `_transcribe_batch` / 流式三件套 / `unload` / aligner 生命周期增加分支；
  - 长音频与说话人分离继续走 `BaseASREngine.transcribe_long_audio`，无需改动。
- `models.json` / 路由 / API 层不动。

## 测试

- 单元：输出解析、长段切片、流式回滚与稳定性、backend 选择（mock HTTP）。
- 集成（GPU）：`ASR_ENGINE=llamacpp` 临时实例，测离线热词、长音频、流式、`word_timestamps=true` 结构、说话人分离。
- 回归：`pytest tests/ -q`，并确保默认 `vllm` 行为不变。

## 风险与回滚

- llama-server 子进程异常：启动失败自动重试（3 次），失败时抛原有错误类型，不影响 vllm 路径。
- 流式稳定性：append-only 校验失败时保留上一轮已确认文本，不中断连接。
- 回滚：`ASR_ENGINE=vllm`（默认）即回到现状；不删除任何现有代码路径。

## 实施结果（已完成）

### 代码

- `app/services/asr/qwen3_llamacpp.py`：`LlamaCppQ8Backend` + `Q8StreamingState`
  - 进程管理（健康检查、240s 启动等待、`shutdown` 终止）
  - 离线/`transcribe_array`/`transcribe_batch`/`transcribe_raw`，>25s 自动切段
  - 流式：首轮 0.32s，之后每 1s 用累积音频全量重解码（保证文本不倒退）
  - 词级时间戳：独立加载 forced aligner，util 保底 0.35，仍不足则 +0.1/+0.2 重试
- `app/services/asr/qwen3_engine.py`：`llamacpp` 分支接入 `transcribe_*`、批量、流式、unload、aligner 生命周期
- `app/core/config.py` + `.env.example`：`ASR_ENGINE`、`QWEN_LLAMACPP_*`
- 文档：`docs/deployment.md`（Q8 部署/回滚）、`CLAUDE.md`

### 实测（RTX 4090）

开发阶段（GPU3，临时实例）：

| 场景 | 结果 |
|---|---|
| 离线 40s 英文（热词 GPT） | 1.5s，`gpt` 正确 |
| 离线 80s（内部 25s 切段） | 1.6s |
| 流式 20s（160ms 帧，每 1s 重解码） | 总耗时 3.5s，每轮约 0.3s，文本持续增长不倒退 |
| 词级时间戳（10s） | 结构一致，17 words，时间合理 |
| 说话人分离（52s 对话） | 2 段，说话人1/2 正确，走 `transcribe_long_audio` + `_transcribe_batch` |
| ASR 显存 | llama-server 约 4.1GB（vLLM bf16 约 8.3GB） |

生产切换验证（18109，`ASR_ENGINE=llamacpp`，真实 HTTP/WS）：

| 场景 | 结果 |
|---|---|
| `/v1/models` | 200，`qwen3-asr-confucius4-r2t2` |
| 离线 10s（首请求，含 llama-server 启动） | 3.19s，热词 `gpt` 正确 |
| 离线 40s | 1.20s，`speaker=说话人1`，`segments` 正常 |
| 词级时间戳（10s） | 26.96s（首次含 aligner 加载），17 words |
| 流式 40s（160ms 帧） | 首字 3.68s（前 4s 静音）、3 段、`full_text` 正确 |
| 显存 | ASR 4.1GB；词级峰值约 10.6GB，空闲 25s 后回落至 4.5GB |
| 日志 | 0 ERROR |

### 切换过程中修复的问题

1. `Qwen3ASREngine.supports_realtime` 未包含 `llamacpp`，WebSocket 连接被拒绝 → 已加入。
2. 音频开头长静音时 FireRedVAD 立即报句尾，产生 <1s 的极短段并被误识别（`ttranscribe and gbt`）→ 现在段内有效语音 ≥0.8s 才落定，vLLM 路径同样受益。
3. Q8 aligner 沿用 vLLM 的 `gpu_memory_utilization=0.2` 会因 KV cache 不足空等约 90s → 保底 0.35。
4. 流式 append-only prefix 在 llama.cpp 上不稳定（英文单词被字符级回滚切断）→ 先临时改为每 1s 全量重解码。
5. 全量重解码导致 partial 文本回撤（用户实测 `写了一个信息的媒` → `又写了一个信息…`）→ 移植 T8 `StreamingSession` 协议改为 **token 级 append-only**（`QWEN_LLAMACPP_STREAM_MODE=append`）：标签独立维护 + 正文回滚 1 token + `stable` 落后候选 1 token + 回撤候选整轮丢弃 + 首轮静音继续累积（≤6s）。实测 3m44s/16 段/158 partial/0 回撤，每帧约 200ms；tokenizer 用 `QWEN_LLAMACPP_TOKENIZER_PATH`（HF 原目录）。

### 已知差异

- Q8 质量略低于 bf16（示例音频出现 `tendbreak`、`fifty seven`），热词与专名仍生效；
- 流式默认 append-only（160ms chunk、token 级续写、partial 只增不减），`redecode` 模式可回退为 1s 全量重解码；
- `word_timestamps=true` 时 aligner 与 llama-server 同卡，峰值约 11GB，靠对齐后 sleep 回落。


### 回归测试（2026-10-18 晚）

单测：`.venv/bin/python -m pytest tests/ -q` → **152 passed**。

生产接口（18109，Q8）回归 14 项，全部 PASS：

| 用例 | 结果 |
|---|---|
| `/v1/models` | 200 |
| OpenAI 10s / 40s | 200，文本与 segments 正常 |
| 热词（`prompt` / `vocabulary_id`） | 200 |
| `word_timestamps=true` | 首次 29.4s（含 aligner 加载）、连续第二次 21.4s（修复后）、第三次 0.7s，顶层 `words[]` 32 项 |
| `enable_itn=false` / `enable_speaker_diarization=false` | 200 |
| `response_format=text/verbose_json/srt` | 200 |
| 阿里云 `/stream/v1/asr` | 200 |
| 并发 2 路 10s | 200，wall 0.57s |
| 非法音频 | 修复后 400（原 500） |

流式（10 分钟音频、append-only）：`language` 残留 0；partial 前缀变化仅来自 ITN 数字规范化（`一→1`、`七→70`），非识别回撤。

修复的问题：

1. **aligner 休眠后 `encode` 挂起**：`_get_aligner()` / `align_transcript()` 增加唤醒（第二次 `word_timestamps` 曾卡死 >300s）。
2. **非法音频返回 500**：`normalize_audio_for_asr` / `convert_audio_to_wav` 改为 `InvalidParameterException`，OpenAI 兼容接口新增 `APIException` 状态码映射 → 400。
