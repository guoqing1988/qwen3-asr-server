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
  - 词级时间戳：独立加载 forced aligner，显存不足自动提高 util 重试
- `app/services/asr/qwen3_engine.py`：`llamacpp` 分支接入 `transcribe_*`、批量、流式、unload、aligner 生命周期
- `app/core/config.py` + `.env.example`：`ASR_ENGINE`、`QWEN_LLAMACPP_*`
- 文档：`docs/deployment.md`（Q8 部署/回滚）、`CLAUDE.md`

### 实测（RTX 4090，GPU3）

| 场景 | 结果 |
|---|---|
| 离线 40s 英文（热词 GPT） | 1.5s，`gpt` 正确 |
| 离线 80s（内部 25s 切段） | 1.6s |
| 流式 20s（160ms 帧，每 1s 重解码） | 总耗时 3.5s，每轮约 0.3s，文本持续增长不倒退 |
| 词级时间戳（10s） | 结构一致，17 words，时间合理；aligner 加载约 25s（含重试） |
| 说话人分离（52s 对话） | 2 段，说话人1/2 正确，走 `transcribe_long_audio` + `_transcribe_batch` |
| ASR 显存 | llama-server 约 4.1GB（vLLM bf16 约 8.3GB） |

### 已知差异

- Q8 质量略低于 bf16（示例音频出现 `tendbreak`、`fifty seven`），热词与专名仍生效；
- 流式为 1s 粒度（非 vLLM 的 160ms 增量），整段文本替换，前端已按 `current_segment_text` 全量刷新，兼容；
- `word_timestamps=true` 时 aligner 与 llama-server 同卡，峰值约 11GB，靠对齐后 sleep 回落。
