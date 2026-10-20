# 计划：实时说话人标签 + 可配置对齐（移植上游 5ef0ce0 / ff8a493）

## 背景

上游 `AsrServe` 在 Nemotron 迁移后新增两项能力，我们尚未移植：

1. `5ef0ce0 feat: stream Nemotron speaker labels for realtime utterances`
   —— Nemotron 与 ASR 并行跑同一路 PCM，每个「停顿边界的话」结束后给出说话人标签，
   diarization 失败只让标签为 null，不影响转写。
2. `ff8a493 feat: add configurable alignment`（其中的对齐部分，macOS 服务不适用）
   —— `ALIGNMENT_MODE=forced|uniform`：uniform 按字均分时长的兜底词级时间戳，
   响应新增 `word_timestamp_method` 说明来源。

## 一、可配置对齐（uniform 兜底）

**新增** `app/services/asr/uniform_alignment.py`：`uniform_word_timestamps(text, duration)`
（CJK 逐字、拉丁按词、标点附到前一个单元，时长均分；最后单元对齐到段尾）。

**接线**：
- `settings.ALIGNMENT_MODE`（默认 `forced`）：`forced` = 现有 forced aligner；
  `uniform` = 不做声学对齐，直接用均匀时间。
- `Qwen3ASREngine._transcribe_batch`（vllm/llamacpp 分支）与 `transcribe_file_with_vad`：
  `uniform` 时跳过 aligner 调用，用 `uniform_word_timestamps(text, 段时长)`。
- `ASRSegmentResult.word_timestamp_method`（`forced` / `uniform`）→
  `/v1/audio/transcriptions` 的 `verbose_json` 新增可选字段 `word_timestamp_method`
  （新增字段，既有客户端不受影响）。

**收益**：无 aligner（显存紧张/未下载）时仍能返回 `words[]`；代价是时间精度差。

## 二、实时说话人标签

**`app/utils/speaker_diarizer.py`**：新增 `streaming_parts()`，返回
`(model, processor)`（复用懒加载实例，供流式推理取 `audio_chunk_start` /
`num_samples_first_audio_chunk` / `num_samples_per_audio_chunk`）。

**新增** `app/services/asr/realtime_speakers.py`：移植上游 `SpeakerStream`
- `feed(pcm_int16_bytes)` 累积音频
- `utterance(index, start_ms, end_ms)` 登记待标注的话
- `advance()` 按 Nemotron 流式 chunk 边界推进推理（`speaker_cache` 跨块复用），
  已覆盖的话按「概率 argmax 出现最多的说话人」打标签
- 失败只置 `finished`，标签返回 `None`，绝不影响转写

**接线 `app/services/asr/openai_realtime.py`**：
- 会话建立时创建 `SpeakerStream`（`settings.REALTIME_SPEAKER_LABELS` 控制，默认开）
- `_feed`：把 16k PCM 同时喂给 SpeakerStream
- `_finalize`：登记本段（绝对采样起止）→ 拉起 two-pass 重转（并行）→ 取标签（超时
  `REALTIME_SPEAKER_LABEL_TIMEOUT`，默认 2s）→ `completed` 事件新增 `speaker` 字段
  （`null` 表示未知；OpenAI 客户端忽略未知字段，兼容）
- `session.created/updated` 的 `session` 里加 `speaker_diarization` 布尔，告知能力

**收益**：实时转录界面可直接显示「谁在说」，与离线 diarization 标签体系一致
（`说话人N`），声纹命名仍只在离线接口生效。

## 验证矩阵

1. `pytest tests/ -q` 全绿；新增 `test_uniform_alignment.py`、`test_realtime_speakers.py`
   （标签聚合用假 model，不依赖 GPU）。
2. 生产 `word_timestamps=true`：
   - 默认 `forced`：行为与现在一致，`word_timestamp_method=forced`
   - `ALIGNMENT_MODE=uniform`：返回 `words[]`、时间单调、`word_timestamp_method=uniform`
3. `/v1/realtime` 30s 混人音频（Q8）：`completed` 带 `speaker`，两人对话切换正确；
   关闭 `REALTIME_SPEAKER_LABELS` 后字段为 `null`。
4. 显存：实时标签让 Nemotron 常驻（约 0.4GB），确认空闲休眠策略仍生效。
5. 日志无 ERROR；diarization 失败场景（模型缺失）不影响转写。

## 回滚

- 单项开关：`ALIGNMENT_MODE=forced`、`REALTIME_SPEAKER_LABELS=false`，重启即回到旧行为。
- 代码回滚：`git revert` 对应提交。

## 风险

- Nemotron 流式 processor API（`audio_chunk_start` / `is_streaming`）与离线路径不同，
  需在真实音频上校验帧对齐；标签延迟约 1s（Nemotron 缓冲），
  two-pass 重转通常已覆盖这段等待。
- 实时标签把 diarization 引入会话生命周期，需确保会话结束/异常时释放。

## 实施结果

### 已完成

- `app/services/asr/uniform_alignment.py`：CJK 逐字 / 拉丁按词均分，末单元对齐段尾。
- `ALIGNMENT_MODE`（`forced` 默认 / `uniform`）：vLLM `transcribe_file`/`transcribe_batch`、
  llama.cpp `_word_tokens`、rust/llama.cpp `transcribe_file_with_vad` 四处分支；
  `verbose_json` 新增 `word_timestamp_method`（`forced_alignment` / `uniform_fallback`）。
- `app/services/asr/realtime_speakers.py::SpeakerStream`：流式帧累积、utterance 覆盖判定、
  标签聚合、`flush` 收尾、`names` 跨流继承。
- `SpeakerDiarizer.streaming_parts()`：复用懒加载的 Nemotron model/processor。
- `/v1/realtime`：会话级 `speaker_diarization` 能力字段；`_feed` 并行喂 PCM 并推进推理；
  `_finalize` 登记段区间 → two-pass 重转（并行）→ 取标签 → `completed.speaker`。
- `REALTIME_SPEAKER_LABELS` / `REALTIME_SPEAKER_LABEL_TIMEOUT` 开关。

### 实测

| 场景 | 结果 |
|------|------|
| `ALIGNMENT_MODE=uniform` + `word_timestamps=true`（5min） | 1073 words，`word_timestamp_method=uniform_fallback`，首词 1.94s（段偏移正确） |
| 默认 `forced`（10s） | 19 words，`word_timestamp_method=forced_alignment` |
| `/v1/realtime` 30s（Q8，1x 实时速率，混人） | 2 段 completed，`speaker` 依次 `说话人1` / `说话人2` ✅ |
| `/v1/realtime` 快速灌音频（10x）后停止 | 末段靠 flush 拿到标签（`说话人1`） |
| `REALTIME_SPEAKER_LABELS=false` | `speaker_diarization=false`，`speaker=null`，转写不受影响 |
| `pytest tests/ -q` | 221 passed, 11 subtests |

### 已知局限

- 标签比文本晚约 1s（Nemotron 右上下文 / `fifo_length`）。
- 实时标签只给「每段一个主导说话人」，段内换人不切分（依赖 R2T2 停顿边界）。
- 会话超过 1 小时时 activity 缓冲打满，后续标签退化为 null（不影响转写）。
- `ALIGNMENT_MODE=uniform` 同时降低说话人归属精度（词级时间变成均分）。
