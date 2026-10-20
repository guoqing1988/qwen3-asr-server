# Nemotron-3 Diarization 集成（方案 A：全量升级 + 替换 CAM++ 分段）

日期：2026-10-19
状态：实施中

## 背景

现有离线说话人分离用 CAM++ 自动聚类：3 人录音实测只出 **2 人**（≥2048 embeddings 走
HDBSCAN，欠分），且 `merge_short_segments` 合并到 60s 导致段内夹他人。

隔离验证（`/tmp/nemo_deps`，未动生产环境）结论：

| | CAM++ | Nemotron-3（CPU） |
|---|---|---|
| 人数 | 2 | 4 类（前 10min 为 3 类） |
| 粒度 | 窗口级 | **帧级 10ms，支持重叠** |
| 4524s 音频耗时 | — | 推理 62s（RTF 0.014），含解码 89s |
| 重叠检测 | 无 | S0–S2 重叠 194.6s（真实抢话） |

用户决定走方案 A（上游已验证：vLLM 0.30 + torch 2.13 + transformers 指定 commit）。

## 环境升级（隔离验证后切换生产）

驱动已满足（RTX 4090 / 驱动 580.178.04 / CUDA 13.0）。

- `torch 2.13.0` / `torchaudio 2.11.0` / `torchvision 0.28.0`（cu130 源）
- `vllm[audio] 0.30.0` + `flashinfer-cubin 0.6.18.post1` + `nvidia-nccl-cu13 2.29.7`
- `transformers` 固定 commit `27166ea03f12c940f23176a904ab1d2ff1a3dcbb`
- `huggingface_hub 1.33.0`、`tokenizers>=0.23.1,<0.24`、`safetensors>=0.8`
- `fastapi 0.136.3`、`requests 2.34.2`、`modelscope 1.34.0`、`scipy 1.15.3`（Linux）
- 移除 `hdbscan`（CAM++ diarization 不再使用；声纹 CAM++ 走 modelscope pipeline 保留）
- 同步 `uv.lock` 与 `requirements.txt`（版本范围一致）
- **隔离验证**：`UV_PROJECT_ENVIRONMENT=/data1/venvs/qwen3-asr-upgrade uv sync`，验证通过后再切换 `.venv`

## 代码迁移

1. `app/utils/speaker_diarizer.py`：替换为上游 Nemotron 实现
   （`DiarizationResult` / `SpeakerSegment(confidence)` / `diarize()` / `speech_intervals_ms()`，
   模型 `NEMOTRON_MODEL_PATH=/data1/models/Nemotron-3-Diarization`，8 说话人上限、
   帧级概率、重叠区间）。
2. 新增 `app/services/asr/speaker_attribution.py`：`assign_speakers`（词级时间→说话人）、
   `consolidate_speaker_turns`（同人跨停顿合并、≤2s 插话吸收、≤60s 段落句末切分）。
3. `app/services/asr/engines/base.py::transcribe_long_audio`：
   - 启用 diarization 时：`diarization.speech_intervals_ms()` 作为切分提示 +
     `AudioSplitter` 有界切分 → ASR chunk（≤60s）→ 词级时间戳 → `assign_speakers`
     → `consolidate_speaker_turns` 组装最终 segments。
   - 关闭 diarization 时保持现有 VAD 分段。
4. `app/core/config.py`：新增 `NEMOTRON_MODEL_PATH`；移除 CAM++ diarization 相关配置
   （保留声纹 `VOICEPRINT_*`）。
5. `app/utils/model_loader.py`：预加载/清理路径改为 Nemotron；删除
   `get_global_diarization_pipeline` 引用。
6. `app/services/asr/model_capabilities.py`：diarization 能力声明适配（帧级、需词级时间）。
7. 响应结构不变：`segments[].speaker_id`、`speaker_segments`（若对外暴露）、
   `word_timestamps` 顶层 `words[]` 保持现状。

## 保留功能（不得降级）

- 离线/流式 Q8（`ASR_ENGINE=llamacpp`）、vLLM bf16 路径
- 声纹识别（CAM++ sv + sqlite-vec，独立于 diarization）
- `/v1/realtime`（OpenAI Realtime）、`/ws/v1/asr`、`/ws/v1/asr/funasr`
- 词级时间戳（forced aligner）、离线句末标点、热词、非法音频 400
- 空闲休眠/显存策略（sleep mode、两级释放）

## 验证矩阵

1. `pytest tests/ -q` 全绿（含新增 diarization/attribution 单测）。
2. 生产接口回归（`/tmp/q8_regression.py` 14 项）：Q8 离线/流式、热词、词级、ITN、diarization、
   response_format、阿里云、并发。
3. 混人音频 `/home/ubuntu/test_audio_3.mp3`（前 10min）：
   - 人数 = 3（S0/S1/S2），段内说话人单一（不再夹他人）
   - 与 Nemotron 原始分段一致性检查（段边界落在说话人切换点）
4. 声纹：注册声纹后离线转写能正确命名（GPU CAM++ 路径）。
5. 词级时间戳：`word_timestamps=true` 结构与时间正常；aligner 休眠/唤醒正常。
6. 流式：`/v1/realtime` commit + server_vad 各推 30s，0 回撤；Q8 长音频 3min。
7. 显存：休眠态/唤醒态读数对比升级前（vLLM 0.30 允许变化，但需记录）。
8. 日志：无 ERROR/异常；Nemotron 推理耗时记录。

## 回滚

- 分支 `feature/nemotron-diarization`，环境升级独立 commit；回滚 =
  `git checkout main` + `uv sync`（lock 回到旧版）+ `systemctl restart qwen3-asr`。
- 升级前备份生产 `.venv`（`/data1/venvs/qwen3-asr-backup-<date>`）。

## 风险

- vLLM 0.30 API 变化：`LLM`、`SamplingParams`、sleep/wake、`kv_cache_memory_bytes`、
  `mm_processor_cache_gb`、`engine_core.shutdown` 需逐一验证。
- transformers 5.x 可能影响 FunASR/modelscope（它们通常要求 transformers<5）→ 需验证
  FunASR 与声纹 pipeline。
- 首次运行 vLLM 0.30 会重新编译/生成 torch.compile 缓存（启动变慢，属一次性）。

## 实施结果（2026-10-20）

### 环境

- 隔离 venv `/data1/venvs/qwen3-asr-upgrade`：torch 2.13.0+cu130 / torchaudio 2.11.0 /
  torchvision 0.28.0 / vllm 0.30.0 / transformers 5.18.0.dev0（commit `27166ea0`）/
  huggingface_hub 1.33.0 / tokenizers 0.23.3 / safetensors 0.8.0 / fastapi 0.136.3
- **`flashinfer-cubin` 移除**：只发 GitHub release（1.5GB，实测直连 16KB/s），改为让
  FlashInfer JIT 编译采样 kernel；为此新增 `ninja` 依赖，`start.py` 启动时把 venv 的
  `bin` 注入 `PATH`（systemd 下也生效）。首启因此多约 60s，产物缓存于 `~/.cache/flashinfer`
- `pytest` 全量 **183 passed**（新 env）

### 代码

- `app/utils/speaker_diarizer.py`：Nemotron 帧级实现（`DiarizationResult` /
  `SpeakerSegment(confidence)` / `speech_intervals_ms()` / 懒加载 + 全局实例 + `close()`），
  取代 CAM++ 自动聚类（HDBSCAN 长音频欠分）
- `app/services/asr/speaker_attribution.py`：新增 `assign_speakers` /
  `consolidate_speaker_turns`（词级归属 + 段落合并 + ≤2s 插话并入 + 60s 段落切分），
  并对词级时间做段内钳制（原先毫秒级越界会让整单 500）
- `app/utils/audio_splitter.py`：`split_audio_file(..., speech_segments=)` 支持外部语音区间；
  `merge_segments_greedy` 改为「边界切点覆盖整条时间轴」（间隙也送 ASR，避免低音量语音丢失）
- `app/services/asr/engines/base.py`：diarization → 切分 → 批量识别（`needs_words =
  word_timestamps or diarization`）→ 归属 → 按请求恢复 `word_tokens`
- `app/core/config.py`：新增 `NEMOTRON_MODEL_PATH`；`app/utils/model_loader.py` 预加载改 Nemotron
- `app/services/asr/qwen3_vllm.py`：流式 `max_tokens` 预留 12 token 头部；prefix 改用
  「已解析正文 + 显式语言头部」重建；对齐失败重试一次（Q8 同步）
- `app/services/asr/punctuation.py`：句末标点补齐（承接上一轮）

### 实测（新 env）

| 场景 | 结果 |
|------|------|
| 离线 Q8 5min（含 diarization） | 42.7s 首次 / 10.6s 热态，5 段 |
| 离线 vLLM 5min | 18.4s，5 段；10min 含 diarization 35.3s，25 段 3 人 |
| 词级时间戳 5min | Q8 35s / 1073 words；vLLM 41.9s（含 aligner 冷加载）/ 1043 words |
| Nemotron diarization | 10min 6.1s；75min 全量 62s（4 人）；176min 全量见下 |
| 全量 B（75min, 3 人） | 190s，148 段，16801 字，4 人，尾段 gap 0.5s |
| 全量 A（176min） | 440s，251 段，41968 字，8 人（模型上限），尾段 gap 0.7s |
| `/v1/realtime`（Q8） | commit 16k/24k、server_vad 39s（3s 静音）各 1 段 completed，delta 单调 |
| FunASR `/ws/v1/asr/funasr` | 8 partial + 1 final |
| 废弃 `/ws/v1/asr/qwen` | `endpoint_deprecated` + 1008 |
| 声纹 | 注册 3 样本 → 5min 转写命中 `测试说话人A` 7 段；删除正常 |
| 非法音频 | 400 `INVALID_PARAMETER` |

### 已知问题（本轮未解决）

- **vLLM 0.30 流式退化**：短 chunk 下出现英文碎片/丢字（Q8 正常）。已做两项加固
  （头部余量、prefix 用正文重建）但未根治；实时场景建议 `ASR_ENGINE=llamacpp`
- **`flashinfer-cubin` 缺位**：首启 JIT ~60s（已通过 `ninja` + PATH 解决可用性）
- 长会议音频说话人数易顶到 8 人上限（模型上限，非 bug）
