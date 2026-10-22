# 计划：移除 FunASR 实时栈/阿里云端点，分段统一走 Nemotron

日期：2026-10-22　状态：已批准执行

## 目标

对外接口与上游 `AsrServe` 对齐，只保留 OpenAI 兼容（转写 / realtime / models）+ 声纹；
同时修掉「段内含大段静音导致整段输出为空」的问题。

## 一、删除阿里云协议与 Paraformer 实时栈

**端点下线**：`/stream/v1/asr`、`/stream/v1/asr/models`、`/stream/v1/asr/health`、
`/ws/v1/asr`、`/ws/v1/asr/funasr`、`/ws/v1/asr/qwen`（原已废弃）

**删除文件**：
- `app/api/v1/asr.py`、`app/models/asr.py`
- `app/api/v1/websocket_asr.py`、`app/services/websocket_asr.py`
- `app/services/asr/engines/funasr.py`

**接线清理**：`app/api/v1/__init__.py`、`app/services/asr/engines/__init__.py`、
`app/services/asr/manager.py`、`app/services/asr/model_plan.py`（去掉 `paraformer-large`）

**保留**：`funasr==1.3.1` 依赖（离线标点 CT-Transformer 仍用它）；
`X-NLS-Token` 鉴权（声纹/OpenAI 兼容接口仍在用，删它属于无收益的兼容性破坏）。

## 二、分段统一走 Nemotron，删除 FunASR VAD

- `app/services/asr/engines/base.py`：Nemotron **始终**跑 diarize；
  `enable_speaker_diarization` 只决定是否输出说话人标签（与上游一致）
- `app/utils/audio_splitter.py`：删除 `get_vad_segments`；`split_audio_file` 的
  `speech_segments` 变为必填
- `app/services/asr/engines/global_models.py`：删除 `get_global_vad_model` /
  `get_vad_inference_lock` / `get_global_punc_realtime_model`，保留离线 `get_global_punc_model`
- `app/services/asr/model_capabilities.py`：模型资产只留 R2T2 / aligner / Nemotron /
  CT-Transformer 标点 / CAM++ 声纹 SV
- `app/utils/model_loader.py`：去掉 VAD、Paraformer、实时标点预加载；离线标点改为总是预加载
- `app/core/config.py`：删除 `VAD_MODEL`、`FUNASR_*` 等仅供实时栈使用的配置

## 三、段间长静音断句（修空输出）

`merge_segments_greedy`：相邻语音区间之间静音超过 `MAX_SILENCE_GAP_SEC`（默认 5s）时
**断开为独立段**，不把长静音并进同一段送 ASR。

- 39s「14s 语音 + 24s 静音」→ 切成语音段 + 静音尾段，静音段识别为空后被丢弃，语音不丢
- 静音断开也减少无谓推理（长静音不再占用 60s 段预算）

## 验证矩阵

1. `pytest tests/ -q` 全绿；新增「长静音断句」单测；删除引用 websocket_asr 的旧测试
2. 生产 `journalctl` 无 ERROR；启动不再加载 VAD / Paraformer / 实时标点
3. 离线：`enable_speaker_diarization=true/false` 都能转写；39s 含长静音音频不再整段为空
4. `/v1/realtime`、词级时间戳、声纹注册/匹配、非法音频 400 均正常
5. 端点确认已下线（`/stream/v1/asr`、`/ws/v1/asr*` → 404/关闭）

## 回滚

代码层面 `git revert`；若只需临时恢复阿里云端点，回滚到本提交之前的分支即可。

## 风险

- 阿里云 SDK / `/ws/v1/asr` 客户端会断（用户已确认放弃）
- 关闭说话人分离的请求也会跑 Nemotron（GPU ~0.4GB，RTF≈0.014）
- 删除 VAD 后不再有 CPU 纯语音检测兜底：Nemotron 不可用时整段音频直接送 ASR

## 实施结果（2026-10-22 完成）

**代码**
- 删除 `app/api/v1/asr.py`、`app/models/asr.py`、`app/api/v1/websocket_asr.py`、`app/services/websocket_asr.py`、`app/services/asr/engines/funasr.py`、`app/services/qwen3_websocket_asr.py`（旧 WS 服务，仅 `/ws/v1/asr/qwen` 用过）
- 全局标点模块 `global_models.py` 只留离线标点；`model_loader` 预加载只剩 `punc_model` + `speaker_diarization_model`
- `base.py`：Nemotron 始终跑（分段依据），`enable_speaker_diarization` 只控制是否输出标签
- `audio_splitter.py`：删 `get_vad_segments`；`split_audio_file(speech_segments=)` 变必填；新增 `find_long_gap()` 与 `speech_coverage()`；无语音区间时走固定时长切分
- 对外路由仅剩 `/v1/audio/transcriptions`、`/v1/models`、`/v1/realtime`、`/api/v1/voiceprint-speakers*`

**验证**
| 项 | 结果 |
|----|------|
| `pytest tests/ -q` | 216 passed（新增 `tests/test_silence_gap_split.py` 8 例；删除 2 个 WS 测试文件与 3 个依赖旧 WS 服务的用例） |
| 生产启动 | 预加载 `declared=3 runtime=1` + `punc_model` + `speaker_diarization_model`，不再加载 VAD / Paraformer / 实时标点；无 ERROR（除 transformers 已知 image processor 警告） |
| 端点下线 | `/stream/v1/asr`、`/stream/v1/asr/health` → 404；`/ws/v1/asr*` 路由已不存在 |
| 离线转写 | 5min 音频 1207 字符 / 5 段 / 说话人「老板」（声纹命中）；`enable_speaker_diarization=false` 正常 |
| 39s 稀疏音频 | 修复前 `text=""`、0 段；修复后切 15 段、51 字符、`说话人1` |
| `/v1/realtime` | 2 段 completed，`speaker` = 说话人1/说话人2 |
| 词级时间戳 / 格式 / 非法音频 | `verbose_json` words=34；json/text/srt/vtt 正常；非法音频 400 |

**关键补充**：仅靠「长静音间隙」判据不够——该 39s 音频最大间隙 4350ms（<5s 阈值）、尾部 4290ms，Nemotron 把噪声当活动，活动散布整条时间轴。因此追加「语音覆盖率 < 50% 且时长 > 15s 也切分」判据（`AudioSplitter.MIN_SPEECH_COVERAGE` / `COVERAGE_CHECK_MIN_SEC`），实测覆盖率 37% 触发切分。
