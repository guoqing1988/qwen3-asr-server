# R2T2 原生用法改造（prompt / 流式 / VAD / 热词）

> 日期：2026-10-18
> 状态：实施中（不提交 git）

## 一、目标

保持现有功能（离线长音频、说话人分离、声纹、词级时间戳、显存休眠），把
Confucius4-R2T2 的官方推荐用法落到服务里：

1. prompt 对齐官方（system=热词原文，强制语言用 `language X<asr_text>` 后缀）→ 修复热词失效；
2. 流式默认 160ms chunk + 动态 max_tokens → 出字延迟 2s 降到约 0.4s；
3. 流式切段改用 FireRedVAD 静音边界（替代 12s 强切 + 1s overlap + 字符去重）；
4. 可选 two-pass：段定稿时用离线路径重转，保证热词/专名；
5. 超长音频：离线保持现有 VAD/说话人分段；流式用 VAD + 60s 时长兜底。

## 二、不引入官方 pip 包的原因（实测）

- PyPI `qwen-asr==0.0.6` 的 `streaming_transcribe(pcm, state)` 没有 `max_new_tokens`，
  而 Confucius4-R2T2 的 `r2t2_asr.py` 传 3 个参数 → `TypeError`，官方包按 README 跑不通；
- `qwen-asr` 强依赖 `nagisa` → `dynet`（重 C++ 依赖），并在 import 时注册 vLLM 0.14
  版模型实现，可能覆盖现有 vLLM 0.19 内置实现；
- 其收益（prompt/流式算法/热词）可直接移植，无需引入依赖。

## 三、设计

### 3.1 Prompt（`qwen3_vllm._build_chat_prompt`）

```
<|im_start|>system\n{context}<|im_end|>\n
<|im_start|>user\n<|audio_start|><|audio_pad|><|audio_end|><|im_end|>\n
<|im_start|>assistant\n
```
- `context` 直接作为 system（热词原文），不再包 "Transcribe the speech..."；
- `language` 非空时在末尾追加 `language {X}<asr_text>`，模型只续写正文；
- 离线、流式共用。

### 3.2 流式参数（默认值）

| 参数 | 新默认 | 旧默认 |
|---|---|---|
| `chunk_size_sec` | 0.16 | 2.0 |
| `lookahead_sec`（首 chunk 额外音频） | 0.16 | 无 |
| `unfixed_chunk_num` | 0 | 2 |
| `unfixed_token_num` | 1 | 5 |
| `max_new_tokens` | 动态 `(chunk+lookahead)/1280*2`，配置值作为上限 | 固定 32/2048 |
| 预热 warmup/initial | 删除 | 8/32 |

### 3.3 流式切段（`qwen3_websocket_asr`）

- **静音边界**：FireRedVAD `detect_chunk` 检测 `is_speech_end` → `finish_streaming_transcribe`
  落定当前段、重建 state；
- **时长兜底**：单段 ≥ `QWEN_STREAM_SEGMENT_MAX_SEC`（默认 60s）→ 落定并重建；
- **去掉** 1s overlap 预置、`_strip_overlap` 字符去重（append-only 本身不重复，字符去重可能误删）；
- VAD 不可用（依赖/模型缺失）时回退自适应能量阈值。

### 3.4 two-pass（可选，默认开启）

段落定时用该段 `audio_accum` 走离线路径（官方 prompt + 热词）重转，替换段文本；
只影响 `segment_end`/`final` 的定稿，不影响实时预览。

### 3.5 后处理

移植官方：`_normalize_punct_by_context`（中英标点按前字符归一）、
中文相邻字去空格、`split("|")`、U+FFFD token 跳过、`language None` 回退。

### 3.6 超长音频

- 离线：保持 `MAX_SEGMENT_SEC=60` 的 VAD/说话人分段 + 声纹，仅换 prompt；
- 流式：段内全量累积（≤60s），段间无重叠、文本自然拼接，可无限长。

## 四、改动清单

| 文件 | 改动 |
|---|---|
| `app/services/asr/qwen3_vllm.py` | prompt/parse/prefix/标点/动态 max_tokens/默认参数/`transcribe_array` |
| `app/services/asr/qwen3_engine.py` | 默认参数、`Qwen3StreamingState`、`transcribe_array` |
| `app/services/qwen3_websocket_asr.py` | 默认参数、FireRedVAD、段落定/two-pass、去 overlap |
| `app/services/asr/stream_vad.py` | 新增 FireRedVAD 封装（懒加载、失败回退） |
| `app/core/config.py` | `QWEN_STREAM_CHUNK_SEC` / `LOOKAHEAD_SEC` / `UNFIXED_*` / `VAD_*` / `FINAL_PASS`；`SEGMENT_MAX_SEC=60` |
| `.env.example` / `README.md` / `docs/deployment.md` | 新参数与默认值 |
| `tests/` | 更新 window/dedup 测试；新增 prompt、动态 token、VAD 段逻辑测试 |

## 五、依赖与模型

- `fireredvad==0.0.2`（增量安装到 `.venv`，同步 `pyproject.toml` + `requirements.txt`）；
- `FireRedTeam/FireRedVAD` 的 `Stream-VAD`（约 2.3MB）下载到 `/data1/models/FireRedVAD/Stream-VAD`；
- 模型/依赖缺失时自动回退能量阈值，不阻断启动。

## 六、验证

| 测试点 | 方式 |
|---|---|
| prompt 格式/强制语言 parse | 单元 |
| 动态 max_tokens、首 chunk lookahead | 单元 |
| 段落定无重复、VAD 事件驱动 | 单元（fake VAD） |
| 热词离线生效 | 集成（真实 GPU，40s 英文 `gpt` 对比） |
| 流式延迟/文本 | 集成（真实 GPU，首字 ≤0.6s） |
| 超长（拼接 5 分钟音频） | 集成（真实 GPU，段文本无重复、无丢段） |
| 现有回归 | `pytest tests/ -v` |

## 七、风险与回滚

- 流式默认 160ms 会提高出字频率（客户端需能处理），payload 可覆盖；
- VAD 在极低音量音频上可能漏检，保留能量阈值回退；
- two-pass 增加段结束时的推理量，可用 `QWEN_STREAM_FINAL_PASS=false` 关闭；
- 全部改动不改变对外 API 字段，旧客户端参数仍可用。

---

## 八、实施结果（2026-10-18 实机验证）

### 代码与依赖

- `_build_chat_prompt` 已改为官方格式（system=热词原文 + `language X<asr_text>` 后缀）；
- 流式默认 `chunk_size_sec=0.16`、`lookahead_sec=0.16`、`unfixed_chunk_num=0`、`unfixed_token_num=1`，生成量按 chunk 时长动态计算（配置值仅作上限）；
- 新增 `app/services/asr/stream_vad.py`（FireRedVAD `Stream-VAD` 封装），VAD 句尾触发落定；依赖/模型缺失自动回退能量阈值；
- 段落定改为 VAD 静音边界，移除 1s overlap 与字符去重；段定稿可选 two-pass（`QWEN_STREAM_FINAL_PASS=true`，默认开）；
- `fireredvad==0.0.2` 已加入 `pyproject.toml` / `requirements.txt`，模型在 `/data1/models/FireRedVAD/Stream-VAD`。

### 实测数据（真实 GPU，临时实例，未影响生产服务）

| 项目 | 结果 |
|---|---|
| 离线热词（40s 英文，`GPT, transcription`） | `gpt` 全部正确（改造前为 `gbt`） |
| 流式首字延迟（5.4s 中文，热态） | 0.20s（0.5s 帧）/ 0.44s（160ms 帧）；冷启动约 1.2s（kernel 预热） |
| 流式分段（6.7s 中文） | 2 段，文本完整自然 |
| 超长音频（67.4s，10 遍拼接） | 20 段；`之前` ×10、`不让喝` ×10，无重复、无丢段 |
| two-pass | 段定稿把 `之前有顾客自己带酒水` 修正为带句号版本，日志可见 `Final pass corrected segment` |
| 单测 | `pytest tests/ -q` → 127 passed |

### 遗留事项（需人工确认）

- 生产 `.env` 仍保留旧的 `QWEN_STREAM_*` 参数项（部分已废弃/默认值不同），重启后它们会覆盖新默认。建议同步更新（尤其 `QWEN_STREAM_SEGMENT_MAX_SEC`、`QWEN_STREAM_OVERLAP_SEC`、`QWEN_STREAM_MAX_NEW_TOKENS`），按项目规则修改 `.env` 需用户确认。
- 本次改造未执行任何 git 提交。
