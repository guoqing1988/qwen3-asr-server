# CLAUDE.md

Qwen3-ASR 服务（GPU 版）：Confucius4-R2T2（vLLM bf16 默认，可选 llama.cpp Q8，`ASR_ENGINE` 切换）+ FunASR 实时流式 + Nemotron-3 说话人分离 + 声纹识别（sqlite-vec），支持 **Docker Compose** 与 **本地 systemd** 两种部署方式。

运行环境：**torch 2.13.0+cu130 / vLLM 0.30.0 / transformers 5.18.0.dev0（固定 commit）/ CUDA 13.0**；依赖统一由 `pyproject.toml` + `uv.lock` 管理（`uv sync`）。

## 开发与运行

### 本地 systemd 部署（当前生产环境）

- **启动**：`sudo systemctl start qwen3-asr`
- **重启**：`sudo systemctl restart qwen3-asr`（代码修改即时生效；`start.py` 每次启动都执行 `load_dotenv()`，`.env` 修改也随重启生效，2026-08-13 实测验证）
- **停止**：`sudo systemctl stop qwen3-asr`
- **日志**：`sudo journalctl -u qwen3-asr -f`
- **状态**：`sudo systemctl status qwen3-asr`
- **端口**：由 `.env` 的 `PORT` 决定（uvicorn 直接监听；不要写死在代码里）
- **部署目录**：`/data1/www/qwen3-asr-server`。unit 用**软链接**方式安装：
  `sudo ln -sfn /data1/www/qwen3-asr-server/qwen3-asr.service /etc/systemd/system/qwen3-asr.service`
  （改仓库里的 unit 文件后 `daemon-reload` 即生效，无需再 cp）
- **模型缓存**：统一放在 `/data1/models/`（`MODELSCOPE_CACHE=/data1/models/modelscope/hub`、`HF_HOME=/data1/models/huggingface`）
  - ASR 主模型：`/data1/models/Confucius4-R2T2`（`QWEN3_ASR_MODEL_PATH`）
  - 词级时间戳对齐器：`/data1/models/Qwen3-ForcedAligner-0.6B`（`QWEN3_FORCED_ALIGNER_PATH`）
  - 说话人分离：`/data1/models/Nemotron-3-Diarization`（`NEMOTRON_MODEL_PATH`，帧级/Nemotron-3，397MB）
  - VAD / 声纹 CAM++ / Paraformer / 标点：`/data1/models/modelscope/hub/models/{iic,damo}`
- **启动流程**：`start.py` 加载 `.env` → 检查模型缺失则报错（离线模式不自动下载）→ main.py lifespan 预加载引擎 → 就绪
- **配置**：`.env` 文件，`start.py` 通过 `load_dotenv()` 加载
- **GPU 选择**：由 `.env` 的 `CUDA_VISIBLE_DEVICES` 决定（unit 里**不设** `CUDA_VISIBLE_DEVICES` / `DEVICE` 的 `Environment=`，否则 systemd 的 `Environment` 优先级更高，会架空 `.env`）
- **服务控制**：`Restart=on-failure` + `StartLimitBurst=3` / `StartLimitIntervalSec=300`（写在 `[Unit]` 段；5 分钟内重启超 3 次会 `start-limit-hit` 拒绝启动，用 `systemctl reset-failed` 清除）

### Docker 部署（备选）

- **启动/重建**：`docker compose up -d`（改 `.env` 或镜像标签变化会重建容器，可写层会丢）
- **重启**：`docker restart qwen3-asr`（代码挂载，重启即生效；不重读 `.env`）
- **日志**：`docker logs -f qwen3-asr`
- **端口**：宿主 `9101` → 容器 nginx `8000` → uvicorn `18000`
- **模型缓存**：`./models/modelscope`、`./models/huggingface` 挂载到容器 `/root/.cache/`（宿主机持久化）
- **启动流程**：镜像内置 entrypoint 启动 uvicorn → start.py 检查模型缺失则自动下载 → main.py lifespan 预加载引擎 → 就绪。宿主机代码通过 `./app:/app/app` 挂载覆盖镜像内旧代码即时生效

### 新增 Python 依赖流程

项目通过 `pyproject.toml` 管理全部依赖（`uv.lock` 锁定版本）。本地和 Docker 统一用 `uv sync` 安装。

**运行环境增量加包（无需 rebuild 镜像）**：
1. 修改 `pyproject.toml` 添加依赖
2. `uv lock` 更新 uv.lock
3. **同步更新 `requirements.txt`**（版本范围与 pyproject.toml 一致，Docker entrypoint 增量补装用）
4. 本地：`uv sync` → `sudo systemctl restart qwen3-asr`
5. Docker：commit → `docker compose up -d` → entrypoint 自动补装

**正式发布（rebuild 镜像）**：
1. 上述步骤 + `docker build -f Dockerfile.gpu -t quantatrisk/qwen3-asr:gpu-latest .`
2. 若 venv 已包含（不再需要运行时补装），可从 `requirements.txt` 移除该依赖

⚠️ `requirements.txt` 与 `pyproject.toml` 版本范围必须一致。

## 测试

- 项目 Python 3.12，`tests/` 目录下的测试在本地 `.venv` 中运行
- 本地 `.venv` 不含 pytest：`uv pip install pytest`
- 跑测试：`.venv/bin/python -m pytest tests/ -v`
- Docker 环境：`docker cp tests/<file> qwen3-asr:/tmp/` → `docker exec qwen3-asr /opt/venv/bin/python -m pytest /tmp/test_*.py -v`
- 相关测试：`test_idle_unload.py`（空闲/压力卸载）、`test_lazy_load_retry.py`（懒加载重试）、`test_startup_graceful_degrade.py`（降级启动）、`test_audio_decode_fallback.py`（音频解码三级兜底）

## 推理后端：vLLM（默认）与 llama.cpp Q8

`ASR_ENGINE=vllm`（默认）/ `llamacpp`，开关键在 `.env`，对外 API 与返回结构不变（含 `segments[].start/end`；`word_timestamps` 默认关，开启时返回 `words` 与真实时间）。

- **llamacpp**：官方 Q8 GGUF（`/data1/models/Confucius4-R2T2-GGUF/`）+ 托管 `llama-server` 子进程（`app/services/asr/qwen3_llamacpp.py`）；ASR 显存 **4.1GB**（vLLM 约 8.3GB），离线/流式/长音频/说话人分离全支持；单次音频上限约 60s，后端内部按 `QWEN_LLAMACPP_MAX_SEGMENT_SEC`（25s）切段；流式默认 append-only（`QWEN_LLAMACPP_STREAM_MODE=append`，移植 T8 协议：token 级回滚 + stable 落后候选 1 token + 回撤候选整轮丢弃 + 首轮静音跳过），实测 3m44s 16 段 0 回撤；`redecode` 可回退为每 1s 全量重解码；段内有效语音 <0.8s 不落定；词级时间戳复用 `Qwen3-ForcedAligner-0.6B`，aligner util 保底 0.35（否则 KV cache 不足会白等约 90s）。生产实测：离线 40s 1.2s、流式 40s 首字 3.7s/3 段、词级首次 27s（17 words）、词级峰值约 10.6GB 且空闲 25s 回落至 4.5GB。
- **回滚**：`.env` 改回 `ASR_ENGINE=vllm`（或删除）重启即可。

## 实时接口（OpenAI Realtime）

- 对外的实时端点统一为 **`ws /v1/realtime`**（`intent=transcription`，OpenAI Realtime transcription 子集），实现在 `app/services/asr/openai_realtime.py` + `app/api/v1/realtime.py`；识别复用 `Qwen3ASREngine` 流式接口、FireRedVAD 断句、two-pass 定稿与热词合并
- 客户端事件：`session.update`（`format.rate` 16k/24k、`transcription.{prompt,language,keywords}`、`turn_detection` 为 `null` 或 `server_vad`）、`input_audio_buffer.append`（base64 PCM16，单次 ≤10s）、`commit`、`clear`
- 服务端事件：`session.created/updated`、`conversation.item.input_audio_transcription.delta/completed/failed`、`input_audio_buffer.committed/cleared`、`error`
- **实时说话人标签**（`REALTIME_SPEAKER_LABELS`，默认开）：`app/services/asr/realtime_speakers.py::SpeakerStream` 用 Nemotron 流式（10ms 帧、`speaker_cache` 跨块复用）与 ASR 并行消费同一路 PCM；每段 `completed` 事件带 `speaker`（`说话人N` / `null`），标签比文本晚约 1s（Nemotron 右上下文）。音频持续流入时无需等待；客户端 commit 不再发音频时用 `flush` 把尾部音频按最后一块推理一次（精度略降），随后该流终结并在下一段重建（说话人编号 `names` 继承）。等待上限 `REALTIME_SPEAKER_LABEL_TIMEOUT`（2s），分离失败只让标签为 null，绝不影响转写
- 鉴权：`Authorization: Bearer`、`?token=`、子协议 `openai-insecure-api-key.<key>`；握手回 `subprotocol=realtime`
- 流式 partial **关闭 ITN**（保证 delta 单调，避免 `一`→`1` 改写前缀），`.completed` 才做 ITN/标点；`server_vad` 模式句尾静音自动落定，连续讲话按 `QWEN_STREAM_SEGMENT_MAX_SEC`（60s）兜底
- **`ws /ws/v1/asr/qwen` 已废弃**：连接后返回 `{"type":"error","error":{"code":"endpoint_deprecated"}}` 并 `close(1008)`；`/ws/v1/asr`、`/ws/v1/asr/funasr`（FunASR 阿里云协议）保持不变

## 说话人分离与说话人归属（Nemotron-3）

- **模型**：`app/utils/speaker_diarizer.py` 用 `AutoModelForAudioFrameClassification`（`Nemotron3DiarizationForAudioFrameClassification`）加载 `/data1/models/Nemotron-3-Diarization`，**帧级 10ms** 输出 8 路说话人概率，`diarize(path)` 返回 `DiarizationResult(segments, probabilities, frame_seconds, duration, speaker_ids)`，`speech_intervals_ms()` 给出合并重叠后的语音区间（说话人上限 8）；标签为「说话人N」，失败抛 `DefaultServerErrorException`
- **取代 CAM++ 自动聚类**：旧 CAM++ diarization 在长音频（≥2048 embeddings）走 HDBSCAN 会欠分（实测 3 人音频只出 2 人）；Nemotron 实测同一 4524s 音频出 4 人、RTF≈0.014（62s/75min）、加载 18s（首次）、之后 6s/10min
- **切分**：`AudioSplitter.merge_segments_greedy` 把说话人活动区间边界当切点，**覆盖整条时间轴**（区间之间的间隙也送 ASR——未检测到活动不等于静音），仅对 <1s 短段做邻段合并，超长段按 `MAX_SEGMENT_SEC` 切分；`split_audio_file(..., speech_segments=)` 直接吃 Nemotron 的区间，未开分离时回退 FunASR VAD
- **归属**：`app/services/asr/speaker_attribution.py`（`assign_speakers` + `consolidate_speaker_turns`）按词级时间把 ASR 块切到说话人（覆盖度 ≥0.5 才切换、模糊时沿用当前说话人），再把同人跨停顿合并成段落、≤2s 插话并入主说话人、段落过长在句末切分；词级时间戳偶有毫秒级越界会钳制到段内（不再整单失败）
- 启用 diarization 时**自动跑 forced aligner**（`needs_words = word_timestamps or diarization`），返回给客户端前按请求恢复（未请求词级则清掉 `word_tokens`），对外结构不变
- 声纹（CAM++）仍独立负责「说话人命名」：先 diarization 分段 → 按 speaker 分组提取 embedding → 命中已注册声纹则替换 `speaker_id`

## 引擎容错与显存策略（已实现）

- **懒加载自动重试**：引擎创建失败重试 3 次（间隔 5s/10s）
- **降级启动**：qwen3 预加载失败不阻断启动，funasr 兜底
- **forced aligner 懒加载**：aligner（另一个 vLLM 实例，约 3GB）不再在启动时预热，只在 `word_timestamps=true` 的请求到来时才加载（`QWEN_WARMUP_FORCED_ALIGNER=false`）——这是启动显存从 16GB 降到 6GB 的关键
- **对齐模式 `ALIGNMENT_MODE`**：`forced`（默认，声学对齐）| `uniform`（`app/services/asr/uniform_alignment.py` 按字/词均分时长，不加载 aligner，省约 6GB 显存但精度低）；`verbose_json` 用 `word_timestamp_method`（`forced_alignment` / `uniform_fallback`）标明来源。注意 `uniform` 下说话人归属所用的词级时间也是均分的，归属精度会下降
- **空闲释放（两级，sleep mode 默认开启）**：
  - 一级 `QWEN_IDLE_UNLOAD_TIMEOUT`（秒，当前 `.env` 设 **15**）：`sleep`——权重 offload 到 CPU、丢弃 KV cache，**vLLM 进程常驻**，下次请求 `wake_up` 仅 **0.1~0.4s**（对比销毁重建 11~21s）
  - 二级 `QWEN_SHUTDOWN_AFTER_S`（秒，默认 **600**，0 禁用）：仍空闲则彻底 `shutdown` 子进程，把 sleep 残留的 CUDA context（两引擎合计约 1.9GB）一并归还
  - `QWEN_ENABLE_SLEEP_MODE=false` 可整体回退为「空闲即销毁子进程」
- **aligner 独立生命周期**：aligner 有自己的活跃计数与空闲计时，忙时绝不释放、也不被主引擎连坐（旧实现只在主引擎卸载时跟着 `shutdown`，对齐阶段主引擎一旦空闲就会把正在干活的 aligner 一起关掉）
- **声纹（CAM++）按 ASR 分段提取**：逐段提取 embedding 后按时长加权融合（单段超 30s 再细分、<1s 丢弃、全短则拼接兑底、每说话人最多 30 段等间隔采样）——把同一说话人数小时拼接音频整体送入会让 TDNN 中间激活爆显存（实测 CUDA OOM，且失败后 ~17GB reserved 不归还，声纹一直被降级为「仅保留 diarization 标签」）。分段限长后 GPU 安全：`.env` 默认 `VOICEPRINT_DEVICE=auto`（有 CUDA 用 GPU：30s 音频峰值约 840MiB、峰值 712MiB、快 CPU 5~8 倍），提取后 `empty_cache` 归还缓存
- **声纹让路**：离线转写在声纹（CAM++）推理前主动让 vLLM 引擎 `sleep` 归还显存，避免 CUDA OOM（实测曾因两引擎常驻导致声纹识别降级为仅保留 diarization 标签）
- **显存压力休眠**：可用显存低于 `QWEN_LOW_VRAM_THRESHOLD_GB` 且空闲超过 `QWEN_VRAM_PRESSURE_COOLDOWN_S` 即休眠让位；不支持 sleep 的引擎回退为卸载
- **sleep 与多模态缓存的坑**：启用 sleep 时必须同时 `mm_processor_cache_gb=0`（代码内固定），否则唤醒后重复音频会命中失效的 multimodal processor cache 并抛 `AssertionError`（已实测；该缓存是 CPU 内存，禁用不影响显存）
- **显存控制参数**（全部走 `.env`，不写死在 `models.json`）：
  - `QWEN_GPU_MEMORY_UTILIZATION`（当前 0.35）
  - `QWEN_KV_CACHE_MEMORY_BYTES`（当前 1.5GiB；精确控制 KV cache，避免按 utilization 吃满显存）
  - `QWEN_MAX_NUM_SEQS`（当前 5）/ `QWEN_MAX_NUM_BATCHED_TOKENS`（当前 2048）/ `QWEN_MAX_MODEL_LEN`（当前 4096）
  - `QWEN_ENFORCE_EAGER`（当前 true，省 CUDA graph 显存）
  - `QWEN_FORCE_ALIGNER_GPU_MEMORY_UTILIZATION`（当前 0.20）/ `QWEN_FORCE_ALIGNER_MAX_MODEL_LEN`（当前 2048）
  - `QWEN_STREAM_CHUNK_SEC`（当前 0.16，官方推荐 80ms~2s）/ `QWEN_STREAM_LOOKAHEAD_SEC`（当前 0.16，首 chunk 额外音频）
  - `QWEN_STREAM_UNFIXED_CHUNK_NUM`（当前 0）/ `QWEN_STREAM_UNFIXED_TOKEN_NUM`（当前 1）：官方 prefix 策略
  - `QWEN_STREAM_MAX_NEW_TOKENS`（当前 512，仅作**上限**；实际按 chunk 时长动态取 `(chunk+lookahead)/1280*2`）
  - **VAD 静音边界**：FireRedVAD `Stream-VAD`（`/data1/models/FireRedVAD/Stream-VAD`，2.3MB）在句尾静音处落定当前段并重建 state；模型/依赖缺失时回退 `QWEN_STREAM_SILENCE_SEC` 能量阈值；`QWEN_STREAM_SEGMENT_MAX_SEC`（当前 60s）作为连续讲话的时长兜底；`QWEN_STREAM_OVERLAP_SEC=0`（官方 append-only 不需要重叠）
  - **two-pass 定稿**：`QWEN_STREAM_FINAL_PASS=true` 时，段落定用该段音频走离线路径重转（热词生效），修正流式 append-only 无法回改的专名错误
- **实测显存**（RTX 4090 24GB）：**休眠态**仅主进程约 1.9GB + 引擎残留约 0.95GB ≈ **2.8GB**；**唤醒态** ASR 引擎约 **8.8GB**（权重 3.87 + KV 1.5 + 长音频 encoder buffer）、aligner 约 **6.0GB**（权重 1.73 + KV 1.79）：显式请求词级时间戳时两者合计峰值约 17GB（靠 15s 空闲休眠回落）
- `gpu_memory_utilization` 有下限：主引擎约 0.16-0.20，对齐器约 0.15（调低会 KV cache 不足启动失败）；已用 `QWEN_KV_CACHE_MEMORY_BYTES` 精确控制 KV cache

## 已知问题与排障（详见 docs/troubleshooting.md）

- **vLLM 显存峰值**：推理后主进程持有 encoder buffer（reserved 非 allocated），`empty_cache` 无效；靠空闲/压力卸载释放
- **镜像膨胀**：`uv pip install` 缓存 `/root/.cache/uv`（~21GB）未清理，Dockerfile 需 `uv cache clean`
- **uv.lock 版本滞后**：lock 锁定的旧版 C 扩展与当前环境不兼容，**严禁运行时 `uv sync --frozen` 大面积降级**（会导致段错误）
- 容器重建丢可写层：`docker exec` 装的包（pytest 等）需重装；`.env` 备份放 `.env.bak.*`
- **CAM++ 离线加载（仅声纹）**：说话人分离已改 Nemotron，以下只影响声纹命名用的 CAM++ SV 模型；`iic/speech_campplus_speaker-diarization_common/configuration.json` 里的子模型路径（`speaker_model` / `change_locator` / `vad_model`）可能残留旧路径（旧 Docker 路径，或缓存目录多一层 `models`）。`fix_camplusplus_config()` 启动时按路径末尾的 `<org>/<name>`（`damo` / `iic`）提取 model id，再用当前 `MODELSCOPE_CACHE` 重建，自动修正；若仍未修复，检查子模型目录权限（`sudo chown`）
- **ModelScope 缓存路径**：库真实结构是 `$MODELSCOPE_CACHE/models/<id>`，`settings.MODELSCOPE_PATH` 已按此拼接（`MODELSCOPE_CACHE` 指到 `.../modelscope/hub` 即可）；模型统一放 `/data1/models`，不再放项目内 `models/`
- **vLLM 0.30 流式退化（已知）**：`/v1/realtime` 在 vLLM 后端上短 chunk 会出现幻觉词/丢字（每轮 `language X<asr_text>` 头部需重新生成，动态 `max_tokens` 已预留 12 token，但仍不如 Q8 稳定）。实时场景请用 Q8（`ASR_ENGINE=llamacpp`，T8 append-only + 首轮静音跳过 + <0.8s 不落定），vLLM 后端以离线转写为主
- **vLLM 0.30 显存口径变化**：默认开启 CUDA graph memory profiling，`gpu_memory_utilization=0.35` 等效 0.3346；配 `kv_cache_memory_bytes` 时会跳过 profiling，若显存仍不足会报 `No available memory for the cache blocks`（调大 `QWEN_GPU_MEMORY_UTILIZATION` 或减小 `QWEN_MAX_NUM_BATCHED_TOKENS`/encoder 预算）
- **FlashInfer 未装预编译 cubin**：`flashinfer-cubin`（1.5GB，仅 GitHub release）未列入依赖，首次推理前 FlashInfer 会 JIT 编译采样 kernel，需要 PATH 里能找到 `ninja`（`pyproject.toml` 已加 `ninja`；`start.py` 启动时把 venv 的 bin 目录注入 PATH，systemd 下也生效），首启因此慢约 60s
- **音频解码兜底**：`convert_audio_to_wav` 为三级链路（librosa/soundfile → PyAV 逐帧容错 → ffmpeg CLI），损坏 mp3 由 PyAV/ffmpeg 兜底；`av` 已提升为显式依赖，本机 ffmpeg 已通过 `apt` 安装

## 离线分段与标点（2026-10-19 同步上游修正）

- **分段有界不丢尾音**（上游 `534405a`/`c02da1e`）：`AudioSplitter.merge_segments_greedy` 合并后再按 `MAX_SEGMENT_SEC` 切分；`_split_by_fixed_duration` 把不足最小时长的尾段并入前段（旧实现会直接丢弃），毫秒/采样点换算改整数；切分逻辑已全部收敛到 `AudioSplitter`（Nemotron 只提供语音区间，不再自己切段）
- **句末标点补齐**（上游 `9b4ee87`/`1854214`/`ba1dba8` 简化版）：`app/services/asr/punctuation.py::restore_sentence_endings` 复用 FunASR CT-Transformer（`PUNC_MODEL`，CPU），离线结果缺句末标点时**只补一个标点**，不复制模型对正文的改写；英文/数字结尾用 ASCII 标点，结尾是逗号/分号则替换为句末标点，收尾引号保持在标点之后（`…"好的"` → `…"好的。"`）。批量一次调用（缺失段少于 1 条不加载模型），失败仅告警不阻断转写。开关 `ASR_RESTORE_SENTENCE_ENDING`（默认 true，全量列表长度不影响，实测 46 段 5min 音频 +0.2s）
- 未搬运：上游「离线尾部 pad 二次解码」（`OFFLINE_TAIL_SAMPLES`，每段 2x 推理）——我们离线 `max_new_tokens=1024` 且分段 ≤60s，未见尾部截断，收益不确定不引入

## 约定

- Python 3.12+ 现代类型注解（`X | None`、内置泛型、`collections.abc`），不写 `from __future__ import annotations`
- git commit 格式 `<type>: <描述>`（feat/fix/refactor/docs/test/chore）
- 中改及以上走完整流程：编码 → Review → 测试 → 更新文档（docs/deployment.md、docs/troubleshooting.md）
