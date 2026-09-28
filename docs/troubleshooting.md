# 排障经验记录

本文记录实际生产中遇到的故障与排查方法，供后续快速定位类似问题。

## 2026-08-10：容器间歇性段错误（SIGSEGV）排查全过程

### 现象

- 容器启动时 `Failed core proc(s): {'EngineCore': -11}`（SIGSEGV）或 `Failed core proc(s): {}`（空集）
- 偶发 `Aborted (core dumped)` / `double free or corruption (!prev)`（glibc 堆损坏）
- 崩溃发生在 CAM++/PUNC 等 modelscope 模型加载阶段，或 vLLM EngineCore 初始化阶段
- 同一时段 Claude Code 客户端（Bun）也段错误崩溃
- 用 `docker run` 无挂载启动测试容器正常，用 `docker compose`（挂载代码）崩

### 排查过程

1. **初期怀疑硬件**：宿主多个无关进程（apport、systemd-udevd、python）同时段 GPF，且为 i9-14900K。查内核日志：
   ```bash
   grep -E "general protection fault|segfault" /var/log/kern.log
   ```
   但服务崩溃模式高度一致（7 次都是同一个 python3.12 内存地址 `[420000+2e3000]`），不像硬件随机性。

2. **排除文件损坏**：`md5sum` 对比容器内与宿主机挂载目录——文件完全一致。

3. **锁定真正根因：`uv sync --frozen` 运行时降级**。镜像 build 时 `uv pip install vllm[audio]` 装了新版 C 扩展（xgrammar 0.2.3 等），entrypoint 中 `uv sync --frozen` 按 lock 将 91 个包**降级到旧版**——新版 .so 文件被旧 Python wrapper 加载，版本不兼容导致段错误。证据：
   - `docker run --rm <镜像> uv sync --frozen --dry-run` 显示 Would download 91 packages
   - 去掉运行时 `uv sync` 后**立即正常**
   - 在 Dockerfile build 末尾加 `uv sync --frozen` 对齐后，build 出来的镜像反而崩溃（build 时降级同样不兼容）

### 解决

- **入口点**：不执行运行时大面积依赖同步（只保留 sqlite-vec 按需检查）
- **依赖对齐**：不应在构建时或运行时降级——`uv.lock` 锁定的旧版本与当前 PyTorch 2.10 + CUDA 12.8 环境不兼容
- **正确的依赖策略**：新增依赖需**重新构建镜像**（含 `uv lock` 更新 lock），不在运行时改 venv
- 详见 `CLAUDE.md`「新增 Python 依赖流程」和 `docs/deployment.md`「引擎加载容错说明」

### 关键教训

- **`uv sync --frozen` 在运行时做是大忌**：lock 与镜像 venv 版本不一致时，原地降级 C 扩展极大概率段错误
- **C 扩展 Crash 的特征**：崩溃地址固定（`python3.12[420000+2e3000]`），不是硬件随机性
- **隔离测试 vs compose 的差异**：`docker run` 用镜像内代码，`docker compose` 用挂载代码——排查时间歇性表现可用此方法排除代码变动嫌疑

## vLLM 显存占用排查

### 现象

- 早期默认配置下：推理后 GPU 占用 29GB（主进程 15GB + 两个 EngineCore 14GB），其他 GPU 服务（ComfyUI）可能 OOM
- `torch.cuda.empty_cache()` 在推理后调用**无效**

### 结论

- `nvidia-smi` 显示的是 reserved（保留）而非 allocated（实际使用）
- vLLM 0.19 双进程架构：主进程持有音频编码器（mm encoder）按 max_num_batched_tokens 预算预分配的活跃 buffer，非空闲块，`empty_cache` 释放不了
- 引擎空闲卸载后主进程回落（2GB），证明非内存泄漏
- `gpu_memory_utilization` 有下限：调太低报 `No available memory for the cache blocks` / `To serve at least one request ... (0.88 GiB KV cache is needed ...)`，启动失败。主引擎 48GB 卡下限约 0.16-0.20，对齐器因 encoder cache 开销大下限约 0.15
- 显存实测（2026-09-28）：**休眠态约 2.8GB**（主进程 1.9GB + 引擎残留 0.95GB）；唤醒态 ASR 引擎 8.8GB、aligner 6.0GB
- 解决（三级）：
  1. **启动即降显存**（2026-09-25 完成）：forced aligner 改懒加载 + KV cache 精确控制（`QWEN_KV_CACHE_MEMORY_BYTES`）+ `enforce_eager` + 降 `max_num_batched_tokens`，启动显存 16GB → 约 6GB
  2. **运行时让位**：显存压力休眠（可用显存 < `QWEN_LOW_VRAM_THRESHOLD_GB`（默认 6GB）且引擎空闲超 `QWEN_VRAM_PRESSURE_COOLDOWN_S`（默认 60s）时自动让位）+ 空闲休眠（`QWEN_IDLE_UNLOAD_TIMEOUT`，生产设 15s）
  3. **sleep mode**（2026-09-28）：空闲只归还显存、进程常驻，唤醒 <1s（详见本文末新章节）

## 镜像体积排查

### 现象

- 重新 build 后镜像从 23GB 膨胀到 42GB，venv 内包列表几乎无差异

### 结论

- `uv pip install` 的下载缓存默认保存在 `/root/.cache/uv`，Dockerfile 未清理 → 镜像携带 21GB 缓存
- 修复：Dockerfile 在依赖安装后执行 `uv cache clean`（或删除 `/root/.cache/uv`）
- 排查命令：`docker exec <容器> du -sh /root/.cache/uv`；`docker history <镜像>` 对比各层大小

## 容器重建注意事项

- `docker compose up -d` 在 `.env` 变化或镜像标签更新时会**重建容器**（丢弃可写层）
- 可写层中的临时安装（`docker exec pip install` 的包、pytest 等）重建后丢失，需重新安装
- 本项目 `tests/` 目录未挂载进容器，容器内跑测试需 `docker cp` 测试文件（或在 CI 环境跑）
- `.env` 修改需重建容器才生效（`docker restart` 不重读 .env）

## 2026-08-12：说话人声纹匹配失败（sqlite-vec KNN 查询报错）

**现象**：长音频转录结果中说话人未匹配到注册声纹，日志出现
```
Voiceprint enrichment failed; keep diarization labels: A LIMIT or 'k = ?' constraint is required on vec0 knn queries.
```

**根因**：sqlite-vec 0.1.9 的 `vec0` KNN 查询**不支持 `ORDER BY distance LIMIT ?` 语法**（即使 LIMIT 为常量也报错），必须使用 `k = ?` 约束。`SqliteVecVoiceprintStore.search()` 中的 KNN 查询沿用了旧版语法，导致整个声纹匹配链路报错降级为保留 diarization 标签。

**解决**：KNN 查询改为 `WHERE embedding MATCH ? AND k = ?`（`k` 绑定为候选数上限），外层 JOIN 过滤逻辑不变。

**排查命令**：`.venv/bin/python -m pytest tests/test_voiceprint_matching.py -v`（`test_sqlite_vec_store_groups_multiple_samples_by_speaker` 覆盖完整 KNN 链路）

## 2026-08-12：Forced aligner 反转时间戳警告

**现象**：日志出现
```
Forced aligner produced reversed timestamps for token='，': start_ms=1120.0 end_ms=960.0
```
反转均发生在标点上，幅度 2-3 个分类 bin（`timestamp_segment_time=80ms`，即 160-240ms）。

**根因**：与官方 `qwen_asr` 实现有两处差异：
1. 本项目把独立标点（`，`、`。` 等）作为对齐单元送入 forced aligner——标点无语音内容，模型的 token_classify 预测分布平坦、argmax 不稳定，偶发 start/end 乱序。官方 `Qwen3ForceAlignProcessor` 会剔除标点（仅保留 `'`、字母、数字）。
2. 缺少官方 `fix_timestamp` 后处理——官方对整条时间戳序列做 LIS 单调化 + 异常段插值，本项目原先仅逐 token 交换 start/end。

**解决**：`_split_alignment_units` 剔除独立标点（词内符号如 `'`、`-` 随词保留）+ 移植官方 `fix_timestamp`（`_fix_timestamp`，全局 LIS 单调化 + 就近填充/线性插值），解析时不再产生反转警告。

**边界情况（非故障）**：若音频段含较长静音前缀（VAD 未切分），ASR 文本开头 token 需对齐到静音区，模型预测乱序，`_fix_timestamp` 的 LIS 会把这些位置归零（输出单调但时间戳为 0）。这是模型无法对齐静音区的固有行为，纯语音段（VAD 正常切分）无此现象；旧实现此时输出乱飞的非零值（更差）。

**排查命令**：`.venv/bin/python -m pytest tests/test_qwen3_align.py -v`（切分、单调化、align 解析链路全覆盖）

## 2026-08-10：Docker → 本地 systemd 迁移问题

### CAM++ 说话人分离模型加载失败（离线环境 400 错误）

**现象**：
```
ERROR - 400 Client Error: Bad Request for url: https://www.modelscope.cn/api/v1/models//root?Revision=master
ERROR - CAM++ 模型加载失败: SegmentationClusteringPipeline: invalid model repo path
```

**根因**：CAM++ 的 `configuration.json` 中子模型路径（`speaker_model`、`change_locator`、`vad_model`）仍指向旧路径（Docker 容器路径 `/root/.cache/modelscope/hub/models/damo/...`，或缓存目录多一层 `models`），本地运行时 `MODELSCOPE_CACHE` 已变，modelscope 在本地找不到这些路径时会把绝对路径当 model id 去联网解析。

**解决**：`fix_camplusplus_config()` 在每次启动时自动修正——按路径末尾的 `<org>/<name>`（`damo` / `iic`）提取 model id，再用当前 `MODELSCOPE_CACHE` 重建。2026-09-25 修复了正则（原来只认 `/hub/models/damo/...`，匹配不到多一层 `models` 的旧路径）。若仍有问题：
```bash
# 检查子模型目录权限（Docker 挂载的模型属 root）
sudo chown -R $(whoami):$(whoami) /data1/models/
```

### 模型文件权限问题（Permission denied）

**现象**：`MODELSCOPE_CACHE` 指向的模型目录无法读取。

**根因**：`models/` 下的文件是 Docker 挂载时由容器写入的，属 root:root。

**解决**：
```bash
sudo chown -R $(whoami):$(whoami) /data1/models/
sudo chown -R $(whoami):$(whoami) /data1/www/qwen3-asr-server/logs/
sudo chown -R $(whoami):$(whoami) /data1/www/qwen3-asr-server/temp/
sudo chown -R $(whoami):$(whoami) /data1/www/qwen3-asr-server/data/
```

### 服务频繁重启（systemd StartLimit）

**现象**：`systemctl status` 显示 `start limit hit`，服务不再自动重启。

**根因**：服务配置了 `StartLimitBurst=3`，5 分钟内重启超过 3 次触发保护。调试期间手动连续 `systemctl restart` 也会计数（不一定是启动失败）。注意这两个参数应写在 `[Unit]` 段（写在 `[Service]` 段不生效）。

**解决**：
```bash
# 先修复启动失败根因，然后重置计数
sudo systemctl reset-failed qwen3-asr
sudo systemctl start qwen3-asr
```

## 2026-08-13：vLLM 引擎启动失败（bad marshal data，pyc 缓存损坏）

### 现象

- 引擎空闲卸载后懒加载失败，或重启后预加载失败：日志出现
  `EngineCore failed to start` + `ValueError: bad marshal data (unknown type code)`，
  回溯指向 `importlib._bootstrap_external._compile_bytecode`
- 转写请求返回空响应（流式响应已发出心跳后中断：`Caught handled exception, but response already started`）
- 该错误与早期 Docker 部署时遇到的同类错误一致（当时未能解决，是迁移到本地部署的原因之一）

### 排查

```bash
# 1. 用全新进程复现导入失败，定位损坏模块
cd /data1/www/qwen3-asr-server
.venv/bin/python -c "from vllm._aiter_ops import rocm_aiter_ops"   # → bad marshal data

# 2. 检查 pyc 文件时间戳（ctime=mtime 说明文件自生成后未变，损坏发生在写入时）
stat -c '%y %z' .venv/lib/python3.12/site-packages/vllm/__pycache__/_aiter_ops.cpython-312.pyc

# 3. 批量体检 vllm 包全部 pyc（输出损坏文件列表）
.venv/bin/python -c "
import marshal, os
for dp, _, fns in os.walk('.venv/lib/python3.12/site-packages/vllm'):
    for fn in fns:
        if fn.endswith('.pyc'):
            p = os.path.join(dp, fn)
            try: marshal.loads(open(p, 'rb').read()[16:])
            except Exception as e: print(p, e)
"
```

### 根因

`.pyc` 字节码缓存写入时被中断（进程被杀、并发编译冲突，或容器构建/文件拷贝期间编译）
导致正文损坏，而文件头部（magic/时间戳/大小）完好，导入时通过头部校验后才在
marshal 解析处报错。

### 解决

用完好源码重新编译覆盖损坏的 pyc（不删除文件），然后重启服务验证：

```bash
.venv/bin/python -c "
import py_compile
py_compile.compile(
    '.venv/lib/python3.12/site-packages/vllm/_aiter_ops.py',
    cfile='.venv/lib/python3.12/site-packages/vllm/__pycache__/_aiter_ops.cpython-312.pyc',
    doraise=True,
)
"
# 验证导入通过后再重启
.venv/bin/python -c "from vllm._aiter_ops import rocm_aiter_ops"
sudo systemctl restart qwen3-asr
```

也可直接删除损坏的 pyc（导入时自动从源码重建，效果相同）。若损坏面较大，
可对整个包清理缓存：`find .venv -name '__pycache__' -type d -exec rm -rf {} +`
（缓存会自动重建，无数据损失）。

## 2026-09-25：迁移 Confucius4-R2T2 并优化启动显存

### 启动显存 16GB → 6GB

**现象**：服务启动即占 16GB（主引擎 ~8GB + forced aligner ~8GB），而 5 并发场景没必要常驻 aligner。

**根因**：`Qwen3ASREngine._warmup_forced_aligner()` 在启动时**无条件**预热 forced aligner（另一个 vLLM 实例）。

**解决**（全部由 `.env` 驱动，不写死在 `models.json`）：
- `QWEN_WARMUP_FORCED_ALIGNER=false`：aligner 改为首次 `word_timestamps=true` 请求时懒加载
- `QWEN_KV_CACHE_MEMORY_BYTES=1610612736`（1.5GiB）：精确控制 KV cache，不再按 `gpu_memory_utilization` 吃满
- `QWEN_MAX_NUM_BATCHED_TOKENS=2048`：直接决定 encoder cache 预算
- `QWEN_ENFORCE_EAGER=true`：省 CUDA graph 约 0.5GB

**实测**（RTX 4090 24GB，GPU7）：基线 0.86GB → 仅 ASR 引擎 6.0GB → 完整服务 8.87GB → 词级时间戳峰值 12.2GB。

### 配置分层：models.json 只放模型定义，运维参数走 .env

**踩坑**：早期把 `gpu_memory_utilization` 等运维参数写进 `models.json`，每次调显存都要改模型定义文件。

**约定**：`models.json` 只放模型定义（路径、`max_model_len`、`max_inference_batch_size`、`forced_aligner_path`）；所有运维参数（显存、并发、卸载策略）走 `.env`。`models.json` 支持 `${VAR:-default}` 占位符，例如 `"offline": "${QWEN3_ASR_MODEL_PATH:-netease-youdao/Confucius4-R2T2}"`，换路径只改 `.env`。

### systemd unit 用软链接 + 参数不要与 .env 重复

**踩坑 1**：unit 里写了 `Environment="CUDA_VISIBLE_DEVICES=7"`，而 systemd 的 `Environment=` 优先级高于 `.env`（`load_dotenv()` 默认不覆盖已有环境变量），导致 `.env` 里配的 GPU 被架空。**unit 里只保留进程级开关**（`FUNASR_STARTUP_UI`、`PYTHONUNBUFFERED`），GPU / 端口 / 模型路径全部交给 `.env`。

**踩坑 2**：调试期间频繁 `systemctl restart` 触发 `start-limit-hit`（见上一节）。

**安装方式**：用软链接而非 `cp`，让仓库内 unit 成为唯一来源：
```bash
sudo ln -sfn /data1/www/qwen3-asr-server/qwen3-asr.service /etc/systemd/system/qwen3-asr.service
sudo systemctl daemon-reload
```

### 流式（实时转录）输出乱码 / 退化成标点

**现象**：`/ws/v1/asr/qwen` 流式返回的文本是 `"language None"`、`"None"`、`"."`，而不是实际转写内容；同一音频的离线转写完全正常。

**三个根因（2026-09-25 逐一定位）**：

1. **prefix 语义错误（主因）**：`Qwen3VLLMBackend._decode_stream` 用 `raw_decoded` 的**最后 `unfixed_token_num` 个 token** 作为下一轮 prompt 前缀。这只有在「vLLM KV 增量续写」时才成立；本项目每轮都是 `generate(prompt + prefix, audio=累积音频)` 的**全量重算**，只给末尾 1 个 token 等于丢掉全部历史——文本逐轮丢内容，最后只剩 `.`。
   修复：prefix 改为**已定稿历史**（完整 `raw_decoded` 去掉末尾 `unfixed_token_num` 个 token），抽出 `_build_stream_prefix()` 便于测试。

2. **`max_new_tokens` 太小**：曾按「官方实时推荐 2」配置 `QWEN_STREAM_MAX_NEW_TOKENS=2`。本项目每轮全量重算，该值是「整段累积文本的生成上限」，给 2 连 `language X<asr_text>` 前缀都吐不完。
   修复：默认改 `512`（**上限而非耗时**，遇 EOS 即停，不影响出字延迟），另加预热期参数 `QWEN_STREAM_MAX_NEW_TOKENS_INITIAL=32` / `QWEN_STREAM_WARMUP_CHUNKS=8`。

3. **解析残留**：流式 prefix 用 `skip_special_tokens=False` 解码（必须保留 `<asr_text>` 以免解析失败），会带出 `|` 等标记字符；音频不足时模型还会把语言预测成字面量 `None`。
   修复：`_parse_asr_output()` 清理 `|` 并将 `None/null` 视为无效，回退到请求指定语言。

**验证**：用 `/tmp/ws_dump.py` 逐 chunk 打印，8s 音频（`chunk_size_sec=0.16`）文本持续推进且与离线结果一致，处理速度 ≈0.156s/chunk（跟得上实时）。

**回归测试**：`.venv/bin/python -m pytest tests/test_qwen3_stream_parse.py -v`（覆盖 `_parse_asr_output` 解析与 `_build_stream_prefix` 边界）。

### 长时实时任务：解码窗口必须滑动（2026-09-25）

**现象**：52.5s 连续语音（`chunk_size_sec=0.16`）流式转写，发送结束后服务端仍滞后 **6.2s（≈12%）** 才处理完。

**根因**：每轮 `generate(prompt + prefix, audio=全部累积音频)` 是**全量重算**，prefill 开销随段内音频线性增长。原实现虽有分段机制，但单段上限硬编码 60s（`MAX_BUFFER=960000`）、静音阈值 2s（`SILENCE_THRESHOLD=32000`）——连续语音无静音时单段一直涨到 60s，窗口内后期严重落后，也不适合小时/天级实时任务。

**改造（滑动窗口 + 段间重叠去重）**：

- `QWEN_STREAM_SEGMENT_MAX_SEC`（默认 12s）：单段上限，超时强制落定并开新段 → 每轮解码窗口恒定，不随会话时长增长
- `QWEN_STREAM_SILENCE_SEC`（默认 2s）：静音优先落定（自然句边界）
- `QWEN_STREAM_OVERLAP_SEC`（默认 1s）：新段开头预置上一段尾部音频，减少强制切段处丢词；识别结果按字符级后缀/前缀匹配去重，**大小写不敏感**（重叠音频处模型可能重排大小写，如 `Really` vs `really`，敏感匹配会漏去重）
- 硬编码的 `MAX_BUFFER` / `SILENCE_THRESHOLD` 类常量移除，全部走 `.env`

**效果**：52.5s 音频滞后 **6.2s → 0.1s**，分 5 段，段边界重复文本被正确去除，`FULL` 文本连贯（如段边界 `really hard work` + `Really hard work getting...` → `getting our technology...`）。

**回归测试**：`.venv/bin/python -m pytest tests/test_qwen3_stream_window.py -v`（`_strip_overlap` 中英文/大小写/边界，`_need_truncate` 阈值，`_extract_segment_tail` 尾部切片）。

**残余限制**：单段内仍是全量重算，所以窗口大小直接决定单轮成本；若要进一步降延迟，需改用 vLLM 异步引擎做 KV 增量续写（大改造）。

## 2026-09-28：音频格式转换失败导致 500（mp3 损坏 + 本地缺 ffmpeg）

### 现象

`POST /v1/audio/transcriptions` 返回 500，日志：

```
Note: Illegal Audio-MPEG-Header 0x2c203733 at offset 5179245.
Note: Trying to resync...
Note: Skipped 1024 bytes in input.
error: Giving up resync after 1024 bytes - your stream is not nice...
[OpenAI API] 转写失败: 音频标准化失败: 音频格式转换失败:
```

**错误信息尾部为空**，仅凭日志看不出真实原因。

### 根因（三层叠加）

1. **音频文件本身损坏**：mp3 在 offset 5179245 处混入非 MPEG 数据（`0x2c203733` = ASCII `, 73`），libsndfile/libmpg123 在 resync 限制（1024 字节）内无法重新同步，直接报错。
2. **librosa 的回退无效**：`librosa.load()` → soundfile 失败后回退 `audioread`，而本机 audioread 只有 `rawread` 后端（不认 mp3），抛出无信息的异常 → 日志里的空错误串由此而来。
3. **ffmpeg 兜底双重失效**：
   - 本地 systemd 部署**没装 ffmpeg**（ffmpeg 只在 Docker 镜像里，见 `Dockerfile.gpu`）；
   - `convert_audio_to_wav` 的兜底命令参数错误：`ffmpeg -f s16le ... -i input`，`-f s16le` 强制按**裸 PCM** 解析 mp3，即使装了 ffmpeg 也会失败。

> 注意：`get_container_duration()` 依赖 `ffprobe`，本地未安装时静默返回 `None`（仅影响 m4a 时间戳缩放），不影响本次故障。

### 修复

`app/utils/audio.py` 的 `convert_audio_to_wav()` 改为三级解码链路，任一成功即返回：

| 顺序 | 解码器 | 适用场景 |
|------|--------|----------|
| 1 | librosa / soundfile | 正常文件（精度最好，主路径不变） |
| 2 | **PyAV 逐帧容错解码**（`_decode_audio_with_pyav`） | 含损坏帧/非法 MPEG 头的音频：坏包只丢弃当前帧，继续解码后续数据 |
| 3 | ffmpeg CLI | Docker 镜像内置；命令修正为 `ffmpeg -i input -vn -ar 16000 -ac 1 -acodec pcm_s16le -y output` |

同时：错误信息聚合三层失败原因（不再是空串）；`av` 提升为 `pyproject.toml` 显式依赖（此前仅由 vllm 传递引入）。

### 验证

构造损坏 mp3（在 20KB 处插入 40KB 垃圾字节）复现：

- 修复前：soundfile 报错 → audioread 无后端 → ffmpeg 不存在 → 500
- 修复后：PyAV 容错解码丢弃 9 个坏包，8s 音频恢复 7.92s，转换成功

回归测试：`.venv/bin/python -m pytest tests/test_audio_decode_fallback.py -v`（正常 mp3 走 librosa、损坏 mp3 走 PyAV、不可解码文件报完整分层错误、ffmpeg 命令不含 `-f s16le`）。

### 关键教训

- 损坏音频文件是常态（下载中断、拼接错误），解码链路必须有**容错层**，且不依赖可能缺失的系统二进制；
- fallback 分支要保留真实异常（原实现 `str(e)` 引用的是外层异常，日志为空，掩盖了根因）。后续已在本机 `sudo apt install -y ffmpeg`（6.1.1）使 ffmpeg CLI 路径可用（Docker 镜像本就内置）。

## 2026-09-28：显存策略改为「sleep mode + 15s 空闲休眠」

### 背景

同一张卡上还有 ComfyUI、llama.cpp 等服务，vLLM 引擎空闲常驻 17GB（ASR 8.8GB + aligner 6.0GB + 主进程 2.1GB）会挤爆其它服务，并导致本项目声纹（CAM++）CUDA OOM 降级：

```
Voiceprint enrichment failed; keep diarization labels: CUDA out of memory. Tried to allocate 3.84 GiB.
```

### 方案：进程常驻 + 只归还显存（vLLM sleep mode）

| | 旧行为（销毁子进程） | 新行为（sleep mode） |
|---|---|---|
| 空闲动作 | `shutdown()` 杀 EngineCore | `sleep(level=1)`：权重 offload 到 CPU、丢弃 KV cache |
| 释放显存 | 全部（除残留 ~1.84GiB） | ASR 6.45GiB / aligner 3.91GiB（各残留 ~1.84GiB） |
| 恢复耗时 | 重载 11~21s | `wake_up()` **0.2~0.5s** |
| 空闲阈值 | 900s | **15s**（`QWEN_IDLE_UNLOAD_TIMEOUT`） |

实现要点：

- `Qwen3VLLMBackend` 新增 `sleep_main` / `sleep_aligner` / `wake_main` / `wake_aligner`，用 `threading.RLock` 与推理入口（`_main_in_use` / `_aligner_in_use`）互斥：使用中不睡、睡眠中自动唤醒（覆盖 `_run_generate`、`_decode_stream`、`align_transcript` 全部推理入口）；
- `RuntimeRouter` 空闲监控改为调用 `sleep`（不支持 sleep 的引擎回退 `unload`）；`release()` 时刷新 `_last_use`（空闲从「请求结束」而非「请求开始」起算）；
- **aligner 独立生命周期**：自己的活跃计数与空闲计时，忙时绝不释放。旧实现只在主引擎卸载时跟着 `shutdown`，对齐阶段主引擎一旦空闲就会把正在干活的 aligner 一起关掉；
- 声纹推理前主动 `release_vllm_gpu()` 让引擎休眠腾显存；
- `QWEN_ENABLE_SLEEP_MODE=false` 可整体回退旧行为。

### 踩坑：sleep + multimodal cache = AssertionError

启用 sleep 后，唤醒再次推理相同音频时报：

```
ERROR [core.py] AssertionError: Expected a cached item for mm_hash='48bfbe6c...'
```

根因：vLLM 的 multimodal processor cache（`mm_processor_cache_gb`，默认 4GiB）内部状态在 sleep/wake 后失效，重复音频命中失效项（`vllm/multimodal/cache.py:360`）。

解决：启用 sleep 时固定传 `mm_processor_cache_gb=0`（`qwen3_vllm.py`）。该缓存位于 CPU 内存，禁用不影响显存。

实测（本机 vLLM 0.19.0 + Confucius4-R2T2）：

```
load=11.0s → before-sleep infer ok → sleep=3.7s（释放 6.45GiB）→ wake=0.38s → after-wake infer ok
```

### 效果

休眠态显存 **17GB → 2.8GB**（主进程 1.9GB + 引擎残留 0.95GB）；请求到来 0.2~0.5s 唤醒，基本无感。

回归测试：`.venv/bin/python -m pytest tests/test_idle_sleep.py -v`（14 项：空闲 sleep 而非卸载、活跃请求不睡、无 sleep 能力回退卸载、aligner 忙不睡/可单独睡、空闲计时刷新、声纹让路、sleep 与推理互斥与自动唤醒、vLLM 参数组装）。

### 附带修复：声纹（CAM++）OOM 的真正根因

实测（2026-09-28 09:47）发现：引擎 `sleep` 后整卡只剩 4.51GB，但 2 秒后声纹仍然 OOM：

```
09:47:53 Sleep mode freed 7.6 GiB, 4.51 GiB still in use   ← 引擎已全部归还
09:47:54 Loading CAM++ speaker embedding provider: device=cuda:0
09:47:56 Voiceprint enrichment failed: CUDA OOM. Tried to allocate 1.92 GiB ... 1.62 GiB free
```

根因不在 vLLM：`embedding_service.py` 把**同一说话人的所有语句拼接成一整段音频**（去掉静音，可能几千秒）后整体送入 CAM++，TDNN 中间激活直接爆显存；失败后 ~17GB 变成 torch reserved **不会自动归还**（下次请求继续 OOM）。即：声纹从未成功过，一直在“撞 OOM → 降级”。

修复（两层）：

1. **按 ASR 分段提取**（`SpeakerEmbeddingService._extract_segment_embeddings`）：用识别出的分段作为提取单位（每段通常几秒，符合 CAM++ 短时假设），逐段提取后按时长加权融合；单段超 `VOICEPRINT_MAX_CHUNK_SEC`(30s) 再细分，短于 `VOICEPRINT_MIN_CHUNK_SEC`(1s) 的段跳过（全部过短时退化为拼接提取，避免整人丢失）。
2. **推理设备默认 `auto`**（`VOICEPRINT_DEVICE=auto`）：有 CUDA 用 GPU、无则 CPU。分段限长后 GPU 不再有风险：实测 30s 音频峰值 840MiB（reserved）/ 712MiB，比 CPU 快 5~8 倍（30s 音频 37ms vs 291ms；实测 10 段×24s GPU 2.6s，含首次 CUDA 初始化）。提取完 `torch.cuda.empty_cache()` 归还缓存（reserved 不会自动释放）。

回归测试：`.venv/bin/python -m pytest tests/test_voiceprint_embedding_chunking.py -v`（11 项：短音频单次、长音频切片与时长加权、尾片丢弃、参考音频全路径、按分段提取、超长单段细分、短段跳过、全短段兑底、低于最小时长跳过、设备默认 CPU、provider 使用配置设备）。
