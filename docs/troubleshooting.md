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

- 推理后 GPU 占用 29GB（主进程 15GB + 两个 EngineCore 14GB），其他 GPU 服务（ComfyUI）可能 OOM
- `torch.cuda.empty_cache()` 在推理后调用**无效**

### 结论

- `nvidia-smi` 显示的是 reserved（保留）而非 allocated（实际使用）
- vLLM 0.19 双进程架构：主进程持有音频编码器（mm encoder）按 max_num_batched_tokens 预算预分配的活跃 buffer，非空闲块，`empty_cache` 释放不了
- 引擎空闲卸载后主进程回落（2GB），证明非内存泄漏
- `gpu_memory_utilization` 有下限：调太低报 `No available memory for the cache blocks` / `To serve at least one request ... (0.88 GiB KV cache is needed ...)`，启动失败。主引擎 48GB 卡下限约 0.16-0.20，对齐器因 encoder cache 开销大下限约 0.15
- 解决：**显存压力自适应卸载**（可用显存 <15GB 且引擎空闲超 60s 时自动卸载让位）+ 空闲卸载兜底

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

**根因**：CAM++ 的 `configuration.json` 中子模型路径（`speaker_model`、`change_locator`、`vad_model`）仍指向 Docker 容器路径 `/root/.cache/modelscope/hub/models/damo/...`，本地运行时 `MODELSCOPE_CACHE` 指向 `/data/www/qwen3-asr/models/modelscope/hub/models/`，modelscope 在本地找不到这些路径时尝试联网解析。

**解决**：`fix_camplusplus_config()` 在每次启动时自动修正，支持从旧绝对路径中提取 model ID 再映射到当前 `MODELSCOPE_CACHE` 路径。若仍有问题：
```bash
# 检查子模型目录权限（Docker 挂载的模型属 root）
sudo chown -R $(whoami):$(whoami) /data/www/qwen3-asr/models/
```

### 模型文件权限问题（Permission denied）

**现象**：`MODELSCOPE_CACHE` 指向的模型目录无法读取。

**根因**：`models/` 下的文件是 Docker 挂载时由容器写入的，属 root:root。

**解决**：
```bash
sudo chown -R $(whoami):$(whoami) /data/www/qwen3-asr/models/
sudo chown -R $(whoami):$(whoami) /data/www/qwen3-asr/logs/
sudo chown -R $(whoami):$(whoami) /data/www/qwen3-asr/temp/
sudo chown -R $(whoami):$(whoami) /data/www/qwen3-asr/data/
```

### 服务频繁重启（systemd StartLimit）

**现象**：`systemctl status` 显示 `start limit hit`，服务不再自动重启。

**根因**：服务配置了 `StartLimitBurst=3`，5 分钟内重启超过 3 次触发保护。

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
cd /data/www/qwen3-asr
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
