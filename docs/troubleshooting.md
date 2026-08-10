# 排障经验记录

本文记录实际生产中遇到的故障与排查方法，供后续快速定位类似问题。

## 2026-08-10：vLLM EngineCore 间歇性段错误与宿主硬件故障

### 现象

- 容器启动时 `Failed core proc(s): {'EngineCore': -11}`（SIGSEGV）或 `Failed core proc(s): {}`（空集）
- 错误在加载 Qwen3-ForcedAligner 引擎阶段，主引擎 Qwen3-ASR 正常
- 偶发伴随 `double free or corruption (!prev)`（glibc 堆损坏）
- 同一时段 Claude Code 客户端（Bun）也段错误崩溃
- 空闲 300s 后引擎卸载，下次请求懒加载重载同样间歇性失败

### 根因（排查顺序）

1. **不要只盯着应用层**。多个无关进程（apport、systemd-udevd、python、Bun、vLLM EngineCore）在同一时段段错误，强烈提示宿主系统级问题。查内核日志：

   ```bash
   grep -E "general protection fault|segfault" /var/log/kern.log
   ```

   结果：宿主 Intel i9-14900K（13/14 代已知 CPU 电压不稳问题）在 13:17 起间歇性产生 GPF，波及所有进程。

2. **区分"容器问题"与"宿主问题"**：
   - `docker inspect qwen3-asr --format '{{.Created}}'` 判断容器是否被重建（重建清空可写层）
   - `docker diff qwen3-asr` 查看可写层变更（nvidia runtime 注入宿主驱动库是正常行为，非污染）
   - `md5sum` 对比容器内与挂载目录文件，排除挂载/文件损坏嫌疑

3. **vLLM 崩溃点定位**：`-11` = 子进程 SIGSEGV；`{}` 空集 = EngineCore 未留下退出码（崩溃更早）。Rust 栈（`tokenizers BpeBuilder::build` 中 String::clone 崩溃）= 堆内存损坏，是宿主数据损坏的典型表现。

### 解决（应用层容错，宿主需 BIOS/微码更新治本）

- **懒加载自动重试**：引擎创建失败重试 3 次（间隔递增），落在故障窗口间隙
- **预加载失败降级启动**：qwen3 失败不再拒绝启动，funasr 兜底，失败引擎懒加载恢复
- 详见 `docs/deployment.md`「引擎加载容错说明」

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
