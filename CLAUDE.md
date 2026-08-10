# CLAUDE.md

Qwen3-ASR 服务（GPU 版）：Qwen3-ASR-1.7B（vLLM）+ FunASR 实时流式 + 声纹识别（sqlite-vec），Docker Compose 部署，代码 bind mount 到宿主机 `/data/www/qwen3-asr`。

## 开发与运行

- **启动/重建**：`docker compose up -d`（改 `.env` 或镜像标签变化会重建容器，可写层会丢）
- **重启**：`docker restart qwen3-asr`（代码挂载，重启即生效；不重读 `.env`）
- **日志**：`docker logs -f qwen3-asr`
- **端口**：宿主 `9101` → 容器 nginx `8000` → uvicorn `18000`
- **模型缓存**：`./models/modelscope`、`./models/huggingface` 挂载到容器 `/root/.cache/`（宿主机持久化）
- **启动流程**：镜像内置 entrypoint 启动 uvicorn → start.py 检查模型缺失则自动下载 → main.py lifespan 预加载引擎 → 就绪。宿主机代码通过 `./app:/app/app` 挂载覆盖镜像内旧代码即时生效

### 新增 Python 依赖流程

项目通过 `requirements.txt` 管理运行时额外依赖（`pyproject.toml` 的子集）。entrypoint 启动时 `uv pip install -r` 增量补装，已安装的秒过。

**运行环境增量加包（无需 rebuild 镜像）**：
1. 修改 `pyproject.toml` 添加依赖
2. `uv lock` 更新 uv.lock（宿主机无 uv 时：`docker exec qwen3-asr sh -c "cd /app && uv lock"`）
3. **同步更新 `requirements.txt`**（版本范围与 pyproject.toml 一致）
4. commit → `docker compose up -d` → entrypoint 自动补装

**正式发布（rebuild 镜像）**：
1. 上述步骤 + `docker build -f Dockerfile.gpu -t quantatrisk/qwen3-asr:gpu-latest .`
2. 若 venv 已包含（不再需要运行时补装），可从 `requirements.txt` 移除该依赖

⚠️ `requirements.txt` 与 `pyproject.toml` 版本范围必须一致，否则两个文件指向不同版本。

## 测试

- 宿主机 Python 3.13 与项目（3.10-3.12）不匹配，**测试在容器内跑**
- `tests/` 未挂载进容器：`docker cp tests/<file> qwen3-asr:/tmp/` 后执行
- pytest 不在镜像内：`docker exec qwen3-asr uv pip install --python /opt/venv/bin/python pytest`
- 跑测试：`docker exec qwen3-asr /opt/venv/bin/python -m pytest /tmp/test_*.py -v`
- 相关测试：`test_idle_unload.py`（空闲/压力卸载）、`test_lazy_load_retry.py`（懒加载重试）、`test_startup_graceful_degrade.py`（降级启动）

## 引擎容错与显存策略（已实现）

- **懒加载自动重试**：引擎创建失败重试 3 次（间隔 5s/10s）
- **降级启动**：qwen3 预加载失败不阻断启动，funasr 兜底
- **空闲卸载**：`QWEN_IDLE_UNLOAD_TIMEOUT`（默认 300s），0 禁用
- **显存压力卸载**：可用显存 <15GB 且引擎空闲 >60s 自动让位（阈值/冷却在 `router.py` 常量 `_LOW_VRAM_THRESHOLD_GB` / `_VRAM_PRESSURE_COOLDOWN_S`）
- `gpu_memory_utilization` 有下限：主引擎 48GB 卡约 0.16-0.20，对齐器约 0.15（调低会 KV cache 不足启动失败）

## 已知问题与排障（详见 docs/troubleshooting.md）

- **vLLM 显存峰值**：推理后主进程持有 encoder buffer（reserved 非 allocated），`empty_cache` 无效；靠空闲/压力卸载释放
- **镜像膨胀**：`uv pip install` 缓存 `/root/.cache/uv`（~21GB）未清理，Dockerfile 需 `uv cache clean`
- **uv.lock 版本滞后**：lock 锁定的旧版 C 扩展与当前环境不兼容，**严禁运行时 `uv sync --frozen` 大面积降级**（会导致段错误）
- 容器重建丢可写层：`docker exec` 装的包（pytest 等）需重装；`.env` 备份放 `.env.bak.*`

## 约定

- Python 3.12+ 现代类型注解（`X | None`、内置泛型、`collections.abc`），不写 `from __future__ import annotations`
- git commit 格式 `<type>: <描述>`（feat/fix/refactor/docs/test/chore）
- 中改及以上走完整流程：编码 → Review → 测试 → 更新文档（docs/deployment.md、docs/troubleshooting.md）
