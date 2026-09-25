# CLAUDE.md

Qwen3-ASR 服务（GPU 版）：Confucius4-R2T2（vLLM，默认且唯一 ASR 模型）+ FunASR 实时流式 + 声纹识别（sqlite-vec），支持 **Docker Compose** 与 **本地 systemd** 两种部署方式。

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
  - VAD / CAM++ / Paraformer / 标点：`/data1/models/modelscope/hub/models/{iic,damo}`
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
- 相关测试：`test_idle_unload.py`（空闲/压力卸载）、`test_lazy_load_retry.py`（懒加载重试）、`test_startup_graceful_degrade.py`（降级启动）

## 引擎容错与显存策略（已实现）

- **懒加载自动重试**：引擎创建失败重试 3 次（间隔 5s/10s）
- **降级启动**：qwen3 预加载失败不阻断启动，funasr 兜底
- **forced aligner 懒加载**：aligner（另一个 vLLM 实例，约 3GB）不再在启动时预热，只在 `word_timestamps=true` 的请求到来时才加载（`QWEN_WARMUP_FORCED_ALIGNER=false`）——这是启动显存从 16GB 降到 6GB 的关键
- **空闲卸载**：`QWEN_IDLE_UNLOAD_TIMEOUT`（秒），0 禁用；当前 `.env` 设 **900（15 分钟无请求自动卸载 vLLM 引擎释放显存）**
- **显存压力卸载**：可用显存低于 `QWEN_LOW_VRAM_THRESHOLD_GB`（默认 6GB）且引擎空闲超过 `QWEN_VRAM_PRESSURE_COOLDOWN_S`（默认 60s）自动让位
- **显存控制参数**（全部走 `.env`，不写死在 `models.json`）：
  - `QWEN_GPU_MEMORY_UTILIZATION`（当前 0.35）
  - `QWEN_KV_CACHE_MEMORY_BYTES`（当前 1.5GiB；精确控制 KV cache，避免按 utilization 吃满显存）
  - `QWEN_MAX_NUM_SEQS`（当前 5）/ `QWEN_MAX_NUM_BATCHED_TOKENS`（当前 2048）/ `QWEN_MAX_MODEL_LEN`（当前 4096）
  - `QWEN_ENFORCE_EAGER`（当前 true，省 CUDA graph 显存）
  - `QWEN_FORCE_ALIGNER_GPU_MEMORY_UTILIZATION`（当前 0.20）/ `QWEN_FORCE_ALIGNER_MAX_MODEL_LEN`（当前 2048）
  - `QWEN_STREAM_MAX_NEW_TOKENS`（流式单 chunk 最大生成 token，当前 2，越小出字延迟越低；客户端可在 start 消息覆盖）
- **实测显存**（RTX 4090 24GB，`CUDA_VISIBLE_DEVICES=7`）：仅 ASR 引擎约 **6.0GB**；完整服务（含 VAD/CAM++/Paraformer/标点）约 **8.7GB**；显式请求词级时间戳时峰值约 12GB
- `gpu_memory_utilization` 有下限：主引擎约 0.16-0.20，对齐器约 0.15（调低会 KV cache 不足启动失败）；已用 `QWEN_KV_CACHE_MEMORY_BYTES` 精确控制 KV cache

## 已知问题与排障（详见 docs/troubleshooting.md）

- **vLLM 显存峰值**：推理后主进程持有 encoder buffer（reserved 非 allocated），`empty_cache` 无效；靠空闲/压力卸载释放
- **镜像膨胀**：`uv pip install` 缓存 `/root/.cache/uv`（~21GB）未清理，Dockerfile 需 `uv cache clean`
- **uv.lock 版本滞后**：lock 锁定的旧版 C 扩展与当前环境不兼容，**严禁运行时 `uv sync --frozen` 大面积降级**（会导致段错误）
- 容器重建丢可写层：`docker exec` 装的包（pytest 等）需重装；`.env` 备份放 `.env.bak.*`
- **CAM++ 离线加载**：`iic/speech_campplus_speaker-diarization_common/configuration.json` 里的子模型路径（`speaker_model` / `change_locator` / `vad_model`）可能残留旧路径（旧 Docker 路径，或缓存目录多一层 `models`）。`fix_camplusplus_config()` 启动时按路径末尾的 `<org>/<name>`（`damo` / `iic`）提取 model id，再用当前 `MODELSCOPE_CACHE` 重建，自动修正；若仍未修复，检查子模型目录权限（`sudo chown`）
- **ModelScope 缓存路径**：库真实结构是 `$MODELSCOPE_CACHE/models/<id>`，`settings.MODELSCOPE_PATH` 已按此拼接（`MODELSCOPE_CACHE` 指到 `.../modelscope/hub` 即可）；模型统一放 `/data1/models`，不再放项目内 `models/`

## 约定

- Python 3.12+ 现代类型注解（`X | None`、内置泛型、`collections.abc`），不写 `from __future__ import annotations`
- git commit 格式 `<type>: <描述>`（feat/fix/refactor/docs/test/chore）
- 中改及以上走完整流程：编码 → Review → 测试 → 更新文档（docs/deployment.md、docs/troubleshooting.md）
