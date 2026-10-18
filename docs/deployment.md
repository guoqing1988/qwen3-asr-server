# Qwen3-ASR 部署指南

快速部署 Qwen3-ASR 语音识别服务，支持 CPU/macOS 与 NVIDIA GPU 两种运行形态。

依赖安装现在改成根目录默认 GPU，CPU 为单独特化环境：

| 模式 | 命令 | 说明 |
|------|------|------|
| GPU | `uv sync` | Linux/NVIDIA 运行时，默认锁定 CUDA 12.8/cu128 `torch/torchaudio/torchvision` + `vllm[audio]==0.19.0` |
| CPU | `./scripts/sync_cpu_env.sh` | Linux/CPU 运行时 |

## 快速部署

### GPU 版本部署（推荐）

适用于生产环境，提供更快的推理速度：

**前置要求：**
- NVIDIA GPU（默认镜像面向 CUDA 12.8+；CUDA 12.6 / 13.0 可通过构建参数覆盖）
- 已安装 NVIDIA Container Toolkit
- 显存 8GB+（默认模型 Confucius4-R2T2；24GB 卡实测：仅 ASR 引擎约 6GB，含 VAD/声纹/Paraformer/标点完整栈约 8.7GB，启用词级时间戳峰值约 12GB）

```bash
# 使用 docker run（带模型挂载）
docker run -d --name qwen3-asr \
  --gpus all \
  -p 9101:8000 \
  -v ./models/modelscope:/root/.cache/modelscope \
  -v ./models/huggingface:/root/.cache/huggingface \
  -v ./data:/app/data \
  -v ./logs:/app/logs \
  -v ./temp:/app/temp \
  quantatrisk/qwen3-asr:gpu-latest

# 或使用 docker compose（推荐）
docker compose up -d
```

### 多 GPU 自动并行部署（推荐）

适用于并发量较高场景。该方案通过容器 entrypoint 自动完成：
- 根据 `CUDA_VISIBLE_DEVICES` 拉起多个 ASR 实例（每张卡 1 个实例）
- 容器内自动生成 Nginx upstream 并负载均衡到各实例
- 对外仍只暴露一个服务端口（默认 `9101`，由 NGINX_PORT 控制）

你不需要手工维护多个 `docker compose` 服务块或手工维护 nginx upstream。

```bash
# 4 卡示例：GPU0,1,2,3 各启动 1 个实例
CUDA_VISIBLE_DEVICES=0,1,2,3 docker compose up -d
```

常用组合：
- 单卡（保持默认）：`CUDA_VISIBLE_DEVICES=0`
- 双卡：`CUDA_VISIBLE_DEVICES=0,1`
- 四卡：`CUDA_VISIBLE_DEVICES=0,1,2,3`

**服务访问地址：**
- API 服务: `http://localhost:9101`
- API 文档: `http://localhost:9101/docs`

### CPU 版本部署

适用于开发测试或无 GPU 环境：

```bash
docker run -d --name qwen3-asr \
  -p 9101:8000 \
  -v ./models/modelscope:/root/.cache/modelscope \
  -v ./models/huggingface:/root/.cache/huggingface \
  -v ./data:/app/data \
  -v ./logs:/app/logs \
  -v ./temp:/app/temp \
  quantatrisk/qwen3-asr:cpu-latest
```

CPU 镜像使用 QwenASR Rust，并自动选择 `qwen3-asr-0.6b`。x86_64 需要
`avx2` 与 `fma`；`word_timestamps=true` 会加载 forced aligner。构建目标与
运行时选择见主 README。

### Linux 本地部署（systemd，无需 Docker）

适用于已安装 NVIDIA GPU 的 Linux 服务器。直接使用 uv 创建虚拟环境在宿主机运行，
通过 systemd 管理服务生命周期。

**前置要求：**
- Ubuntu 22.04+ / 同等 Linux 发行版
- NVIDIA GPU + 驱动（显存 8GB+，推荐 12GB+ 以启用词级时间戳）
- CUDA 12.8（按 PyTorch cu128 编译）
- [uv](https://docs.astral.sh/uv/) 包管理器（安装：`curl -LsSf https://astral.sh/uv/install.sh | sh`）
- `ffmpeg` / `ffprobe`（音频解码兜底与容器时长探测；Docker 镜像已内置）：`sudo apt install -y ffmpeg`

**1. 创建虚拟环境并安装依赖**

```bash
cd /data1/www/qwen3-asr-server

# 创建虚拟环境（Python 3.12）
uv venv --python 3.12

# 按 pyproject.toml + uv.lock 安装全部依赖
uv sync
```

依赖完全由 `pyproject.toml` 定义，`uv sync` 一键安装，无需单独 `pip install`。
`requirements.txt` 仅用于 Docker 镜像运行时的增量补装，本地部署不涉及。

**2. 配置环境变量**

```bash
cp .env.example .env   # 如不存在则手动创建
```

`.env` 必要配置项：

| 变量 | 本地部署取值 | 说明 |
|------|-------------|------|
| `PORT` | `9101`（按需修改） | 服务监听端口，无任何硬编码，随便改 |
| `CUDA_VISIBLE_DEVICES` | `7` | 指定使用的 GPU（本机为 8×RTX 4090） |
| `DEVICE` | `cuda:0` | 推理设备（单卡可见时即 `cuda:0`） |
| `HF_HUB_OFFLINE` | `1` | 离线模式，不走 HuggingFace 联网 |
| `HF_ENDPOINT` | `https://hf-mirror.com` | HuggingFace 镜像（下载模型时临时关离线） |
| `MODELSCOPE_CACHE` | `/data1/models/modelscope/hub` | ModelScope 缓存根目录（库内部再拼 `/models/<id>`） |
| `HF_HOME` | `/data1/models/huggingface` | HuggingFace 缓存路径 |
| `QWEN3_ASR_MODEL_PATH` | `/data1/models/Confucius4-R2T2` | ASR 主模型本地路径 |
| `QWEN3_FORCED_ALIGNER_PATH` | `/data1/models/Qwen3-ForcedAligner-0.6B` | 词级时间戳对齐器路径 |
| `API_KEY` | 与服务端鉴权一致 | 服务会校验此 Key |
| `QWEN_GPU_MEMORY_UTILIZATION` | `0.35` | 主引擎显存占用比（运维参数一律走 `.env`） |
| `QWEN_KV_CACHE_MEMORY_BYTES` | `1610612736` | KV cache 精确字节数（1.5GiB） |
| `QWEN_MAX_NUM_SEQS` | `5` | 最大并发序列数（5 并发） |
| `QWEN_MAX_NUM_BATCHED_TOKENS` | `2048` | 单批最大 token 数（= encoder cache 预算） |
| `QWEN_MAX_MODEL_LEN` | `4096` | 上下文长度 |
| `QWEN_ENFORCE_EAGER` | `true` | 关闭 CUDA graph/torch.compile，省约 0.5GB 显存 |
| `QWEN_FORCE_ALIGNER_GPU_MEMORY_UTILIZATION` | `0.20` | 对齐器显存占用比 |
| `QWEN_FORCE_ALIGNER_MAX_MODEL_LEN` | `2048` | 对齐器上下文长度 |
| `QWEN_WARMUP_FORCED_ALIGNER` | `false` | 不启动预热对齐器，改为首次词级时间戳请求时懒加载（启动显存 16GB→6GB 的关键） |
| `QWEN_STREAM_CHUNK_SEC` | `0.16` | 流式解码粒度（官方推荐的 80ms~2s 范围） |
| `QWEN_STREAM_LOOKAHEAD_SEC` | `0.16` | 首个 chunk 额外音频，帮助首段判断语言/专名 |
| `QWEN_STREAM_UNFIXED_CHUNK_NUM` | `0` | 前 N 个 chunk 不接历史 prefix |
| `QWEN_STREAM_UNFIXED_TOKEN_NUM` | `1` | prefix 回滚 token 数（官方默认 1） |
| `QWEN_STREAM_MAX_NEW_TOKENS` | `512` | 生成**上限**；实际按 chunk 时长动态取小值，0 表示不设上限 |
| `QWEN_STREAM_SEGMENT_MAX_SEC` | `60` | 单段最大时长兜底（主边界是 VAD 静音） |
| `QWEN_STREAM_SILENCE_SEC` | `2.0` | VAD 不可用时的能量阈值回退 |
| `QWEN_STREAM_OVERLAP_SEC` | `0` | 段间重叠音频；官方 append-only 不需要 |
| `QWEN_STREAM_VAD_MODEL_PATH` | `/data1/models/FireRedVAD/Stream-VAD` | 流式 VAD 模型目录（不存在则回退能量阈值） |
| `QWEN_STREAM_VAD_MIN_SILENCE_FRAMES` | `20` | 句尾静音帧数（10ms/帧，20 = 200ms） |
| `QWEN_STREAM_FINAL_PASS` | `true` | 段定稿时用离线路径重转（two-pass，保证热词/专名） |
| `QWEN_IDLE_UNLOAD_TIMEOUT` | `15` | 一级释放：空闲休眠超时（秒），0 禁用；`15` = 15 秒无请求即 sleep（权重 offload 到 CPU，唤醒 <1s） |
| `QWEN_SHUTDOWN_AFTER_S` | `600` | 二级释放：仍空闲超过该秒数则彻底 shutdown 子进程（连 sleep 残留的 CUDA context 一起归还），0 禁用；代价是下次请求重载 11~21s |
| `QWEN_ENABLE_SLEEP_MODE` | `true` | 启用 vLLM sleep mode（进程常驻、只归还显存）；`false` 回退为销毁子进程（重载 11~21s） |
| `QWEN_LOW_VRAM_THRESHOLD_GB` | `6.0` | 显存压力休眠阈值 |

> **路径说明**：模型统一放在 `/data1/models`（不放在项目目录内）。
> `models.json` 支持 `${VAR:-default}` 占位符，换模型/换路径只改 `.env`：
> `QWEN3_ASR_MODEL_PATH`（ASR 主模型）、`QWEN3_FORCED_ALIGNER_PATH`（对齐器）、
> `MODELSCOPE_CACHE`（ModelScope 根目录）、`HF_HOME`（HuggingFace 缓存）。
> `app/core/config.py` 的 `MODELSCOPE_PATH` = `$MODELSCOPE_CACHE/models`（ModelScope
> 库内部结构如此），仅在未设置 `MODELSCOPE_CACHE` 时用默认 `~/.cache/modelscope/hub/models`。

**3. 修复模型目录权限**

Docker 挂载的模型文件通常属 root，本地运行时需要修正：

```bash
sudo chown -R $(whoami):$(whoami) /data1/models/
sudo chown -R $(whoami):$(whoami) /data1/www/qwen3-asr-server/logs/
sudo chown -R $(whoami):$(whoami) /data1/www/qwen3-asr-server/temp/
sudo chown -R $(whoami):$(whoami) /data1/www/qwen3-asr-server/data/
```

**4. 启动验证**

先直接用命令行确认能启动，再注册 systemd 服务：

```bash
cd /data1/www/qwen3-asr-server
.venv/bin/python start.py
```

看到 `Worker [main] 已就绪` 且日志中 `extra_failed=0` 表示所有模型加载成功。

**5. 注册 systemd 服务**

```bash
sudo ln -sfn /data1/www/qwen3-asr-server/qwen3-asr.service /etc/systemd/system/qwen3-asr.service
sudo systemctl daemon-reload
sudo systemctl enable qwen3-asr.service
sudo systemctl start qwen3-asr.service
```

> 用**软链接**而不是 `cp`：仓库里的 `qwen3-asr.service` 成为唯一来源，
> 以后改 unit（端口无关，主要是 GPU、路径、限流参数）只需 `daemon-reload`，
> 不会再出现仓库与 `/etc/systemd/system` 两份文件不一致的问题。

服务配置要点：
- unit 文件用**软链接**安装（见上），改仓库中的 unit 后 `daemon-reload` 即生效
- `Restart=on-failure`：进程异常退出自动重启，间隔 10s
- `StartLimitBurst=3` / `StartLimitIntervalSec=300`（写在 `[Unit]` 段）：5 分钟内重启超过 3 次则停止重试；调试期间频繁 `restart` 会触发 `start-limit-hit`，用 `sudo systemctl reset-failed qwen3-asr` 清除后重试
- `.env` 通过 `start.py` 中的 `load_dotenv()` 加载，不需要 `EnvironmentFile=`
- GPU 设备由 `.env` 的 `CUDA_VISIBLE_DEVICES` 驱动（unit 里**不要**再写 `Environment="CUDA_VISIBLE_DEVICES=..."`，因为 systemd 的 `Environment=` 优先级高于 `.env`，会把 `.env` 里的值架空）

**6. 日常运维**

```bash
# 查看状态
sudo systemctl status qwen3-asr

# 查看日志
sudo journalctl -u qwen3-asr -f

# 重启服务（代码 bind mount 已生效时）
sudo systemctl restart qwen3-asr

# 健康检查
curl -s http://127.0.0.1:9101/stream/v1/asr/health \
  -H "Authorization: Bearer $API_KEY"

# 服务不断重启时重置计数
sudo systemctl reset-failed qwen3-asr
```

**7. 从 Docker 迁移到本地**

如果之前使用 Docker 运行，迁移只需两步：

```bash
# ① 修正模型文件权限
sudo chown -R $(whoami):$(whoami) /data1/models/

# ② 确保 CAM++ 子模型路径正确
# 启动时 fix_camplusplus_config() 会自动修正 configuration.json 中的旧路径
# （支持旧 Docker 路径，以及缓存目录多一层 models 的情况）
```

其余无需变动：模型缓存、声纹数据库、`.env` 配置均兼容。

### macOS / Apple Silicon 本地部署

适用于 M1/M2/M3/M4 机器上的本地 Qwen3-ASR 推理。当前 macOS 已统一走 vendored QwenASR Rust CPU backend。

```bash
./scripts/sync_cpu_env.sh
source .venv/bin/activate
python start.py
```

### 验证部署

```bash
# 健康检查
curl http://localhost:9101/stream/v1/asr/health

# 查看可用模型
curl http://localhost:9101/stream/v1/asr/models

# 测试语音识别（阿里云协议）
curl -X POST "http://localhost:9101/stream/v1/asr" \
  -H "Content-Type: application/octet-stream" \
  --data-binary @test.wav

# 测试 OpenAI 兼容接口
curl -X POST "http://localhost:9101/v1/audio/transcriptions" \
  -H "Authorization: Bearer any" \
  -F "file=@test.wav" \
  -F "model=qwen3-asr-confucius4-r2t2"
```

### 引擎加载容错说明

vLLM 引擎初始化在特定宿主环境（如 CPU 电压不稳、驱动兼容问题）下可能偶发段错误（SIGSEGV）。服务内置两层容错：

1. **懒加载自动重试**：引擎被空闲卸载后，请求触发懒加载时创建失败会自动重试 3 次（间隔 5s/10s 递增），提高恢复概率。
2. **预加载失败降级启动**：启动时 qwen3 引擎预加载失败不再拒绝启动（降级继续，funasr 等其余引擎照常服务），失败引擎在后续请求时由懒加载路径自动重试恢复。

### 显存策略

默认模型为 **Confucius4-R2T2**，配合「forced aligner 懒加载 + KV cache 精确控制」，启动显存已从 16GB 降到约 6GB（含基线的 nvidia-smi 读数）。实测（RTX 4090 24GB，`CUDA_VISIBLE_DEVICES=7`）：

| 场景 | 显存 |
|------|------|
| 基线（空载） | 0.86GB |
| 仅 ASR 引擎 | ~6.0GB |
| 完整服务（+ VAD / CAM++ / Paraformer / 标点） | ~8.7GB |
| 显式请求词级时间戳（+ aligner） | ~12GB（峰值） |

压缩手段（全部走 `.env`，不写死在 `models.json`）：

- `QWEN_WARMUP_FORCED_ALIGNER=false`：对齐器（独立 vLLM 实例）不在启动时预热，首次 `word_timestamps=true` 请求才加载；`word_timestamps=true` 在首次加载后 aligner 会自动重试最大 20 次（间隔 10s）
- `QWEN_KV_CACHE_MEMORY_BYTES=1610612736`：精确控制 KV cache，避免按 utilization 吃满显存
- `QWEN_MAX_NUM_BATCHED_TOKENS=2048`：直接决定 encoder cache 预算
- `QWEN_ENFORCE_EAGER=true`：关闭 CUDA graph / torch.compile，省约 0.5GB

服务按两级策略归还显存，让位给同一 GPU 上的其他服务（如 ComfyUI）：

1. **空闲休眠（一级）**：`QWEN_IDLE_UNLOAD_TIMEOUT` 秒无请求即 `sleep`（当前生产设 `15`）——权重 offload 到 CPU、丢弃 KV cache，**vLLM 子进程常驻**，下次请求 `wake_up` 仅 0.1~0.4s（实测）。关闭 `QWEN_ENABLE_SLEEP_MODE` 则回退为旧行为（销毁子进程，重载 11~21s）。
2. **长时间空闲彻底卸载（二级）**：仍空闲超过 `QWEN_SHUTDOWN_AFTER_S`（默认 `600` = 10 分钟）则 shutdown 子进程，把 sleep 残留的 CUDA context（两引擎合计约 1.9GB）也归还；下次请求走懒加载（11~21s）。设 `0` 禁用二级。
3. **显存压力休眠**：GPU 可用显存低于 `QWEN_LOW_VRAM_THRESHOLD_GB`（默认 6GB）且引擎空闲超过 `QWEN_VRAM_PRESSURE_COOLDOWN_S`（默认 60s）时休眠引擎（日志标记 `(vram pressure)`）；不支持 sleep 的引擎回退为卸载。

> `forced aligner` 有独立的活跃计数与空闲计时：忙时绝不释放，也不会被主引擎连坐；声纹（CAM++）推理前会主动让 vLLM 引擎休眠以避让显存。
> 启用 sleep 时项目固定传 `mm_processor_cache_gb=0`：否则唤醒后重复音频会命中失效的 multimodal cache 并报 `AssertionError`。

## 从源码构建镜像

### 使用构建脚本

项目提供了一个更薄的 `build.sh` 包装层，用于统一 `docker buildx` 参数：

```bash
# 构建所有版本（CPU + GPU）
./build.sh

# 仅构建 GPU 版本
./build.sh -t gpu

# 构建指定版本并推送
./build.sh -t all -v 1.0.3 -p

# 查看帮助
./build.sh -h
```

**构建脚本参数：**

| 参数 | 说明 | 默认值 |
|------|------|--------|
| `-a, --arch` | 目标架构: `amd64`, `arm64`, `multi` | `amd64` |
| `-t, --type` | 构建类型: `cpu`, `gpu`, `all` | `all` |
| `-v, --version` | 版本标签 | `latest` |
| `-p, --push` | 构建后推送到 Docker Hub | 否 |
| `-e, --export` | 导出单架构镜像为 tar.gz | 否 |
| `-o, --output` | 导出目录 | `.` |
| `-r, --registry` | 镜像仓库 | `quantatrisk` |
| `-n, --no-cache` | 禁用 Docker 构建缓存 | 否 |

### 手动构建

```bash
# 构建 CPU 版本
docker build -t qwen3-asr:cpu-latest -f Dockerfile.cpu .

# 构建绑定当前机器指令集的 CPU 版本（仅适合同构部署）
docker build -t qwen3-asr:cpu-native -f Dockerfile.cpu \
  --build-arg QWENASR_RUST_TARGET_CPU=native \
  .

# 构建默认 GPU 版本（CUDA 12.8 / PyTorch cu128）
docker build -t qwen3-asr:gpu-latest -f Dockerfile.gpu .

# 构建 CUDA 12.6 版本
docker build -t qwen3-asr:gpu-cu126 -f Dockerfile.gpu \
  --build-arg PYTORCH_BASE_IMAGE=pytorch/pytorch:2.10.0-cuda12.6-cudnn9-runtime \
  --build-arg PYTORCH_CUDA_INDEX=https://download.pytorch.org/whl/cu126 \
  --build-arg CUDA_NVCC_PACKAGE=cuda-nvcc-12-6 \
  --build-arg TORCH_CUDA_ARCH_LIST="8.0;8.6;8.9" \
  .

# 构建 CUDA 13.0 版本
docker build -t qwen3-asr:gpu-cu130 -f Dockerfile.gpu \
  --build-arg PYTORCH_BASE_IMAGE=pytorch/pytorch:2.10.0-cuda13.0-cudnn9-runtime \
  --build-arg PYTORCH_CUDA_INDEX=https://download.pytorch.org/whl/cu130 \
  --build-arg CUDA_NVCC_PACKAGE=cuda-nvcc-13-0 \
  --build-arg TORCH_CUDA_ARCH_LIST="12.0+PTX" \
  .
```

`Dockerfile.cpu` 可覆盖的 CPU 构建参数：

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `QWENASR_RUST_TARGET_CPU` | `x86-64-v2` | amd64 Rust backend 编译目标；可设为 `native` 构建绑定当前 CPU 的镜像 |

`Dockerfile.gpu` 可覆盖的 GPU 构建参数：

| 参数 | 默认值 | 用途 |
|------|--------|------|
| `PYTORCH_BASE_IMAGE` | `pytorch/pytorch:2.10.0-cuda12.8-cudnn9-runtime` | 选择 PyTorch/CUDA 基础镜像 |
| `PYTORCH_CUDA_INDEX` | `https://download.pytorch.org/whl/cu128` | 选择 PyTorch wheel CUDA 后端 |
| `CUDA_NVCC_PACKAGE` | `cuda-nvcc-12-8` | 安装匹配的 nvcc，用于 vLLM/FlashInfer JIT |
| `TORCH_CUDA_ARCH_LIST` | `12.0+PTX` | 指定 JIT 编译目标架构 |
| `VLLM_PACKAGE` | `vllm[audio]==0.19.0` | 覆盖 vLLM 包版本或来源 |

### 生产部署到其他服务器

**依赖会自动包含**：Dockerfile 使用 `uv sync --frozen` 按 `pyproject.toml` +
`uv.lock` 安装全部依赖，因此重建镜像后 `sqlite-vec`（声纹向量库）等新依赖
已内置，无需手工安装。

**生产环境不要挂载源码卷**。仓库内 `docker-compose.yml` 的
`./app:/app/app` 挂载是本地开发模式（改代码即生效）；生产部署请使用本地
构建的镜像，代码以镜像内为准：

```yaml
# docker-compose.prod.yml（生产示例）
services:
  qwen3-asr:
    image: qwen3-asr:gpu-latest          # 本地构建的镜像 tag
    container_name: qwen3-asr
    ports:
      - "${NGINX_PORT:-9101}:8000"
    volumes:
      # 仅挂载运行时数据目录，不挂载源码
      - ./temp:/app/temp
      - ./logs:/app/logs
      - ./data:/app/data
      - ./models/modelscope:/root/.cache/modelscope
      - ./models/huggingface:/root/.cache/huggingface
    runtime: nvidia
    environment:
      API_KEY: ${API_KEY:-}
      HF_HUB_OFFLINE: ${HF_HUB_OFFLINE:-0}
      HF_ENDPOINT: ${HF_ENDPOINT:-}
      QWEN3_ASR_MODEL: ${QWEN3_ASR_MODEL:-}
      CUDA_VISIBLE_DEVICES: ${CUDA_VISIBLE_DEVICES:-0}
      QWEN_GPU_MEMORY_UTILIZATION: ${QWEN_GPU_MEMORY_UTILIZATION:-}
      QWEN_FORCE_ALIGNER_GPU_MEMORY_UTILIZATION: ${QWEN_FORCE_ALIGNER_GPU_MEMORY_UTILIZATION:-}
      QWEN_IDLE_UNLOAD_TIMEOUT: ${QWEN_IDLE_UNLOAD_TIMEOUT:-300}
      VOICEPRINT_ENABLED: ${VOICEPRINT_ENABLED:-true}
      VOICEPRINT_DB_PATH: ${VOICEPRINT_DB_PATH:-./data/voiceprints.sqlite3}
      VOICEPRINT_MATCH_THRESHOLD: ${VOICEPRINT_MATCH_THRESHOLD:-0.70}
      VOICEPRINT_DEVICE: ${VOICEPRINT_DEVICE:-auto}
    restart: unless-stopped
```

部署清单（新服务器功能一致的四个条件）：

```bash
# 1. 拉代码并切到目标分支（含 pyproject.toml 新依赖）
git clone <your-repo> && git checkout feature/voiceprint

# 2. 重建镜像（uv sync 自动包含 sqlite-vec 等依赖）
#    注意：新增的 av（PyAV，音频解码兑底）属 C 扩展，不在 requirements.txt 的运行时
#    增量安装列表里，必须 rebuild 镜像才能生效（未 rebuild 时仍有 ffmpeg CLI 兑底）
./build.sh -t gpu -v latest

# 3. 准备 .env（端口、显存控制、声纹、鉴权）
cp .env.example .env && vim .env

# 4. 迁移声纹注册数据（可选：保留已注册的说话人名字）
scp ./data/voiceprints.sqlite3 user@new-server:/opt/qwen3-asr/data/

# 5. 模型缓存：在线环境启动时自动下载；离线环境用
./scripts/prepare-models.sh   # 打包后 scp 到新服务器解压
```

**声纹数据库（`./data/voiceprints.sqlite3`）只存 embedding 向量与显示名，
不含音频本身**；不迁移则需在新服务器重新注册说话人。

### 模型下载

启动时会先检测当前运行计划所需模型；如果本地缓存缺失，会自动下载。离线部署可显式设置 `HF_HUB_OFFLINE=1` 并提前准备模型缓存。
手动准备方式：

```bash
# 交互式导出当前运行计划所需模型
./scripts/prepare-models.sh

# 或直接使用项目 CLI
uv run python -m app.utils.download_models
uv run python -m app.utils.download_models --export-dir ./models
```

**模型目录结构**（统一放在 `/data1/models`，不放在项目目录内）：

```text
/data1/models/
├── Confucius4-R2T2/            # 默认 ASR 主模型 ~3.9GB（bf16）
├── Qwen3-ForcedAligner-0.6B/   # 词级时间戳对齐器 ~1.8GB（懒加载）
├── huggingface/                # HuggingFace 缓存（HF_HOME）
└── modelscope/
    └── hub/models/             # MODELSCOPE_CACHE=/data1/models/modelscope/hub
        ├── iic/speech_paraformer-large_...          # 中文实时流式 ~849MB
        ├── iic/punc_ct-transformer_...（×2）        # 离线/实时标点
        ├── iic/speech_campplus_speaker-diarization_common  # 说话人分离
        ├── damo/speech_campplus_sv_...              # 声纹验证（声纹匹配）
        ├── damo/speech_campplus-transformer_scl_... # 说话人聚类
        └── damo/speech_fsmn_vad_...                 # VAD ~69MB
```

> 完整模型用途与使用场景见 README「模型清单」章节。换模型或换路径只改 `.env`
> 的 `QWEN3_ASR_MODEL_PATH` / `QWEN3_FORCED_ALIGNER_PATH` / `MODELSCOPE_CACHE` /
> `HF_HOME`（`models.json` 用 `${VAR:-default}` 占位符引用），无需改代码。

离线部署时，把上述 `/data1/models` 整体拷贝到目标机器，并在 `.env` 中把
`QWEN3_ASR_MODEL_PATH` / `QWEN3_FORCED_ALIGNER_PATH` / `MODELSCOPE_CACHE` /
`HF_HOME` 指向对应路径即可。

## 环境变量配置

复制 `.env.example` 为 `.env`：**`.env` 是运行时配置（含所有运维调优参数）的唯一来源**，不需要改 `models.json` 或代码。端口、显存、并发、卸载策略全部在此调整，改完重启服务即生效（`start.py` 每次启动都执行 `load_dotenv()`）。

### 声纹数据库配置

| 环境变量 | 默认值 | 说明 |
|----------|--------|------|
| `VOICEPRINT_ENABLED` | `true` | 是否启用 ASR 结果声纹身份匹配 |
| `VOICEPRINT_DB_PATH` | `./data/voiceprints.sqlite3` | SQLite + sqlite-vec 声纹数据库路径 |
| `VOICEPRINT_MATCH_THRESHOLD` | `0.70` | 说话人身份匹配阈值 |
| `VOICEPRINT_DEVICE` | `auto` | 声纹推理设备。默认 `auto`：有 CUDA 用 GPU（分段已限长 ≤30s，峰值约 840MiB），无 CUDA 降级 CPU（慢 5~8 倍） |

启用后通过 `/api/v1/voiceprint-speakers` 注册说话人（名字 + 单人音频样本），
ASR 转写结果中匹配到的 `speaker_id` 会自动替换为注册名；未注册或匹配不
确定的说话人保留原始标签。接口与匹配策略详见
[voiceprint-architecture.md](voiceprint-architecture.md)。

### 热词上下文配置

| 环境变量 | 默认值 | 说明 |
|----------|--------|------|
| `ASR_DEFAULT_HOTWORDS` | （空） | 服务端预设热词（英文逗号分隔），与请求热词（OpenAI `prompt` / 原生 `vocabulary_id` / 流式 WebSocket `context`）合并后注入识别上下文；建议 ≤512 字符 |

热词用**英文逗号**分隔（如 `阿里巴巴,OpenAI,Bank of China`），短语内部空格保留
（`Bank of China` 整体作为一个热词）。热词为"倾向性采纳"而非强制替换，建议只收录
高频关键术语（公司名、行业专有名词、英文缩写等）。修改 `.env` 后重启服务即生效
（`start.py` 每次启动都会执行 `load_dotenv()` 重读 `.env`）。

**注入方式（2026-10-18 对齐官方）**：热词现在按 Confucius4-R2T2 官方约定
**直接作为 system prompt 原文**，不再包一层 "Transcribe the speech... / Use this
context..." 指令；强制语言时在生成提示后追加 `language X<asr_text>`。实测同样
40s 英文音频 + 热词 `GPT, transcription`：旧写法输出全部 `gbt`（热词无效），
官方写法输出 `gpt`（热词生效）。

**合并规则**：离线（`/v1/audio/transcriptions` 的 `prompt`、`/stream/v1/asr` 的
`vocabulary_id`）与流式（`/ws/v1/asr/qwen` 的 `context`）都按
`merge_hotwords(ASR_DEFAULT_HOTWORDS, 请求热词)` 合并：默认词在前、去重、保留短语内空格。

**流式热词限制**：流式为 append-only，首段一旦输出错误（如 `gbt`）后续不会改写。
`QWEN_STREAM_FINAL_PASS=true` 会在段定稿时用离线路径对该段音频重转，可校正
后续段的专名；首段若音频本身不清晰则无法完全修复，建议对热词要求高的场景
使用离线接口或保留 two-pass。

**稀释与加权（实测经验，2026-08-13）：**
- 列表过长会稀释单个热词的偏向权重：实测 40+ 词列表中的 `Dazed` 被识别成"电视"，
  精简到 16 个高频词或对关键词重复加权后恢复正常
- 关键专有词可在**单次请求**的 `prompt` 中重复 2-3 遍加权（重复项用空格连成一个
  短语，避免与预设列表去重），如 `prompt="Dazed Dazed Dazed"`（OpenAI 接口）或
  `vocabulary_id=Dazed Dazed Dazed`（原生接口）
- 热词拼写必须准确（大小写、空格），拼错的词等于无效词（如 `Globa Vision` 应为
  `Global Vision`）

## 服务监控

### 健康检查

```bash
curl http://localhost:9101/stream/v1/asr/health
```

### 日志监控

```bash
# 实时查看日志
docker logs -f qwen3-asr

# 查看错误日志
docker logs qwen3-asr 2>&1 | grep -i error
```

### 资源监控

```bash
# 容器资源使用
docker stats qwen3-asr

# GPU 使用情况
docker exec -it qwen3-asr nvidia-smi
```

## Docker 常用命令

> 以下命令使用 Compose v2 写法（`docker compose`）。若本机只安装了旧版
> `docker-compose`（v1），把 `docker compose` 替换为 `docker-compose` 即可，
> 两者功能等价。

### 服务生命周期

```bash
# 启动服务（后台运行；已存在且配置未变的容器会直接复用）
docker compose up -d

# 配置变更后强制重建（改端口/环境变量/挂载/镜像时必须用）
docker compose up -d --force-recreate

# 停止容器进程（保留容器，下次 start 秒级恢复；仅适用于配置未变的场景）
docker compose stop
docker compose start

# 重启容器进程
docker compose restart

# 停止并删除容器与默认网络（数据卷与挂载目录保留）
docker compose down

# 停止并删除容器、网络、匿名卷（慎用，会丢容器内未挂载的数据）
docker compose down -v
```

**区分两个概念**：`stop/start` 只是重启**同一个容器**，创建时固化的端口、
环境变量、挂载定义都不会更新；`up --force-recreate` 会**销毁旧容器并按最新
配置重建**。改了 `docker-compose.yml` 或 `.env` 中的端口、环境变量、volumes
等容器级配置后，必须用 `--force-recreate`（或 `down` + `up -d`）才能生效。

### 状态与日志

```bash
# 查看容器状态与端口映射
docker compose ps

# 实时查看日志
docker compose logs -f qwen3-asr

# 查看最近 N 行日志
docker compose logs --tail 100 qwen3-asr

# 查看错误日志
docker compose logs qwen3-asr 2>&1 | grep -i error
```

### 容器内操作

```bash
# 进入容器终端
docker compose exec -T qwen3-asr bash

# 在容器内执行单条命令
docker compose exec -T qwen3-asr nvidia-smi
docker compose exec -T qwen3-asr printenv QWEN_GPU_MEMORY_UTILIZATION

# 宿主机 <-> 容器复制文件
docker compose cp qwen3-asr:/app/demo/demo.mp4 /tmp/demo.mp4
docker compose cp ./data/file.wav qwen3-asr:/tmp/
```

### 镜像管理

```bash
# 查看本地镜像
docker images

# 删除不再使用的悬空镜像
docker image prune

# 删除指定镜像（先确认无容器使用）
docker rmi quantatrisk/qwen3-asr:gpu-latest
```

### GPU 状态

```bash
# 宿主机查看 GPU 与显存占用（含各进程明细）
nvidia-smi

# 只看显存占用数值（脚本场景）
nvidia-smi --query-gpu=memory.used --format=csv,noheader
```

## 资源需求

### 最小配置（CPU 版本）

- CPU: 4 核
- 内存: 16GB
- 磁盘: 20GB

### 推荐配置（GPU 版本）

- CPU: 4 核
- 内存: 16GB
- GPU: NVIDIA GPU (8GB+ 显存；启用词级时间戳约需 12GB)
- 磁盘: 20GB

## 故障排除

### 常见问题

| 问题 | 症状 | 解决方案 |
|------|------|----------|
| GPU 内存不足 | CUDA OOM 错误 | 调低 `QWEN_KV_CACHE_MEMORY_BYTES` / `QWEN_MAX_NUM_SEQS`，或使用 `qwen3-asr-0.6b`，或部署 CPU 镜像 |
| 模型加载失败 / 缓慢 | 本地模型缓存缺失 | 先运行 `./scripts/prepare-models.sh` 或 `uv run python -m app.utils.download_models` 预准备模型 |
| 端口被占用 | 端口冲突错误 | 修改端口映射：`"8080:8000"` |
| 说话人分离失败 | CAM++ 模型错误 | 检查模型是否完整下载，显存是否充足 |

### 调试模式

```bash
# 启用调试模式
docker run -e DEBUG=true -e LOG_LEVEL=DEBUG ...

# 进入容器调试
docker exec -it qwen3-asr /bin/bash
```

## 更新服务

```bash
# 拉取最新镜像（GPU 版本）
docker pull quantatrisk/qwen3-asr:gpu-latest

# 拉取最新镜像（CPU 版本）
docker pull quantatrisk/qwen3-asr:cpu-latest

# 重启服务
docker compose down && docker compose up -d
```
