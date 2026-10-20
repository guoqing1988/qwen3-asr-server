# -*- coding: utf-8 -*-
"""
统一配置管理
ASR语音识别配置选项
"""

import os
from typing import Optional
from pathlib import Path


class Settings:
    """统一应用配置类"""

    # 应用信息
    APP_NAME: str = "Qwen3-ASR Server"
    APP_VERSION: str = "1.0.3"
    APP_DESCRIPTION: str = "Qwen3-ASR speech recognition API service"

    # 服务器配置
    HOST: str = "0.0.0.0"
    PORT: int = 8000
    DEBUG: bool = False

    # 鉴权配置
    API_KEY: Optional[str] = None  # 从环境变量API_KEY读取，如果为None则鉴权可选

    # 设备配置
    DEVICE: str = "auto"  # auto, cpu, cuda:0, npu:0

    # 路径配置
    BASE_DIR: Path = Path(__file__).parent.parent.parent
    TEMP_DIR: str = "temp"
    # ModelScope 模型目录。与 modelscope 库语义保持一致：
    # 库实际下载到 ``$MODELSCOPE_CACHE/models/<model_id>``，
    # 因此这里默认取 ``$MODELSCOPE_CACHE/models``（未设置时为 ~/.cache/modelscope/hub/models）。
    # 也允许用 ``MODELSCOPE_PATH`` 显式指定最终目录。
    MODELSCOPE_PATH: str = os.getenv(
        "MODELSCOPE_PATH",
        os.path.join(
            os.getenv(
                "MODELSCOPE_CACHE", os.path.expanduser("~/.cache/modelscope/hub")
            ),
            "models",
        ),
    )

    # 日志配置
    LOG_LEVEL: str = "INFO"
    LOG_FILE: Optional[str] = str(BASE_DIR / "logs" / "qwen3-asr.log")
    LOG_MAX_BYTES: int = 20 * 1024 * 1024  # 20MB
    LOG_BACKUP_COUNT: int = 50  # 保留50个备份文件

    FUNASR_AUTOMODEL_KWARGS = {
        "trust_remote_code": False,
        "disable_update": True,
        "disable_pbar": True,
        "disable_log": True,  # 禁用FunASR的tables输出
        "local_files_only": True,  # 强制使用本地模型，禁止联网下载
    }
    ASR_MODELS_CONFIG: str = str(BASE_DIR / "app/services/asr/models.json")
    VAD_MODEL: str = "damo/speech_fsmn_vad_zh-cn-16k-common-pytorch"
    PUNC_MODEL: str = "iic/punc_ct-transformer_zh-cn-common-vocab272727-pytorch"
    PUNC_REALTIME_MODEL: str = (
        "iic/punc_ct-transformer_zh-cn-common-vad_realtime-vocab272727"
    )

    # 流式ASR远场过滤配置
    ASR_ENABLE_NEARFIELD_FILTER: bool = True  # 是否启用远场声音过滤
    ASR_NEARFIELD_RMS_THRESHOLD: float = 0.01  # RMS能量阈值（宽松模式，适合大多数场景）
    # 音频处理配置
    MAX_AUDIO_SIZE: int = 2048 * 1024 * 1024  # 2GB

    # 批处理推理配置（GPU 真并行）
    ASR_BATCH_SIZE: int = 4  # ASR 批处理大小（同时推理的片段数），建议 2-8

    # 服务端预设热词（英文逗号分隔，与请求热词合并后注入识别上下文；建议 ≤512 字符）
    ASR_DEFAULT_HOTWORDS: str = ""

    # Nemotron-3 说话人分离模型目录（帧级、支持重叠，8 说话人上限）
    NEMOTRON_MODEL_PATH: str = "/data1/models/Nemotron-3-Diarization"

    # 离线转写后补句末标点（R2T2 段尾常缺标点；复用 FunASR 标点模型）
    ASR_RESTORE_SENTENCE_ENDING: bool = True

    # 音频分段配置
    MAX_SEGMENT_SEC: float = 60.0  # Max offline ASR segment duration in seconds.

    # Runtime 并发配置（按 backend 独立控制）
    QWEN_RUST_CPU_WORKERS: int = 4
    FUNASR_WORKERS: int = 1

    # 空闲自动归还显存配置（秒；超过该时长无请求则 sleep/卸载 vLLM 引擎释放显存，0=禁用）
    QWEN_IDLE_UNLOAD_TIMEOUT: int = 300
    # sleep mode：启用后空闲时把权重 offload 到 CPU（进程常驻，唤醒 <1s）；
    # 关闭则回退为销毁子进程（重载需十余秒）
    QWEN_ENABLE_SLEEP_MODE: bool = True
    # sleep 之后多久彻底卸载（秒，0=禁用）：sleep 会留下 CUDA context 等残留
    # （两个引擎合计约 1.9GB），长时间空闲改走 shutdown 可连残留一起归还，
    # 代价是下次请求需重新加载（11~21s）
    QWEN_SHUTDOWN_AFTER_S: int = 600

    # 显存压力卸载配置：可用显存低于阈值且引擎空闲时让位其他 GPU 服务
    QWEN_LOW_VRAM_THRESHOLD_GB: float = 15.0
    QWEN_VRAM_PRESSURE_COOLDOWN_S: float = 60.0

    # vLLM 显存利用率（0.0-1.0）。None 表示按模型权重文件大小自动估算。
    # 属于部署/运维参数，只从环境变量读取，不写入 models.json。
    QWEN_GPU_MEMORY_UTILIZATION: Optional[float] = None
    # forced aligner 的显存利用率；None 表示继承主模型配置
    QWEN_FORCE_ALIGNER_GPU_MEMORY_UTILIZATION: Optional[float] = None

    # --- vLLM 运行时调优（部署/运维参数，均经 .env 配置）---
    # 最大并发序列数（同时处理的请求数）
    QWEN_MAX_NUM_SEQS: Optional[int] = None
    # 单批最大 token 数；vLLM 中它同时决定音频 encoder cache 预算，调小可省显存
    QWEN_MAX_NUM_BATCHED_TOKENS: Optional[int] = None
    # 覆盖 models.json 中的 max_model_len
    QWEN_MAX_MODEL_LEN: Optional[int] = None
    # 禁用 CUDA graph 以省显存（吞吐略降）
    QWEN_ENFORCE_EAGER: bool = False
    # 直接指定 KV cache 字节数（比 gpu_memory_utilization 更省），None 表示不用
    QWEN_KV_CACHE_MEMORY_BYTES: Optional[int] = None
    # forced aligner 的 max_model_len
    QWEN_FORCE_ALIGNER_MAX_MODEL_LEN: Optional[int] = None
    # ASR 推理后端：vllm（默认，bf16）| llamacpp（官方 GGUF Q8，低显存）
    ASR_ENGINE: str = "vllm"
    # llama.cpp Q8 后端（ASR_ENGINE=llamacpp 时生效）
    QWEN_LLAMACPP_MODEL_PATH: str = (
        "/data1/models/Confucius4-R2T2-GGUF/Confucius4-R2T2-Q8_0.gguf"
    )
    QWEN_LLAMACPP_MMPROJ_PATH: str = (
        "/data1/models/Confucius4-R2T2-GGUF/mmproj-Confucius4-R2T2-Q8_0.gguf"
    )
    QWEN_LLAMACPP_SERVER_BIN: str = "/data1/www/llama.cpp/build/bin/llama-server"
    # llama-server 监听端口；0 表示自动选择空闲端口
    QWEN_LLAMACPP_PORT: int = 18090
    QWEN_LLAMACPP_CTX: int = 8192
    QWEN_LLAMACPP_PARALLEL: int = 1
    QWEN_LLAMACPP_GPU_LAYERS: int = 99
    # 单次送入 llama.cpp 的最大音频时长（实测单次上限约 60s，取 25s 留余量）
    QWEN_LLAMACPP_MAX_SEGMENT_SEC: float = 25.0
    # 流式解码模式：append=token 级 append-only（默认，已发布文本不回撤）；redecode=每秒全量重解码
    QWEN_LLAMACPP_STREAM_MODE: str = "append"
    # 流式 token 回滚用的 tokenizer（默认复用 HF 原模型目录）
    QWEN_LLAMACPP_TOKENIZER_PATH: str = "/data1/models/Confucius4-R2T2"
    # 流式识别：单 chunk 生成上限（配置值作为硬上限，实际按 chunk 时长动态取小值）
    QWEN_STREAM_MAX_NEW_TOKENS: int = 512
    # 流式识别：解码粒度（秒）。官方 R2T2 推荐 0.16（80ms~2s 可调）
    QWEN_STREAM_CHUNK_SEC: float = 0.16
    # 流式识别：首个 chunk 额外 lookahead 音频（秒），用于首段语言/专名判断
    QWEN_STREAM_LOOKAHEAD_SEC: float = 0.16
    # 流式识别：前 N 个 chunk 不接历史 prefix（官方默认 0）
    QWEN_STREAM_UNFIXED_CHUNK_NUM: int = 0
    # 流式识别：prefix 回滚 token 数（官方默认 1）
    QWEN_STREAM_UNFIXED_TOKEN_NUM: int = 1
    # 流式识别：单段最大时长（秒）。超过即落定该段并开新段
    QWEN_STREAM_SEGMENT_MAX_SEC: float = 60.0
    # 流式识别：能量阈值回退时的静音切段阈值（秒），VAD 不可用时生效
    QWEN_STREAM_SILENCE_SEC: float = 2.0
    # 流式识别：段间重叠音频（秒）。官方 append-only 不需要重叠，默认关闭
    QWEN_STREAM_OVERLAP_SEC: float = 0.0
    # 流式 VAD：FireRedVAD Stream-VAD 模型目录（不存在时回退能量阈值）
    QWEN_STREAM_VAD_MODEL_PATH: str = "/data1/models/FireRedVAD/Stream-VAD"
    # 流式 VAD：连续静音帧数（10ms/帧，官方 ws_server 用 20；项目转录场景
    # 取 50（500ms）避免中文句内短停顿把一句话切碎）
    QWEN_STREAM_VAD_MIN_SILENCE_FRAMES: int = 50
    # 流式：段定稿时用离线路径重转（two-pass，提升热词/专名；关闭可省一次推理）
    QWEN_STREAM_FINAL_PASS: bool = True
    # Forced aligner（词级时间戳）配置
    # 是否在启动完整性检查中要求 aligner 模型存在
    QWEN_FORCED_ALIGNER_ENABLED: bool = True
    # 是否在引擎加载时预热 aligner（默认 False：请求 word_timestamps 时才懒加载，节省显存）
    QWEN_WARMUP_FORCED_ALIGNER: bool = False

    # Voiceprint identification configuration.
    VOICEPRINT_ENABLED: bool = True
    VOICEPRINT_DB_PATH: str = str(BASE_DIR / "data" / "voiceprints.sqlite3")
    VOICEPRINT_MATCH_THRESHOLD: float = 0.70
    # 声纹推理设备：默认 auto（有 CUDA 用 GPU，否则 CPU）。
    # 分段提取保证单次输入 ≤ VOICEPRINT_MAX_CHUNK_SEC(30s)，GPU 峰值显存实测约 840MiB；
    # 而把数小时拼接音频整体送入会爆 17GB+（CUDA OOM 且 reserved 不归还）。
    # CPU 慢 5-8 倍（30s 音频 291ms vs GPU 37ms），无 GPU 环境自动降级。
    VOICEPRINT_DEVICE: str = "auto"

    def __init__(self):
        """从环境变量读取配置"""
        self._load_from_env()
        self._ensure_directories()

    def _load_from_env(self):
        """从环境变量加载配置"""
        # 服务器配置
        self.HOST = os.getenv("HOST", self.HOST)
        self.PORT = int(os.getenv("PORT", str(self.PORT)))
        self.DEBUG = os.getenv("DEBUG", "false").lower() == "true"

        # 日志配置
        self.LOG_LEVEL = os.getenv("LOG_LEVEL", self.LOG_LEVEL)
        self.LOG_FILE = os.getenv("LOG_FILE", self.LOG_FILE)
        self.LOG_MAX_BYTES = int(os.getenv("LOG_MAX_BYTES", str(self.LOG_MAX_BYTES)))
        self.LOG_BACKUP_COUNT = int(
            os.getenv("LOG_BACKUP_COUNT", str(self.LOG_BACKUP_COUNT))
        )

        # 鉴权配置：空值/空白统一视为未配置
        self.API_KEY = (os.getenv("API_KEY") or "").strip() or None

        # 设备配置
        self.DEVICE = os.getenv("DEVICE", self.DEVICE)

        # 远场过滤配置
        self.ASR_ENABLE_NEARFIELD_FILTER = (
            os.getenv("ASR_ENABLE_NEARFIELD_FILTER", "true").lower() == "true"
        )
        self.ASR_NEARFIELD_RMS_THRESHOLD = float(
            os.getenv(
                "ASR_NEARFIELD_RMS_THRESHOLD", str(self.ASR_NEARFIELD_RMS_THRESHOLD)
            )
        )

        # 音频处理配置
        # 支持简化格式：纯数字表示MB，或带单位（如 2048MB, 2GB）
        max_audio_size_str = os.getenv("MAX_AUDIO_SIZE")
        if max_audio_size_str:
            self.MAX_AUDIO_SIZE = self._parse_size(max_audio_size_str)

        self.ASR_BATCH_SIZE = int(
            os.getenv("ASR_BATCH_SIZE", str(self.ASR_BATCH_SIZE))
        )

        self.ASR_DEFAULT_HOTWORDS = (
            os.getenv("ASR_DEFAULT_HOTWORDS") or ""
        ).strip()

        self.NEMOTRON_MODEL_PATH = os.path.expanduser(
            os.getenv("NEMOTRON_MODEL_PATH", self.NEMOTRON_MODEL_PATH)
        )

        self.ASR_RESTORE_SENTENCE_ENDING = (
            os.getenv("ASR_RESTORE_SENTENCE_ENDING", str(self.ASR_RESTORE_SENTENCE_ENDING))
            .strip()
            .lower()
            not in {"0", "false", "no", "off"}
        )

        self.MAX_SEGMENT_SEC = float(
            os.getenv("MAX_SEGMENT_SEC", str(self.MAX_SEGMENT_SEC))
        )

        self.QWEN_RUST_CPU_WORKERS = int(
            os.getenv("QWEN_RUST_CPU_WORKERS", str(self.QWEN_RUST_CPU_WORKERS))
        )
        self.FUNASR_WORKERS = int(
            os.getenv("FUNASR_WORKERS", str(self.FUNASR_WORKERS))
        )

        self.QWEN_IDLE_UNLOAD_TIMEOUT = int(
            os.getenv("QWEN_IDLE_UNLOAD_TIMEOUT", str(self.QWEN_IDLE_UNLOAD_TIMEOUT))
        )
        self.QWEN_ENABLE_SLEEP_MODE = (
            os.getenv("QWEN_ENABLE_SLEEP_MODE", str(self.QWEN_ENABLE_SLEEP_MODE)).lower()
            in {"1", "true", "yes", "on"}
        )
        self.QWEN_SHUTDOWN_AFTER_S = int(
            os.getenv("QWEN_SHUTDOWN_AFTER_S", str(self.QWEN_SHUTDOWN_AFTER_S))
        )

        self.QWEN_LOW_VRAM_THRESHOLD_GB = float(
            os.getenv(
                "QWEN_LOW_VRAM_THRESHOLD_GB", str(self.QWEN_LOW_VRAM_THRESHOLD_GB)
            )
        )
        self.QWEN_VRAM_PRESSURE_COOLDOWN_S = float(
            os.getenv(
                "QWEN_VRAM_PRESSURE_COOLDOWN_S",
                str(self.QWEN_VRAM_PRESSURE_COOLDOWN_S),
            )
        )

        gpu_utilization = (os.getenv("QWEN_GPU_MEMORY_UTILIZATION") or "").strip()
        self.QWEN_GPU_MEMORY_UTILIZATION = (
            float(gpu_utilization) if gpu_utilization else None
        )
        aligner_utilization = (
            os.getenv("QWEN_FORCE_ALIGNER_GPU_MEMORY_UTILIZATION") or ""
        ).strip()
        self.QWEN_FORCE_ALIGNER_GPU_MEMORY_UTILIZATION = (
            float(aligner_utilization) if aligner_utilization else None
        )

        self.QWEN_MAX_NUM_SEQS = self._optional_int("QWEN_MAX_NUM_SEQS")
        self.QWEN_MAX_NUM_BATCHED_TOKENS = self._optional_int(
            "QWEN_MAX_NUM_BATCHED_TOKENS"
        )
        self.QWEN_MAX_MODEL_LEN = self._optional_int("QWEN_MAX_MODEL_LEN")
        self.QWEN_KV_CACHE_MEMORY_BYTES = self._optional_int(
            "QWEN_KV_CACHE_MEMORY_BYTES"
        )
        self.QWEN_FORCE_ALIGNER_MAX_MODEL_LEN = self._optional_int(
            "QWEN_FORCE_ALIGNER_MAX_MODEL_LEN"
        )
        self.QWEN_ENFORCE_EAGER = (
            os.getenv("QWEN_ENFORCE_EAGER", str(self.QWEN_ENFORCE_EAGER)).lower()
            in {"1", "true", "yes", "on"}
        )
        self.QWEN_STREAM_MAX_NEW_TOKENS = int(
            os.getenv(
                "QWEN_STREAM_MAX_NEW_TOKENS", str(self.QWEN_STREAM_MAX_NEW_TOKENS)
            )
        )
        self.QWEN_STREAM_CHUNK_SEC = float(
            os.getenv("QWEN_STREAM_CHUNK_SEC", str(self.QWEN_STREAM_CHUNK_SEC))
        )
        self.QWEN_STREAM_LOOKAHEAD_SEC = float(
            os.getenv(
                "QWEN_STREAM_LOOKAHEAD_SEC", str(self.QWEN_STREAM_LOOKAHEAD_SEC)
            )
        )
        self.QWEN_STREAM_UNFIXED_CHUNK_NUM = int(
            os.getenv(
                "QWEN_STREAM_UNFIXED_CHUNK_NUM",
                str(self.QWEN_STREAM_UNFIXED_CHUNK_NUM),
            )
        )
        self.QWEN_STREAM_UNFIXED_TOKEN_NUM = int(
            os.getenv(
                "QWEN_STREAM_UNFIXED_TOKEN_NUM",
                str(self.QWEN_STREAM_UNFIXED_TOKEN_NUM),
            )
        )
        self.QWEN_STREAM_SEGMENT_MAX_SEC = float(
            os.getenv(
                "QWEN_STREAM_SEGMENT_MAX_SEC", str(self.QWEN_STREAM_SEGMENT_MAX_SEC)
            )
        )
        self.QWEN_STREAM_SILENCE_SEC = float(
            os.getenv("QWEN_STREAM_SILENCE_SEC", str(self.QWEN_STREAM_SILENCE_SEC))
        )
        self.QWEN_STREAM_OVERLAP_SEC = float(
            os.getenv("QWEN_STREAM_OVERLAP_SEC", str(self.QWEN_STREAM_OVERLAP_SEC))
        )
        self.QWEN_STREAM_VAD_MODEL_PATH = os.getenv(
            "QWEN_STREAM_VAD_MODEL_PATH", self.QWEN_STREAM_VAD_MODEL_PATH
        )
        self.QWEN_STREAM_VAD_MIN_SILENCE_FRAMES = int(
            os.getenv(
                "QWEN_STREAM_VAD_MIN_SILENCE_FRAMES",
                str(self.QWEN_STREAM_VAD_MIN_SILENCE_FRAMES),
            )
        )
        self.QWEN_STREAM_FINAL_PASS = (
            os.getenv("QWEN_STREAM_FINAL_PASS", str(self.QWEN_STREAM_FINAL_PASS)).lower()
            in {"1", "true", "yes", "on"}
        )
        self.ASR_ENGINE = (
            os.getenv("ASR_ENGINE", self.ASR_ENGINE) or self.ASR_ENGINE
        ).strip().lower()
        self.QWEN_LLAMACPP_MODEL_PATH = os.getenv(
            "QWEN_LLAMACPP_MODEL_PATH", self.QWEN_LLAMACPP_MODEL_PATH
        )
        self.QWEN_LLAMACPP_MMPROJ_PATH = os.getenv(
            "QWEN_LLAMACPP_MMPROJ_PATH", self.QWEN_LLAMACPP_MMPROJ_PATH
        )
        self.QWEN_LLAMACPP_SERVER_BIN = os.getenv(
            "QWEN_LLAMACPP_SERVER_BIN", self.QWEN_LLAMACPP_SERVER_BIN
        )
        self.QWEN_LLAMACPP_PORT = int(
            os.getenv("QWEN_LLAMACPP_PORT", str(self.QWEN_LLAMACPP_PORT))
        )
        self.QWEN_LLAMACPP_CTX = int(
            os.getenv("QWEN_LLAMACPP_CTX", str(self.QWEN_LLAMACPP_CTX))
        )
        self.QWEN_LLAMACPP_PARALLEL = int(
            os.getenv("QWEN_LLAMACPP_PARALLEL", str(self.QWEN_LLAMACPP_PARALLEL))
        )
        self.QWEN_LLAMACPP_GPU_LAYERS = int(
            os.getenv("QWEN_LLAMACPP_GPU_LAYERS", str(self.QWEN_LLAMACPP_GPU_LAYERS))
        )
        self.QWEN_LLAMACPP_STREAM_MODE = os.getenv(
            "QWEN_LLAMACPP_STREAM_MODE", self.QWEN_LLAMACPP_STREAM_MODE
        )
        self.QWEN_LLAMACPP_TOKENIZER_PATH = os.getenv(
            "QWEN_LLAMACPP_TOKENIZER_PATH", self.QWEN_LLAMACPP_TOKENIZER_PATH
        )
        self.QWEN_LLAMACPP_MAX_SEGMENT_SEC = float(
            os.getenv(
                "QWEN_LLAMACPP_MAX_SEGMENT_SEC",
                str(self.QWEN_LLAMACPP_MAX_SEGMENT_SEC),
            )
        )

        self.QWEN_FORCED_ALIGNER_ENABLED = (
            os.getenv(
                "QWEN_FORCED_ALIGNER_ENABLED", str(self.QWEN_FORCED_ALIGNER_ENABLED)
            ).lower()
            in {"1", "true", "yes", "on"}
        )
        self.QWEN_WARMUP_FORCED_ALIGNER = (
            os.getenv(
                "QWEN_WARMUP_FORCED_ALIGNER", str(self.QWEN_WARMUP_FORCED_ALIGNER)
            ).lower()
            in {"1", "true", "yes", "on"}
        )

        self.VOICEPRINT_ENABLED = (
            os.getenv("VOICEPRINT_ENABLED", str(self.VOICEPRINT_ENABLED)).lower()
            in {"1", "true", "yes", "on"}
        )
        self.VOICEPRINT_DB_PATH = (
            os.getenv("VOICEPRINT_DB_PATH") or self.VOICEPRINT_DB_PATH
        ).strip()
        self.VOICEPRINT_MATCH_THRESHOLD = float(
            os.getenv(
                "VOICEPRINT_MATCH_THRESHOLD",
                str(self.VOICEPRINT_MATCH_THRESHOLD),
            )
        )
        self.VOICEPRINT_DEVICE = (
            os.getenv("VOICEPRINT_DEVICE") or self.VOICEPRINT_DEVICE
        ).strip()


    @staticmethod
    def _optional_int(name: str) -> Optional[int]:
        """读取可选整数环境变量，空值或非法值返回 None。"""
        raw = (os.getenv(name) or "").strip()
        if not raw:
            return None
        try:
            return int(raw)
        except ValueError:
            return None

    def _parse_size(self, size_str: str) -> int:
        """解析带单位的大小字符串

        支持格式：
        - 纯数字：视为 MB（如 2048 = 2048MB = 2147483648 bytes）
        - 带单位：如 2GB, 2048MB, 1.5GB
        """
        size_str = size_str.strip().upper()

        # 如果纯数字，视为 MB
        if size_str.isdigit():
            return int(size_str) * 1024 * 1024

        # 带单位的处理
        if size_str.endswith('GB'):
            return int(float(size_str[:-2]) * 1024 * 1024 * 1024)
        elif size_str.endswith('MB'):
            return int(float(size_str[:-2]) * 1024 * 1024)
        elif size_str.endswith('KB'):
            return int(float(size_str[:-2]) * 1024)
        else:
            # 默认视为字节
            return int(size_str)

    def _ensure_directories(self):
        """确保必需的目录存在"""
        os.makedirs(self.TEMP_DIR, exist_ok=True)

    @property
    def models_config_path(self) -> str:
        """获取模型配置文件的完整路径"""
        return str(self.BASE_DIR / self.ASR_MODELS_CONFIG)

    @property
    def docs_url(self) -> Optional[str]:
        """获取文档URL"""
        return "/docs"

    @property
    def redoc_url(self) -> Optional[str]:
        """获取ReDoc URL"""
        return "/redoc"


# 全局配置实例
settings = Settings()
