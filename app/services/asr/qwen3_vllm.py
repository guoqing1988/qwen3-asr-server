# -*- coding: utf-8 -*-
"""Official vLLM adapter for CUDA Qwen3-ASR."""

from __future__ import annotations

import contextlib
import importlib
import importlib.util
import logging
import os
import re
import threading
import time
from dataclasses import dataclass, field
from typing import Any, Optional

import librosa
import numpy as np
import torch

from app.infrastructure import resolve_huggingface_snapshot_dir
from app.utils.text_processing import normalize_asr_text

from .engines import ASRRawResult, ASRSegmentResult, WordToken

logger = logging.getLogger(__name__)

_DEFAULT_SAMPLE_RATE = 16000
_LANGUAGE_ALIASES = {
    "zh": "Chinese",
    "zh-cn": "Chinese",
    "zh-hans": "Chinese",
    "zh-hant": "Chinese",
    "cn": "Chinese",
    "en": "English",
    "en-us": "English",
    "en-gb": "English",
    "ja": "Japanese",
    "jp": "Japanese",
    "ko": "Korean",
    "yue": "Cantonese",
    "fr": "French",
    "de": "German",
    "es": "Spanish",
    "ru": "Russian",
}


def is_vllm_available() -> bool:
    """Return True when the official vLLM runtime is installed."""
    return importlib.util.find_spec("vllm") is not None


def _normalize_language_name(language: Optional[str]) -> Optional[str]:
    if not language:
        return None
    normalized = language.strip()
    if not normalized:
        return None
    alias = _LANGUAGE_ALIASES.get(normalized.lower())
    if alias:
        return alias
    if " " in normalized:
        return " ".join(part.capitalize() for part in normalized.split())
    return normalized.capitalize()


def _load_audio(audio_path: str) -> np.ndarray:
    audio, _sample_rate = librosa.load(audio_path, sr=_DEFAULT_SAMPLE_RATE, mono=True)
    return audio.astype(np.float32)


def _build_chat_prompt(context: str = "", language: Optional[str] = None) -> str:
    """构造官方 qwen_asr 等价 prompt。

    - system 直接承载 context/热词原文（官方约定），不附加任何自然语言指令；
    - 指定语言时在生成提示后追加 `language X<asr_text>`，让模型直接续写正文，
      避免每轮重复预测语言标签（R2T2 流式稳定前缀的关键）。
    """
    context_text = (context or "").strip()
    prompt = (
        f"<|im_start|>system\n{context_text}<|im_end|>\n"
        "<|im_start|>user\n<|audio_start|><|audio_pad|><|audio_end|><|im_end|>\n"
        "<|im_start|>assistant\n"
    )
    if language:
        prompt += f"language {language}<asr_text>"
    return prompt


def _build_stream_prefix(tokenizer: Any, raw_decoded: str, unfixed_token_num: int) -> str:
    """构造流式解码的 prefix：已定稿的历史文本（去掉末尾 unfixed_token_num 个 token）。

    对齐官方 R2T2 实现：先按 `|` 截断，再按 token 回滚；解码若出现替换字符
    U+FFFD（截断在多字节字符中间）则继续多回滚一个 token，保证 prefix 合法。
    """
    raw = (raw_decoded or "").split("|")[0]
    if not raw:
        return ""
    token_ids = tokenizer.encode(raw, add_special_tokens=False)
    rollback = max(0, int(unfixed_token_num))
    while True:
        end = max(0, len(token_ids) - rollback)
        prefix = (
            tokenizer.decode(token_ids[:end], skip_special_tokens=False)
            if end > 0
            else ""
        )
        if "\ufffd" not in prefix:
            return prefix.split("|")[0]
        if end == 0:
            return ""
        rollback += 1


_EN2ZH_PUNCT = {
    ",": "，",
    ".": "。",
    "!": "！",
    "?": "？",
    ";": "；",
    ":": "：",
    "(": "（",
    ")": "）",
}
_ZH2EN_PUNCT = {v: k for k, v in _EN2ZH_PUNCT.items()}
_ALL_PUNCT_PAT = re.compile(r"[,\.!?;:()\uff0c\u3002\uff01\uff1f\uff1b\uff1a\uff08\uff09]")


def _normalize_punct_by_context(text: str) -> str:
    """按前一个非空白字符的语言归一标点（官方 R2T2 实现）。

    中文后用全角标点，英文/数字后用半角标点，其余保持原样。
    """
    if not text:
        return text

    def _replace(match: re.Match[str]) -> str:
        punct = match.group()
        prev_char = ""
        for i in range(match.start() - 1, -1, -1):
            if not text[i].isspace():
                prev_char = text[i]
                break
        if not prev_char:
            return punct
        if "\u4e00" <= prev_char <= "\u9fff":
            return _EN2ZH_PUNCT.get(punct, punct)
        if prev_char.isascii() and (prev_char.isalnum() or prev_char in "\"'"):
            return _ZH2EN_PUNCT.get(punct, punct)
        return punct

    return _ALL_PUNCT_PAT.sub(_replace, text)


def _strip_cjk_spaces(text: str) -> str:
    """去掉中文字符之间被模型插入的空格（保留中英混排中的英文空格）。"""
    return re.sub(r"(?<=[\u4e00-\u9fff])\s+(?=[\u4e00-\u9fff])", "", text)


def _build_alignment_prompt(tokens: list[str]) -> str:
    body = "<timestamp><timestamp>".join(tokens) + "<timestamp><timestamp>"
    return f"<|audio_start|><|audio_pad|><|audio_end|>{body}"


def _parse_asr_output(raw_text: str, language: Optional[str]) -> tuple[str, str]:
    """解析 ASR 输出（格式：`language <lang><asr_text><正文>`）。

    流式路径的 prefix 用 skip_special_tokens=False 解码，语言段与正文首尾
    可能残留 '|' 之类的标记字符，这里一并清理。
    """
    text = (raw_text or "").strip()
    if "<asr_text>" in text:
        left, right = text.split("<asr_text>", 1)
        detected = left.strip().strip("|").strip()
        if detected.lower().startswith("language "):
            detected = detected[len("language ") :].strip()
        # 模型偶尔会把语言预测成字面量 None/null（音频不足时），视为无效
        if detected.lower() in {"none", "null"}:
            detected = ""
        return detected or (language or ""), right.strip().strip("|").strip()
    return (language or ""), text.strip("|").strip()


def _split_alignment_units(text: str) -> list[str]:
    if not text:
        return []

    # Mixed Chinese/English transcripts should not fall back to whitespace-only
    # tokenization, otherwise a long CJK sentence with a single embedded English
    # word can collapse into one giant alignment unit.
    # 独立标点不作为对齐单元（无语音内容，模型预测不稳定，见 docs/troubleshooting.md）；
    # 词内符号（' . _ + -）随词保留，如 "don't"、"state-of-the-art"、"1.5"。
    token_pattern = re.compile(
        r"[\u4e00-\u9fff]"                    # CJK ideographs, align per character
        r"|[A-Za-z0-9]+(?:['._+-][A-Za-z0-9]+)*",  # Latin / alnum words
        re.UNICODE,
    )
    return token_pattern.findall(text)


def _fix_timestamp(data: list[float]) -> list[float]:
    """修复 forced aligner 时间戳序列的乱序（移植官方 qwen_asr fix_timestamp）。

    对整条时间戳序列（每对齐单元 2 个：start/end）求最长递增子序列（LIS），
    不在 LIS 上的位置视为异常：连续异常 ≤2 个时用左右最近正常值就近填充，
    否则按左右正常值线性插值，保证输出全局单调递增。
    标点等无语音单元的单点乱序（start > end 或跨单元回退）在此统一修复。
    """
    n = len(data)
    if n <= 1:
        return data.copy()

    # LIS 动态规划（O(n²)，对齐按 segment 分批进行，n 为数百量级，开销可忽略）
    dp = [1] * n
    parent = [-1] * n
    for i in range(1, n):
        for j in range(i):
            if data[j] <= data[i] and dp[j] + 1 > dp[i]:
                dp[i] = dp[j] + 1
                parent[i] = j

    max_length = max(dp)
    max_idx = dp.index(max_length)

    lis_indices: list[int] = []
    idx = max_idx
    while idx != -1:
        lis_indices.append(idx)
        idx = parent[idx]
    lis_indices.reverse()

    is_normal = [False] * n
    for idx in lis_indices:
        is_normal[idx] = True

    result = data.copy()
    i = 0
    while i < n:
        if not is_normal[i]:
            j = i
            while j < n and not is_normal[j]:
                j += 1
            anomaly_count = j - i

            left_val: float | None = None
            for k in range(i - 1, -1, -1):
                if is_normal[k]:
                    left_val = result[k]
                    break
            right_val: float | None = None
            for k in range(j, n):
                if is_normal[k]:
                    right_val = result[k]
                    break

            if anomaly_count <= 2:
                # 就近取左右正常值（平局取左）；异常段两侧至少一侧有正常值
                for k in range(i, j):
                    if left_val is None:
                        result[k] = right_val
                    elif right_val is None:
                        result[k] = left_val
                    else:
                        result[k] = left_val if (k - (i - 1)) <= (j - k) else right_val
            elif left_val is not None and right_val is not None:
                # 连续异常较多时按左右正常值线性插值
                step = (right_val - left_val) / (anomaly_count + 1)
                for k in range(i, j):
                    result[k] = left_val + step * (k - i + 1)
            elif left_val is not None:
                for k in range(i, j):
                    result[k] = left_val
            elif right_val is not None:
                for k in range(i, j):
                    result[k] = right_val
            i = j
        else:
            i += 1

    return result


def _resolve_forced_aligner_gpu_memory_utilization(
    primary_utilization: float,
    explicit_utilization: float | None = None,
) -> float:
    """解析 forced aligner 的显存利用率。

    优先级：显式传入值（来自 settings 中的 .env 配置） > 继承主模型。
    """
    if explicit_utilization is not None:
        if 0.0 < explicit_utilization <= 1.0:
            logger.info(
                "Using configured forced aligner gpu_memory_utilization=%s",
                explicit_utilization,
            )
            return explicit_utilization
        logger.warning(
            "Invalid configured forced_aligner_gpu_memory_utilization=%s, ignoring",
            explicit_utilization,
        )

    return primary_utilization


@dataclass
class _GeneratedTranscript:
    text: str
    language: str


@dataclass
class VLLMRealtimeState:
    prompt_raw: str
    language: str
    chunk_size_sec: float
    unfixed_chunk_num: int
    unfixed_token_num: int
    max_new_tokens: int
    force_language: str | None = None
    lookahead_samples: int = 0
    # 兼容旧调用方：预热参数已废弃（官方算法用动态 max_new_tokens）
    max_new_tokens_initial: int | None = None
    warmup_chunks: int | None = None
    chunk_id: int = 0
    text: str = ""
    raw_decoded: str = ""
    audio_buffer: np.ndarray = field(default_factory=lambda: np.array([], dtype=np.float32))
    audio_accum: np.ndarray = field(default_factory=lambda: np.array([], dtype=np.float32))


class Qwen3VLLMBackend:
    """Thin adapter over official vLLM APIs for Qwen3-ASR."""

    def __init__(
        self,
        model_path: str,
        forced_aligner_path: Optional[str],
        gpu_memory_utilization: float,
        max_inference_batch_size: int,
        max_new_tokens: int,
        max_model_len: Optional[int] = None,
        forced_aligner_gpu_memory_utilization: float | None = None,
        max_num_seqs: int | None = None,
        max_num_batched_tokens: int | None = None,
        enforce_eager: bool = False,
        kv_cache_memory_bytes: int | None = None,
        forced_aligner_max_model_len: int | None = None,
        enable_sleep_mode: bool = False,
    ) -> None:
        try:
            vllm_module = importlib.import_module("vllm")
            transformers_module = importlib.import_module("transformers")
        except ImportError as exc:
            raise RuntimeError(
                "CUDA Qwen3-ASR now requires official vLLM with Qwen3 forced aligner support. "
                "Install it with: pip install 'vllm[audio]==0.19.0'"
            ) from exc

        local_model_path = str(resolve_huggingface_snapshot_dir(model_path))
        local_forced_aligner_path = (
            str(resolve_huggingface_snapshot_dir(forced_aligner_path))
            if forced_aligner_path
            else None
        )

        self._llm_cls = getattr(vllm_module, "LLM")
        self._sampling_params_cls = getattr(vllm_module, "SamplingParams")
        self._tokenizer = getattr(transformers_module, "AutoTokenizer").from_pretrained(
            local_model_path,
            trust_remote_code=True,
            local_files_only=True,
        )

        llm_kwargs: dict[str, Any] = {
            "model": local_model_path,
            "gpu_memory_utilization": gpu_memory_utilization,
        }
        if max_model_len is not None:
            llm_kwargs["max_model_len"] = max_model_len
        if max_num_seqs is not None:
            llm_kwargs["max_num_seqs"] = max_num_seqs
        if max_num_batched_tokens is not None:
            llm_kwargs["max_num_batched_tokens"] = max_num_batched_tokens
        if enforce_eager:
            llm_kwargs["enforce_eager"] = True
        if kv_cache_memory_bytes is not None:
            llm_kwargs["kv_cache_memory_bytes"] = kv_cache_memory_bytes
        # sleep mode：进程常驻，空闲时把权重 offload 到 CPU、丢弃 KV cache 以归还显存
        if enable_sleep_mode:
            llm_kwargs["enable_sleep_mode"] = True
            # sleep 丢弃 KV cache 后 multimodal processor cache 内部状态失效，
            # 唤醒后重复音频会命中失效项并抛 AssertionError（实测），故禁用该缓存
            llm_kwargs["mm_processor_cache_gb"] = 0

        self._llm = self._llm_cls(**llm_kwargs)
        self._sampling_params = self._sampling_params_cls(
            temperature=0.01,
            max_tokens=max_new_tokens,
        )
        self._max_inference_batch_size = max_inference_batch_size
        self._gpu_memory_utilization = gpu_memory_utilization
        self._forced_aligner_gpu_memory_utilization = (
            forced_aligner_gpu_memory_utilization
        )
        self._forced_aligner_path = local_forced_aligner_path
        self._forced_aligner_max_model_len = forced_aligner_max_model_len
        self._max_num_batched_tokens = max_num_batched_tokens
        self._forced_aligner: Any | None = None
        self._timestamp_token_id: int | None = None
        self._timestamp_segment_time: float | None = None

        # 显存让位状态（sleep/wake）
        self._enable_sleep_mode = enable_sleep_mode
        self._sleep_lock = threading.RLock()
        self._main_active = 0
        self._main_sleeping = False
        # aligner 独立生命周期：自己的活跃计数与最后使用时间，不被主引擎连坐
        self._aligner_state_lock = threading.RLock()
        self._aligner_active = 0
        self._aligner_last_use = time.monotonic()
        self._aligner_sleeping = False

    def _get_forced_aligner_gpu_memory_utilization(self) -> float:
        configured = _resolve_forced_aligner_gpu_memory_utilization(
            self._gpu_memory_utilization,
            explicit_utilization=self._forced_aligner_gpu_memory_utilization,
        )
        logger.info(
            "Resolved forced aligner gpu_memory_utilization=%s (primary=%s)",
            configured,
            self._gpu_memory_utilization,
        )
        return configured

    def _get_forced_aligner(self) -> Any:
        if not self._forced_aligner_path:
            raise RuntimeError("word_timestamps requires a configured forced aligner model")

        if self._forced_aligner is None:
            forced_aligner_gpu_memory_utilization = self._get_forced_aligner_gpu_memory_utilization()
            logger.info(
                "Loading Qwen3 forced aligner via official vLLM: %s (gpu_memory_utilization=%s)",
                self._forced_aligner_path,
                forced_aligner_gpu_memory_utilization,
            )
            aligner_kwargs: dict[str, Any] = {
                "model": self._forced_aligner_path,
                "runner": "pooling",
                "enforce_eager": True,
                "gpu_memory_utilization": forced_aligner_gpu_memory_utilization,
                "hf_overrides": {
                    "architectures": ["Qwen3ASRForcedAlignerForTokenClassification"],
                },
            }
            if self._forced_aligner_max_model_len is not None:
                aligner_kwargs["max_model_len"] = self._forced_aligner_max_model_len
            if self._max_num_batched_tokens is not None:
                aligner_kwargs["max_num_batched_tokens"] = self._max_num_batched_tokens
            if self._enable_sleep_mode:
                aligner_kwargs["enable_sleep_mode"] = True
                # 同主引擎：sleep 后 multimodal 缓存会失效，禁用
                aligner_kwargs["mm_processor_cache_gb"] = 0
            self._forced_aligner = self._llm_cls(**aligner_kwargs)
            llm_engine = getattr(self._forced_aligner, "llm_engine", None)
            if llm_engine is None:
                raise RuntimeError("Forced aligner did not expose a vLLM engine instance")
            config = llm_engine.vllm_config.model_config.hf_config
            self._timestamp_token_id = int(config.timestamp_token_id)
            self._timestamp_segment_time = float(config.timestamp_segment_time)
            self._aligner_last_use = time.monotonic()
            self._aligner_sleeping = False

        return self._forced_aligner

    def ensure_forced_aligner_loaded(self) -> None:
        if self._forced_aligner_path:
            self._get_forced_aligner()

    def shutdown(self) -> None:
        """释放 vLLM 引擎占用的 GPU 显存。

        终止主模型与 forced aligner 的 EngineCore 子进程（显存随之归还），
        并清空本进程内的 CUDA 缓存。卸载后引擎不可用，需重建 Qwen3VLLMBackend。
        """
        import gc

        for name, llm in (("main", self._llm), ("forced_aligner", self._forced_aligner)):
            if llm is None:
                continue
            engine_core = getattr(getattr(llm, "llm_engine", None), "engine_core", None)
            if engine_core is not None and hasattr(engine_core, "shutdown"):
                try:
                    engine_core.shutdown(timeout=10.0)
                    logger.info("vLLM %s engine core shutdown complete", name)
                except Exception as exc:  # noqa: BLE001 - 卸载失败不应阻断流程
                    logger.warning("Failed to shutdown %s vLLM engine core: %s", name, exc)

        self._forced_aligner = None
        self._llm = None
        self._main_sleeping = False
        self._aligner_sleeping = False
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        logger.info("Qwen3VLLMBackend shutdown complete, GPU memory released")

    # ------------------------------------------------------------------
    # sleep / wake：进程常驻，空闲时只归还显存（不销毁 vLLM 子进程）
    # ------------------------------------------------------------------

    def supports_sleep(self) -> bool:
        return self._enable_sleep_mode and self._llm is not None

    @contextlib.contextmanager
    def _main_in_use(self):
        """标记主引擎使用中（必要时唤醒），与 sleep 互斥。"""
        with self._sleep_lock:
            self._main_active += 1
            if self._main_sleeping and self._llm is not None:
                started = time.monotonic()
                self._llm.wake_up()
                self._main_sleeping = False
                logger.info("Main engine woke up in %.2fs", time.monotonic() - started)
        try:
            yield
        finally:
            with self._sleep_lock:
                self._main_active = max(0, self._main_active - 1)

    def is_main_sleeping(self) -> bool:
        return self._main_sleeping

    def sleep_main(self) -> None:
        """归还主引擎显存（权重 offload 到 CPU、丢弃 KV cache）。"""
        if not self._enable_sleep_mode or self._llm is None:
            return
        with self._sleep_lock:
            if self._main_sleeping or self._main_active > 0:
                return
            started = time.monotonic()
            self._llm.sleep(level=1)
            self._main_sleeping = True
            logger.info(
                "Main engine slept in %.1fs, GPU memory released",
                time.monotonic() - started,
            )

    def wake_main(self) -> None:
        if self._llm is None or not self._main_sleeping:
            return
        with self._sleep_lock:
            if not self._main_sleeping:
                return
            started = time.monotonic()
            self._llm.wake_up()
            self._main_sleeping = False
            logger.info("Main engine woke up in %.2fs", time.monotonic() - started)

    # ---- forced aligner（独立生命周期：可单独 sleep，不被主引擎连坐）----

    def is_aligner_loaded(self) -> bool:
        return self._forced_aligner is not None

    def is_aligner_sleeping(self) -> bool:
        return self._aligner_sleeping

    def is_aligner_busy(self) -> bool:
        return self._aligner_active > 0

    def aligner_idle_seconds(self) -> float:
        if self._forced_aligner is None:
            return 0.0
        return time.monotonic() - self._aligner_last_use

    @contextlib.contextmanager
    def _aligner_in_use(self):
        """标记 aligner 使用中（必要时唤醒），与 sleep 互斥。"""
        with self._aligner_state_lock:
            self._aligner_active += 1
            self._aligner_last_use = time.monotonic()
            if self._aligner_sleeping and self._forced_aligner is not None:
                started = time.monotonic()
                self._forced_aligner.wake_up()
                self._aligner_sleeping = False
                logger.info("Forced aligner woke up in %.2fs", time.monotonic() - started)
        try:
            yield
        finally:
            with self._aligner_state_lock:
                self._aligner_active = max(0, self._aligner_active - 1)
                self._aligner_last_use = time.monotonic()

    def sleep_aligner(self) -> None:
        """单独归还 aligner 显存（主引擎不受影响）。"""
        if not self._enable_sleep_mode or self._forced_aligner is None:
            return
        with self._aligner_state_lock:
            if self._aligner_sleeping or self._aligner_active > 0:
                return
            started = time.monotonic()
            self._forced_aligner.sleep(level=1)
            self._aligner_sleeping = True
            logger.info(
                "Forced aligner slept in %.1fs, GPU memory released",
                time.monotonic() - started,
            )

    def wake_aligner(self) -> None:
        if self._forced_aligner is None or not self._aligner_sleeping:
            return
        with self._aligner_state_lock:
            if not self._aligner_sleeping:
                return
            started = time.monotonic()
            self._forced_aligner.wake_up()
            self._aligner_sleeping = False
            logger.info("Forced aligner woke up in %.2fs", time.monotonic() - started)

    def sleep_all(self) -> None:
        """归还两个引擎的显存（供声纹等需要大块显存的步骤让路）。"""
        try:
            self.sleep_aligner()
        except Exception as exc:  # noqa: BLE001 - 让路失败不应影响主流程
            logger.warning("Failed to sleep forced aligner: %s", exc)
        try:
            self.sleep_main()
        except Exception as exc:  # noqa: BLE001
            logger.warning("Failed to sleep main engine: %s", exc)

    def _run_generate(
        self,
        audio_items: list[tuple[np.ndarray, str, Optional[str]]],
    ) -> list[_GeneratedTranscript]:
        prompts: list[dict[str, Any]] = []
        for audio, context, language in audio_items:
            prompts.append(
                {
                    "prompt": _build_chat_prompt(context=context, language=_normalize_language_name(language)),
                    "multi_modal_data": {"audio": [audio]},
                }
            )

        with self._main_in_use():
            outputs = self._llm.generate(
                prompts,
                sampling_params=self._sampling_params,
                use_tqdm=False,
            )

        transcripts: list[_GeneratedTranscript] = []
        for output, (_audio, _context, language) in zip(outputs, audio_items):
            raw_text = str(output.outputs[0].text if output.outputs else "")
            parsed_language, parsed_text = _parse_asr_output(raw_text, _normalize_language_name(language))
            transcripts.append(_GeneratedTranscript(text=parsed_text, language=parsed_language))
        # 推理完成后释放本进程 CUDA 缓存池，避免峰值显存长期占用
        # （宿主 GPU 与其他服务共享，如 ComfyUI 推理需要大显存）
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        return transcripts

    def transcribe_text(
        self,
        audio_path: str,
        context: str = "",
        language: Optional[str] = None,
        enable_itn: bool = False,
    ) -> str:
        transcript = self._run_generate([(_load_audio(audio_path), context, language)])[0]
        return normalize_asr_text(transcript.text, enable_itn=enable_itn)

    def transcribe_array(
        self,
        audio: np.ndarray,
        context: str = "",
        language: Optional[str] = None,
        enable_itn: bool = False,
    ) -> str:
        """对内存音频做一次离线转写（用于流式段落 two-pass 定稿）。"""
        transcript = self._run_generate(
            [(np.asarray(audio, dtype=np.float32), context, language)]
        )[0]
        return normalize_asr_text(transcript.text, enable_itn=enable_itn)

    def transcribe_raw(
        self,
        audio_path: str,
        context: str = "",
        language: Optional[str] = None,
        word_timestamps: bool = False,
        enable_itn: bool = False,
    ) -> ASRRawResult:
        audio = _load_audio(audio_path)
        transcript = self._run_generate([(audio, context, language)])[0]
        text = normalize_asr_text(transcript.text, enable_itn=enable_itn)
        if not word_timestamps:
            return ASRRawResult(
                text=text,
                segments=[ASRSegmentResult(text=text, start_time=0.0, end_time=0.0)] if text else [],
            )

        aligned = self.align_transcript(audio_path=audio_path, text=text, language=language, audio=audio)
        word_tokens = [
            WordToken(
                text=str(item["text"]),
                start_time=round(float(item["start_ms"]) / 1000.0, 3),
                end_time=round(float(item["end_ms"]) / 1000.0, 3),
            )
            for item in aligned
        ]
        if not word_tokens:
            return ASRRawResult(
                text=text,
                segments=[ASRSegmentResult(text=text, start_time=0.0, end_time=0.0)] if text else [],
            )
        return ASRRawResult(
            text=text,
            segments=[
                ASRSegmentResult(
                    text=text,
                    start_time=word_tokens[0].start_time,
                    end_time=word_tokens[-1].end_time,
                    word_tokens=word_tokens,
                )
            ],
        )

    def transcribe_batch(
        self,
        audio_paths: list[str],
        context: str = "",
        language: Optional[str] = None,
        word_timestamps: bool = False,
        enable_itn: bool = False,
    ) -> list[ASRSegmentResult]:
        audios = [_load_audio(path) for path in audio_paths]
        results: list[ASRSegmentResult] = []
        for start in range(0, len(audios), self._max_inference_batch_size):
            chunk = audios[start:start + self._max_inference_batch_size]
            transcripts = self._run_generate([(audio, context, language) for audio in chunk])
            for audio_path, audio, transcript in zip(audio_paths[start:start + len(chunk)], chunk, transcripts):
                text = normalize_asr_text(transcript.text, enable_itn=enable_itn)
                if not word_timestamps:
                    results.append(ASRSegmentResult(text=text, start_time=0.0, end_time=0.0))
                    continue
                aligned = self.align_transcript(
                    audio_path=audio_path,
                    text=text,
                    language=language,
                    audio=audio,
                )
                word_tokens = [
                    WordToken(
                        text=str(item["text"]),
                        start_time=round(float(item["start_ms"]) / 1000.0, 3),
                        end_time=round(float(item["end_ms"]) / 1000.0, 3),
                    )
                    for item in aligned
                ]
                results.append(
                    ASRSegmentResult(
                        text=text,
                        start_time=word_tokens[0].start_time if word_tokens else 0.0,
                        end_time=word_tokens[-1].end_time if word_tokens else 0.0,
                        word_tokens=word_tokens or None,
                    )
                )
        return results

    def align_transcript(
        self,
        audio_path: str,
        text: str,
        language: Optional[str] = None,
        audio: Optional[np.ndarray] = None,
    ) -> list[dict[str, float | str]]:
        tokens = _split_alignment_units(text)
        if not tokens:
            return []

        aligner = self._get_forced_aligner()
        prompt = _build_alignment_prompt(tokens)
        audio_array = audio if audio is not None else _load_audio(audio_path)
        with self._aligner_in_use():
            outputs = aligner.encode(
                [{"prompt": prompt, "multi_modal_data": {"audio": audio_array}}],
                pooling_task="token_classify",
            )
        output = outputs[0]
        logits = output.outputs.data
        predictions = logits.argmax(dim=-1) if hasattr(logits, "argmax") else np.argmax(logits, axis=-1)
        ts_predictions = [
            float(pred.item() if hasattr(pred, "item") else pred) * float(self._timestamp_segment_time or 0.0)
            for tid, pred in zip(output.prompt_token_ids, predictions)
            if int(tid) == int(self._timestamp_token_id or -1)
        ]

        expected_timestamps = len(tokens) * 2
        if len(ts_predictions) < expected_timestamps:
            raise RuntimeError(
                "Forced aligner returned fewer timestamp predictions than expected: "
                f"expected={expected_timestamps}, got={len(ts_predictions)}, tokens={len(tokens)}"
            )

        # 整条序列全局单调化（标点等无语音单元偶发乱序在此修复），
        # 替换原先仅逐 token 交换 start/end 的局部处理
        fixed_predictions = _fix_timestamp(ts_predictions)
        anomaly_count = sum(
            1 for original, fixed in zip(ts_predictions, fixed_predictions) if original != fixed
        )
        if anomaly_count:
            logger.info(
                "Forced aligner timestamps fixed: %d/%d anomaly predictions corrected",
                anomaly_count,
                len(ts_predictions),
            )

        aligned = [
            {
                "text": token,
                "start_ms": fixed_predictions[index * 2],
                "end_ms": fixed_predictions[index * 2 + 1],
            }
            for index, token in enumerate(tokens)
        ]
        # 对齐器推理后同样释放本进程 CUDA 缓存池
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        return aligned

    def init_streaming_state(
        self,
        *,
        context: str = "",
        language: Optional[str] = None,
        chunk_size_sec: float = 0.16,
        unfixed_chunk_num: int = 0,
        unfixed_token_num: int = 1,
        max_new_tokens: int = 512,
        lookahead_sec: float = 0.16,
        max_new_tokens_initial: int | None = None,
        warmup_chunks: int | None = None,
        initial_audio: np.ndarray | None = None,
    ) -> VLLMRealtimeState:
        """初始化 R2T2 流式状态（官方参数：160ms chunk / unfixed 0+1）。

        - ``chunk_size_sec``：解码粒度，官方推荐 0.16（80ms~2s 可调）；
        - ``lookahead_sec``：首个 chunk 额外音频，用于首段语言/专名判断；
        - ``max_new_tokens``：配置的生成上限，实际按 chunk 时长动态取小值；
        - ``initial_audio``：兼容旧段间重叠逻辑，新调用方不再使用。
        """
        normalized_language = _normalize_language_name(language)
        force_language = normalized_language or None
        return VLLMRealtimeState(
            prompt_raw=_build_chat_prompt(context=context, language=force_language),
            language=normalized_language or "",
            force_language=force_language,
            chunk_size_sec=float(chunk_size_sec),
            unfixed_chunk_num=int(unfixed_chunk_num),
            unfixed_token_num=int(unfixed_token_num),
            max_new_tokens=int(max_new_tokens),
            lookahead_samples=int(max(0.0, float(lookahead_sec)) * _DEFAULT_SAMPLE_RATE),
            max_new_tokens_initial=max_new_tokens_initial,
            warmup_chunks=warmup_chunks,
            audio_accum=(
                np.asarray(initial_audio, dtype=np.float32)
                if initial_audio is not None
                else np.array([], dtype=np.float32)
            ),
            audio_buffer=np.array([], dtype=np.float32),
        )

    def _stream_max_new_tokens(self, state: VLLMRealtimeState, chunk_samples: int) -> int:
        """按 chunk 时长计算生成上限（官方约 80ms 音频 = 1 token）。

        ``max_new_tokens`` 配置值作为硬上限（<=0 表示不设上限），默认动态值
        约为 ``(chunk + lookahead) / 1280 * 2``，兼顾中英文（中文 token 更密）。
        """
        dynamic = max(2, int(chunk_samples / 1280) * 2)
        cap = int(state.max_new_tokens or 0)
        if cap > 0:
            return max(1, min(cap, dynamic))
        return dynamic

    def _decode_stream(
        self, state: VLLMRealtimeState, chunk_samples: int
    ) -> VLLMRealtimeState:
        # prefix = 已定稿的历史文本（见 _build_stream_prefix 说明）
        prefix = ""
        if state.chunk_id >= state.unfixed_chunk_num and state.raw_decoded:
            prefix = _build_stream_prefix(
                self._tokenizer, state.raw_decoded, state.unfixed_token_num
            )

        # 官方 R2T2：每轮生成量按当前 chunk 时长自适应（几 token 即可），
        # 配置值只作为上限，避免历史上 2048 这类大值拖慢单轮解码。
        max_tokens = self._stream_max_new_tokens(state, chunk_samples)
        with self._main_in_use():
            output = self._llm.generate(
                [
                    {
                        "prompt": state.prompt_raw + prefix,
                        "multi_modal_data": {"audio": [state.audio_accum]},
                    }
                ],
                sampling_params=self._sampling_params_cls(
                    temperature=0.0,
                    max_tokens=max_tokens,
                    skip_special_tokens=True,
                ),
                use_tqdm=False,
            )[0]
        generated = _normalize_punct_by_context(
            str(output.outputs[0].text if output.outputs else "")
        ).replace("\ufffd", "")
        raw = (prefix + generated).split("|")[0]
        parsed_language, parsed_text = _parse_asr_output(
            raw, state.force_language or state.language or None
        )
        if state.force_language:
            parsed_language = state.force_language
        if state.force_language == "Chinese" or parsed_language == "Chinese":
            raw = _strip_cjk_spaces(raw)
            parsed_text = _strip_cjk_spaces(parsed_text)
        state.raw_decoded = raw
        state.text = parsed_text
        state.language = parsed_language or state.language
        state.chunk_id += 1
        return state

    def feed_stream(self, pcm: np.ndarray, state: VLLMRealtimeState) -> VLLMRealtimeState:
        state.audio_buffer = np.concatenate([state.audio_buffer, pcm.astype(np.float32)])
        segment_size = int(max(state.chunk_size_sec, 0.02) * _DEFAULT_SAMPLE_RATE)
        while True:
            # 首个 chunk 额外带 lookahead 音频，后续按固定 chunk 大小
            need = segment_size + (state.lookahead_samples if state.chunk_id == 0 else 0)
            if len(state.audio_buffer) < need:
                break
            segment = state.audio_buffer[:need].copy()
            state.audio_buffer = state.audio_buffer[need:]
            state.audio_accum = np.concatenate([state.audio_accum, segment])
            state = self._decode_stream(state, chunk_samples=len(segment))
        return state

    def finish_stream(self, state: VLLMRealtimeState) -> VLLMRealtimeState:
        step_samples = int(max(state.chunk_size_sec, 0.02) * _DEFAULT_SAMPLE_RATE)
        if len(state.audio_buffer) > 0:
            tail = state.audio_buffer
            state.audio_accum = np.concatenate([state.audio_accum, tail])
            state.audio_buffer = np.array([], dtype=np.float32)
            # 尾块可能很短，仍按一个完整 chunk 给足生成长度，避免句尾被截断
            state = self._decode_stream(
                state, chunk_samples=max(len(tail), step_samples)
            )
        elif state.chunk_id == 0 and len(state.audio_accum) > 0:
            state = self._decode_stream(
                state,
                chunk_samples=max(len(state.audio_accum), step_samples),
            )
        return state
