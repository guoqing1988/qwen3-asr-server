# -*- coding: utf-8 -*-
"""Official vLLM adapter for CUDA Qwen3-ASR."""

from __future__ import annotations

import importlib
import importlib.util
import logging
import os
import re
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
    instructions: list[str] = []
    if language:
        instructions.append(f"Transcribe the speech in {language}.")
    else:
        instructions.append("Transcribe the speech accurately.")
    if context.strip():
        instructions.append(f"Use this context when resolving named entities: {context.strip()}")
    system_text = " ".join(instructions).strip()
    return (
        f"<|im_start|>system\n{system_text}<|im_end|>\n"
        "<|im_start|>user\n<|audio_start|><|audio_pad|><|audio_end|><|im_end|>\n"
        "<|im_start|>assistant\n"
    )


def _build_stream_prefix(tokenizer: Any, raw_decoded: str, unfixed_token_num: int) -> str:
    """构造流式解码的 prefix：已定稿的历史文本（去掉末尾 unfixed_token_num 个 token）。

    本项目每轮都是「全量音频 + 全量文本重算」（无 vLLM KV 增量续写），prefix 必须
    是完整历史；只取末尾 N 个 token 会丢掉全部历史，导致输出逐轮丢内容。

    注：原始输出用 skip_special_tokens=False 解码，`<asr_text>` 等标记必须保留，
    否则 `_parse_asr_output` 无法切分语言段与正文。
    """
    if not raw_decoded:
        return ""
    token_ids = tokenizer.encode(raw_decoded, add_special_tokens=False)
    if unfixed_token_num > 0:
        keep = token_ids[: -unfixed_token_num] if len(token_ids) > unfixed_token_num else []
    else:
        keep = token_ids
    if not keep:
        return ""
    return tokenizer.decode(keep, skip_special_tokens=False).replace("\ufffd", "")


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
    max_new_tokens_initial: int = 32
    warmup_chunks: int = 8
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
            self._forced_aligner = self._llm_cls(**aligner_kwargs)
            llm_engine = getattr(self._forced_aligner, "llm_engine", None)
            if llm_engine is None:
                raise RuntimeError("Forced aligner did not expose a vLLM engine instance")
            config = llm_engine.vllm_config.model_config.hf_config
            self._timestamp_token_id = int(config.timestamp_token_id)
            self._timestamp_segment_time = float(config.timestamp_segment_time)

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
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        logger.info("Qwen3VLLMBackend shutdown complete, GPU memory released")

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
        chunk_size_sec: float = 2.0,
        unfixed_chunk_num: int = 2,
        unfixed_token_num: int = 5,
        max_new_tokens: int = 32,
        max_new_tokens_initial: int = 32,
        warmup_chunks: int = 8,
    ) -> VLLMRealtimeState:
        normalized_language = _normalize_language_name(language) or ""
        return VLLMRealtimeState(
            prompt_raw=_build_chat_prompt(context=context, language=normalized_language or None),
            language=normalized_language,
            chunk_size_sec=chunk_size_sec,
            unfixed_chunk_num=unfixed_chunk_num,
            unfixed_token_num=unfixed_token_num,
            max_new_tokens=max_new_tokens,
            max_new_tokens_initial=max_new_tokens_initial,
            warmup_chunks=warmup_chunks,
            audio_buffer=np.array([], dtype=np.float32),
            audio_accum=np.array([], dtype=np.float32),
        )

    def _decode_stream(self, state: VLLMRealtimeState) -> VLLMRealtimeState:
        # prefix = 已定稿的历史文本（见 _build_stream_prefix 说明）；预热 chunk 内不定稿
        prefix = ""
        if state.chunk_id >= state.unfixed_chunk_num and state.raw_decoded:
            prefix = _build_stream_prefix(
                self._tokenizer, state.raw_decoded, state.unfixed_token_num
            )

        # 预热期给足 token，让模型先吐出 `language X<asr_text>` 前缀；稳定期只做
        # 增量生成以降低出字延迟（前缀没出来时正文永远解析不到）
        max_tokens = (
            state.max_new_tokens_initial
            if state.chunk_id < state.warmup_chunks
            else state.max_new_tokens
        )
        output = self._llm.generate(
            [
                {
                    "prompt": state.prompt_raw + prefix,
                    "multi_modal_data": {"audio": [state.audio_accum]},
                }
            ],
            sampling_params=self._sampling_params_cls(
                temperature=0.01,
                max_tokens=max_tokens,
            ),
            use_tqdm=False,
        )[0]
        generated = str(output.outputs[0].text if output.outputs else "")
        parsed_language, parsed_text = _parse_asr_output(prefix + generated, state.language or None)
        state.raw_decoded = prefix + generated
        state.text = parsed_text
        state.language = parsed_language or state.language
        state.chunk_id += 1
        return state

    def feed_stream(self, pcm: np.ndarray, state: VLLMRealtimeState) -> VLLMRealtimeState:
        state.audio_buffer = np.concatenate([state.audio_buffer, pcm.astype(np.float32)])
        segment_size = int(max(state.chunk_size_sec, 0.1) * _DEFAULT_SAMPLE_RATE)
        while len(state.audio_buffer) >= segment_size:
            segment = state.audio_buffer[:segment_size].copy()
            state.audio_buffer = state.audio_buffer[segment_size:]
            state.audio_accum = np.concatenate([state.audio_accum, segment])
            state = self._decode_stream(state)
        return state

    def finish_stream(self, state: VLLMRealtimeState) -> VLLMRealtimeState:
        if len(state.audio_buffer) > 0:
            state.audio_accum = np.concatenate([state.audio_accum, state.audio_buffer])
            state.audio_buffer = np.array([], dtype=np.float32)
            state = self._decode_stream(state)
        elif state.chunk_id == 0 and len(state.audio_accum) > 0:
            state = self._decode_stream(state)
        return state
