# -*- coding: utf-8 -*-
"""llama.cpp Q8 backend for Confucius4-R2T2.

Runs the official Q8 GGUF pair through a managed ``llama-server`` subprocess.
Public behaviour (offline / long audio / streaming / word timestamps) matches the
vLLM backend; switch with ``ASR_ENGINE=vllm|llamacpp``.
"""

import base64
import io
import logging
import os
import socket
import subprocess
import threading
import time
import wave
from dataclasses import dataclass, field
from typing import Any

import numpy as np
import requests

from app.core.config import settings
from app.services.asr.engines.base import ASRRawResult, ASRSegmentResult, WordToken
from app.utils.text_processing import normalize_asr_text

from .qwen3_vllm import (
    _build_alignment_prompt,
    _fix_timestamp,
    _load_audio,
    _normalize_language_name,
    _split_alignment_units,
    _strip_cjk_spaces,
)

logger = logging.getLogger(__name__)

_SAMPLE_RATE = 16000
_SERVER_LOG = "/tmp/qwen3-asr-llamacpp-server.log"


def _pcm_to_wav_bytes(audio: np.ndarray) -> bytes:
    """float32 [-1, 1] mono 16k -> 16-bit PCM WAV bytes."""
    pcm = (np.clip(np.asarray(audio, dtype=np.float32), -1.0, 1.0) * 32768.0).astype(
        np.int16
    )
    buf = io.BytesIO()
    with wave.open(buf, "wb") as writer:
        writer.setnchannels(1)
        writer.setsampwidth(2)
        writer.setframerate(_SAMPLE_RATE)
        writer.writeframes(pcm.tobytes())
    return buf.getvalue()


def _split_audio(audio: np.ndarray, max_seconds: float) -> list[np.ndarray]:
    """Split audio into <=max_seconds pieces (llama.cpp audio encoder ~60s cap)."""
    step = max(1, int(max_seconds * _SAMPLE_RATE))
    if len(audio) <= step:
        return [audio]
    return [audio[start : start + step] for start in range(0, len(audio), step)]


def _join_segments(parts: list[str], language: str | None) -> str:
    if len(parts) == 1:
        return parts[0]
    spaced = (language or "") in {
        "English",
        "Korean",
        "German",
        "French",
        "Russian",
        "Portuguese",
        "Spanish",
        "Italian",
    }
    text = " ".join(parts) if spaced else "".join(parts)
    return _strip_cjk_spaces(text).strip()


def _parse_output(raw: str, forced_language: str | None) -> tuple[str, str]:
    """Parse `language X<asr_text>body` (same shape as the vLLM backend)."""
    text = (raw or "").strip()
    if "<asr_text>" in text:
        left, right = text.split("<asr_text>", 1)
        detected = left.strip().strip("|").strip()
        if detected.lower().startswith("language "):
            detected = detected[len("language ") :].strip()
        if detected.lower() in {"none", "null"}:
            detected = ""
        return forced_language or detected, right.strip().strip("|").strip()
    # 首轮可能只生成了未闭合的 `language X<asr_text`（max_tokens 截断）
    lines = text.splitlines()
    if len(lines) > 1 and lines[0].strip().lower().startswith("language "):
        detected = lines[0].strip()[len("language ") :].strip()
        return forced_language or detected, "\n".join(lines[1:]).strip()
    if text.lower().startswith("language "):
        detected = text[len("language ") :].split("<", 1)[0].strip()
        if detected.lower() in {"none", "null"}:
            detected = ""
        return forced_language or detected, ""
    return forced_language or "", text.strip("|").strip()


@dataclass
class Q8StreamingState:
    """Append-only streaming state, mirroring ``VLLMRealtimeState`` fields used by the engine."""

    context: str = ""
    language: str | None = None
    force_language: str | None = None
    chunk_size_sec: float = 0.16
    lookahead_sec: float = 0.16
    unfixed_token_num: int = 1
    max_new_tokens: int = 512
    buffer: np.ndarray = field(default_factory=lambda: np.empty(0, dtype=np.float32))
    audio: np.ndarray = field(default_factory=lambda: np.empty(0, dtype=np.float32))
    raw: str = ""
    text: str = ""
    preview: str = ""
    detected_language: str = ""
    chunk_id: int = 0


class LlamaCppQ8Backend:
    """Q8 GGUF inference through a managed llama-server process."""

    def __init__(
        self,
        model_path: str,
        mmproj_path: str,
        server_bin: str,
        *,
        context_size: int = 8192,
        parallel: int = 1,
        gpu_layers: int = 99,
        port: int = 0,
        max_segment_sec: float = 25.0,
        forced_aligner_path: str | None = None,
        forced_aligner_gpu_memory_utilization: float | None = None,
        forced_aligner_max_model_len: int | None = None,
        request_timeout: float = 600.0,
    ) -> None:
        for path in (model_path, mmproj_path, server_bin):
            if not path or not os.path.exists(path):
                raise RuntimeError(f"llama.cpp Q8 backend requirement missing: {path}")
        self._model_path = model_path
        self._mmproj_path = mmproj_path
        self._server_bin = server_bin
        self._context_size = int(context_size)
        self._parallel = int(parallel)
        self._gpu_layers = int(gpu_layers)
        self._configured_port = int(port)
        self._max_segment_sec = float(max_segment_sec)
        self._request_timeout = float(request_timeout)

        self._forced_aligner_path = forced_aligner_path
        self._aligner_gpu_util = forced_aligner_gpu_memory_utilization
        self._aligner_max_model_len = forced_aligner_max_model_len
        self._aligner: Any | None = None
        self._aligner_sleeping = False
        self._aligner_busy = 0
        self._aligner_last_use = 0.0
        self._timestamp_token_id: int | None = None
        self._timestamp_segment_time: float | None = None
        self._aligner_lock = threading.RLock()

        self._lock = threading.RLock()
        self._proc: subprocess.Popen[bytes] | None = None
        self._port = 0
        self._log_handle: Any | None = None

    # ------------------------------------------------------------------
    # llama-server lifecycle
    # ------------------------------------------------------------------

    @staticmethod
    def _free_port() -> int:
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
            sock.bind(("127.0.0.1", 0))
            return int(sock.getsockname()[1])

    def _healthy(self) -> bool:
        if self._port <= 0:
            return False
        try:
            response = requests.get(
                f"http://127.0.0.1:{self._port}/health", timeout=2
            )
            return response.status_code == 200
        except requests.RequestException:
            return False

    def _ensure_server(self) -> None:
        with self._lock:
            if self._proc is not None and self._proc.poll() is None and self._healthy():
                return
            self._stop_server_locked()
            port = self._configured_port or self._free_port()
            command = [
                self._server_bin,
                "-m",
                self._model_path,
                "--mmproj",
                self._mmproj_path,
                "-ngl",
                str(self._gpu_layers),
                "-c",
                str(self._context_size),
                "--parallel",
                str(self._parallel),
                "--host",
                "127.0.0.1",
                "--port",
                str(port),
                "--no-webui",
            ]
            logger.info("Starting llama-server Q8 on port %s: %s", port, " ".join(command))
            self._log_handle = open(_SERVER_LOG, "ab")
            self._proc = subprocess.Popen(
                command,
                stdout=self._log_handle,
                stderr=subprocess.STDOUT,
                env=os.environ.copy(),
            )
            self._port = port
            deadline = time.monotonic() + 240.0
            while time.monotonic() < deadline:
                if self._proc.poll() is not None:
                    raise RuntimeError(
                        f"llama-server exited during startup (code={self._proc.returncode}); "
                        f"see {_SERVER_LOG}"
                    )
                if self._healthy():
                    logger.info("llama-server Q8 ready on port %s", port)
                    return
                time.sleep(0.5)
            self._stop_server_locked()
            raise RuntimeError("llama-server did not become healthy within 240s")

    def _stop_server_locked(self) -> None:
        proc = self._proc
        self._proc = None
        if proc is not None and proc.poll() is None:
            proc.terminate()
            try:
                proc.wait(timeout=15)
            except subprocess.TimeoutExpired:
                proc.kill()
                try:
                    proc.wait(timeout=5)
                except subprocess.TimeoutExpired:
                    pass
        if self._log_handle is not None:
            try:
                self._log_handle.close()
            except Exception:
                pass
            self._log_handle = None
        self._port = 0

    def shutdown(self) -> None:
        with self._lock:
            self._stop_server_locked()
        with self._aligner_lock:
            if self._aligner is not None:
                try:
                    self._aligner.shutdown()
                except Exception:
                    pass
                self._aligner = None
                self._aligner_sleeping = False

    # ------------------------------------------------------------------
    # HTTP generation
    # ------------------------------------------------------------------

    def _chat(
        self,
        audio: np.ndarray,
        *,
        context: str = "",
        prefix: str = "",
        max_tokens: int = 512,
    ) -> str:
        """One chat completion; returns raw text including the assistant prefix."""
        self._ensure_server()
        if len(audio) == 0:
            return prefix
        payload = {
            "messages": [],
            "max_tokens": int(max_tokens),
            "temperature": 0.0,
            "stream": False,
            "add_generation_prompt": not bool(prefix),
        }
        if context.strip():
            payload["messages"].append({"role": "system", "content": context.strip()})
        payload["messages"].append(
            {
                "role": "user",
                "content": [
                    {
                        "type": "input_audio",
                        "input_audio": {
                            "data": base64.b64encode(_pcm_to_wav_bytes(audio)).decode(),
                            "format": "wav",
                        },
                    },
                    {"type": "text", "text": "Transcribe this audio."},
                ],
            }
        )
        if prefix:
            payload["messages"].append({"role": "assistant", "content": prefix})
        response = requests.post(
            f"http://127.0.0.1:{self._port}/v1/chat/completions",
            json=payload,
            timeout=self._request_timeout,
        )
        if response.status_code != 200:
            raise RuntimeError(
                f"llama-server chat failed: HTTP {response.status_code}: {response.text[:300]}"
            )
        return str(response.json()["choices"][0]["message"]["content"] or "")

    def _transcribe_audio(
        self,
        audio: np.ndarray,
        *,
        context: str = "",
        language: str | None = None,
        enable_itn: bool = False,
    ) -> str:
        forced = _normalize_language_name(language)
        pieces = _split_audio(audio, self._max_segment_sec)
        texts: list[str] = []
        for index, piece in enumerate(pieces):
            prefix = f"language {forced}<asr_text>" if forced else ""
            raw = self._chat(
                piece,
                context=context,
                prefix=prefix,
                max_tokens=1024 if index == 0 else 512,
            )
            _, body = _parse_output(raw, forced)
            if forced == "Chinese" or not forced:
                body = _strip_cjk_spaces(body)
            if body.strip():
                texts.append(body.strip())
        text = _join_segments(texts, forced) if texts else ""
        return normalize_asr_text(text, enable_itn=enable_itn)

    # ------------------------------------------------------------------
    # Offline API (called by Qwen3ASREngine)
    # ------------------------------------------------------------------

    def transcribe_text(
        self,
        audio_path: str,
        context: str = "",
        language: str | None = None,
        enable_itn: bool = True,
    ) -> str:
        return self._transcribe_audio(
            _load_audio(audio_path),
            context=context,
            language=language,
            enable_itn=enable_itn,
        )

    def transcribe_array(
        self,
        audio: np.ndarray,
        context: str = "",
        language: str | None = None,
        enable_itn: bool = False,
    ) -> str:
        return self._transcribe_audio(
            np.asarray(audio, dtype=np.float32),
            context=context,
            language=language,
            enable_itn=enable_itn,
        )

    def transcribe_raw(
        self,
        audio_path: str,
        context: str = "",
        language: str | None = None,
        word_timestamps: bool = False,
        enable_itn: bool = False,
    ) -> ASRRawResult:
        audio = _load_audio(audio_path)
        text = self._transcribe_audio(
            audio, context=context, language=language, enable_itn=enable_itn
        )
        if not word_timestamps or not text:
            return ASRRawResult(
                text=text,
                segments=[ASRSegmentResult(text=text, start_time=0.0, end_time=0.0)]
                if text
                else [],
            )
        word_tokens = self._word_tokens(audio_path, text, language, audio)
        if not word_tokens:
            return ASRRawResult(
                text=text,
                segments=[ASRSegmentResult(text=text, start_time=0.0, end_time=0.0)],
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
        language: str | None = None,
        word_timestamps: bool = False,
        enable_itn: bool = False,
    ) -> list[ASRSegmentResult]:
        results: list[ASRSegmentResult] = []
        for path in audio_paths:
            audio = _load_audio(path)
            text = self._transcribe_audio(
                audio, context=context, language=language, enable_itn=enable_itn
            )
            if not word_timestamps or not text:
                results.append(ASRSegmentResult(text=text, start_time=0.0, end_time=0.0))
                continue
            word_tokens = self._word_tokens(path, text, language, audio)
            results.append(
                ASRSegmentResult(
                    text=text,
                    start_time=word_tokens[0].start_time if word_tokens else 0.0,
                    end_time=word_tokens[-1].end_time if word_tokens else 0.0,
                    word_tokens=word_tokens or None,
                )
            )
        return results

    def _word_tokens(
        self,
        audio_path: str,
        text: str,
        language: str | None,
        audio: np.ndarray,
    ) -> list[WordToken]:
        try:
            aligned = self.align_transcript(
                audio_path=audio_path, text=text, language=language, audio=audio
            )
        except Exception as exc:  # noqa: BLE001 - 对齐失败不阻断文本返回
            logger.warning("Q8 forced alignment failed: %s", exc)
            return []
        return [
            WordToken(
                text=str(item["text"]),
                start_time=round(float(item["start_ms"]) / 1000.0, 3),
                end_time=round(float(item["end_ms"]) / 1000.0, 3),
            )
            for item in aligned
        ]

    # ------------------------------------------------------------------
    # Forced aligner (same Qwen3-ForcedAligner used by the vLLM backend)
    # ------------------------------------------------------------------

    def _get_aligner(self) -> Any:
        if not self._forced_aligner_path:
            raise RuntimeError("word_timestamps requires a configured forced aligner model")
        with self._aligner_lock:
            return self._get_aligner_locked()

    def _get_aligner_locked(self) -> Any:
        if self._aligner is None:
            from vllm import LLM

            kwargs: dict[str, Any] = {
                "model": self._forced_aligner_path,
                "runner": "pooling",
                "enforce_eager": True,
                "hf_overrides": {
                    "architectures": ["Qwen3ASRForcedAlignerForTokenClassification"],
                },
                "enable_sleep_mode": True,
                "mm_processor_cache_gb": 0,
                "disable_log_stats": True,
            }
            util = (
                self._aligner_gpu_util
                if self._aligner_gpu_util is not None
                else settings.QWEN_FORCE_ALIGNER_GPU_MEMORY_UTILIZATION
            )
            if util is None:
                util = settings.QWEN_GPU_MEMORY_UTILIZATION
            # Q8 ASR（llama-server）常驻约 4GB，面向 vLLM 主引擎 sleep 后的 0.2
            # 会因 KV cache 不足白等约 90s 才报错；直接保底 0.35。
            util = max(float(util if util is not None else 0.35), 0.35)
            if self._aligner_max_model_len is not None:
                kwargs["max_model_len"] = int(self._aligner_max_model_len)
            logger.info("Loading Qwen3 forced aligner for Q8 backend: %s", self._forced_aligner_path)
            # Q8 主模型（llama-server）常驻约 4GB，配置值（面向 vLLM 主引擎 sleep 后）
            # 可能不够 KV cache，按需逐步提高 utilization 重试。
            last_error: Exception | None = None
            for attempt, candidate_util in enumerate((util, util + 0.1, util + 0.2)):
                try:
                    self._aligner = LLM(
                        **{**kwargs, "gpu_memory_utilization": candidate_util}
                    )
                    break
                except Exception as exc:  # noqa: BLE001 - 仅在显存不足时重试
                    last_error = exc
                    message = str(exc)
                    if (
                        "No available memory for the cache blocks" not in message
                        and "Engine core initialization failed" not in message
                    ):
                        raise
                    logger.warning(
                        "Forced aligner KV cache insufficient at gpu_memory_utilization=%s "
                        "(attempt %s/3), retrying with more memory",
                        candidate_util,
                        attempt + 1,
                    )
            if self._aligner is None:
                raise RuntimeError(f"Failed to initialize forced aligner: {last_error}")
            engine = getattr(self._aligner, "llm_engine", None)
            if engine is None:
                raise RuntimeError("Forced aligner did not expose a vLLM engine instance")
            model_config = getattr(
                getattr(engine, "vllm_config", None), "model_config", None
            ) or getattr(engine, "model_config", None)
            config = model_config.hf_config
            self._timestamp_token_id = int(config.timestamp_token_id)
            self._timestamp_segment_time = float(config.timestamp_segment_time)
            self._aligner_sleeping = False
        return self._aligner

    def align_transcript(
        self,
        audio_path: str,
        text: str,
        language: str | None = None,
        audio: np.ndarray | None = None,
    ) -> list[dict[str, float | str]]:
        tokens = _split_alignment_units(text)
        if not tokens:
            return []
        aligner = self._get_aligner()
        prompt = _build_alignment_prompt(tokens)
        audio_array = audio if audio is not None else _load_audio(audio_path)
        with self._aligner_lock:
            self._aligner_busy += 1
            try:
                outputs = aligner.encode(
                    [{"prompt": prompt, "multi_modal_data": {"audio": audio_array}}],
                    pooling_task="token_classify",
                )
            finally:
                self._aligner_busy -= 1
        output = outputs[0]
        logits = output.outputs.data
        predictions = (
            logits.argmax(dim=-1)
            if hasattr(logits, "argmax")
            else np.argmax(logits, axis=-1)
        )
        ts_predictions = [
            float(pred.item() if hasattr(pred, "item") else pred)
            * float(self._timestamp_segment_time or 0.0)
            for tid, pred in zip(output.prompt_token_ids, predictions)
            if int(tid) == int(self._timestamp_token_id or -1)
        ]
        if len(ts_predictions) < len(tokens) * 2:
            raise RuntimeError(
                "Forced aligner returned fewer timestamp predictions than expected: "
                f"expected={len(tokens) * 2}, got={len(ts_predictions)}"
            )
        fixed = _fix_timestamp(ts_predictions)
        self._aligner_last_use = time.monotonic()
        return [
            {
                "text": token,
                "start_ms": fixed[index * 2],
                "end_ms": fixed[index * 2 + 1],
            }
            for index, token in enumerate(tokens)
        ]

    @property
    def is_aligner_loaded(self) -> bool:
        return self._aligner is not None

    @property
    def is_aligner_busy(self) -> bool:
        with self._aligner_lock:
            return self._aligner_busy > 0

    def aligner_idle_seconds(self) -> float:
        if self._aligner_last_use <= 0:
            return 0.0
        return max(0.0, time.monotonic() - self._aligner_last_use)

    def sleep_aligner(self) -> None:
        with self._aligner_lock:
            if self._aligner is None or self._aligner_sleeping:
                return
            started = time.monotonic()
            self._aligner.sleep(level=1)
            self._aligner_sleeping = True
            logger.info(
                "Q8 forced aligner slept in %.1fs, GPU memory released",
                time.monotonic() - started,
            )

    def wake_aligner(self) -> None:
        with self._aligner_lock:
            if self._aligner is None or not self._aligner_sleeping:
                return
            started = time.monotonic()
            self._aligner.wake_up()
            self._aligner_sleeping = False
            logger.info(
                "Q8 forced aligner woke up in %.2fs", time.monotonic() - started
            )

    def supports_sleep(self) -> bool:
        return False

    def sleep_main(self) -> None:
        return None

    def wake_main(self) -> None:
        return None

    def is_main_sleeping(self) -> bool:
        return False

    # ------------------------------------------------------------------
    # Streaming
    # ------------------------------------------------------------------

    def init_streaming_state(
        self,
        *,
        context: str = "",
        language: str | None = None,
        chunk_size_sec: float = 0.16,
        unfixed_chunk_num: int = 0,
        unfixed_token_num: int = 1,
        max_new_tokens: int = 512,
        lookahead_sec: float = 0.16,
        max_new_tokens_initial: int | None = None,
        warmup_chunks: int | None = None,
        initial_audio: np.ndarray | None = None,
    ) -> Q8StreamingState:
        forced = _normalize_language_name(language)
        return Q8StreamingState(
            context=context or "",
            language=forced or "",
            force_language=forced,
            chunk_size_sec=float(chunk_size_sec),
            lookahead_sec=float(lookahead_sec),
            unfixed_token_num=int(unfixed_token_num),
            max_new_tokens=int(max_new_tokens),
            audio=(
                np.asarray(initial_audio, dtype=np.float32)
                if initial_audio is not None
                else np.empty(0, dtype=np.float32)
            ),
        )

    # 流式重解码间隔（秒）。llama-server 每轮做全量 prefill + 生成，
    # 20s 音频约 0.3s；用 1s 间隔在 160ms 帧预算内保持实时，且避免
    # append-only prefix 在 llama.cpp 上不稳定导致的文本回退。
    _STREAM_DECODE_INTERVAL_SEC = 1.0

    def feed_stream(self, pcm: np.ndarray, state: Q8StreamingState) -> Q8StreamingState:
        state.buffer = np.concatenate(
            [state.buffer, np.asarray(pcm, dtype=np.float32).reshape(-1)]
        )
        interval = (
            max(0.2, float(state.chunk_size_sec) + float(state.lookahead_sec))
            if state.chunk_id == 0
            else self._STREAM_DECODE_INTERVAL_SEC
        )
        if len(state.buffer) < int(interval * _SAMPLE_RATE):
            return state
        state.audio = np.concatenate([state.audio, state.buffer])
        state.buffer = np.empty(0, dtype=np.float32)
        state.text = self._transcribe_audio(
            state.audio,
            context=state.context,
            language=state.force_language,
            enable_itn=False,
        )
        state.raw = state.text
        state.preview = state.text
        state.chunk_id += 1
        return state

    def finish_stream(self, state: Q8StreamingState) -> Q8StreamingState:
        if len(state.buffer) > 0:
            state.audio = np.concatenate([state.audio, state.buffer])
            state.buffer = np.empty(0, dtype=np.float32)
        if len(state.audio) == 0:
            return state
        # 收尾用完整音频从头解码，保证句尾完整（避免 append-only 截断）
        state.text = self._transcribe_audio(
            state.audio,
            context=state.context,
            language=state.force_language,
            enable_itn=False,
        )
        state.raw = state.text
        state.preview = state.text
        state.chunk_id += 1
        return state
