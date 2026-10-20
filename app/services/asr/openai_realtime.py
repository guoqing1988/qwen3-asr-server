# -*- coding: utf-8 -*-
"""OpenAI Realtime transcription (WebSocket subset) on the shared Qwen3 engine.

只实现 transcription 子集（`intent=transcription`），事件与字段对齐 OpenAI
Realtime API，便于现有客户端直接接入；识别仍复用 `Qwen3ASREngine` 的流式接口。

流式 partial 关闭 ITN（数字/标点规范化会改写前缀，破坏 delta 单调性），
段定稿时再开启 ITN，完成事件里给最终文本。
"""

from __future__ import annotations

import base64
import binascii
import json
import logging
import uuid
from dataclasses import dataclass, field
from typing import Any

import numpy as np
from fastapi import WebSocket, WebSocketDisconnect

from app.core.config import settings
from .realtime_speakers import SpeakerStream
from app.core.executor import run_sync
from app.services.asr.model_selection import validate_realtime_model_id
from app.services.asr.qwen3_engine import Qwen3ASREngine, Qwen3StreamingState
from app.services.asr.runtime import RuntimeEngineLease, get_runtime_router
from app.services.asr.stream_vad import StreamVAD, create_stream_vad
from app.utils.text_processing import merge_hotwords, normalize_asr_text

logger = logging.getLogger(__name__)

_SAMPLE_RATE = 16000
_SUPPORTED_RATES = (16000, 24000)
_MAX_APPEND_SECONDS = 10
_MIN_SEGMENT_VOICE_SEC = 0.8
_VOICE_RMS_THRESHOLD = 0.015


def _event_id() -> str:
    return "event_" + uuid.uuid4().hex


def _item_id() -> str:
    return "item_" + uuid.uuid4().hex


@dataclass
class RealtimeTranscriptionConfig:
    """`session.audio.input.format` + `session.audio.input.transcription` 的解析结果。"""

    model: str = "qwen3-asr"
    prompt: str = ""
    language: str | None = None
    keywords: list[str] = field(default_factory=list)
    input_rate: int = _SAMPLE_RATE
    turn_detection: str | None = None  # None=客户端 commit；"server_vad"=静音自动落定

    def context(self) -> str:
        parts = [
            self.prompt.strip(),
            ", ".join(self.keywords),
        ]
        return merge_hotwords(
            settings.ASR_DEFAULT_HOTWORDS, "，".join(part for part in parts if part)
        )


class _Resampler:
    """PCM16 输入重采样到 16k（仅 24k 输入需要）。"""

    def __init__(self, rate: int) -> None:
        self._stream = None
        if rate != _SAMPLE_RATE:
            import soxr

            self._stream = soxr.ResampleStream(rate, _SAMPLE_RATE, 1, dtype="float32")

    def push(self, pcm: np.ndarray) -> np.ndarray:
        if self._stream is None:
            return pcm
        return self._stream.resample_chunk(pcm.astype(np.float32), last=False)


class OpenAIRealtimeSession:
    """单条 WebSocket 连接的 OpenAI Realtime 转录会话。"""

    def __init__(self, websocket: WebSocket, task_id: str) -> None:
        self._ws = websocket
        self._task_id = task_id
        self._config = RealtimeTranscriptionConfig()
        self._engine: Qwen3ASREngine | None = None
        self._lease: RuntimeEngineLease | None = None
        self._vad: StreamVAD | None = None
        self._state: Qwen3StreamingState | None = None
        self._resampler = _Resampler(_SAMPLE_RATE)
        self._buffer = np.zeros(0, dtype=np.float32)
        self._segment_audio = np.zeros(0, dtype=np.float32)
        self._silence_samples = 0
        self._segment_voice_samples = 0
        self._total_samples = 0
        self._item_id = _item_id()
        self._partial_text = ""
        self._closed = False
        # 实时说话人标签（Nemotron 与 ASR 并行消费同一路 PCM）
        self._speakers: SpeakerStream | None = (
            SpeakerStream() if settings.REALTIME_SPEAKER_LABELS else None
        )
        self._speaker_index = 0
        self._segment_start_sample = 0

    # ------------------------------------------------------------------
    # 引擎与段状态
    # ------------------------------------------------------------------

    async def _ensure_engine(self) -> Qwen3ASREngine:
        if self._engine is not None:
            return self._engine
        model = validate_realtime_model_id("qwen3-asr")
        self._lease = await get_runtime_router().acquire_engine(model)
        engine = self._lease.engine
        if not isinstance(engine, Qwen3ASREngine):
            raise RuntimeError("Current model is not Qwen3-ASR")
        if not engine.supports_realtime:
            raise RuntimeError(
                f"Current device {engine.device} does not support Qwen3-ASR realtime streaming"
            )
        self._engine = engine
        self._vad = self._vad or create_stream_vad()
        self._state = self._new_state()
        return engine

    def _new_state(self) -> Qwen3StreamingState | None:
        if self._engine is None:
            return None
        return self._engine.init_streaming_state(
            context=self._config.context(),
            language=self._config.language,
            chunk_size_sec=settings.QWEN_STREAM_CHUNK_SEC,
            lookahead_sec=settings.QWEN_STREAM_LOOKAHEAD_SEC,
            unfixed_chunk_num=settings.QWEN_STREAM_UNFIXED_CHUNK_NUM,
            unfixed_token_num=settings.QWEN_STREAM_UNFIXED_TOKEN_NUM,
            max_new_tokens=settings.QWEN_STREAM_MAX_NEW_TOKENS,
        )

    def _reset_segment(self, *, keep_buffer: bool) -> None:
        if not keep_buffer:
            self._buffer = np.zeros(0, dtype=np.float32)
        self._segment_audio = np.zeros(0, dtype=np.float32)
        self._silence_samples = 0
        self._segment_voice_samples = 0
        # 段长计时：buffer 中未消费的部分属于下一段，保留其长度
        self._total_samples = len(self._buffer)
        self._partial_text = ""
        self._item_id = _item_id()
        if self._vad is not None:
            self._vad.reset()
        self._state = self._new_state()
        # 下一段起点 = 已喂入总量 - 尚未消费的 buffer（该部分属于下一段）
        if self._speakers is not None:
            if self._speakers.flushed:
                # flush 后流已终结（末段收尾推理用掉了剩余音频）：重建并沿用编号。
                # buffer 中未消费的音频属于下一段，补喂给新流以保持时间轴对齐。
                names = self._speakers.names
                self._speakers = SpeakerStream(names)
                if self._buffer.size:
                    self._speakers.feed(
                        (np.clip(self._buffer, -1.0, 1.0) * 32767.0)
                        .astype("<i2")
                        .tobytes()
                    )
            self._segment_start_sample = self._speakers.samples - len(self._buffer)

    def _text(self, text: str, *, itn: bool) -> str:
        return normalize_asr_text(text, enable_itn=itn)

    # ------------------------------------------------------------------
    # 事件发送
    # ------------------------------------------------------------------

    async def _send(self, payload: dict[str, Any]) -> None:
        payload.setdefault("event_id", _event_id())
        await self._ws.send_json(payload)

    async def _send_error(
        self, code: str, message: str, *, error_type: str = "invalid_request_error"
    ) -> None:
        await self._send(
            {
                "type": "error",
                "error": {"type": error_type, "code": code, "message": message},
            }
        )

    def _session_payload(self) -> dict[str, Any]:
        return {
            "object": "realtime.transcription_session",
            "id": "sess_" + self._task_id,
            "model": self._config.model,
            "speaker_diarization": self._speakers is not None,
            "audio": {
                "input": {
                    "format": {"type": "audio/pcm", "rate": self._config.input_rate},
                    "transcription": {
                        "model": self._config.model,
                        "prompt": self._config.prompt,
                        "language": self._config.language,
                        "keywords": self._config.keywords,
                    },
                    "turn_detection": (
                        {"type": "server_vad"}
                        if self._config.turn_detection == "server_vad"
                        else None
                    ),
                }
            },
        }

    # ------------------------------------------------------------------
    # 会话主循环
    # ------------------------------------------------------------------

    async def handle(self) -> None:
        await self._ws.accept(
            subprotocol=(
                "realtime"
                if "realtime" in self._ws.scope.get("subprotocols", [])
                else None
            )
        )
        await self._send(
            {"type": "session.created", "session": self._session_payload()}
        )
        try:
            while not self._closed:
                message = await self._ws.receive()
                if message.get("type") == "websocket.disconnect":
                    break
                raw = message.get("text")
                if raw is None:
                    await self._send_error(
                        "invalid_event", "Only JSON text events are supported"
                    )
                    continue
                await self._dispatch(raw)
        except WebSocketDisconnect:
            logger.info("[%s] Realtime client disconnected", self._task_id)
        except Exception as exc:  # noqa: BLE001 - 单连接失败不影响服务
            logger.error("[%s] Realtime session failed: %s", self._task_id, exc)
            try:
                await self._send_error("stream_failed", str(exc), error_type="server_error")
            except Exception:  # noqa: BLE001
                pass
        finally:
            await self._close()

    async def _dispatch(self, raw: str) -> None:
        try:
            event = json.loads(raw)
        except json.JSONDecodeError:
            await self._send_error("invalid_event", "Event must be valid JSON")
            return
        if not isinstance(event, dict):
            await self._send_error("invalid_event", "Event must be a JSON object")
            return
        kind = event.get("type")
        if kind == "session.update":
            await self._handle_session_update(event)
        elif kind == "input_audio_buffer.append":
            await self._handle_append(event)
        elif kind == "input_audio_buffer.commit":
            await self._handle_commit()
        elif kind == "input_audio_buffer.clear":
            self._reset_segment(keep_buffer=False)
            await self._send({"type": "input_audio_buffer.cleared"})
        else:
            await self._send_error(
                "unsupported_event",
                f"Unsupported event type: {kind!r}; supported: session.update, "
                "input_audio_buffer.append, input_audio_buffer.commit, "
                "input_audio_buffer.clear",
            )

    async def _handle_session_update(self, event: dict[str, Any]) -> None:
        session = event.get("session") or {}
        audio = (session.get("audio") or {}).get("input") or {}
        transcription = audio.get("transcription") or {}
        fmt = audio.get("format") or {}

        rate = int(fmt.get("rate") or self._config.input_rate)
        if rate not in _SUPPORTED_RATES:
            await self._send_error("invalid_rate", f"Unsupported sample rate: {rate}")
            return
        keywords = transcription.get("keywords") or []
        if isinstance(keywords, str):
            keywords = [item.strip() for item in keywords.split(",") if item.strip()]
        if not isinstance(keywords, list) or any(
            not isinstance(item, str)
            or not item.strip()
            or "<" in item
            or ">" in item
            or "\n" in item
            for item in keywords
        ):
            await self._send_error(
                "invalid_keywords", "keywords must be nonempty single-line strings"
            )
            return
        language = transcription.get("language")
        turn_detection = audio.get("turn_detection")
        mode = None
        if isinstance(turn_detection, dict) and turn_detection.get("type") == "server_vad":
            mode = "server_vad"

        self._config = RealtimeTranscriptionConfig(
            model=str(transcription.get("model") or self._config.model),
            prompt=str(transcription.get("prompt") or ""),
            language=str(language) if language else None,
            keywords=[str(item).strip() for item in keywords],
            input_rate=rate,
            turn_detection=mode,
        )
        self._resampler = _Resampler(rate)
        # 与 OpenAI 一致：session.update 丢弃未提交音频，按新参数重建状态
        self._state = None
        self._reset_segment(keep_buffer=False)
        await self._send({"type": "session.updated", "session": self._session_payload()})

    async def _handle_append(self, event: dict[str, Any]) -> None:
        encoded = event.get("audio")
        if not isinstance(encoded, str) or not encoded:
            await self._send_error("invalid_audio", "audio must be base64-encoded PCM16")
            return
        max_bytes = _SUPPORTED_RATES[-1] * 2 * _MAX_APPEND_SECONDS
        if len(encoded) > (max_bytes + 2) // 3 * 4 + 4096:
            await self._send_error("audio_too_large", "append exceeds 10s of audio")
            return
        try:
            pcm_bytes = base64.b64decode(encoded, validate=True)
        except (binascii.Error, ValueError):
            await self._send_error("invalid_audio", "audio is not valid base64")
            return
        if len(pcm_bytes) % 2:
            await self._send_error(
                "invalid_audio", "PCM16 payload must have even length"
            )
            return
        samples = np.frombuffer(pcm_bytes, dtype="<i2").astype(np.float32) / 32768.0
        await self._feed(self._resampler.push(samples))

    async def _handle_commit(self) -> None:
        # OpenAI 顺序：先 committed，再 completed（两者同一 item_id）
        await self._send(
            {"type": "input_audio_buffer.committed", "item_id": self._item_id}
        )
        await self._finalize(reason="commit")
        self._reset_segment(keep_buffer=False)

    async def _feed(self, audio: np.ndarray) -> None:
        if audio.size == 0:
            return
        engine = await self._ensure_engine()
        if self._speakers is not None:
            # Nemotron 与 ASR 并行：把同一路 16k PCM16 喂入流式分离，
            # 并立刻推进已完整 chunk 的推理（否则会积压到定稿时一次性补齐而超时）
            self._speakers.feed(
                (np.clip(audio, -1.0, 1.0) * 32767.0).astype("<i2").tobytes()
            )
            await self._speakers.advance()
        self._buffer = np.concatenate([self._buffer, audio.astype(np.float32)])
        if float(np.sqrt(np.mean(audio**2))) >= _VOICE_RMS_THRESHOLD:
            self._silence_samples = 0
            self._segment_voice_samples += int(audio.size)
        else:
            self._silence_samples += int(audio.size)

        chunk_size = int(settings.QWEN_STREAM_CHUNK_SEC * _SAMPLE_RATE)
        max_samples = int(settings.QWEN_STREAM_SEGMENT_MAX_SEC * _SAMPLE_RATE)
        while self._buffer.size >= chunk_size:
            chunk = self._buffer[:chunk_size]
            self._buffer = self._buffer[chunk_size:]
            self._total_samples += int(chunk.size)
            speech_ended = False
            if self._vad is not None:
                speech_ended = bool(await run_sync(self._vad.detect_speech_end, chunk))
            else:
                speech_ended = self._silence_samples >= int(
                    settings.QWEN_STREAM_SILENCE_SEC * _SAMPLE_RATE
                )
            self._segment_audio = np.concatenate([self._segment_audio, chunk])
            self._state = await run_sync(
                engine.streaming_transcribe, chunk, self._state
            )
            current = self._text(self._state.last_text or "", itn=False)
            if current and current.startswith(self._partial_text) and current != self._partial_text:
                delta = current[len(self._partial_text) :]
                self._partial_text = current
                await self._send(
                    {
                        "type": "conversation.item.input_audio_transcription.delta",
                        "item_id": self._item_id,
                        "content_index": 0,
                        "delta": delta,
                    }
                )
            elif current and not current.startswith(self._partial_text):
                # 前缀被改写（极少见；流式已关闭 ITN）→ 重发全文
                self._partial_text = current
                await self._send(
                    {
                        "type": "conversation.item.input_audio_transcription.delta",
                        "item_id": self._item_id,
                        "content_index": 0,
                        "delta": current,
                    }
                )
            # 连续讲话无静音时用段长上限强制落定
            force_finalize = self._total_samples >= max_samples
            if not speech_ended and not force_finalize:
                continue
            if self._config.turn_detection != "server_vad":
                # 手动 commit 模式：句尾只暂停消费，不落定
                break
            if not force_finalize and self._segment_voice_samples < int(
                _MIN_SEGMENT_VOICE_SEC * _SAMPLE_RATE
            ):
                # 段首静音（有效语音不足）：继续等待真实语音
                break
            await self._send(
                {"type": "input_audio_buffer.committed", "item_id": self._item_id}
            )
            await self._finalize(reason="silence" if speech_ended else "max_duration")
            # 剩余 buffer 属于下一段（VAD 在 chunk 边界落定）
            self._reset_segment(keep_buffer=True)
            break

    async def _finalize(self, *, reason: str) -> str | None:
        engine = self._engine
        if engine is None or self._state is None or self._state.chunk_count == 0:
            return None
        speaker_index = self._speaker_index
        speaker: str | None = None
        if self._speakers is not None and self._segment_audio.size > 0:
            end_sample = self._segment_start_sample + int(self._segment_audio.size)
            self._speakers.utterance(
                speaker_index,
                round(self._segment_start_sample * 1000 / _SAMPLE_RATE),
                round(end_sample * 1000 / _SAMPLE_RATE),
            )
            self._speaker_index += 1
        try:
            self._state = await run_sync(engine.finish_streaming_transcribe, self._state)
            text = self._text(self._state.last_text or "", itn=True)
            if settings.QWEN_STREAM_FINAL_PASS and self._segment_audio.size > 0 and text.strip():
                try:
                    refined = await run_sync(
                        engine.transcribe_array,
                        self._segment_audio,
                        self._config.context(),
                        self._config.language,
                        True,
                    )
                    refined = self._text(refined, itn=True)
                    if refined.strip():
                        text = refined
                except Exception as exc:  # noqa: BLE001 - 重转失败保留流式结果
                    logger.warning("[%s] Final pass failed: %s", self._task_id, exc)
        except Exception as exc:  # noqa: BLE001
            logger.error("[%s] Segment finalize failed: %s", self._task_id, exc)
            await self._send(
                {
                    "type": "conversation.item.input_audio_transcription.failed",
                    "item_id": self._item_id,
                    "content_index": 0,
                    "error": {"type": "server_error", "message": str(exc)},
                }
            )
            return None
        if len(text.strip()) < 2:
            return None
        if self._speakers is not None and self._segment_audio.size > 0:
            # two-pass 重转已耗时，标签大概率就绪；超时/失败则为 null
            speaker = await self._speakers.label(
                speaker_index, settings.REALTIME_SPEAKER_LABEL_TIMEOUT
            )
        await self._send(
            {
                "type": "conversation.item.input_audio_transcription.completed",
                "item_id": self._item_id,
                "content_index": 0,
                "transcript": text,
                "speaker": speaker,
            }
        )
        logger.info("[%s] Segment completed (%s): %r", self._task_id, reason, text)
        return text

    async def _close(self) -> None:
        self._closed = True
        close_stream = getattr(self._state, "internal_state", None)
        if close_stream is not None and callable(getattr(close_stream, "close", None)):
            close_stream.close()
        if self._speakers is not None:
            self._speakers.close()
        if self._lease is not None:
            await self._lease.close()
        logger.info("[%s] Realtime session closed", self._task_id)
