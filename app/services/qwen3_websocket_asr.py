# -*- coding: utf-8 -*-
"""Qwen3-ASR websocket streaming service."""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field
from enum import IntEnum
from typing import Any, Dict, List, Optional

import numpy as np
from fastapi import WebSocket, WebSocketDisconnect

from app.core.config import settings
from app.core.exceptions import create_error_response
from app.core.executor import run_sync
from app.services.asr.model_selection import validate_realtime_model_id
from app.services.asr.qwen3_engine import Qwen3ASREngine, Qwen3StreamingState
from app.services.asr.runtime import RuntimeEngineLease, get_runtime_router
from app.services.asr.stream_vad import StreamVAD, create_stream_vad
from app.utils.text_processing import merge_hotwords, normalize_asr_text

logger = logging.getLogger(__name__)

# 流式音频固定按 16kHz 处理；段/静音阈值以秒配置，此处换算为采样点数
_SAMPLE_RATE = 16000


def _convert_audio(
    audio_bytes: bytes,
    fmt: str,
    sample_rate: int,
) -> Optional[np.ndarray]:
    try:
        if fmt == "wav" and len(audio_bytes) > 44:
            audio_bytes = audio_bytes[44:]

        audio = np.frombuffer(audio_bytes, dtype=np.int16).astype(np.float32) / 32768.0

        if sample_rate != 16000:
            import scipy.signal

            num = int(len(audio) * 16000 / sample_rate)
            audio = scipy.signal.resample(audio, num)
            if isinstance(audio, tuple):
                audio = audio[0]

        return audio
    except Exception as exc:
        logger.error("Audio conversion failed: %s", exc)
        return None


class ConnectionState(IntEnum):
    READY = 1
    STARTED = 2
    STREAMING = 3


@dataclass
class ConnectionContext:
    state: ConnectionState = ConnectionState.READY
    params: Dict[str, Any] = field(default_factory=dict)
    engine_lease: Optional[RuntimeEngineLease] = None
    engine: Optional[Qwen3ASREngine] = None
    audio_buffer: np.ndarray = field(
        default_factory=lambda: np.array([], dtype=np.float32)
    )
    streaming_state: Optional[Qwen3StreamingState] = None
    vad: Optional[StreamVAD] = None
    # 当前段的完整音频（two-pass 定稿重转用）
    segment_audio: np.ndarray = field(
        default_factory=lambda: np.array([], dtype=np.float32)
    )
    silence_samples: int = 0
    total_samples: int = 0
    confirmed_segments: List[Dict[str, Any]] = field(default_factory=list)
    segment_index: int = 0
    last_partial_text: str = ""



class Qwen3ASRService:
    async def _ensure_engine(self, ctx: ConnectionContext) -> Qwen3ASREngine:
        if ctx.engine is not None:
            return ctx.engine

        runtime_router = get_runtime_router()
        model = validate_realtime_model_id("qwen3-asr")
        logger.info("Using Qwen3-ASR model: %s", model)

        ctx.engine_lease = await runtime_router.acquire_engine(model)
        engine = ctx.engine_lease.engine
        if not isinstance(engine, Qwen3ASREngine):
            raise RuntimeError("Current model is not Qwen3-ASR")
        if not engine.supports_realtime:
            raise RuntimeError(
                f"Current device {engine.device} does not support Qwen3-ASR realtime streaming; "
                "only CUDA vLLM and CPU Rust paths are supported"
            )

        ctx.engine = engine
        if ctx.vad is None:
            ctx.vad = create_stream_vad()
        return engine

    def _has_voice(self, audio: np.ndarray) -> bool:
        return np.sqrt(np.mean(audio**2)) >= 0.015

    def _normalize_output_text(self, text: str, ctx: ConnectionContext) -> str:
        return normalize_asr_text(
            text,
            enable_itn=bool(ctx.params.get("enable_inverse_text_normalization", True)),
        )

    async def _finalize_segment(
        self,
        websocket: WebSocket,
        ctx: ConnectionContext,
        task_id: str,
        reason: str,
    ) -> None:
        """落定当前段：finish 流式状态 → 可选离线重转 → 推送 → 重建 state。

        VAD 在 chunk 边界触发，audio_buffer 中尚未处理的音频属于下一段，
        因此这里不清空 audio_buffer。
        """
        try:
            engine = await self._ensure_engine(ctx)
            segment_text = ""
            if ctx.streaming_state is not None and ctx.streaming_state.chunk_count > 0:
                ctx.streaming_state = await run_sync(
                    engine.finish_streaming_transcribe,
                    ctx.streaming_state,
                )
                segment_text = self._normalize_output_text(
                    ctx.streaming_state.last_text or "", ctx
                )

            # two-pass：段音频走离线路径重转，修正 append-only 流式无法回改的
            # 专名/热词错误；失败时保留流式结果
            if (
                settings.QWEN_STREAM_FINAL_PASS
                and ctx.segment_audio.size > 0
                and segment_text.strip()
            ):
                try:
                    refined = await run_sync(
                        engine.transcribe_array,
                        ctx.segment_audio,
                        ctx.params.get("context", ""),
                        ctx.params.get("language"),
                        bool(
                            ctx.params.get("enable_inverse_text_normalization", True)
                        ),
                    )
                    refined = self._normalize_output_text(refined, ctx)
                    if refined.strip():
                        if refined.strip() != segment_text.strip():
                            logger.info(
                                "[%s] Final pass corrected segment: %r -> %r",
                                task_id,
                                segment_text,
                                refined,
                            )
                        segment_text = refined
                except Exception as exc:  # noqa: BLE001 - 重转失败不影响流式结果
                    logger.warning(
                        "[%s] Segment final pass failed: %s", task_id, exc
                    )

            is_valid = len(segment_text.strip()) >= 2
            if is_valid:
                ctx.confirmed_segments.append(
                    {
                        "index": ctx.segment_index,
                        "text": segment_text,
                        "language": ctx.streaming_state.last_language,
                        "reason": reason,
                    }
                )

                full_text = "\n".join(
                    segment["text"] for segment in ctx.confirmed_segments
                )

                await websocket.send_json(
                    {
                        "type": "segment_end",
                        "task_id": task_id,
                        "segment_index": ctx.segment_index,
                        "reason": reason,
                        "result": {
                            "text": full_text,
                            "segment_text": segment_text,
                            "language": ctx.streaming_state.last_language,
                        },
                        "confirmed_texts": [
                            segment["text"] for segment in ctx.confirmed_segments
                        ],
                    }
                )

                ctx.segment_index += 1
                logger.info(
                    "[%s] Segment finalized: reason=%s text=%r",
                    task_id,
                    reason,
                    segment_text,
                )
            else:
                logger.debug("[%s] Short segment filtered: %r", task_id, segment_text)

            ctx.silence_samples = 0
            ctx.total_samples = 0
            ctx.last_partial_text = ""
            ctx.segment_audio = np.array([], dtype=np.float32)
            if ctx.vad is not None:
                ctx.vad.reset()
            ctx.streaming_state = engine.init_streaming_state(
                context=ctx.params.get("context", ""),
                language=ctx.params.get("language"),
                chunk_size_sec=ctx.params.get(
                    "chunk_size_sec", settings.QWEN_STREAM_CHUNK_SEC
                ),
                lookahead_sec=ctx.params.get(
                    "lookahead_sec", settings.QWEN_STREAM_LOOKAHEAD_SEC
                ),
                unfixed_chunk_num=ctx.params.get(
                    "unfixed_chunk_num", settings.QWEN_STREAM_UNFIXED_CHUNK_NUM
                ),
                unfixed_token_num=ctx.params.get(
                    "unfixed_token_num", settings.QWEN_STREAM_UNFIXED_TOKEN_NUM
                ),
                max_new_tokens=(
                    ctx.params["max_new_tokens"]
                    if ctx.params.get("max_new_tokens") is not None
                    else settings.QWEN_STREAM_MAX_NEW_TOKENS
                ),
            )

            if is_valid:
                await websocket.send_json(
                    {
                        "type": "segment_start",
                        "task_id": task_id,
                        "segment_index": ctx.segment_index,
                    }
                )

        except Exception as exc:
            logger.error("[%s] Finalize segment failed: %s", task_id, exc)
            raise

    async def _send_error(
        self,
        websocket: WebSocket,
        message: str,
        task_id: str,
        code: str = "DEFAULT_SERVER_ERROR",
    ) -> None:
        try:
            error = create_error_response(
                error_code=code, message=message, task_id=task_id
            )
            error["type"] = "error"
            await websocket.send_json(error)
        except Exception:
            pass

    async def handle_connection(self, websocket: WebSocket, task_id: str) -> None:
        await websocket.accept()
        logger.info("[%s] Qwen3 websocket connected", task_id)

        ctx = ConnectionContext()

        try:
            while True:
                message = await websocket.receive()

                if "text" in message:
                    data = json.loads(message["text"])
                    msg_type = data.get("type", "")

                    if msg_type == "start":
                        if ctx.state != ConnectionState.READY:
                            await self._send_error(
                                websocket, "识别已在进行中", task_id, "INVALID_STATE"
                            )
                            continue

                        payload = data.get("payload", {})
                        ctx.params = {
                            "format": payload.get("format", "pcm"),
                            "sample_rate": payload.get("sample_rate", 16000),
                            "language": payload.get("language"),
                            "context": merge_hotwords(
                                settings.ASR_DEFAULT_HOTWORDS,
                                payload.get("context", ""),
                            ),
                            "enable_inverse_text_normalization": payload.get(
                                "enable_inverse_text_normalization",
                                True,
                            ),
                            "chunk_size_sec": payload.get(
                                "chunk_size_sec", settings.QWEN_STREAM_CHUNK_SEC
                            ),
                            "lookahead_sec": payload.get(
                                "lookahead_sec", settings.QWEN_STREAM_LOOKAHEAD_SEC
                            ),
                            "unfixed_chunk_num": payload.get(
                                "unfixed_chunk_num", settings.QWEN_STREAM_UNFIXED_CHUNK_NUM
                            ),
                            "unfixed_token_num": payload.get(
                                "unfixed_token_num", settings.QWEN_STREAM_UNFIXED_TOKEN_NUM
                            ),
                            "max_new_tokens": payload.get(
                                "max_new_tokens", settings.QWEN_STREAM_MAX_NEW_TOKENS
                            ),
                        }

                        engine = await self._ensure_engine(ctx)
                        ctx.streaming_state = engine.init_streaming_state(
                            context=ctx.params["context"],
                            language=ctx.params["language"],
                            chunk_size_sec=ctx.params["chunk_size_sec"],
                            lookahead_sec=ctx.params["lookahead_sec"],
                            unfixed_chunk_num=ctx.params["unfixed_chunk_num"],
                            unfixed_token_num=ctx.params["unfixed_token_num"],
                            max_new_tokens=(
                                ctx.params["max_new_tokens"]
                                if ctx.params.get("max_new_tokens") is not None
                                else settings.QWEN_STREAM_MAX_NEW_TOKENS
                            ),
                        )

                        await websocket.send_json(
                            {
                                "type": "started",
                                "task_id": task_id,
                                "params": ctx.params,
                            }
                        )
                        ctx.state = ConnectionState.STARTED
                        logger.info("[%s] Recognition started: %s", task_id, ctx.params)

                    elif msg_type == "stop":
                        if ctx.state in (
                            ConnectionState.STARTED,
                            ConnectionState.STREAMING,
                        ):
                            await self._stop(websocket, ctx, task_id)
                        break

                    else:
                        await self._send_error(
                            websocket,
                            f"未知消息类型: {msg_type}",
                            task_id,
                            "INVALID_MESSAGE",
                        )

                elif "bytes" in message:
                    if ctx.state not in (
                        ConnectionState.STARTED,
                        ConnectionState.STREAMING,
                    ):
                        await self._send_error(
                            websocket, "请先发送 start", task_id, "INVALID_STATE"
                        )
                        continue

                    audio = _convert_audio(
                        message["bytes"],
                        ctx.params["format"],
                        ctx.params["sample_rate"],
                    )
                    if audio is None:
                        continue

                    ctx.total_samples += len(audio)
                    ctx.segment_audio = np.concatenate([ctx.segment_audio, audio])

                    if self._has_voice(audio):
                        ctx.silence_samples = 0
                    else:
                        ctx.silence_samples += len(audio)

                    ctx.audio_buffer = np.concatenate([ctx.audio_buffer, audio])

                    chunk_size = int(ctx.params["chunk_size_sec"] * _SAMPLE_RATE)
                    results = []
                    speech_ended = False
                    engine = await self._ensure_engine(ctx)

                    while len(ctx.audio_buffer) >= chunk_size:
                        chunk = ctx.audio_buffer[:chunk_size]
                        ctx.audio_buffer = ctx.audio_buffer[chunk_size:]

                        # 优先用 FireRedVAD 的句尾静音事件；不可用时回退能量阈值
                        if ctx.vad is not None:
                            speech_ended = await run_sync(
                                ctx.vad.detect_speech_end, chunk
                            )
                        else:
                            speech_ended = ctx.silence_samples >= int(
                                settings.QWEN_STREAM_SILENCE_SEC * _SAMPLE_RATE
                            )

                        ctx.streaming_state = await run_sync(
                            engine.streaming_transcribe,
                            chunk,
                            ctx.streaming_state,
                        )

                        # 句尾已出现：停止消费缓冲区，剩余音频属于下一段
                        if speech_ended:
                            break

                        current = self._normalize_output_text(
                            ctx.streaming_state.last_text or "",
                            ctx,
                        )
                        confirmed = "\n".join(
                            [segment["text"] for segment in ctx.confirmed_segments]
                        )
                        full = confirmed + "\n" + current if confirmed else current

                        if full != ctx.last_partial_text:
                            ctx.last_partial_text = full
                            results.append(
                                {
                                    "text": full,
                                    "current_segment_text": current,
                                    "language": ctx.streaming_state.last_language,
                                    "chunk_id": ctx.streaming_state.chunk_count,
                                    "is_partial": True,
                                    "segment_index": ctx.segment_index,
                                }
                            )

                        if speech_ended:
                            break

                    max_samples = int(settings.QWEN_STREAM_SEGMENT_MAX_SEC * _SAMPLE_RATE)

                    # 先推 partial，再落定段（保证客户端看到 result → segment_end 的顺序）
                    if results:
                        await websocket.send_json(
                            {
                                "type": "result",
                                "task_id": task_id,
                                "results": results,
                                "segment_index": ctx.segment_index,
                                "confirmed_segments_count": len(ctx.confirmed_segments),
                            }
                        )
                        ctx.state = ConnectionState.STREAMING

                    if speech_ended or ctx.total_samples >= max_samples:
                        reason = "silence" if speech_ended else "max_duration"
                        await self._finalize_segment(websocket, ctx, task_id, reason)
                        # VAD 在 chunk 边界落定：buffer 中剩余音频属于下一段
                        ctx.segment_audio = ctx.audio_buffer.copy()
                        ctx.total_samples = len(ctx.audio_buffer)

        except WebSocketDisconnect:
            logger.info("[%s] WebSocket disconnected", task_id)
        except Exception as exc:
            logger.error("[%s] Connection error: %s", task_id, exc)
            await self._send_error(websocket, str(exc), task_id)
        finally:
            stream_handle = getattr(
                getattr(ctx, "streaming_state", None), "internal_state", None
            )
            close_stream = getattr(stream_handle, "close", None)
            if callable(close_stream):
                close_stream()

            if ctx.engine_lease is not None:
                await ctx.engine_lease.close()
            logger.info("[%s] Connection closed", task_id)

    async def _stop(
        self,
        websocket: WebSocket,
        ctx: ConnectionContext,
        task_id: str,
    ) -> None:
        try:
            engine = await self._ensure_engine(ctx)

            if len(ctx.audio_buffer) > 0:
                ctx.streaming_state = await run_sync(
                    engine.streaming_transcribe,
                    ctx.audio_buffer,
                    ctx.streaming_state,
                )

            # 末段走与静音边界相同的落定流程（finish + 可选 two-pass）
            await self._finalize_segment(websocket, ctx, task_id, "final")

            final = (
                ctx.confirmed_segments[-1]["text"]
                if ctx.confirmed_segments
                else ""
            )
            all_texts = [
                segment["text"]
                for segment in ctx.confirmed_segments
                if segment["text"].strip()
            ]
            full_text = "\n".join(all_texts)

            await websocket.send_json(
                {
                    "type": "final",
                    "task_id": task_id,
                    "result": {
                        "text": final,
                        "full_text": full_text,
                        "language": ctx.streaming_state.last_language,
                        "total_chunks": ctx.streaming_state.chunk_count,
                        "total_segments": len(ctx.confirmed_segments),
                        "segments": ctx.confirmed_segments,
                    },
                }
            )

            logger.info(
                "[%s] Recognition completed, segments=%s",
                task_id,
                len(ctx.confirmed_segments),
            )

        except Exception as exc:
            logger.error("[%s] Stop failed: %s", task_id, exc)
            await self._send_error(websocket, f"结束识别失败: {exc}", task_id)
