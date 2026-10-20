# -*- coding: utf-8 -*-
"""R2T2 流式：VAD 静音边界落定 / 无重叠拼接 / 动态生成长度测试。

背景（2026-10-18）：流式切段由「固定 12s 强切 + 1s 重叠 + 字符去重」改为
官方推荐做法——FireRedVAD 句尾静音触发落定、段间无重叠、文本自然拼接；
每轮生成上限按 chunk 时长动态计算（配置值仅作上限）。
"""

import asyncio
import unittest
from typing import Callable

import numpy as np

from app.core.config import settings
from app.services.asr.qwen3_vllm import (
    Qwen3VLLMBackend,
    VLLMRealtimeState,
    _build_stream_prefix,
    _normalize_punct_by_context,
    _strip_cjk_spaces,
)
from app.services.qwen3_websocket_asr import ConnectionContext, Qwen3ASRService


async def _run_inline(
    func: Callable[..., object],
    *args: object,
    **kwargs: object,
) -> object:
    return func(*args, **kwargs)


class _FakeStreamingState:
    def __init__(self, text: str = "段一文本") -> None:
        self.last_text = text
        self.last_language = "Chinese"
        self.chunk_count = 1


class _FakeVAD:
    def __init__(self) -> None:
        self.reset_count = 0
        self.detect_results: list[bool] = []
        self.detected: list[int] = []

    def reset(self) -> None:
        self.reset_count += 1

    def detect_speech_end(self, pcm: np.ndarray) -> bool:
        self.detected.append(len(pcm))
        return self.detect_results.pop(0) if self.detect_results else False


class _FakeEngine:
    supports_realtime = True
    device = "cuda"

    def __init__(self) -> None:
        self.refined_text = "段一文本（离线校正）"
        self.transcribe_calls: list[np.ndarray] = []
        self.init_calls = 0

    def init_streaming_state(self, **kwargs: object) -> _FakeStreamingState:
        self.init_calls += 1
        return _FakeStreamingState("")

    def streaming_transcribe(
        self, audio: np.ndarray, state: _FakeStreamingState
    ) -> _FakeStreamingState:
        state.chunk_count += 1
        return state

    def finish_streaming_transcribe(
        self, state: _FakeStreamingState
    ) -> _FakeStreamingState:
        return state

    def transcribe_array(
        self,
        audio: np.ndarray,
        context: str = "",
        language: str | None = None,
        enable_itn: bool = False,
    ) -> str:
        self.transcribe_calls.append(audio)
        return self.refined_text


class _FakeWebSocket:
    def __init__(self) -> None:
        self.events: list[dict] = []

    async def send_json(self, event: dict) -> None:
        self.events.append(event)


class StreamVadDefaultsTest(unittest.TestCase):
    def test_official_default_stream_params(self) -> None:
        self.assertAlmostEqual(settings.QWEN_STREAM_CHUNK_SEC, 0.16)
        self.assertAlmostEqual(settings.QWEN_STREAM_LOOKAHEAD_SEC, 0.16)
        self.assertEqual(settings.QWEN_STREAM_UNFIXED_CHUNK_NUM, 0)
        self.assertEqual(settings.QWEN_STREAM_UNFIXED_TOKEN_NUM, 1)
        self.assertGreaterEqual(settings.QWEN_STREAM_SEGMENT_MAX_SEC, 60.0)

    def test_dynamic_max_tokens_scales_with_chunk(self) -> None:
        backend = object.__new__(Qwen3VLLMBackend)
        state = VLLMRealtimeState(
            prompt_raw="",
            language="Chinese",
            chunk_size_sec=0.16,
            unfixed_chunk_num=0,
            unfixed_token_num=1,
            max_new_tokens=2048,
        )
        # 160ms(2560) + 160ms lookahead → 5120/1280*2 = 8，另加语言头部余量 12
        self.assertEqual(backend._stream_max_new_tokens(state, 5120), 20)
        # 配置值小于动态值时，配置值生效（作为硬上限）
        state.max_new_tokens = 2
        self.assertEqual(backend._stream_max_new_tokens(state, 5120), 2)
        # <=0 表示不设上限
        state.max_new_tokens = 0
        self.assertEqual(backend._stream_max_new_tokens(state, 5120), 20)

    def test_prefix_rolls_back_and_trims_pipe(self) -> None:
        class _Tokenizer:
            def encode(self, text: str, add_special_tokens: bool = False) -> list[str]:
                return list(text)

            def decode(
                self, ids: list[str], skip_special_tokens: bool = False
            ) -> str:
                return "".join(ids)

        self.assertEqual(_build_stream_prefix(_Tokenizer(), "abc|", 1), "ab")
        self.assertEqual(_build_stream_prefix(_Tokenizer(), "", 1), "")

    def test_punctuation_and_cjk_space_normalization(self) -> None:
        self.assertEqual(_normalize_punct_by_context("你好,world."), "你好，world.")
        self.assertEqual(_strip_cjk_spaces("你 好 世 界"), "你好世界")
        self.assertEqual(_strip_cjk_spaces("你 好 world"), "你好 world")


class SegmentFinalizeTest(unittest.IsolatedAsyncioTestCase):
    async def asyncSetUp(self) -> None:
        asyncio.get_running_loop().slow_callback_duration = 1.0

    async def _finalize(self, *, final_pass: bool) -> tuple:
        service = Qwen3ASRService()
        engine = _FakeEngine()
        vad = _FakeVAD()
        ctx = ConnectionContext(
            params={
                "context": "某某公司",
                "language": "Chinese",
                "enable_inverse_text_normalization": True,
            },
            engine=engine,
            streaming_state=_FakeStreamingState("段一文本"),
            vad=vad,
            segment_audio=np.zeros(16000, dtype=np.float32),
        )
        websocket = _FakeWebSocket()

        from unittest.mock import patch

        with (
            patch(
                "app.services.qwen3_websocket_asr.run_sync",
                new=_run_inline,
            ),
            patch.object(settings, "QWEN_STREAM_FINAL_PASS", final_pass),
        ):
            await service._finalize_segment(websocket, ctx, "task", "silence")
        return service, ctx, engine, vad, websocket

    async def test_finalize_uses_two_pass_text(self) -> None:
        _, ctx, engine, vad, websocket = await self._finalize(final_pass=True)
        self.assertEqual(len(engine.transcribe_calls), 1)
        self.assertEqual(ctx.confirmed_segments[0]["text"], "段一文本（离线校正）")
        self.assertEqual(ctx.confirmed_segments[0]["reason"], "silence")
        self.assertEqual(vad.reset_count, 1)
        self.assertEqual(websocket.events[0]["type"], "segment_end")
        self.assertEqual(
            websocket.events[0]["result"]["segment_text"], "段一文本（离线校正）"
        )
        self.assertEqual(websocket.events[1]["type"], "segment_start")

    async def test_finalize_keeps_streaming_text_without_final_pass(self) -> None:
        _, ctx, engine, _, _ = await self._finalize(final_pass=False)
        self.assertEqual(engine.transcribe_calls, [])
        self.assertEqual(ctx.confirmed_segments[0]["text"], "段一文本")

    async def test_finalize_does_not_clear_pending_audio(self) -> None:
        """VAD 在 chunk 边界落定，buffer 中剩余音频属于下一段，不能清空。"""
        service = Qwen3ASRService()
        remaining = np.ones(800, dtype=np.float32)
        ctx = ConnectionContext(
            params={"language": "Chinese"},
            engine=_FakeEngine(),
            streaming_state=_FakeStreamingState("段一文本"),
            segment_audio=np.zeros(16000, dtype=np.float32),
            audio_buffer=remaining,
        )
        websocket = _FakeWebSocket()

        from unittest.mock import patch

        with (
            patch("app.services.qwen3_websocket_asr.run_sync", new=_run_inline),
            patch.object(settings, "QWEN_STREAM_FINAL_PASS", False),
        ):
            await service._finalize_segment(websocket, ctx, "task", "silence")

        self.assertEqual(ctx.audio_buffer.tolist(), remaining.tolist())
        self.assertEqual(ctx.segment_audio.size, 0)


if __name__ == "__main__":
    unittest.main()
