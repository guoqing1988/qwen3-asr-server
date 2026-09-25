# -*- coding: utf-8 -*-
"""Qwen3 流式滑动窗口测试。

背景（2026-09-25）：实时任务可能持续数小时甚至一天，不能把全部音频和文本一直
累积重算（实测 52.5s 连续语音滞后 6.2s）。改为「固定段窗口 + 静音/时长落定 +
段间重叠去重」：

- 单段超过 `QWEN_STREAM_SEGMENT_MAX_SEC` 秒，或累计静音超过
  `QWEN_STREAM_SILENCE_SEC` 秒，即落定当前段并开新段 —— 保证每轮解码窗口恒定，
  不随会话时长增长；
- 新段开头预置上一段尾部 `QWEN_STREAM_OVERLAP_SEC` 秒音频（重叠），
  识别结果做边界去重，避免强制切段处丢词/断句。
"""

import unittest

import numpy as np

from app.core.config import settings
from app.services.asr.qwen3_engine import Qwen3StreamingState
from app.services.asr.qwen3_vllm import VLLMRealtimeState
from app.services.qwen3_websocket_asr import (
    _SAMPLE_RATE,
    ConnectionContext,
    Qwen3ASRService,
    _strip_overlap,
)


class StripOverlapTest(unittest.TestCase):
    def test_english_tail_overlap(self) -> None:
        self.assertEqual(
            _strip_overlap("So good to see you.", "you. Good to see you too."),
            "Good to see you too.",
        )

    def test_chinese_tail_overlap(self) -> None:
        self.assertEqual(_strip_overlap("我们去看电影。", "电影。然后回家。"), "然后回家。")

    def test_no_overlap_keeps_text(self) -> None:
        self.assertEqual(_strip_overlap("abc", "xyz"), "xyz")

    def test_empty_inputs(self) -> None:
        self.assertEqual(_strip_overlap("", "abc"), "abc")
        self.assertEqual(_strip_overlap("abc", ""), "")

    def test_case_insensitive_overlap(self) -> None:
        """重叠音频处模型可能重排大小写（Really vs really），仍应去重。"""
        self.assertEqual(
            _strip_overlap(
                "...really hard work",
                "Really hard work getting our technology",
            ),
            "getting our technology",
        )

    def test_overlap_below_min_chars_is_kept(self) -> None:
        """低于 min_chars 的巧合重叠不裁剪，避免误删真实内容。"""
        self.assertEqual(_strip_overlap("abc", "cdef"), "cdef")


class WindowTruncateTest(unittest.TestCase):
    def setUp(self) -> None:
        self.service = Qwen3ASRService()
        self.max_samples = int(settings.QWEN_STREAM_SEGMENT_MAX_SEC * _SAMPLE_RATE)
        self.silence_samples = int(settings.QWEN_STREAM_SILENCE_SEC * _SAMPLE_RATE)

    def test_truncate_on_max_duration(self) -> None:
        ctx = ConnectionContext()
        ctx.total_samples = self.max_samples
        self.assertTrue(self.service._need_truncate(ctx))

    def test_truncate_on_silence(self) -> None:
        ctx = ConnectionContext()
        ctx.silence_samples = self.silence_samples
        self.assertTrue(self.service._need_truncate(ctx))

    def test_no_truncate_below_thresholds(self) -> None:
        ctx = ConnectionContext()
        ctx.total_samples = self.max_samples - 1
        ctx.silence_samples = self.silence_samples - 1
        self.assertFalse(self.service._need_truncate(ctx))


class SegmentTailTest(unittest.TestCase):
    def _make_state(self, samples: int) -> Qwen3StreamingState:
        inner = VLLMRealtimeState(
            prompt_raw="",
            language="English",
            chunk_size_sec=0.16,
            unfixed_chunk_num=0,
            unfixed_token_num=1,
            max_new_tokens=2048,
        )
        inner.audio_accum = np.arange(samples, dtype=np.float32)
        return Qwen3StreamingState(internal_state=inner)

    def test_returns_tail_slice(self) -> None:
        service = Qwen3ASRService()
        ctx = ConnectionContext()
        ctx.streaming_state = self._make_state(_SAMPLE_RATE * 5)
        tail = service._extract_segment_tail(ctx)
        expected_len = int(settings.QWEN_STREAM_OVERLAP_SEC * _SAMPLE_RATE)
        self.assertIsNotNone(tail)
        assert tail is not None
        self.assertEqual(len(tail), expected_len)
        # 取的应是尾部（最后 expected_len 个样本）
        self.assertEqual(tail[-1], np.float32(_SAMPLE_RATE * 5 - 1))

    def test_returns_none_without_vllm_state(self) -> None:
        """Rust 流状态没有 audio_accum，应安全返回 None（不预置重叠）。"""
        service = Qwen3ASRService()
        ctx = ConnectionContext()
        ctx.streaming_state = None
        self.assertIsNone(service._extract_segment_tail(ctx))


if __name__ == "__main__":
    unittest.main()
