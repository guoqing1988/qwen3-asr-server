# -*- coding: utf-8 -*-
"""长静音断句：段间静音超过 MAX_SILENCE_GAP_SEC 时必须切成独立段。

背景：39s 音频里 14s 语音 + 25s 静音，整段送 ASR 会被判空、语音一起丢失。
"""

import unittest
from unittest.mock import patch

import numpy as np
import soundfile as sf
import tempfile

from app.core.config import settings
from app.utils.audio_splitter import AudioSplitter


def _write_wav(path: str, seconds: float) -> None:
    sf.write(path, np.zeros(int(seconds * 16000), dtype=np.float32), 16000)


class SilenceGapTest(unittest.TestCase):
    def setUp(self) -> None:
        self._tmp = tempfile.TemporaryDirectory()
        self.path = f"{self._tmp.name}/silence.wav"
        _write_wav(self.path, 39.0)
        self.splitter = AudioSplitter()

    def tearDown(self) -> None:
        self._tmp.cleanup()

    def test_find_long_gap_detects_trailing_silence(self) -> None:
        gap = self.splitter.find_long_gap([(50, 14400)], 39000)
        self.assertEqual(gap, (14400, 39000))

    def test_find_long_gap_ignores_short_pauses(self) -> None:
        spans = [(0, 5000), (6000, 12000), (14000, 20000)]
        self.assertIsNone(self.splitter.find_long_gap(spans, 20000))

    def test_find_long_gap_detects_middle_silence(self) -> None:
        gap = self.splitter.find_long_gap([(0, 5000), (30000, 36000)], 39000)
        self.assertEqual(gap, (5000, 30000))

    def test_short_audio_with_long_silence_is_split(self) -> None:
        # 39s < MAX_SEGMENT_SEC(60s)，但存在 24s 静音 → 必须切段
        with patch.object(settings, "MAX_SEGMENT_SEC", 60), patch.object(
            settings, "MAX_SILENCE_GAP_SEC", 5
        ):
            splitter = AudioSplitter()
            segments = splitter.split_audio_file(
                self.path, speech_segments=[(50, 14400)]
            )
        self.assertGreater(len(segments), 1)
        # 覆盖完整时间轴，语音区间保持在独立段内
        self.assertEqual(segments[0].start_ms, 0)
        self.assertEqual(segments[-1].end_ms, 39000)
        voice_segments = [s for s in segments if s.start_ms < 14400]
        self.assertTrue(all(s.duration_ms <= 60000 for s in segments))

    def test_short_audio_without_long_silence_stays_single_segment(self) -> None:
        with patch.object(settings, "MAX_SEGMENT_SEC", 60), patch.object(
            settings, "MAX_SILENCE_GAP_SEC", 5
        ):
            splitter = AudioSplitter()
            segments = splitter.split_audio_file(
                self.path, speech_segments=[(0, 39000)]
            )
        self.assertEqual(len(segments), 1)

    def test_sparse_speech_is_split_even_without_long_gap(self) -> None:
        # 39s 内只有零星活动（覆盖率 37%）、最大间隙 4.3s：仍须切分
        # 8 段共 9.6s 活动，间隔 4.2s（< 阈值 5s），覆盖率 25%
        spans = [(i * 5400, i * 5400 + 1200) for i in range(8)]
        with patch.object(settings, "MAX_SEGMENT_SEC", 60), patch.object(
            settings, "MAX_SILENCE_GAP_SEC", 5
        ):
            splitter = AudioSplitter()
            self.assertIsNone(splitter.find_long_gap(spans, 39000))
            self.assertLess(splitter.speech_coverage(spans, 39000), 0.5)
            segments = splitter.split_audio_file(self.path, speech_segments=spans)
        self.assertGreater(len(segments), 1)

    def test_dense_speech_stays_single_segment(self) -> None:
        spans = [(0, 18000), (18200, 38000)]  # 覆盖率 >95%
        splitter = AudioSplitter()
        self.assertGreaterEqual(splitter.speech_coverage(spans, 39000), 0.5)
        with patch.object(settings, "MAX_SEGMENT_SEC", 60):
            segments = splitter.split_audio_file(self.path, speech_segments=spans)
        self.assertEqual(len(segments), 1)

    def test_empty_speech_segments_fall_back_to_fixed_split(self) -> None:
        with patch.object(settings, "MAX_SEGMENT_SEC", 10):
            splitter = AudioSplitter()
            segments = splitter.split_audio_file(self.path, speech_segments=[])
        self.assertGreater(len(segments), 1)
        self.assertEqual(segments[0].start_ms, 0)
        self.assertEqual(segments[-1].end_ms, 39000)


if __name__ == "__main__":
    unittest.main()
