# -*- coding: utf-8 -*-
"""实时说话人标签（SpeakerStream）单元测试。

用假 processor/推理注入，验证帧累积、待标注话的覆盖条件、标签聚合与超时；
真实 Nemotron 推理由集成测试覆盖，这里不依赖 GPU/模型。
"""

import asyncio
from collections import deque

import numpy as np

from app.services.asr import realtime_speakers as module
from app.services.asr.realtime_speakers import SpeakerStream

SR = 16000


class _FeatureExtractor:
    win_length = 400


class _FakeProcessor:
    """chunk 边界：首块 200ms，之后每块 100ms（与 Nemotron 流式语义一致）。"""

    feature_extractor = _FeatureExtractor()
    num_samples_first_audio_chunk = 3200
    num_samples_per_audio_chunk = 1600

    def audio_chunk_start(self, frames: int) -> int:
        """已消费 frames 帧后，下一个 chunk 的起始采样点（单调）。"""
        first_frames = self.num_samples_first_audio_chunk // 160
        if frames < first_frames:
            return 0
        return self.num_samples_first_audio_chunk + (frames - first_frames) * 160


def _make_stream(columns: list[int]) -> SpeakerStream:
    """构造一个绕过 model/processor 的流，按 columns 依次产出说话人帧。"""
    stream = SpeakerStream.__new__(SpeakerStream)
    stream.model = object()  # 非 None：标签可用
    stream.processor = _FakeProcessor()
    stream.audio = np.empty(0, dtype=np.float32)
    stream.offset = 0
    stream.samples = 0
    stream.frames = 0
    stream.activity = np.zeros((4000, 8), dtype=bool)
    stream.cache = None
    stream.ended = False
    stream.finished = False
    stream.pending = deque()
    stream.labels = {}
    stream.resolved = set()
    stream.names = {}
    stream.flushed = False
    calls = {"n": 0}

    def _infer(audio, first, last):
        column = columns[min(calls["n"], len(columns) - 1)]
        calls["n"] += 1
        frames = max(1, len(audio) // 160)
        return _activity(frames, column), None

    stream._infer = _infer
    return stream


def _activity(frames: int, column: int) -> np.ndarray:
    active = np.zeros((frames, 8), dtype=bool)
    active[:, column] = True
    return active


def _pcm(seconds: float) -> bytes:
    return np.zeros(int(seconds * SR), dtype="<i2").tobytes()


def test_model_missing_marks_finished(monkeypatch):
    monkeypatch.setattr(
        module, "get_speaker_diarizer", lambda: (_ for _ in ()).throw(RuntimeError("boom"))
    )
    stream = SpeakerStream()
    assert stream.finished is True
    stream.utterance(0, 0, 500)
    assert stream.ready() == [{"utterance": 0, "speaker": None}]


def test_feed_accumulates_float_audio():
    stream = _make_stream([0])
    stream.feed(_pcm(0.5))
    assert stream.samples == int(0.5 * SR)
    stream.end()
    stream.feed(_pcm(0.5))  # ended 之后仍可喂入，直到 finished
    assert stream.samples == SR


def test_advance_only_returns_covered_utterances():
    stream = _make_stream([0])
    stream.feed(_pcm(1.0))
    stream.end()
    stream.utterance(0, 0, 500)
    stream.utterance(1, 600, 900)

    events = asyncio.run(stream.advance())

    assert [event["utterance"] for event in events] == [0, 1]
    assert events[0]["speaker"] == "说话人1"


def test_utterance_not_returned_before_audio_arrives():
    stream = _make_stream([0])
    stream.feed(_pcm(0.3))
    stream.utterance(0, 0, 4000)  # 登记 4s，但音频只喂了 0.3s 且尚未 end()

    assert asyncio.run(stream.advance()) == []


def test_label_times_out_without_audio():
    stream = _make_stream([0])
    stream.utterance(3, 0, 400)

    assert asyncio.run(stream.label(3, 0.1)) is None
    assert 3 not in stream.labels


def test_labels_are_stable_per_speaker_column():
    stream = _make_stream([0])
    # 帧级概率已经就位（帧 = 10ms）：前 0.5s 说话人列 0，随后 0.5s 列 1
    stream.frames = 100
    stream.activity[0:50, 0] = True
    stream.activity[50:100, 1] = True

    assert stream._label(0, 500) == "说话人1"
    assert stream._label(500, 1000) == "说话人2"
    # 同一列始终映射到同一标签
    assert stream._label(0, 500) == "说话人1"


def test_no_activity_yields_none_label():
    stream = _make_stream([0])
    stream.frames = 10
    assert stream._label(0, 100) is None


def test_label_resolves_via_advance():
    stream = _make_stream([0])
    stream.feed(_pcm(0.8))
    stream.end()
    stream.utterance(0, 0, 500)

    assert asyncio.run(stream.label(0, 1.0)) == "说话人1"


def test_flush_labels_tail_audio_without_more_input():
    stream = _make_stream([0])
    stream.feed(_pcm(0.5))
    stream.utterance(0, 0, 600)  # 超出已推理帧（右上下文不足），不 flush 等不到

    assert asyncio.run(stream.label(0, 0.1)) == "说话人1"
    assert stream.flushed is True


def test_names_are_inherited_when_stream_restarts():
    stream = _make_stream([0])
    stream.frames = 100
    stream.activity[0:100, 3] = True
    assert stream._label(0, 1000) == "说话人1"

    restarted = SpeakerStream.__new__(SpeakerStream)
    SpeakerStream.__init__(restarted, stream.names)
    assert restarted.names == {3: "说话人1"}


def test_close_clears_state():
    stream = _make_stream([0])
    stream.feed(_pcm(0.5))
    stream.utterance(0, 0, 300)
    stream.close()

    assert stream.finished is True
    assert stream.samples == 16000 or stream.audio.size == 0
    assert not stream.pending
