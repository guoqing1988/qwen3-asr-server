# -*- coding: utf-8 -*-
"""Unit tests for the llama.cpp Q8 backend (no server / GPU required)."""

import numpy as np
import pytest

from app.core.config import settings
from app.services.asr.qwen3_engine import Qwen3ASREngine
from app.services.asr.qwen3_llamacpp import (
    LlamaCppQ8Backend,
    Q8StreamingState,
    _join_segments,
    _parse_output,
    _split_audio,
)


def _make_backend(monkeypatch) -> LlamaCppQ8Backend:
    monkeypatch.setattr(
        "app.services.asr.qwen3_llamacpp._load_audio",
        lambda path: np.zeros(16000, dtype=np.float32),
    )
    backend = object.__new__(LlamaCppQ8Backend)
    backend._max_segment_sec = 25.0
    return backend


class TestParseOutput:
    def test_detected_language(self):
        assert _parse_output("language English<asr_text>hello world", None) == (
            "English",
            "hello world",
        )

    def test_forced_language_wins(self):
        assert _parse_output("language Chinese<asr_text>你好", "English") == (
            "English",
            "你好",
        )

    def test_plain_text(self):
        assert _parse_output("just text", None) == ("", "just text")

    def test_language_without_tag(self):
        assert _parse_output("language Chinese\n你好", None) == ("Chinese", "你好")

    def test_none_language(self):
        assert _parse_output("language None<asr_text>hi", None) == ("", "hi")

    def test_truncated_tag_first_chunk(self):
        # 首轮 max_tokens 截断，只吐出未闭合标签
        assert _parse_output("language None<asr_text", None) == ("", "")


class TestAudioHelpers:
    def test_split_respects_limit(self):
        audio = np.zeros(16000 * 60, dtype=np.float32)
        pieces = _split_audio(audio, 25.0)
        assert len(pieces) == 3
        assert max(len(piece) for piece in pieces) <= 16000 * 25
        assert sum(len(piece) for piece in pieces) == len(audio)

    def test_short_audio_not_split(self):
        audio = np.zeros(16000 * 5, dtype=np.float32)
        assert len(_split_audio(audio, 25.0)) == 1

    def test_join_latin_uses_spaces(self):
        assert _join_segments(["hello", "world"], "English") == "hello world"

    def test_join_chinese_has_no_spaces(self):
        assert _join_segments(["你好", "世界"], "Chinese") == "你好世界"



class TestTranscribeAudio:
    def test_long_audio_is_chunked_and_joined(self, monkeypatch):
        backend = _make_backend(monkeypatch)
        calls: list[int] = []

        def fake_chat(audio, *, context="", language=None, prefix="", max_tokens=512):
            calls.append(len(audio))
            return f"language Chinese<asr_text>第{len(calls)}段。"

        monkeypatch.setattr(backend, "_chat", fake_chat)
        text = backend._transcribe_audio(
            np.zeros(16000 * 60, dtype=np.float32), language="zh"
        )
        assert calls == [16000 * 25, 16000 * 25, 16000 * 10]
        assert text == "第1段。第2段。第3段。"

    def test_hotword_context_forwarded(self, monkeypatch):
        backend = _make_backend(monkeypatch)
        seen: dict[str, str] = {}

        def fake_chat(audio, *, context="", language=None, prefix="", max_tokens=512):
            seen["context"] = context
            seen["prefix"] = prefix
            return "language English<asr_text>gpt"

        monkeypatch.setattr(backend, "_chat", fake_chat)
        text = backend._transcribe_audio(
            np.zeros(16000, dtype=np.float32), context="GPT", language="en"
        )
        assert seen["context"] == "GPT"
        assert seen["prefix"] == "language English<asr_text>"
        assert text == "gpt"


class TestStreaming:
    def test_feed_stream_redecodes_at_interval(self, monkeypatch):
        backend = _make_backend(monkeypatch)
        texts = iter(["你好", "你好世界"])
        calls: list[int] = []

        def fake_transcribe(audio, **kwargs):
            calls.append(len(audio))
            return next(texts)

        monkeypatch.setattr(backend, "_transcribe_audio", fake_transcribe)
        state = backend.init_streaming_state(chunk_size_sec=0.16, lookahead_sec=0.16)
        # 首轮阈值 0.32s：0.2s 不解码，0.4s 解码
        state = backend.feed_stream(np.zeros(3200, dtype=np.float32), state)
        assert calls == []
        state = backend.feed_stream(np.zeros(3200, dtype=np.float32), state)
        assert calls == [6400]
        assert state.text == "你好"
        # 之后每 1s 解码一次
        for _ in range(5):
            state = backend.feed_stream(np.zeros(3200, dtype=np.float32), state)
        assert len(calls) == 2
        assert state.text == "你好世界"

    def test_finish_stream_decodes_full_audio(self, monkeypatch):
        backend = _make_backend(monkeypatch)
        backend._chat = lambda *a, **k: "language Chinese<asr_text>完整句子。"
        state = backend.init_streaming_state(chunk_size_sec=0.16, lookahead_sec=0.0)
        state.buffer = np.zeros(1600, dtype=np.float32)
        state = backend.finish_stream(state)
        assert state.text == "完整句子。"
        assert len(state.audio) == 1600
        assert len(state.buffer) == 0


class TestBackendSelection:
    def test_llamacpp_supports_realtime(self):
        engine = object.__new__(Qwen3ASREngine)
        engine._backend = "llamacpp"
        assert engine.supports_realtime is True

    def test_llamacpp_selected_by_config(self, monkeypatch):
        monkeypatch.setattr(settings, "ASR_ENGINE", "llamacpp")
        engine = object.__new__(Qwen3ASREngine)
        engine._device = "cuda:0"
        assert engine._select_backend() == "llamacpp"

    def test_vllm_still_default(self, monkeypatch):
        monkeypatch.setattr(settings, "ASR_ENGINE", "vllm")
        monkeypatch.setattr("app.services.asr.qwen3_engine.is_vllm_available", lambda: True)
        engine = object.__new__(Qwen3ASREngine)
        engine._device = "cuda:0"
        assert engine._select_backend() == "vllm"

    def test_llamacpp_requires_no_vllm_import(self, monkeypatch):
        monkeypatch.setattr(settings, "ASR_ENGINE", "llamacpp")
        monkeypatch.setattr("app.services.asr.qwen3_engine.is_vllm_available", lambda: False)
        engine = object.__new__(Qwen3ASREngine)
        engine._device = "cpu"
        assert engine._select_backend() == "llamacpp"


def test_streaming_state_defaults():
    state = Q8StreamingState()
    assert state.chunk_size_sec == pytest.approx(0.16)
    assert state.unfixed_token_num == 1
    assert state.buffer.size == 0
