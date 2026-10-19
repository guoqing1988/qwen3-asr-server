# -*- coding: utf-8 -*-
"""Unit tests for the OpenAI Realtime transcription gateway (no GPU/engine)."""

from __future__ import annotations

import asyncio
import base64

import numpy as np
import pytest

from app.core.config import settings
from app.services.asr import openai_realtime as gateway
from app.services.asr.openai_realtime import (
    OpenAIRealtimeSession,
    RealtimeTranscriptionConfig,
    _Resampler,
)


class FakeWebSocket:
    def __init__(self) -> None:
        self.sent: list[dict] = []
        self.scope = {"subprotocols": []}
        self.accepted_subprotocol: str | None = None

    async def accept(self, subprotocol: str | None = None) -> None:
        self.accepted_subprotocol = subprotocol

    async def send_json(self, payload: dict) -> None:
        self.sent.append(payload)


class FakeState:
    def __init__(self) -> None:
        self.last_text = ""
        self.last_language = "Chinese"
        self.chunk_count = 0
        self.internal_state = None


class FakeEngine:
    device = "cuda:0"

    @property
    def supports_realtime(self) -> bool:
        return True

    def init_streaming_state(self, **kwargs) -> FakeState:
        return FakeState()

    def streaming_transcribe(self, chunk, state: FakeState) -> FakeState:
        state.chunk_count += 1
        state.last_text = "你好" * state.chunk_count
        return state

    def finish_streaming_transcribe(self, state: FakeState) -> FakeState:
        state.last_text = "你好世界。"
        return state

    def transcribe_array(self, *args, **kwargs) -> str:
        return "你好世界。"


class FakeVAD:
    def __init__(self, end: bool = False) -> None:
        self.end = end
        self.resets = 0

    def detect_speech_end(self, chunk) -> bool:
        return self.end

    def reset(self) -> None:
        self.resets += 1


def _pcm(seconds: float, amplitude: float = 0.2) -> str:
    samples = (np.ones(int(seconds * 16000), dtype=np.float32) * amplitude * 32768).astype("<i2")
    return base64.b64encode(samples.tobytes()).decode()


def _session(monkeypatch, vad_end: bool = False) -> tuple[OpenAIRealtimeSession, FakeWebSocket, FakeEngine]:
    ws = FakeWebSocket()
    session = OpenAIRealtimeSession(ws, "test")
    engine = FakeEngine()

    async def fake_ensure_engine() -> FakeEngine:
        session._engine = engine
        session._vad = FakeVAD(end=vad_end)
        session._state = engine.init_streaming_state()
        return engine

    monkeypatch.setattr(session, "_ensure_engine", fake_ensure_engine)
    monkeypatch.setattr(settings, "QWEN_STREAM_FINAL_PASS", False)
    return session, ws, engine


def _types(ws: FakeWebSocket) -> list[str]:
    return [event["type"] for event in ws.sent]


class TestResampler:
    def test_16k_passthrough(self):
        data = np.ones(1600, dtype=np.float32)
        assert np.array_equal(_Resampler(16000).push(data), data)

    def test_24k_resamples(self):
        data = np.ones(2400, dtype=np.float32)
        resampler = _Resampler(24000)
        first = resampler.push(data)
        # 首块包含滤波器延迟，输出略少于 2/3
        assert 1300 < first.size <= 1600
        second = resampler.push(data)
        ratio = (first.size + second.size) / 4800
        assert 0.55 < ratio < 0.70


class TestConfig:
    def test_context_merges_prompt_and_keywords(self, monkeypatch):
        monkeypatch.setattr(settings, "ASR_DEFAULT_HOTWORDS", "Bloomberg")
        config = RealtimeTranscriptionConfig(prompt="Transcribe", keywords=["gpt", "Coaching"])
        context = config.context()
        assert "Bloomberg" in context and "Transcribe" in context
        assert "gpt" in context and "Coaching" in context


class TestSessionUpdate:
    def test_unsupported_rate(self, monkeypatch):
        session, ws, _ = _session(monkeypatch)
        asyncio.run(
            session._dispatch(
                '{"type":"session.update","session":{"audio":{"input":{"format":{"rate":44100}}}}}'
            )
        )
        assert ws.sent[-1]["type"] == "error"
        assert ws.sent[-1]["error"]["code"] == "invalid_rate"

    def test_keywords_validation(self, monkeypatch):
        session, ws, _ = _session(monkeypatch)
        asyncio.run(
            session._dispatch(
                '{"type":"session.update","session":{"audio":{"input":'
                '{"transcription":{"keywords":["bad<term"]}}}}}'
            )
        )
        assert ws.sent[-1]["error"]["code"] == "invalid_keywords"

    def test_server_vad_and_24k(self, monkeypatch):
        session, ws, _ = _session(monkeypatch)
        asyncio.run(
            session._dispatch(
                '{"type":"session.update","session":{"audio":{"input":'
                '{"format":{"type":"audio/pcm","rate":24000},'
                '"transcription":{"prompt":"hi","keywords":["gpt"]},'
                '"turn_detection":{"type":"server_vad"}}}}}'
            )
        )
        assert ws.sent[-1]["type"] == "session.updated"
        assert session._config.turn_detection == "server_vad"
        assert session._config.input_rate == 24000
        assert isinstance(session._resampler, _Resampler)


class TestAppend:
    def test_delta_events_are_incremental(self, monkeypatch):
        session, ws, _ = _session(monkeypatch)
        asyncio.run(
            session._dispatch(
                f'{{"type":"input_audio_buffer.append","audio":"{_pcm(0.32)}"}}'
            )
        )
        deltas = [
            event["delta"]
            for event in ws.sent
            if event["type"].endswith("transcription.delta")
        ]
        assert deltas, "expected delta events"
        assert "".join(deltas) == "你好你好"

    def test_invalid_base64(self, monkeypatch):
        session, ws, _ = _session(monkeypatch)
        asyncio.run(
            session._dispatch('{"type":"input_audio_buffer.append","audio":"@@@"}')
        )
        assert ws.sent[-1]["error"]["code"] == "invalid_audio"

    def test_odd_pcm_length(self, monkeypatch):
        session, ws, _ = _session(monkeypatch)
        encoded = base64.b64encode(b"\x01\x02\x03").decode()
        asyncio.run(
            session._dispatch(
                f'{{"type":"input_audio_buffer.append","audio":"{encoded}"}}'
            )
        )
        assert ws.sent[-1]["error"]["code"] == "invalid_audio"

    def test_unknown_event(self, monkeypatch):
        session, ws, _ = _session(monkeypatch)
        asyncio.run(session._dispatch('{"type":"response.create"}'))
        assert ws.sent[-1]["error"]["code"] == "unsupported_event"


class TestCommitAndClear:
    def test_commit_emits_committed_then_completed(self, monkeypatch):
        session, ws, _ = _session(monkeypatch)
        asyncio.run(
            session._dispatch(
                f'{{"type":"input_audio_buffer.append","audio":"{_pcm(0.32)}"}}'
            )
        )
        asyncio.run(session._dispatch('{"type":"input_audio_buffer.commit"}'))
        types = _types(ws)
        assert "input_audio_buffer.committed" in types
        completed = [e for e in ws.sent if e["type"].endswith(".completed")]
        assert completed and completed[-1]["transcript"] == "你好世界。"
        # committed 必须早于 completed
        assert types.index("input_audio_buffer.committed") < types.index(
            completed[-1]["type"]
        )

    def test_clear(self, monkeypatch):
        session, ws, _ = _session(monkeypatch)
        asyncio.run(
            session._dispatch(
                f'{{"type":"input_audio_buffer.append","audio":"{_pcm(0.32)}"}}'
            )
        )
        asyncio.run(session._dispatch('{"type":"input_audio_buffer.clear"}'))
        assert ws.sent[-1]["type"] == "input_audio_buffer.cleared"
        assert session._buffer.size == 0


class TestServerVad:
    def test_auto_finalize_on_silence(self, monkeypatch):
        session, ws, _ = _session(monkeypatch, vad_end=True)
        asyncio.run(
            session._dispatch(
                '{"type":"session.update","session":{"audio":{"input":'
                '{"turn_detection":{"type":"server_vad"}}}}}'
            )
        )
        asyncio.run(
            session._dispatch(
                f'{{"type":"input_audio_buffer.append","audio":"{_pcm(1.0)}"}}'
            )
        )
        types = _types(ws)
        assert "input_audio_buffer.committed" in types
        assert any(t.endswith(".completed") for t in types)

    def test_max_duration_force_finalize(self, monkeypatch):
        monkeypatch.setattr(settings, "QWEN_STREAM_SEGMENT_MAX_SEC", 1.0)
        session, ws, _ = _session(monkeypatch, vad_end=False)
        asyncio.run(
            session._dispatch(
                '{"type":"session.update","session":{"audio":{"input":'
                '{"turn_detection":{"type":"server_vad"}}}}}'
            )
        )
        asyncio.run(
            session._dispatch(
                f'{{"type":"input_audio_buffer.append","audio":"{_pcm(1.5)}"}}'
            )
        )
        assert any(t.endswith(".completed") for t in _types(ws))

    def test_manual_mode_never_auto_finalizes(self, monkeypatch):
        session, ws, _ = _session(monkeypatch, vad_end=True)
        asyncio.run(
            session._dispatch(
                f'{{"type":"input_audio_buffer.append","audio":"{_pcm(1.0)}"}}'
            )
        )
        assert not any(t.endswith(".completed") for t in _types(ws))
