# -*- coding: utf-8 -*-
"""离线句末标点补齐的单元测试（fake 标点模型，不加载真实 FunASR 模型）。"""

from __future__ import annotations

import contextlib

import pytest

from app.core.config import settings
from app.services.asr import punctuation
from app.services.asr.punctuation import restore_sentence_endings


class FakePuncModel:
    def __init__(self, rewrite: dict[str, str] | None = None, count: int | None = None):
        self.rewrite = rewrite or {}
        self.count = count
        self.calls: list[list[str]] = []

    def generate(self, input: list[str]) -> list[dict[str, str]]:
        self.calls.append(list(input))
        results = [
            {"text": self.rewrite.get(text, text + "。")} for text in input
        ]
        if self.count is not None:
            results = results[: self.count]
        return results


@pytest.fixture()
def patch_model(monkeypatch):
    def _install(model: FakePuncModel) -> FakePuncModel:
        monkeypatch.setattr(punctuation, "get_global_punc_model", lambda device: model)
        monkeypatch.setattr(
            punctuation, "get_punc_inference_lock", lambda: contextlib.nullcontext()
        )
        return model

    return _install


class TestRestoreSentenceEndings:
    def test_no_missing_mark_skips_model(self, patch_model):
        model = patch_model(FakePuncModel())
        texts = ["已经有标点。", "问句？", "英文.", ""]
        assert restore_sentence_endings(texts) == texts
        assert model.calls == []

    def test_appends_only_final_mark(self, patch_model):
        # 标点模型会改写正文（去空格等），我们只取最后一个标点
        model = patch_model(FakePuncModel(rewrite={"今天 天气": "今天天气。"}))
        assert restore_sentence_endings(["今天 天气"]) == ["今天 天气。"]
        assert model.calls == [["今天 天气"]]
        _ = model

    def test_english_and_digit_endings_use_ascii_mark(self, patch_model):
        patch_model(FakePuncModel())
        result = restore_sentence_endings(["hello world", "10 percent"])
        assert result == ["hello world.", "10 percent."]

    def test_chinese_ending_keeps_full_width_mark(self, patch_model):
        patch_model(FakePuncModel())
        assert restore_sentence_endings(["你好世界"]) == ["你好世界。"]

    def test_trailing_comma_is_replaced(self, patch_model):
        patch_model(FakePuncModel())
        assert restore_sentence_endings(["这句话没说完，"]) == ["这句话没说完。"]

    def test_closing_quote_stays_after_mark(self, patch_model):
        patch_model(FakePuncModel())
        assert restore_sentence_endings(['他说"好的"']) == ['他说"好的。"']

    def test_model_failure_keeps_original(self, patch_model, monkeypatch):
        def boom(device):
            raise RuntimeError("model missing")

        monkeypatch.setattr(punctuation, "get_global_punc_model", boom)
        texts = ["没有标点"]
        assert restore_sentence_endings(texts) == texts

    def test_result_count_mismatch_keeps_original(self, patch_model):
        patch_model(FakePuncModel(count=0))
        assert restore_sentence_endings(["没有标点"]) == ["没有标点"]

    def test_batch_single_call(self, patch_model):
        model = patch_model(FakePuncModel())
        restore_sentence_endings(["第一句", "第二句。", "第三句"])
        assert model.calls == [["第一句", "第三句"]]


class TestOfflineServiceIntegration:
    def test_service_restores_segment_and_full_text(self, monkeypatch):
        from app.services.asr import offline_transcription_service as module
        from app.services.asr.engines import ASRFullResult, ASRSegmentResult

        class FakeRouter:
            async def run_offline(self, request):
                return ASRFullResult(
                    text="第一句\n第二句。",
                    segments=[
                        ASRSegmentResult(text="第一句", start_time=0.0, end_time=1.0),
                        ASRSegmentResult(text="第二句。", start_time=1.0, end_time=2.0),
                    ],
                    duration=2.0,
                )

        class FakePrepared:
            normalized_path = "/tmp/x.wav"
            timestamp_scale = 1.0
            duration = 2.0
            original_path = "/tmp/x.wav"

        class FakeOptions:
            hotwords = ""
            sample_rate = 16000
            enable_speaker_diarization = False
            word_timestamps = False
            task_id = "t"

        monkeypatch.setattr(module, "get_runtime_router", lambda: FakeRouter())
        monkeypatch.setattr(module.settings, "VOICEPRINT_ENABLED", False)
        monkeypatch.setattr(
            module,
            "get_default_offline_model_id",
            lambda: "qwen3-asr-confucius4-r2t2",
        )
        monkeypatch.setattr(module.settings, "ASR_RESTORE_SENTENCE_ENDING", True)
        monkeypatch.setattr(
            punctuation,
            "restore_sentence_endings",
            lambda texts: [text if text[-1:] in "。？！" else text + "。" for text in texts],
        )

        import asyncio

        result = asyncio.run(
            module.OfflineTranscriptionService().transcribe(FakePrepared(), FakeOptions())
        )
        assert result.segments[0].text == "第一句。"
        assert result.segments[1].text == "第二句。"
        assert result.text == "第一句。\n第二句。"

    def test_service_switch_off(self, monkeypatch):
        from app.services.asr import offline_transcription_service as module
        from app.services.asr.engines import ASRFullResult, ASRSegmentResult

        class FakeRouter:
            async def run_offline(self, request):
                return ASRFullResult(
                    text="没有标点",
                    segments=[ASRSegmentResult(text="没有标点", start_time=0.0, end_time=1.0)],
                    duration=1.0,
                )

        class FakePrepared:
            normalized_path = "/tmp/x.wav"
            timestamp_scale = 1.0
            duration = 1.0
            original_path = "/tmp/x.wav"

        class FakeOptions:
            hotwords = ""
            sample_rate = 16000
            enable_speaker_diarization = False
            word_timestamps = False
            task_id = "t"

        monkeypatch.setattr(module, "get_runtime_router", lambda: FakeRouter())
        monkeypatch.setattr(module.settings, "VOICEPRINT_ENABLED", False)
        monkeypatch.setattr(
            module,
            "get_default_offline_model_id",
            lambda: "qwen3-asr-confucius4-r2t2",
        )
        monkeypatch.setattr(module.settings, "ASR_RESTORE_SENTENCE_ENDING", False)

        import asyncio

        result = asyncio.run(
            module.OfflineTranscriptionService().transcribe(FakePrepared(), FakeOptions())
        )
        assert result.text == "没有标点"
