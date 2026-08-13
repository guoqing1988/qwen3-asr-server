# -*- coding: utf-8 -*-
"""merge_hotwords 纯函数与服务层/接口层热词接线测试。"""

import os
import unittest
from unittest import mock

from app.services.asr.offline_transcription_service import merge_hotwords


class MergeHotwordsTest(unittest.TestCase):
    """merge_hotwords 纯函数单元测试。"""

    def test_both_empty(self) -> None:
        self.assertEqual(merge_hotwords("", ""), "")

    def test_default_only(self) -> None:
        self.assertEqual(merge_hotwords("阿里巴巴 腾讯", ""), "阿里巴巴 腾讯")

    def test_request_only(self) -> None:
        self.assertEqual(merge_hotwords("", "OpenAI Kubernetes"), "OpenAI Kubernetes")

    def test_merge_order_default_first(self) -> None:
        result = merge_hotwords("阿里巴巴 腾讯", "OpenAI")
        self.assertEqual(result, "阿里巴巴 腾讯 OpenAI")

    def test_dedup_exact_match(self) -> None:
        # 请求词与默认词重复时只保留首次出现（默认在前）
        result = merge_hotwords("阿里巴巴 OpenAI", "OpenAI 腾讯")
        self.assertEqual(result, "阿里巴巴 OpenAI 腾讯")

    def test_dedup_case_sensitive(self) -> None:
        # 大小写敏感：Kubernetes 与 kubernetes 是不同词，避免破坏英文写法
        result = merge_hotwords("Kubernetes", "kubernetes")
        self.assertEqual(result, "Kubernetes kubernetes")

    def test_whitespace_cleaning(self) -> None:
        result = merge_hotwords("  阿里巴巴   腾讯  ", "  OpenAI   ")
        self.assertEqual(result, "阿里巴巴 腾讯 OpenAI")


class SettingsDefaultHotwordsTest(unittest.TestCase):
    """ASR_DEFAULT_HOTWORDS 环境变量解析测试。"""

    def test_load_from_env_strips(self) -> None:
        from app.core.config import Settings

        with mock.patch.dict(os.environ, {"ASR_DEFAULT_HOTWORDS": "  阿里巴巴  腾讯  "}):
            self.assertEqual(Settings().ASR_DEFAULT_HOTWORDS, "阿里巴巴  腾讯")

    def test_default_is_empty_string(self) -> None:
        from app.core.config import Settings

        with mock.patch.dict(os.environ, {"ASR_DEFAULT_HOTWORDS": ""}):
            self.assertEqual(Settings().ASR_DEFAULT_HOTWORDS, "")


class ServiceHotwordsMergeTest(unittest.IsolatedAsyncioTestCase):
    """服务层转写：默认热词与请求热词合并后传给路由。"""

    async def _run_transcribe(self, default: str, request: str) -> str:
        from app.core.config import settings
        from app.services.asr.engines import ASRFullResult
        from app.services.asr.offline_transcription_service import (
            OfflineTranscriptionOptions,
            OfflineTranscriptionService,
            PreparedAudio,
        )
        from app.services.asr.runtime import get_runtime_router

        captured: dict[str, object] = {}

        async def fake_run_offline(asr_request):
            captured["hotwords"] = asr_request.hotwords
            return ASRFullResult(text="ok", segments=[], duration=0.0)

        router = get_runtime_router()
        service = OfflineTranscriptionService()
        audio = PreparedAudio(
            normalized_path="/tmp/fake.wav",
            duration=1.0,
            original_path="/tmp/fake.wav",
        )
        options = OfflineTranscriptionOptions(hotwords=request)
        with (
            mock.patch.object(router, "run_offline", new=fake_run_offline),
            mock.patch.object(settings, "ASR_DEFAULT_HOTWORDS", default),
            mock.patch.object(settings, "VOICEPRINT_ENABLED", False),
        ):
            await service.transcribe(audio, options)
        return str(captured["hotwords"])

    async def test_default_plus_request_merged(self) -> None:
        result = await self._run_transcribe("阿里巴巴 腾讯", "OpenAI")
        self.assertEqual(result, "阿里巴巴 腾讯 OpenAI")

    async def test_no_default_uses_request(self) -> None:
        result = await self._run_transcribe("", "OpenAI Kubernetes")
        self.assertEqual(result, "OpenAI Kubernetes")

    async def test_no_request_uses_default(self) -> None:
        result = await self._run_transcribe("阿里巴巴 腾讯", "")
        self.assertEqual(result, "阿里巴巴 腾讯")


if __name__ == "__main__":
    unittest.main()
