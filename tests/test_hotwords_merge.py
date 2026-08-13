# -*- coding: utf-8 -*-
"""merge_hotwords 纯函数与服务层/接口层热词接线测试。"""

import unittest

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


if __name__ == "__main__":
    unittest.main()
