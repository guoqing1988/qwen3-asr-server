# -*- coding: utf-8 -*-
"""Qwen3/Confucius4 流式（vLLM 路径）解析与 prefix 逻辑测试。

背景（2026-09-25 实测修复的三个问题）：
1. `_build_stream_prefix` 原先只取 `raw_decoded` 末尾 N 个 token 作 prefix，
   等于丢掉全部历史 —— 本项目是「每轮全量音频 + 全量文本重算」（无 vLLM KV
   增量续写），导致流式文本逐轮丢内容、退化成标点（如只剩 "."）。
2. `_parse_asr_output` 未清理 prefix 残留的 `|`（prefix 用
   skip_special_tokens=False 解码会带出标记字符），污染语言字段与正文。
3. 音频不足时模型会把语言预测成字面量 `None`，需回退到请求指定的语言。
"""

import unittest
from typing import Any

from app.services.asr.qwen3_vllm import _build_stream_prefix, _parse_asr_output


class _CharTokenizer:
    """按字符切分的极简 tokenizer，便于精确断言 prefix 边界。"""

    def encode(self, text: str, add_special_tokens: bool = False) -> list[str]:
        return list(text)

    def decode(self, token_ids: list[Any], skip_special_tokens: bool = False) -> str:
        return "".join(str(token) for token in token_ids)


class ParseAsrOutputTest(unittest.TestCase):
    def test_standard_format(self) -> None:
        language, text = _parse_asr_output(
            "language English<asr_text>So good to see you.", None
        )
        self.assertEqual(language, "English")
        self.assertEqual(text, "So good to see you.")

    def test_strips_pipe_residue(self) -> None:
        language, text = _parse_asr_output("|language English<asr_text>|So good", None)
        self.assertEqual(language, "English")
        self.assertEqual(text, "So good")

    def test_none_literal_language_falls_back_to_requested(self) -> None:
        language, text = _parse_asr_output("language None<asr_text>So", "English")
        self.assertEqual(language, "English")
        self.assertEqual(text, "So")

    def test_missing_marker_keeps_requested_language(self) -> None:
        language, text = _parse_asr_output("So good", "English")
        self.assertEqual(language, "English")
        self.assertEqual(text, "So good")

    def test_empty_input(self) -> None:
        self.assertEqual(_parse_asr_output("", "English"), ("English", ""))


class BuildStreamPrefixTest(unittest.TestCase):
    def setUp(self) -> None:
        self.tokenizer = _CharTokenizer()

    def test_keeps_full_history_minus_rollback(self) -> None:
        """回归点：prefix 必须是完整历史（去掉末尾 unfixed_token_num 个 token）。"""
        prefix = _build_stream_prefix(self.tokenizer, "language English<asr_text>abcdefg", 1)
        self.assertEqual(prefix, "language English<asr_text>abcdef")

    def test_keeps_everything_when_no_rollback(self) -> None:
        self.assertEqual(_build_stream_prefix(self.tokenizer, "abc", 0), "abc")

    def test_empty_when_rollback_covers_all(self) -> None:
        self.assertEqual(_build_stream_prefix(self.tokenizer, "a", 5), "")

    def test_empty_input(self) -> None:
        self.assertEqual(_build_stream_prefix(self.tokenizer, "", 1), "")


if __name__ == "__main__":
    unittest.main()
