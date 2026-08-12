"""Qwen3 forced aligner 对齐逻辑测试：对齐单元切分、时间戳全局单调化、align_transcript 解析链路。

GPU 模型推理不可控且重（需加载 forced aligner 模型），align_transcript 链路测试
通过 mock aligner.encode 的输出构造分类 logits，验证项目侧解析与修复逻辑（注释见
CLAUDE.md 测试规范：对外部不可控服务允许 Mock）。
"""

import unittest
from unittest import mock

import numpy as np
import torch

from app.services.asr.qwen3_vllm import (
    Qwen3VLLMBackend,
    _fix_timestamp,
    _split_alignment_units,
)

_TIMESTAMP_TOKEN_ID = 151705  # 与 Qwen3-ForcedAligner-0.6B config.json 一致
_SEGMENT_TIME_MS = 80.0


class SplitAlignmentUnitsTest(unittest.TestCase):
    def test_cjk_with_punctuation_keeps_only_characters(self) -> None:
        self.assertEqual(_split_alignment_units("你好，世界。"), ["你", "好", "世", "界"])

    def test_latin_words_with_punctuation_keeps_only_words(self) -> None:
        self.assertEqual(_split_alignment_units("Hello, world!"), ["Hello", "world"])

    def test_mixed_cjk_latin_with_punctuation(self) -> None:
        self.assertEqual(
            _split_alignment_units("state-of-the-art, 很好。"),
            ["state-of-the-art", "很", "好"],
        )

    def test_internal_word_symbols_are_kept(self) -> None:
        self.assertEqual(
            _split_alignment_units("don't stop 1.5 倍"),
            ["don't", "stop", "1.5", "倍"],
        )

    def test_punctuation_only_text_yields_no_units(self) -> None:
        self.assertEqual(_split_alignment_units("。。。，"), [])

    def test_empty_text_yields_no_units(self) -> None:
        self.assertEqual(_split_alignment_units(""), [])


class FixTimestampTest(unittest.TestCase):
    def test_normal_monotonic_sequence_unchanged(self) -> None:
        data = [50.0, 60.0, 70.0, 80.0]
        self.assertEqual(_fix_timestamp(data), data)

    def test_single_reversed_pair_is_fixed(self) -> None:
        # [100, 120, 90, 140]: 第二个单元 start=90 < end=120 反转
        fixed = _fix_timestamp([100.0, 120.0, 90.0, 140.0])
        self.assert_monotonic(fixed)
        self.assertEqual(fixed[1], fixed[2])  # 反转点被就近填充

    def test_tail_regression_is_fixed(self) -> None:
        fixed = _fix_timestamp([90.0, 100.0, 110.0, 95.0, 120.0])
        self.assert_monotonic(fixed)
        self.assertEqual(fixed[3], 110.0)  # 回退点取左侧正常值

    def test_consecutive_anomalies_filled_by_nearest_value(self) -> None:
        fixed = _fix_timestamp([100.0, 90.0, 80.0, 140.0])
        self.assert_monotonic(fixed)
        # 异常段 (90,80)：90 距左 100 一步取 100；80 距右 140 一步取 140
        self.assertEqual(fixed, [100.0, 100.0, 140.0, 140.0])

    def test_long_anomaly_run_uses_linear_interpolation(self) -> None:
        # LIS 为 [100,200,210]（降序段 90,80,70 无法接入），
        # 连续异常 3 个 > 2 → 按 100/200 线性插值 125/150/175
        fixed = _fix_timestamp([100.0, 90.0, 80.0, 70.0, 200.0, 210.0])
        self.assert_monotonic(fixed)
        self.assertEqual(fixed, [100.0, 125.0, 150.0, 175.0, 200.0, 210.0])

    def test_head_anomaly_filled_by_right_value(self) -> None:
        # 头部 120 异常（LIS 为 90→100→140），左侧无正常值 → 取右侧 90
        fixed = _fix_timestamp([120.0, 90.0, 100.0, 140.0])
        self.assertEqual(fixed, [90.0, 90.0, 100.0, 140.0])

    def assert_monotonic(self, data: list[float]) -> None:
        for i in range(len(data) - 1):
            self.assertLessEqual(
                data[i], data[i + 1],
                f"sequence not monotonic at index {i}: {data}",
            )


class AlignTranscriptTest(unittest.TestCase):
    """通过 mock forced aligner 输出，验证 align_transcript 完整解析链路。"""

    def _make_backend(self, prompt_token_ids: list[int], bins: list[int]) -> Qwen3VLLMBackend:
        """绕过 __init__ 构造 backend（构造会加载真实 vLLM 模型，见 CLAUDE.md Mock 说明）。"""
        backend = object.__new__(Qwen3VLLMBackend)
        backend._timestamp_token_id = _TIMESTAMP_TOKEN_ID
        backend._timestamp_segment_time = _SEGMENT_TIME_MS
        backend._get_forced_aligner = lambda: _FakeAligner(prompt_token_ids, bins)
        return backend

    def test_aligns_tokens_with_normal_timestamps(self) -> None:
        backend = self._make_backend(
            prompt_token_ids=self._prompt_ids(),
            bins=[5, 7, 12, 16],  # ms: [400, 560, 960, 1280]，单调正常
        )
        with mock.patch(
            "app.services.asr.qwen3_vllm._load_audio",
            return_value=np.zeros(16000, dtype=np.float32),
        ):
            aligned = backend.align_transcript(
                audio_path="unused.wav",
                text="你好。",
                language="Chinese",
            )

        self.assertEqual([item["text"] for item in aligned], ["你", "好"])
        self.assertEqual(aligned[0]["start_ms"], 400.0)
        self.assertEqual(aligned[0]["end_ms"], 560.0)
        self.assertEqual(aligned[1]["start_ms"], 960.0)
        self.assertEqual(aligned[1]["end_ms"], 1280.0)

    def test_reversed_predictions_are_globally_fixed(self) -> None:
        # 第一个单元的 end=240 < start=400（反转）；LIS 为 [5,12,16]，
        # 位置 1（值 3）为单点异常，就近取左 5 → 修复为 [5,5,12,16]
        backend = self._make_backend(
            prompt_token_ids=self._prompt_ids(),
            bins=[5, 3, 12, 16],
        )
        with mock.patch(
            "app.services.asr.qwen3_vllm._load_audio",
            return_value=np.zeros(16000, dtype=np.float32),
        ):
            aligned = backend.align_transcript(
                audio_path="unused.wav",
                text="你好。",
                language="Chinese",
            )

        self.assertEqual(aligned[0]["start_ms"], 400.0)
        self.assertEqual(aligned[0]["end_ms"], 400.0)
        self.assertEqual(aligned[1]["start_ms"], 960.0)
        self.assertEqual(aligned[1]["end_ms"], 1280.0)
        # 全序列单调
        spans = [(item["start_ms"], item["end_ms"]) for item in aligned]
        flat = [v for span in spans for v in span]
        self.assertEqual(flat, sorted(flat))

    def test_punctuation_does_not_become_alignment_unit(self) -> None:
        backend = self._make_backend(
            prompt_token_ids=self._prompt_ids(unit_count=4),
            bins=[5, 7, 12, 16, 20, 24, 28, 32],
        )
        with mock.patch(
            "app.services.asr.qwen3_vllm._load_audio",
            return_value=np.zeros(16000, dtype=np.float32),
        ):
            aligned = backend.align_transcript(
                audio_path="unused.wav",
                text="你好，世界！",
                language="Chinese",
            )

        # 标点不作为对齐单元，输出仅 4 个字
        self.assertEqual([item["text"] for item in aligned], ["你", "好", "世", "界"])

    @staticmethod
    def _prompt_ids(unit_count: int = 2) -> list[int]:
        """构造对齐 prompt token id 序列（与 _build_alignment_prompt 的 join+尾部结构一致）。

        prompt: <|audio_start|><|audio_pad|><|audio_end|> +
                <ts><ts>单元1<ts><ts>单元2 ... <ts><ts>
        → [audio x3] + 每单元 2 个 ts + 文本占位 + 尾部 2 个 ts
        """
        ids = [100, 101, 102]  # audio placeholder
        for unit_id in range(1000, 1000 + unit_count):  # 文本 token 占位
            ids.extend([_TIMESTAMP_TOKEN_ID, _TIMESTAMP_TOKEN_ID, unit_id])
        ids.extend([_TIMESTAMP_TOKEN_ID, _TIMESTAMP_TOKEN_ID])
        return ids


class _FakeAligner:
    """模拟 forced aligner encode 输出：每 prompt token 一个分类 logits 行。"""

    def __init__(self, prompt_token_ids: list[int], bins: list[int]) -> None:
        self._prompt_token_ids = prompt_token_ids
        self._bins = bins

    def encode(self, prompts: list[dict], **kwargs) -> list[object]:
        # bins 只填充到 prompt 中的 timestamp 位置；prompt 尾部多出的
        # timestamp（句末对）不在解析范围，pad 为 0。
        # 用 torch tensor 模拟真实 logits（解析路径走 argmax(dim=-1)）。
        bins_iter = iter(self._bins)
        logits = torch.zeros((len(self._prompt_token_ids), 5000), dtype=torch.float32)
        for i, tid in enumerate(self._prompt_token_ids):
            if tid == _TIMESTAMP_TOKEN_ID:
                bin_idx = next(bins_iter, 0)
                logits[i, bin_idx] = 1.0
        return [_FakeOutput(self._prompt_token_ids, logits)]


class _FakeOutput:
    def __init__(self, prompt_token_ids: list[int], logits: torch.Tensor) -> None:
        self.prompt_token_ids = prompt_token_ids
        self.outputs = _FakeLogits(logits)


class _FakeLogits:
    def __init__(self, data: torch.Tensor) -> None:
        self.data = data


if __name__ == "__main__":
    unittest.main()
