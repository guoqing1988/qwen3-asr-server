# -*- coding: utf-8 -*-
"""离线转写的句末标点补齐（复用 FunASR CT-Transformer 标点模型）。

R2T2 在段尾没说完时通常不给标点，离线输出常见「…是不是所以你看」这种
无结尾标点的句子。这里只**补一个句末标点**，绝不复制标点模型对正文的改写
（它会改写空格/数字写法），也不改动已有标点。
"""

from __future__ import annotations

import logging

from app.services.asr.engines.global_models import (
    get_global_punc_model,
    get_punc_inference_lock,
)

logger = logging.getLogger(__name__)

_TERMINALS = frozenset(".?!。？！…")
# 结尾是这些标点时先去掉再补句末（避免「，。」）
_CONTINUATIONS = frozenset(",;:，；：、")
# 英文/数字结尾用 ASCII 标点
_ENGLISH_MARKS = str.maketrans("。？！", ".?!")
_CLOSING_QUOTES = "\"'’”」』】)]}"


def _ending(text: str) -> str:
    """去掉尾部空白与收尾引号/括号后的有效结尾部分。"""
    return text.rstrip().rstrip(_CLOSING_QUOTES).rstrip()


def _copy_final_mark(text: str, result: object) -> str:
    """只把标点模型给出的句末标点搬到原文，正文保持原样。"""
    if (
        not isinstance(result, dict)
        or not isinstance(result.get("text"), str)
        or not result["text"].strip()
    ):
        return text
    ending = _ending(text)
    restored = _ending(result["text"])
    if not ending or not restored or restored[-1] not in _TERMINALS:
        return text
    mark = restored[-1]
    if ending[-1].isascii() and ending[-1].isalnum():
        mark = mark.translate(_ENGLISH_MARKS)
    suffix = text[len(ending) :]
    if ending[-1] in _CONTINUATIONS:
        ending = ending[:-1]
    return ending + mark + suffix


def restore_sentence_endings(texts: list[str]) -> list[str]:
    """批量补齐缺失的句末标点（一次模型调用，失败时原样返回）。

    Args:
        texts: 各段识别文本

    Returns:
        与输入等长的文本列表
    """
    restored = list(texts)
    missing = [
        index
        for index, text in enumerate(texts)
        if (ending := _ending(text)) and ending[-1] not in _TERMINALS
    ]
    if not missing:
        return restored
    try:
        model = get_global_punc_model("cpu")
        with get_punc_inference_lock():
            results = model.generate(input=[texts[index] for index in missing])
    except Exception as exc:  # noqa: BLE001 - 标点补齐失败不影响转写结果
        logger.warning("Sentence-ending restoration unavailable: %s", exc)
        return restored
    if not isinstance(results, list) or len(results) != len(missing):
        logger.warning(
            "Punctuation model returned %s results for %s inputs",
            len(results) if isinstance(results, list) else type(results).__name__,
            len(missing),
        )
        return restored
    for index, result in zip(missing, results):
        restored[index] = _copy_final_mark(texts[index], result)
    return restored
