"""均匀词级时间戳（无语声对齐）：按字/词均分时长，作为 aligner 不可用时的兜底。

- CJK 逐字、拉丁按词、标点/符号并入前一个单元
- 最后单元对齐到段尾，保证覆盖整段
- 精度明显低于 forced aligner，仅用于 `ALIGNMENT_MODE=uniform` 或降级场景
"""

import unicodedata

from .engines import WordToken


def _is_cjk_character(character: str) -> bool:
    codepoint = ord(character)
    return (
        0x3400 <= codepoint <= 0x4DBF
        or 0x4E00 <= codepoint <= 0x9FFF
        or 0xF900 <= codepoint <= 0xFAFF
    )


def _alignment_units(text: str) -> list[str]:
    units: list[str] = []
    current_word: list[str] = []
    leading_punctuation = ""

    def flush_word() -> None:
        nonlocal leading_punctuation
        if not current_word:
            return
        units.append(f"{leading_punctuation}{''.join(current_word)}")
        current_word.clear()
        leading_punctuation = ""

    for character in text:
        if character.isspace():
            flush_word()
            continue
        if _is_cjk_character(character):
            flush_word()
            units.append(f"{leading_punctuation}{character}")
            leading_punctuation = ""
            continue

        category = unicodedata.category(character)
        if category[0] in {"L", "M", "N"}:
            current_word.append(character)
            continue

        flush_word()
        if units:
            units[-1] += character
        else:
            leading_punctuation += character

    flush_word()
    if leading_punctuation and units:
        units[-1] += leading_punctuation
    return units


def uniform_word_timestamps(text: str, duration: float) -> list[WordToken]:
    # ponytail: pauses are distributed across words; use a validated acoustic aligner for accuracy.
    units = _alignment_units(text)
    if not units or duration <= 0:
        return []
    step = duration / len(units)
    return [
        WordToken(
            unit,
            index * step,
            duration if index == len(units) - 1 else (index + 1) * step,
        )
        for index, unit in enumerate(units)
    ]
