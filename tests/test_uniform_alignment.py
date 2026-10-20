import pytest
from app.services.asr.uniform_alignment import uniform_word_timestamps


def test_cjk_split_and_fill_duration():
    words = uniform_word_timestamps("你好，世界", 4.0)
    assert [w.text for w in words] == ["你", "好，", "世", "界"]
    assert words[0].start_time == 0.0
    assert words[-1].end_time == pytest.approx(4.0)
    assert all(a.end_time <= b.start_time + 1e-9 for a, b in zip(words, words[1:]))


def test_latin_words_and_punctuation_attach():
    words = uniform_word_timestamps("hello, world!", 3.0)
    assert [w.text for w in words] == ["hello,", "world!"]
    assert words[-1].end_time == pytest.approx(3.0)


def test_empty_text_returns_empty():
    assert uniform_word_timestamps("   ", 2.0) == []
