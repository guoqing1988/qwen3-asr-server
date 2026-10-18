# -*- coding: utf-8 -*-
"""音频解码兜底链路测试

覆盖 convert_audio_to_wav 的三级解码策略：
1. librosa/soundfile（正常文件）
2. PyAV 逐帧容错解码（含损坏帧的 mp3）
3. ffmpeg CLI（命令参数正确性 + 未安装时的降级）
"""

import os
import subprocess
import tempfile
import unittest
from fractions import Fraction
from unittest.mock import patch

import av
import numpy as np
import soundfile as sf

from app.core.exceptions import InvalidParameterException
from app.utils.audio import convert_audio_to_wav


def _make_mp3(path: str, seconds: int = 8, sr: int = 16000) -> None:
    """用 PyAV 编码一段正弦波 mp3 作为测试素材"""
    t = np.arange(sr * seconds) / sr
    samples = (0.3 * np.sin(2 * np.pi * 440 * t)).astype(np.float32)
    container = av.open(path, "w", format="mp3")
    try:
        stream = container.add_stream("libmp3lame", rate=sr)
        frame = av.AudioFrame.from_ndarray(samples.reshape(1, -1), format="fltp", layout="mono")
        frame.sample_rate = sr
        frame.pts = 0
        frame.time_base = Fraction(1, sr)
        for packet in stream.encode(frame):
            container.mux(packet)
        for packet in stream.encode(None):
            container.mux(packet)
    finally:
        container.close()


def _corrupt_mp3(src: str, dst: str, offset: int = 20000, junk_size: int = 40960) -> None:
    """在 mp3 中间插入一段垃圾字节，模拟损坏文件（非法 MPEG 头 + 后续可恢复数据）"""
    raw = open(src, "rb").read()
    rng = np.random.default_rng(0)
    junk = rng.integers(0, 256, size=junk_size, dtype=np.uint8).tobytes()
    with open(dst, "wb") as f:
        f.write(raw[:offset] + junk + raw[offset:])


class AudioDecodeFallbackTest(unittest.TestCase):
    def setUp(self) -> None:
        self._tmpdir = tempfile.TemporaryDirectory()
        self.tmp = self._tmpdir.name

    def tearDown(self) -> None:
        self._tmpdir.cleanup()

    def _new_mp3(self, name: str = "normal.mp3", **kwargs: int) -> str:
        path = os.path.join(self.tmp, name)
        try:
            _make_mp3(path, **kwargs)
        except Exception as e:  # pragma: no cover - 环境缺少 libmp3lame 编码器
            self.skipTest(f"PyAV 无法编码 mp3（缺少 libmp3lame）: {e}")
        return path

    def test_normal_mp3_converted_by_librosa(self) -> None:
        """正常 mp3 走 librosa，输出 16k 单声道 wav 且时长一致"""
        src = self._new_mp3(seconds=8)
        dst = os.path.join(self.tmp, "normal.wav")

        out = convert_audio_to_wav(src, dst, target_sr=16000)

        self.assertEqual(out, dst)
        info = sf.info(dst)
        self.assertEqual(info.samplerate, 16000)
        self.assertEqual(info.channels, 1)
        self.assertAlmostEqual(info.duration, 8.0, delta=0.2)

    def test_corrupted_mp3_falls_back_to_pyav(self) -> None:
        """损坏 mp3 在 librosa 失败后由 PyAV 容错解码救回"""
        normal = self._new_mp3(seconds=8)
        broken = os.path.join(self.tmp, "broken.mp3")
        _corrupt_mp3(normal, broken)
        dst = os.path.join(self.tmp, "broken.wav")

        # 强制模拟 librosa/soundfile 解码失败（否则本机装了 ffmpeg 时可能由
        # audioread 的 ffmpeg 后端直接成功，测不到 PyAV 分支）
        with (
            patch(
                "app.utils.audio.librosa.load",
                side_effect=RuntimeError("soundfile failed"),
            ),
            self.assertLogs("app.utils.audio", level="WARNING") as logs,
        ):
            out = convert_audio_to_wav(broken, dst, target_sr=16000)

        self.assertEqual(out, dst)
        self.assertTrue(os.path.exists(dst))
        joined = "\n".join(logs.output)
        self.assertIn("PyAV 容错解码", joined)

        info = sf.info(dst)
        self.assertEqual(info.samplerate, 16000)
        self.assertEqual(info.channels, 1)
        # 容错解码应恢复大部分音频（而非直接失败或只剩极短片段）
        self.assertGreater(info.duration, 4.0)

    def test_undecodable_input_reports_layered_errors(self) -> None:
        """完全无法解码时，错误信息须包含各解码层的真实原因（不能是空串）"""
        bad = os.path.join(self.tmp, "garbage.mp3")
        with open(bad, "wb") as f:
            f.write(bytes(np.random.default_rng(1).integers(0, 256, 4096, dtype=np.uint8)))

        # 输入音频不可解码属客户端错误 → InvalidParameterException（HTTP 400）
        with self.assertRaises(InvalidParameterException) as ctx:
            convert_audio_to_wav(bad, os.path.join(self.tmp, "garbage.wav"))

        message = str(ctx.exception)
        self.assertIn("音频格式转换失败", message)
        self.assertIn("librosa:", message)
        self.assertIn("pyav:", message)
        self.assertIn("ffmpeg:", message)

    def test_ffmpeg_command_does_not_force_raw_input(self) -> None:
        """ffmpeg 兜底命令不得再出现把 mp3 当裸 PCM 的 `-f s16le`"""
        src = self._new_mp3(seconds=2)
        dst = os.path.join(self.tmp, "ffmpeg.wav")
        captured: dict[str, list[str]] = {}

        def fake_run(cmd, **kwargs):  # type: ignore[no-untyped-def]
            captured["cmd"] = cmd
            # 模拟 ffmpeg 成功产出文件
            sf.write(dst, np.zeros(16000, dtype=np.float32), 16000, format="WAV")
            return subprocess.CompletedProcess(cmd, 0, b"", b"")

        with (
            patch("app.utils.audio.librosa.load", side_effect=RuntimeError("soundfile failed")),
            patch(
                "app.utils.audio._decode_audio_with_pyav",
                side_effect=RuntimeError("pyav failed"),
            ),
            patch("app.utils.audio.subprocess.run", side_effect=fake_run),
        ):
            out = convert_audio_to_wav(src, dst, target_sr=16000)

        self.assertEqual(out, dst)
        cmd = captured["cmd"]
        self.assertEqual(cmd[0], "ffmpeg")
        self.assertNotIn("-f", cmd)
        self.assertIn("-i", cmd)
        self.assertIn("pcm_s16le", cmd)
        self.assertIn(str(16000), cmd)
        self.assertEqual(cmd[-1], dst)


if __name__ == "__main__":
    unittest.main()
