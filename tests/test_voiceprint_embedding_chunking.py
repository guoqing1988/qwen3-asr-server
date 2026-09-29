"""声纹 embedding 长音频切片融合 + 设备选择测试。

背景（2026-09-28 实测）：CAM++ 对「同一说话人全部语句拼接后的超长音频」整体
推理时，TDNN 中间激活爆显存（`Tried to allocate 1.92 GiB` 后 CUDA OOM），且失败
后数 GB reserved 显存不会自动归还。修复：切片（≤30s）分别提取 + 按时长加权融合，
并把声纹推理默认放到 CPU。
"""

import sys
import tempfile
import unittest
import wave
from pathlib import Path
from unittest import mock

import numpy as np
import torch

from app.core.config import settings
from app.core.device import detect_device
from app.services.asr.engines import ASRFullResult, ASRSegmentResult
from app.services.speaker.config import (
    VOICEPRINT_MAX_CHUNK_SEC,
    VOICEPRINT_MAX_SEGMENTS_PER_SPEAKER,
    VOICEPRINT_MIN_CHUNK_SEC,
)
from app.services.speaker.domain import SpeakerEmbedding
from app.services.speaker.embedding_service import SpeakerEmbeddingService
from app.services.speaker.providers.campplus_embedding_provider import (
    CampplusEmbeddingProvider,
)

SAMPLE_RATE = 16000


class _FakeProvider:
    """记录每次 extract 的音频长度，并用音频均值构造可预测的向量。"""

    provider_name = "fake"
    model_name = "fake-model"
    sample_rate = SAMPLE_RATE

    def __init__(self) -> None:
        self.calls: list[int] = []

    def extract(self, audio_data: np.ndarray) -> SpeakerEmbedding:
        self.calls.append(int(audio_data.shape[0]))
        mean = float(np.mean(audio_data)) if audio_data.size else 0.0
        return SpeakerEmbedding(
            vector=np.array([mean, 1.0], dtype=np.float32),
            provider=self.provider_name,
            model_name=self.model_name,
            sample_rate=self.sample_rate,
            duration_sec=audio_data.shape[0] / self.sample_rate,
        )


def _write_wav(path: Path, audio: np.ndarray, sample_rate: int = SAMPLE_RATE) -> None:
    with wave.open(str(path), "wb") as handle:
        handle.setnchannels(1)
        handle.setsampwidth(2)
        handle.setframerate(sample_rate)
        handle.writeframes((np.clip(audio, -1.0, 1.0) * 32767).astype("<i2").tobytes())


class LongAudioChunkingTest(unittest.TestCase):
    def test_short_audio_still_uses_single_extract(self) -> None:
        provider = _FakeProvider()
        service = SpeakerEmbeddingService(provider)

        audio = np.full(5 * SAMPLE_RATE, 0.5, dtype=np.float32)
        embedding = service._extract_embedding_for_audio(audio, sample_rate=SAMPLE_RATE)

        self.assertEqual(provider.calls, [5 * SAMPLE_RATE])
        self.assertAlmostEqual(float(embedding.vector[0]), 0.5, places=5)

    def test_long_audio_is_chunked_and_duration_weighted(self) -> None:
        provider = _FakeProvider()
        service = SpeakerEmbeddingService(provider)

        # 75s：前 60s 为 0、后 15s 为 1 → 加权均值 15/75 = 0.2（简单平均会是 0.5）
        audio = np.concatenate(
            [
                np.zeros(60 * SAMPLE_RATE, dtype=np.float32),
                np.ones(15 * SAMPLE_RATE, dtype=np.float32),
            ]
        )
        embedding = service._extract_embedding_for_audio(audio, sample_rate=SAMPLE_RATE)

        self.assertEqual(len(provider.calls), 3)  # 30 + 30 + 15
        self.assertTrue(
            all(size <= int(VOICEPRINT_MAX_CHUNK_SEC * SAMPLE_RATE) for size in provider.calls)
        )
        self.assertAlmostEqual(float(embedding.vector[0]), 0.2, places=4)
        self.assertAlmostEqual(embedding.duration_sec, 75.0, places=3)

    def test_tiny_tail_chunk_is_dropped(self) -> None:
        provider = _FakeProvider()
        service = SpeakerEmbeddingService(provider)

        audio = np.zeros(int(60.5 * SAMPLE_RATE), dtype=np.float32)
        service._extract_embedding_for_audio(audio, sample_rate=SAMPLE_RATE)

        # 30 + 30 + 0.5s：尾片短于 VOICEPRINT_MIN_CHUNK_SEC，不送去提取
        self.assertEqual(len(provider.calls), 2)
        self.assertTrue(
            all(size >= int(VOICEPRINT_MIN_CHUNK_SEC * SAMPLE_RATE) for size in provider.calls)
        )

    def test_reference_embedding_full_path_is_chunked(self) -> None:
        provider = _FakeProvider()
        service = SpeakerEmbeddingService(provider)

        audio = np.zeros(70 * SAMPLE_RATE, dtype=np.float32)
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "reference.wav"
            _write_wav(path, audio)
            embedding = service.extract_reference_embedding(audio_path=str(path))

        self.assertEqual(len(provider.calls), 3)  # 30 + 30 + 10
        self.assertAlmostEqual(embedding.duration_sec, 70.0, places=2)
        # 向量已 L2 归一化：[0, 1] → norm 1
        self.assertAlmostEqual(float(np.linalg.norm(embedding.vector)), 1.0, places=5)

    def test_local_speaker_embeddings_follow_asr_segments(self) -> None:
        provider = _FakeProvider()
        service = SpeakerEmbeddingService(provider)

        # 段1 = 10s 的全 0，段2 = 15s 的全 1 → 加权均值 15/25 = 0.6（简单平均 0.5）
        audio = np.concatenate(
            [
                np.zeros(10 * SAMPLE_RATE, dtype=np.float32),
                np.ones(15 * SAMPLE_RATE, dtype=np.float32),
            ]
        )
        asr_result = ASRFullResult(
            text="hello world",
            duration=25.0,
            segments=[
                ASRSegmentResult(
                    text="hello", start_time=0.0, end_time=10.0, speaker_id="Speaker1"
                ),
                ASRSegmentResult(
                    text="world", start_time=10.0, end_time=25.0, speaker_id="Speaker1"
                ),
            ],
        )

        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "dialog.wav"
            _write_wav(path, audio)
            embeddings = service.extract_local_speaker_embeddings(
                audio_path=str(path),
                asr_result=asr_result,
                timestamp_scale=1.0,
            )

        self.assertEqual(set(embeddings), {"Speaker1"})
        # 按分段提取：两段各一次，长度与分段一致
        self.assertEqual(provider.calls, [10 * SAMPLE_RATE, 15 * SAMPLE_RATE])
        vector = embeddings["Speaker1"].vector
        self.assertAlmostEqual(float(vector[0] / vector[1]), 0.6, places=3)
        self.assertAlmostEqual(embeddings["Speaker1"].duration_sec, 25.0, places=2)

    def test_excessive_segments_are_sampled(self) -> None:
        provider = _FakeProvider()
        service = SpeakerEmbeddingService(provider)

        seg_count = 50  # > VOICEPRINT_MAX_SEGMENTS_PER_SPEAKER
        seg_sec = 4.0
        audio = np.full(int(seg_count * seg_sec * SAMPLE_RATE), 0.3, dtype=np.float32)
        asr_result = ASRFullResult(
            text="x",
            duration=seg_count * seg_sec,
            segments=[
                ASRSegmentResult(
                    text=str(index),
                    start_time=index * seg_sec,
                    end_time=(index + 1) * seg_sec,
                    speaker_id="Speaker1",
                )
                for index in range(seg_count)
            ],
        )

        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "many.wav"
            _write_wav(path, audio)
            embeddings = service.extract_local_speaker_embeddings(
                audio_path=str(path),
                asr_result=asr_result,
                timestamp_scale=1.0,
            )

        # 分段过多 → 只提取采样后的数量，但时长仍按全部语音统计
        self.assertEqual(len(provider.calls), VOICEPRINT_MAX_SEGMENTS_PER_SPEAKER)
        self.assertAlmostEqual(
            embeddings["Speaker1"].duration_sec, seg_count * seg_sec, places=1
        )

    def test_sample_chunks_evenly_spaced(self) -> None:
        chunks = [np.full(index + 1, float(index), dtype=np.float32) for index in range(50)]

        selected = SpeakerEmbeddingService._sample_chunks(chunks, 30)

        self.assertEqual(len(selected), 30)
        self.assertEqual(float(selected[0][0]), 0.0)  # 首段入选
        self.assertGreater(float(selected[-1][0]), 40.0)  # 覆盖到时间轴尾部
        # 不超限（含 limit=0 表示不限制）时原样返回
        self.assertIs(SpeakerEmbeddingService._sample_chunks(chunks, 0), chunks)
        self.assertIs(SpeakerEmbeddingService._sample_chunks(chunks, 50), chunks)

    def test_oversized_single_segment_is_split(self) -> None:
        provider = _FakeProvider()
        service = SpeakerEmbeddingService(provider)

        audio = np.full(70 * SAMPLE_RATE, 0.4, dtype=np.float32)
        asr_result = ASRFullResult(
            text="long",
            duration=70.0,
            segments=[
                ASRSegmentResult(
                    text="long", start_time=0.0, end_time=70.0, speaker_id="Speaker1"
                ),
            ],
        )

        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "single.wav"
            _write_wav(path, audio)
            embeddings = service.extract_local_speaker_embeddings(
                audio_path=str(path),
                asr_result=asr_result,
                timestamp_scale=1.0,
            )

        # 超长单段再细分：30 + 30 + 10
        self.assertEqual(
            provider.calls, [30 * SAMPLE_RATE, 30 * SAMPLE_RATE, 10 * SAMPLE_RATE]
        )
        vector = embeddings["Speaker1"].vector
        # 全 0.4 → 融合后仍 0.4 : 1.0
        self.assertAlmostEqual(float(vector[0] / vector[1]), 0.4, places=4)

    def test_tiny_segment_is_skipped(self) -> None:
        provider = _FakeProvider()
        service = SpeakerEmbeddingService(provider)

        audio = np.concatenate(
            [
                np.full(5 * SAMPLE_RATE, 0.4, dtype=np.float32),
                np.zeros(int(0.5 * SAMPLE_RATE), dtype=np.float32),
            ]
        )
        asr_result = ASRFullResult(
            text="hi",
            duration=5.5,
            segments=[
                ASRSegmentResult(
                    text="hi", start_time=0.0, end_time=5.0, speaker_id="Speaker1"
                ),
                ASRSegmentResult(
                    text="!", start_time=5.0, end_time=5.5, speaker_id="Speaker1"
                ),
            ],
        )

        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "tiny.wav"
            _write_wav(path, audio)
            service.extract_local_speaker_embeddings(
                audio_path=str(path),
                asr_result=asr_result,
                timestamp_scale=1.0,
            )

        # 0.5s 的段太短，不送去提取
        self.assertEqual(provider.calls, [5 * SAMPLE_RATE])

    def test_all_tiny_segments_fall_back_to_concat(self) -> None:
        provider = _FakeProvider()
        service = SpeakerEmbeddingService(provider)

        seg_count = 5
        seg_sec = 0.9
        audio = np.full(int(seg_count * seg_sec * SAMPLE_RATE), 0.3, dtype=np.float32)
        asr_result = ASRFullResult(
            text="a b c d e",
            duration=seg_count * seg_sec,
            segments=[
                ASRSegmentResult(
                    text=str(index),
                    start_time=index * seg_sec,
                    end_time=(index + 1) * seg_sec,
                    speaker_id="Speaker1",
                )
                for index in range(seg_count)
            ],
        )

        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "tiny_all.wav"
            _write_wav(path, audio)
            embeddings = service.extract_local_speaker_embeddings(
                audio_path=str(path),
                asr_result=asr_result,
                timestamp_scale=1.0,
            )

        # 全部过短 → 退化拼接后提取（否则整个说话人会丢失）
        self.assertEqual(len(provider.calls), 1)
        self.assertAlmostEqual(
            provider.calls[0], int(seg_count * seg_sec * SAMPLE_RATE), delta=SAMPLE_RATE
        )
        self.assertEqual(set(embeddings), {"Speaker1"})

    def test_speaker_below_min_speech_is_skipped(self) -> None:
        provider = _FakeProvider()
        service = SpeakerEmbeddingService(provider)

        audio = np.full(int(1.5 * SAMPLE_RATE), 0.4, dtype=np.float32)
        asr_result = ASRFullResult(
            text="hi",
            duration=1.5,
            segments=[
                ASRSegmentResult(
                    text="hi", start_time=0.0, end_time=1.5, speaker_id="Speaker1"
                ),
            ],
        )

        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "short.wav"
            _write_wav(path, audio)
            embeddings = service.extract_local_speaker_embeddings(
                audio_path=str(path),
                asr_result=asr_result,
                timestamp_scale=1.0,
            )

        self.assertEqual(embeddings, {})
        self.assertEqual(provider.calls, [])


class VoiceprintDeviceTest(unittest.TestCase):
    def test_default_device_is_auto(self) -> None:
        self.assertEqual(settings.VOICEPRINT_DEVICE, "auto")
        # auto：有 CUDA 时用 GPU（分段已限长，峰值约 840MiB），否则降级 CPU
        expected = "cuda:0" if torch.cuda.is_available() else "cpu"
        self.assertEqual(detect_device(settings.VOICEPRINT_DEVICE), expected)

    def test_gpu_cache_released_after_reference_extraction(self) -> None:
        provider = _FakeProvider()
        service = SpeakerEmbeddingService(provider)

        audio = np.full(5 * SAMPLE_RATE, 0.4, dtype=np.float32)
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "reference.wav"
            _write_wav(path, audio)
            with mock.patch.object(
                SpeakerEmbeddingService, "_release_gpu_cache"
            ) as release:
                service.extract_reference_embedding(audio_path=str(path))

        release.assert_called_once()

    def test_gpu_cache_released_after_local_extraction(self) -> None:
        provider = _FakeProvider()
        service = SpeakerEmbeddingService(provider)

        audio = np.full(6 * SAMPLE_RATE, 0.4, dtype=np.float32)
        asr_result = ASRFullResult(
            text="hi",
            duration=6.0,
            segments=[
                ASRSegmentResult(
                    text="hi", start_time=0.0, end_time=6.0, speaker_id="Speaker1"
                ),
            ],
        )

        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "local.wav"
            _write_wav(path, audio)
            with mock.patch.object(
                SpeakerEmbeddingService, "_release_gpu_cache"
            ) as release:
                service.extract_local_speaker_embeddings(
                    audio_path=str(path),
                    asr_result=asr_result,
                    timestamp_scale=1.0,
                )

        release.assert_called_once()

    def test_campplus_provider_uses_voiceprint_device(self) -> None:
        fake_pipeline = object()
        with (
            mock.patch.dict(
                sys.modules,
                {
                    "modelscope.pipelines": mock.MagicMock(
                        pipeline=mock.MagicMock(return_value=fake_pipeline)
                    ),
                    "modelscope.utils.constant": mock.MagicMock(
                        Tasks=mock.MagicMock(speaker_verification="sv")
                    ),
                },
            ),
            mock.patch(
                "app.services.speaker.providers.campplus_embedding_provider.resolve_model_path",
                return_value="/tmp/fake-model",
            ),
            mock.patch(
                "app.services.speaker.providers.campplus_embedding_provider.detect_device",
                side_effect=lambda configured: f"resolved:{configured}",
            ) as detect_device,
        ):
            CampplusEmbeddingProvider()._get_pipeline()

        detect_device.assert_called_once_with(settings.VOICEPRINT_DEVICE)


if __name__ == "__main__":
    unittest.main()
