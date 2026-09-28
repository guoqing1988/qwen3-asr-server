"""Speaker embedding extraction and aggregation."""

from __future__ import annotations

import logging
from collections import defaultdict
from typing import Protocol

import librosa
import numpy as np

from app.services.asr.engines import ASRFullResult

from .config import (
    VOICEPRINT_MAX_CHUNK_SEC,
    VOICEPRINT_MIN_CHUNK_SEC,
    VOICEPRINT_MIN_SPEECH_SEC,
)
from .domain import SpeakerEmbedding

logger = logging.getLogger(__name__)


class SpeakerEmbeddingProvider(Protocol):
    provider_name: str
    model_name: str
    sample_rate: int

    def extract(self, audio_data: np.ndarray) -> SpeakerEmbedding:
        ...


class SpeakerEmbeddingService:
    """Extract one centroid speaker embedding per local diarization label."""

    def __init__(self, provider: SpeakerEmbeddingProvider) -> None:
        self._provider = provider

    def extract_local_speaker_embeddings(
        self,
        *,
        audio_path: str,
        asr_result: ASRFullResult,
        timestamp_scale: float,
    ) -> dict[str, SpeakerEmbedding]:
        grouped_ranges = self._group_speaker_ranges(
            asr_result=asr_result,
            timestamp_scale=timestamp_scale,
        )
        if not grouped_ranges:
            return {}

        audio_data, loaded_sample_rate = librosa.load(
            audio_path,
            sr=self._provider.sample_rate,
        )
        sample_rate = int(loaded_sample_rate)
        if sample_rate <= 0 or audio_data.size == 0:
            return {}

        embeddings: dict[str, SpeakerEmbedding] = {}
        for speaker_id, ranges in grouped_ranges.items():
            duration_sec = self._speech_seconds(
                audio_data=audio_data,
                ranges=ranges,
                sample_rate=sample_rate,
            )
            if duration_sec < VOICEPRINT_MIN_SPEECH_SEC:
                logger.info(
                    "Skip voiceprint matching for %s: speech %.2fs < %.2fs",
                    speaker_id,
                    duration_sec,
                    VOICEPRINT_MIN_SPEECH_SEC,
                )
                continue

            embedding = self._extract_segment_embeddings(
                audio_data=audio_data,
                ranges=ranges,
                sample_rate=sample_rate,
                duration_sec=duration_sec,
            )
            if embedding is None:
                continue

            embeddings[speaker_id] = SpeakerEmbedding(
                vector=self._normalize(embedding.vector),
                provider=embedding.provider,
                model_name=embedding.model_name,
                sample_rate=sample_rate,
                duration_sec=duration_sec,
            )

        self._release_gpu_cache()
        return embeddings

    @staticmethod
    def _release_gpu_cache() -> None:
        """归还 CAM++ 在 GPU 上留下的缓存块。

        torch 的 caching allocator 不会自动释放 reserved（单次 30s 音频峰值约
        840MiB），空闲时会白占显存；用完即还，代价只是下次提取重新分配。
        """
        try:
            import torch
        except ImportError:  # pragma: no cover - 无 torch 环境（纯 CPU 部署）
            return
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    @staticmethod
    def _speech_seconds(
        *,
        audio_data: np.ndarray,
        ranges: list[tuple[float, float]],
        sample_rate: int,
    ) -> float:
        """该说话人所有分段的有效语音总时长（秒），超出音频范围的部分被截断。"""
        if sample_rate <= 0:
            return 0.0
        total_samples = 0
        for start_sec, end_sec in ranges:
            start_sample = max(0, int(start_sec * sample_rate))
            end_sample = min(audio_data.shape[0], int(end_sec * sample_rate))
            if end_sample > start_sample:
                total_samples += end_sample - start_sample
        return float(total_samples / sample_rate)

    def _extract_segment_embeddings(
        self,
        *,
        audio_data: np.ndarray,
        ranges: list[tuple[float, float]],
        sample_rate: int,
        duration_sec: float,
    ) -> SpeakerEmbedding | None:
        """按 ASR 分段逐段提取 embedding，再按时长加权融合成一个说话人向量。

        用识别出的分段作为提取单位：每段通常只有几秒，符合 CAM++ 的短时假设，
        不会像把数小时音频拼接后整体送入那样让 TDNN 中间激活爆显存（实测单次
        尝试分配 1.92GiB 后 CUDA OOM）；分段边界本身就是语义边界，比人工切片
        更自然。

        过短的分段（< ``VOICEPRINT_MIN_CHUNK_SEC``）的 embedding 不可靠，会跳过；
        若该说话人所有分段都过短，则退化为拼接后按上限切分提取，避免整人丢失。
        """
        max_samples = max(1, int(VOICEPRINT_MAX_CHUNK_SEC * sample_rate))
        min_samples = int(VOICEPRINT_MIN_CHUNK_SEC * sample_rate)

        vectors: list[np.ndarray] = []
        weights: list[float] = []
        template: SpeakerEmbedding | None = None
        skipped: list[np.ndarray] = []
        total_samples = 0

        for start_sec, end_sec in ranges:
            start_sample = max(0, int(start_sec * sample_rate))
            end_sample = min(audio_data.shape[0], int(end_sec * sample_rate))
            if end_sample <= start_sample:
                continue
            segment = np.asarray(audio_data[start_sample:end_sample], dtype=np.float32)
            total_samples += int(segment.size)
            # 单段超长时再细分，保证每片不超上限
            for offset in range(0, segment.size, max_samples):
                chunk = segment[offset : offset + max_samples]
                if chunk.size < min_samples:
                    skipped.append(chunk)
                    continue
                chunk_embedding = self._provider.extract(chunk)
                if template is None:
                    template = chunk_embedding
                vectors.append(
                    np.asarray(chunk_embedding.vector, dtype=np.float32).reshape(-1)
                )
                weights.append(float(chunk.size))

        if total_samples == 0:
            return None

        if not vectors:
            # 所有分段都过短：退化为拼接后提取（短段拼起来通常也不长）
            fallback = np.concatenate(skipped).astype(np.float32, copy=False)
            return self._extract_embedding_for_audio(fallback, sample_rate=sample_rate)

        weight_array = np.asarray(weights, dtype=np.float32)
        weight_array = weight_array / float(weight_array.sum())
        fused = (np.stack(vectors, axis=0) * weight_array[:, None]).sum(axis=0)
        assert template is not None
        return SpeakerEmbedding(
            vector=fused,
            provider=template.provider,
            model_name=template.model_name,
            sample_rate=template.sample_rate,
            duration_sec=duration_sec,
        )

    def extract_reference_embedding(self, *, audio_path: str) -> SpeakerEmbedding:
        audio_data, loaded_sample_rate = librosa.load(
            audio_path,
            sr=self._provider.sample_rate,
        )
        sample_rate = int(loaded_sample_rate)
        embedding = self._extract_embedding_for_audio(
            audio_data,
            sample_rate=sample_rate,
        )
        self._release_gpu_cache()
        return SpeakerEmbedding(
            vector=self._normalize(embedding.vector),
            provider=embedding.provider,
            model_name=embedding.model_name,
            sample_rate=sample_rate,
            duration_sec=float(audio_data.shape[0] / sample_rate) if sample_rate else 0.0,
        )

    def _extract_embedding_for_audio(
        self,
        audio: np.ndarray,
        *,
        sample_rate: int,
    ) -> SpeakerEmbedding:
        """提取音频的说话人 embedding；长音频按时长加权融合多段结果。

        CAM++ 期望短时输入：把数小时拼接音频整体送入会让 TDNN 中间激活爆显存
        （实测单次尝试分配 1.92GiB 后 CUDA OOM），且数小时拼接出的单一 embedding
        本身也不具代表性。这里按 ``VOICEPRINT_MAX_CHUNK_SEC`` 切片分别提取，
        再按各片时长加权平均。
        """
        audio = np.asarray(audio, dtype=np.float32).reshape(-1)
        rate = sample_rate if sample_rate > 0 else self._provider.sample_rate
        max_samples = max(1, int(VOICEPRINT_MAX_CHUNK_SEC * rate))
        if audio.size <= max_samples:
            return self._provider.extract(audio)

        min_samples = int(VOICEPRINT_MIN_CHUNK_SEC * rate)
        vectors: list[np.ndarray] = []
        weights: list[float] = []
        template: SpeakerEmbedding | None = None
        for start in range(0, audio.size, max_samples):
            chunk = audio[start : start + max_samples]
            if chunk.size < min_samples:
                continue  # 尾片太短，提取出的 embedding 不可靠
            chunk_embedding = self._provider.extract(chunk)
            if template is None:
                template = chunk_embedding
            vectors.append(
                np.asarray(chunk_embedding.vector, dtype=np.float32).reshape(-1)
            )
            weights.append(float(chunk.size))

        if template is None:
            return self._provider.extract(audio)

        weight_array = np.asarray(weights, dtype=np.float32)
        weight_array = weight_array / float(weight_array.sum())
        fused = (np.stack(vectors, axis=0) * weight_array[:, None]).sum(axis=0)
        return SpeakerEmbedding(
            vector=fused,
            provider=template.provider,
            model_name=template.model_name,
            sample_rate=template.sample_rate,
            duration_sec=float(audio.size / rate),
        )

    @staticmethod
    def _group_speaker_ranges(
        *,
        asr_result: ASRFullResult,
        timestamp_scale: float,
    ) -> dict[str, list[tuple[float, float]]]:
        scale = timestamp_scale if timestamp_scale > 0 else 1.0
        ranges: dict[str, list[tuple[float, float]]] = defaultdict(list)
        for segment in asr_result.segments:
            if not segment.speaker_id:
                continue
            start_sec = max(0.0, float(segment.start_time) / scale)
            end_sec = max(start_sec, float(segment.end_time) / scale)
            if end_sec > start_sec:
                ranges[segment.speaker_id].append((start_sec, end_sec))
        return dict(ranges)

    @staticmethod
    def _normalize(vector: np.ndarray) -> np.ndarray:
        value = np.asarray(vector, dtype=np.float32).reshape(-1)
        norm = float(np.linalg.norm(value))
        if norm <= 0:
            return value
        return value / norm
