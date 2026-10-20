# -*- coding: utf-8 -*-
"""Nemotron-3 说话人分离（帧级、支持重叠，替代 CAM++ 自动聚类）。

模型：`nvidia/Nemotron-3-Diarization`（transformers 指定 commit，8 说话人上限）。
输出原始时间轴上的说话人区间（可重叠）与逐帧概率；`speech_intervals_ms()` 把
所有说话人活动并集作为离线切分提示（只作切分提示，不丢弃未覆盖音频）。
"""

import logging
import math
import threading
from dataclasses import dataclass
from typing import TYPE_CHECKING

import librosa
import numpy as np
import torch

from ..core.config import settings
from ..core.device import detect_device
from ..core.exceptions import DefaultServerErrorException

if TYPE_CHECKING:
    from transformers import (
        Nemotron3DiarizationForAudioFrameClassification,
        Nemotron3DiarizationProcessor,
    )

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class SpeakerSegment:
    start_sec: float
    end_sec: float
    speaker_id: str
    confidence: float


@dataclass(frozen=True)
class DiarizationResult:
    segments: list[SpeakerSegment]
    probabilities: np.ndarray
    frame_seconds: float
    duration: float
    speaker_ids: tuple[str | None, ...]

    def speech_intervals_ms(self) -> list[tuple[int, int]]:
        """把所有说话人活动并集成语音区间（毫秒），供离线切分使用。

        区间之间没有活动不代表静音，切分器仍按有界策略保留音频。
        """
        merged: list[list[int]] = []
        for span in sorted(self.segments, key=lambda span: span.start_sec):
            start = max(0, round(span.start_sec * 1000))
            end = round(span.end_sec * 1000)
            if end <= start:
                continue
            if merged and start <= merged[-1][1]:
                merged[-1][1] = max(merged[-1][1], end)
            else:
                merged.append([start, end])
        return [(start, end) for start, end in merged]


class SpeakerDiarizer:
    SAMPLE_RATE = 16000
    MAX_SPEAKERS = 8

    def __init__(self) -> None:
        self._model: Nemotron3DiarizationForAudioFrameClassification | None = None
        self._processor: Nemotron3DiarizationProcessor | None = None
        self._lock = threading.RLock()
        self._closed = False
        self._frame_seconds = 0.01

    def warmup(self) -> None:
        with self._lock:
            if self._closed:
                raise RuntimeError("Speaker diarizer is closed")
            if self._model is not None:
                return
            from transformers import AutoModelForAudioFrameClassification, AutoProcessor

            device = detect_device(settings.DEVICE)
            processor = AutoProcessor.from_pretrained(
                settings.NEMOTRON_MODEL_PATH, local_files_only=True
            )
            model = (
                AutoModelForAudioFrameClassification.from_pretrained(
                    settings.NEMOTRON_MODEL_PATH,
                    local_files_only=True,
                    dtype=torch.float32,
                )
                .to(device)
                .eval()
            )
            extractor = processor.feature_extractor
            if extractor.sampling_rate != self.SAMPLE_RATE:
                raise ValueError("Nemotron requires the 16 kHz feature extractor")
            if model.config.head_config.num_speakers != self.MAX_SPEAKERS:
                raise ValueError("Nemotron requires an eight-speaker checkpoint")
            frame_seconds = extractor.hop_length / extractor.sampling_rate
            if not math.isfinite(frame_seconds) or frame_seconds <= 0:
                raise ValueError("Invalid Nemotron frame duration")
            self._processor = processor
            self._model = model
            self._frame_seconds = frame_seconds
            logger.info(
                "Loaded Nemotron diarization: device=%s, frame=%.3fs", device, frame_seconds
            )

    def diarize(self, audio_path: str) -> DiarizationResult:
        try:
            with self._lock:
                if self._closed:
                    raise RuntimeError("Speaker diarizer is closed")
                audio, sample_rate = librosa.load(audio_path, sr=self.SAMPLE_RATE)
                if audio.ndim != 1 or not np.isfinite(audio).all():
                    raise ValueError("Diarization requires finite mono audio")
                duration = len(audio) / sample_rate
                if not audio.size:
                    return DiarizationResult(
                        [],
                        np.empty((0, self.MAX_SPEAKERS), dtype=np.float32),
                        self._frame_seconds,
                        duration,
                        (None,) * self.MAX_SPEAKERS,
                    )
                self.warmup()
                model, processor = self._model, self._processor
                assert model is not None and processor is not None
                inputs = processor(audio, sampling_rate=sample_rate).to(
                    model.device, dtype=model.dtype
                )
                # 不传 cache：离线 forward 自己维护整段录音的局部状态
                with torch.inference_mode():
                    logits = model(**inputs).logits
                    if (
                        logits.ndim != 3
                        or logits.shape[0] != 1
                        or logits.shape[2] != self.MAX_SPEAKERS
                        or inputs.attention_mask.shape != logits.shape[:2]
                        or not torch.isfinite(logits).all()
                    ):
                        raise ValueError(
                            "Invalid Nemotron frame logits or attention mask"
                        )
                    raw_segments = processor.extract_speaker_dict(
                        logits, inputs.attention_mask
                    )[0]
                    probabilities = logits.sigmoid()[0]
                    probabilities = (
                        probabilities.masked_fill(
                            ~inputs.attention_mask[0].bool()[:, None], 0
                        )
                        .float()
                        .cpu()
                        .numpy()
                    )
                return self._make_result(raw_segments, probabilities, duration)
        except Exception as exc:  # noqa: BLE001 - 统一包装为服务异常
            logger.exception("Nemotron diarization failed")
            raise DefaultServerErrorException(
                f"Speaker diarization failed: {exc}"
            ) from exc

    def _make_result(
        self,
        raw_segments: list[dict[str, float | int]],
        probabilities: np.ndarray,
        duration: float,
    ) -> DiarizationResult:
        if (
            probabilities.ndim != 2
            or probabilities.shape[1] != self.MAX_SPEAKERS
            or not np.isfinite(probabilities).all()
            or np.any((probabilities < 0) | (probabilities > 1))
        ):
            raise ValueError("Invalid Nemotron speaker probabilities")
        intervals: list[tuple[float, float, int]] = []
        for segment in raw_segments:
            start, end = float(segment["Start"]), float(segment["End"])
            column = int(segment["Speaker"])
            if (
                not math.isfinite(start)
                or not math.isfinite(end)
                or start < 0
                or end <= start
                or column != segment["Speaker"]
                or not 0 <= column < self.MAX_SPEAKERS
            ):
                raise ValueError("Invalid Nemotron speaker interval")
            end = min(duration, end)
            if end > start:
                intervals.append((start, end, column))
        intervals.sort(key=lambda segment: (segment[0], segment[2], segment[1]))
        labels: dict[int, str] = {}
        segments: list[SpeakerSegment] = []
        for start, end, column in intervals:
            if column not in labels:
                labels[column] = f"说话人{len(labels) + 1}"
            first = round(start / self._frame_seconds)
            last = min(len(probabilities), round(end / self._frame_seconds))
            # 收尾帧可能不足一个完整 hop
            last = max(first + 1, last)
            if first >= len(probabilities):
                raise ValueError("Nemotron interval exceeds its probability timeline")
            confidence = float(probabilities[first:last, column].mean())
            segments.append(SpeakerSegment(start, end, labels[column], confidence))
        return DiarizationResult(
            segments,
            probabilities,
            self._frame_seconds,
            duration,
            tuple(labels.get(column) for column in range(self.MAX_SPEAKERS)),
        )

    def close(self) -> None:
        with self._lock:
            self._model = None
            self._processor = None
            self._closed = True


_global_diarizer: SpeakerDiarizer | None = None
_global_lock = threading.Lock()


def get_speaker_diarizer() -> SpeakerDiarizer:
    global _global_diarizer
    with _global_lock:
        if _global_diarizer is None:
            _global_diarizer = SpeakerDiarizer()
        return _global_diarizer


def close_speaker_diarizer() -> None:
    global _global_diarizer
    with _global_lock:
        if _global_diarizer is not None:
            _global_diarizer.close()
            _global_diarizer = None
