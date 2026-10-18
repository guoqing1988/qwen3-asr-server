# -*- coding: utf-8 -*-
"""流式语音端点检测（FireRedVAD Stream-VAD 封装）。

用于实时识别在自然句尾静音处落定当前段，替代历史上「固定时长强切 + 重叠去重」。
依赖或模型缺失时返回 None，调用方回退到能量阈值。
"""

from __future__ import annotations

import logging
import os

import numpy as np

from app.core.config import settings

logger = logging.getLogger(__name__)


class StreamVAD:
    """FireRedVAD 流式 VAD 的薄封装（单连接单实例，内部有状态）。"""

    def __init__(self, model_path: str, min_silence_frames: int = 20) -> None:
        from fireredvad import FireRedStreamVad, FireRedStreamVadConfig

        config = FireRedStreamVadConfig(
            use_gpu=False,
            speech_threshold=0.4,
            min_silence_frame=max(1, int(min_silence_frames)),
        )
        self._vad = FireRedStreamVad.from_pretrained(model_path, config)

    def reset(self) -> None:
        self._vad.reset()

    def detect_speech_end(self, pcm: np.ndarray) -> bool:
        """送入一段 16k float32 PCM，返回其中是否出现句尾静音。"""
        audio = np.asarray(pcm, dtype=np.float32)
        # FireRedVAD 按 25ms（400 样本）帧做 fbank，过短输入会得到 0 帧并抛错
        if audio.size < 400:
            return False
        int16 = (np.clip(audio, -1.0, 1.0) * 32768.0).astype(np.int16)
        results = self._vad.detect_chunk(int16)
        return any(bool(getattr(item, "is_speech_end", False)) for item in results)


def create_stream_vad() -> StreamVAD | None:
    """按配置创建 VAD；依赖或模型不可用时返回 None（回退能量阈值）。"""
    model_path = (settings.QWEN_STREAM_VAD_MODEL_PATH or "").strip()
    if not model_path or not os.path.isdir(model_path):
        logger.info(
            "Stream VAD model not found at %s, falling back to energy threshold",
            model_path,
        )
        return None
    try:
        vad = StreamVAD(
            model_path,
            min_silence_frames=settings.QWEN_STREAM_VAD_MIN_SILENCE_FRAMES,
        )
        logger.info("Stream VAD loaded: %s", model_path)
        return vad
    except Exception as exc:  # noqa: BLE001 - VAD 不可用不应阻断流式识别
        logger.warning("Failed to load stream VAD: %s", exc)
        return None
