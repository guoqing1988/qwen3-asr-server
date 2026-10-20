# -*- coding: utf-8 -*-
"""实时说话人标签：Nemotron 与 ASR 并行消费同一路 PCM，为每个「停顿边界的话」打标签。

R2T2 的流式 delta 不含词级时间，因此标签只挂在按停顿切分出的整段话上，
不会跟着最新文字漂移。标签比文本滞后约 Nemotron 的缓冲长度（~1s）。
diarization 失败只让标签为 `None`，绝不影响转写。
"""

import asyncio
import logging
from collections import deque
from typing import Any

import numpy as np
import torch

from ...utils.speaker_diarizer import SpeakerDiarizer, get_speaker_diarizer

logger = logging.getLogger(__name__)

SAMPLE_RATE = 16000
FRAME_SAMPLES = SAMPLE_RATE // 100  # Nemotron 每 10ms 输出一帧 logit
# activity 缓冲上限：1 小时（约 1.7MB/连接）。超出后超长会话的标签退化为 null，
# 不影响转写（实时会话极少超过 1 小时）。
_MAX_FRAMES = 60 * 60 * 1000 // 10 + 200


class SpeakerStream:
    """单条连接的 Nemotron 流式状态。

    每个登记的话最终得到一个标签；未知或 diarization 失败时为 `None`。
    """

    def __init__(self, names: dict[int, str] | None = None) -> None:
        self.model: Any = None
        self.processor: Any = None
        try:
            self.model, self.processor = get_speaker_diarizer().streaming_parts()
        except Exception:  # noqa: BLE001 - 模型缺失只降级标签
            logger.warning("实时说话人分离不可用，标签将为 null", exc_info=True)
        self.audio = np.empty(0, dtype=np.float32)
        self.offset = 0  # self.audio[0] 对应的绝对采样点
        self.samples = 0
        self.frames = 0
        self.activity = np.zeros(
            (_MAX_FRAMES, SpeakerDiarizer.MAX_SPEAKERS), dtype=bool
        )
        self.cache: Any = None
        self.ended = False
        self.finished = self.model is None
        self.pending: deque[tuple[int, int, int]] = deque()
        self.labels: dict[int, str] = {}
        self.resolved: set[int] = set()
        # 说话人列 → 显示名；段间重建流时继承，保证同一人编号不变
        self.names: dict[int, str] = dict(names or {})
        self.flushed = False

    # ------------------------------------------------------------------
    # 输入
    # ------------------------------------------------------------------

    def feed(self, pcm: bytes) -> None:
        """喂入 16k PCM16 字节流（与 ASR 同一路音频）。"""
        if self.finished or not pcm:
            return
        audio = np.frombuffer(pcm, dtype="<i2").astype(np.float32) / 32768
        self.audio = np.concatenate([self.audio, audio])
        self.samples += len(audio)

    def end(self) -> None:
        """标记音频结束，允许处理最后不足一个 chunk 的尾巴。"""
        self.ended = True

    def utterance(self, index: int, start_ms: int, end_ms: int) -> None:
        """登记一段待标注的话（毫秒，会话内绝对时间）。"""
        if end_ms > start_ms:
            self.pending.append((index, start_ms, end_ms))

    # ------------------------------------------------------------------
    # 推理
    # ------------------------------------------------------------------

    async def advance(self, *, flush: bool = False) -> list[dict]:
        """推进所有完整 chunk 的推理，返回音频已被覆盖的标签事件。

        `flush=True` 时把剩余不足一个 chunk 的音频也按最后一块推理（精度略降），
        用于客户端 commit 后不再发音频、标签否则永远等不到右上下文的场景。
        """
        if self.finished:
            return self.ready()
        if flush:
            self.flushed = True
        try:
            while (chunk := self._next_chunk(flush=flush)) is not None:
                active, self.cache = await asyncio.to_thread(self._infer, *chunk)
                self._accept(active, last=chunk[2])
        except Exception:  # noqa: BLE001 - 分离失败不影响转写
            logger.warning("实时说话人推理失败，标签置空", exc_info=True)
            self.finished, self.model = True, None
        return self.ready()

    def ready(self) -> list[dict]:
        """取出音频已被 Nemotron 覆盖的标签（不做推理）。"""
        events: list[dict] = []
        while self.pending and (
            self.finished or self.frames * FRAME_SAMPLES >= self._end_sample(self.pending[0])
        ):
            index, start, end = self.pending.popleft()
            events.append({"utterance": index, "speaker": self._label(start, end)})
        return events

    async def label(self, index: int, timeout: float) -> str | None:
        """等待指定话的标签；超时或失败返回 None。"""
        loop = asyncio.get_running_loop()
        deadline = loop.time() + max(0.0, timeout)
        while True:
            for event in await self.advance():
                resolved = int(event["utterance"])
                self.resolved.add(resolved)
                if event["speaker"] is not None:
                    self.labels[resolved] = str(event["speaker"])
            if index in self.resolved or self.finished:
                return self.labels.get(index)
            if loop.time() >= deadline:
                # 客户端停止发送后右上下文永远补不齐：用尾部音频收尾推理一次
                for event in await self.advance(flush=True):
                    resolved = int(event["utterance"])
                    self.resolved.add(resolved)
                    if event["speaker"] is not None:
                        self.labels[resolved] = str(event["speaker"])
                return self.labels.get(index)
            await asyncio.sleep(0.05)

    # ------------------------------------------------------------------
    # 内部
    # ------------------------------------------------------------------

    @staticmethod
    def _end_sample(item: tuple[int, int, int]) -> int:
        return item[2] * SAMPLE_RATE // 1000

    def _next_chunk(self, *, flush: bool = False) -> tuple[np.ndarray, bool, bool] | None:
        if self.finished:
            return None
        processor = self.processor
        first = self.frames == 0
        start = 0 if first else processor.audio_chunk_start(self.frames)
        size = (
            processor.num_samples_first_audio_chunk
            if first
            else processor.num_samples_per_audio_chunk
        )
        last = start + size > self.samples
        if last and not (self.ended or flush):
            return None
        chunk = self.audio[start - self.offset :]  # start >= offset 恒成立
        if last and len(chunk) < processor.feature_extractor.win_length:
            self.finished = True  # 比一个分析窗还短，无法推理
            return None
        return (chunk if last else chunk[:size]).copy(), first, last

    def _infer(self, audio: np.ndarray, first: bool, last: bool) -> tuple[np.ndarray, Any]:
        inputs = self.processor(
            audio,
            sampling_rate=SAMPLE_RATE,
            is_streaming=True,
            is_first_audio_chunk=first,
            is_last_audio_chunk=last,
        ).to(self.model.device, dtype=self.model.dtype)
        with torch.inference_mode():
            output = self.model(**inputs, speaker_cache=self.cache)
            active = (output.logits[0].sigmoid() > 0.5).cpu().numpy()
        return active, output.speaker_cache

    def _accept(self, active: np.ndarray, last: bool) -> None:
        end = min(len(self.activity), self.frames + len(active))
        self.activity[self.frames : end] = active[: end - self.frames]
        self.frames = end
        self.finished = last
        # 只保留 processor 还需要的历史音频（chunk cache 长度）
        keep = self.processor.audio_chunk_start(self.frames)
        if keep > self.offset:
            self.audio = self.audio[keep - self.offset :]
            self.offset = keep

    def _label(self, start_ms: int, end_ms: int) -> str | None:
        if self.model is None:
            return None
        start = start_ms * SAMPLE_RATE // 1000 // FRAME_SAMPLES
        end = end_ms * SAMPLE_RATE // 1000 // FRAME_SAMPLES
        counts = self.activity[start:end].sum(axis=0)
        if not counts.any():
            return None
        # 一段话内以主导说话人为准；无停顿换人时跟随占多数的一路
        column = int(counts.argmax())
        if column not in self.names:
            self.names[column] = f"说话人{len(self.names) + 1}"
        return self.names[column]

    def close(self) -> None:
        self.finished = True
        self.audio = np.empty(0, dtype=np.float32)
        self.activity = np.zeros((0, SpeakerDiarizer.MAX_SPEAKERS), dtype=bool)
        self.pending.clear()
