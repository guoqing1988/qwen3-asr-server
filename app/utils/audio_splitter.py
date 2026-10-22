# -*- coding: utf-8 -*-
"""
音频分割模块
基于 VAD 的智能音频分割，支持长音频分段识别
"""

import logging
import numpy as np
import librosa
import soundfile as sf
import tempfile
import os
from itertools import pairwise
from typing import List, Tuple, Optional
from dataclasses import dataclass

from ..core.config import settings
from ..core.exceptions import DefaultServerErrorException

logger = logging.getLogger(__name__)


@dataclass
class AudioSegment:
    """音频片段信息"""

    start_ms: int  # 开始时间（毫秒）
    end_ms: int  # 结束时间（毫秒）
    audio_data: Optional[np.ndarray] = None  # 音频数据
    temp_file: Optional[str] = None  # 临时文件路径
    speaker_id: Optional[str] = None  # 说话人ID（多说话人模式）

    @property
    def start_sec(self) -> float:
        """开始时间（秒）"""
        return self.start_ms / 1000.0

    @property
    def end_sec(self) -> float:
        """结束时间（秒）"""
        return self.end_ms / 1000.0

    @property
    def duration_ms(self) -> int:
        """时长（毫秒）"""
        return self.end_ms - self.start_ms

    @property
    def duration_sec(self) -> float:
        """时长（秒）"""
        return self.duration_ms / 1000.0


class AudioSplitter:
    """音频分割器

    以 Nemotron 说话人活动区间为切分依据（VAD 已随实时栈移除），
    按有界策略切分长音频，并避免把超长静音并入语音段。
    """

    # 默认配置
    DEFAULT_MIN_SEGMENT_SEC = 1.0  # 每段最小时长（秒）
    DEFAULT_SAMPLE_RATE = 16000  # 默认采样率
    # 语音覆盖率低于该值且时长超过 COVERAGE_CHECK_MIN_SEC 时也切分：
    # 稀疏语音的整段（如 14s 语音 + 25s 静音）会被模型判成空输出
    MIN_SPEECH_COVERAGE = 0.5
    COVERAGE_CHECK_MIN_SEC = 15.0

    def __init__(
        self,
        min_segment_sec: float = DEFAULT_MIN_SEGMENT_SEC,
        device: str = "auto",
    ):
        """初始化音频分割器

        Args:
            min_segment_sec: 每段最小时长（秒）
            device: 计算设备（"cuda", "cpu", "auto"）
        """
        split_trigger_sec = settings.MAX_SEGMENT_SEC

        self.split_trigger_sec = split_trigger_sec
        self.min_segment_sec = min_segment_sec
        self.split_trigger_ms = int(split_trigger_sec * 1000)
        self.min_segment_ms = int(min_segment_sec * 1000)
        self.max_silence_gap_ms = int(settings.MAX_SILENCE_GAP_SEC * 1000)
        self.device = device

    @staticmethod
    def speech_coverage(
        speech_segments: List[Tuple[int, int]],
        total_duration_ms: int,
    ) -> float:
        """语音活动占总时长的比例（0~1）；区间重叠不重复计算。"""
        if total_duration_ms <= 0 or not speech_segments:
            return 0.0
        covered = 0
        cursor = 0
        for start, end in sorted(
            (max(0, int(s)), min(total_duration_ms, int(e))) for s, e in speech_segments
        ):
            start = max(start, cursor)
            if end > start:
                covered += end - start
            cursor = max(cursor, end)
        return min(1.0, covered / total_duration_ms)

    def find_long_gap(
        self,
        speech_segments: List[Tuple[int, int]],
        total_duration_ms: int,
    ) -> Tuple[int, int] | None:
        """返回第一处超过阈值的静音间隙（毫秒），没有则返回 None。

        间隙包含首段之前与尾段之后：这两处同样可能把长静音并进语音段。
        """
        if not speech_segments:
            return None
        spans = sorted(
            (max(0, int(start)), min(total_duration_ms, int(end)))
            for start, end in speech_segments
        )
        cursor = 0
        for start, end in spans:
            if start - cursor > self.max_silence_gap_ms:
                return (cursor, start)
            cursor = max(cursor, end)
        if total_duration_ms - cursor > self.max_silence_gap_ms:
            return (cursor, total_duration_ms)
        return None

    def merge_segments_greedy(
        self, speech_segments: List[Tuple[int, int]], total_duration_ms: int
    ) -> List[Tuple[int, int]]:
        """按语音活动区间边界切分整条时间轴

        策略：
        1. 活动区间边界只作为切点；区间之间的间隙同样作为独立片段送识别，
           因为未检测到说话人活动不等于静音（低音量语音常落在间隙里）
        2. 不把分离的语音区间打包进同一请求，段落拼接在识别之后进行
        3. 仅对短片段（< min_segment_ms）做邻段合并，超长片段按最大时长切分
        4. 间隙本身已是独立片段：超过 MAX_SILENCE_GAP_SEC 的静音不会与语音段合并

        Args:
            speech_segments: 已检测到的语音段列表 [(start_ms, end_ms), ...]
            total_duration_ms: 音频总时长（毫秒）

        Returns:
            覆盖 [0, total_duration_ms] 的连续段列表 [(start_ms, end_ms), ...]
        """
        edges = {
            min(total_duration_ms, max(0, int(edge)))
            for span in speech_segments
            for edge in span
        }
        merged = list(pairwise(sorted(edges | {0, total_duration_ms})))

        # 只处理短片段：与相邻片段合并（不基于静音间隙）
        idx = 0
        while idx < len(merged):
            start_ms, end_ms = merged[idx]
            duration = end_ms - start_ms

            if duration >= self.min_segment_ms or len(merged) == 1:
                idx += 1
                continue

            if idx == 0:
                # 首段过短：并入后段
                next_end = merged[idx + 1][1]
                merged[idx + 1] = (start_ms, next_end)
                del merged[idx]
                continue

            if idx == len(merged) - 1:
                # 尾段过短：并入前段
                prev_start, _ = merged[idx - 1]
                merged[idx - 1] = (prev_start, end_ms)
                del merged[idx]
                idx = max(0, idx - 1)
                continue

            # 中间短段：优先并入时长更短的一侧，避免单段过长
            prev_start = merged[idx - 1][0]
            next_end = merged[idx + 1][1]
            merged_with_prev_duration = end_ms - prev_start
            merged_with_next_duration = next_end - start_ms

            if merged_with_prev_duration <= merged_with_next_duration:
                merged[idx - 1] = (prev_start, end_ms)
                del merged[idx]
                idx = max(0, idx - 1)
            else:
                merged[idx + 1] = (start_ms, next_end)
                del merged[idx]

        return [
            (start_ms + start, start_ms + end)
            for start_ms, end_ms in merged
            for start, end in self._split_by_fixed_duration(end_ms - start_ms)
        ]

    def _split_by_fixed_duration(self, total_duration_ms: int) -> List[Tuple[int, int]]:
        """按固定时长切分（无 VAD 时的 fallback）

        Args:
            total_duration_ms: 音频总时长（毫秒）

        Returns:
            切分后的段列表
        """
        if self.split_trigger_ms < 1:
            raise ValueError("Maximum segment duration must be at least one millisecond")
        segments = []
        current = 0
        # 末尾残段并入前一段，避免短尾音被丢弃（丢尾音会让最后一句识别缺失）
        min_tail_ms = min(self.min_segment_ms, self.split_trigger_ms)
        while current < total_duration_ms:
            end = min(current + self.split_trigger_ms, total_duration_ms)
            remaining = total_duration_ms - end
            if 0 < remaining < min_tail_ms:
                end = total_duration_ms - min_tail_ms
            segments.append((current, end))
            current = end
        return segments

    def split_audio_file(
        self,
        audio_path: str,
        output_dir: Optional[str] = None,
        *,
        speech_segments: List[Tuple[int, int]],
    ) -> List[AudioSegment]:
        """分割音频文件

        Args:
            audio_path: 音频文件路径
            output_dir: 输出目录（可选，默认使用临时目录）
            speech_segments: Nemotron 说话人活动区间（毫秒，必填）；只作切分提示，
                输出片段覆盖完整音频

        Returns:
            音频片段列表
        """
        try:
            # 加载音频
            audio_data, sr = librosa.load(audio_path, sr=self.DEFAULT_SAMPLE_RATE)
            total_duration_ms = (len(audio_data) * 1000 + int(sr) - 1) // int(sr)

            logger.info(f"音频总时长: {total_duration_ms / 1000:.2f}秒")

            speech_spans = list(speech_segments or [])

            # 时长在限制内、且没有超长静音间隙时才整段返回；长静音必须切段，
            # 否则整段会被静音拖成空输出（模型对「少量语音 + 大量静音」长段输出空）
            long_gap = self.find_long_gap(speech_spans, total_duration_ms)
            # 稀疏语音：时长不短但活动占比很小，整段送 ASR 会被判空（静音占主导）
            sparse = (
                total_duration_ms > self.COVERAGE_CHECK_MIN_SEC * 1000
                and self.speech_coverage(speech_spans, total_duration_ms)
                < self.MIN_SPEECH_COVERAGE
            )
            if (
                len(audio_data) * 1000 <= self.split_trigger_ms * int(sr)
                and long_gap is None
                and not sparse
            ):
                logger.info("音频时长在限制内且无长静音，无需分割")
                return [
                    AudioSegment(
                        start_ms=0,
                        end_ms=total_duration_ms,
                        audio_data=audio_data,
                        temp_file=audio_path,
                    )
                ]
            if sparse and long_gap is None:
                logger.info(
                    "语音覆盖率 %.0f%% 偏低（阈值 %.0f%%），按语音区间切分",
                    self.speech_coverage(speech_spans, total_duration_ms) * 100,
                    self.MIN_SPEECH_COVERAGE * 100,
                )
            if long_gap is not None:
                logger.info(
                    "检测到长静音间隙 %d-%d ms（阈值 %.1fs），按语音区间切分",
                    long_gap[0],
                    long_gap[1],
                    self.max_silence_gap_ms / 1000,
                )

            merged_segments = (
                self.merge_segments_greedy(speech_spans, total_duration_ms)
                if speech_spans
                else self._split_by_fixed_duration(total_duration_ms)
            )
            logger.info(
                f"重分段完成: 原始语音区间={len(speech_spans)}, 输出={len(merged_segments)}"
            )

            # 切分音频并保存到临时文件
            logger.info("开始切分音频并保存临时文件...")
            output_dir = output_dir or settings.TEMP_DIR
            os.makedirs(output_dir, exist_ok=True)

            audio_segments = []
            for idx, (start_ms, end_ms) in enumerate(merged_segments):
                # 计算采样点范围
                start_sample = start_ms * int(sr) // 1000
                end_sample = end_ms * int(sr) // 1000

                # 提取音频片段
                segment_data = audio_data[start_sample:end_sample]

                # 保存到临时文件
                temp_file = tempfile.NamedTemporaryFile(
                    delete=False,
                    suffix=".wav",
                    dir=output_dir,
                    prefix=f"segment_{idx:03d}_",
                )
                temp_path = temp_file.name
                temp_file.close()

                sf.write(temp_path, segment_data, sr)

                segment = AudioSegment(
                    start_ms=start_ms,
                    end_ms=end_ms,
                    audio_data=segment_data,
                    temp_file=temp_path,
                )
                audio_segments.append(segment)

                logger.debug(
                    f"分段 {idx + 1}/{len(merged_segments)}: "
                    f"{start_ms / 1000:.2f}s - {end_ms / 1000:.2f}s "
                    f"(时长: {segment.duration_sec:.2f}s)"
                )

            logger.info(f"音频切分完成，共 {len(audio_segments)} 个分段")
            return audio_segments

        except Exception as e:
            logger.error(f"音频分割失败: {e}")
            raise DefaultServerErrorException(f"音频分割失败: {str(e)}")

    @staticmethod
    def cleanup_segments(segments: List[AudioSegment]) -> None:
        """清理临时文件

        Args:
            segments: 音频片段列表
        """
        for segment in segments:
            if segment.temp_file and os.path.exists(segment.temp_file):
                try:
                    os.remove(segment.temp_file)
                except Exception as e:
                    logger.warning(f"清理临时文件失败: {segment.temp_file}, {e}")


def split_long_audio(
    audio_path: str,
    device: str = "auto",
) -> List[AudioSegment]:
    """分割长音频的便捷函数

    Args:
        audio_path: 音频文件路径
        device: 计算设备

    Returns:
        音频片段列表
    """
    splitter = AudioSplitter(device=device)
    return splitter.split_audio_file(audio_path)
