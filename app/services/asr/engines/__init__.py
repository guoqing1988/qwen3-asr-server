# -*- coding: utf-8 -*-
"""
ASR引擎模块
支持多种ASR引擎实现
"""

# 基础类和数据类
from .base import (
    BaseASREngine,
    RealTimeASREngine,
    WordToken,
    ASRSegmentResult,
    ASRFullResult,
    ASRRawResult,
)

# 全局模型管理（仅离线标点；VAD/实时标点已随 Paraformer 实时栈移除）
from .global_models import (
    get_global_punc_model,
    get_punc_inference_lock,
)

__all__ = [
    # 基础类
    "BaseASREngine",
    "RealTimeASREngine",
    # 数据类
    "WordToken",
    "ASRSegmentResult",
    "ASRFullResult",
    "ASRRawResult",
    # 全局模型管理
    "get_global_punc_model",
    "get_punc_inference_lock",
]
