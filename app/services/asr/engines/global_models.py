# -*- coding: utf-8 -*-
"""全局标点模型管理（FunASR CT-Transformer，离线句末标点补齐）。

VAD 与实时标点随 Paraformer 实时栈移除：离线分段改用 Nemotron 语音区间。
"""

import logging
import threading
from funasr import AutoModel

from app.core.config import settings
from app.infrastructure import resolve_model_path


logger = logging.getLogger(__name__)


# 全局标点符号模型缓存（避免重复加载）
_global_punc_model = None
_punc_model_lock = threading.Lock()
_punc_inference_lock = threading.Lock()  # 推理互斥锁，防止并发状态混乱


def _resolve_device(device: str) -> str:
    """解析设备字符串，将 auto 转换为实际的设备"""
    from app.core.device import detect_device

    return detect_device(device)


def get_global_punc_model(device: str):
    """获取全局标点符号模型实例（离线版，线程安全，双重检查锁定）"""
    global _global_punc_model

    if _global_punc_model is None:
        with _punc_model_lock:
            if _global_punc_model is None:
                try:
                    # 解析模型路径：优先使用本地缓存
                    resolved_punc_path = resolve_model_path(settings.PUNC_MODEL)
                    logger.info(f"正在加载全局标点符号模型（离线）: {resolved_punc_path}")

                    # 解析 auto 设备
                    resolved_device = _resolve_device(device)

                    _global_punc_model = AutoModel(
                        model=resolved_punc_path,
                        device=resolved_device,
                        **settings.FUNASR_AUTOMODEL_KWARGS,
                    )
                    logger.info("全局标点符号模型（离线）加载成功")
                except Exception as e:
                    logger.error(f"全局标点符号模型（离线）加载失败: {str(e)}")
                    _global_punc_model = None
                    raise

    return _global_punc_model


def get_punc_inference_lock():
    """获取PUNC模型推理锁（线程安全）"""
    return _punc_inference_lock
