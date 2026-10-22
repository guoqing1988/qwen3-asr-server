# -*- coding: utf-8 -*-
"""Single-source deployment model planning."""

from __future__ import annotations

import json
import os
import platform
from pathlib import Path
from typing import Optional

from app.core.config import settings

QWEN_MODEL_OVERRIDE_ENV = "QWEN3_ASR_MODEL"
_QWEN_MODEL_ALIASES = {
    "qwen3-asr-0.6b": "qwen3-asr-0.6b",
    "0.6b": "qwen3-asr-0.6b",
    "0.6": "qwen3-asr-0.6b",
    "qwen/qwen3-asr-0.6b": "qwen3-asr-0.6b",
    "qwen3-asr-confucius4-r2t2": "qwen3-asr-confucius4-r2t2",
    "confucius4-r2t2": "qwen3-asr-confucius4-r2t2",
    "confucius4": "qwen3-asr-confucius4-r2t2",
    "r2t2": "qwen3-asr-confucius4-r2t2",
    "netease-youdao/confucius4-r2t2": "qwen3-asr-confucius4-r2t2",
}


def _load_declared_default_qwen_model() -> Optional[str]:
    """读取 models.json 中显式标记 ``default: true`` 的 qwen3 引擎模型 id。

    使默认模型的选择由配置驱动，而非代码内硬编码的模型名。
    """
    models_file = Path(settings.models_config_path)
    if not models_file.exists():
        return None
    try:
        with open(models_file, "r", encoding="utf-8") as f:
            config = json.load(f)
    except (json.JSONDecodeError, OSError):
        return None

    for model_id, model_config in config.get("models", {}).items():
        if not isinstance(model_config, dict):
            continue
        if str(model_config.get("engine", "")).lower() != "qwen3":
            continue
        if model_config.get("default") is True:
            return model_id
    return None


def load_supported_model_ids() -> list[str]:
    """Load declared model ids from models.json."""
    models_file = Path(settings.models_config_path)
    if not models_file.exists():
        return []

    with open(models_file, "r", encoding="utf-8") as f:
        config = json.load(f)

    return list(config.get("models", {}).keys())


def get_qwen_model_override() -> Optional[str]:
    """Return the explicit Qwen model override from the environment."""
    raw_value = (os.getenv(QWEN_MODEL_OVERRIDE_ENV) or "").strip()
    if not raw_value:
        return None

    normalized = _QWEN_MODEL_ALIASES.get(raw_value.lower(), raw_value)
    return normalized


def detect_qwen_model_by_vram(all_model_ids: Optional[list[str]] = None) -> Optional[str]:
    """Pick the active Qwen model for the current machine."""
    from app.core.device import detect_device, get_vram_gb

    model_ids = all_model_ids or load_supported_model_ids()
    override_model = get_qwen_model_override()
    if override_model:
        return override_model if override_model in model_ids else None

    # models.json 显式声明的默认模型优先（配置驱动）
    declared_default = _load_declared_default_qwen_model()
    if declared_default and declared_default in model_ids:
        return declared_default

    resolved_device = detect_device(settings.DEVICE)

    # macOS defaults to the lighter Rust CPU path unless QWEN3_ASR_MODEL is set.
    if platform.system() == "Darwin":
        return _pick_lightweight_model(model_ids)

    if resolved_device == "cpu":
        return _pick_lightweight_model(model_ids)

    vram = get_vram_gb()
    preferred = _pick_large_model(model_ids) if vram >= 32 else _pick_lightweight_model(model_ids)
    if preferred:
        return preferred
    return model_ids[0] if model_ids else None


def _pick_lightweight_model(model_ids: list[str]) -> Optional[str]:
    """在候选模型中优先选择轻量版本（命名含 0.6b），用于低显存/CPU 环境。"""
    for model_id in model_ids:
        if "0.6b" in model_id.lower():
            return model_id
    return model_ids[0] if model_ids else None


def _pick_large_model(model_ids: list[str]) -> Optional[str]:
    """在候选模型中优先选择非轻量版本，用于大显存环境。"""
    for model_id in model_ids:
        if "0.6b" not in model_id.lower():
            return model_id
    return model_ids[0] if model_ids else None


def get_active_qwen_model(all_model_ids: Optional[list[str]] = None) -> str:
    """Return the required Qwen model for the current machine."""
    model_ids = all_model_ids or load_supported_model_ids()
    qwen_model = detect_qwen_model_by_vram(model_ids)
    if not qwen_model:
        override_model = get_qwen_model_override()
        if override_model:
            available_qwen_models = ", ".join(
                model_id for model_id in model_ids if model_id.startswith("qwen")
            )
            raise RuntimeError(
                f"{QWEN_MODEL_OVERRIDE_ENV}={override_model} 不在可用 Qwen3-ASR 模型中: "
                f"{available_qwen_models}"
            )
        raise RuntimeError("当前环境未找到可运行的 Qwen3-ASR 模型")
    return qwen_model


def get_runtime_model_ids(all_model_ids: Optional[list[str]] = None) -> list[str]:
    """Return the runtime model/capability plan for the current machine."""
    model_ids = all_model_ids or load_supported_model_ids()
    return [get_active_qwen_model(model_ids)]


def get_default_model_id(all_model_ids: Optional[list[str]] = None) -> str:
    """Return the single default offline model for API/UI selection."""
    return get_active_qwen_model(all_model_ids)
