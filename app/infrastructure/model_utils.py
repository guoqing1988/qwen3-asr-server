# -*- coding: utf-8 -*-
"""Shared model path resolution helpers."""

import logging
import os
import re
from pathlib import Path
from typing import Any, Optional

from app.core.config import settings

logger = logging.getLogger(__name__)

# 支持 ${VAR} 与 ${VAR:-default} 两种占位符，供 models.json 引用部署环境变量
_ENV_PLACEHOLDER_PATTERN = re.compile(
    r"\$\{([A-Za-z_][A-Za-z0-9_]*)(?::-(.*?))?\}"
)


def expand_env_placeholders(value: str) -> str:
    """展开字符串中的 ``${VAR}`` / ``${VAR:-default}`` 占位符。

    - 环境变量已设置且非空 → 取环境变量值
    - 未设置 → 取 ``:-`` 后的默认值
    - 未设置且无默认值 → 保留原占位符（避免静默产生空路径）
    """
    if not isinstance(value, str) or "${" not in value:
        return value

    def _replace(match: re.Match[str]) -> str:
        name, default = match.group(1), match.group(2)
        env_value = os.getenv(name)
        if env_value:
            return env_value
        if default is not None:
            return default
        return match.group(0)

    return _ENV_PLACEHOLDER_PATTERN.sub(_replace, value)


def expand_env_placeholders_in_mapping(
    mapping: dict[str, Any],
) -> dict[str, Any]:
    """递归展开映射中所有字符串值的环境变量占位符。"""
    expanded: dict[str, Any] = {}
    for key, value in mapping.items():
        if isinstance(value, str):
            expanded[key] = expand_env_placeholders(value)
        elif isinstance(value, dict):
            expanded[key] = expand_env_placeholders_in_mapping(value)
        else:
            expanded[key] = value
    return expanded


def resolve_local_model_dir(model_ref: str) -> Path | None:
    """若 ``model_ref`` 指向已存在的本地目录则返回其绝对路径，否则返回 None。

    用于区分「本地路径模型」与「HF/ModelScope 模型引用」：
    完整性检查、下载流程对两者的目录结构假设不同。
    """
    if not model_ref:
        return None
    candidate = Path(os.path.expandvars(model_ref)).expanduser()
    if candidate.is_dir():
        return candidate.resolve()
    return None


def _is_truthy(value: str | None) -> bool:
    return (value or "").strip().lower() in {"1", "true", "yes", "on"}


def _expand_path(value: str) -> Path:
    return Path(os.path.expandvars(value)).expanduser()


def is_huggingface_offline() -> bool:
    return _is_truthy(os.getenv("HF_HUB_OFFLINE"))


def get_huggingface_cache_root() -> Path:
    hf_hub_cache = (os.getenv("HF_HUB_CACHE") or "").strip()
    if hf_hub_cache:
        return _expand_path(hf_hub_cache)

    hf_home = (os.getenv("HF_HOME") or "").strip()
    if hf_home:
        return _expand_path(hf_home) / "hub"

    xdg_cache_home = (os.getenv("XDG_CACHE_HOME") or "").strip()
    if xdg_cache_home:
        return _expand_path(xdg_cache_home) / "huggingface" / "hub"

    return Path.home() / ".cache" / "huggingface" / "hub"


def get_huggingface_model_cache_dir(model_id: str) -> Path:
    org, model = model_id.split("/", 1)
    return get_huggingface_cache_root() / f"models--{org}--{model}"


def find_huggingface_snapshot_dir(model_ref_or_path: str) -> Optional[Path]:
    raw_path = Path(model_ref_or_path).expanduser()
    if raw_path.exists():
        return raw_path.resolve()
    if "/" not in model_ref_or_path:
        return None

    base_dir = get_huggingface_model_cache_dir(model_ref_or_path)
    if not base_dir.exists():
        return None

    ref_main = base_dir / "refs" / "main"
    if ref_main.exists():
        snapshot_name = ref_main.read_text(encoding="utf-8").strip()
        snapshot_dir = base_dir / "snapshots" / snapshot_name
        if snapshot_dir.exists():
            return snapshot_dir.resolve()

    snapshots_dir = base_dir / "snapshots"
    if not snapshots_dir.exists():
        return None

    snapshots = [path for path in snapshots_dir.iterdir() if path.is_dir()]
    snapshots.sort(key=lambda path: path.stat().st_mtime, reverse=True)
    return snapshots[0].resolve() if snapshots else None


def resolve_huggingface_snapshot_dir(model_ref_or_path: str) -> Path:
    snapshot_dir = find_huggingface_snapshot_dir(model_ref_or_path)
    if snapshot_dir is not None:
        return snapshot_dir

    raise FileNotFoundError(
        f"Hugging Face model path not found for '{model_ref_or_path}'. "
        f"Checked direct path and cache root: {get_huggingface_cache_root()}."
    )


def resolve_model_path(model_id: Optional[str]) -> str:
    if not model_id:
        raise ValueError("model_id is required")

    local_path = Path(settings.MODELSCOPE_PATH) / model_id

    if local_path.exists() and local_path.is_dir():
        resolved = str(local_path)
        logger.info("Using local ModelScope cache for %s: %s", model_id, resolved)
        return resolved

    logger.warning("ModelScope cache missing for %s; runtime may download it", model_id)
    return model_id
