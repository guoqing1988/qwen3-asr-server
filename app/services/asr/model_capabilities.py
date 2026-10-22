# -*- coding: utf-8 -*-
"""Shared capability-to-model asset definitions."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal, Optional

from app.core.config import settings
from app.infrastructure import resolve_local_model_dir
from app.services.asr.manager import get_model_manager
from app.services.asr.model_plan import get_active_qwen_model, get_runtime_model_ids


ModelSource = Literal["modelscope", "huggingface"]


@dataclass(frozen=True)
class ModelAsset:
    source: ModelSource
    model_id: str
    description: str
    revision: Optional[str] = None
    required_patterns: tuple[str, ...] = ()
    alternative_required_patterns: tuple[tuple[str, ...], ...] = ()
    min_total_size_bytes: int = 0
    # 非空表示该模型已存在于本地目录（不经 HF 缓存 
    # 结构，完整性检查与下载流程据此区分）
    local_dir: Optional[str] = None


def _build_huggingface_asset(
    model_id: str,
    description: str,
    *,
    min_total_size_bytes: int,
) -> ModelAsset:
    """构造 HuggingFace 资产声明，自动区分本地目录与 HF 缓存。"""
    local_dir = resolve_local_model_dir(model_id)
    if local_dir is not None:
        return ModelAsset(
            source="huggingface",
            model_id=model_id,
            description=description,
            local_dir=str(local_dir),
            required_patterns=("config.json",),
            alternative_required_patterns=(
                ("model.safetensors",),
                ("model.safetensors.index.json", "model-*.safetensors"),
            ),
            min_total_size_bytes=min_total_size_bytes,
        )

    return ModelAsset(
        source="huggingface",
        model_id=model_id,
        description=description,
        required_patterns=("snapshots/*/config.json",),
        alternative_required_patterns=(
            ("snapshots/*/model.safetensors",),
            (
                "snapshots/*/model.safetensors.index.json",
                "snapshots/*/model-*.safetensors",
            ),
        ),
        min_total_size_bytes=min_total_size_bytes,
    )


# 离线标点用的 CT-Transformer（FunASR AutoModel，CPU）
_OFFLINE_PUNC_ASSET = ModelAsset(
    source="modelscope",
    model_id=settings.PUNC_MODEL,
    description="Punctuation Offline",
    required_patterns=("configuration.json", "config.yaml", "model.pt", "tokens.json"),
    min_total_size_bytes=50_000_000,
)

# 说话人分离用本地 Nemotron-3（见 get_enabled_qwen_huggingface_assets），
# 这里只保留声纹命名需要的 CAM++ SV 模型。
_DIARIZATION_ASSETS = (
    ModelAsset(
        source="modelscope",
        model_id="damo/speech_campplus_sv_zh-cn_16k-common",
        description="CAM++ Speaker Verification",
        required_patterns=("configuration.json", "config.yaml", "campplus_cn_common.bin"),
        min_total_size_bytes=10_000_000,
    ),
)


def get_download_modelscope_assets() -> list[ModelAsset]:
    """Return the full static ModelScope export set used by predownload/export."""
    return [*_DIARIZATION_ASSETS, _OFFLINE_PUNC_ASSET]


def get_runtime_required_modelscope_assets(
    *,
    include_realtime_punc: bool = False,
) -> list[ModelAsset]:
    """Return ModelScope assets required by the current runtime plan.

    `include_realtime_punc` 仅为兼容旧调用方保留：实时标点随 Paraformer 实时栈一并移除。
    """
    return [*_DIARIZATION_ASSETS, _OFFLINE_PUNC_ASSET]


def get_enabled_qwen_huggingface_assets(
    *,
    include_forced_aligner: bool = True,
) -> list[ModelAsset]:
    """Return HuggingFace assets required by the runtime Qwen plan."""
    manager = get_model_manager()
    assets: list[ModelAsset] = []
    model_id = get_active_qwen_model()
    model_config = manager.get_declared_entry_config(model_id)
    offline_model = model_config.offline_model_path
    if offline_model:
        assets.append(
            _build_huggingface_asset(
                offline_model,
                f"{model_config.name} Offline",
                min_total_size_bytes=500_000_000,
            )
        )
    forced_aligner = str(model_config.extra_kwargs.get("forced_aligner_path") or "").strip()
    if forced_aligner and include_forced_aligner:
        assets.append(
            _build_huggingface_asset(
                forced_aligner,
                f"{model_config.name} Forced Aligner",
                min_total_size_bytes=500_000_000,
            )
        )
    # 离线说话人分离用本地 Nemotron-3（不落 HF 缓存结构，直接校验目录内容）
    assets.append(
        ModelAsset(
            source="huggingface",
            model_id="nvidia/Nemotron-3-Diarization",
            description="Nemotron-3 Diarization",
            local_dir=settings.NEMOTRON_MODEL_PATH,
            required_patterns=("config.json",),
            alternative_required_patterns=(
                ("model.safetensors",),
                ("model.safetensors.index.json", "model-*.safetensors"),
            ),
            min_total_size_bytes=100_000_000,
        )
    )
    return assets


def get_camplusplus_replacement_paths(cache_dir: str) -> dict[str, str]:
    """Return the CAM++ offline replacement map for local cache paths."""
    return {
        "damo/speech_campplus_sv_zh-cn_16k-common": f"{cache_dir}/damo/speech_campplus_sv_zh-cn_16k-common",
        "damo/speech_campplus-transformer_scl_zh-cn_16k-common": f"{cache_dir}/damo/speech_campplus-transformer_scl_zh-cn_16k-common",
        "damo/speech_fsmn_vad_zh-cn-16k-common-pytorch": f"{cache_dir}/damo/speech_fsmn_vad_zh-cn-16k-common-pytorch",
    }
