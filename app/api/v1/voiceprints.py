"""Voiceprint management API."""

from __future__ import annotations

import logging
from typing import Optional

from fastapi import APIRouter, File, Form, Request, UploadFile
from fastapi.responses import JSONResponse

from app.core.exceptions import (
    AuthenticationException,
    InvalidParameterException,
    create_error_response,
    get_http_status_code,
)
from app.core.security import validate_token
from app.models.voiceprint import (
    VoiceprintDeleteResponse,
    VoiceprintRegistrationResponse,
    VoiceprintSampleRegistrationResponse,
    VoiceprintSpeakerItem,
    VoiceprintSpeakerListResponse,
)
from app.services.audio import get_audio_service
from app.services.audio.audio_service import AudioProcessingResult
from app.services.speaker import get_speaker_identification_service
from app.services.speaker.identification_service import VoiceprintSampleSource
from app.utils.common import generate_task_id

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api/v1", tags=["Voiceprints"])


async def _prepare_voiceprint_samples(
    *,
    files: list[UploadFile],
    task_id: str,
) -> tuple[list[VoiceprintSampleSource], list[AudioProcessingResult]]:
    if not files:
        raise InvalidParameterException("at least one voiceprint sample is required", task_id)

    audio_service = get_audio_service()
    samples: list[VoiceprintSampleSource] = []
    prepared_audios: list[AudioProcessingResult] = []

    for file in files:
        audio_data = await file.read()
        prepared_audio = await audio_service.process_upload_file(
            audio_data=audio_data,
            filename=file.filename,
            task_id=task_id,
            sample_rate=16000,
        )
        prepared_audios.append(prepared_audio)
        samples.append(
            VoiceprintSampleSource(
                audio_path=prepared_audio.normalized_path,
                source_bytes=audio_data,
            )
        )

    return samples, prepared_audios


def _cleanup_prepared_audios(prepared_audios: list[AudioProcessingResult]) -> None:
    audio_service = get_audio_service()
    for prepared_audio in prepared_audios:
        audio_service.cleanup(
            prepared_audio.original_path,
            prepared_audio.normalized_path,
        )


@router.post(
    "/voiceprint-speakers",
    response_model=VoiceprintRegistrationResponse,
    summary="Create a speaker and register voiceprint samples",
)
async def create_voiceprint_speaker(
    request: Request,
    display_name: str = Form(..., description="Speaker display name"),
    description: Optional[str] = Form(None, description="Speaker description"),
    files: list[UploadFile] = File(
        ...,
        alias="file",
        description="One or more single-speaker reference audio files",
    ),
) -> JSONResponse:
    """创建说话人并注册声纹样本。

    注册后，ASR 转写结果中与该样本声纹匹配的分段，其 `speaker_id` 会
    自动替换为该显示名；未注册或匹配不确定的说话人保留原始标签
    （如 `说话人1`）。

    参数（multipart/form-data）:
    - `display_name`: 说话人显示名，将出现在 ASR 结果的 `speaker_id` 中
    - `description`: 可选，说话人备注
    - `file`: 一个或多个**单人**参考音频（wav/mp3/mp4 等，自动归一化到
      16kHz）；建议每段 2 秒以上纯语音，多个样本可提升匹配准确率

    鉴权: 需要 `X-NLS-Token` 请求头（未配置 API_KEY 时免鉴权）。
    注意：本接口与 `/stream/*` 一致仅接受 `X-NLS-Token`，不接受
    `Authorization: Bearer`（OpenAI 兼容的 `/v1/*` 接口才两者皆可）

    示例:
        curl -X POST 'http://localhost:9101/api/v1/voiceprint-speakers' \\
          -H "X-NLS-Token: your_key" \\
          -F 'display_name=张三' -F 'file=@sample.wav'

    返回: `speaker_id`（后续追加样本/删除时使用）、`voiceprint_id(s)`、
    `voiceprint_count`。鉴权失败返回 400（AUTHENTICATION_FAILED）。
    """
    task_id = generate_task_id()
    prepared_audios: list[AudioProcessingResult] = []

    try:
        auth_ok, auth_content = validate_token(request, task_id)
        if not auth_ok:
            raise AuthenticationException(auth_content, task_id)

        normalized_display_name = display_name.strip()
        if not normalized_display_name:
            raise InvalidParameterException("display_name cannot be empty", task_id)

        samples, prepared_audios = await _prepare_voiceprint_samples(
            files=files,
            task_id=task_id,
        )
        result = get_speaker_identification_service().register_speaker_samples(
            display_name=normalized_display_name,
            description=description.strip() if description else None,
            samples=samples,
        )
        payload = VoiceprintRegistrationResponse(
            speaker_id=result.speaker.id,
            display_name=result.speaker.display_name,
            voiceprint_id=result.voiceprint_id,
            voiceprint_ids=result.voiceprint_ids,
            voiceprint_count=len(result.voiceprint_ids),
        ).model_dump()
        return JSONResponse(content=payload, headers={"task_id": task_id})

    except (AuthenticationException, InvalidParameterException) as exc:
        exc.task_id = task_id
        return JSONResponse(
            content=exc.to_dict(),
            headers={"task_id": task_id},
            status_code=get_http_status_code(exc.status_code),
        )
    except Exception as exc:
        logger.error("[%s] Voiceprint registration failed: %s", task_id, exc)
        return JSONResponse(
            content=create_error_response(
                error_code="DEFAULT_SERVER_ERROR",
                message=f"Voiceprint registration failed: {exc}",
                task_id=task_id,
            ),
            headers={"task_id": task_id},
            status_code=500,
        )
    finally:
        _cleanup_prepared_audios(prepared_audios)


@router.post(
    "/voiceprint-speakers/{speaker_id}/samples",
    response_model=VoiceprintSampleRegistrationResponse,
    summary="Add voiceprint samples to an existing speaker",
)
async def add_voiceprint_speaker_samples(
    request: Request,
    speaker_id: str,
    files: list[UploadFile] = File(
        ...,
        alias="file",
        description="One or more single-speaker reference audio files",
    ),
) -> JSONResponse:
    """为已有说话人追加声纹样本。

    同一说话人注册的样本越多，匹配越稳定（内部分数按
    `max_score * 0.7 + top3_mean_score * 0.3` 聚合）。

    参数:
    - `speaker_id`: 路径参数，创建说话人时返回的 ID
    - `file`: 一个或多个**单人**参考音频，格式与创建接口相同

    示例:
        curl -X POST 'http://localhost:9101/api/v1/voiceprint-speakers/{speaker_id}/samples' \\
          -H "X-NLS-Token: your_key" \\
          -F 'file=@another_sample.wav'

    返回: `speaker_id`、更新后的 `voiceprint_ids`、`voiceprint_count`。
    说话人不存在时返回 400（INVALID_PARAMETER）。
    """
    task_id = generate_task_id()
    prepared_audios: list[AudioProcessingResult] = []

    try:
        auth_ok, auth_content = validate_token(request, task_id)
        if not auth_ok:
            raise AuthenticationException(auth_content, task_id)

        samples, prepared_audios = await _prepare_voiceprint_samples(
            files=files,
            task_id=task_id,
        )
        voiceprint_ids = get_speaker_identification_service().add_speaker_samples(
            speaker_id=speaker_id,
            samples=samples,
        )
        payload = VoiceprintSampleRegistrationResponse(
            speaker_id=speaker_id,
            voiceprint_ids=voiceprint_ids,
            voiceprint_count=len(voiceprint_ids),
        ).model_dump()
        return JSONResponse(content=payload, headers={"task_id": task_id})
    except (AuthenticationException, InvalidParameterException) as exc:
        exc.task_id = task_id
        return JSONResponse(
            content=exc.to_dict(),
            headers={"task_id": task_id},
            status_code=get_http_status_code(exc.status_code),
        )
    except Exception as exc:
        logger.error("[%s] Voiceprint sample registration failed: %s", task_id, exc)
        return JSONResponse(
            content=create_error_response(
                error_code="DEFAULT_SERVER_ERROR",
                message=f"Voiceprint sample registration failed: {exc}",
                task_id=task_id,
            ),
            headers={"task_id": task_id},
            status_code=500,
        )
    finally:
        _cleanup_prepared_audios(prepared_audios)


@router.get(
    "/voiceprint-speakers",
    response_model=VoiceprintSpeakerListResponse,
    summary="List registered voiceprint speakers",
)
async def list_voiceprint_speakers(request: Request) -> JSONResponse:
    """列出已注册的声纹说话人。

    返回每个说话人的 `speaker_id`、`display_name`、`description` 和
    `voiceprint_count`（已注册的声纹样本数）。已软删除的说话人不返回。

    示例:
        curl 'http://localhost:9101/api/v1/voiceprint-speakers' \\
          -H "X-NLS-Token: your_key"
    """
    task_id = generate_task_id()
    try:
        auth_ok, auth_content = validate_token(request, task_id)
        if not auth_ok:
            raise AuthenticationException(auth_content, task_id)

        speakers = get_speaker_identification_service().list_speakers()
        payload = VoiceprintSpeakerListResponse(
            speakers=[
                VoiceprintSpeakerItem(
                    speaker_id=item.id,
                    display_name=item.display_name,
                    description=item.description,
                    voiceprint_count=item.voiceprint_count,
                )
                for item in speakers
            ]
        ).model_dump()
        return JSONResponse(content=payload, headers={"task_id": task_id})
    except AuthenticationException as exc:
        exc.task_id = task_id
        return JSONResponse(
            content=exc.to_dict(),
            headers={"task_id": task_id},
            status_code=get_http_status_code(exc.status_code),
        )
    except Exception as exc:
        logger.error("[%s] Voiceprint speaker list failed: %s", task_id, exc)
        return JSONResponse(
            content=create_error_response(
                error_code="DEFAULT_SERVER_ERROR",
                message=f"Voiceprint speaker list failed: {exc}",
                task_id=task_id,
            ),
            headers={"task_id": task_id},
            status_code=500,
        )


@router.delete(
    "/voiceprint-speakers/{speaker_id}",
    response_model=VoiceprintDeleteResponse,
    summary="Delete a voiceprint speaker",
)
async def delete_voiceprint_speaker(
    request: Request,
    speaker_id: str,
) -> JSONResponse:
    """软删除声纹说话人。

    删除后 ASR 结果不再使用该显示名替换 `speaker_id`，回退为原始说话人
    标签。仅标记 `status='deleted'`，声纹向量保留在数据库中，可通过
    重建数据库清理。

    参数:
    - `speaker_id`: 路径参数，要删除的说话人 ID

    示例:
        curl -X DELETE 'http://localhost:9101/api/v1/voiceprint-speakers/{speaker_id}' \\
          -H "X-NLS-Token: your_key"
    """
    task_id = generate_task_id()
    try:
        auth_ok, auth_content = validate_token(request, task_id)
        if not auth_ok:
            raise AuthenticationException(auth_content, task_id)

        get_speaker_identification_service().delete_speaker(speaker_id=speaker_id)
        payload = VoiceprintDeleteResponse(speaker_id=speaker_id).model_dump()
        return JSONResponse(content=payload, headers={"task_id": task_id})
    except AuthenticationException as exc:
        exc.task_id = task_id
        return JSONResponse(
            content=exc.to_dict(),
            headers={"task_id": task_id},
            status_code=get_http_status_code(exc.status_code),
        )
    except Exception as exc:
        logger.error("[%s] Voiceprint speaker delete failed: %s", task_id, exc)
        return JSONResponse(
            content=create_error_response(
                error_code="DEFAULT_SERVER_ERROR",
                message=f"Voiceprint speaker delete failed: {exc}",
                task_id=task_id,
            ),
            headers={"task_id": task_id},
            status_code=500,
        )
