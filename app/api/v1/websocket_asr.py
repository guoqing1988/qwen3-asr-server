# -*- coding: utf-8 -*-
"""WebSocket ASR API routes."""

import logging
import time
from typing import Optional

from fastapi import APIRouter, WebSocket, WebSocketDisconnect

from ...services.websocket_asr import AliyunWebSocketASRService

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/ws/v1/asr", tags=["WebSocket ASR"])


@router.websocket("")
@router.websocket("/funasr")
async def funasr_websocket(websocket: WebSocket) -> None:
    await websocket.accept()
    service = AliyunWebSocketASRService()
    task_id = f"funasr_ws_{int(time.time())}_{id(websocket)}"

    try:
        await service.handle_connection(websocket, task_id)
    except WebSocketDisconnect:
        logger.info("[%s] Client disconnected", task_id)
    except Exception as exc:
        logger.error("[%s] Connection error: %s", task_id, exc)
    finally:
        await service.cleanup()


@router.websocket("/qwen")
async def qwen_asr_websocket(
    websocket: WebSocket,
    task_id: Optional[str] = None,
) -> None:
    """已废弃：改用标准 OpenAI Realtime 端点 `ws /v1/realtime`。"""
    _ = task_id
    await websocket.accept()
    await websocket.send_json(
        {
            "type": "error",
            "error": {
                "code": "endpoint_deprecated",
                "message": (
                    "ws /ws/v1/asr/qwen 已废弃，请改用 ws /v1/realtime"
                    "（OpenAI Realtime transcription 协议，intent=transcription）"
                ),
            },
        }
    )
    await websocket.close(code=1008)
