# -*- coding: utf-8 -*-
"""OpenAI Realtime transcription WebSocket endpoint (`/v1/realtime`)."""

from __future__ import annotations

import logging
import uuid
from typing import Optional

from fastapi import APIRouter, WebSocket

from ...core.security import validate_token_value, validate_websocket_token
from ...services.asr.openai_realtime import OpenAIRealtimeSession

logger = logging.getLogger(__name__)
router = APIRouter(tags=["OpenAI Realtime"])


def _authorized(websocket: WebSocket) -> bool:
    """兼容项目 token（header/query）与 OpenAI 风格（Bearer / 不安全子协议）。"""
    valid, _ = validate_websocket_token(websocket)
    if valid:
        return True
    headers = getattr(websocket, "headers", None)
    if headers is not None:
        authorization = headers.get("authorization") or ""
        scheme, _, value = authorization.partition(" ")
        if scheme.lower() == "bearer" and validate_token_value(value.strip()):
            return True
    for protocol in websocket.scope.get("subprotocols", []):
        if protocol.startswith("openai-insecure-api-key.") and validate_token_value(
            protocol.removeprefix("openai-insecure-api-key.")
        ):
            return True
    return False


@router.websocket("/v1/realtime")
async def openai_realtime(
    websocket: WebSocket,
    intent: str = "transcription",
    task_id: Optional[str] = None,
) -> None:
    if not _authorized(websocket):
        await websocket.close(code=1008)
        return
    if intent != "transcription":
        await websocket.close(code=1008, reason="Only transcription sessions are supported")
        return
    session = OpenAIRealtimeSession(websocket, task_id or str(uuid.uuid4())[:8])
    await session.handle()
