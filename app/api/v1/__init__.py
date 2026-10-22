# -*- coding: utf-8 -*-
"""API v1版本路由"""

from fastapi import APIRouter
from .realtime import router as realtime_router
from .openai_compatible import router as openai_router
from .voiceprints import router as voiceprints_router

api_router = APIRouter()

# 标准 OpenAI Realtime 转录（WebSocket）
api_router.include_router(realtime_router)

# OpenAI 兼容 API
api_router.include_router(openai_router)

# Voiceprint management API
api_router.include_router(voiceprints_router)
