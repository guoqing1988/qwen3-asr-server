# -*- coding: utf-8 -*-
"""启动预加载失败降级测试。

背景：宿主环境不稳定时 qwen3 引擎预加载可能失败，
降级策略：不阻断启动（其余引擎如 funasr 继续服务），失败模型由懒加载重试恢复。
"""

from __future__ import annotations

import unittest
from unittest.mock import MagicMock, patch

from app import main as main_module


class StartupGracefulDegradeTest(unittest.IsolatedAsyncioTestCase):
    async def test_preload_failure_does_not_block_startup(self) -> None:
        """qwen3 预加载失败时 lifespan 不抛错，正常就绪并启动空闲卸载监控"""
        fake_router = MagicMock()

        with (
            patch.object(main_module, "emit_boot_event"),
            patch.object(main_module, "cleanup_temp_directory"),
            patch.object(main_module, "shutdown_executor"),
            patch(
                "app.utils.model_loader.verify_required_models_integrity",
                return_value={"invalid_models": []},
            ),
            patch(
                "app.utils.model_loader.preload_models",
                return_value={
                    "asr_models": {
                        "qwen3-asr-1.7b": {
                            "loaded": False,
                            "error": "Engine core initialization failed (-11)",
                        },
                        "paraformer-large": {"loaded": True},
                    },
                    "extra_models": {},
                },
            ),
            patch(
                "app.services.asr.runtime.router.get_runtime_router",
                return_value=fake_router,
            ),
        ):
            # lifespan 不抛异常即通过（降级启动成功）
            async with main_module.lifespan(MagicMock()):
                pass

        fake_router.start_idle_unload_monitor.assert_called_once()

    async def test_all_models_loaded_startup_normal(self) -> None:
        """全部预加载成功时行为不变（正常就绪）"""
        fake_router = MagicMock()

        with (
            patch.object(main_module, "emit_boot_event"),
            patch.object(main_module, "cleanup_temp_directory"),
            patch.object(main_module, "shutdown_executor"),
            patch(
                "app.utils.model_loader.verify_required_models_integrity",
                return_value={"invalid_models": []},
            ),
            patch(
                "app.utils.model_loader.preload_models",
                return_value={
                    "asr_models": {
                        "qwen3-asr-1.7b": {"loaded": True},
                        "paraformer-large": {"loaded": True},
                    },
                    "extra_models": {},
                },
            ),
            patch(
                "app.services.asr.runtime.router.get_runtime_router",
                return_value=fake_router,
            ),
        ):
            async with main_module.lifespan(MagicMock()):
                pass

        fake_router.start_idle_unload_monitor.assert_called_once()
