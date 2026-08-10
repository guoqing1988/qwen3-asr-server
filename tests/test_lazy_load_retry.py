# -*- coding: utf-8 -*-
"""懒加载失败自动重试测试。

背景：宿主环境不稳定时 vLLM 引擎初始化偶发段错误（SIGSEGV/-11），
懒加载路径需自动重试以提高恢复概率（router._ensure_shared_engine_loaded）。
"""

from __future__ import annotations

import unittest
from unittest.mock import MagicMock, patch

from app.services.asr.runtime.router import RuntimeFamily, RuntimeRouter


class _RetryEngine:
    """模拟 vLLM 引擎：is_model_loaded 返回 True"""

    def __init__(self, name: str) -> None:
        self.name = name

    @property
    def is_model_loaded(self) -> bool:
        return True


async def _no_sleep(_seconds: float) -> None:
    """跳过真实 sleep，加速测试"""


class LazyLoadRetryTest(unittest.IsolatedAsyncioTestCase):
    async def _build_router(self, create_impl) -> RuntimeRouter:
        router = RuntimeRouter()
        manager = MagicMock()
        manager.create_engine.side_effect = create_impl
        router._manager = manager  # type: ignore[assignment]
        return router

    async def test_retry_succeeds_after_transient_failures(self) -> None:
        """前 2 次创建失败（模拟 EngineCore 崩溃），第 3 次成功"""
        calls: list[str] = []

        def create_impl(model_id: str) -> _RetryEngine:
            calls.append(model_id)
            if len(calls) < 3:
                raise RuntimeError("Engine core initialization failed (-11)")
            return _RetryEngine("ok")

        router = await self._build_router(create_impl)
        with patch(
            "app.services.asr.runtime.router.asyncio.sleep", side_effect=_no_sleep
        ):
            engine, _sem = await router._ensure_shared_engine_loaded(  # noqa: SLF001
                RuntimeFamily.QWEN_VLLM, "qwen3-asr-test"
            )

        self.assertEqual(len(calls), 3, "应重试至第 3 次成功")
        self.assertEqual(engine.name, "ok")

    async def test_retry_exhausts_then_raises(self) -> None:
        """3 次全部失败 → 向上抛出异常（接口收到错误而非挂起）"""

        def create_impl(_model_id: str) -> _RetryEngine:
            raise RuntimeError("Engine core initialization failed (-11)")

        router = await self._build_router(create_impl)
        with patch(
            "app.services.asr.runtime.router.asyncio.sleep", side_effect=_no_sleep
        ):
            with self.assertRaisesRegex(RuntimeError, "Engine core initialization"):
                await router._ensure_shared_engine_loaded(  # noqa: SLF001
                    RuntimeFamily.QWEN_VLLM, "qwen3-asr-test"
                )

    async def test_first_attempt_success_no_retry(self) -> None:
        """首次创建成功 → 不触发重试"""
        calls: list[str] = []

        def create_impl(model_id: str) -> _RetryEngine:
            calls.append(model_id)
            return _RetryEngine("ok")

        router = await self._build_router(create_impl)
        with patch(
            "app.services.asr.runtime.router.asyncio.sleep", side_effect=_no_sleep
        ):
            engine, _sem = await router._ensure_shared_engine_loaded(  # noqa: SLF001
                RuntimeFamily.QWEN_VLLM, "qwen3-asr-test"
            )

        self.assertEqual(len(calls), 1, "首次成功不应重试")
        self.assertEqual(engine.name, "ok")
