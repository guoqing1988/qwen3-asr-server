# -*- coding: utf-8 -*-
"""空闲自动卸载与懒加载重建的单元测试。"""

from __future__ import annotations

import asyncio
import time
import unittest
from unittest.mock import MagicMock

from app.services.asr.runtime.router import RuntimeFamily, RuntimeRouter


class _MockEngine:
    """可切换加载状态的 mock 引擎"""

    def __init__(self, loaded: bool = True) -> None:
        self.loaded = loaded
        self.unload_calls = 0

    def is_model_loaded(self) -> bool:
        return self.loaded

    def unload(self) -> None:
        self.unload_calls += 1
        self.loaded = False


def _build_router() -> tuple[RuntimeRouter, _MockEngine]:
    router = RuntimeRouter()
    engine = _MockEngine()
    key = (RuntimeFamily.QWEN_VLLM, "qwen3-asr-test")
    router._shared_engines[key] = engine
    router._shared_limits[key] = asyncio.Semaphore(8)
    router._loaded_model_ids.add("qwen3-asr-test")
    router._resolve_family = lambda _model_id: RuntimeFamily.QWEN_VLLM  # type: ignore[method-assign]
    return router, engine


class IdleUnloadRouterTest(unittest.IsolatedAsyncioTestCase):
    async def test_idle_timeout_unloads_engine(self) -> None:
        router, engine = _build_router()
        key = (RuntimeFamily.QWEN_VLLM, "qwen3-asr-test")
        router._last_use[key] = time.monotonic() - 400  # 超过 300s 阈值

        await router._unload_idle_engines(300)

        self.assertEqual(engine.unload_calls, 1)
        self.assertNotIn(key, router._shared_engines)
        self.assertNotIn("qwen3-asr-test", router._loaded_model_ids)

    async def test_active_request_prevents_unload(self) -> None:
        router, engine = _build_router()
        key = (RuntimeFamily.QWEN_VLLM, "qwen3-asr-test")
        router._last_use[key] = time.monotonic() - 400
        router._active_requests[key] = 1

        await router._unload_idle_engines(300)

        self.assertEqual(engine.unload_calls, 0)
        self.assertIn(key, router._shared_engines)

    async def test_under_timeout_prevents_unload(self) -> None:
        router, engine = _build_router()
        key = (RuntimeFamily.QWEN_VLLM, "qwen3-asr-test")
        router._last_use[key] = time.monotonic() - 10

        await router._unload_idle_engines(300)

        self.assertEqual(engine.unload_calls, 0)

    async def test_semaphore_waiter_prevents_unload(self) -> None:
        router, engine = _build_router()
        key = (RuntimeFamily.QWEN_VLLM, "qwen3-asr-test")
        router._last_use[key] = time.monotonic() - 400
        semaphore = router._shared_limits[key]
        # 占满全部 slot 并挂起一个等待者，模拟有请求排队使用引擎
        for _ in range(8):
            await semaphore.acquire()
        waiter = asyncio.create_task(semaphore.acquire())
        await asyncio.sleep(0.01)
        try:
            await router._unload_idle_engines(300)
            self.assertEqual(engine.unload_calls, 0)
        finally:
            waiter.cancel()
            for _ in range(8):
                semaphore.release()

    async def test_unloaded_engine_is_lazy_reloaded_on_acquire(self) -> None:
        router, engine = _build_router()
        key = (RuntimeFamily.QWEN_VLLM, "qwen3-asr-test")
        router._last_use[key] = time.monotonic() - 400
        await router._unload_idle_engines(300)
        self.assertNotIn(key, router._shared_engines)

        # 请求到来 → 懒加载重建（create_engine 在后台线程执行）
        new_engine = _MockEngine()
        manager = MagicMock()
        manager.create_engine = MagicMock(return_value=new_engine)
        router._manager = manager

        lease = await router.acquire_engine("qwen3-asr-test")
        try:
            manager.create_engine.assert_called_once_with("qwen3-asr-test")
            self.assertIn(key, router._shared_engines)
            self.assertIs(lease.engine, new_engine)
            self.assertIn("qwen3-asr-test", router._loaded_model_ids)
        finally:
            await lease.close()

        self.assertEqual(router._active_requests.get(key, 0), 0)

    async def test_start_stop_monitor_lifecycle(self) -> None:
        router, _engine = _build_router()

        router.start_idle_unload_monitor()
        self.assertIsNotNone(router._unload_monitor_task)

        router.stop_idle_unload_monitor()
        self.assertIsNone(router._unload_monitor_task)


class BackendShutdownTest(unittest.TestCase):
    """Qwen3VLLMBackend.shutdown / Qwen3ASREngine.unload 的释放行为。"""

    def test_backend_shutdown_calls_engine_core(self) -> None:
        from app.services.asr.qwen3_vllm import Qwen3VLLMBackend

        backend = Qwen3VLLMBackend.__new__(Qwen3VLLMBackend)
        engine_core = MagicMock()
        llm = MagicMock()
        llm.llm_engine.engine_core = engine_core
        backend._llm = llm
        backend._forced_aligner = None

        backend.shutdown()

        engine_core.shutdown.assert_called_once_with(timeout=10.0)
        self.assertIsNone(backend._llm)

    def test_backend_shutdown_without_engine_core_does_not_raise(self) -> None:
        from app.services.asr.qwen3_vllm import Qwen3VLLMBackend

        backend = Qwen3VLLMBackend.__new__(Qwen3VLLMBackend)
        backend._llm = MagicMock()  # 无 llm_engine 属性
        backend._forced_aligner = None

        backend.shutdown()  # 不应抛异常

        self.assertIsNone(backend._llm)

    def test_engine_unload_clears_model(self) -> None:
        from app.services.asr.qwen3_engine import Qwen3ASREngine

        engine = Qwen3ASREngine.__new__(Qwen3ASREngine)
        backend = MagicMock()
        engine._backend = "vllm"
        engine.model = backend

        engine.unload()

        backend.shutdown.assert_called_once_with()
        self.assertIsNone(engine.model)

    def test_engine_unload_when_already_unloaded_is_noop(self) -> None:
        from app.services.asr.qwen3_engine import Qwen3ASREngine

        engine = Qwen3ASREngine.__new__(Qwen3ASREngine)
        engine._backend = "vllm"
        engine.model = None

        engine.unload()  # 不应抛异常
