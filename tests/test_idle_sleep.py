# -*- coding: utf-8 -*-
"""空闲休眠/唤醒策略测试

覆盖：
1. RuntimeRouter 空闲时调用 sleep（进程常驻）而非销毁引擎；不支持 sleep 才回退 unload；
2. aligner 独立生命周期：忙时不释放、可单独释放、不被主引擎连坐；
3. 请求结束刷新空闲计时；声纹前主动归还显存；
4. Backend 层 sleep 与推理互斥、使用中自动唤醒。
"""

import asyncio
import threading
import time
import unittest
from unittest import mock

from app.core.config import settings
from app.services.asr.qwen3_vllm import Qwen3VLLMBackend
from app.services.asr.runtime.router import RuntimeFamily, RuntimeRouter


class _FakeLLM:
    def __init__(self) -> None:
        self.slept = 0
        self.woke = 0

    def sleep(self, level: int = 1) -> None:
        self.slept += 1

    def wake_up(self) -> None:
        self.woke += 1


class _FakeEngine:
    def __init__(self, supports_sleep: bool = True, aligner_loaded: bool = False):
        self.slept = 0
        self.aligner_slept = 0
        self.unloaded = 0
        self.sleep_all_calls = 0
        self._supports_sleep = supports_sleep
        self._aligner_loaded = aligner_loaded
        self._aligner_busy = False
        self._aligner_idle = 999.0

    def is_model_loaded(self) -> bool:
        return True

    def supports_sleep(self) -> bool:
        return self._supports_sleep

    def sleep(self) -> None:
        self.slept += 1

    def unload(self) -> None:
        self.unloaded += 1

    def sleep_aligner(self) -> None:
        self.aligner_slept += 1

    def aligner_is_loaded(self) -> bool:
        return self._aligner_loaded

    def aligner_is_busy(self) -> bool:
        return self._aligner_busy

    def aligner_idle_seconds(self) -> float:
        return self._aligner_idle

    def sleep_all(self) -> None:
        self.sleep_all_calls += 1


def _make_router(
    engine: _FakeEngine, key: tuple[RuntimeFamily, str] = (RuntimeFamily.QWEN_VLLM, "m")
) -> tuple[RuntimeRouter, tuple[RuntimeFamily, str]]:
    """构造只含必要状态的 RuntimeRouter（跳过 __init__ 的模型加载）"""
    router = RuntimeRouter.__new__(RuntimeRouter)
    router._shared_engines = {key: engine}
    router._shared_limits = {}
    router._active_requests = {}
    router._last_use = {key: time.monotonic()}
    router._loaded_model_ids = {"m"}
    router._load_locks = {}
    return router, key


class IdleSleepRouterTest(unittest.TestCase):
    def test_idle_engine_sleeps_instead_of_unloading(self) -> None:
        engine = _FakeEngine()
        router, key = _make_router(engine)
        router._last_use[key] = time.monotonic() - 100  # 已空闲超过阈值

        asyncio.run(router._unload_idle_engines(15))

        self.assertEqual(engine.slept, 1)
        self.assertEqual(engine.unloaded, 0)
        # sleep 后引擎对象保留，供下次请求唤醒
        self.assertIn(key, router._shared_engines)

    def test_active_request_prevents_sleep(self) -> None:
        engine = _FakeEngine()
        router, key = _make_router(engine)
        router._last_use[key] = time.monotonic() - 100
        router._active_requests[key] = 1

        asyncio.run(router._unload_idle_engines(15))

        self.assertEqual(engine.slept, 0)

    def test_engine_without_sleep_falls_back_to_unload(self) -> None:
        engine = _FakeEngine(supports_sleep=False)
        router, key = _make_router(engine)
        router._last_use[key] = time.monotonic() - 100

        asyncio.run(router._unload_idle_engines(15))

        self.assertEqual(engine.unloaded, 1)
        self.assertNotIn(key, router._shared_engines)

    def test_long_idle_fully_unloads_after_shutdown_window(self) -> None:
        engine = _FakeEngine()
        router, key = _make_router(engine)
        with mock.patch.object(settings, "QWEN_SHUTDOWN_AFTER_S", 60):
            router._last_use[key] = time.monotonic() - 90

            asyncio.run(router._unload_idle_engines(15))

        # 二级：超过 shutdown 窗口 → 彻底卸载（连 sleep 残留一起归还）
        self.assertEqual(engine.unloaded, 1)
        self.assertEqual(engine.slept, 0)
        self.assertNotIn(key, router._shared_engines)

    def test_shutdown_window_disabled_keeps_sleeping(self) -> None:
        engine = _FakeEngine()
        router, key = _make_router(engine)
        router._last_use[key] = time.monotonic() - 10_000

        with mock.patch.object(settings, "QWEN_SHUTDOWN_AFTER_S", 0):
            asyncio.run(router._unload_idle_engines(15))

        self.assertEqual(engine.slept, 1)
        self.assertEqual(engine.unloaded, 0)

    def test_busy_aligner_blocks_full_unload(self) -> None:
        engine = _FakeEngine(aligner_loaded=True)
        engine._aligner_busy = True
        router, key = _make_router(engine)
        with mock.patch.object(settings, "QWEN_SHUTDOWN_AFTER_S", 60):
            router._last_use[key] = time.monotonic() - 90

            asyncio.run(router._unload_idle_engines(15))

        # 对齐进行中：不连坐卸载，但主引擎本身仍可休眠
        self.assertEqual(engine.unloaded, 0)
        self.assertEqual(engine.slept, 1)
        self.assertEqual(engine.aligner_slept, 0)

    def test_busy_aligner_is_not_slept(self) -> None:
        engine = _FakeEngine(aligner_loaded=True)
        engine._aligner_busy = True
        router, key = _make_router(engine)
        router._last_use[key] = time.monotonic() - 100

        asyncio.run(router._unload_idle_engines(15))

        self.assertEqual(engine.aligner_slept, 0)

    def test_aligner_sleeps_independently_while_main_engine_busy(self) -> None:
        engine = _FakeEngine(aligner_loaded=True)
        engine._aligner_idle = 100.0
        router, key = _make_router(engine)
        router._active_requests[key] = 1  # 主引擎正忙

        asyncio.run(router._unload_idle_engines(15))

        self.assertEqual(engine.slept, 0)  # 主引擎不释放
        self.assertEqual(engine.aligner_slept, 1)  # aligner 单独释放

    def test_release_refreshes_idle_timer(self) -> None:
        engine = _FakeEngine()
        router, key = _make_router(engine)
        router._last_use[key] = 0.0

        release = router._make_vllm_release(key, asyncio.Semaphore(1))
        release()

        self.assertGreater(router._last_use[key], 0.0)

    def test_release_vllm_gpu_calls_sleep_all(self) -> None:
        engine = _FakeEngine()
        router, _key = _make_router(engine)

        asyncio.run(router.release_vllm_gpu())

        self.assertEqual(engine.sleep_all_calls, 1)


class _FakeAligner:
    def __init__(self) -> None:
        self.slept = 0
        self.woke = 0

    def sleep(self, level: int = 1) -> None:
        self.slept += 1

    def wake_up(self) -> None:
        self.woke += 1


def _make_backend() -> Qwen3VLLMBackend:
    backend = Qwen3VLLMBackend.__new__(Qwen3VLLMBackend)
    backend._enable_sleep_mode = True
    backend._llm = _FakeLLM()
    backend._sleep_lock = threading.RLock()
    backend._main_active = 0
    backend._main_sleeping = False
    backend._aligner_state_lock = threading.RLock()
    backend._aligner_active = 0
    backend._aligner_last_use = time.monotonic()
    backend._aligner_sleeping = False
    backend._forced_aligner = _FakeAligner()
    return backend


class BackendSleepTest(unittest.TestCase):
    def test_sleep_main_skipped_while_in_use(self) -> None:
        backend = _make_backend()
        with backend._main_in_use():
            backend.sleep_main()
            self.assertEqual(backend._llm.slept, 0)
        backend.sleep_main()
        self.assertEqual(backend._llm.slept, 1)

    def test_main_wakes_automatically_on_use(self) -> None:
        backend = _make_backend()
        backend.sleep_main()
        self.assertTrue(backend.is_main_sleeping())

        with backend._main_in_use():
            self.assertFalse(backend.is_main_sleeping())

        self.assertEqual(backend._llm.woke, 1)

    def test_sleep_aligner_skipped_while_busy(self) -> None:
        backend = _make_backend()
        with backend._aligner_in_use():
            backend.sleep_aligner()
            self.assertEqual(backend._forced_aligner.slept, 0)
            self.assertTrue(backend.is_aligner_busy())
        backend.sleep_aligner()
        self.assertEqual(backend._forced_aligner.slept, 1)

    def test_aligner_wakes_automatically_on_use(self) -> None:
        backend = _make_backend()
        backend.sleep_aligner()
        self.assertTrue(backend.is_aligner_sleeping())

        with backend._aligner_in_use():
            self.assertFalse(backend.is_aligner_sleeping())

        self.assertEqual(backend._forced_aligner.woke, 1)

    def test_supports_sleep_flag(self) -> None:
        backend = _make_backend()
        self.assertTrue(backend.supports_sleep())
        backend._enable_sleep_mode = False
        self.assertFalse(backend.supports_sleep())


class SleepModeArgsTest(unittest.TestCase):
    """确认 sleep mode 开启时传给 vLLM 的参数正确。

    `mm_processor_cache_gb=0` 是实测必需的：sleep 丢弃 KV cache 后，
    multimodal processor cache 内部状态失效，唤醒后重复音频会抛 AssertionError。
    """

    def _build(self, enable_sleep_mode: bool) -> dict:
        captured: dict = {}

        class _FakeLLM:
            def __init__(self, **kwargs) -> None:
                captured.update(kwargs)

        class _FakeVLLM:
            LLM = _FakeLLM

            @staticmethod
            def SamplingParams(**kwargs):  # noqa: N802 - 模拟 vLLM API
                return object()

        class _FakeTokenizer:
            @staticmethod
            def from_pretrained(*args, **kwargs):
                return object()

        class _FakeTransformers:
            AutoTokenizer = _FakeTokenizer

        def fake_import(name: str):
            return _FakeVLLM if name == "vllm" else _FakeTransformers

        class _FakeImportlib:
            # 只替换 qwen3_vllm 模块内的 importlib，不动全局（否则 mock 自身会受影响）
            import_module = staticmethod(fake_import)

        with (
            mock.patch("app.services.asr.qwen3_vllm.importlib", _FakeImportlib),
            mock.patch(
                "app.services.asr.qwen3_vllm.resolve_huggingface_snapshot_dir",
                side_effect=lambda path: path,
            ),
        ):
            Qwen3VLLMBackend(
                model_path="fake-model",
                forced_aligner_path=None,
                gpu_memory_utilization=0.35,
                max_inference_batch_size=4,
                max_new_tokens=32,
                enable_sleep_mode=enable_sleep_mode,
            )
        return captured

    def test_sleep_mode_enables_sleep_and_disables_mm_cache(self) -> None:
        captured = self._build(enable_sleep_mode=True)
        self.assertTrue(captured["enable_sleep_mode"])
        self.assertEqual(captured["mm_processor_cache_gb"], 0)

    def test_sleep_mode_disabled_keeps_vllm_defaults(self) -> None:
        captured = self._build(enable_sleep_mode=False)
        self.assertNotIn("enable_sleep_mode", captured)
        self.assertNotIn("mm_processor_cache_gb", captured)


if __name__ == "__main__":
    unittest.main()
