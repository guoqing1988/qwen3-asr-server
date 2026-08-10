# -*- coding: utf-8 -*-
"""Runtime router for pooled ASR execution."""

from __future__ import annotations
import asyncio
import logging
import threading
import time
from dataclasses import dataclass
from enum import Enum
from typing import Awaitable, Callable, Optional

import torch

from app.core.config import settings
from app.core.device import detect_device
from app.core.executor import run_sync
from app.services.asr.engines import ASRFullResult, BaseASREngine
from app.services.asr.manager import get_model_manager
from app.services.asr.qwenasr_rust import is_qwenasr_rust_available
from .local_pool import LocalEnginePool

logger = logging.getLogger(__name__)

_VLLM_SHARED_CONCURRENCY = 8
# 显存压力自适应卸载：可用显存低于该值且引擎空闲时，立即卸载释放（让位其他 GPU 服务）
_LOW_VRAM_THRESHOLD_GB = 15.0
# 压力卸载冷却期：距引擎最后一次使用低于该值时不因压力卸载（避免刚用完就重载）
_VRAM_PRESSURE_COOLDOWN_S = 60.0


class RuntimeFamily(str, Enum):
    QWEN_VLLM = "qwen_vllm"
    QWEN_RUST_CPU = "qwen_rust_cpu"
    FUNASR = "funasr"


@dataclass
class OfflineASRRequest:
    model_id: str
    audio_path: str
    hotwords: str = ""
    enable_punctuation: bool = True
    enable_itn: bool = True
    sample_rate: int = 16000
    enable_speaker_diarization: bool = True
    word_timestamps: bool = False
    timestamp_scale: float = 1.0
    task_id: Optional[str] = None


class RuntimeEngineLease:
    """Lifecycle wrapper around a pooled engine instance."""

    def __init__(
        self,
        engine: BaseASREngine,
        release_callback: Callable[[], None | Awaitable[None]],
    ):
        self.engine = engine
        self._release_callback = release_callback
        self._closed = False

    async def close(self) -> None:
        if self._closed:
            return
        self._closed = True
        result = self._release_callback()
        if asyncio.iscoroutine(result):
            await result

    async def __aenter__(self) -> BaseASREngine:
        return self.engine

    async def __aexit__(self, exc_type, exc, tb) -> None:
        await self.close()


class RuntimeRouter:
    """Central backend router for all ASR entrypoints."""

    def __init__(self):
        self._manager = get_model_manager()
        self._pools: dict[tuple[RuntimeFamily, str], LocalEnginePool[BaseASREngine]] = (
            {}
        )
        self._shared_engines: dict[tuple[RuntimeFamily, str], BaseASREngine] = {}
        self._shared_limits: dict[tuple[RuntimeFamily, str], asyncio.Semaphore] = {}
        self._vllm_offline_locks: dict[str, asyncio.Lock] = {}
        self._pool_lock = threading.Lock()
        self._loaded_model_ids: set[str] = set()

        # 空闲自动卸载状态（仅 vLLM 共享引擎）：
        # - _last_use: 引擎最后一次被使用的时间（monotonic）
        # - _active_requests: 当前持有引擎的活跃请求数
        # - _load_locks: 每引擎重建锁，防止卸载/加载并发竞态
        # - _unload_monitor_task: 空闲监控后台任务
        self._last_use: dict[tuple[RuntimeFamily, str], float] = {}
        self._active_requests: dict[tuple[RuntimeFamily, str], int] = {}
        self._load_locks: dict[tuple[RuntimeFamily, str], asyncio.Lock] = {}
        self._unload_monitor_task: asyncio.Task | None = None

    def resolve_model_id(self, model_id: Optional[str]) -> str:
        if model_id:
            return model_id
        config = self._manager.get_declared_entry_config()
        return config.model_id

    def _resolve_family(self, model_id: str) -> RuntimeFamily:
        device = detect_device(settings.DEVICE)
        if model_id.startswith("qwen3-asr-"):
            if device.startswith("cuda"):
                return RuntimeFamily.QWEN_VLLM
            if device == "cpu" and is_qwenasr_rust_available():
                return RuntimeFamily.QWEN_RUST_CPU
            raise RuntimeError(f"Qwen3-ASR is not available on device '{device}'")
        return RuntimeFamily.FUNASR

    def _pool_size_for_family(self, family: RuntimeFamily) -> int:
        if family == RuntimeFamily.QWEN_VLLM:
            return 1
        if family == RuntimeFamily.QWEN_RUST_CPU:
            return settings.QWEN_RUST_CPU_WORKERS
        return settings.FUNASR_WORKERS

    def _create_pool(
        self, family: RuntimeFamily, model_id: str
    ) -> LocalEnginePool[BaseASREngine]:
        pool_key = (family, model_id)
        existing = self._pools.get(pool_key)
        if existing is not None:
            return existing

        with self._pool_lock:
            existing = self._pools.get(pool_key)
            if existing is not None:
                return existing
            pool = LocalEnginePool(
                size=self._pool_size_for_family(family),
                factory=lambda: self._manager.create_engine(model_id),
            )
            self._pools[pool_key] = pool
            self._loaded_model_ids.add(model_id)
            return pool

    @staticmethod
    def _engine_loaded(engine: BaseASREngine | None) -> bool:
        return engine is not None and getattr(engine, "is_model_loaded", lambda: True)()

    def _get_shared_engine(
        self, family: RuntimeFamily, model_id: str
    ) -> tuple[BaseASREngine, asyncio.Semaphore]:
        """同步获取共享引擎（warmup/启动路径）。引擎被卸载后自动重建。"""
        runtime_key = (family, model_id)
        engine = self._shared_engines.get(runtime_key)
        semaphore = self._shared_limits.get(runtime_key)
        if self._engine_loaded(engine) and semaphore is not None:
            return engine, semaphore

        with self._pool_lock:
            engine = self._shared_engines.get(runtime_key)
            semaphore = self._shared_limits.get(runtime_key)
            if not self._engine_loaded(engine):
                engine = self._manager.create_engine(model_id)
                self._shared_engines[runtime_key] = engine
                self._loaded_model_ids.add(model_id)
                self._last_use[runtime_key] = time.monotonic()
            if semaphore is None:
                semaphore = asyncio.Semaphore(_VLLM_SHARED_CONCURRENCY)
                self._shared_limits[runtime_key] = semaphore
            return engine, semaphore

    async def _ensure_shared_engine_loaded(
        self, family: RuntimeFamily, model_id: str
    ) -> tuple[BaseASREngine, asyncio.Semaphore]:
        """请求路径获取共享引擎；已卸载时懒加载重建（异步，不阻塞事件循环）。

        与空闲卸载共用 per-key 锁，保证卸载/重建互斥。
        """
        runtime_key = (family, model_id)
        lock = self._load_locks.setdefault(runtime_key, asyncio.Lock())
        async with lock:
            engine = self._shared_engines.get(runtime_key)
            semaphore = self._shared_limits.get(runtime_key)
            if semaphore is None:
                semaphore = asyncio.Semaphore(_VLLM_SHARED_CONCURRENCY)
                self._shared_limits[runtime_key] = semaphore
            if self._engine_loaded(engine):
                return engine, semaphore
            logger.info("Engine %s was unloaded, lazy loading...", model_id)
            # 模型加载耗时长，放入线程池避免阻塞事件循环；
            # 宿主环境不稳定时引擎初始化偶发段错误，失败自动重试（间隔递增）
            max_attempts = 3
            for attempt in range(1, max_attempts + 1):
                try:
                    engine = await asyncio.to_thread(
                        self._manager.create_engine, model_id
                    )
                    break
                except Exception:
                    logger.warning(
                        "Engine %s lazy load attempt %d/%d failed",
                        model_id,
                        attempt,
                        max_attempts,
                    )
                    if attempt == max_attempts:
                        raise
                    await asyncio.sleep(attempt * 5)
            self._shared_engines[runtime_key] = engine
            self._loaded_model_ids.add(model_id)
            self._last_use[runtime_key] = time.monotonic()
            logger.info("Engine %s lazy loaded", model_id)
            return engine, semaphore

    def warmup_model(self, model_id: Optional[str] = None) -> None:
        resolved_model_id = self.resolve_model_id(model_id)
        family = self._resolve_family(resolved_model_id)
        if family == RuntimeFamily.QWEN_VLLM:
            self._get_shared_engine(family, resolved_model_id)
            return
        pool = self._create_pool(family, resolved_model_id)
        pool.warmup()

    def get_loaded_model_ids(self) -> list[str]:
        return sorted(self._loaded_model_ids)

    def get_memory_usage(self) -> dict[str, object]:
        memory_info: dict[str, object] = {
            "model_list": self.get_loaded_model_ids(),
            "loaded_count": len(self._loaded_model_ids),
        }

        if torch.cuda.is_available():
            memory_info["gpu_memory"] = {
                "allocated": f"{torch.cuda.memory_allocated() / 1024**3:.2f}GB",
                "cached": f"{torch.cuda.memory_reserved() / 1024**3:.2f}GB",
                "max_allocated": f"{torch.cuda.max_memory_allocated() / 1024**3:.2f}GB",
            }

        return memory_info

    async def acquire_engine(
        self, model_id: Optional[str] = None
    ) -> RuntimeEngineLease:
        resolved_model_id = self.resolve_model_id(model_id)
        family = self._resolve_family(resolved_model_id)
        if family == RuntimeFamily.QWEN_VLLM:
            runtime_key = (family, resolved_model_id)
            # 等待信号量期间引擎可能被空闲监控卸载，拿到后需确认仍已加载
            while True:
                engine, semaphore = await self._ensure_shared_engine_loaded(
                    family, resolved_model_id
                )
                await semaphore.acquire()
                if self._engine_loaded(engine):
                    break
                logger.warning(
                    "Engine %s was unloaded while waiting for semaphore, reloading",
                    resolved_model_id,
                )
                semaphore.release()
            self._active_requests[runtime_key] = (
                self._active_requests.get(runtime_key, 0) + 1
            )
            self._last_use[runtime_key] = time.monotonic()
            return RuntimeEngineLease(
                engine=engine,
                release_callback=self._make_vllm_release(runtime_key, semaphore),
            )
        pool = self._create_pool(family, resolved_model_id)
        engine = await pool.acquire()
        return RuntimeEngineLease(
            engine=engine,
            release_callback=lambda: pool.release(engine),
        )

    def _make_vllm_release(
        self, runtime_key: tuple[RuntimeFamily, str], semaphore: asyncio.Semaphore
    ) -> Callable[[], None]:
        def release() -> None:
            self._active_requests[runtime_key] = max(
                0, self._active_requests.get(runtime_key, 0) - 1
            )
            semaphore.release()

        return release

    async def run_offline(self, request: OfflineASRRequest) -> ASRFullResult:
        model_id = self.resolve_model_id(request.model_id)
        if self._resolve_family(model_id) == RuntimeFamily.QWEN_VLLM:
            lock = self._vllm_offline_locks.setdefault(model_id, asyncio.Lock())
            async with lock:
                return await self._run_offline(request, model_id)
        return await self._run_offline(request, model_id)

    async def _run_offline(
        self,
        request: OfflineASRRequest,
        model_id: str,
    ) -> ASRFullResult:
        async with await self.acquire_engine(model_id) as engine:
            return await run_sync(
                engine.transcribe_long_audio,
                audio_path=request.audio_path,
                hotwords=request.hotwords,
                enable_punctuation=request.enable_punctuation,
                enable_itn=request.enable_itn,
                sample_rate=request.sample_rate,
                enable_speaker_diarization=request.enable_speaker_diarization,
                word_timestamps=request.word_timestamps,
                timestamp_scale=request.timestamp_scale,
                task_id=request.task_id,
            )

    # ------------------------------------------------------------------
    # 空闲自动卸载（仅 vLLM 共享引擎）
    # ------------------------------------------------------------------

    def start_idle_unload_monitor(self) -> None:
        """启动空闲自动卸载监控后台任务。"""
        timeout = settings.QWEN_IDLE_UNLOAD_TIMEOUT
        if timeout <= 0:
            logger.info("QWEN_IDLE_UNLOAD_TIMEOUT=%s, idle unload disabled", timeout)
            return
        if self._unload_monitor_task is None or self._unload_monitor_task.done():
            self._unload_monitor_task = asyncio.create_task(self._idle_unload_loop())
            logger.info(
                "Idle unload monitor started (timeout=%ss, family=%s)",
                timeout,
                RuntimeFamily.QWEN_VLLM.value,
            )

    def stop_idle_unload_monitor(self) -> None:
        """停止空闲自动卸载监控后台任务。"""
        if self._unload_monitor_task is not None:
            self._unload_monitor_task.cancel()
            self._unload_monitor_task = None
            logger.info("Idle unload monitor stopped")

    @staticmethod
    def _detect_vram_pressure() -> bool:
        """GPU 可用显存低于阈值时视为显存压力（其他服务需要大显存）。"""
        if not torch.cuda.is_available():
            return False
        try:
            free_bytes, _total = torch.cuda.mem_get_info()
            return free_bytes / (1024**3) < _LOW_VRAM_THRESHOLD_GB
        except Exception:  # noqa: BLE001 - 检测失败不阻断监控
            return False

    async def _idle_unload_loop(self) -> None:
        timeout = settings.QWEN_IDLE_UNLOAD_TIMEOUT
        # 间隔上限 30s：保证显存压力变化能被及时响应
        check_interval = min(max(timeout / 3, 5), 30)
        while True:
            await asyncio.sleep(check_interval)
            try:
                vram_pressure = self._detect_vram_pressure()
                await self._unload_idle_engines(timeout, vram_pressure=vram_pressure)
            except Exception as exc:  # noqa: BLE001 - 监控任务不应因单次异常退出
                logger.warning("Idle unload check failed: %s", exc)

    async def _unload_idle_engines(self, timeout: float, vram_pressure: bool = False) -> None:
        """卸载空闲的 vLLM 引擎。

        普通模式：超过 idle 阈值且无活跃请求/等待者时卸载。
        显存压力模式（vram_pressure=True）：可用显存紧张时，冷却期过后即卸载，
        将显存让位给其他 GPU 服务（如 ComfyUI）。
        """
        now = time.monotonic()
        for runtime_key, engine in list(self._shared_engines.items()):
            if runtime_key[0] != RuntimeFamily.QWEN_VLLM:
                continue
            if not self._engine_loaded(engine):
                continue
            if self._active_requests.get(runtime_key, 0) > 0:
                continue
            semaphore = self._shared_limits.get(runtime_key)
            # 有请求在等待信号量时不可卸载
            if semaphore is not None and semaphore.locked():
                continue
            last_use = self._last_use.get(runtime_key)
            if last_use is None:
                continue
            idle_for = now - last_use
            if vram_pressure:
                # 压力模式：冷却期（60s）过后即卸载，不等待完整 idle 超时
                if idle_for < _VRAM_PRESSURE_COOLDOWN_S:
                    continue
            elif idle_for < timeout:
                continue

            model_id = runtime_key[1]
            lock = self._load_locks.setdefault(runtime_key, asyncio.Lock())
            async with lock:
                # 等待锁期间可能有新请求进来，二次检查
                if not self._engine_loaded(engine):
                    continue
                if self._active_requests.get(runtime_key, 0) > 0:
                    continue
                if semaphore is not None and semaphore.locked():
                    continue
                logger.info(
                    "Engine %s idle for %.0fs, unloading to release GPU memory%s",
                    model_id,
                    idle_for,
                    " (vram pressure)" if vram_pressure else "",
                )
                try:
                    await asyncio.to_thread(engine.unload)
                except Exception as exc:  # noqa: BLE001 - 卸载失败不阻断监控
                    logger.error("Failed to unload engine %s: %s", model_id, exc)
                    continue
                self._shared_engines.pop(runtime_key, None)
                self._loaded_model_ids.discard(model_id)
                logger.info("Engine %s unloaded, GPU memory released", model_id)


_runtime_router: Optional[RuntimeRouter] = None
_runtime_router_lock = threading.Lock()


def get_runtime_router() -> RuntimeRouter:
    global _runtime_router
    if _runtime_router is None:
        with _runtime_router_lock:
            if _runtime_router is None:
                _runtime_router = RuntimeRouter()
    return _runtime_router
