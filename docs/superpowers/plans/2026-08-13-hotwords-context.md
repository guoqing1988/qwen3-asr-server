# 热词上下文功能实现计划

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** OpenAI 转写接口 `prompt` 参数接入热词链路，并新增 `ASR_DEFAULT_HOTWORDS` 服务端预设热词，与请求热词合并后注入 Qwen3-ASR 识别上下文。

**Architecture:** 在离线转写服务层（`OfflineTranscriptionService.transcribe()`）用纯函数 `merge_hotwords()` 合并 `settings.ASR_DEFAULT_HOTWORDS` 与请求热词（OpenAI `prompt` / 原生 `vocabulary_id`），合并结果经既有 `hotwords` → vLLM `context` 链路注入模型 system prompt。两个离线接口一处生效，原生 `/v1/asr` 端点代码零改动。

**Tech Stack:** Python 3.12、FastAPI（Form 参数 + TestClient）、pydantic-free 自研 Settings、unittest（无 pytest-asyncio）、pytest。

**规格文档:** `docs/superpowers/specs/2026-08-13-hotwords-context-design.md`

## Global Constraints

- Python 3.12：新增代码用现代类型注解（内置泛型 `list[str]`、`X | None`）；**例外**：`openai_compatible.py` 中被修改的 `prompt` 表单字段保持该函数既有的 `Optional[str]` 风格，避免同一签名内注解风格混用
- 不新增 `from __future__ import annotations` 导入（`offline_transcription_service.py` 已存在该导入，保持不动）
- 关键业务逻辑必须有中文注释
- 测试运行命令：`.venv/bin/python -m pytest tests/test_hotwords_merge.py -v`（本地 .venv，pytest 9.1.1，**无 pytest-asyncio**，异步测试用 `unittest.IsolatedAsyncioTestCase` + `unittest.mock`，与 tests/test_runtime_router.py 风格一致）
- git commit 格式 `<type>: <描述>`，结尾加 `Co-Authored-By: Claude <noreply@anthropic.com>`
- 不安装新依赖（TestClient 依赖 httpx 0.28.1 已就绪）
- 合并语义（来自规格）：默认词在前、请求词在后；按空白分词；精确匹配去重（大小写敏感）；合并结果不做硬截断
- 非目标：不改 WebSocket 实时流 `context`、不改 FunASR / CPU Rust 后端忽略热词的现状、不做热词权重与管理 API

---

### Task 1: merge_hotwords 纯函数（TDD）

**Files:**
- Modify: `app/services/asr/offline_transcription_service.py`（模块级函数，放在 import 区之后、`PreparedAudio` 类定义之前）
- Test: `tests/test_hotwords_merge.py`（新建）

**Interfaces:**
- Produces: `merge_hotwords(default: str, request: str) -> str` —— Task 3 服务层接线依赖此函数。返回合并去重后的热词串，空格分隔。

- [ ] **Step 1: 编写失败测试**

创建 `tests/test_hotwords_merge.py`：

```python
# -*- coding: utf-8 -*-
"""merge_hotwords 纯函数与服务层/接口层热词接线测试。"""

from __future__ import annotations

import unittest

from app.services.asr.offline_transcription_service import merge_hotwords


class MergeHotwordsTest(unittest.TestCase):
    """merge_hotwords 纯函数单元测试。"""

    def test_both_empty(self) -> None:
        self.assertEqual(merge_hotwords("", ""), "")

    def test_default_only(self) -> None:
        self.assertEqual(merge_hotwords("阿里巴巴 腾讯", ""), "阿里巴巴 腾讯")

    def test_request_only(self) -> None:
        self.assertEqual(merge_hotwords("", "OpenAI Kubernetes"), "OpenAI Kubernetes")

    def test_merge_order_default_first(self) -> None:
        result = merge_hotwords("阿里巴巴 腾讯", "OpenAI")
        self.assertEqual(result, "阿里巴巴 腾讯 OpenAI")

    def test_dedup_exact_match(self) -> None:
        # 请求词与默认词重复时只保留首次出现（默认在前）
        result = merge_hotwords("阿里巴巴 OpenAI", "OpenAI 腾讯")
        self.assertEqual(result, "阿里巴巴 OpenAI 腾讯")

    def test_dedup_case_sensitive(self) -> None:
        # 大小写敏感：Kubernetes 与 kubernetes 是不同词，避免破坏英文写法
        result = merge_hotwords("Kubernetes", "kubernetes")
        self.assertEqual(result, "Kubernetes kubernetes")

    def test_whitespace_cleaning(self) -> None:
        result = merge_hotwords("  阿里巴巴   腾讯  ", "  OpenAI   ")
        self.assertEqual(result, "阿里巴巴 腾讯 OpenAI")


if __name__ == "__main__":
    unittest.main()
```

- [ ] **Step 2: 运行测试确认失败**

Run: `.venv/bin/python -m pytest tests/test_hotwords_merge.py -v`
Expected: FAIL，报 `ImportError: cannot import name 'merge_hotwords' from 'app.services.asr.offline_transcription_service'`

- [ ] **Step 3: 实现 merge_hotwords**

在 `app/services/asr/offline_transcription_service.py` 的 `logger = logging.getLogger(__name__)` 之后新增：

```python
def merge_hotwords(default: str, request: str) -> str:
    """合并默认热词与请求热词。

    按空白分词、精确匹配去重（大小写敏感，避免破坏英文写法）、
    保持顺序（默认在前、请求在后）。
    """
    seen: set[str] = set()
    merged: list[str] = []
    for word in f"{default} {request}".split():
        if word not in seen:
            seen.add(word)
            merged.append(word)
    return " ".join(merged)
```

- [ ] **Step 4: 运行测试确认通过**

Run: `.venv/bin/python -m pytest tests/test_hotwords_merge.py -v`
Expected: PASS（7 个用例全部通过）

- [ ] **Step 5: 提交**

```bash
git add app/services/asr/offline_transcription_service.py tests/test_hotwords_merge.py
git commit -m "feat: 新增热词合并函数 merge_hotwords

Co-Authored-By: Claude <noreply@anthropic.com>"
```

---

### Task 2: ASR_DEFAULT_HOTWORDS 配置（TDD）

**Files:**
- Modify: `app/core/config.py`（`Settings` 类属性 + `_load_from_env()`）
- Modify: `.env.example`（新增注释段）
- Test: `tests/test_hotwords_merge.py`（追加测试类）

**Interfaces:**
- Produces: `settings.ASR_DEFAULT_HOTWORDS: str`（已 `.strip()`，未配置时为空串）—— Task 3 依赖

- [ ] **Step 1: 编写失败测试**

在 `tests/test_hotwords_merge.py` 顶部 import 区追加：

```python
import os
from unittest import mock
```

并在文件末尾（`if __name__ == "__main__"` 之前）追加测试类：

```python
class SettingsDefaultHotwordsTest(unittest.TestCase):
    """ASR_DEFAULT_HOTWORDS 环境变量解析测试。"""

    def test_load_from_env_strips(self) -> None:
        from app.core.config import Settings

        with mock.patch.dict(os.environ, {"ASR_DEFAULT_HOTWORDS": "  阿里巴巴  腾讯  "}):
            self.assertEqual(Settings().ASR_DEFAULT_HOTWORDS, "阿里巴巴  腾讯")

    def test_default_is_empty_string(self) -> None:
        from app.core.config import Settings

        with mock.patch.dict(os.environ, {"ASR_DEFAULT_HOTWORDS": ""}):
            self.assertEqual(Settings().ASR_DEFAULT_HOTWORDS, "")
```

- [ ] **Step 2: 运行测试确认失败**

Run: `.venv/bin/python -m pytest tests/test_hotwords_merge.py -v`
Expected: 两个新用例 FAIL，报 `AttributeError: 'Settings' object has no attribute 'ASR_DEFAULT_HOTWORDS'`；Task 1 的 7 个用例仍 PASS

- [ ] **Step 3: 实现配置读取**

`app/core/config.py` 中，在 `ASR_BATCH_SIZE` 类属性（第 67 行）之后新增：

```python
    # 服务端预设热词（与请求热词合并后注入识别上下文；建议 ≤512 字符）
    ASR_DEFAULT_HOTWORDS: str = ""
```

在 `_load_from_env()` 的 `self.ASR_BATCH_SIZE = ...` 块（第 126-128 行）之后新增：

```python
        self.ASR_DEFAULT_HOTWORDS = (
            os.getenv("ASR_DEFAULT_HOTWORDS") or ""
        ).strip()
```

- [ ] **Step 4: 更新 .env.example**

在 `.env.example` 文件末尾追加（保持该文件英文注释风格）：

```
# -----------------------------------------------------------------------------
# Hotwords context.
# Server-side preset hotwords, merged with per-request hotwords
# (OpenAI `prompt` / native `vocabulary_id`) before recognition.
# Keep concise (recommended ≤512 chars) for best effect.
# -----------------------------------------------------------------------------
# ASR_DEFAULT_HOTWORDS=Alibaba Tencent OpenAI Kubernetes
```

- [ ] **Step 5: 运行测试确认通过**

Run: `.venv/bin/python -m pytest tests/test_hotwords_merge.py -v`
Expected: PASS（9 个用例全部通过）

- [ ] **Step 6: 提交**

```bash
git add app/core/config.py .env.example tests/test_hotwords_merge.py
git commit -m "feat: 新增 ASR_DEFAULT_HOTWORDS 服务端预设热词配置

Co-Authored-By: Claude <noreply@anthropic.com>"
```

---

### Task 3: 服务层接线（默认热词 + 请求热词合并生效）

**Files:**
- Modify: `app/services/asr/offline_transcription_service.py`（`transcribe()` 方法，约第 95-107 行）
- Test: `tests/test_hotwords_merge.py`（追加测试类）

**Interfaces:**
- Consumes: `merge_hotwords`（Task 1）、`settings.ASR_DEFAULT_HOTWORDS`（Task 2）
- Produces: `transcribe()` 传给运行时路由的 `OfflineASRRequest.hotwords` 恒为合并结果

- [ ] **Step 1: 编写失败测试**

在 `tests/test_hotwords_merge.py` 末尾（`if __name__ == "__main__"` 之前）追加：

```python
class ServiceHotwordsMergeTest(unittest.IsolatedAsyncioTestCase):
    """服务层转写：默认热词与请求热词合并后传给路由。"""

    async def _run_transcribe(self, default: str, request: str) -> str:
        from app.core.config import settings
        from app.services.asr.engines import ASRFullResult
        from app.services.asr.offline_transcription_service import (
            OfflineTranscriptionOptions,
            OfflineTranscriptionService,
            PreparedAudio,
        )
        from app.services.asr.runtime import get_runtime_router

        captured: dict[str, object] = {}

        async def fake_run_offline(asr_request):
            captured["hotwords"] = asr_request.hotwords
            return ASRFullResult(text="ok", segments=[], duration=0.0)

        router = get_runtime_router()
        service = OfflineTranscriptionService()
        audio = PreparedAudio(
            normalized_path="/tmp/fake.wav",
            duration=1.0,
            original_path="/tmp/fake.wav",
        )
        options = OfflineTranscriptionOptions(hotwords=request)
        with (
            mock.patch.object(router, "run_offline", new=fake_run_offline),
            mock.patch.object(settings, "ASR_DEFAULT_HOTWORDS", default),
            mock.patch.object(settings, "VOICEPRINT_ENABLED", False),
        ):
            await service.transcribe(audio, options)
        return str(captured["hotwords"])

    async def test_default_plus_request_merged(self) -> None:
        result = await self._run_transcribe("阿里巴巴 腾讯", "OpenAI")
        self.assertEqual(result, "阿里巴巴 腾讯 OpenAI")

    async def test_no_default_uses_request(self) -> None:
        result = await self._run_transcribe("", "OpenAI Kubernetes")
        self.assertEqual(result, "OpenAI Kubernetes")

    async def test_no_request_uses_default(self) -> None:
        result = await self._run_transcribe("阿里巴巴 腾讯", "")
        self.assertEqual(result, "阿里巴巴 腾讯")
```

- [ ] **Step 2: 运行测试确认失败**

Run: `.venv/bin/python -m pytest tests/test_hotwords_merge.py::ServiceHotwordsMergeTest -v`
Expected: FAIL（`test_default_plus_request_merged` 实际得到 `OpenAI` 而非合并结果，AssertionError）

- [ ] **Step 3: 实现服务层合并**

`app/services/asr/offline_transcription_service.py` 的 `transcribe()` 中，把：

```python
                hotwords=options.hotwords,
```

改为：

```python
                hotwords=merge_hotwords(settings.ASR_DEFAULT_HOTWORDS, options.hotwords),
```

（`settings` 与 `merge_hotwords` 均已在本模块可用，无需新增 import）

- [ ] **Step 4: 运行测试确认通过**

Run: `.venv/bin/python -m pytest tests/test_hotwords_merge.py -v`
Expected: PASS（12 个用例全部通过）

- [ ] **Step 5: 提交**

```bash
git add app/services/asr/offline_transcription_service.py tests/test_hotwords_merge.py
git commit -m "feat: 离线转写服务接入默认热词合并

Co-Authored-By: Claude <noreply@anthropic.com>"
```

---

### Task 4: OpenAI 接口 prompt 参数接线

**Files:**
- Modify: `app/api/v1/openai_compatible.py`（`prompt` 字段、`_ = (...)` 丢弃行、`OfflineTranscriptionOptions` 构造、`_get_transcription_description()`）
- Test: `tests/test_hotwords_merge.py`（追加测试类）

**Interfaces:**
- Consumes: `OfflineTranscriptionOptions.hotwords`（既有字段）
- Produces: `POST /v1/audio/transcriptions` 的 `prompt` 表单参数（≤512 字符）流入 `OfflineTranscriptionOptions.hotwords`，后续合并由 Task 3 的服务层完成

- [ ] **Step 1: 编写失败测试**

在 `tests/test_hotwords_merge.py` 末尾（`if __name__ == "__main__"` 之前）追加：

```python
def _make_wav_bytes(duration_sec: float) -> bytes:
    """生成 16kHz 单声道正弦 WAV 音频字节（供 TestClient 上传）。"""
    import io
    import math
    import struct
    import wave

    buf = io.BytesIO()
    with wave.open(buf, "wb") as wav:
        wav.setnchannels(1)
        wav.setsampwidth(2)
        wav.setframerate(16000)
        frames = bytearray()
        for i in range(int(16000 * duration_sec)):
            sample = int(32767 * 0.3 * math.sin(2 * math.pi * 440 * i / 16000))
            frames += struct.pack("<h", sample)
        wav.writeframes(bytes(frames))
    return buf.getvalue()


class OpenAIEndpointHotwordsTest(unittest.TestCase):
    """OpenAI 接口接线：prompt 表单参数流入 OfflineTranscriptionOptions.hotwords。"""

    def test_prompt_flows_into_hotwords(self) -> None:
        from fastapi.testclient import TestClient

        from app.core.config import settings
        from app.main import app
        from app.services.asr.engines import ASRFullResult
        from app.services.asr.offline_transcription_service import (
            get_offline_transcription_service,
        )

        captured: dict[str, object] = {}

        async def fake_transcribe(prepared_audio, options):
            captured["options"] = options
            return ASRFullResult(text="测试", segments=[], duration=1.0)

        service = get_offline_transcription_service()
        with (
            mock.patch.object(service, "transcribe", new=fake_transcribe),
            mock.patch.object(settings, "API_KEY", None),
        ):
            # 不用 with 包裹 TestClient，避免触发 lifespan（会预加载引擎）
            client = TestClient(app)
            response = client.post(
                "/v1/audio/transcriptions",
                files={"file": ("test.wav", _make_wav_bytes(0.5), "audio/wav")},
                data={"response_format": "text", "prompt": "阿里巴巴 OpenAI"},
            )

        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.text, "测试")
        options = captured["options"]
        self.assertEqual(options.hotwords, "阿里巴巴 OpenAI")
```

- [ ] **Step 2: 运行测试确认失败**

Run: `.venv/bin/python -m pytest tests/test_hotwords_merge.py::OpenAIEndpointHotwordsTest -v`
Expected: FAIL（`options.hotwords` 实际为 `""`，AssertionError）。若报 ffmpeg 缺失类错误（音频预处理失败），**先向用户确认**后再安装 ffmpeg（生产环境已具备，属系统依赖安装红线）

- [ ] **Step 3: 实现接口接线（四处小改）**

`app/api/v1/openai_compatible.py`：

1. `prompt` 表单字段（第 496 行）改为：

```python
    prompt: Optional[str] = Form(
        None,
        description="热词/上下文提示（与 ASR_DEFAULT_HOTWORDS 预设热词合并后注入识别上下文，≤512 字符）",
        max_length=512,
    ),
```

（保持 `Optional[str]` 风格与该函数其他参数一致，不混用新式注解）

2. `_ = (prompt, temperature, timestamp_granularities)`（第 506 行）改为：

```python
    _ = (temperature, timestamp_granularities)
```

3. `OfflineTranscriptionOptions(...)` 构造（第 559-563 行）改为：

```python
            OfflineTranscriptionOptions(
                sample_rate=16000,
                hotwords=prompt or "",
                enable_speaker_diarization=enable_speaker_diarization,
                word_timestamps=word_timestamps,
            ),
```

4. `_get_transcription_description()` 中（第 412-413 行）：

```markdown
**暂不支持的参数：**
`prompt`、`temperature`、`timestamp_granularities` 参数已保留但暂不生效
```

改为：

```markdown
**热词上下文：**
- `prompt` 可传公司/行业专有名词（如 `阿里巴巴 OpenAI Kubernetes`），与 `ASR_DEFAULT_HOTWORDS` 预设热词合并后注入识别上下文，改善专有名词与英文短语识别
- 热词为"倾向性采纳"而非强制替换，建议精选 ≤512 字符的高频关键术语

**暂不支持的参数：**
`temperature`、`timestamp_granularities` 参数已保留但暂不生效
```

- [ ] **Step 4: 运行测试确认通过**

Run: `.venv/bin/python -m pytest tests/test_hotwords_merge.py -v`
Expected: PASS（13 个用例全部通过）

- [ ] **Step 5: 提交**

```bash
git add app/api/v1/openai_compatible.py tests/test_hotwords_merge.py
git commit -m "feat: OpenAI 转写接口 prompt 参数接入热词链路

Co-Authored-By: Claude <noreply@anthropic.com>"
```

---

### Task 5: 文档更新

**Files:**
- Modify: `docs/deployment.md`（环境变量配置节，第 448-459 行之后）
- Modify: `app/api/v1/asr.py`（端点描述与 `vocabulary_id` 参数描述）

**Interfaces:** 无（纯文档）。前置依赖：Task 2 的 `ASR_DEFAULT_HOTWORDS` 语义

- [ ] **Step 1: 更新 docs/deployment.md**

在"### 声纹数据库配置"小节（第 459 行 `[voiceprint-architecture.md]` 段）之后、"## 服务监控"之前新增：

```markdown
### 热词上下文配置

| 环境变量 | 默认值 | 说明 |
|----------|--------|------|
| `ASR_DEFAULT_HOTWORDS` | （空） | 服务端预设热词，与请求热词（OpenAI `prompt` / 原生 `vocabulary_id`）合并后注入识别上下文；建议 ≤512 字符 |

热词为"倾向性采纳"而非强制替换，建议只收录高频关键术语（公司名、行业专有名词、
英文缩写等），过长列表会稀释效果。修改 `.env` 后需 systemd **stop → start**
才会重读（`restart` 不重读 `.env`）。
```

- [ ] **Step 2: 更新 app/api/v1/asr.py 端点描述**

1. 第 112 行注意事项 bullet：

```python
- `vocabulary_id` 参数用于传递无权重热词上下文（如：`阿里巴巴 腾讯`）。[Deprecated] 数字权重语法不受支持，传入时会被忽略
```

改为：

```python
- `vocabulary_id` 参数用于传递无权重热词上下文（如：`阿里巴巴 腾讯`），与 `ASR_DEFAULT_HOTWORDS` 预设热词合并生效。[Deprecated] 数字权重语法不受支持，传入时会被忽略
```

2. 第 174 行 `vocabulary_id` 参数 description：

```python
                "description": "无权重热词上下文，例如：`阿里巴巴 腾讯`。[Deprecated] 数字权重语法不受支持，传入时会被忽略",
```

改为：

```python
                "description": "无权重热词上下文，例如：`阿里巴巴 腾讯`；与 ASR_DEFAULT_HOTWORDS 预设热词合并生效。[Deprecated] 数字权重语法不受支持，传入时会被忽略",
```

- [ ] **Step 3: 回归确认（文档改动不破坏代码）**

Run: `.venv/bin/python -m pytest tests/test_hotwords_merge.py -v`
Expected: PASS（13 个用例全部通过）

- [ ] **Step 4: 提交**

```bash
git add docs/deployment.md app/api/v1/asr.py
git commit -m "docs: 更新热词上下文文档与环境变量说明

Co-Authored-By: Claude <noreply@anthropic.com>"
```

---

## 完成后的整体验证（部署前）

- [ ] 全量跑本项目测试：`.venv/bin/python -m pytest tests/ -v`（确认无回归）
- [ ] 部署：`sudo systemctl restart qwen3-asr`（代码改动生效）
- [ ] 在 `.env` 写入 `ASR_DEFAULT_HOTWORDS=<公司词表>` 后 **stop → start** 使其生效
- [ ] 手工验证：带 `prompt` 调用 `/v1/audio/transcriptions`，对比热词词项识别结果；再不带 `prompt` 验证预设词兜底生效
- [ ] 日志确认：`sudo journalctl -u qwen3-asr -f` 无异常
