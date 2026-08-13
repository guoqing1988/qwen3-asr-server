# 热词上下文功能设计规格（prompt 接线 + 服务端预设热词）

- 日期：2026-08-13
- 状态：已获用户批准
- 变更级别：中改（功能增强）

## 背景与目标

用户通过 `/v1/audio/transcriptions`（OpenAI 兼容接口）做音频转录，公司/行业专有名词与英文短语识别不准。Qwen3-ASR 是"指令跟随"式模型，热词通过 system prompt 的 context 注入（`qwen3_vllm.py` 的 `_build_chat_prompt` → `"Use this context when resolving named entities: {context}"`），可显著改善专有名词识别。

**现状**：热词参数链路（`hotwords` → vLLM `context`）已存在；原生 `/v1/asr` 已通过 `vocabulary_id` 暴露；但 OpenAI 兼容接口接收 `prompt` 后直接丢弃（`openai_compatible.py:506`），且无服务端预设能力。

**目标**：
1. OpenAI 接口的 `prompt` 参数接入热词链路
2. 新增 `ASR_DEFAULT_HOTWORDS` 环境变量实现服务端预设，请求未传热词时兜底
3. 请求热词与预设热词**合并**生效

## 已确认的决策

| 决策点 | 结论 |
|--------|------|
| 请求热词与预设热词关系 | 合并（默认在前、请求在后、去重） |
| 生效范围 | 两个离线接口都生效（OpenAI 兼容 + 原生 `/v1/asr`） |
| 合并逻辑位置 | 服务层（`OfflineTranscriptionService.transcribe()`），一处实现 |

## 详细设计

### 1. 配置（app/core/config.py + .env.example）

- `Settings` 新增类属性 `ASR_DEFAULT_HOTWORDS: str = ""`（注释：服务端预设热词，与请求热词合并后注入识别上下文）
- `_load_from_env()` 中读取：`self.ASR_DEFAULT_HOTWORDS = (os.getenv("ASR_DEFAULT_HOTWORDS") or "").strip()`，与 `API_KEY` 处理风格一致
- `.env.example` 新增注释段：说明变量用途、建议 ≤512 字符、示例（如 `阿里巴巴 腾讯 OpenAI Kubernetes`）

### 2. 合并函数（app/services/asr/offline_transcription_service.py）

模块级纯函数，中文注释说明合并语义：

```python
def merge_hotwords(default: str, request: str) -> str:
    """合并默认热词与请求热词。

    按空白分词、精确匹配去重（大小写敏感，避免破坏英文写法）、保持顺序（默认在前、请求在后）。
    """
```

- 顺序：默认词在前、请求词在后
- 去重：精确匹配（`Kubernetes` 与 `kubernetes` 视为不同词）
- 任一参数为空均正常返回；不抛异常
- 合并结果不做硬截断（请求侧 `prompt` 限 512、预设值建议 ≤512，模型上下文窗口足够容纳）

### 3. 服务层接线（offline_transcription_service.py）

`transcribe()` 构造 `OfflineASRRequest` 时将：

```python
hotwords=options.hotwords,
```

改为：

```python
hotwords=merge_hotwords(settings.ASR_DEFAULT_HOTWORDS, options.hotwords),
```

两个离线接口自动同时生效，原生 `/v1/asr` 端点代码零改动。

### 4. OpenAI 接口接线（app/api/v1/openai_compatible.py）

- `prompt` 表单字段：描述改为"热词/上下文提示，与 ASR_DEFAULT_HOTWORDS 合并后注入识别上下文"，加 `max_length=512`（与 `vocabulary_id` 的 512 上限一致）
- `_ = (prompt, temperature, timestamp_granularities)` 中移除 `prompt`（保留后两个为暂不支持）
- `OfflineTranscriptionOptions` 增加 `hotwords=prompt or ""`
- `_get_transcription_description()`：将 `prompt` 从"暂不支持的参数"中移出，补充用法说明

### 5. 错误处理

无新增异常路径：合并是纯函数；`ASR_DEFAULT_HOTWORDS` 未配置时行为与现状完全一致。

### 6. 测试（tests/test_hotwords_merge.py）

- **纯函数单测**：双空 / 仅默认 / 仅请求 / 去重 / 空白清洗 / 顺序（默认在前）
- **服务层集成**：monkeypatch 运行时路由 `run_offline`，验证传给路由的 `OfflineASRRequest.hotwords` 为合并结果（含 `ASR_DEFAULT_HOTWORDS` 置值/置空两场景）
- **接口接线**：FastAPI TestClient + monkeypatch `transcribe`，验证 `prompt` 流入 `OfflineTranscriptionOptions.hotwords`（实现时先确认 .venv 具备 TestClient 依赖；不具备则仅做前两层，并在测试文件中注释原因）

### 7. 文档

- `docs/deployment.md` 环境变量表新增 `ASR_DEFAULT_HOTWORDS` 行（含默认值、说明、合并语义）
- 注明：修改 `.env` 需走 systemd **stop → start** 才会重读（restart 不重读 `.env`）
- `app/api/v1/asr.py` 端点描述补充：未传 `vocabulary_id` 时用 `ASR_DEFAULT_HOTWORDS` 兜底，传入则合并

## 非目标（明确不做）

- WebSocket 实时流的 `context` 参数不动
- FunASR 引擎、CPU Rust 后端忽略热词是现状，本次不改
- 不做热词权重语法（数字权重本就标注 Deprecated）
- 不做热词管理/持久化 API（YAGNI）

## 部署注意事项

- 代码修改：`sudo systemctl restart qwen3-asr` 即时生效
- `.env` 新增 `ASR_DEFAULT_HOTWORDS`：需 stop → start 才重读
- 热词效果为"倾向性采纳"而非强制替换；建议精选几十个高频关键术语，过长列表会稀释效果
