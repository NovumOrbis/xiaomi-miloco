"""omni 图片输入能力探测（可选、需用户显式触发）。

``probe.probe_omni`` 只验证「地址可达 + 鉴权通过 + 模型名被接受」，**从不发图**：纯文本
模型也会通过。本模块补上缺的一环——发一张内置的合成纯色小图并问「主色是什么」，据此判断
当前配置的模型是否真的接受图片输入，而不是只看 HTTP 200。

- 图片：代码内生成的纯色 JPEG（cv2 编码，与运行时视觉输入同格式；无任何真实画面），
  经 ``prompt_builder._jpeg_block`` 包装成与生产同形的 image_url 块。
- 请求体：经 provider adapter 的 ``build_request_body`` 构造（OpenAI 兼容族 / Qwen 强制流式 /
  Gemini 原生 ``inline_data`` 各自拿到正确线上格式），鉴权头复用 ``build_request_headers``。
- 单次请求（无 GET /models 预检），会消耗极少量 token，**不得被自动触发**（只由 admin
  ``/omni-config/test-image`` 在用户点按钮时调用）。

返回 ``{status, reason, http_status?, latency_ms?}``，``status`` 四选一：

- ``supported``     —— 200 且回答正确（主色匹配）。
- ``unsupported``   —— provider 以 400/422 + 图片/多模态相关报错明确拒绝了图片输入。
  若报错指向「这张图」本身（格式 / 尺寸 / 解码失败），说明模型其实收图，归 ``inconclusive``。
- ``failed``        —— 鉴权 / 网络 / 超时 / 限流 / 5xx 等，**与图片能力无关**，无结论。
- ``inconclusive``  —— 200 但回答错误 / 为空 / 无法解析（看图能力存疑 ≠ 不支持图片），
  或被拒但看不出是否因图片、或拒的是这张图的格式/尺寸。

``reason`` 为短机器码（前端本地化）；刻意**不回传** provider 原始响应文本，避免泄漏
key / 内部信息。
"""

from __future__ import annotations

import json
import random
import re
import time
from typing import Any

import httpx

from miloco.perception.engine.omni.error_classifier import (
    classify_exception,
    classify_response,
)
from miloco.perception.engine.omni.probe import _normalize_base_url

# 单次请求预算：比连通性探测宽松（图片推理略慢），connect 仍保持短。
_TIMEOUT = httpx.Timeout(30.0, connect=10.0)
_MAX_TOKENS = 32
_IMAGE_SIZE = 96
_JPEG_QUALITY = 90

# 合成图候选：四种互相区分度高的纯色。每次探测随机取一种，避免纯文本模型靠固定答案蒙对。
_COLORS: dict[str, tuple[int, int, int]] = {
    "red": (220, 30, 30),
    "green": (30, 170, 60),
    "blue": (30, 60, 220),
    "yellow": (240, 210, 20),
}
# 中文颜色字 → 英文名（模型无视「用英文作答」时的宽松兜底）。
_ZH_COLORS = {"红": "red", "绿": "green", "蓝": "blue", "黄": "yellow"}

_QUESTION = (
    "What is the single dominant colour of this image? "
    "Answer with exactly one lowercase English word: red, green, blue or yellow."
)

# 400/422 报错文本里出现这些词，视为「provider 明确不接受图片/多模态输入」。
# 刻意只含通用词，不含任何厂商特定逻辑。
_IMAGE_REJECT_HINTS = (
    "image",
    "vision",
    "visual",
    "multimodal",
    "multi-modal",
    "modalit",
    "unknown variant",
    "image_url",
)
# 报错虽提到图片，但指向这张图本身（格式/尺寸/解码）→ 模型其实收图，不能判「不支持」。
_IMAGE_INVALID_HINTS = (
    "format",
    "decode",
    "corrupt",
    "too large",
    "too big",
    "size",
    "dimension",
    "resolution",
    "base64",
    "mime",
    "invalid image",
)
# 400 里夹带「key 无效」时（Gemini 对无效 key 返回 400）归为鉴权失败，不是图片问题。
_KEY_HINTS = ("api key", "api_key", "apikey", "invalid key", "unauthorized")


def build_probe_jpeg(rgb: tuple[int, int, int], size: int = _IMAGE_SIZE) -> bytes:
    """生成 ``size``×``size`` 纯色 JPEG（cv2 编码，与运行时视觉输入同格式）。"""
    import cv2
    import numpy as np

    bgr = np.full((size, size, 3), rgb[::-1], dtype=np.uint8)
    ok, buf = cv2.imencode(".jpg", bgr, [cv2.IMWRITE_JPEG_QUALITY, _JPEG_QUALITY])
    if not ok:  # pragma: no cover — 纯色图编码不会失败
        raise RuntimeError("probe JPEG encoding failed")
    return buf.tobytes()


def build_probe_messages(color: str) -> list[dict[str, Any]]:
    """OpenAI 形态的内部 IR：一个 image_url(JPEG data URI) 块 + 一个提问文本块。

    图片块直接复用运行时 ``prompt_builder._jpeg_block``，与生产请求同形；Gemini 的
    inline_data 转换由 ``GeminiAdapter.build_request_body`` 完成。
    """
    from miloco.perception.engine.omni.prompt_builder import _jpeg_block

    return [
        {
            "role": "user",
            "content": [
                _jpeg_block(build_probe_jpeg(_COLORS[color])),
                {"type": "text", "text": _QUESTION},
            ],
        }
    ]


def _result(
    status: str,
    reason: str,
    *,
    http_status: int | None = None,
    latency_ms: int | None = None,
) -> dict[str, Any]:
    out: dict[str, Any] = {"status": status, "reason": reason}
    if http_status is not None:
        out["http_status"] = http_status
    if latency_ms is not None:
        out["latency_ms"] = latency_ms
    return out


def _error_text(body: bytes) -> str:
    """从错误响应体抽出小写错误文本，**仅用于内部分类**，不对外返回。"""
    text = ""
    try:
        data = json.loads(body)
        err = data.get("error", data) if isinstance(data, dict) else data
        if isinstance(err, dict):
            text = " ".join(
                str(err.get(k, "")) for k in ("message", "status", "code", "type")
            )
        else:
            text = str(err)
    except Exception:  # noqa: BLE001 — 非 JSON 体回退原文
        text = body[:500].decode("utf-8", errors="replace")
    return text.lower()


def _classify_http_error(resp: httpx.Response, body: bytes, latency_ms: int) -> dict[str, Any]:
    """非 200 → 结果。基础分类复用运行时 ``classify_response``（与熔断同口径），
    只在 400/422（``rejected_authed``）上细分是否与图片有关。"""
    kw = {"http_status": resp.status_code, "latency_ms": latency_ms}
    classified = classify_response(resp)
    code = classified.code if classified is not None else "http_error"
    if code != "rejected_authed":
        return _result("failed", code, **kw)
    text = _error_text(body)
    if any(h in text for h in _KEY_HINTS):
        return _result("failed", "bad_key", **kw)
    if any(h in text for h in _IMAGE_REJECT_HINTS):
        if any(h in text for h in _IMAGE_INVALID_HINTS):
            return _result("inconclusive", "image_invalid", **kw)
        return _result("unsupported", "image_rejected", **kw)
    # 被拒但报错里看不出与图片有关（可能是模型名 / 参数问题）→ 不下结论。
    return _result("inconclusive", "rejected_unknown", **kw)


def _text_of(content: Any) -> str:
    """OpenAI message.content 可能是 str 或 [{type:text,text}] 列表。"""
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        return "".join(
            p.get("text", "")
            for p in content
            if isinstance(p, dict) and isinstance(p.get("text"), str)
        )
    return ""


def _colors_in(answer: str) -> set[str]:
    found = {w for w in re.findall(r"[a-z]+", answer.lower()) if w in _COLORS}
    found.update(en for zh, en in _ZH_COLORS.items() if zh in answer)
    return found


def _judge_answer(answer: str, expected: str, latency_ms: int) -> dict[str, Any]:
    kw = {"http_status": 200, "latency_ms": latency_ms}
    if not answer.strip():
        return _result("inconclusive", "empty_answer", **kw)
    mentioned = _colors_in(answer)
    if mentioned == {expected}:
        return _result("supported", "answer_correct", **kw)
    return _result("inconclusive", "wrong_answer", **kw)


async def _read_stream_answer(adapter: Any, resp: httpx.Response) -> str | None:
    """读 SSE 流并拼接增量文本（复用运行时 ``_iter_sse_chunks``）；
    一条可解析 chunk 都没有则返回 None。"""
    from miloco.perception.engine.omni.omni_client import _iter_sse_chunks

    parts: list[str] = []
    parsed_any = False
    async for chunk in _iter_sse_chunks(resp):
        if not isinstance(chunk, dict):
            continue
        parsed_any = True
        delta, _usage = adapter.parse_stream_chunk(chunk)
        if delta:
            parts.append(delta)
    return "".join(parts) if parsed_any else None


async def probe_image_input(
    model: str,
    base_url: str,
    api_key: str,
    *,
    color: str | None = None,
    transport: httpx.AsyncBaseTransport | None = None,
) -> dict[str, Any]:
    """发一次内置纯色图 + 「主色是什么」，判断模型是否真的接受图片输入。

    ``color`` / ``transport`` 仅供测试注入（固定答案 / MockTransport）。
    """
    base, err = _normalize_base_url(base_url)
    if err is not None:
        return _result("failed", "invalid_url")

    # 延迟 import，与 probe.py 同款（避免顶层循环依赖）。
    from miloco.perception.engine.omni.provider import (
        build_request_headers,
        get_adapter,
    )

    expected = color or random.choice(list(_COLORS))
    adapter = get_adapter(model)
    body = adapter.build_request_body(
        build_probe_messages(expected),
        model=model,
        max_tokens=_MAX_TOKENS,
        temperature=0.0,
        top_p=1.0,
        stream=False,  # adapter 若强制流式 (Qwen) 会覆盖
    )
    forced_stream = bool(body.get("stream", False))
    url = adapter.endpoint(base, model, stream=forced_stream)
    headers = build_request_headers(adapter, base, api_key)

    t0 = time.monotonic()
    try:
        async with httpx.AsyncClient(timeout=_TIMEOUT, transport=transport) as client:
            async with client.stream("POST", url, headers=headers, json=body) as resp:
                if resp.status_code != 200:
                    raw = await resp.aread()
                    latency_ms = round((time.monotonic() - t0) * 1000)
                    return _classify_http_error(resp, raw, latency_ms)
                if forced_stream:
                    answer = await _read_stream_answer(adapter, resp)
                    latency_ms = round((time.monotonic() - t0) * 1000)
                    if answer is None:
                        return _result(
                            "inconclusive",
                            "bad_response",
                            http_status=200,
                            latency_ms=latency_ms,
                        )
                    return _judge_answer(answer, expected, latency_ms)
                raw = await resp.aread()
    except Exception as e:  # noqa: BLE001 — 只回传分类码，不回传异常文本
        if isinstance(e, httpx.TimeoutException):
            return _result("failed", "timeout")
        return _result("failed", classify_exception(e).code)

    latency_ms = round((time.monotonic() - t0) * 1000)
    try:
        payload = adapter.parse_response(json.loads(raw))
        content = (
            payload["choices"][0]["message"]["content"]
            if payload.get("choices")
            else None
        )
    except Exception:  # noqa: BLE001 — 非 JSON / 形状异常
        return _result(
            "inconclusive", "bad_response", http_status=200, latency_ms=latency_ms
        )
    if content is None:
        return _result(
            "inconclusive", "empty_answer", http_status=200, latency_ms=latency_ms
        )
    return _judge_answer(_text_of(content), expected, latency_ms)
