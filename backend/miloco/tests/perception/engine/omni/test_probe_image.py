"""probe_image.py 单元测试:httpx.MockTransport 抓真实上线的请求体,断言各 provider 格式
里确实带了图片块,并覆盖 supported / unsupported / failed / inconclusive 的状态映射。"""

from __future__ import annotations

import base64
import json

import cv2
import httpx
import numpy as np
import pytest
from miloco.perception.engine.omni import probe_image

OPENAI_BASE = "https://llm.example/v1"
GEMINI_BASE = "https://generativelanguage.googleapis.com/v1beta"
KEY = "sk-test-secret-123"


def _transport(handler):
    return httpx.MockTransport(handler)


def _chat_ok(text):
    return httpx.Response(200, json={"choices": [{"message": {"content": text}}]})


def _capture(response_factory):
    """返回 (handler, seen);seen 收集每次请求的 url / headers / json body。"""
    seen: list[dict] = []

    def handler(request: httpx.Request) -> httpx.Response:
        seen.append(
            {
                "url": str(request.url),
                "headers": dict(request.headers),
                "body": json.loads(request.content),
            }
        )
        return response_factory()

    return handler, seen


def _decode_jpeg(data_uri: str) -> tuple[int, int, tuple[int, int, int]]:
    """解出 data URI 里的 JPEG,返回 (w, h, 平均 RGB);JPEG 有损,颜色按容差比较。"""
    assert data_uri.startswith("data:image/jpeg;base64,")
    raw = base64.b64decode(data_uri.split(",", 1)[1])
    assert raw[:3] == b"\xff\xd8\xff" and raw[-2:] == b"\xff\xd9"
    img = cv2.imdecode(np.frombuffer(raw, dtype=np.uint8), cv2.IMREAD_COLOR)
    assert img is not None
    b, g, r = (int(round(c)) for c in img.reshape(-1, 3).mean(axis=0))
    return img.shape[1], img.shape[0], (r, g, b)


def _close(actual: tuple[int, int, int], expected: tuple[int, int, int]) -> bool:
    return all(abs(a - e) <= 12 for a, e in zip(actual, expected))


# ─── 内置图片 ────────────────────────────────────────────────────────────────


def test_build_probe_jpeg_is_valid_solid_color():
    uri = (
        "data:image/jpeg;base64,"
        + base64.b64encode(probe_image.build_probe_jpeg((10, 20, 30), 16)).decode()
    )
    w, h, rgb = _decode_jpeg(uri)
    assert (w, h) == (16, 16) and _close(rgb, (10, 20, 30))


@pytest.mark.parametrize("color", sorted(probe_image._COLORS))
def test_probe_colours_survive_jpeg_and_stay_distinct(color):
    # 四种候选色经 JPEG 编码后仍贴近原色,且彼此不会被混淆
    w, h, rgb = _decode_jpeg(probe_image.build_probe_messages(color)[0]["content"][0]["image_url"]["url"])
    assert (w, h) == (probe_image._IMAGE_SIZE, probe_image._IMAGE_SIZE)
    assert _close(rgb, probe_image._COLORS[color])
    for other, other_rgb in probe_image._COLORS.items():
        if other != color:
            assert not _close(rgb, other_rgb)


# ─── 请求格式:各 provider 的图片块 ──────────────────────────────────────────


async def test_openai_compat_body_contains_image_url_block():
    handler, seen = _capture(lambda: _chat_ok("red"))
    r = await probe_image.probe_image_input(
        "mimo-v2.6-flash", OPENAI_BASE, KEY, color="red", transport=_transport(handler)
    )
    assert r["status"] == "supported" and r["reason"] == "answer_correct"
    assert len(seen) == 1
    req = seen[0]
    assert req["url"] == f"{OPENAI_BASE}/chat/completions"
    assert req["headers"]["authorization"] == f"Bearer {KEY}"
    body = req["body"]
    assert body["stream"] is False and body["max_tokens"] == probe_image._MAX_TOKENS
    content = body["messages"][0]["content"]
    img = next(b for b in content if b["type"] == "image_url")
    w, h, rgb = _decode_jpeg(img["image_url"]["url"])
    assert (w, h) == (96, 96) and _close(rgb, (220, 30, 30))
    assert any(b["type"] == "text" and "colour" in b["text"] for b in content)


async def test_qwen_forced_stream_body_contains_image_and_reads_sse():
    sse = (
        'data: {"choices":[{"delta":{"content":"Bl"}}]}\n\n'
        'data: {"choices":[{"delta":{"content":"ue"}}]}\n\n'
        "data: [DONE]\n\n"
    )
    handler, seen = _capture(
        lambda: httpx.Response(
            200, content=sse, headers={"content-type": "text/event-stream"}
        )
    )
    r = await probe_image.probe_image_input(
        "qwen3.5-omni-flash",
        OPENAI_BASE,
        KEY,
        color="blue",
        transport=_transport(handler),
    )
    assert r["status"] == "supported"
    body = seen[0]["body"]
    assert body["stream"] is True and body["modalities"] == ["text"]
    img = next(b for b in body["messages"][0]["content"] if b["type"] == "image_url")
    assert _close(_decode_jpeg(img["image_url"]["url"])[2], (30, 60, 220))


async def test_gemini_native_body_contains_inline_data_image():
    handler, seen = _capture(
        lambda: httpx.Response(
            200, json={"candidates": [{"content": {"parts": [{"text": "Green"}]}}]}
        )
    )
    r = await probe_image.probe_image_input(
        "gemini-3-flash-preview",
        GEMINI_BASE,
        KEY,
        color="green",
        transport=_transport(handler),
    )
    assert r["status"] == "supported"
    req = seen[0]
    assert req["url"] == f"{GEMINI_BASE}/models/gemini-3-flash-preview:generateContent"
    assert req["headers"]["x-goog-api-key"] == KEY
    parts = req["body"]["contents"][0]["parts"]
    inline = next(p["inline_data"] for p in parts if "inline_data" in p)
    assert inline["mime_type"] == "image/jpeg"
    assert _close(_decode_jpeg("data:image/jpeg;base64," + inline["data"])[2], (30, 170, 60))
    assert any("text" in p for p in parts)


# ─── 状态映射 ────────────────────────────────────────────────────────────────


async def _run(response_or_exc, *, model="mimo-v2.6-flash", color="red"):
    def handler(request: httpx.Request) -> httpx.Response:
        if isinstance(response_or_exc, Exception):
            raise response_or_exc
        return response_or_exc

    return await probe_image.probe_image_input(
        model, OPENAI_BASE, KEY, color=color, transport=_transport(handler)
    )


@pytest.mark.parametrize(
    "answer", ["red", "Red.", "The dominant colour is RED.", "红色"]
)
async def test_supported_answer_variants(answer):
    r = await _run(_chat_ok(answer))
    assert r["status"] == "supported" and r["http_status"] == 200
    assert isinstance(r["latency_ms"], int)


async def test_supported_with_content_part_list():
    r = await _run(_chat_ok([{"type": "text", "text": "red"}]))
    assert r["status"] == "supported"


@pytest.mark.parametrize(
    "message",
    [
        "Image input is not supported by this model",
        "unknown variant `image_url`, expected `text`",
        "This model does not support multimodal content",
        "vision is not enabled for model",
    ],
)
@pytest.mark.parametrize("status_code", [400, 422])
async def test_unsupported_when_provider_rejects_image(status_code, message):
    r = await _run(httpx.Response(status_code, json={"error": {"message": message}}))
    assert r["status"] == "unsupported" and r["reason"] == "image_rejected"
    assert r["http_status"] == status_code


async def test_unsupported_gemini_style_error_body():
    r = await _run(
        httpx.Response(
            400,
            json={
                "error": {
                    "code": 400,
                    "message": "Unable to process input image.",
                    "status": "INVALID_ARGUMENT",
                }
            },
        ),
        model="gemini-3-flash-preview",
    )
    assert r["status"] == "unsupported"


@pytest.mark.parametrize(
    "message",
    [
        "Invalid image format: expected png",
        "Image is too large",
        "Failed to decode image data",
        "image dimensions exceed the maximum resolution",
        "Multimodal data is corrupted: image",
    ],
)
async def test_rejecting_this_image_is_inconclusive_not_unsupported(message):
    # 报错指向这张图本身(格式/尺寸/解码) → 模型其实收图,不能判「不支持图片」
    r = await _run(httpx.Response(400, json={"error": {"message": message}}))
    assert r["status"] == "inconclusive" and r["reason"] == "image_invalid"


async def test_wrong_visual_answer_is_never_unsupported():
    # 看错颜色 = 视觉理解存疑,不等于不支持图片输入
    r = await _run(_chat_ok("blue"), color="red")
    assert r["status"] == "inconclusive" and r["reason"] == "wrong_answer"


async def test_400_invalid_api_key_is_failed_not_unsupported():
    r = await _run(
        httpx.Response(400, json={"error": {"message": "API key not valid. image"}}),
        model="gemini-3-flash-preview",
    )
    assert r["status"] == "failed" and r["reason"] == "bad_key"


async def test_400_without_image_hint_is_inconclusive():
    r = await _run(
        httpx.Response(400, json={"error": {"message": "max_tokens too small"}})
    )
    assert r["status"] == "inconclusive" and r["reason"] == "rejected_unknown"


@pytest.mark.parametrize(
    ("resp", "reason"),
    [
        (httpx.Response(401, text="nope"), "bad_key"),
        (httpx.Response(403, text="nope"), "bad_key"),
        (httpx.Response(404, text="nope"), "not_found"),
        (httpx.Response(429, text="slow down"), "rate_limited"),
        (httpx.Response(500, text="boom"), "http_error"),
        (httpx.Response(503, text="boom"), "http_error"),
    ],
)
async def test_failed_statuses(resp, reason):
    r = await _run(resp)
    assert r["status"] == "failed" and r["reason"] == reason


async def test_failed_timeout_and_unreachable():
    t = await _run(httpx.ReadTimeout("slow"))
    assert t["status"] == "failed" and t["reason"] == "timeout"
    u = await _run(httpx.ConnectError("dns"))
    assert u["status"] == "failed" and u["reason"] == "unreachable"


async def test_failed_invalid_scheme():
    r = await probe_image.probe_image_input("m", "ftp://x/v1", KEY)
    assert r == {"status": "failed", "reason": "invalid_url"}


async def test_inconclusive_wrong_answer():
    r = await _run(_chat_ok("blue"), color="red")
    assert r["status"] == "inconclusive" and r["reason"] == "wrong_answer"


async def test_inconclusive_multiple_colours_named():
    r = await _run(_chat_ok("red or blue"), color="red")
    assert r["status"] == "inconclusive" and r["reason"] == "wrong_answer"


@pytest.mark.parametrize("content", ["", "   ", None])
async def test_inconclusive_empty_answer(content):
    r = await _run(_chat_ok(content))
    assert r["status"] == "inconclusive" and r["reason"] == "empty_answer"


async def test_inconclusive_unparseable_200():
    r = await _run(httpx.Response(200, text="<html>gateway</html>"))
    assert r["status"] == "inconclusive" and r["reason"] == "bad_response"


async def test_inconclusive_stream_without_data():
    r = await _run(
        httpx.Response(200, content=": keep-alive\n\n"), model="qwen3.5-omni-flash"
    )
    assert r["status"] == "inconclusive" and r["reason"] == "bad_response"


async def test_stream_non_200_is_classified():
    r = await _run(
        httpx.Response(400, json={"error": {"message": "image not supported"}}),
        model="qwen3.5-omni-flash",
    )
    assert r["status"] == "unsupported"


async def test_result_never_echoes_response_text_or_key():
    secret_msg = "leak-me-please image " + KEY
    r = await _run(httpx.Response(400, json={"error": {"message": secret_msg}}))
    dumped = json.dumps(r)
    assert KEY not in dumped and "leak-me" not in dumped
    assert set(r) <= {"status", "reason", "http_status", "latency_ms"}
