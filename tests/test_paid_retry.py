"""Offline verification of retry and model fallback for paid image POSTs."""

import base64
import io
import json
import unittest
from unittest.mock import AsyncMock, patch

import httpx
from PIL import Image

from astrbot_plugin_gpt_image_studio.core.http_utils import (
    AmbiguousRequestError,
    ClientError,
    MAX_JSON_IMAGE_BYTES,
)
from astrbot_plugin_gpt_image_studio.core.image_client import ImageClient


_SECRET = "upstream detail sk-test-should-not-leak"
_API = {
    "base_url": "https://gateway.test",
    "api_key": "local-test-key",
    "image_model": "primary",
    "fallback_models": ["backup"],
}


class _BrokenBody(httpx.AsyncByteStream):
    async def __aiter__(self):
        yield b'{"data":'
        # Even a connection-looking exception is ambiguous once headers arrive.
        raise httpx.ConnectError(_SECRET)


class PaidRetryTests(unittest.IsolatedAsyncioTestCase):
    async def _exercise(self, operation, handler, expected_error=None, message=None):
        calls = []

        def record(request):
            if request.method == "POST":
                if operation == "generate":
                    self.assertEqual(request.url.path, "/v1/images/generations")
                    model = json.loads(request.content)["model"]
                else:
                    self.assertEqual(request.url.path, "/v1/images/edits")
                    model = (
                        request.content.split(b'name="model"\r\n\r\n', 1)[1]
                        .split(b"\r\n", 1)[0].decode()
                    )
                calls.append(model)
            return handler(request, len(calls))

        async with httpx.AsyncClient(transport=httpx.MockTransport(record)) as transport_client:
            client = ImageClient(_API, transport_client)
            args = ("一张图片", [b"reference-image"]) if operation == "edit" else ("一张图片",)
            with patch("astrbot_plugin_gpt_image_studio.core.http_utils.asyncio.sleep", new_callable=AsyncMock):
                if expected_error is None:
                    return calls, await getattr(client, operation)(*args)
                with self.assertRaises(expected_error) as caught:
                    await getattr(client, operation)(*args)
        text = str(caught.exception)
        self.assertIn(message, text)
        self.assertNotIn(_SECRET, text)
        if expected_error is AmbiguousRequestError:
            self.assertIn("可能已计费", text)
            self.assertIn("未自动重试或切换模型", text)
        return calls

    async def test_ambiguous_http_status_never_resubmits(self):
        for op in ("generate", "edit"):
            for status in (408, 429, 500, 502, 503, 504, 599):
                with self.subTest(op=op, status=status):
                    calls = await self._exercise(
                        op, lambda _r, _n: httpx.Response(status, text=_SECRET, headers={"Retry-After": "0"}),
                        AmbiguousRequestError, f"HTTP {status}",
                    )
                    self.assertEqual(calls, ["primary"])

    async def test_other_http_status_never_falls_back_or_redirects(self):
        for op in ("generate", "edit"):
            for status in (302, 307, 400, 401, 403, 404, 409, 422):
                with self.subTest(op=op, status=status):
                    calls = await self._exercise(
                        op, lambda _r, _n: httpx.Response(status, text=_SECRET, headers={"Location": "/next"}),
                        ClientError, f"HTTP {status}",
                    )
                    self.assertEqual(calls, ["primary"])

    async def test_read_write_or_remote_disconnect_never_resubmits(self):
        for op in ("generate", "edit"):
            for exc in (httpx.ReadTimeout, httpx.ReadError, httpx.WriteTimeout,
                        httpx.WriteError, httpx.RemoteProtocolError, httpx.CloseError):
                with self.subTest(op=op, exc=exc):
                    def broken(request, _n):
                        raise exc(_SECRET, request=request)

                    calls = await self._exercise(op, broken, AmbiguousRequestError, "中断")
                    self.assertEqual(calls, ["primary"])

    async def test_malformed_success_never_resubmits(self):
        factories = (
            lambda: httpx.Response(200, content=_SECRET),
            lambda: httpx.Response(204),
            lambda: httpx.Response(200, json=[]),
            lambda: httpx.Response(200, content=b'{"data":[],"data":[]}'),
            lambda: httpx.Response(200, json={"error": _SECRET}),
            lambda: httpx.Response(200, json={"data": []}),
            lambda: httpx.Response(200, json={"data": [{}]}),
            lambda: httpx.Response(200, json={"data": [{"b64_json": "&&"}]}),
            lambda: httpx.Response(200, json={"data": [{"b64_json": "YWJj"}]}),
            lambda: httpx.Response(200, headers={"Content-Length": str(MAX_JSON_IMAGE_BYTES + 1)}),
            lambda: httpx.Response(200, stream=_BrokenBody()),
        )
        for op in ("generate", "edit"):
            for i, response in enumerate(factories):
                with self.subTest(op=op, case=i):
                    calls = await self._exercise(
                        op, lambda _r, _n: response(), AmbiguousRequestError, "生图 API" if i != 8 else "图片",
                    )
                    self.assertEqual(calls, ["primary"])

    async def test_download_failure_after_success_never_resubmits(self):
        for op in ("generate", "edit"):
            with self.subTest(op=op):
                def handler(request, _n):
                    if request.method == "GET":
                        return httpx.Response(404, text=_SECRET)
                    return httpx.Response(200, json={"data": [{"url": "https://gateway.test/output"}]})

                calls = await self._exercise(op, handler, AmbiguousRequestError, "下载生成结果 HTTP 404")
                self.assertEqual(calls, ["primary"])

    async def test_only_pre_send_failures_can_retry_and_fall_back(self):
        stream = io.BytesIO()
        Image.new("RGB", (2, 3)).save(stream, format="PNG")
        output = base64.b64encode(stream.getvalue()).decode()
        for op in ("generate", "edit"):
            with self.subTest(op=op):
                def handler(request, n):
                    if n <= 3:
                        exc = (httpx.ConnectError, httpx.ConnectTimeout, httpx.PoolTimeout)[n - 1]
                        raise exc(_SECRET, request=request)
                    return httpx.Response(200, json={"data": [{"b64_json": output}]})

                calls, result = await self._exercise(op, handler)
                self.assertEqual(calls, ["primary", "primary", "primary", "backup"])
                self.assertEqual((result.model, result.width, result.height), ("backup", 2, 3))

    async def test_retry_budget_does_not_override_ambiguous_result(self):
        for op in ("generate", "edit"):
            with self.subTest(op=op):
                def handler(request, n):
                    if n == 1:
                        raise httpx.ConnectError(_SECRET, request=request)
                    return httpx.Response(504, text=_SECRET)

                calls = await self._exercise(op, handler, AmbiguousRequestError, "HTTP 504")
                self.assertEqual(calls, ["primary", "primary"])
