"""Bounded JSON / multipart helpers for the image API.

Everything here treats the remote endpoint as untrusted: bodies are size-capped,
JSON is parsed strictly and errors never echo remote bodies or credentials.
Derived from the daily-newspaper plugin's hardened client.
"""

from __future__ import annotations

import asyncio
import json
from contextlib import asynccontextmanager, suppress
from typing import Any, AsyncIterator, Callable
from urllib.parse import urlsplit, urlunsplit

import httpx

MAX_IMAGE_BYTES = 20 * 1024 * 1024
MAX_JSON_IMAGE_BYTES = ((MAX_IMAGE_BYTES + 2) // 3) * 4 + 1024 * 1024


class ClientError(RuntimeError):
    """Safe user-facing error: message contains no remote body/url/api key."""


class AmbiguousRequestError(ClientError):
    """A request may already have been billed; do not automatically repeat it."""


class HTTPRequestError(ClientError):
    def __init__(self, status: int):
        self.status = status
        super().__init__(f"API HTTP {status}")


class ConnectionFailure(ClientError):
    """Connection establishment failed before any response was obtained."""


def positive_int(value: Any, default: int, maximum: int) -> int:
    try:
        return max(1, min(int(value), maximum))
    except (ValueError, TypeError, OverflowError):
        return default


def api_endpoint(base_url: str, resource: str) -> str:
    """Accept an API root or a root already ending in /v1; never duplicate /v1."""
    raw = str(base_url or "").strip().rstrip("/")
    try:
        parsed = urlsplit(raw)
        if (
            parsed.scheme not in {"https", "http"}
            or not parsed.hostname
            or parsed.username is not None
            or parsed.password is not None
            or parsed.query
            or parsed.fragment
        ):
            raise ValueError
        parsed.port  # validate malformed ports without exposing the URL
    except ValueError:
        raise ClientError("API base_url 不合法（只允许无凭据的 HTTP/HTTPS 根地址）") from None
    path = parsed.path.rstrip("/")
    if not path.endswith("/v1"):
        path += "/v1"
    return urlunsplit((parsed.scheme, parsed.netloc, path + "/" + resource.lstrip("/"), "", ""))


@asynccontextmanager
async def use_client(client: httpx.AsyncClient | None) -> AsyncIterator[httpx.AsyncClient]:
    """Only close clients we create; never inherit env proxies for local gateways."""
    if client is not None:
        yield client
    else:
        async with httpx.AsyncClient(trust_env=False, follow_redirects=False) as created:
            yield created


async def read_bounded(response: httpx.Response, limit: int) -> bytes:
    if response.headers.get("content-encoding", "identity").lower() not in {"", "identity"}:
        raise ClientError("响应使用未允许的压缩编码")
    length = response.headers.get("content-length")
    if length:
        try:
            if int(length) > limit:
                raise ClientError("响应超过大小上限")
        except ValueError:
            raise ClientError("响应 Content-Length 不合法") from None
    data = bytearray()
    async for chunk in response.aiter_bytes():
        if len(data) + len(chunk) > limit:
            raise ClientError("响应超过大小上限")
        data.extend(chunk)
    return bytes(data)


def _no_duplicate_pairs(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise ValueError("duplicate JSON key")
        result[key] = value
    return result


def _invalid_constant(_: str) -> None:
    raise ValueError("non-finite JSON number")


def strict_json_loads(text: str | bytes) -> Any:
    return json.loads(text, object_pairs_hook=_no_duplicate_pairs, parse_constant=_invalid_constant)


def uncertain_result(detail: str) -> AmbiguousRequestError:
    """Use only locally constructed, safe detail strings."""
    return AmbiguousRequestError(f"{detail}；任务可能已计费，未自动重试或切换模型，请先核实")


async def _send(
    client: httpx.AsyncClient,
    request_factory: Callable[[], httpx.Request],
    *,
    timeout: int,
    max_bytes: int,
    attempts: int,
) -> dict[str, Any]:
    """Retry only proven pre-send connection failures, never a received status.

    A gateway status (even 429) doesn't prove the upstream didn't accept a POST.
    """
    for attempt in range(attempts):
        request = request_factory()
        request.extensions = {"timeout": httpx.Timeout(timeout).as_dict()}
        try:
            response = await client.send(request, auth=None, follow_redirects=False, stream=True)
        except (httpx.ConnectError, httpx.ConnectTimeout, httpx.PoolTimeout):
            if attempt + 1 >= attempts:
                raise ConnectionFailure("API 连接建立失败，未发送生图请求") from None
            await asyncio.sleep(2 ** min(attempt, 3))
            continue
        except httpx.HTTPError:
            raise uncertain_result("生图 API 请求中断或读写超时") from None
        try:
            status = response.status_code
            if status == 408 or status >= 500:
                raise uncertain_result(f"生图 API HTTP {status}，上游处理结果不明")
            if not 200 <= status < 300:
                raise HTTPRequestError(status)
            try:
                raw = await read_bounded(response, max_bytes)
                value = strict_json_loads(raw)
            except (ClientError, httpx.HTTPError, ValueError, RecursionError):
                raise uncertain_result("生图 API 成功响应无法完整读取或不是有效 JSON") from None
            if not isinstance(value, dict):
                raise uncertain_result("生图 API 成功响应不是 JSON 对象")
            return value
        finally:
            # An expected transport cleanup error must not mask the outcome.
            with suppress(httpx.HTTPError):
                await response.aclose()
    raise ConnectionFailure("API 连接建立失败，未发送生图请求")


def _auth_headers(key: str, content_type: str) -> dict[str, str]:
    if not key:
        raise ClientError("尚未配置 API Key")
    return {"Authorization": f"Bearer {key}", "Content-Type": content_type, "Accept-Encoding": "identity"}


async def post_json(
    client: httpx.AsyncClient,
    url: str,
    key: str,
    payload: dict[str, Any],
    *,
    timeout: int,
    max_bytes: int = MAX_JSON_IMAGE_BYTES,
    attempts: int = 3,
) -> dict[str, Any]:
    return await _send(
        client,
        lambda: httpx.Request("POST", url, headers=_auth_headers(key, "application/json"), json=payload),
        timeout=timeout,
        max_bytes=max_bytes,
        attempts=attempts,
    )


async def post_multipart(
    client: httpx.AsyncClient,
    url: str,
    key: str,
    data: dict[str, Any],
    files: list[tuple[str, tuple[str, bytes, str]]],
    *,
    timeout: int,
    max_bytes: int = MAX_JSON_IMAGE_BYTES,
    attempts: int = 3,
) -> dict[str, Any]:
    """Build the multipart body once per attempt so retries are byte-fresh."""

    def factory() -> httpx.Request:
        stream = httpx._multipart.MultipartStream(  # noqa: SLF001 - build fresh per attempt
            data=dict(data),
            files=list(files),
        )
        body = b"".join(stream.iter_chunks())
        content_type = f"multipart/form-data; boundary={stream.boundary.decode('ascii')}"
        headers = _auth_headers(key, content_type)
        # Fully-render the multipart body: httpx only injects Host for its own
        # builders, and AsyncClient rejects a sync-ish content stream.
        return httpx.Request("POST", url, headers=headers, content=body)

    return await _send(client, factory, timeout=timeout, max_bytes=max_bytes, attempts=attempts)
