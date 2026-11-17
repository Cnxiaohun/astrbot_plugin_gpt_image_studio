"""Image generation / editing client for an OpenAI-compatible images API.

Two operations, both verified against a Magpie gateway serving `gpt-image-2`:

* ``generate``  -> POST /v1/images/generations  (text-to-image)
* ``edit``      -> POST /v1/images/edits        (1..N reference images, optional mask)

Both accept the same optional ``size`` / ``quality`` hints. gpt-image-2 treats
``size`` as advisory (the prompt drives framing), so the client reports the real
returned pixel size rather than echoing what was requested.
"""

from __future__ import annotations

import asyncio
import base64
import binascii
from dataclasses import dataclass, field
from typing import Any

import httpx

from .http_utils import (
    MAX_JSON_IMAGE_BYTES,
    AmbiguousRequestError,
    ClientError,
    ConnectionFailure,
    HTTPRequestError,
    api_endpoint,
    positive_int,
    post_json,
    post_multipart,
    use_client,
)
from .images import ImageError, inspect_image

# Fields the caller may not smuggle in through extra_body.
_RESERVED = {"model", "prompt", "n", "image", "image[]", "mask"}


@dataclass
class ImageResult:
    data: bytes
    model: str
    mime: str
    width: int
    height: int
    revised_prompt: str = ""
    usage: dict[str, Any] = field(default_factory=dict)


def _decode_base64(value: Any) -> bytes:
    if not isinstance(value, str) or not value:
        raise ImageError("生图 API 返回的 b64_json 为空")
    payload = value.strip()
    if payload.startswith("data:"):
        comma = payload.find(",")
        payload = payload[comma + 1:] if comma >= 0 else payload
    try:
        data = base64.b64decode(payload, validate=True)
    except (binascii.Error, ValueError):
        raise ImageError("生图 API 返回的 b64_json 不是合法 base64") from None
    if not data:
        raise ImageError("生图 API 返回了空图片")
    return data


def _first_entry(response: dict[str, Any]) -> dict[str, Any]:
    entries = response.get("data")
    if not isinstance(entries, list) or len(entries) != 1 or not isinstance(entries[0], dict):
        raise ImageError("生图 API 应返回且仅返回一张图片")
    return entries[0]


class ImageClient:
    def __init__(self, api: dict, client: httpx.AsyncClient | None = None):
        self.api = api
        self.client = client

    async def close(self) -> None:
        """Injected clients belong to the caller; temporary ones close per call."""

    # ---- shared helpers -------------------------------------------------
    def _models(self, override: str = "") -> list[str]:
        primary = str(override or self.api.get("image_model") or "").strip()
        if not primary:
            raise ImageError("尚未配置生图模型")
        fb = self.api.get("fallback_models") or []
        if not isinstance(fb, list) or any(not isinstance(m, str) for m in fb):
            raise ImageError("备用模型配置必须为字符串列表")
        return list(dict.fromkeys([primary] + [m.strip() for m in fb if m.strip()]))

    def _extra(self) -> dict[str, Any]:
        extra = self.api.get("extra_body") or {}
        if not isinstance(extra, dict):
            raise ImageError("额外参数必须是 JSON 对象")
        return {k: v for k, v in extra.items() if k not in _RESERVED}

    def _timeout(self, override: Any = None) -> int:
        return positive_int(override if override else self.api.get("timeout_seconds", 300), 300, 1800)

    async def _finish(self, response: dict[str, Any], model: str) -> ImageResult:
        entry = _first_entry(response)
        if entry.get("b64_json") is not None:
            data = _decode_base64(entry["b64_json"])
        elif isinstance(entry.get("url"), str) and entry["url"].startswith(("http://", "https://")):
            data = await self._download(entry["url"])
        else:
            raise ImageError("生图 API 缺少 b64_json/url")
        mime, width, height = await asyncio.to_thread(inspect_image, data)
        usage = response.get("usage") if isinstance(response.get("usage"), dict) else {}
        actual = response.get("model") if isinstance(response.get("model"), str) else model
        revised = entry.get("revised_prompt") if isinstance(entry.get("revised_prompt"), str) else ""
        return ImageResult(data, actual, mime, width, height, revised, usage)

    async def _download(self, url: str) -> bytes:
        # Same-host gateway URLs only; never follow to arbitrary internals.
        async with use_client(self.client) as client:
            try:
                response = await client.get(url, timeout=60, follow_redirects=False)
            except httpx.HTTPError:
                raise ImageError("下载生成结果失败") from None
            if response.status_code >= 300:
                raise ImageError(f"下载生成结果 HTTP {response.status_code}")
            data = response.content
        if len(data) > 20 * 1024 * 1024:
            raise ImageError("生成结果超过 20MB")
        return data

    # ---- public API -----------------------------------------------------
    async def generate(
        self,
        prompt: str,
        *,
        model_override: str = "",
        size: str = "",
        quality: str = "",
    ) -> ImageResult:
        if not isinstance(prompt, str) or not prompt.strip():
            raise ImageError("提示词不能为空")
        models = self._models(model_override)
        extra = self._extra()
        endpoint = api_endpoint(self.api.get("base_url", ""), "images/generations")
        timeout = self._timeout()
        last: ClientError | None = None
        async with use_client(self.client) as client:
            for model in models:
                payload = dict(extra)
                payload.update({"model": model, "prompt": prompt.strip(), "n": 1})
                if size:
                    payload["size"] = size
                if quality:
                    payload["quality"] = quality
                try:
                    response = await post_json(
                        client, endpoint, self.api.get("api_key", ""), payload,
                        timeout=timeout, max_bytes=MAX_JSON_IMAGE_BYTES,
                    )
                except AmbiguousRequestError:
                    raise
                except HTTPRequestError as exc:
                    if exc.status in {401, 403}:
                        raise ImageError(f"API HTTP {exc.status}：认证/权限失败，请检查 Key") from None
                    last = exc
                    continue
                except ConnectionFailure as exc:
                    last = exc
                    continue
                except ClientError as exc:
                    raise ImageError(str(exc)) from None
                return await self._finish(response, model)
        raise ImageError(f"所有生图模型均失败：{last or '无可用模型'}")

    async def edit(
        self,
        prompt: str,
        images: list[bytes],
        *,
        mask: bytes | None = None,
        model_override: str = "",
        size: str = "",
        quality: str = "",
    ) -> ImageResult:
        if not isinstance(prompt, str) or not prompt.strip():
            raise ImageError("提示词不能为空")
        if not images:
            raise ImageError("参考图不能为空")
        if len(images) > 8:
            raise ImageError("一次最多引用 8 张参考图")
        models = self._models(model_override)
        extra = self._extra()
        endpoint = api_endpoint(self.api.get("base_url", ""), "images/edits")
        timeout = self._timeout()
        last: ClientError | None = None
        async with use_client(self.client) as client:
            for model in models:
                data: dict[str, Any] = dict(extra)
                data.update({"model": model, "prompt": prompt.strip(), "n": "1"})
                if size:
                    data["size"] = size
                if quality:
                    data["quality"] = quality
                files: list[tuple[str, tuple[str, bytes, str]]] = []
                if len(images) == 1:
                    files.append(("image", ("ref-1.png", images[0], "image/png")))
                else:
                    for idx, blob in enumerate(images, 1):
                        files.append(("image[]", (f"ref-{idx}.png", blob, "image/png")))
                if mask:
                    files.append(("mask", ("mask.png", mask, "image/png")))
                try:
                    response = await post_multipart(
                        client, endpoint, self.api.get("api_key", ""), data, files,
                        timeout=timeout, max_bytes=MAX_JSON_IMAGE_BYTES,
                    )
                except AmbiguousRequestError:
                    raise
                except HTTPRequestError as exc:
                    if exc.status in {401, 403}:
                        raise ImageError(f"API HTTP {exc.status}：认证/权限失败，请检查 Key") from None
                    last = exc
                    continue
                except ConnectionFailure as exc:
                    last = exc
                    continue
                except ClientError as exc:
                    raise ImageError(str(exc)) from None
                return await self._finish(response, model)
        raise ImageError(f"所有生图模型均失败：{last or '无可用模型'}")
