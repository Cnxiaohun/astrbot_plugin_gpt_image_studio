"""Fail-closed reference collection; downloads never inherit gateway credentials."""
from __future__ import annotations

import base64
import binascii
import hashlib
import os
from urllib.parse import unquote, urlsplit

import httpx

from .http_utils import MAX_IMAGE_BYTES, ClientError
from .images import inspect_image

MAX_REFERENCES = 6
HISTORY_IMAGE_SCAN = 12
_DOWNLOAD_TIMEOUT = 20


class ReferenceError(ClientError):
    """Reference input could not be fully read; do not submit paid generation."""


def _checked(data: bytes) -> bytes:
    if not data or len(data) > MAX_IMAGE_BYTES:
        raise ReferenceError("参考图为空或超过 20MB 上限")
    try:
        inspect_image(data)
    except ClientError:
        raise ReferenceError("参考图不是完整有效的 PNG/JPEG/WebP 单帧图片") from None
    return data


def _decode_base64(value: str) -> bytes:
    if len(value) > ((MAX_IMAGE_BYTES + 2) // 3) * 4 + 4:
        raise ReferenceError("参考图超过 20MB 上限")
    try:
        return _checked(base64.b64decode(value, validate=True))
    except (binascii.Error, ValueError):
        raise ReferenceError("参考图 base64 编码无效") from None


def _decode_data_url(value: str) -> bytes | None:
    if not value.lower().startswith("data:image/"):
        return None
    header, sep, payload = value.partition(",")
    if not sep or ";base64" not in header.lower():
        raise ReferenceError("参考图 data URL 不是 base64 编码")
    return _decode_base64(payload)


def _read_local(path: str) -> bytes | None:
    if not os.path.isfile(path):
        return None
    try:
        with open(path, "rb") as stream:
            return _checked(stream.read(MAX_IMAGE_BYTES + 1))
    except OSError:
        raise ReferenceError("无法读取本地参考图") from None


async def _download(url: str, client: httpx.AsyncClient | None) -> bytes:
    try:
        parsed = urlsplit(url)
        if (parsed.scheme not in {"http", "https"} or not parsed.hostname
                or parsed.username is not None or parsed.password is not None):
            raise ValueError
        parsed.port
    except ValueError:
        raise ReferenceError("参考图 URL 不合法或包含凭据") from None
    own = client is None
    http = client or httpx.AsyncClient(trust_env=False, follow_redirects=False)
    response = None
    try:
        # Do NOT use client.get/build_request: those merge gateway default
        # Authorization, x-api-key, Cookie, query parameters and client auth.
        request = httpx.Request(
            "GET", url, headers={"Accept-Encoding": "identity"},
            extensions={"timeout": httpx.Timeout(_DOWNLOAD_TIMEOUT).as_dict()},
        )
        response = await http.send(request, auth=None, follow_redirects=False, stream=True)
        if response.status_code >= 300:
            raise ReferenceError(f"下载参考图 HTTP {response.status_code}（不跟随重定向）")
        if response.headers.get("content-encoding", "identity").lower() not in {"", "identity"}:
            raise ReferenceError("参考图下载不允许压缩响应")
        length = response.headers.get("content-length")
        if length is not None:
            try:
                if int(length) < 0 or int(length) > MAX_IMAGE_BYTES:
                    raise ReferenceError("参考图超过 20MB 上限")
            except ValueError:
                raise ReferenceError("参考图 Content-Length 无效") from None
        data = bytearray()
        async for chunk in response.aiter_raw(chunk_size=65536):
            if len(data) + len(chunk) > MAX_IMAGE_BYTES:
                raise ReferenceError("参考图超过 20MB 上限")
            data.extend(chunk)
        return _checked(bytes(data))
    except httpx.HTTPError:
        raise ReferenceError("下载参考图失败，链接可能已过期") from None
    finally:
        if response is not None:
            await response.aclose()
        if own:
            await http.aclose()


async def _read_ref(value: str, client: httpx.AsyncClient | None) -> bytes:
    value = value.strip()
    if value.startswith("base64://"):
        return _decode_base64(value[9:])
    if value.lower().startswith("data:image/"):
        return _decode_data_url(value)
    if value.startswith(("http://", "https://")):
        return await _download(value, client)
    if value.startswith("file://"):
        parsed = urlsplit(value)
        if parsed.netloc not in {"", "localhost"}:
            raise ReferenceError("不支持远程 file URL")
        value = unquote(parsed.path)
    if os.path.isabs(value):
        data = _read_local(value)
        if data is not None:
            return data
    raise ReferenceError("参考图无法读取；请重新发送图片后重试")


async def component_to_bytes(component, client: httpx.AsyncClient | None = None) -> bytes | None:
    from astrbot.api.message_components import Image
    if not isinstance(component, Image):
        return None
    # Never call convert_to_file_path: its internal HTTP reader is unbounded.
    for name in ("path", "url", "file"):
        value = getattr(component, name, None)
        if isinstance(value, str) and value.strip():
            # url/file may be platform IDs; a real URL, path or base64 must be
            # explicitly available rather than delegating an unbounded download.
            if value.startswith(("http://", "https://", "base64://", "data:image/", "file://")) or os.path.isabs(value):
                return await _read_ref(value, client)
    raise ReferenceError("参考图链接不可用，请重新发送图片")


def _dedupe(blobs: list[bytes]) -> list[bytes]:
    seen = set()
    out = []
    for blob in blobs:
        digest = hashlib.sha256(blob).digest()
        if digest not in seen:
            seen.add(digest)
            out.append(blob)
    return out


def _cap(count: int) -> None:
    if count > MAX_REFERENCES:
        raise ReferenceError(f"一次最多 {MAX_REFERENCES} 张参考图，实际 {count} 张；请减少后重试")


async def _components(components, client) -> list[bytes]:
    from astrbot.api.message_components import Image
    images = [c for c in components if isinstance(c, Image)]
    _cap(len(images))
    # An individual failure must propagate, never silently drop one image.
    return [await component_to_bytes(c, client) for c in images]


async def collect_message_images(event, client: httpx.AsyncClient | None = None) -> list[bytes]:
    return await _components(event.get_messages(), client)


def _bot(event):
    for obj in (getattr(event, "bot", None), getattr(getattr(event, "message_obj", None), "bot", None)):
        if obj is not None and callable(getattr(obj, "call_action", None)):
            return obj
    return None


def _payload(payload):
    if not isinstance(payload, dict):
        raise ReferenceError("平台返回的引用消息格式无效")
    if isinstance(payload.get("data"), dict) and "message" not in payload:
        payload = payload["data"]
    return payload


def _check_group(event, payload, *, required: bool):
    current = str(event.get_group_id() or "")
    actual = str(payload.get("group_id") or "")
    if (current and ((required and not actual) or (actual and actual != current))) or (not current and actual):
        raise ReferenceError("引用消息群号无法验证或与当前群不符，已拒绝读取")


async def _segments(segments, client) -> list[bytes]:
    if not isinstance(segments, list):
        raise ReferenceError("平台消息不是结构化消息段，无法完整读取参考图")
    refs = []
    for seg in segments:
        if isinstance(seg, dict) and seg.get("type") == "image":
            data = seg.get("data") or {}
            value = data.get("url") or data.get("file")
            if not isinstance(value, str) or not value:
                raise ReferenceError("引用消息中有图片缺少可用链接")
            refs.append(value)
    _cap(len(refs))
    return [await _read_ref(ref, client) for ref in refs]


async def collect_quoted_images(event, client: httpx.AsyncClient | None = None) -> list[bytes]:
    from astrbot.api.message_components import Reply
    quote = next((c for c in event.get_messages() if isinstance(c, Reply)), None)
    if quote is None:
        return []
    bot = _bot(event)
    if bot is not None:
        # get_msg is the cross-version fallback for 4.29.3. Verify the group
        # BEFORE parsing/downloading any media from the returned message.
        try:
            payload = _payload(await bot.call_action("get_msg", message_id=quote.id))
        except ReferenceError:
            raise
        except Exception:
            raise ReferenceError("无法获取被引用消息，请重新发送参考图") from None
        _check_group(event, payload, required=True)
        blobs = await _segments(payload.get("message"), client)
    else:
        chain = getattr(quote, "chain", None) or []
        blobs = await _components(chain, client)
        if not blobs:
            # Avoid relying on version-specific official extractors that may
            # silently drop images, fetch forward chains, or download unbounded.
            raise ReferenceError("无法完整解析被引用消息，请直接带图发送指令")
    if not blobs:
        raise ReferenceError("被引用消息没有可用图片；不会回退到其它消息")
    return _dedupe(blobs)


async def _recent_history_images(event, client: httpx.AsyncClient | None, scan_limit: int) -> list[bytes]:
    bot = _bot(event)
    group_id = event.get_group_id()
    if not group_id or bot is None:
        raise ReferenceError("当前平台无法读取群历史图")
    try:
        payload = _payload(await bot.call_action("get_group_msg_history", group_id=int(group_id), count=int(scan_limit)))
    except ReferenceError:
        raise
    except Exception:
        raise ReferenceError("读取群历史图失败") from None
    messages = payload.get("messages")
    if not isinstance(messages, list):
        raise ReferenceError("群历史消息格式无效")
    segments = []
    for message in reversed(messages):
        if not isinstance(message, dict):
            raise ReferenceError("群历史消息格式无效")
        _check_group(event, message, required=True)
        chain = message.get("message")
        if not isinstance(chain, list):
            raise ReferenceError("群历史消息段格式无效")
        segments.extend(chain)
    return _dedupe(await _segments(segments, client))


async def gather_references(event, mode: str, client: httpx.AsyncClient | None = None, *, history_scan: int = HISTORY_IMAGE_SCAN) -> list[bytes]:
    """auto = explicit quote OR current attachments; history requires opt-in."""
    from astrbot.api.message_components import Reply
    mode = (mode or "auto").lower()
    if mode == "none":
        return []
    if mode == "message":
        return await collect_message_images(event, client)
    if mode == "quote":
        return await collect_quoted_images(event, client)
    if mode == "history":
        return await _recent_history_images(event, client, history_scan)
    if mode == "auto":
        if any(isinstance(c, Reply) for c in event.get_messages()):
            return await collect_quoted_images(event, client)
        return await collect_message_images(event, client)
    raise ReferenceError("未知的引用图模式")
