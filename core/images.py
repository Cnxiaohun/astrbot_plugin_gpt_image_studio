"""Validate and normalize images before they reach or leave the plugin."""

from __future__ import annotations

import asyncio
import io
import warnings

from PIL import Image, UnidentifiedImageError

from .http_utils import MAX_IMAGE_BYTES, ClientError

MAX_PIXELS = 32_000_000
MAX_EDGE = 8192
# Uploaded references are re-encoded down before sending to keep requests small.
UPLOAD_MAX_EDGE = 2048
UPLOAD_QUALITY = 92


class ImageError(ClientError):
    """Generation/validation failed; safe to surface without leaking prompt/creds."""


_MIME = {"PNG": "image/png", "JPEG": "image/jpeg", "WEBP": "image/webp"}


def inspect_image(data: bytes) -> tuple[str, int, int]:
    """Verify container and pixels; reject SVG, animated images and huge rasters."""
    if not data or len(data) > MAX_IMAGE_BYTES:
        raise ImageError("图片为空或超过 20MB 上限")
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("error", Image.DecompressionBombWarning)
            with Image.open(io.BytesIO(data)) as image:
                fmt = image.format
                width, height = image.size
                if fmt not in _MIME:
                    raise ImageError("仅接受 PNG/JPEG/WebP 图片")
                if width < 1 or height < 1 or max(width, height) > MAX_EDGE or width * height > MAX_PIXELS:
                    raise ImageError("图片尺寸超过安全上限")
                if getattr(image, "n_frames", 1) != 1:
                    raise ImageError("只接受单帧图片")
                image.verify()
            with Image.open(io.BytesIO(data)) as image:
                image.load()
        return _MIME[fmt], width, height
    except ImageError:
        raise
    except (UnidentifiedImageError, OSError, ValueError, SyntaxError,
            Image.DecompressionBombWarning, Image.DecompressionBombError):
        raise ImageError("内容不是完整有效的图片") from None


def _flatten_to_png(data: bytes) -> bytes:
    """Validate, downscale oversized references and re-encode as PNG.

    Alpha is flattened onto white so the model never gets a transparent canvas.
    """
    inspect_image(data)
    with Image.open(io.BytesIO(data)) as source:
        source.load()
        image = source.convert("RGBA")
        rgb = Image.new("RGB", image.size, "white")
        rgb.paste(image, mask=image.getchannel("A"))
        longest = max(rgb.size)
        if longest > UPLOAD_MAX_EDGE:
            scale = UPLOAD_MAX_EDGE / longest
            new_size = (max(1, round(rgb.width * scale)), max(1, round(rgb.height * scale)))
            rgb = rgb.resize(new_size, Image.LANCZOS)
        out = io.BytesIO()
        rgb.save(out, format="PNG", optimize=True)
    return out.getvalue()


async def prepare_reference(data: bytes) -> bytes:
    """Downscale/validate a user-supplied reference image off the event loop."""
    return await asyncio.to_thread(_flatten_to_png, data)
