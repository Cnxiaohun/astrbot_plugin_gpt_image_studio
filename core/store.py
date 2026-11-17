"""Save generated images and build chat reply chains (zh text + image)."""

from __future__ import annotations

import os
import time
import uuid
from pathlib import Path

from .http_utils import ClientError


class Store:
    def __init__(self, base_dir: Path, keep: int = 200):
        self.dir = Path(base_dir) / "outputs"
        self.dir.mkdir(parents=True, exist_ok=True)
        self.keep = max(10, int(keep))

    def save(self, data: bytes, mime: str) -> Path:
        ext = {"image/png": ".png", "image/jpeg": ".jpg", "image/webp": ".webp"}.get(mime, ".png")
        name = f"{time.strftime('%Y%m%d-%H%M%S')}-{uuid.uuid4().hex[:8]}{ext}"
        path = self.dir / name
        try:
            path.write_bytes(data)
        except OSError:
            raise ClientError("保存生成图片到磁盘失败") from None
        self._prune()
        return path

    def _prune(self) -> None:
        try:
            files = sorted(self.dir.glob("*"), key=os.path.getmtime, reverse=True)
        except OSError:
            return
        for old in files[self.keep:]:
            try:
                old.unlink()
            except OSError:
                pass


def build_reply(path: Path, text: str = ""):
    """Return a MessageChain with optional leading text and the image file."""
    from astrbot.api.event import MessageChain
    from astrbot.api.message_components import Image, Plain

    chain = MessageChain()
    if text:
        chain.chain.append(Plain(text))
    chain.chain.append(Image.fromFileSystem(str(path.resolve())))
    return chain
