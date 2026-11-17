"""End-to-end smoke test with a fake AstrBot event + real gateway (optional).

Set GPT_IMAGE_TEST_KEY to run against a live gateway; otherwise only the
offline parsing / reply-assembly paths that don't need network are exercised.
"""
from __future__ import annotations

import asyncio
import base64
import json
import os
import sys
from pathlib import Path

sys.path.insert(0, "/tmp/AstrBot")
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from astrbot.api.message_components import Image, Plain, Reply  # noqa: E402

from core.image_client import ImageClient  # noqa: E402
from core.references import component_to_bytes, collect_message_images  # noqa: E402

REF_PNG_B64 = None


def _ref_png() -> bytes:
    """A tiny solid-colour PNG generated on the fly (no fixture files)."""
    import io

    from PIL import Image as PILImage

    buf = io.BytesIO()
    PILImage.new("RGB", (64, 64), (220, 60, 60)).save(buf, format="PNG")
    return buf.getvalue()


class FakeSender:
    user_id = "10086"
    nickname = "测试员"


class FakeMessageObj:
    group_id = "123456"
    self_id = "99999"
    sender = FakeSender()
    type = None


class FakeEvent:
    def __init__(self, components, message_str=""):
        self.message_obj = FakeMessageObj()
        self.message_obj.message = components
        self.message_str = message_str
        self.session_id = "aiocqhttp:GroupMessage:123456"
        self.role = "admin"
        self._extras = {}
        self.sent = []

    def get_messages(self):
        return self.message_obj.message

    def get_sender_id(self):
        return "10086"

    def get_group_id(self):
        return "123456"

    def get_session_id(self):
        return self.session_id

    def is_admin(self):
        return True

    def set_extra(self, key, value):
        self._extras[key] = value

    def plain_result(self, text):
        return ("plain", text)

    def chain_result(self, chain):
        return ("chain", chain)


async def test_component_roundtrip():
    png = _ref_png()
    img = Image.fromBytes(png)
    data = await component_to_bytes(img)
    assert data == png, "fromBytes roundtrip failed"
    ev = FakeEvent([Plain("/生图 一只猫"), img, img], message_str="/生图 一只猫")
    blobs = await collect_message_images(ev)
    assert len(blobs) == 2, f"expected 2 images, got {len(blobs)}"
    print("OK component roundtrip + multi-image collect:", len(blobs))


async def test_data_url_and_reply_chain():
    png = _ref_png()
    data_url = "data:image/png;base64," + base64.b64encode(png).decode()
    comp = Image(file=data_url)
    assert await component_to_bytes(comp) == png
    # Reply with an embedded chain of two *distinct* images
    import io

    from PIL import Image as PILImage

    buf = io.BytesIO()
    PILImage.new("RGB", (64, 64), (60, 120, 220)).save(buf, format="PNG")
    png2 = buf.getvalue()
    reply = Reply(id="1", sender_nickname="someone")
    reply.chain = [Image.fromBytes(png), Image.fromBytes(png2)]
    ev = FakeEvent([reply, Plain("/生图 改一下")], message_str="/生图 改一下")
    from core.references import collect_quoted_images

    blobs = await collect_quoted_images(ev)
    assert len(blobs) == 2, f"reply chain expected 2, got {len(blobs)}"
    print("OK data url + reply embedded chain multi-image:", len(blobs))


async def test_live_generate_and_edit():
    key = os.environ.get("GPT_IMAGE_TEST_KEY")
    if not key:
        print("SKIP live (no GPT_IMAGE_TEST_KEY)")
        return
    api = {
        "base_url": os.environ.get("GPT_IMAGE_TEST_BASE", "http://127.0.0.1:3425"),
        "api_key": key,
        "image_model": "gpt-image-2",
        "fallback_models": [],
        "timeout_seconds": 300,
        "extra_body": {},
    }
    client = ImageClient(api)
    gen = await client.generate("一只戴墨镜的橘猫，扁平插画风", quality="low")
    print("OK live generate:", gen.model, gen.width, gen.height)
    from core.images import prepare_reference

    ref = await prepare_reference(_ref_png())
    edit = await client.edit("把这张图改成水彩风格", [ref], quality="low")
    print("OK live edit:", edit.width, edit.height)
    multi = await client.edit("把这几张图拼成九宫格", [ref, ref, ref], quality="low")
    print("OK live multi-edit:", multi.width, multi.height)


async def main():
    await test_component_roundtrip()
    await test_data_url_and_reply_chain()
    await test_live_generate_and_edit()
    print("ALL SMOKE TESTS PASSED")


if __name__ == "__main__":
    asyncio.run(main())
