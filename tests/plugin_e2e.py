"""Manual end-to-end smoke test with a fake AstrBot event and a real gateway.

Requires GPT_IMAGE_TEST_KEY (will make real, paid image calls). Not part of the
offline unit suite. Commands now submit detached jobs, so this test drains
``plugin._jobs`` and inspects the final messages sent back to the original event.
"""
from __future__ import annotations

import asyncio
import io
import os
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, "/tmp/AstrBot")
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from PIL import Image as PILImage  # noqa: E402
from astrbot.api.message_components import Image, Plain, Reply  # noqa: E402
from astrbot.api.star import StarTools  # noqa: E402

from astrbot_plugin_gpt_image_studio import main as plugin_main  # noqa: E402


def _png(color) -> bytes:
    buf = io.BytesIO()
    PILImage.new("RGB", (80, 80), color).save(buf, format="PNG")
    return buf.getvalue()


class FakeConfig(dict):
    pass


class FakeSender:
    user_id = "10086"
    nickname = "测试员"


class FakeMessageObj:
    group_id = "123456"
    self_id = "99999"
    sender = FakeSender()
    type = None


class FakeEvent:
    def __init__(self, components, message_str):
        self.message_obj = FakeMessageObj()
        self.message_obj.message = components
        self.message_str = message_str
        self.session_id = "aiocqhttp:GroupMessage:123456"
        self.role = "admin"
        self._extras = {}
        self.sent = []

    async def send(self, result):
        self.sent.append(result)

    def stop_event(self):
        self.stopped = True

    def get_messages(self):
        return self.message_obj.message

    def get_sender_id(self):
        return "10086"

    def get_group_id(self):
        return "123456"

    def get_session_id(self):
        return self.session_id

    def get_platform_id(self):
        return "aiocqhttp"

    def is_admin(self):
        return True

    def set_extra(self, key, value):
        self._extras[key] = value

    def get_extra(self, key, default=None):
        return self._extras.get(key, default)

    def plain_result(self, text):
        return {"kind": "plain", "text": text}

    def chain_result(self, chain):
        return {"kind": "chain", "chain": chain}


def build_config():
    key = os.environ["GPT_IMAGE_TEST_KEY"]
    return FakeConfig({
        "api": {
            "base_url": os.environ.get("GPT_IMAGE_TEST_BASE", "http://127.0.0.1:3425"),
            "api_key": key,
            "image_model": "gpt-image-2",
            "fallback_models": [],
            "timeout_seconds": 300,
            "extra_body": {},
        },
        "generation": {"size": "1024x1024", "quality": "low", "reference_mode": "auto"},
        "access": {"generate_admin_only": True, "admin_user_ids": "", "show_prompt_in_reply": False},
        "limits": {"cooldown_seconds": 0, "max_daily_per_user": 100, "max_concurrency": 2},
        "diagnosis": {"provider_id": "", "timeout_seconds": 120, "max_output_chars": 6000, "max_daily_per_user": 30},
    })


def inspect_result(result):
    """Return (text, image_path) from a plain/chain result dict."""
    if result["kind"] == "plain":
        return result["text"], None
    chain = result["chain"]
    texts = [c.text for c in chain if isinstance(c, Plain)]
    images = [c for c in chain if isinstance(c, Image)]
    path = images[0].path if images else None
    return "".join(texts), path


async def collect(agen):
    return [r async for r in agen]


async def finish_jobs(plugin):
    """Wait for every detached job; returns nothing, asserts tasks clean up."""
    tasks = list(plugin._jobs.values())
    if tasks:
        await asyncio.gather(*tasks, return_exceptions=True)
    assert not plugin._jobs, "detached jobs not cleaned up"


async def run():
    data_dir = Path(tempfile.mkdtemp(prefix="gptimg-"))
    original = StarTools.get_data_dir
    StarTools.get_data_dir = classmethod(lambda cls, name=None: data_dir)
    plugin = plugin_main.GptImageStudio(context=None, config=build_config())
    await plugin.initialize()

    try:
        # 1) pure text-to-image
        ev = FakeEvent([Plain("/生图 创建 一只戴墨镜的橘猫，扁平插画风")], "/生图 创建 一只戴墨镜的橘猫，扁平插画风")
        await collect(plugin.cmd_generate(ev))
        await finish_jobs(plugin)
        text, path = inspect_result(ev.sent[-1])
        assert path and os.path.isfile(path), f"no image produced: {text}"
        print("1. text-to-image OK ->", text, "|", Path(path).name)

        # 2) multi-image in ONE message => edit
        png1, png2 = _png((220, 50, 50)), _png((50, 50, 220))
        ev = FakeEvent(
            [Plain("/生图 编辑 把这两张图里各放一个圆点到中央"), Image.fromBytes(png1), Image.fromBytes(png2)],
            "/生图 编辑 把这两张图里各放一个圆点到中央",
        )
        await collect(plugin.cmd_generate(ev))
        await finish_jobs(plugin)
        text, path = inspect_result(ev.sent[-1])
        assert path and "参考图 2 张" in text, f"expected 2 references: {text}"
        print("2. multi-image edit OK ->", text)

        # 3) quoted message with two distinct images => /改图
        reply = Reply(id="1", sender_nickname="甲")
        reply.chain = [Image.fromBytes(png1), Image.fromBytes(png2)]
        ev = FakeEvent([reply, Plain("/改图 把背景换成海边")], "/改图 把背景换成海边")
        await collect(plugin.cmd_edit(ev))
        await finish_jobs(plugin)
        text, path = inspect_result(ev.sent[-1])
        assert path and "参考图 2 张" in text, f"quoted 2 refs: {text}"
        print("3. quoted multi-image edit OK ->", text)

        # 4) /画风 sugar requires a reference
        ev = FakeEvent([Plain("/画风 水彩")], "/画风 水彩")
        await collect(plugin.cmd_style(ev))
        await finish_jobs(plugin)
        text, path = inspect_result(ev.sent[-1])
        assert path is None and "参考图" in text, f"style without ref should ask: {text}"
        print("4. /画风 without ref guarded OK ->", text)

        # 5) 改图 with no image at all => friendly error
        ev = FakeEvent([Plain("/改图 加个帽子")], "/改图 加个帽子")
        await collect(plugin.cmd_edit(ev))
        await finish_jobs(plugin)
        text, path = inspect_result(ev.sent[-1])
        assert path is None and "参考图" in text, text
        print("5. /改图 no ref guarded OK ->", text)

        print("PLUGIN E2E ALL PASSED")
    finally:
        StarTools.get_data_dir = original
        await plugin.terminate()


if __name__ == "__main__":
    if "GPT_IMAGE_TEST_KEY" not in os.environ:
        print("Set GPT_IMAGE_TEST_KEY to run")
        sys.exit(0)
    asyncio.run(run())
