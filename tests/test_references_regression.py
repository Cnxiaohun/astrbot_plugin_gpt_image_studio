"""Offline reference regressions: mock HTTP/OneBot; never call image APIs."""
import asyncio
import base64
import io
import unittest
from types import SimpleNamespace
from unittest.mock import AsyncMock, patch

import httpx
from PIL import Image as PILImage
from astrbot.api.message_components import Image, Plain, Reply

try:
    from ..core import references as refs
except ImportError:
    from core import references as refs


def png(color="red"):
    output = io.BytesIO()
    PILImage.new("RGB", (8, 8), color).save(output, "PNG")
    return output.getvalue()


def event(chain=(), bot=None):
    return SimpleNamespace(get_messages=lambda: list(chain), get_group_id=lambda: "123", bot=bot)


class Stream(httpx.AsyncByteStream):
    def __init__(self, chunks):
        self.chunks = chunks
        self.reads = 0
        self.closed = False

    async def __aiter__(self):
        for chunk in self.chunks:
            self.reads += 1
            yield chunk

    async def aclose(self):
        self.closed = True


class ReferenceRegressions(unittest.IsolatedAsyncioTestCase):
    async def test_message_mode_reads_all_images(self):
        e = event([Plain("text"), Image.fromBytes(png()), Image.fromBytes(png("blue"))])
        self.assertEqual(await refs.gather_references(e, "message"), [png(), png("blue")])

    async def test_none_ignores_all_reference_inputs(self):
        bot = SimpleNamespace(call_action=AsyncMock())
        self.assertEqual(await refs.gather_references(event([Reply(id=1)], bot), "none"), [])
        bot.call_action.assert_not_called()

    async def test_auto_never_scans_unrelated_history(self):
        bot = SimpleNamespace(call_action=AsyncMock())
        self.assertEqual(await refs.gather_references(event(bot=bot), "auto"), [])
        bot.call_action.assert_not_called()

    async def test_bad_quote_never_falls_back_to_current_or_history(self):
        e = event([Reply(id="42"), Image.fromBytes(png())])
        with self.assertRaises(refs.ReferenceError):
            await refs.gather_references(e, "auto")

    async def test_get_msg_all_images_without_official_extractor(self):
        payload = {"group_id": 123, "message": [
            {"type": "image", "data": {"file": "base64://" + base64.b64encode(png(c)).decode()}}
            for c in ("red", "blue")
        ]}
        bot = SimpleNamespace(call_action=AsyncMock(return_value=payload))
        data = await refs.gather_references(event([Reply(id="42")], bot), "auto")
        self.assertEqual(data, [png(), png("blue")])
        bot.call_action.assert_awaited_once_with("get_msg", message_id="42")

    async def test_cross_group_quote_rejected_before_download(self):
        for group in (456, None):
            bot = SimpleNamespace(call_action=AsyncMock(return_value={"group_id": group, "message": [
                {"type": "image", "data": {"url": "https://images.test/x.png"}}
            ]}))
            with patch.object(refs, "_download", new_callable=AsyncMock) as dl:
                with self.assertRaises(refs.ReferenceError):
                    await refs.gather_references(event([Reply(id=1)], bot), "quote")
                dl.assert_not_called()

    async def test_quote_lookup_failure_is_not_silently_dropped(self):
        bot = SimpleNamespace(call_action=AsyncMock(side_effect=RuntimeError("not available")))
        with self.assertRaises(refs.ReferenceError):
            await refs.gather_references(event([Reply(id=1)], bot), "auto")

    async def test_one_bad_image_fails_whole_batch(self):
        for chain in ([Image.fromBytes(png()), Image.fromBytes(b"broken")],
                      [Image.fromBytes(png()), Image(file="expired-file-id")]):
            with self.assertRaises(refs.ReferenceError):
                await refs.gather_references(event(chain), "message")

    async def test_quote_one_bad_image_fails_whole_batch(self):
        quote = Reply(id=1)
        quote.chain = [Image.fromBytes(png()), Image.fromBytes(b"bad")]
        with self.assertRaises(refs.ReferenceError):
            await refs.gather_references(event([quote]), "quote")

    async def test_cap_rejected_not_truncated_even_duplicates(self):
        with self.assertRaisesRegex(refs.ReferenceError, "最多 6"):
            await refs.gather_references(event([Image.fromBytes(png())] * 7), "message")

    async def test_explicit_history_collects_and_checks_group(self):
        bot = SimpleNamespace(call_action=AsyncMock(return_value={"messages": [{
            "group_id": 123, "message": [{"type": "image", "data": {
                "file": "base64://" + base64.b64encode(png()).decode()}}]
        }]}))
        self.assertEqual(await refs.gather_references(event(bot=bot), "history"), [png()])
        self.assertEqual(bot.call_action.await_args.args, ("get_group_msg_history",))

    async def test_download_never_inherits_client_credentials(self):
        captured = []
        def handle(request):
            captured.append(request)
            return httpx.Response(200, stream=Stream([png()]))
        async with httpx.AsyncClient(
            transport=httpx.MockTransport(handle), headers={"Authorization": "Bearer secret", "x-api-key": "secret"},
            cookies={"session": "secret"}, auth=("user", "pass"), params={"api_key": "secret"},
        ) as client:
            self.assertEqual(await refs._download("https://images.test/x.png", client), png())
        request = captured[0]
        for name in ("authorization", "x-api-key", "cookie"):
            self.assertNotIn(name, request.headers)
        self.assertEqual(str(request.url), "https://images.test/x.png")

    async def test_content_length_limit_prevents_body_read(self):
        stream = Stream([b"x" * 2048])
        async with httpx.AsyncClient(transport=httpx.MockTransport(
            lambda r: httpx.Response(200, headers={"content-length": str(refs.MAX_IMAGE_BYTES + 1)}, stream=stream)
        )) as client:
            with self.assertRaises(refs.ReferenceError):
                await refs._download("https://images.test/x", client)
        self.assertEqual(stream.reads, 0)
        self.assertTrue(stream.closed)

    async def test_stream_limit_without_content_length(self):
        stream = Stream([b"x" * 65536] * 5)
        with patch.object(refs, "MAX_IMAGE_BYTES", 70000):
            async with httpx.AsyncClient(transport=httpx.MockTransport(
                lambda r: httpx.Response(200, stream=stream)
            )) as client:
                with self.assertRaises(refs.ReferenceError):
                    await refs._download("https://images.test/x", client)
        self.assertEqual(stream.reads, 2)
        self.assertTrue(stream.closed)

    async def test_no_redirect_or_url_credentials(self):
        calls = []
        def handle(request):
            calls.append(request)
            return httpx.Response(302, headers={"location": "https://elsewhere.test"})
        async with httpx.AsyncClient(transport=httpx.MockTransport(handle), follow_redirects=True) as client:
            with self.assertRaises(refs.ReferenceError):
                await refs._download("https://images.test/x", client)
            with self.assertRaises(refs.ReferenceError):
                await refs._download("https://user:secret@images.test/x", client)
        self.assertEqual(len(calls), 1)

    async def test_wrapped_get_msg_and_reference_cap_before_download(self):
        bot = SimpleNamespace(call_action=AsyncMock(return_value={"data": {
            "group_id": 123, "message": [{"type": "image", "data": {
                "url": "https://images.test/x"}}] * 7}}))
        with patch.object(refs, "_download", new_callable=AsyncMock) as download:
            with self.assertRaisesRegex(refs.ReferenceError, "最多 6"):
                await refs.gather_references(event([Reply(id=1)], bot), "quote")
            download.assert_not_called()

    async def test_download_failure_does_not_submit_partial_references(self):
        calls = []
        def handle(request):
            calls.append(str(request.url))
            if request.url.path == "/bad":
                return httpx.Response(404, stream=Stream([]))
            return httpx.Response(200, stream=Stream([png()]))
        async with httpx.AsyncClient(transport=httpx.MockTransport(handle)) as client:
            e = event([Image(file="https://images.test/good"), Image(file="https://images.test/bad")])
            with self.assertRaises(refs.ReferenceError):
                await refs.gather_references(e, "message", client)
        self.assertEqual(len(calls), 2)

    async def test_cross_group_history_is_rejected(self):
        bot = SimpleNamespace(call_action=AsyncMock(return_value={"messages": [{
            "group_id": 999, "message": []}]}))
        with self.assertRaises(refs.ReferenceError):
            await refs.gather_references(event(bot=bot), "history")

    async def test_do_not_use_unbounded_astrbot_converter(self):
        image = Image.fromBytes(png())
        with patch.object(Image, "convert_to_file_path", new_callable=AsyncMock) as converter:
            self.assertEqual(await refs.component_to_bytes(image), png())
            converter.assert_not_called()


if __name__ == "__main__":
    unittest.main()
