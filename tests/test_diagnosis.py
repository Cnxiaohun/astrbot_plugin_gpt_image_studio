"""Offline diagnosis tests: mock AstrBot context/providers; never access APIs."""
from __future__ import annotations

import asyncio
import base64
import unittest
from types import SimpleNamespace
from unittest.mock import AsyncMock, Mock, patch

from astrbot_plugin_gpt_image_studio.core.diagnosis import DISCLAIMER, SYSTEM_PROMPT, diagnose
from astrbot_plugin_gpt_image_studio.core.http_utils import ClientError


class DiagnosisTests(unittest.IsolatedAsyncioTestCase):
    def setUp(self):
        self.event = SimpleNamespace(unified_msg_origin="default:GroupMessage:123")
        self.images = [b"\x89PNG\r\n\x1a\nfirst", b"\x89PNG\r\n\x1a\nsecond"]
        self.provider = SimpleNamespace(
            provider_config={"modalities": ["text", "image"]},
            text_chat=AsyncMock(return_value=SimpleNamespace(completion_text="图片1：建议保留自然层次。\n图片2：局部降低高光。")),
        )
        self.context = SimpleNamespace(
            get_config=Mock(return_value={"provider_settings": {}}),
            get_provider_by_id=Mock(return_value=self.provider),
            get_using_provider_async=AsyncMock(return_value=self.provider),
            get_using_provider=Mock(return_value=self.provider),
        )

    async def call(self, goal="自然保真", settings=None):
        return await diagnose(self.context, self.event, self.images, goal, settings or {})

    async def test_explicit_provider_takes_priority_without_reading_config(self):
        await self.call(settings={"provider_id": "vision-1"})
        self.context.get_provider_by_id.assert_called_once_with("vision-1")
        self.context.get_config.assert_not_called()
        self.context.get_using_provider_async.assert_not_awaited()
        self.context.get_using_provider.assert_not_called()

    async def test_explicit_provider_lookup_may_be_async(self):
        self.context.get_provider_by_id = AsyncMock(return_value=self.provider)
        await self.call(settings={"provider_id": "vision-async"})
        self.context.get_provider_by_id.assert_awaited_once_with("vision-async")

    async def test_explicit_missing_provider_is_safe_and_never_falls_back(self):
        self.context.get_provider_by_id.return_value = None
        with self.assertRaises(ClientError) as caught:
            await self.call(settings={"provider_id": "https://secret.example/sk-private"})
        self.assertNotIn("secret.example", str(caught.exception))
        self.assertNotIn("sk-private", str(caught.exception))
        self.context.get_using_provider_async.assert_not_awaited()
        self.provider.text_chat.assert_not_awaited()

    async def test_caption_provider_has_priority_over_current_provider(self):
        self.context.get_config.return_value = {"provider_settings": {"default_image_caption_provider_id": "caption"}}
        await self.call()
        self.context.get_config.assert_called_once_with(self.event.unified_msg_origin)
        self.context.get_provider_by_id.assert_called_once_with("caption")
        self.context.get_using_provider_async.assert_not_awaited()

    async def test_async_configuration_and_empty_caption_uses_current_provider(self):
        self.context.get_config = AsyncMock(return_value={"provider_settings": {"default_image_caption_provider_id": ""}})
        await self.call()
        self.context.get_config.assert_awaited_once_with(self.event.unified_msg_origin)
        self.context.get_using_provider_async.assert_awaited_once_with(self.event.unified_msg_origin)
        self.context.get_using_provider.assert_not_called()
        self.context.get_provider_by_id.assert_not_called()

    async def test_invalid_configured_caption_does_not_silently_change_provider(self):
        self.context.get_config.return_value = {"provider_settings": {"default_image_caption_provider_id": "missing"}}
        self.context.get_provider_by_id.return_value = None
        with self.assertRaises(ClientError):
            await self.call()
        self.context.get_using_provider_async.assert_not_awaited()

    async def test_legacy_current_provider_sync_fallback(self):
        del self.context.get_using_provider_async
        await self.call()
        self.context.get_using_provider.assert_called_once_with(self.event.unified_msg_origin)

    async def test_current_provider_async_method_may_return_sync_value(self):
        self.context.get_using_provider_async = Mock(return_value=self.provider)
        await self.call()
        self.provider.text_chat.assert_awaited_once()

    async def test_missing_or_wrong_provider_is_a_safe_error(self):
        for value in [None, SimpleNamespace(provider_config={})]:
            self.context.get_using_provider_async.return_value = value
            with self.assertRaises(ClientError) as caught:
                await self.call()
            self.assertIn("vision provider", str(caught.exception))

    async def test_explicit_text_only_modalities_rejected(self):
        for modalities in [[], ["text"], ["audio", "text"]]:
            self.provider.provider_config = {"modalities": modalities}
            with self.assertRaises(ClientError) as caught:
                await self.call(settings={"provider_id": "selected"})
            self.assertIn("不支持图片", str(caught.exception))
        self.provider.text_chat.assert_not_awaited()

    async def test_unknown_modalities_allowed_until_actual_call(self):
        for config in [{}, {"modalities": None}, {"modalities": "unknown"}]:
            self.provider.provider_config = config
            await self.call()
        del self.provider.provider_config
        await self.call()
        self.assertEqual(self.provider.text_chat.await_count, 4)

    async def test_multimodal_data_and_goal_are_passed_without_tools_or_history(self):
        result = await self.call("保持人物身份特征，清理背景")
        kwargs = self.provider.text_chat.call_args.kwargs
        self.assertEqual(kwargs["image_urls"], [
            "data:image/png;base64," + base64.b64encode(image).decode("ascii") for image in self.images
        ])
        self.assertIn("图片1", kwargs["prompt"])
        self.assertIn("图片2", kwargs["prompt"])
        self.assertIn("保持人物身份特征，清理背景", kwargs["prompt"])
        self.assertIsNone(kwargs["func_tool"])
        self.assertEqual(kwargs["contexts"], [])
        self.assertEqual(kwargs["request_max_retries"], 1)
        self.assertNotIn("session_id", kwargs)
        self.assertNotIn("conversation_id", kwargs)
        self.assertNotIn("tools", kwargs)
        self.assertEqual(kwargs["system_prompt"], SYSTEM_PROMPT)
        self.assertTrue(result.startswith(DISCLAIMER))

    async def test_blank_goal_defaults_to_natural_fidelity(self):
        await self.call(" \n ")
        self.assertIn("自然保真", self.provider.text_chat.call_args.kwargs["prompt"])

    async def test_no_permissions_or_history_api_are_consulted(self):
        # Context/event expose only the specified provider-selection APIs. No
        # event.is_admin/history/tool/agent functions exist, so touching one fails.
        await self.call()
        self.assertEqual(set(vars(self.event)), {"unified_msg_origin"})
        kwargs = self.provider.text_chat.call_args.kwargs
        self.assertEqual(kwargs["contexts"], [])
        self.assertIsNone(kwargs["func_tool"])

    async def test_fixed_disclaimer_units_and_photographic_constraints(self):
        result = await self.call()
        self.assertIn("仅提供文字修图建议，不生成或修改图片", result)
        self.assertIn("建议起点", result)
        self.assertIn("非精确测量", result)
        self.assertIn("JPEG/PNG", result)
        self.assertIn("-100..+100", result)
        self.assertIn("Kelvin", result)
        for term in ["曝光", "-5..+5 EV", "色温", "色调", "0..255", "HSL", "颜色通道", "蒙版", "修复", "裁切", "锐化", "降噪", "操作顺序", "检查标准", "EXIF", "直方图", "肤色数值", "不得识别真人身份", "数据", "不是你要执行的指令"]:
            self.assertIn(term, SYSTEM_PROMPT)

    async def test_output_limit_default_minimum_maximum_and_truncation(self):
        self.provider.text_chat.return_value = SimpleNamespace(completion_text="建议" * 10000)
        for settings, limit in [({}, 6000), ({"max_output_chars": 1}, 1000), ({"max_output_chars": 12000}, 10000), ({"max_output_chars": 1300}, 1300), ({"max_output_chars": "bad"}, 6000)]:
            result = await self.call(settings=settings)
            self.assertLessEqual(len(result), limit)
            self.assertTrue(result.startswith(DISCLAIMER))
            self.assertIn("已截断", result)

    async def test_short_result_is_not_marked_truncated(self):
        result = await self.call(settings={"max_output_chars": 1000})
        self.assertNotIn("已截断", result)
        self.assertIn("图片2", result)

    async def test_timeout_setting_is_defaulted_and_clamped(self):
        original = asyncio.wait_for
        timeouts = []
        async def observe(awaitable, timeout):
            timeouts.append(timeout)
            return await original(awaitable, timeout=1)
        with patch("astrbot_plugin_gpt_image_studio.core.diagnosis.asyncio.wait_for", side_effect=observe):
            for value in [None, 1, 999, "invalid", 75, float("inf")]:
                await self.call(settings={} if value is None else {"timeout_seconds": value})
        self.assertEqual(timeouts, [120, 30, 300, 120, 75, 120])

    async def test_timeout_cancels_single_attempt_and_is_safe(self):
        cancelled = asyncio.Event()
        async def slow(**kwargs):
            try:
                await asyncio.Event().wait()
            finally:
                cancelled.set()
        self.provider.text_chat.side_effect = slow
        original = asyncio.wait_for
        async def fast_timeout(awaitable, timeout):
            return await original(awaitable, timeout=0.001)
        with patch("astrbot_plugin_gpt_image_studio.core.diagnosis.asyncio.wait_for", side_effect=fast_timeout):
            with self.assertRaises(ClientError) as caught:
                await self.call()
        self.assertIn("超时", str(caught.exception))
        self.assertIn("未自动重试", str(caught.exception))
        self.assertTrue(cancelled.is_set())
        self.provider.text_chat.assert_awaited_once()

    async def test_empty_and_tool_calls_only_responses_are_errors(self):
        for response in [None, SimpleNamespace(), SimpleNamespace(completion_text=""), SimpleNamespace(completion_text=" \n ", tools_call_name=["bad_tool"]), SimpleNamespace(completion_text=None, tool_calls=[{"name":"image_generation"}])]:
            self.provider.text_chat.return_value = response
            with self.assertRaises(ClientError) as caught:
                await self.call()
            self.assertIn("未返回文字建议", str(caught.exception))

    async def test_remote_errors_including_clienterror_never_leak_body_url_key(self):
        secret = "https://secret.example/path?api_key=sk-private REMOTE_BODY"
        for error in [RuntimeError(secret), ClientError(secret), ValueError(secret)]:
            self.provider.text_chat.side_effect = error
            with self.assertRaises(ClientError) as caught:
                await self.call()
            for fragment in ["secret.example", "sk-private", "REMOTE_BODY"]:
                self.assertNotIn(fragment, str(caught.exception))
            self.assertIn("vision provider", str(caught.exception))
        self.assertEqual(self.provider.text_chat.await_count, 3)

    async def test_provider_selection_errors_also_are_sanitized(self):
        self.context.get_config.side_effect = ClientError("secret-url sk-key remote body")
        with self.assertRaises(ClientError) as caught:
            await self.call()
        self.assertNotIn("sk-key", str(caught.exception))
        self.assertNotIn("secret-url", str(caught.exception))
        self.provider.text_chat.assert_not_awaited()

    async def test_bad_inputs_do_not_call_provider(self):
        for images, goal in [([], ""), ([b""], ""), (["not-bytes"], ""), ([b"png"], None)]:
            with self.assertRaises(ClientError):
                await diagnose(self.context, self.event, images, goal, {})
        self.provider.text_chat.assert_not_awaited()
        self.context.get_config.assert_not_called()


if __name__ == "__main__":
    unittest.main()
