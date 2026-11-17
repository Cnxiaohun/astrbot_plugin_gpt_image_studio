"""Regression tests using real AstrBot filters, without paid/network calls.

Run from a temporary cwd with workspace and AstrBot checkout on PYTHONPATH.
"""
import asyncio
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import AsyncMock, patch

from astrbot.core.star.filter.command import CommandFilter
from astrbot.core.star.filter.regex import RegexFilter
from astrbot.core.star.star_handler import star_handlers_registry
from astrbot.core.star.register.star_handler import llm_tools
from astrbot_plugin_gpt_image_studio.main import GptImageStudio
from astrbot_plugin_gpt_image_studio.core.quota import Quota


class Event:
    def __init__(self, text, admin=True):
        self.message_str = text
        self.is_at_or_wake_command = False
        self.admin = admin
        self.extras = {}
        self.stopped = False
        self.send = AsyncMock()
    def get_message_str(self): return self.message_str
    def get_sender_id(self): return "test-user"
    def get_session_id(self): return "test-session"
    def get_group_id(self): return "test-group"
    def is_admin(self): return self.admin
    def get_messages(self): return []
    def stop_event(self): self.stopped = True
    def get_extra(self, key, default=None): return self.extras.get(key, default)
    def set_extra(self, key, value): self.extras[key] = value
    def plain_result(self, text): return SimpleNamespace(text=text)
    def chain_result(self, chain): return SimpleNamespace(chain=chain)


def plugin():
    p = GptImageStudio(None, {"access": {"generate_admin_only": True}})
    p.quota = Quota(0, 100)
    return p


async def finish_jobs(p):
    """Await the plugin-owned worker tasks without timing assumptions or sleeps."""
    await asyncio.gather(*list(p._jobs.values()))


def registered_filters(handler):
    return next(md.event_filters for md in star_handlers_registry
                if md.handler_name == handler and 'gpt_image_studio' in md.handler_module_path)


class DispatchTests(unittest.IsolatedAsyncioTestCase):
    def test_real_filter_empty_wake_prefix_regression(self):
        slash = registered_filters('slash_fallback')
        for text in ['/改图 加帽子', '/生图状态', '/生图帮助', '/修图 改背景']:
            ev = Event(text)
            self.assertTrue(all(f.filter(ev, {}) for f in slash), text)
        old = registered_filters('cmd_edit')
        self.assertFalse(all(f.filter(Event('/改图 加帽子'), {}) for f in old))
        for text in ['/改图片 加帽子', '我想/改图', '/生图状态错误']:
            self.assertFalse(all(f.filter(Event(text), {}) for f in slash), text)

    async def test_slash_status_stops_llm(self):
        p = plugin()
        ev = Event('/生图状态')
        out = [x async for x in p.slash_fallback(ev)]
        self.assertTrue(ev.stopped)
        self.assertIn('今日已用', out[0].text)
        self.assertEqual([], [x async for x in p.slash_fallback(ev)])

    async def test_slash_edit_dispatch_and_permission(self):
        p = plugin()
        p._generate = AsyncMock(return_value=(None, 'reference test'))
        ev = Event('/改图 给第2张加帽子')
        out = [x async for x in p.slash_fallback(ev)]
        self.assertTrue(ev.stopped)
        self.assertEqual(out, [])
        self.assertEqual(len(p._jobs), 1)
        p._generate.assert_not_awaited()
        await finish_jobs(p)
        p._generate.assert_awaited_once_with(ev, '给第2张加帽子', force_reference=True)
        self.assertIn('reference test', ev.send.call_args.args[0].text)
        self.assertFalse(p._jobs)
        ev = Event('/改图 加帽子', admin=False)
        out = [x async for x in p.slash_fallback(ev)]
        self.assertTrue(ev.stopped)
        self.assertIn('仅管理员', out[0].text)
        self.assertEqual(p._generate.await_count, 1)

    def test_llm_tools_registered_with_searchable_descriptions(self):
        for name in ['gpt_image_generate', 'gpt_image_edit']:
            tool = next(t for t in llm_tools.func_list if t.name == name)
            self.assertIn('GPT 生图大师', tool.description)
            self.assertIn('prompt', str(tool.parameters))

    async def test_tool_permission_and_single_attempt(self):
        p = plugin()
        p._generate = AsyncMock(return_value=(None, '模拟网络错误，勿重试'))
        denied = await p.tool_edit(Event('帮我改图', admin=False), '加帽子')
        self.assertIn('仅管理员', denied)
        p._generate.assert_not_awaited()
        ev = Event('帮我改图')
        self.assertIn('已受理', await p.tool_edit(ev, '加帽子'))
        self.assertEqual(len(p._jobs), 1)
        p._generate.assert_not_awaited()
        self.assertIn('勿重复', await p.tool_edit(ev, '加帽子'))
        await finish_jobs(p)
        self.assertEqual(p._generate.await_count, 1)
        self.assertIn('模拟网络错误', ev.send.call_args.args[0].text)
        self.assertEqual(ev.send.await_count, 1)
        self.assertFalse(p._jobs)

    async def test_tool_sends_real_result_chain(self):
        p = plugin()
        with tempfile.TemporaryDirectory() as d:
            path = Path(d) / 'image.png'
            path.write_bytes(b'test')
            p._generate = AsyncMock(return_value=(path, '出图完成'))
            ev = Event('帮我画猫')
            result = await p.tool_generate(ev, '猫')
            self.assertIn('已受理', result)
            self.assertEqual(len(p._jobs), 1)
            ev.send.assert_not_awaited()
            await finish_jobs(p)
            self.assertEqual(ev.send.await_count, 1)
            chain = ev.send.call_args.args[0].chain
            from astrbot.api.message_components import Image, Plain
            self.assertTrue(any(isinstance(c, Image) and c.path == str(path.resolve()) for c in chain))
            self.assertTrue(any(isinstance(c, Plain) and '出图完成' in c.text for c in chain))
            p._generate.assert_awaited_once_with(ev, '猫', force_reference=False, reference_mode='none')
            self.assertFalse(p._jobs)

    async def test_start_notice_precedes_reference_download(self):
        from astrbot_plugin_gpt_image_studio.core.http_utils import ClientError
        p = plugin()
        p.client = SimpleNamespace(client=None)
        p.store = object()
        p._sem = asyncio.Semaphore(1)
        ev = Event('/改图 加帽子')
        order = []
        async def send(message):
            order.append('notice')
            self.assertIn('正在处理生图任务', message.text)
            self.assertIn('本群', message.text)
        async def gather(*a, **k):
            order.append('download')
            raise ClientError('引用链接失效，未提交生图')
        ev.send.side_effect = send
        with patch('astrbot_plugin_gpt_image_studio.main.gather_references', side_effect=gather):
            path, note = await p._generate(ev, '加帽子', force_reference=True)
        self.assertEqual(order, ['notice', 'download'])
        self.assertIsNone(path)
        self.assertIn('未提交', note)
        self.assertEqual(ev.send.await_count, 1)

    async def test_denied_or_empty_prompt_sends_no_start_notice(self):
        p = plugin()
        p.client = SimpleNamespace(client=None)
        p.store = object()
        p._sem = asyncio.Semaphore(1)
        for text, admin in [('猫', False), ('', True)]:
            ev = Event(text, admin=admin)
            path, note = await p._generate(ev, text)
            self.assertIsNone(path)
            ev.send.assert_not_awaited()

    async def test_queue_notice_and_failed_notice_stop_before_paid_work(self):
        p = plugin()
        p.client = SimpleNamespace(client=None)
        p.store = object()
        p._sem = asyncio.Semaphore(1)
        await p._sem.acquire()
        ev = Event('/改图 加帽子')
        with patch('astrbot_plugin_gpt_image_studio.main.gather_references', new=AsyncMock(return_value=[])):
            path, note = await p._generate(ev, '加帽子', force_reference=True)
        self.assertIn('排队', ev.send.call_args.args[0].text)
        self.assertIsNone(path)
        p._sem.release()
        ev = Event('/生图 猫')
        ev.send.side_effect = RuntimeError('send failed')
        with patch('astrbot_plugin_gpt_image_studio.main.gather_references', new=AsyncMock()) as gather:
            path, note = await p._generate(ev, '猫')
        self.assertIn('任务未提交', note)
        gather.assert_not_awaited()
        self.assertFalse(p._busy_users)

    async def test_busy_user_cannot_submit_concurrently(self):
        p = plugin()
        started, finish = asyncio.Event(), asyncio.Event()
        async def work(*a, **k):
            started.set()
            await finish.wait()
            return None, 'done'
        p._run_generation = work
        ev = Event('/生图 猫')
        task = asyncio.create_task(p._generate(ev, '猫'))
        await started.wait()
        try:
            result = await p._generate(ev, '另一只猫')
            self.assertIn('进行中', result[1])
        finally:
            finish.set()
            await task
        self.assertFalse(p._busy_users)


if __name__ == '__main__':
    unittest.main()
