"""Unified command dispatch, reference selection and replay regression tests."""
import asyncio
import time
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import AsyncMock, Mock, patch

from astrbot_plugin_gpt_image_studio.main import GptImageStudio, JobSnapshot
from astrbot_plugin_gpt_image_studio.core.prompts import parse_studio_command
from astrbot_plugin_gpt_image_studio.tests.test_dispatch import Event, plugin, finish_jobs


class StudioTests(unittest.IsolatedAsyncioTestCase):
    async def test_canonical_command_matrix(self):
        for action, mode, required, minimum in [('创建','none',False,0),('编辑','auto',True,0),('风格','auto',True,0),('融合','auto',True,2)]:
            p = plugin()
            p._generate = AsyncMock(return_value=(None, 'mock-result'))
            ev = Event(f'/生图 {action} --比例 2:3 --质量 高 海报')
            out = [r async for r in p.slash_fallback(ev)]
            self.assertEqual(out, [])
            self.assertEqual(len(p._jobs), 1)
            p._generate.assert_not_awaited()
            await finish_jobs(p)
            self.assertIn('mock-result', ev.send.call_args.args[0].text)
            self.assertFalse(p._jobs)
            kw = p._generate.call_args.kwargs
            self.assertEqual((kw['reference_mode'], kw['force_reference'], kw['min_references']), (mode,required,minimum))
            self.assertEqual(kw['request'].size, '1024x1536')
            self.assertEqual(kw['request'].quality, 'high')
            if action == '风格': self.assertIn('画风', kw['request'].prompt)
            self.assertTrue(ev.stopped)

    async def test_help_status_and_invalid_commands_never_generate(self):
        for tail in ['', '帮助', '状态', '错别字 一只猫', '创建 --未知 高 猫', '编辑']:
            p = plugin()
            p._generate = AsyncMock()
            out = [r async for r in p.slash_fallback(Event('/生图' + (' '+tail if tail else '')))]
            self.assertTrue(out)
            p._generate.assert_not_awaited()
            self.assertFalse(p._jobs)

    def configured(self):
        p = plugin()
        p._sem = asyncio.Semaphore(1)
        result = SimpleNamespace(data=b'output', mime='image/png', width=10, height=10)
        p.client = SimpleNamespace(client=None, edit=AsyncMock(return_value=result), generate=AsyncMock(return_value=result))
        p.store = SimpleNamespace(save=Mock(return_value=Path('/tmp/studio-test-output.png')))
        return p

    async def test_reference_selection_and_cached_retry_use_same_bytes(self):
        p = self.configured()
        ev = Event('/生图 编辑 --参考 3,1 --比例 2:3 海报')
        req = parse_studio_command('编辑 --参考 3,1 --比例 2:3 海报')
        with patch('astrbot_plugin_gpt_image_studio.main.gather_references', new=AsyncMock(return_value=[b'one',b'two',b'three'])), patch('astrbot_plugin_gpt_image_studio.main.prepare_reference', new=AsyncMock(side_effect=lambda x:x)):
            path, _ = await p._generate(ev, req.prompt, request=req, reference_mode='auto', force_reference=True)
        self.assertIsNotNone(path)
        args = p.client.edit.call_args
        self.assertEqual(args.args[1], [b'three',b'one'])
        self.assertEqual(args.kwargs['size'], '1024x1536')
        job = p._previous_job(ev)
        self.assertEqual(job.refs, (b'three',b'one'))
        with patch('astrbot_plugin_gpt_image_studio.main.gather_references', new=AsyncMock()) as gather:
            path, _ = await p._generate(ev, job.prompt, replay=job)
        gather.assert_not_awaited()
        self.assertIsNotNone(path)
        self.assertEqual(p.client.edit.call_args, args)

    async def test_invalid_reference_or_merge_count_not_billed(self):
        for command, refs, minimum in [('编辑 --参考 3 海报',[b'one'],0),('融合 海报',[b'one'],2)]:
            p = self.configured()
            req = parse_studio_command(command)
            with patch('astrbot_plugin_gpt_image_studio.main.gather_references', new=AsyncMock(return_value=refs)), patch('astrbot_plugin_gpt_image_studio.main.prepare_reference', new=AsyncMock(side_effect=lambda x:x)):
                path, note = await p._generate(Event('/生图 '+command), req.prompt, request=req, force_reference=True, min_references=minimum)
            self.assertIsNone(path)
            p.client.edit.assert_not_awaited()
            p.client.generate.assert_not_awaited()
            self.assertIn('未提交',note)

    async def test_create_ignores_attached_references(self):
        p = self.configured()
        req = parse_studio_command('创建 猫')
        from astrbot_plugin_gpt_image_studio.core.references import gather_references
        with patch('astrbot_plugin_gpt_image_studio.main.gather_references', new=AsyncMock(wraps=gather_references)) as gather:
            path, note = await p._generate(Event('/生图 创建 猫'), req.prompt, request=req, reference_mode='none')
        p.client.generate.assert_awaited_once()
        p.client.edit.assert_not_awaited()
        self.assertEqual(gather.call_args.args[1],'none')

    async def test_retry_fee_notice_and_scoped_expiry(self):
        p = plugin()
        ev = Event('/生图 重试')
        snapshot = JobSnapshot('猫','1024x1024','high',(b'one',),time.monotonic())
        p._remember_job(ev,snapshot)
        p._generate = AsyncMock(return_value=(None,'mock'))
        out = [r async for r in p.slash_fallback(ev)]
        self.assertIn('再次计费',out[0].text)
        self.assertEqual(len(p._jobs), 1)
        p._generate.assert_not_awaited()
        await finish_jobs(p)
        self.assertIs(p._generate.call_args.kwargs['replay'],snapshot)
        self.assertIn('mock', ev.send.call_args.args[0].text)
        self.assertFalse(p._jobs)
        other = Event('/生图 重试')
        other.get_session_id = lambda:'another-group'
        self.assertIsNone(p._previous_job(other))
        other.get_sender_id = lambda:'another-user'
        self.assertIsNone(p._previous_job(other))
        p._remember_job(ev,JobSnapshot('猫','','',(),time.monotonic()-1801))
        self.assertIsNone(p._previous_job(ev))


if __name__ == '__main__':
    unittest.main()
