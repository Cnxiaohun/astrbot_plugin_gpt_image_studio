"""Task recovery and ownership without live generation or chat delivery."""
import asyncio
import tempfile
import time
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import AsyncMock, patch

from astrbot_plugin_gpt_image_studio.core.receipts import Receipts, recover_output
from astrbot_plugin_gpt_image_studio.core.store import Store
from astrbot_plugin_gpt_image_studio.tests.test_dispatch import Event, plugin


class ReceiptTests(unittest.IsolatedAsyncioTestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)
        self.db = self.root / 'job_receipts.json'

    def studio(self):
        p = plugin()
        p.store = Store(self.root)
        p.receipts = Receipts(self.db)
        p._sem = asyncio.Semaphore(1)
        p.client = SimpleNamespace(client=None, generate=AsyncMock(return_value=SimpleNamespace(
            data=b'test image', mime='image/png', width=10, height=10)))
        return p

    async def test_lost_delivery_after_restart_recovers_without_generation_or_quota(self):
        p = self.studio()
        ev = Event('/生图 创建 一只猫')
        ev.send.side_effect = [None, RuntimeError('delivery unknown')]
        job = p._submit_image(ev, '一只猫', reference_mode='none')
        await p._jobs[job]
        self.assertEqual(p.client.generate.await_count, 1)
        self.assertIn('发送结果不明', p._receipt_status(ev, job))
        restarted = self.studio()
        status = Event(f'/生图 状态 {job}')
        output = [r async for r in restarted.slash_fallback(status)]
        self.assertIn('发送结果不明', output[0].text)
        self.assertIn('补发', output[0].text)
        resend = Event(f'/生图 补发 {job}')
        output = [r async for r in restarted.slash_fallback(resend)]
        self.assertIn('未调用生图模型', output[0].text)
        resend.send.assert_awaited_once()
        image = resend.send.call_args.args[0].chain[-1]
        self.assertEqual(Path(image.path).read_bytes(), b'test image')
        restarted.client.generate.assert_not_awaited()
        self.assertEqual(restarted.quota.used_today(resend.get_sender_id()), 0)
        self.assertNotIn('一只猫', self.db.read_text())

    async def test_billing_uncertainty_survives_sending_the_error_notice(self):
        from astrbot_plugin_gpt_image_studio.core.http_utils import AmbiguousRequestError

        p = self.studio()
        ev = Event('')
        p.client.generate.side_effect = AmbiguousRequestError('上游超时，可能已计费')
        job = p._submit_image(ev, '一只猫', reference_mode='none')
        await p._jobs[job]
        restarted = self.studio()
        self.assertIn('出图或计费结果不明', restarted._receipt_status(ev, job))
        self.assertIn('没有可取回', await restarted._resend_output(ev, job))
        self.assertEqual(p.client.generate.await_count, 1)
        restarted.client.generate.assert_not_awaited()

    async def test_task_not_disclosed_to_different_sender_or_session(self):
        p = self.studio()
        owner = Event('')
        job = 'aabbccdd11'
        output = p.store.save(b'test', 'image/png')
        p.receipts.put(job, p._receipt_scope(owner), '已出图', output=output.name, final=True)
        for attr in ('get_session_id', 'get_sender_id'):
            other = Event('')
            setattr(other, attr, lambda: 'other')
            self.assertIn('未找到', p._receipt_status(other, job))
            self.assertIn('未找到', await p._resend_output(other, job))
            other.send.assert_not_awaited()

    async def test_in_progress_and_missing_file_do_not_send(self):
        p = self.studio()
        owner = Event('')
        job = 'aabbccdd11'
        p.receipts.put(job, p._receipt_scope(owner), '生成中')
        self.assertIn('请稍候', await p._resend_output(owner, job))
        p.receipts.put(job, p._receipt_scope(owner), '已出图', output='20260101-120000-aabbccdd.png', final=True)
        self.assertIn('已清理', p._receipt_status(owner, job))
        self.assertIn('未重新生成', await p._resend_output(owner, job))
        owner.send.assert_not_awaited()

    def test_restart_does_not_recover_execution_and_receipts_expire(self):
        receipts = Receipts(self.db)
        with patch('astrbot_plugin_gpt_image_studio.core.receipts.time.time', return_value=time.time() - 86401):
            receipts.put('aabbccdd11', 'session', '已出图', final=True)
        receipts.put('aabbccdd22', 'session', '生成中')
        receipts = Receipts(self.db)
        self.assertIsNone(receipts.get('aabbccdd11', 'session'))
        self.assertIn('中断', receipts.get('aabbccdd22', 'session')['status'])

    def test_no_path_escape_or_symlink_recovery(self):
        root = self.root / 'outputs'
        root.mkdir()
        outside = self.root / 'private.png'
        outside.write_bytes(b'private')
        name = '20260101-120000-aabbccdd.png'
        (root / name).symlink_to(outside)
        self.assertIsNone(recover_output(root, name))
        self.assertIsNone(recover_output(root, '../private.png'))
