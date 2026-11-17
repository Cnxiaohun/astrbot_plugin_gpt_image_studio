"""Command integration for text-only multimodal diagnosis; no real paid calls."""
import asyncio
import unittest
from unittest.mock import AsyncMock, patch

from astrbot_plugin_gpt_image_studio.core.prompts import parse_studio_command
from astrbot_plugin_gpt_image_studio.main import GptImageStudio
from astrbot_plugin_gpt_image_studio.tests.test_dispatch import Event, plugin, finish_jobs


class DiagnosisCommandTests(unittest.IsolatedAsyncioTestCase):
    def test_optional_goal_and_ref_options(self):
        for word in ['诊断','修图建议','diagnose','advice']:
            req=parse_studio_command(word)
            self.assertEqual(req.action,'diagnose')
            self.assertEqual(req.prompt,'')
        req=parse_studio_command('诊断 --参考 3,1 调成通透自然')
        self.assertEqual(req.references,(3,1))
        self.assertEqual(req.prompt,'调成通透自然')
        for flags in ['--比例 2:3','--质量 高']:
            with self.assertRaises(ValueError):parse_studio_command('诊断 '+flags)

    async def test_diagnosis_dispatch_does_not_enter_generation(self):
        p=plugin()
        p._generate=AsyncMock()
        p._diagnose=AsyncMock(return_value='曝光建议 +0.3 EV')
        ev=Event('/生图 诊断 --参考 2 自然人像')
        out=[r async for r in p.slash_fallback(ev)]
        self.assertTrue(ev.stopped)
        self.assertEqual(out, [])
        self.assertEqual(len(p._jobs), 1)
        p._diagnose.assert_not_awaited()
        await finish_jobs(p)
        self.assertIn('曝光建议', ev.send.call_args.args[0].text)
        p._generate.assert_not_awaited()
        p._diagnose.assert_awaited_once_with(ev,'自然人像',(2,))
        self.assertFalse(p._jobs)

    async def test_diagnosis_reads_selected_images_and_uses_separate_quota(self):
        p=plugin();p._sem=asyncio.Semaphore(1)
        p.quota.max_daily=1;p.quota.commit('test-user')
        ev=Event('/生图 诊断 --参考 3,1')
        async def model(context,event,images,goal,settings):
            self.assertEqual(ev.send.await_count,1)
            self.assertIn('只提供文字',ev.send.call_args.args[0].text)
            self.assertEqual(images,[b'three',b'one'])
            return '建议起点：曝光+0.3 EV。'
        with patch('astrbot_plugin_gpt_image_studio.main.gather_references',new=AsyncMock(return_value=[b'one',b'two',b'three'])),patch('astrbot_plugin_gpt_image_studio.main.prepare_reference',new=AsyncMock(side_effect=lambda x:x)),patch('astrbot_plugin_gpt_image_studio.main.diagnose',side_effect=model) as diag:
            text=await p._diagnose(ev,'自然保真',(3,1))
        self.assertIn('曝光+0.3 EV',text)
        self.assertIn('3,1',text)
        self.assertEqual(p.quota.used_today('test-user'),1)
        self.assertEqual(p.diagnosis_quota.used_today('test-user'),1)
        self.assertFalse(p._busy_users)
        self.assertFalse(p._active_tasks)

    async def test_permission_missing_images_and_invalid_selection_do_not_call_model(self):
        p=plugin();p._sem=asyncio.Semaphore(1)
        with patch('astrbot_plugin_gpt_image_studio.main.diagnose',new=AsyncMock()) as diag:
            ev=Event('/生图 诊断',admin=False)
            self.assertIn('仅管理员',await p._diagnose(ev))
            ev.send.assert_not_awaited()
            with patch('astrbot_plugin_gpt_image_studio.main.gather_references',new=AsyncMock(return_value=[])):
                self.assertIn('没有找到参考图',await p._diagnose(Event('/生图 诊断')))
            with patch('astrbot_plugin_gpt_image_studio.main.gather_references',new=AsyncMock(return_value=[b'one'])):
                self.assertIn('编号超出范围',await p._diagnose(Event('/生图 诊断'),references=(3,)))
            diag.assert_not_awaited()
        self.assertEqual(p.diagnosis_quota.used_today('test-user'),0)

    async def test_model_error_is_safe_and_not_retried(self):
        p=plugin();p._sem=asyncio.Semaphore(1)
        with patch('astrbot_plugin_gpt_image_studio.main.gather_references',new=AsyncMock(return_value=[b'one'])),patch('astrbot_plugin_gpt_image_studio.main.prepare_reference',new=AsyncMock(return_value=b'one')),patch('astrbot_plugin_gpt_image_studio.main.diagnose',new=AsyncMock(side_effect=RuntimeError('SECRET'))) as diag:
            text=await p._diagnose(Event('/生图 诊断'))
        self.assertNotIn('SECRET',text)
        self.assertIn('未调用生图接口',text)
        self.assertEqual(diag.await_count,1)
        self.assertFalse(p._busy_users)

    async def test_text_parts_and_tool_single_attempt(self):
        p=plugin();p._diagnose=AsyncMock(return_value='参数建议。'*1000)
        ev=Event('帮我给PS参数建议')
        result=await p.tool_diagnose(ev,'自然')
        self.assertIn('已受理',result)
        self.assertEqual(len(p._jobs),1)
        ev.send.assert_not_awaited()
        self.assertIn('已尝试',await p.tool_diagnose(ev,'自然'))
        await finish_jobs(p)
        self.assertGreater(ev.send.await_count,1)
        parts=[call.args[0].text for call in ev.send.call_args_list]
        for part in parts:self.assertLessEqual(len(part),1800)
        self.assertIn('参数建议。'*1000,''.join(parts))
        self.assertEqual(p._diagnose.await_count,1)
        self.assertFalse(p._jobs)


if __name__=='__main__':unittest.main()
