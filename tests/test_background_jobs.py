"""Detached task lifetime, same-user cross-group isolation and delivery tests."""
import asyncio
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import AsyncMock

from astrbot.api.message_components import At, Plain, Reply
from astrbot_plugin_gpt_image_studio.main import SlashCommandFilter
from astrbot_plugin_gpt_image_studio.core.http_utils import ClientError
from astrbot_plugin_gpt_image_studio.tests.test_dispatch import Event, plugin


def group_event(group):
    ev=Event('/生图 创建 猫')
    ev.get_session_id=lambda:group
    ev.get_group_id=lambda:group
    return ev


class BackgroundJobTests(unittest.IsolatedAsyncioTestCase):
    async def test_same_user_two_groups_finish_reverse_order_and_return_to_original_events(self):
        p=plugin();a,b=group_event('group-a'),group_event('group-b')
        starts={g:asyncio.Event() for g in ['group-a','group-b']}
        finish={g:asyncio.Event() for g in starts}
        with tempfile.TemporaryDirectory() as d:
            pa,pb=Path(d)/'a.png',Path(d)/'b.png'
            pa.write_bytes(b'a');pb.write_bytes(b'b')
            async def generate(ev,prompt,**kwargs):
                g=ev.get_group_id();starts[g].set();await finish[g].wait()
                return (pa if g=='group-a' else pb),g
            p._run_generation=generate
            ra=await asyncio.wait_for(p.tool_generate(a,'猫A'),0.1)
            rb=await asyncio.wait_for(p.tool_generate(b,'猫B'),0.1)
            self.assertIn('已受理',ra);self.assertIn('已受理',rb)
            await asyncio.gather(*(x.wait() for x in starts.values()))
            self.assertEqual(len(p._busy_users),2)
            ja,jb=a.get_extra('gpt_image_job_id'),b.get_extra('gpt_image_job_id')
            ta,tb=p._jobs[ja],p._jobs[jb]
            finish['group-b'].set();await tb
            self.assertEqual(a.send.await_count,0)
            self.assertEqual(b.send.await_count,1)
            finish['group-a'].set();await ta
            self.assertEqual(a.send.await_count,1)
            for ev,path,job in [(a,pa,ja),(b,pb,jb)]:
                chain=ev.send.call_args.args[0].chain
                self.assertIn(job,chain[0].text)
                self.assertEqual(chain[1].path,str(path))
                self.assertEqual(p._job_info[job]['status'],'平台发送已返回')
            self.assertFalse(p._busy_users)
            self.assertFalse(p._jobs)
            self.assertFalse(p._pending_scopes)

    async def test_parent_cancel_does_not_cancel_detached_generation(self):
        p=plugin();ev=group_event('group-a');started=asyncio.Event();release=asyncio.Event()
        async def work(*a,**k):started.set();await release.wait();return None,'done'
        p._run_generation=work
        async def parent():
            await p.tool_generate(ev,'猫')
            await asyncio.Event().wait()
        caller=asyncio.create_task(parent())
        await started.wait();job_id=ev.get_extra('gpt_image_job_id');worker=p._jobs[job_id]
        caller.cancel()
        with self.assertRaises(asyncio.CancelledError):await caller
        self.assertFalse(worker.cancelled());self.assertFalse(worker.done())
        release.set();await worker
        self.assertIn('done',ev.send.call_args.args[0].text)

    async def test_same_scope_reserved_before_worker_can_start(self):
        p=plugin();ev=group_event('group-a');release=asyncio.Event()
        async def work(*a,**k):await release.wait();return None,'done'
        p._run_generation=AsyncMock(side_effect=work)
        first=p._submit_image(ev,'猫')
        with self.assertRaises(ClientError):p._submit_image(group_event('group-a'),'猫')
        task=p._jobs[first];release.set();await task
        self.assertEqual(p._run_generation.await_count,1)

    async def test_delivery_failure_is_uncertain_no_regeneration_or_resend(self):
        p=plugin();ev=group_event('group-a')
        p._generate=AsyncMock(return_value=(Path('/tmp/test-saved-result.png'),'done'))
        ev.send.side_effect=RuntimeError('transport ambiguous')
        job=p._submit_image(ev,'猫');await p._jobs[job]
        self.assertEqual(p._generate.await_count,1)
        self.assertEqual(ev.send.await_count,1)
        self.assertEqual(p._job_info[job]['status'],'发送结果不明')
        self.assertEqual(p._job_info[job]['path'],'/tmp/test-saved-result.png')

    async def test_terminate_cancels_model_and_delivery_and_rejects_new_work(self):
        for phase in ['model','delivery']:
            p=plugin();p.client=SimpleNamespace(close=AsyncMock())
            ev=group_event('group-a');started=asyncio.Event()
            async def work(*a,**k):
                if phase=='model':started.set();await asyncio.Event().wait()
                return None,'done'
            async def send(*a,**k):started.set();await asyncio.Event().wait()
            p._run_generation=work
            if phase=='delivery':ev.send.side_effect=send
            job=p._submit_image(ev,'猫');task=p._jobs[job]
            await started.wait();await p.terminate()
            self.assertTrue(task.cancelled())
            self.assertFalse(p._jobs);self.assertFalse(p._busy_users);self.assertFalse(p._pending_scopes)
            with self.assertRaises(ClientError):p._submit_image(ev,'猫')

    async def test_diagnosis_tool_detaches_as_well(self):
        p=plugin();ev=group_event('group-a');release=asyncio.Event()
        async def diagnose(*a):await release.wait();return 'PS文字建议'
        p._diagnose=AsyncMock(side_effect=diagnose)
        response=await asyncio.wait_for(p.tool_diagnose(ev,'自然'),0.1)
        self.assertIn('已受理',response)
        job=ev.get_extra('gpt_image_job_id');task=p._jobs[job]
        release.set();await task
        self.assertIn('PS文字建议',ev.send.call_args.args[0].text)

    def test_at_and_reply_do_not_hide_slash_or_execute_quoted_commands(self):
        ev=Event('@bot /生图 状态')
        quote=Reply(id='1');quote.chain=[Plain('/生图 创建 此处不能执行')]
        ev.get_messages=lambda:[quote,At(qq='bot'),Plain(' /生图 状态')]
        self.assertTrue(SlashCommandFilter().filter(ev,{}))
        ev.get_messages=lambda:[quote,At(qq='bot'),Plain('你觉得如何')]
        self.assertFalse(SlashCommandFilter().filter(ev,{}))


if __name__=='__main__':unittest.main()
