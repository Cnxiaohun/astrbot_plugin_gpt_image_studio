"""GPT 生图大师 —— AstrBot 插件。

管理员在群里：
  * 引用一张图 + `/生图 编辑 <提示词>` → 以引用图为参考改图
  * `/生图 创建 <提示词>` → 纯文生图，忽略附件
  * 一条消息里有多张图    → 全部识别，最多 6 张一起作为参考
  * `/生图 诊断 [目标]`   → 多模态看图，只给文字版 PS/Camera Raw 修图建议
  * `/改图`  `/画风`  等短指令     → 快速改图 / 换画风的语义糖

长任务会转成独立后台任务并绑定原会话，工具/消息层超时不会截断出图或发错群。

所有付费生图指令默认仅管理员可用，可在配置里放开白名单。
"""

from __future__ import annotations

import asyncio
import re
import sys
import time
import uuid
from collections import OrderedDict
from dataclasses import dataclass
from pathlib import Path

from astrbot.api import AstrBotConfig, logger
from astrbot.api.event import AstrMessageEvent, filter
from astrbot.api.star import Context, Star, StarTools

from . import __version__
from .core import config as cfgmod
from .core.image_client import ImageClient
from .core.diagnosis import diagnose
from .core.http_utils import ClientError
from .core.images import ImageError, prepare_reference
from .core.prompts import ParsedArgs, parse_generate_args, prompt_guard, split_reference_mode, parse_studio_command
from .core.quota import Quota
from .core.references import MAX_REFERENCES, gather_references
from .core.store import Store, build_reply  # build_reply used by _reply_image

PLUGIN_NAME = "astrbot_plugin_gpt_image_studio"

HELP = """GPT 生图大师 · /生图 <操作> [参数] <提示词>

/生图 创建 一只戴墨镜的橘猫
/生图 编辑 把背景换成海边，保持人物不变
/生图 风格 水彩
/生图 融合 将第1张人物与第2张场景融合成海报
/生图 诊断 [修图目标]  看图给PS/Camera Raw参数建议，只输出文字
/生图 重试     复用上次成功任务，会再次计费
/生图 状态     查看额度与进行中任务
/生图 帮助     本说明（直接 /生图 也可）

参数（提示词前后均可）：
  --比例 1:1|2:3|3:2|16:9|9:16|横版|竖版
  --质量 低|中|高|自动
  --参考 1,3   按引用消息/本条消息中图片顺序选图
例：/生图 编辑 --参考 1,3 --比例 2:3 做成电影海报

创建只文生图，忽略附件；编辑/风格必须有图，融合至少2张。
优先取被引用消息里的图，否则取本条消息里的图；最多6张。
重试快照仅保留本用户本会话最近30分钟，插件重载后清空。
任务在后台执行，完成后回原会话；可用任务编号在日志中追溯。
旧 /改图、/画风、/重画、/生图状态、/生图帮助 仍可用。
生成操作默认仅 AstrBot 管理员/配置白名单可用。"""


_SLASH_COMMAND = re.compile(r"^\s*/(?:生图状态|生图帮助|生图|改图|画风|重画|画图|绘图|gpt画图|修图)(?:\s|$)")


def slash_command_text(event) -> str:
    from astrbot.api.message_components import Plain
    pieces = [c.text for c in event.get_messages() if isinstance(c, Plain)]
    return "".join(pieces).strip() if pieces else event.get_message_str().strip()


class SlashCommandFilter(filter.CustomFilter):
    def filter(self, event, cfg):
        # Ignore Reply/At/Image placeholders, but never search inside quoted text.
        return bool(_SLASH_COMMAND.match(slash_command_text(event)))


@dataclass(frozen=True)
class JobSnapshot:
    prompt: str
    size: str
    quality: str
    refs: tuple[bytes, ...]
    created: float


class GptImageStudio(Star):
    def __init__(self, context: Context, config: AstrBotConfig):
        super().__init__(context)
        self.config = config
        self.client: ImageClient | None = None
        self.store: Store | None = None
        self.quota = Quota()
        self.diagnosis_quota = Quota(cooldown_seconds=8, max_daily=30)
        self._sem: asyncio.Semaphore | None = None
        self._last_jobs: OrderedDict[str, JobSnapshot] = OrderedDict()
        self._lock = asyncio.Lock()
        self._busy_users: set[str] = set()
        self._active_tasks: set[asyncio.Task] = set()
        self._jobs: dict[str, asyncio.Task] = {}
        self._job_info: OrderedDict[str, dict] = OrderedDict()
        self._pending_scopes: set[str] = set()
        self._closed = False

    # ------------------------------------------------------------------ lifecycle
    async def initialize(self):
        api = cfgmod.api_section(self.config)
        self.client = ImageClient(api)
        data_dir = Path(StarTools.get_data_dir(PLUGIN_NAME))
        self.store = Store(data_dir)
        lim = cfgmod.limits(self.config)
        self.quota = Quota(cooldown_seconds=lim["cooldown_seconds"], max_daily=lim["max_daily"])
        self.quota.load(data_dir / "quota.json")
        diagnosis_settings = cfgmod.sect(self.config, "diagnosis")
        try:
            daily = max(0, min(10000, int(diagnosis_settings.get("max_daily_per_user", 30))))
        except (ValueError, TypeError):
            daily = 30
        self.diagnosis_quota = Quota(cooldown_seconds=lim["cooldown_seconds"], max_daily=daily)
        self.diagnosis_quota.load(data_dir / "diagnosis_quota.json")
        self._sem = asyncio.Semaphore(lim["concurrency"])
        logger.info(
            "GPT 生图大师 v%s 已加载：模型 %s，并发 %d，管理员限定=%s",
            __version__, api.get("image_model"), lim["concurrency"],
            cfgmod.admin_only_generate(self.config),
        )
        if not api.get("base_url") or not api.get("api_key"):
            logger.warning("GPT 生图大师：尚未配置 api.base_url 或 api.api_key，生图会失败")

    async def terminate(self):
        self._closed = True
        tasks = [t for t in (self._active_tasks | set(self._jobs.values())) if t is not asyncio.current_task()]
        for task in tasks:
            task.cancel()
        await asyncio.gather(*tasks, return_exceptions=True)
        if self.client is not None:
            await self.client.close()
            self.client = None

    # ------------------------------------------------------------------ helpers
    def _authorized(self, event: AstrMessageEvent) -> bool:
        allow = cfgmod.admin_allowlist(self.config)
        uid = str(event.get_sender_id() or "")
        if uid and uid in allow:
            return True
        if not cfgmod.admin_only_generate(self.config):
            return True
        try:
            return bool(event.is_admin())
        except Exception:  # noqa: BLE001
            return False

    def _defaults(self) -> dict:
        return cfgmod.defaults(self.config)

    def _uid(self, event: AstrMessageEvent) -> str:
        return str(event.get_sender_id() or event.get_session_id() or "anon")

    def _job_key(self, event) -> str:
        platform = getattr(event, "get_platform_id", lambda: "")()
        return f"{platform}:{event.get_session_id()}:{self._uid(event)}"

    @staticmethod
    def _is_webchat(event) -> bool:
        try:
            if event.get_platform_id() == "webchat":
                return True
        except Exception:  # noqa: BLE001
            pass
        return str(getattr(event, "unified_msg_origin", "")).startswith("webchat:")

    async def _deliver_text(self, event, text: str) -> None:
        """Background text delivery: webchat needs context.send_message to persist
        proactive messages after the originating SSE run has ended; QQ event.send
        is bound to the original group and remains the proven path."""
        umo = getattr(event, "unified_msg_origin", "") or ""
        if self._is_webchat(event) and self.context is not None and umo:
            from astrbot.api.event import MessageChain
            await self.context.send_message(umo, MessageChain().message(text))
            return
        await event.send(event.plain_result(text))

    async def _deliver_chain(self, event, chain) -> None:
        umo = getattr(event, "unified_msg_origin", "") or ""
        if self._is_webchat(event) and self.context is not None and umo:
            await self.context.send_message(umo, chain)
            return
        await event.send(chain)

    def _remember_job(self, event, snapshot: JobSnapshot) -> None:
        key = self._job_key(event)
        self._last_jobs.pop(key, None)
        self._last_jobs[key] = snapshot
        # Bounded, short-lived memory only; no cross-user or cross-group replay.
        while len(self._last_jobs) > 32 or sum(
            sum(len(b) for b in job.refs) for job in self._last_jobs.values()
        ) > 32 * 1024 * 1024:
            self._last_jobs.popitem(last=False)

    def _previous_job(self, event):
        key = self._job_key(event)
        job = self._last_jobs.get(key)
        if job and time.monotonic() - job.created > 1800:
            self._last_jobs.pop(key, None)
            return None
        return job

    async def _generate(
        self, event: AstrMessageEvent, prompt: str, *,
        force_reference: bool = False, reference_mode: str = "",
        request=None, replay: JobSnapshot | None = None, min_references: int = 0,
    ) -> tuple[Path | None, str]:
        """Shared guarded path for commands and LLM tools; one task per user."""
        gate = self._gate(event)
        if gate:
            return None, gate
        user = self._job_key(event)
        if user in self._busy_users:
            return None, "你在本会话已有一个任务进行中，请等它完成，不要重复提交。"
        self._busy_users.add(user)
        task = asyncio.current_task()
        self._active_tasks.add(task)
        try:
            extra = {}
            if request is not None:
                extra["request"] = request
            if replay is not None:
                extra["replay"] = replay
            if min_references:
                extra["min_references"] = min_references
            return await self._run_generation(
                event, prompt, force_reference=force_reference, reference_mode=reference_mode, **extra,
            )
        finally:
            self._busy_users.discard(user)
            self._active_tasks.discard(task)

    async def _run_generation(
        self, event: AstrMessageEvent, prompt: str, *,
        force_reference: bool = False, reference_mode: str = "",
        request=None, replay: JobSnapshot | None = None, min_references: int = 0,
    ) -> tuple[Path | None, str]:
        """Run one generation/edit and return (saved path, note-for-user)."""
        if self.client is None or self.store is None or self._sem is None:
            return None, "插件尚未初始化完成，请稍后重试"
        try:
            prompt = prompt_guard(prompt)
        except ValueError as exc:
            return None, str(exc)

        defaults = self._defaults()
        try:
            if replay is not None:
                parsed = ParsedArgs(replay.prompt, replay.size, replay.quality)
            else:
                parsed = request if request is not None else parse_generate_args(prompt, defaults)
        except ValueError as exc:
            return None, str(exc)
        user = self._uid(event)

        mode = reference_mode or defaults.get("reference_mode", "auto")
        if replay is not None:
            parsed.prompt, parsed.size, parsed.quality = replay.prompt, replay.size, replay.quality
        if not reference_mode and request is None and replay is None:
            text_mode, remainder = split_reference_mode(parsed.prompt)
            if text_mode:
                mode = text_mode
                parsed.prompt = remainder
        size = parsed.size or defaults.get("size", "")
        quality = parsed.quality or defaults.get("quality", "")

        if not parsed.prompt:
            return None, "提示词不能为空，例如：/生图 一只橘猫"

        # Send immediately, before slow image downloads / queueing / paid API I/O.
        # Shared by slash commands and natural-language tools, exactly once/task.
        target = "本群" if event.get_group_id() else "当前会话"
        job_id = getattr(event, "get_extra", lambda *a: "")("gpt_image_job_id") or ""
        state = "正在排队等待生图" if self._sem.locked() else "正在处理生图任务"
        try:
            await self._deliver_text(
                event,
                f"🎨 已收到，{state}（任务 {job_id or '当前'}；有参考图会先读取）。完成后会把图片发到{target}，请稍候，不用重复提交。",
            )
        except Exception as exc:
            logger.warning("GPT 生图大师开始通知发送失败：%s", type(exc).__name__)
            return None, "无法发送生图开始通知，任务未提交，请检查机器人发送权限。"

        refs: list[bytes] = []
        try:
            raw_refs = list(replay.refs) if replay is not None else await gather_references(event, mode, self.client.client)
        except ClientError as exc:
            return None, str(exc)
        selection = getattr(parsed, "references", ()) if replay is None else ()
        if selection:
            if any(i > len(raw_refs) for i in selection):
                return None, f"参考图编号超出范围：当前共 {len(raw_refs)} 张图；未提交生图。"
            raw_refs = [raw_refs[i - 1] for i in selection]
        if len(raw_refs) > MAX_REFERENCES:
            return None, f"一次最多参考 {MAX_REFERENCES} 张图，请减少图片数量。"
        for index, blob in enumerate(raw_refs, 1):
            try:
                refs.append(blob if replay is not None else await prepare_reference(blob))
            except ImageError as exc:
                return None, f"第 {index} 张参考图无效：{exc}；未提交生图请求。"
        if min_references and len(refs) < min_references:
            return None, f"融合至少需要 {min_references} 张参考图，当前只有 {len(refs)} 张；未提交生图。"
        if force_reference and not refs:
            return None, "没有找到可用的参考图：请引用一张图，或直接带上图片再试"

        if not parsed.prompt:
            return None, "提示词不能为空，例如：/生图 一只橘猫"

        logger.info("GPT 生图大师任务开始：mode=%s references=%d", mode, len(refs))
        async with self._sem:
            gate = self.quota.check(user)
            if gate:
                return None, gate
            try:
                if refs:
                    result = await self.client.edit(
                        parsed.prompt, refs,
                        size=size, quality=quality,
                    )
                else:
                    result = await self.client.generate(
                        parsed.prompt, size=size, quality=quality,
                    )
            except (ImageError, ClientError) as exc:
                return None, f"出图失败：{exc}"
            except Exception as exc:  # noqa: BLE001
                logger.error("GPT 生图大师 生图异常：%s", type(exc).__name__)
                return None, "出图失败：内部错误，请查看日志或稍后再试"

        self.quota.commit(user)
        try:
            path = self.store.save(result.data, result.mime)
        except ClientError as exc:
            return None, f"生图已完成但保存失败：{exc}；请勿自动重试（可能再次计费）。"
        logger.info("GPT 生图大师任务完成：references=%d pixels=%dx%d", len(refs), result.width, result.height)
        async with self._lock:
            self._remember_job(event, JobSnapshot(parsed.prompt, size, quality, tuple(refs), time.monotonic()))

        note = f"✅ 出图完成 · {result.width}×{result.height}"
        if refs:
            note += f" · 参考图 {len(refs)} 张"
        if cfgmod.reply_with_source(self.config):
            note += f"\n提示词：{parsed.prompt}"
        return path, note

    def _reply_image(self, event: AstrMessageEvent, path: Path, note: str):
        event.set_extra("gpt_image_reply", str(path))
        yield event.chain_result(list(build_reply(path, note).chain))

    def _spawn_job(self, event, operation, *, kind="生图") -> str:
        """Detach long work from the caller's 30s tool/message lifetime.

        Each closure owns its ORIGINAL event and immutable destination key;
        another group's request can neither replace its target nor cancel it.
        """
        destination = self._job_key(event)
        if self._closed:
            raise ClientError("插件正在停止，请稍后再试。")
        if destination in self._pending_scopes or destination in self._busy_users:
            raise ClientError("你在本会话已有一个任务进行中，请等它完成，不要重复提交。")
        if len(self._jobs) >= 16:
            raise ClientError("任务队列已满，请稍后再试。")
        self._pending_scopes.add(destination)
        job_id = uuid.uuid4().hex[:10]
        info = {"destination": destination, "kind": kind, "status": "已受理", "path": ""}
        self._job_info[job_id] = info
        while len(self._job_info) > 100:
            removable = next((i for i in self._job_info if i not in self._jobs and i != job_id), None)
            if removable is None:
                break
            self._job_info.pop(removable)
        # No mutable global 'current event' or 'current group' is used by worker.
        async def worker():
            try:
                info["status"] = "处理中"
                if kind == "诊断":
                    path, note = None, await operation()
                else:
                    path, note = await operation()
                if path is not None:
                    info["path"] = str(path)
                info["status"] = "发送中"
                logger.info("GPT 生图大师任务回发：job=%s target=%s has_image=%s", job_id, destination, path is not None)
                if path is None:
                    body = note if kind == "诊断" else f"任务 {job_id} · {kind}\n{note}"
                    for part in self._text_parts(body):
                        await self._deliver_text(event, part)
                else:
                    await self._deliver_chain(event, build_reply(path, f"任务 {job_id}\n{note}"))
                info["status"] = "平台发送已返回"
                logger.info("GPT 生图大师任务发送返回：job=%s target=%s", job_id, destination)
            except asyncio.CancelledError:
                info["status"] = "任务被取消，结果可能已计费"
                logger.warning("GPT 生图大师任务取消：job=%s target=%s；不自动重试", job_id, destination)
                raise
            except Exception as exc:
                info["status"] = "发送结果不明" if info["status"] == "发送中" else "处理失败"
                logger.error("GPT 生图大师后台任务异常：job=%s target=%s status=%s error=%s saved=%s", job_id, destination, info["status"], type(exc).__name__, info["path"])
                # Do not automatically resend an image after an ambiguous dispatch.
                if info["status"] != "发送结果不明":
                    try:
                        await self._deliver_text(event, f"任务 {job_id} 处理失败；请查日志确认，不会自动重新生成。")
                    except Exception:
                        pass
            finally:
                self._jobs.pop(job_id, None)
                self._pending_scopes.discard(destination)
        task = asyncio.create_task(worker(), name=f"gpt-image-{job_id}")
        self._jobs[job_id] = task
        def finished(done):
            self._jobs.pop(job_id, None)
            self._pending_scopes.discard(destination)
            if not done.cancelled() and done.exception() is not None:
                logger.error("GPT 生图大师后台任务异常：job=%s type=%s", job_id, type(done.exception()).__name__)
        task.add_done_callback(finished)
        event.set_extra("gpt_image_job_id", job_id)
        logger.info("GPT 生图大师任务受理：job=%s target=%s kind=%s", job_id, destination, kind)
        return job_id

    def _submit_image(self, event, prompt, **kwargs):
        return self._spawn_job(event, lambda: self._generate(event, prompt, **kwargs))

    # ------------------------------------------------------------------ commands
    @filter.command("生图", alias={"画图", "绘图", "gpt画图"}, priority=sys.maxsize)
    async def cmd_generate(self, event: AstrMessageEvent):
        event.stop_event()
        raw = self._tail(event, ("生图", "画图", "绘图", "gpt画图"))
        try:
            request = parse_studio_command(raw)
        except ValueError as exc:
            yield event.plain_result(f"参数错误：{exc}\n用法：/生图 <创建|编辑|风格|融合|诊断|重试|状态|帮助>；/生图 帮助 查看示例。")
            return
        if request.action == "help":
            yield event.plain_result(HELP)
            return
        if request.action == "status":
            async for result in self.cmd_status(event):
                yield result
            return
        if request.action == "retry":
            async for result in self.cmd_again(event):
                yield result
            return
        if request.action == "diagnose":
            if not self._authorized(event):
                yield event.plain_result("抱歉，修图诊断仅管理员可用。")
                return
            try:
                self._spawn_job(event, lambda: self._diagnose(event, request.prompt, request.references), kind="诊断")
            except ClientError as exc:
                yield event.plain_result(str(exc))
            return
        gate = self._gate(event)
        if gate:
            yield event.plain_result(gate)
            return
        if request.action == "style":
            request.prompt = f"把参考图整体转换成{request.prompt}画风，保持主体、构图与内容不变，只改变绘画风格"
        try:
            self._submit_image(
                event, request.prompt, request=request,
                reference_mode="none" if request.action == "create" else "auto",
                force_reference=request.action != "create",
                min_references=2 if request.action == "merge" else 0,
            )
        except ClientError as exc:
            yield event.plain_result(str(exc))

    @filter.command("改图", alias={"修图"}, priority=sys.maxsize)
    async def cmd_edit(self, event: AstrMessageEvent):
        event.stop_event()
        raw = self._tail(event, ("改图", "修图"))
        gate = self._gate(event)
        if gate:
            yield event.plain_result(gate)
            return
        try:
            self._submit_image(event, raw, force_reference=True)
        except ClientError as exc:
            yield event.plain_result(str(exc))

    @filter.command("画风", priority=sys.maxsize)
    async def cmd_style(self, event: AstrMessageEvent):
        event.stop_event()
        raw = self._tail(event, ("画风",)).strip()
        if not raw:
            yield event.plain_result("用法：引用一张图后发 /画风 水彩|油画|赛博朋克|动漫…")
            return
        gate = self._gate(event)
        if gate:
            yield event.plain_result(gate)
            return
        prompt = f"把参考图整体转换成{raw}画风，保持主体、构图与内容不变，只改变绘画风格"
        try:
            self._submit_image(event, prompt, force_reference=True)
        except ClientError as exc:
            yield event.plain_result(str(exc))

    @filter.command("重画", priority=sys.maxsize)
    async def cmd_again(self, event: AstrMessageEvent):
        event.stop_event()
        gate = self._gate(event)
        if gate:
            yield event.plain_result(gate)
            return
        last = self._previous_job(event)
        if last is None:
            yield event.plain_result("本会话没有可重试的成功任务，或快照已过期/重载后清空。请先 /生图 创建 <提示词>。")
            return
        yield event.plain_result("将复用上次成功任务的提示词、参数和参考图重新生成一张；本次会再次计费。")
        try:
            self._submit_image(event, last.prompt, replay=last)
        except ClientError as exc:
            yield event.plain_result(str(exc))

    @filter.command("生图状态", priority=sys.maxsize)
    async def cmd_status(self, event: AstrMessageEvent):
        event.stop_event()
        user = self._uid(event)
        lim = cfgmod.limits(self.config)
        api = cfgmod.api_section(self.config)
        used = self.quota.used_today(user)
        quota_txt = f"{used}/{lim['max_daily']}" if lim["max_daily"] else f"{used}/不限"
        yield event.plain_result(
            f"模型：{api.get('image_model') or '未配置'}\n"
            f"今日已用：{quota_txt}\n"
            f"今日诊断调用：{self.diagnosis_quota.used_today(user)}/{self.diagnosis_quota.max_daily or '不限'}（独立额度）\n"
            f"任务：{'进行中（读图/排队/生图/诊断）' if self._job_key(event) in self._busy_users else '空闲'}\n"
            f"冷却：{lim['cooldown_seconds']} 秒/次 · 并发上限 {lim['concurrency']}"
        )

    @filter.command("生图帮助", priority=sys.maxsize)
    async def cmd_help(self, event: AstrMessageEvent):
        event.stop_event()
        yield event.plain_result(HELP)

    @filter.custom_filter(SlashCommandFilter, priority=sys.maxsize)
    async def slash_fallback(self, event: AstrMessageEvent):
        """Like the newspaper plugin: slash commands work with wake_prefix=[]."""
        if event.get_extra("gpt_image_command_handled"):
            return
        event.set_extra("gpt_image_command_handled", True)
        event.stop_event()
        event.message_str = slash_command_text(event)
        command = event.get_message_str().strip().split(maxsplit=1)[0].lstrip("/")
        handler = {
            "生图": self.cmd_generate, "画图": self.cmd_generate,
            "绘图": self.cmd_generate, "gpt画图": self.cmd_generate,
            "改图": self.cmd_edit, "修图": self.cmd_edit,
            "画风": self.cmd_style, "重画": self.cmd_again,
            "生图状态": self.cmd_status, "生图帮助": self.cmd_help,
        }.get(command)
        if handler is not None:
            logger.info("GPT 生图大师 slash 入口命中：command=%s", command)
            async for result in handler(event):
                yield result

    @filter.llm_tool(name="gpt_image_generate")
    async def tool_generate(self, event: AstrMessageEvent, prompt: str) -> str:
        """GPT 生图大师：图片生成、绘图、AI绘画、文生图。仅在用户明确要求画图/生成图片时调用，管理员权限由工具校验。只生成一张并直接发送到当前会话，不要重复调用。不用于查找图片或解释图片。

        Args:
            prompt(string): 用户明确要求的绘图提示词，保留主体、风格、画幅和文字要求。
        """
        return await self._tool_image(event, prompt, reference_mode="none")

    @filter.llm_tool(name="gpt_image_edit")
    async def tool_edit(self, event: AstrMessageEvent, prompt: str) -> str:
        """GPT 生图大师：改图、图片编辑、参考图生图、图像融合、换背景/画风。仅在用户明确要求修改图片时调用，自动读取当前引用消息或本条消息的所有图片（最多6张）。管理员权限由工具校验；直接把生成图片发到当前会话，失败不自动重试。不接受模型臆造的图片URL。

        Args:
            prompt(string): 修改图片的要求，可按图片顺序描述第1张/第2张参考图的用途。
        """
        return await self._tool_image(event, prompt, force_reference=True)

    async def _tool_image(self, event, prompt, *, force_reference=False, reference_mode=""):
        # Never trust a model to decide access; all tools share command gates.
        if event.get_extra("gpt_image_tool_attempted"):
            return "本条消息已尝试生图，请勿重复调用以免重复计费。"
        gate = self._gate(event)
        if gate:
            return gate
        event.set_extra("gpt_image_tool_attempted", True)
        try:
            job_id = self._submit_image(
                event, prompt, force_reference=force_reference, reference_mode=reference_mode,
            )
        except ClientError as exc:
            return str(exc)
        return f"任务 {job_id} 已受理，后台独立处理并回原会话。请勿等待工具内出图、重复调用或声称已出图。"

    @filter.llm_tool(name="gpt_image_diagnose")
    async def tool_diagnose(self, event: AstrMessageEvent, goal: str = "") -> str:
        """GPT 生图大师：看参考图提供文字版修图建议/PS参数/Camera Raw调色建议。仅当用户请求修图诊断、调整参数或文字建议时调用；自动读引用图或本条图，不生图、不修改图片。会把文字建议直接发到当前会话。

        Args:
            goal(string): 可选修图目标，如通透自然的人像、保留肤质、偏冷电影感；留空做自然保真诊断。
        """
        if event.get_extra("gpt_image_diagnosis_attempted"):
            return "本条消息已尝试修图诊断，请勿重复调用。"
        event.set_extra("gpt_image_diagnosis_attempted", True)
        if not self._authorized(event):
            return "抱歉，修图诊断仅管理员可用。"
        try:
            job_id = self._spawn_job(event, lambda: self._diagnose(event, goal), kind="诊断")
        except ClientError as exc:
            return str(exc)
        return f"诊断任务 {job_id} 已受理，将在原会话发送文字建议；不生图。不要重复调用或臆造诊断结果。"

    @staticmethod
    def _text_parts(text: str):
        """Keep plain-text advice within QQ single-message limits."""
        while len(text) > 1800:
            split = text.rfind("\n", 0, 1800)
            if split < 900:
                split = 1800
            yield text[:split]
            text = text[split:].lstrip("\n")
        if text:
            yield text

    async def _diagnose(self, event, goal: str = "", references=()) -> str:
        if not self._authorized(event):
            return "抱歉，修图诊断仅管理员可用。"
        if len(goal) > 2000:
            return "诊断目标过长，请控制在2000字以内。"
        if self._sem is None:
            return "插件尚未初始化完成，请稍后重试。"
        user = self._uid(event)
        gate = self.diagnosis_quota.check(user)
        if gate:
            return "修图诊断：" + gate
        busy_key = self._job_key(event)
        if busy_key in self._busy_users:
            return "你在本会话已有一个生图/诊断任务进行中，请等它完成。"
        self._busy_users.add(busy_key)
        task = asyncio.current_task()
        self._active_tasks.add(task)
        try:
            try:
                await self._deliver_text(
                    event,
                    "🔎 已收到，正在读取参考图并进行多模态修图诊断；只提供文字建议，不会生成或修改图片。",
                )
                raw = await gather_references(event, "auto")
                if not raw:
                    return "没有找到参考图：请引用图片，或带上图片发送 /生图 诊断 [修图目标]。"
                selection = tuple(references) or tuple(range(1, len(raw) + 1))
                if any(i < 1 or i > len(raw) for i in selection):
                    return f"参考图编号超出范围：当前共 {len(raw)} 张图；未调用诊断模型。"
                images = [await prepare_reference(raw[i - 1]) for i in selection]
            except ClientError as exc:
                return "修图诊断失败：" + str(exc)
            except Exception as exc:
                logger.warning("GPT 生图大师诊断读图失败：%s", type(exc).__name__)
                return "无法发送开始通知或读取参考图，未调用诊断模型。"
            async with self._sem:
                gate = self.diagnosis_quota.check(user)
                if gate:
                    return "修图诊断：" + gate
                # Count model attempts separately from paid image generations.
                self.diagnosis_quota.commit(user)
                try:
                    logger.info("GPT 生图大师诊断开始：references=%d", len(images))
                    result = await diagnose(self.context, event, images, goal, cfgmod.sect(self.config, "diagnosis"))
                except ClientError as exc:
                    return "修图诊断失败：" + str(exc)
                except Exception as exc:
                    logger.warning("GPT 生图大师诊断异常：%s", type(exc).__name__)
                    return "修图诊断失败，请检查所选模型是否支持图片输入；未调用生图接口。"
            logger.info("GPT 生图大师诊断完成：references=%d", len(images))
            if len(selection) > 1 and selection != tuple(range(1, len(selection) + 1)):
                mapping = "，".join(f"图{i}＝原图{source}" for i, source in enumerate(selection, 1))
                return f"{mapping}\n{result}"
            return result
        finally:
            self._busy_users.discard(busy_key)
            self._active_tasks.discard(task)

    # ------------------------------------------------------------------ utilities
    def _gate(self, event: AstrMessageEvent) -> str | None:
        if not self._authorized(event):
            return "抱歉，生图指令仅管理员可用。"
        return self.quota.check(self._uid(event))

    @staticmethod
    def _tail(event: AstrMessageEvent, words: tuple[str, ...]) -> str:
        """Everything after the command word, robust to slash and glued forms."""
        text = (event.message_str or "").strip()
        for word in words:
            if not text:
                break
            head = text.lstrip("/")
            if head == word:
                return ""
            if head.startswith(word):
                return head[len(word):].strip()
            parts = text.split(maxsplit=1)
            if parts and parts[0].lstrip("/") == word:
                return parts[1] if len(parts) > 1 else ""
        # Fallback: strip the first token.
        parts = text.split(maxsplit=1)
        return parts[1].strip() if len(parts) > 1 else ""
