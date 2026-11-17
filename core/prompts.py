"""Framework-independent command and generation-argument parsing.

Options are removed by source span, not by shlex: ordinary quotes, interior
whitespace and prompt line breaks survive unchanged.
"""

from __future__ import annotations

import re
from dataclasses import dataclass

SIZE_ALIASES: dict[str, str] = {
    "1:1": "1024x1024", "方": "1024x1024", "square": "1024x1024",
    "3:2": "1536x1024", "横": "1536x1024", "横版": "1536x1024",
    "landscape": "1536x1024",
    "2:3": "1024x1536", "竖": "1024x1536", "竖版": "1024x1536",
    "portrait": "1024x1536",
    "16:9": "1792x1024", "9:16": "1024x1792",
    "auto": "auto", "自动": "auto",
}
_ALLOWED_SIZES = frozenset(SIZE_ALIASES.values())
QUALITY_ALIASES = {
    "低": "low", "low": "low", "中": "medium", "medium": "medium",
    "高": "high", "high": "high", "自动": "auto", "auto": "auto",
}
MODE_PREFIX = {
    "引用": "quote", "quote": "quote", "引用图": "quote",
    "本图": "message", "图": "message", "message": "message",
    "最近": "history", "history": "history", "历史": "history", "群图": "history",
}
_ACTION_ALIASES = {
    "创建": "create", "create": "create", "new": "create",
    "generate": "create", "gen": "create",
    "编辑": "edit", "edit": "edit", "modify": "edit",
    "风格": "style", "style": "style", "restyle": "style",
    "融合": "merge", "merge": "merge", "blend": "merge",
    "重试": "retry", "retry": "retry", "redo": "retry",
    "诊断": "diagnose", "修图建议": "diagnose", "diagnose": "diagnose", "advice": "diagnose",
    "状态": "status", "status": "status",
    "帮助": "help", "help": "help", "?": "help", "？": "help",
}
_OPTION_NAMES = {
    "--比例": "size", "-r": "size", "--size": "size", "-s": "size",
    "--质量": "quality", "-q": "quality", "--quality": "quality",
    "--参考": "references", "--refs": "references",
}
_TOKEN_RE = re.compile(r"\S+")


@dataclass
class CommandRequest:
    action: str
    prompt: str
    size: str = ""
    quality: str = ""
    references: tuple[int, ...] = ()


@dataclass
class ParsedArgs:
    prompt: str
    size: str = ""
    quality: str = ""
    references: tuple[int, ...] = ()


def _protected_spans(text: str) -> list[tuple[int, int]]:
    """Protect matched literal quotes without requiring balanced shell syntax.

    Apostrophes inside words (don't, cat's) are not opening quotes. Unmatched
    quotes remain literal characters instead of causing a parser error.
    """
    pairs = {'"': '"', "'": "'", "“": "”", "‘": "’"}
    spans = []
    index = 0
    while index < len(text):
        char = text[index]
        if char not in pairs or (index and text[index - 1] == "\\"):
            index += 1
            continue
        if char == "'" and index and text[index - 1].isalnum():
            index += 1
            continue
        end = index + 1
        while end < len(text):
            if text[end] == pairs[char] and text[end - 1] != "\\":
                spans.append((index, end + 1))
                index = end + 1
                break
            end += 1
        else:
            index += 1
    return spans


def _size(value: str) -> str:
    cleaned = value.lower()
    result = SIZE_ALIASES.get(cleaned, cleaned)
    if result not in _ALLOWED_SIZES:
        raise ValueError("比例/尺寸不支持：请用 1:1、3:2、2:3、16:9、9:16、auto 或对应合法尺寸")
    return result


def _quality(value: str) -> str:
    result = QUALITY_ALIASES.get(value.lower())
    if result is None:
        raise ValueError("画质只支持 low/medium/high/auto（低/中/高/自动）")
    return result


def _references(value: str) -> tuple[int, ...]:
    parts = value.replace("，", ",").split(",")
    if any(not re.fullmatch(r"[1-6]", part) for part in parts):
        raise ValueError("参考图编号必须为 1..6 的正整数，用逗号分隔，例如 --参考 1,3")
    result = tuple(int(part) for part in parts)
    if len(set(result)) != len(result):
        raise ValueError("参考图编号不能重复")
    return result


def _parse_options(text: str) -> tuple[ParsedArgs, set[str]]:
    tokens = list(_TOKEN_RE.finditer(text))
    protected = _protected_spans(text)
    removed: list[tuple[int, int]] = []
    values: dict[str, str | tuple[int, ...]] = {}
    index = 0
    while index < len(tokens):
        token = tokens[index]
        raw = token.group()
        if any(start <= token.start() < end for start, end in protected):
            index += 1
            continue
        option, equals, value = raw.partition("=")
        key = _OPTION_NAMES.get(option)
        if key is None:
            if raw.startswith("--") or re.match(r"^-[A-Za-z](?:\b|=)", raw):
                raise ValueError(f"未知选项：{option}")
            index += 1
            continue
        if key in values:
            raise ValueError(f"参数重复：{option}")
        end = token.end()
        if equals:
            if not value:
                raise ValueError(f"参数缺少值：{option}")
        else:
            index += 1
            if index >= len(tokens) or tokens[index].group().startswith("-"):
                raise ValueError(f"参数缺少值：{option}")
            value = tokens[index].group()
            end = tokens[index].end()
        converter = {"size": _size, "quality": _quality, "references": _references}[key]
        values[key] = converter(value)
        # Consume only trailing horizontal separator whitespace. Do not flatten
        # source line breaks or join/reconstruct prompt tokens.
        while end < len(text) and text[end] in " \t":
            end += 1
        removed.append((token.start(), end))
        index += 1

    output = []
    cursor = 0
    for start, end in removed:
        output.append(text[cursor:start])
        cursor = end
    output.append(text[cursor:])
    return ParsedArgs(
        prompt="".join(output).strip(),
        size=values.get("size", ""),
        quality=values.get("quality", ""),
        references=values.get("references", ()),
    ), set(values)


def parse_studio_command(text: str) -> CommandRequest:
    """Parse the tail after /生图. Unknown actions/options always fail closed."""
    if not isinstance(text, str):
        raise ValueError("生图命令必须为文本")
    head = _TOKEN_RE.search(text)
    if head is None:
        return CommandRequest(action="help", prompt="")
    action = _ACTION_ALIASES.get(head.group().lower())
    if action is None:
        raise ValueError(f"未知操作：{head.group()}；可用：创建/编辑/风格/融合/诊断/重试/状态/帮助")
    parsed, options = _parse_options(text[head.end():])
    if action in {"help", "status", "retry"}:
        if parsed.prompt or options:
            raise ValueError("帮助/状态/重试不接受提示词或生成参数")
    elif action == "diagnose":
        if parsed.size or parsed.quality:
            raise ValueError("诊断只输出文字建议，不接受 --比例/--质量；可用 --参考 1,3 选图")
    elif not parsed.prompt:
        raise ValueError("提示词不能为空")
    if action == "create" and parsed.references:
        raise ValueError("创建为纯文生图，不接受 --参考；请使用编辑或融合")
    return CommandRequest(action, parsed.prompt, parsed.size, parsed.quality, parsed.references)


def parse_generate_args(text: str, defaults: dict) -> ParsedArgs:
    """Compatible legacy free-prompt parser with new bilingual option aliases.

    Keeps the prompt/size/quality fields and adds references. The old leading
    size alias syntax (e.g. ``竖版 一只猫``) remains supported.
    """
    parsed, _ = _parse_options(text or "")
    head = _TOKEN_RE.match(parsed.prompt)
    if head and head.group().lower() in SIZE_ALIASES and parsed.prompt[head.end():].strip():
        if not parsed.size:
            parsed.size = SIZE_ALIASES[head.group().lower()]
        parsed.prompt = parsed.prompt[head.end():].lstrip()
    if not parsed.size:
        parsed.size = defaults.get("size", "")
    if not parsed.quality:
        parsed.quality = defaults.get("quality", "")
    return parsed


def split_reference_mode(text: str) -> tuple[str, str]:
    """Peel a legacy source prefix while preserving interior prompt whitespace."""
    stripped = (text or "").strip()
    head = _TOKEN_RE.match(stripped)
    if head is None:
        return "", ""
    mode = MODE_PREFIX.get(head.group().lower().rstrip("：:"), "")
    if mode:
        return mode, stripped[head.end():].lstrip()
    return "", stripped


_BAN_WORDS = ("违规", "裸露", "色情", "暴力血腥", "杀人", "恐怖分子", "毒品制造")


def prompt_guard(prompt: str, limit: int = 2000) -> str:
    """Cheap pre-flight checks before spending a paid call."""
    if not isinstance(prompt, str) or not prompt.strip():
        raise ValueError("提示词不能为空")
    text = prompt.strip()
    if len(text) > limit:
        raise ValueError(f"提示词过长（>{limit} 字），请精简后重试")
    lowered = text.lower()
    for word in _BAN_WORDS:
        if word in lowered:
            raise ValueError("提示词包含不被允许的内容，请修改后再试")
    return text
