"""Offline command-parser coverage; no AstrBot/network/Pillow dependency."""
from __future__ import annotations

import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from core.prompts import (  # noqa: E402
    CommandRequest,
    parse_generate_args,
    parse_studio_command,
    prompt_guard,
    split_reference_mode,
)


class StudioCommandTests(unittest.TestCase):
    def test_empty_is_help(self):
        for text in ("", "   ", "\n\t"):
            with self.subTest(text=text):
                self.assertEqual(parse_studio_command(text), CommandRequest("help", ""))

    def test_canonical_chinese_actions(self):
        for word, canonical in (("创建", "create"), ("编辑", "edit"), ("风格", "style"), ("融合", "merge")):
            with self.subTest(action=word):
                self.assertEqual(parse_studio_command(word + " 一只猫"), CommandRequest(canonical, "一只猫"))
        for word, canonical in (("重试", "retry"), ("状态", "status"), ("帮助", "help")):
            self.assertEqual(parse_studio_command(word), CommandRequest(canonical, ""))

    def test_english_actions_case_insensitive(self):
        for alias, canonical in (("CREATE", "create"), ("new", "create"), ("generate", "create"),
                                 ("gen", "create"), ("EDIT", "edit"), ("modify", "edit"),
                                 ("style", "style"), ("restyle", "style"), ("merge", "merge"), ("blend", "merge")):
            with self.subTest(alias=alias):
                self.assertEqual(parse_studio_command(alias + " cat").action, canonical)
        for alias, canonical in (("retry", "retry"), ("redo", "retry"), ("STATUS", "status"), ("HELP", "help"), ("?", "help")):
            self.assertEqual(parse_studio_command(alias).action, canonical)

    def test_unknown_action_cannot_be_a_prompt(self):
        for text in ("一只猫", "createx cat", "生成 一只猫", "--质量 高 创建 猫", "/生图 创建 猫"):
            with self.subTest(text=text), self.assertRaises(ValueError):
                parse_studio_command(text)

    def test_all_size_flag_aliases_and_equals(self):
        for flag in ("--比例", "-r", "--size", "-s"):
            for separator in (" ", "="):
                with self.subTest(flag=flag, separator=separator):
                    request = parse_studio_command(f"创建 {flag}{separator}2:3 宇航员")
                    self.assertEqual(request.size, "1024x1536")
                    self.assertEqual(request.prompt, "宇航员")

    def test_all_quality_flag_aliases_and_equals(self):
        for flag in ("--质量", "-q", "--quality"):
            for separator in (" ", "="):
                with self.subTest(flag=flag, separator=separator):
                    request = parse_studio_command(f"编辑 {flag}{separator}高 加帽子")
                    self.assertEqual(request.quality, "high")
                    self.assertEqual(request.prompt, "加帽子")

    def test_supported_sizes_and_ratios(self):
        expected = {"1:1": "1024x1024", "3:2": "1536x1024", "2:3": "1024x1536",
                    "16:9": "1792x1024", "9:16": "1024x1792", "竖版": "1024x1536",
                    "横版": "1536x1024", "square": "1024x1024", "自动": "auto"}
        for value, size in expected.items():
            with self.subTest(value=value):
                self.assertEqual(parse_studio_command(f"创建 --比例={value} 猫").size, size)
        for value in ("1024x1024", "1536x1024", "1024x1536", "1792x1024", "1024x1792", "auto"):
            self.assertEqual(parse_studio_command(f"创建 --size {value} 猫").size, value)

    def test_invalid_sizes_fail(self):
        for value in ("0x0", "1x1", "1024x0", "-1x1024", "999999x1024", "4:3", "foo", "2048x2048"):
            with self.subTest(value=value), self.assertRaises(ValueError):
                parse_studio_command(f"创建 --size={value} 猫")

    def test_qualities(self):
        for value, canonical in (("低", "low"), ("中", "medium"), ("高", "high"), ("自动", "auto"),
                                 ("LOW", "low"), ("medium", "medium"), ("high", "high"), ("auto", "auto")):
            self.assertEqual(parse_studio_command(f"创建 -q {value} 猫").quality, canonical)
        for value in ("最高", "max", "", "4"):
            with self.subTest(value=value), self.assertRaises(ValueError):
                parse_studio_command(f"创建 --质量={value} 猫")

    def test_refs_order_and_chinese_comma(self):
        for flag in ("--参考", "--refs"):
            for separator in ("=", " "):
                for value in ("3,1,6", "3，1，6"):
                    r = parse_studio_command(f"融合 {flag}{separator}{value} 做海报")
                    self.assertEqual(r.references, (3, 1, 6))
                    self.assertEqual(r.prompt, "做海报")

    def test_invalid_refs(self):
        for value in ("0", "7", "-1", "1.5", "1,1", "1，1", "1,,3", "1,", ",1", "", "a", "1/3", "1, 3"):
            with self.subTest(value=value), self.assertRaises(ValueError):
                parse_studio_command(f"编辑 --refs={value} 加帽子")

    def test_create_rejects_reference_indices(self):
        with self.assertRaisesRegex(ValueError, "创建"):
            parse_studio_command("创建 --refs 1 猫")

    def test_non_prompt_actions_reject_any_content_or_options(self):
        for action in ("帮助", "状态", "重试", "help", "status", "retry"):
            for tail in ("猫", "--quality 高", "--比例 1:1", "--参考 1", "\n参数"):
                with self.subTest(action=action, tail=tail), self.assertRaises(ValueError):
                    parse_studio_command(f"{action} {tail}")

    def test_prompt_actions_require_prompt_even_with_options(self):
        for action in ("创建", "编辑", "风格", "融合"):
            for tail in ("", "  ", "--quality high", "--size=auto"):
                with self.subTest(action=action, tail=tail), self.assertRaises(ValueError):
                    parse_studio_command(f"{action} {tail}")

    def test_unknown_flags_and_missing_values(self):
        for tail in ("--foo 猫", "--qualityx 高 猫", "--质量", "--size=", "--refs=", "-r",
                     "--size --quality high 猫", "--质量 -r 1:1 猫", "--参考 --refs 1 猫", "-z 猫"):
            with self.subTest(tail=tail), self.assertRaises(ValueError):
                parse_studio_command("编辑 " + tail)

    def test_duplicate_option_aliases_rejected(self):
        for tail in ("--size 1:1 -r 2:3", "--quality high --质量 低", "--refs 1 --参考 2"):
            with self.subTest(tail=tail), self.assertRaises(ValueError):
                parse_studio_command("编辑 " + tail + " 猫")

    def test_multiline_quotes_and_spacing_preserved(self):
        prompt = '海报写上 "Hello  world"\n第二行写上：\'Good day\'\n  缩进内容，“保留中文引号”'
        r = parse_studio_command("创建\n" + prompt)
        self.assertEqual(r.prompt, prompt)

    def test_inline_flag_removal_does_not_flatten_prompt_lines(self):
        r = parse_studio_command('编辑 第一行 --质量 high 背景\n第二行  两个空格\n第三行 --参考=3,1 帽子')
        self.assertEqual(r.prompt, '第一行 背景\n第二行  两个空格\n第三行 帽子')
        self.assertEqual(r.quality, 'high')
        self.assertEqual(r.references, (3, 1))

    def test_quoted_option_text_is_literal_not_parsed(self):
        for literal in ('"--quality bad"', "'--refs 9'", '“--unknown value”', '‘--比例 4:3’'):
            with self.subTest(literal=literal):
                r = parse_studio_command(f"创建 写上 {literal} --质量 高")
                self.assertEqual(r.prompt, f"写上 {literal}")
                self.assertEqual(r.quality, "high")

    def test_apostrophes_and_unmatched_quotes_not_shell_errors(self):
        for prompt in ("A cat's hat", '写一个 " 未配对引号', 'A \\"quoted\\" word', "don't edit the cat's face"):
            with self.subTest(prompt=prompt):
                self.assertEqual(parse_studio_command(f"创建 {prompt} --质量 high").prompt, prompt)

    def test_non_string_rejected(self):
        for text in (None, 1, [], {}):
            with self.subTest(text=text), self.assertRaises(ValueError):
                parse_studio_command(text)


class LegacyCompatibilityTests(unittest.TestCase):
    def test_old_fields_and_positional_defaults_still_work(self):
        p = parse_generate_args("猫", {"size": "1024x1024", "quality": "medium"})
        self.assertEqual((p.prompt, p.size, p.quality, p.references), ("猫", "1024x1024", "medium", ()))

    def test_legacy_supports_new_aliases_and_references(self):
        p = parse_generate_args("加帽子 --比例=2:3 --质量 高 --参考 2，1", {})
        self.assertEqual((p.prompt, p.size, p.quality, p.references), ("加帽子", "1024x1536", "high", (2, 1)))

    def test_legacy_leading_size_alias(self):
        p = parse_generate_args("竖版 猫\n保持第二行", {"quality": "medium"})
        self.assertEqual(p.size, "1024x1536")
        self.assertEqual(p.prompt, "猫\n保持第二行")
        self.assertEqual(p.quality, "medium")

    def test_explicit_size_beats_legacy_prefix_and_defaults(self):
        p = parse_generate_args("竖版 猫 --size=1:1", {"size": "1536x1024", "quality": "low"})
        self.assertEqual(p.size, "1024x1024")
        self.assertEqual(p.prompt, "猫")

    def test_legacy_quotes_and_newlines_not_destroyed(self):
        prompt = '猫说 "Hello  world"\n  第二行'
        self.assertEqual(parse_generate_args(prompt, {}).prompt, prompt)

    def test_legacy_invalid_options_fail_closed(self):
        for text in ("猫 --unknown x", "猫 --size", "猫 -q bad", "猫 --参考 1,1"):
            with self.subTest(text=text), self.assertRaises(ValueError):
                parse_generate_args(text, {})

    def test_reference_mode_interface_preserves_interior_whitespace(self):
        self.assertEqual(split_reference_mode('引用 图1\n图2  保留空白'), ('quote', '图1\n图2  保留空白'))
        self.assertEqual(split_reference_mode("普通\n文本"), ("", "普通\n文本"))
        self.assertEqual(split_reference_mode(""), ("", ""))

    def test_prompt_guard_interface_unchanged(self):
        self.assertEqual(prompt_guard('  猫\n"hello"  '), '猫\n"hello"')
        for text in ("", "裸露", "猫" * 2001):
            with self.subTest(text=text), self.assertRaises(ValueError):
                prompt_guard(text)


if __name__ == "__main__":
    unittest.main()
