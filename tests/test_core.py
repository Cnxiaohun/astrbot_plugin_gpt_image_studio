"""Offline unit tests (no network, no AstrBot import). Run:

    python -m unittest discover -s tests -v
"""
from __future__ import annotations

import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from core.prompts import (  # noqa: E402
    SIZE_ALIASES,
    parse_generate_args,
    prompt_guard,
    split_reference_mode,
)
from core.quota import Quota  # noqa: E402


class PromptTests(unittest.TestCase):
    def test_size_alias_vertical(self):
        args = parse_generate_args("一只猫 --size 竖版", {"size": "", "quality": "medium"})
        self.assertEqual(args.prompt, "一只猫")
        self.assertEqual(args.size, "1024x1536")
        self.assertEqual(args.quality, "medium")

    def test_quality_alias(self):
        args = parse_generate_args("海报 --quality 高 --size 1:1", {})
        self.assertEqual(args.size, "1024x1024")
        self.assertEqual(args.quality, "high")

    def test_defaults_applied(self):
        args = parse_generate_args("随便", {"size": "1024x1024", "quality": "low"})
        self.assertEqual(args.size, "1024x1024")
        self.assertEqual(args.quality, "low")

    def test_mode_prefix_quote(self):
        mode, rest = split_reference_mode("引用 把背景换成海边")
        self.assertEqual(mode, "quote")
        self.assertEqual(rest, "把背景换成海边")

    def test_mode_prefix_none(self):
        mode, rest = split_reference_mode("一只橘猫")
        self.assertEqual(mode, "")
        self.assertEqual(rest, "一只橘猫")

    def test_guard_rejects_empty_and_blocked(self):
        with self.assertRaises(ValueError):
            prompt_guard("   ")
        with self.assertRaises(ValueError):
            prompt_guard("请生成裸露图片")

    def test_guard_length(self):
        with self.assertRaises(ValueError):
            prompt_guard("啊" * 3000)


class QuotaTests(unittest.TestCase):
    def test_cooldown(self):
        q = Quota(cooldown_seconds=60, max_daily=0)
        self.assertIsNone(q.check("u1"))
        q.commit("u1")
        self.assertIsNotNone(q.check("u1"))

    def test_daily_cap(self):
        q = Quota(cooldown_seconds=0, max_daily=2)
        q.commit("u2")
        q.commit("u2")
        self.assertIsNotNone(q.check("u2"))
        self.assertEqual(q.used_today("u2"), 2)


if __name__ == "__main__":
    unittest.main()
