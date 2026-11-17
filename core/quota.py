"""Per-user cooldown and daily-quota accounting, persisted to disk."""

from __future__ import annotations

import json
import time
from dataclasses import dataclass, field
from pathlib import Path


@dataclass
class Quota:
    cooldown_seconds: int = 8
    max_daily: int = 60
    _last: dict[str, float] = field(default_factory=dict)
    _counts: dict[str, dict[str, int]] = field(default_factory=dict)  # user -> {date: n}
    _path: Path | None = None

    def _today(self) -> str:
        return time.strftime("%Y-%m-%d")

    def check(self, user: str) -> str | None:
        """Return an error string if the user may not generate right now."""
        now = time.monotonic()
        last = self._last.get(user, 0.0)
        if self.cooldown_seconds and last and now - last < self.cooldown_seconds:
            wait = int(self.cooldown_seconds - (now - last)) + 1
            return f"慢一点～还需等待 {wait} 秒再试"
        if self.max_daily:
            used = self._counts.get(user, {}).get(self._today(), 0)
            if used >= self.max_daily:
                return f"今天已用完 {self.max_daily} 次生图额度，明天再来吧"
        return None

    def commit(self, user: str) -> None:
        """Record a *successful* generation."""
        self._last[user] = time.monotonic()
        today = self._today()
        bucket = self._counts.setdefault(user, {})
        bucket[today] = bucket.get(today, 0) + 1
        # Drop old days to keep the file small.
        for day in list(bucket):
            if day != today:
                bucket.pop(day, None)
        self._save()

    def used_today(self, user: str) -> int:
        return self._counts.get(user, {}).get(self._today(), 0)

    # -- persistence ------------------------------------------------------
    def load(self, path: Path) -> None:
        self._path = path
        try:
            raw = json.loads(path.read_text("utf-8"))
        except (OSError, ValueError):
            return
        counts = raw.get("counts")
        if isinstance(counts, dict):
            today = self._today()
            self._counts = {
                str(u): {today: int(v.get(today, 0))}
                for u, v in counts.items()
                if isinstance(v, dict)
            }

    def _save(self) -> None:
        if self._path is None:
            return
        try:
            self._path.parent.mkdir(parents=True, exist_ok=True)
            payload = {"counts": self._counts, "saved_at": time.time()}
            self._path.write_text(json.dumps(payload, ensure_ascii=False), "utf-8")
        except OSError:
            pass
