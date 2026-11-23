"""Bounded local job receipts. No prompts, source images, or credentials."""
from __future__ import annotations

import json
import os
import re
import tempfile
import time
from pathlib import Path

JOB_ID = re.compile(r"[a-f0-9]{10}", re.IGNORECASE | re.ASCII)
OUTPUT_NAME = re.compile(r"[0-9]{8}-[0-9]{6}-[a-f0-9]{8}\.(?:png|jpg|webp)")
TTL_SECONDS = 86400
MAX_RECEIPTS = 500
INTERRUPTED = "处理被中断；不会自动恢复或重新生成"
UNKNOWN_DELIVERY = "已出图，但发送结果不明"


def recover_output(directory: Path, name: str) -> Path | None:
    if not OUTPUT_NAME.fullmatch(name):
        return None
    candidate = directory / name
    try:
        if candidate.is_symlink() or not candidate.is_file():
            return None
        resolved = candidate.resolve()
        return resolved if resolved.parent == directory.resolve() else None
    except (OSError, RuntimeError):
        return None


class Receipts:
    def __init__(self, path: Path):
        self.path = path
        self.rows: dict[str, dict] = {}
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
        except (FileNotFoundError, ValueError, RecursionError):
            data = {}
        if isinstance(data, dict):
            for job, row in data.items():
                if not JOB_ID.fullmatch(job) or not isinstance(row, dict):
                    continue
                if not isinstance(row.get("updated"), (float, int)) or not isinstance(row.get("destination"), str):
                    continue
                if not isinstance(row.get("status"), str) or not isinstance(row.get("output"), str):
                    continue
                if not row.get("final"):
                    row["status"] = UNKNOWN_DELIVERY if row["output"] else INTERRUPTED
                    row["final"] = True
                self.rows[job] = row
        self._prune()

    def _prune(self) -> None:
        cutoff = time.time() - TTL_SECONDS
        self.rows = {k: v for k, v in self.rows.items() if not v.get("final") or v["updated"] > cutoff}
        if len(self.rows) > MAX_RECEIPTS:
            finished = sorted((k for k, v in self.rows.items() if v.get("final")), key=lambda k: self.rows[k]["updated"])
            for k in finished[:len(self.rows) - MAX_RECEIPTS]:
                self.rows.pop(k)

    def put(self, job: str, destination: str, status: str, *, output: str = "", final: bool = False) -> None:
        self.rows[job] = {"destination": destination, "status": status, "output": output, "final": final, "updated": time.time()}
        self._prune()
        tmp = None
        try:
            with tempfile.NamedTemporaryFile(mode="w", encoding="utf-8", dir=self.path.parent, delete=False) as file:
                tmp = Path(file.name)
                json.dump(self.rows, file, ensure_ascii=False)
                file.flush()
                os.fsync(file.fileno())
            os.replace(tmp, self.path)
            tmp = None
        finally:
            if tmp is not None:
                tmp.unlink(missing_ok=True)

    def get(self, job: str, destination: str) -> dict | None:
        self._prune()
        row = self.rows.get(job)
        return row.copy() if row is not None and row["destination"] == destination else None
