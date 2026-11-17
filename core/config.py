"""Read a few fields out of the plugin config with sane defaults and no secrets."""

from __future__ import annotations

from typing import Any


def sect(config: Any, name: str) -> dict[str, Any]:
    value = config.get(name) if hasattr(config, "get") else None
    return value if isinstance(value, dict) else {}


def api_section(config: Any) -> dict[str, Any]:
    api = sect(config, "api")
    out = {
        "base_url": str(api.get("base_url") or "").strip(),
        "api_key": str(api.get("api_key") or "").strip(),
        "image_model": str(api.get("image_model") or "gpt-image-2").strip(),
        "fallback_models": api.get("fallback_models") or [],
        "timeout_seconds": api.get("timeout_seconds", 300),
        "extra_body": api.get("extra_body") or {},
    }
    return out


def defaults(config: Any) -> dict[str, str]:
    gen = sect(config, "generation")
    return {
        "size": str(gen.get("size") or ""),
        "quality": str(gen.get("quality") or ""),
        "reference_mode": str(gen.get("reference_mode") or "auto"),
    }


def limits(config: Any) -> dict[str, int]:
    gen = sect(config, "limits")
    def _int(key: str, default: int, lo: int, hi: int) -> int:
        try:
            return max(lo, min(int(gen.get(key, default)), hi))
        except (TypeError, ValueError):
            return default
    return {
        "cooldown_seconds": _int("cooldown_seconds", 8, 0, 600),
        "max_daily": _int("max_daily_per_user", 60, 0, 10000),
        "concurrency": _int("max_concurrency", 2, 1, 8),
    }


def admin_only_generate(config: Any) -> bool:
    gen = sect(config, "access")
    return bool(gen.get("generate_admin_only", True))


def admin_allowlist(config: Any) -> set[str]:
    access = sect(config, "access")
    raw = access.get("admin_user_ids") or []
    if isinstance(raw, str):
        raw = [x for x in raw.replace("，", ",").replace(" ", ",").split(",")]
    return {str(x).strip() for x in raw if str(x).strip()}


def reply_with_source(config: Any) -> bool:
    access = sect(config, "access")
    return bool(access.get("show_prompt_in_reply", False))
