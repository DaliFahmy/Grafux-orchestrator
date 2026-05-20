"""Prompt loader for Grafux-orchestrator.

Parses the [SECTION_NAME] format in configs/Msg_config and exposes
a simple get_system_prompt(section) API.  Results are cached after
the first read so the file is only parsed once per process.
"""
from __future__ import annotations

import re
from functools import lru_cache
from pathlib import Path

_PROMPT_DIR = Path(__file__).parent / "configs"


@lru_cache(maxsize=1)
def _load_msg_config() -> dict[str, str]:
    """Parse Msg_config into a {section_name: text} dict."""
    text = (_PROMPT_DIR / "Msg_config").read_text(encoding="utf-8")
    sections: dict[str, str] = {}
    current_key: str | None = None
    buf: list[str] = []

    for line in text.splitlines():
        m = re.match(r"^\[([A-Za-z0-9_]+)\]$", line.strip())
        if m:
            if current_key is not None:
                sections[current_key] = "\n".join(buf).strip()
            current_key = m.group(1).lower()
            buf = []
        else:
            buf.append(line)

    if current_key is not None:
        sections[current_key] = "\n".join(buf).strip()

    return sections


def get_system_prompt(section: str) -> str:
    """Return the system prompt for *section* (case-insensitive).

    Returns an empty string if the section is not found, so callers
    can always safely concatenate the result.
    """
    return _load_msg_config().get(section.lower(), "")


@lru_cache(maxsize=1)
def get_json_schema() -> str:
    """Return the full JSON format schema as a string."""
    return (_PROMPT_DIR / "json_format_schema.json").read_text(encoding="utf-8")


def list_sections() -> list[str]:
    """Return all available section names (useful for debugging / testing)."""
    return sorted(_load_msg_config().keys())
