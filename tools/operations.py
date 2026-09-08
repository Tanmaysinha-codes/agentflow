"""Small, deterministic operations exposed as Gemini tools."""

from __future__ import annotations

from typing import Any


def record_extraction(title: str, priority: str = "medium") -> dict[str, Any]:
    """Record that an extracted task was accepted for orchestration."""
    return {"recorded": True, "title": title, "priority": priority}


def complete_task(title: str, owner: str | None = None) -> dict[str, Any]:
    """Mark an operational task complete for the current workflow run."""
    return {"completed": True, "title": title, "owner": owner}
