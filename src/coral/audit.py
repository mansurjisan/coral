"""Structured audit logging for CORAL query and tool execution."""

from __future__ import annotations

import json
import logging
import os
from contextlib import contextmanager
from contextvars import ContextVar
from datetime import datetime, timezone
from pathlib import Path
from uuid import uuid4

logger = logging.getLogger(__name__)

_REQUEST_CONTEXT: ContextVar[dict | None] = ContextVar("coral_request_context", default=None)
_SECTION_CONTEXT: ContextVar[str | None] = ContextVar("coral_section_context", default=None)
_AUDIT_PREFIX = "__CORAL_AUDIT__="


def _audit_log_path() -> Path:
    path = os.environ.get("CORAL_AUDIT_LOG", "logs/coral_audit.jsonl").strip()
    return Path(path)


def new_query_id() -> str:
    """Return a short unique query identifier."""
    return uuid4().hex[:12]


@contextmanager
def request_context(
    query_id: str | None = None,
    route: list[str] | None = None,
    mode: str | None = None,
):
    """Attach request-level context to the current execution."""
    existing = _REQUEST_CONTEXT.get()
    context = {
        "query_id": query_id or (existing or {}).get("query_id") or new_query_id(),
        "route": list(route) if route is not None else list((existing or {}).get("route", [])),
        "mode": mode or (existing or {}).get("mode"),
    }
    token = _REQUEST_CONTEXT.set(context)
    try:
        yield context["query_id"]
    finally:
        _REQUEST_CONTEXT.reset(token)


def set_request_route(route: list[str]) -> None:
    """Update the routed sections for the active request."""
    context = _REQUEST_CONTEXT.get()
    if context is None:
        return
    context["route"] = list(route)


@contextmanager
def section_context(section: str):
    """Attach the current routed section to nested tool calls."""
    token = _SECTION_CONTEXT.set(section)
    try:
        yield
    finally:
        _SECTION_CONTEXT.reset(token)


def summarize_arguments(arguments: dict | None, max_length: int = 240) -> str:
    """Serialize tool arguments to a short stable string for audit logs."""
    if arguments is None:
        return "{}"

    summary = json.dumps(arguments, sort_keys=True, default=str)
    if len(summary) <= max_length:
        return summary
    return summary[: max_length - 3] + "..."


def record_audit_event(event: str, **fields) -> None:
    """Append a JSONL audit record with current request context."""
    request = _REQUEST_CONTEXT.get() or {}
    entry = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "event": event,
        "query_id": request.get("query_id"),
        "mode": request.get("mode"),
        "routed_sections": request.get("route", []),
        "section": _SECTION_CONTEXT.get(),
    }
    entry.update(fields)

    path = _audit_log_path()
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("a", encoding="utf-8") as f:
            json.dump(entry, f, sort_keys=True)
            f.write("\n")
    except OSError as exc:
        logger.warning("Failed to write audit log %s: %s", path, exc)


def with_tool_audit_payload(result: str, **payload) -> str:
    """Prefix tool output with machine-readable audit metadata."""
    prefix = _AUDIT_PREFIX + json.dumps(payload, sort_keys=True, separators=(",", ":"))
    if result:
        return prefix + "\n" + result
    return prefix


def split_tool_audit_payload(result: str) -> tuple[dict, str]:
    """Extract audit metadata from a tool result string."""
    if not result.startswith(_AUDIT_PREFIX):
        return {}, result

    first_line, separator, remainder = result.partition("\n")
    try:
        payload = json.loads(first_line[len(_AUDIT_PREFIX):])
    except json.JSONDecodeError:
        return {}, result

    cleaned = remainder if separator else ""
    return payload, cleaned
