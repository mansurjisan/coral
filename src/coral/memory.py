"""Persistent memory for CORAL — stores user preferences across sessions."""

from __future__ import annotations

import json
import logging
from datetime import datetime, timezone
from pathlib import Path

logger = logging.getLogger(__name__)

# Default memory location
_DEFAULT_MEMORY_DIR = Path.home() / ".coral"
_MEMORY_FILE = "memory.json"


class CoralMemory:
    """Simple persistent key-value memory store.

    Stores user preferences, frequently used paths, account info,
    and other context that should persist across chat sessions.
    """

    def __init__(self, memory_dir: str | Path | None = None):
        self.memory_dir = Path(memory_dir) if memory_dir else _DEFAULT_MEMORY_DIR
        self.memory_file = self.memory_dir / _MEMORY_FILE
        self._data: dict = self._load()

    def _load(self) -> dict:
        """Load memory from disk."""
        if not self.memory_file.exists():
            return {"version": 1, "entries": {}, "auto": {}}
        try:
            data = json.loads(self.memory_file.read_text())
            if "entries" not in data:
                data["entries"] = {}
            if "auto" not in data:
                data["auto"] = {}
            return data
        except (json.JSONDecodeError, OSError) as e:
            logger.warning("Failed to load memory: %s", e)
            return {"version": 1, "entries": {}, "auto": {}}

    def _save(self) -> None:
        """Persist memory to disk."""
        self.memory_dir.mkdir(parents=True, exist_ok=True)
        self.memory_file.write_text(json.dumps(self._data, indent=2))

    def set(self, key: str, value: str) -> None:
        """Store a user-provided memory entry."""
        self._data["entries"][key] = {
            "value": value,
            "updated_at": datetime.now(timezone.utc).isoformat(),
        }
        self._save()

    def get(self, key: str) -> str | None:
        """Retrieve a memory entry."""
        entry = self._data["entries"].get(key)
        if entry:
            return entry["value"]
        return self._data["auto"].get(key, {}).get("value")

    def delete(self, key: str) -> bool:
        """Remove a memory entry. Returns True if found."""
        removed = False
        if key in self._data["entries"]:
            del self._data["entries"][key]
            removed = True
        if key in self._data["auto"]:
            del self._data["auto"][key]
            removed = True
        if removed:
            self._save()
        return removed

    def auto_learn(self, key: str, value: str) -> None:
        """Store an auto-learned fact (from tool results, etc.)."""
        self._data["auto"][key] = {
            "value": value,
            "updated_at": datetime.now(timezone.utc).isoformat(),
        }
        self._save()

    def list_all(self) -> dict[str, str]:
        """Return all memory entries (user + auto) as key -> value."""
        result = {}
        for key, entry in self._data.get("auto", {}).items():
            result[f"[auto] {key}"] = entry["value"]
        for key, entry in self._data.get("entries", {}).items():
            result[key] = entry["value"]
        return result

    def to_prompt_context(self) -> str:
        """Format memory as context to inject into system prompts."""
        entries = self.list_all()
        if not entries:
            return ""
        lines = ["The following user preferences and context are known from previous sessions:"]
        for key, value in entries.items():
            lines.append(f"- {key}: {value}")
        return "\n".join(lines)

    def clear_all(self) -> None:
        """Remove all memory entries."""
        self._data = {"version": 1, "entries": {}, "auto": {}}
        self._save()
