"""Tests for CORAL persistent memory."""

import json
import pytest
from pathlib import Path

from coral.memory import CoralMemory


@pytest.fixture
def memory(tmp_path):
    """Create a memory instance using a temp directory."""
    return CoralMemory(memory_dir=tmp_path)


class TestCoralMemory:
    def test_set_and_get(self, memory):
        memory.set("account", "coastal-act")
        assert memory.get("account") == "coastal-act"

    def test_get_missing_returns_none(self, memory):
        assert memory.get("nonexistent") is None

    def test_delete(self, memory):
        memory.set("key", "value")
        assert memory.delete("key") is True
        assert memory.get("key") is None

    def test_delete_missing_returns_false(self, memory):
        assert memory.delete("nonexistent") is False

    def test_auto_learn(self, memory):
        memory.auto_learn("scratch_path", "/scratch5/purged/user")
        assert memory.get("scratch_path") == "/scratch5/purged/user"

    def test_list_all(self, memory):
        memory.set("account", "coastal-act")
        memory.auto_learn("host", "ufe03")
        entries = memory.list_all()
        assert "account" in entries
        assert "[auto] host" in entries

    def test_persistence(self, tmp_path):
        """Memory persists across instances."""
        mem1 = CoralMemory(memory_dir=tmp_path)
        mem1.set("key", "value")

        mem2 = CoralMemory(memory_dir=tmp_path)
        assert mem2.get("key") == "value"

    def test_to_prompt_context_empty(self, memory):
        assert memory.to_prompt_context() == ""

    def test_to_prompt_context(self, memory):
        memory.set("account", "coastal-act")
        memory.set("partition", "hercules")
        ctx = memory.to_prompt_context()
        assert "coastal-act" in ctx
        assert "hercules" in ctx
        assert "previous sessions" in ctx

    def test_clear_all(self, memory):
        memory.set("key1", "val1")
        memory.auto_learn("key2", "val2")
        memory.clear_all()
        assert memory.list_all() == {}

    def test_corrupted_file_recovers(self, tmp_path):
        """Corrupted memory file doesn't crash, returns empty."""
        mem_file = tmp_path / "memory.json"
        mem_file.write_text("not valid json{{{")
        mem = CoralMemory(memory_dir=tmp_path)
        assert mem.list_all() == {}

    def test_user_overrides_auto(self, memory):
        """User-set entries should be separate from auto-learned."""
        memory.auto_learn("account", "auto-value")
        memory.set("account", "user-value")
        # User entry takes precedence in list_all (appears later)
        entries = memory.list_all()
        assert entries["account"] == "user-value"
        assert entries["[auto] account"] == "auto-value"
