"""Tests for config module."""

import pytest

from coral.config import get_all_model_assignments, get_model, get_ollama_host, load_mcp_config


class TestGetModel:
    def test_default_model(self, monkeypatch):
        monkeypatch.delenv("CORAL_MODEL", raising=False)
        monkeypatch.delenv("CORAL_MODEL_CODE", raising=False)
        assert get_model() == "qwen3:32b"

    def test_base_model_from_env(self, monkeypatch):
        monkeypatch.setenv("CORAL_MODEL", "qwen3:30b-a3b")
        assert get_model() == "qwen3:30b-a3b"

    def test_stage_override(self, monkeypatch):
        monkeypatch.setenv("CORAL_MODEL", "qwen3:32b")
        monkeypatch.setenv("CORAL_MODEL_CODE", "qwen3-coder")
        assert get_model("code") == "qwen3-coder"

    def test_stage_falls_back_to_base(self, monkeypatch):
        monkeypatch.setenv("CORAL_MODEL", "qwen3:30b-a3b")
        monkeypatch.delenv("CORAL_MODEL_DATA", raising=False)
        assert get_model("data") == "qwen3:30b-a3b"

    def test_stage_falls_back_to_default(self, monkeypatch):
        monkeypatch.delenv("CORAL_MODEL", raising=False)
        monkeypatch.delenv("CORAL_MODEL_WORKFLOW", raising=False)
        assert get_model("workflow") == "qwen3:32b"

    def test_all_stages_covered(self, monkeypatch):
        monkeypatch.setenv("CORAL_MODEL", "base")
        monkeypatch.setenv("CORAL_MODEL_ROUTER", "router-model")
        monkeypatch.setenv("CORAL_MODEL_SYNTHESIS", "synth-model")
        monkeypatch.setenv("CORAL_MODEL_DATA", "data-model")
        monkeypatch.setenv("CORAL_MODEL_CODE", "code-model")
        monkeypatch.setenv("CORAL_MODEL_WORKFLOW", "wf-model")
        assignments = get_all_model_assignments()
        assert assignments["default"] == "base"
        assert assignments["router"] == "router-model"
        assert assignments["synthesis"] == "synth-model"
        assert assignments["data"] == "data-model"
        assert assignments["code"] == "code-model"
        assert assignments["workflow"] == "wf-model"

    def test_empty_stage_env_falls_back(self, monkeypatch):
        monkeypatch.setenv("CORAL_MODEL", "qwen3:32b")
        monkeypatch.setenv("CORAL_MODEL_CODE", "")
        assert get_model("code") == "qwen3:32b"

    def test_unknown_stage_returns_base(self, monkeypatch):
        monkeypatch.setenv("CORAL_MODEL", "qwen3:32b")
        assert get_model("nonexistent") == "qwen3:32b"


class TestGetOllamaHost:
    def test_default_host(self, monkeypatch):
        monkeypatch.delenv("OLLAMA_HOST", raising=False)
        assert get_ollama_host() == "http://localhost:11434"

    def test_custom_host_from_env(self, monkeypatch):
        monkeypatch.setenv("OLLAMA_HOST", "http://gpu-node:11434")
        assert get_ollama_host() == "http://gpu-node:11434"


class TestLoadMCPConfig:
    def test_loads_valid_config(self, tmp_path):
        config = tmp_path / "config.json"
        config.write_text('{"mcpServers": {"test": {"command": "echo"}}}')
        result = load_mcp_config(str(config))
        assert "mcpServers" in result
        assert "test" in result["mcpServers"]

    def test_missing_file_raises(self):
        with pytest.raises(FileNotFoundError):
            load_mcp_config("/nonexistent/path.json")

    def test_invalid_json_raises(self, tmp_path):
        config = tmp_path / "config.json"
        config.write_text("not json")
        with pytest.raises(Exception):
            load_mcp_config(str(config))
