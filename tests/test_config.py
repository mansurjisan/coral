"""Tests for config module."""

import pytest

from coral.config import get_all_model_assignments, get_model, get_ollama_host, load_mcp_config, set_cli_model


class TestGetModel:
    """Test the four-tier fallback: stage env -> CORAL_MODEL -> CLI --model -> default."""

    def _clear_model_env(self, monkeypatch):
        for var in [
            "CORAL_MODEL", "CORAL_MODEL_ROUTER", "CORAL_MODEL_SYNTHESIS",
            "CORAL_MODEL_DATA", "CORAL_MODEL_CODE", "CORAL_MODEL_WORKFLOW",
            "CORAL_MODEL_ESCALATION",
        ]:
            monkeypatch.delenv(var, raising=False)
        set_cli_model("")

    def test_tier4_hardcoded_default(self, monkeypatch):
        self._clear_model_env(monkeypatch)
        assert get_model() == "qwen3:32b"

    def test_tier3_cli_model(self, monkeypatch):
        self._clear_model_env(monkeypatch)
        set_cli_model("cli-model")
        try:
            assert get_model() == "cli-model"
            assert get_model("data") == "cli-model"
        finally:
            set_cli_model("")

    def test_tier2_coral_model_beats_cli(self, monkeypatch):
        self._clear_model_env(monkeypatch)
        monkeypatch.setenv("CORAL_MODEL", "env-model")
        set_cli_model("cli-model")
        try:
            assert get_model() == "env-model"
            assert get_model("data") == "env-model"
        finally:
            set_cli_model("")

    def test_tier1_stage_env_beats_all(self, monkeypatch):
        self._clear_model_env(monkeypatch)
        monkeypatch.setenv("CORAL_MODEL", "env-model")
        monkeypatch.setenv("CORAL_MODEL_CODE", "stage-coder")
        set_cli_model("cli-model")
        try:
            assert get_model("code") == "stage-coder"
            # Other stages still fall back to CORAL_MODEL
            assert get_model("data") == "env-model"
        finally:
            set_cli_model("")

    def test_all_stages_covered(self, monkeypatch):
        self._clear_model_env(monkeypatch)
        monkeypatch.setenv("CORAL_MODEL", "base")
        monkeypatch.setenv("CORAL_MODEL_ROUTER", "router-model")
        monkeypatch.setenv("CORAL_MODEL_SYNTHESIS", "synth-model")
        monkeypatch.setenv("CORAL_MODEL_DATA", "data-model")
        monkeypatch.setenv("CORAL_MODEL_CODE", "code-model")
        monkeypatch.setenv("CORAL_MODEL_WORKFLOW", "wf-model")
        monkeypatch.setenv("CORAL_MODEL_ESCALATION", "esc-model")
        assignments = get_all_model_assignments()
        assert assignments["default"] == "base"
        assert assignments["router"] == "router-model"
        assert assignments["synthesis"] == "synth-model"
        assert assignments["data"] == "data-model"
        assert assignments["code"] == "code-model"
        assert assignments["workflow"] == "wf-model"
        assert assignments["escalation"] == "esc-model"

    def test_empty_stage_env_falls_back(self, monkeypatch):
        self._clear_model_env(monkeypatch)
        monkeypatch.setenv("CORAL_MODEL", "qwen3:32b")
        monkeypatch.setenv("CORAL_MODEL_CODE", "")
        assert get_model("code") == "qwen3:32b"

    def test_unknown_stage_returns_base(self, monkeypatch):
        self._clear_model_env(monkeypatch)
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
