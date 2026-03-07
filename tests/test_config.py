"""Tests for config module."""

import pytest

from coral.config import get_ollama_host, load_mcp_config


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
