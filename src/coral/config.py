"""Configuration loading for CORAL."""

import json
import os
from pathlib import Path


DEFAULT_MODEL = "qwen3:32b"
DEFAULT_CONFIG_PATH = "coral_config.json"
DEFAULT_OLLAMA_HOST = "http://localhost:11434"


def get_model() -> str:
    """Get model name from environment or default."""
    return os.environ.get("CORAL_MODEL", DEFAULT_MODEL)


def get_ollama_host() -> str:
    """Get Ollama host URL from environment or default."""
    return os.environ.get("OLLAMA_HOST", DEFAULT_OLLAMA_HOST)


def load_mcp_config(config_path: str) -> dict:
    """Load MCP server configuration from JSON file."""
    path = Path(config_path)
    if not path.exists():
        raise FileNotFoundError(f"MCP config not found: {config_path}")
    with open(path) as f:
        return json.load(f)
