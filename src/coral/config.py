"""Configuration loading for CORAL."""

import json
import os
from pathlib import Path


DEFAULT_MODEL = "qwen3:32b"
DEFAULT_CONFIG_PATH = "coral_config.json"
DEFAULT_OLLAMA_HOST = "http://localhost:11434"

# Stage-specific env var names, resolved with fallback chaining:
#   stage env var -> CORAL_MODEL -> CLI --model -> DEFAULT_MODEL
_STAGE_ENV_VARS = {
    "router": "CORAL_MODEL_ROUTER",
    "synthesis": "CORAL_MODEL_SYNTHESIS",
    "data": "CORAL_MODEL_DATA",
    "code": "CORAL_MODEL_CODE",
    "workflow": "CORAL_MODEL_WORKFLOW",
}


def get_model(stage: str | None = None) -> str:
    """Resolve model name with fallback chaining.

    Resolution order:
      1. Stage-specific env var (e.g. CORAL_MODEL_CODE)
      2. CORAL_MODEL
      3. DEFAULT_MODEL

    Args:
        stage: Optional stage name (router, synthesis, data, code, workflow).
               If None, returns the base model.
    """
    if stage:
        env_var = _STAGE_ENV_VARS.get(stage)
        if env_var:
            stage_model = os.environ.get(env_var, "").strip()
            if stage_model:
                return stage_model
    return os.environ.get("CORAL_MODEL", DEFAULT_MODEL)


def get_all_model_assignments() -> dict[str, str]:
    """Return the resolved model for every stage. Useful for audit logging."""
    return {
        "default": get_model(),
        "router": get_model("router"),
        "synthesis": get_model("synthesis"),
        "data": get_model("data"),
        "code": get_model("code"),
        "workflow": get_model("workflow"),
    }


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
