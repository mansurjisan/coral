"""Configuration loading for CORAL."""

import json
import os
from pathlib import Path


DEFAULT_MODEL = "qwen3:32b"
DEFAULT_CONFIG_PATH = "coral_config.json"
DEFAULT_OLLAMA_HOST = "http://localhost:11434"

# Stage-specific env var names.
_STAGE_ENV_VARS = {
    "router": "CORAL_MODEL_ROUTER",
    "synthesis": "CORAL_MODEL_SYNTHESIS",
    "data": "CORAL_MODEL_DATA",
    "code": "CORAL_MODEL_CODE",
    "workflow": "CORAL_MODEL_WORKFLOW",
    "escalation": "CORAL_MODEL_ESCALATION",
}

# Set once by CLI at startup so the resolver has a single source of truth.
_cli_model: str = ""


def set_cli_model(model: str) -> None:
    """Register the model passed via CLI --model flag.

    Called once at CLI startup. All subsequent get_model() calls use this
    as the third tier in the fallback chain.
    """
    global _cli_model
    _cli_model = model.strip() if model else ""


def get_model(stage: str | None = None) -> str:
    """Resolve model name with fallback chaining.

    Resolution order (first non-empty wins):
      1. Stage-specific env var (e.g. CORAL_MODEL_CODE)
      2. CORAL_MODEL env var
      3. CLI --model (registered via set_cli_model)
      4. DEFAULT_MODEL

    Args:
        stage: Optional stage name (router, synthesis, data, code, workflow,
               escalation). If None, skips step 1.
    """
    # 1. Stage-specific override
    if stage:
        env_var = _STAGE_ENV_VARS.get(stage)
        if env_var:
            value = os.environ.get(env_var, "").strip()
            if value:
                return value

    # 2. Base env var
    base_env = os.environ.get("CORAL_MODEL", "").strip()
    if base_env:
        return base_env

    # 3. CLI --model
    if _cli_model:
        return _cli_model

    # 4. Hardcoded default
    return DEFAULT_MODEL


def get_all_model_assignments() -> dict[str, str]:
    """Return the resolved model for every stage. Useful for audit logging."""
    return {
        "default": get_model(),
        "router": get_model("router"),
        "synthesis": get_model("synthesis"),
        "data": get_model("data"),
        "code": get_model("code"),
        "workflow": get_model("workflow"),
        "escalation": get_model("escalation"),
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
