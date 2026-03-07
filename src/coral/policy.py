"""Central authorization policy for CORAL sections and trusted servers."""

from __future__ import annotations

import json
import os
from copy import deepcopy
from functools import lru_cache
from importlib.resources import files
from typing import Iterable


def _manifest_path():
    return files("coral").joinpath("policy_manifest.json")


@lru_cache(maxsize=1)
def load_policy_manifest() -> dict:
    """Load and validate the packaged policy manifest."""
    with _manifest_path().open(encoding="utf-8") as f:
        manifest = json.load(f)
    _validate_manifest(manifest)
    return manifest


def _validate_manifest(manifest: dict) -> None:
    if not isinstance(manifest, dict):
        raise ValueError("Policy manifest must be a JSON object")

    environments = manifest.get("environments")
    sections = manifest.get("sections")
    servers = manifest.get("servers")

    if not isinstance(environments, list) or not environments:
        raise ValueError("Policy manifest must define supported environments")
    if not isinstance(sections, dict) or not sections:
        raise ValueError("Policy manifest must define sections")
    if not isinstance(servers, dict) or not servers:
        raise ValueError("Policy manifest must define servers")

    known_envs = set(environments)
    default_env = manifest.get("default_environment")
    if default_env not in known_envs:
        raise ValueError(
            f"Policy manifest default environment '{default_env}' is not in environments"
        )

    for section_name, section_cfg in sections.items():
        section_servers = section_cfg.get("servers")
        if not isinstance(section_servers, list):
            raise ValueError(f"Section '{section_name}' must define a server list")
        unknown_servers = sorted(set(section_servers) - set(servers))
        if unknown_servers:
            joined = ", ".join(unknown_servers)
            raise ValueError(
                f"Section '{section_name}' references unknown servers: {joined}"
            )

    for server_name, server_cfg in servers.items():
        server_envs = server_cfg.get("environments")
        sandbox_required = server_cfg.get("sandbox_required")
        if not isinstance(server_envs, list) or not server_envs:
            raise ValueError(f"Server '{server_name}' must define environments")

        unknown_envs = sorted(set(server_envs) - known_envs)
        if unknown_envs:
            joined = ", ".join(unknown_envs)
            raise ValueError(
                f"Server '{server_name}' references unsupported environments: {joined}"
            )

        if not isinstance(sandbox_required, dict):
            raise ValueError(
                f"Server '{server_name}' must define sandbox_required by environment"
            )

        missing_sandbox_envs = sorted(set(server_envs) - set(sandbox_required))
        if missing_sandbox_envs:
            joined = ", ".join(missing_sandbox_envs)
            raise ValueError(
                f"Server '{server_name}' is missing sandbox_required values for: {joined}"
            )


def get_runtime_environment() -> str:
    """Return the configured runtime environment."""
    manifest = load_policy_manifest()
    default_env = manifest["default_environment"]
    environment = _normalize_environment(os.environ.get("CORAL_ENV", default_env))
    if environment not in manifest["environments"]:
        supported = ", ".join(manifest["environments"])
        raise ValueError(
            f"Unsupported CORAL_ENV '{environment}'. Expected one of: {supported}"
        )
    return environment


def _normalize_environment(environment: str) -> str:
    return environment.strip().lower()


def _resolve_environment(environment: str | None) -> str:
    """Return a validated runtime environment."""
    manifest = load_policy_manifest()
    if environment is None:
        return get_runtime_environment()
    normalized = _normalize_environment(environment)
    if normalized not in manifest["environments"]:
        supported = ", ".join(manifest["environments"])
        raise ValueError(
            f"Unsupported environment '{normalized}'. Expected one of: {supported}"
        )
    return normalized


def get_server_policy(server_name: str) -> dict:
    """Return the policy metadata for a trusted server."""
    manifest = load_policy_manifest()
    server_key = server_name.strip().lower()
    try:
        return deepcopy(manifest["servers"][server_key])
    except KeyError as exc:
        raise ValueError(f"Unknown server '{server_name}' in policy manifest") from exc


def get_section_servers(section: str, environment: str | None = None) -> list[str]:
    """Return allowed servers for a section in the active environment."""
    manifest = load_policy_manifest()
    section_key = section.strip().lower()
    if section_key not in manifest["sections"]:
        known_sections = ", ".join(sorted(manifest["sections"]))
        raise ValueError(
            f"Unknown section '{section}'. Expected one of: {known_sections}"
        )

    env = _resolve_environment(environment)

    allowed = []
    for server_name in manifest["sections"][section_key]["servers"]:
        if env in manifest["servers"][server_name]["environments"]:
            allowed.append(server_name)
    return allowed


def is_server_authorized(
    section: str,
    server_name: str,
    environment: str | None = None,
) -> bool:
    """Return whether a server is authorized for a section."""
    return server_name.strip().lower() in get_section_servers(section, environment)


def server_requires_sandbox(
    server_name: str,
    environment: str | None = None,
) -> bool:
    """Return whether a server requires sandboxing in the target environment."""
    server_policy = get_server_policy(server_name)
    env = _resolve_environment(environment)
    if env not in server_policy["environments"]:
        return False
    return bool(server_policy["sandbox_required"].get(env, False))


def validate_configured_servers(
    server_names: Iterable[str],
    environment: str | None = None,
) -> None:
    """Fail closed if config includes unknown or disallowed servers."""
    manifest = load_policy_manifest()
    env = _resolve_environment(environment)
    known_servers = set(manifest["servers"])

    configured = {name.strip().lower() for name in server_names}
    unknown = sorted(configured - known_servers)
    if unknown:
        joined = ", ".join(unknown)
        raise ValueError(f"Unapproved MCP servers in config: {joined}")

    blocked = sorted(
        name for name in configured
        if env not in manifest["servers"][name]["environments"]
    )
    if blocked:
        joined = ", ".join(blocked)
        raise ValueError(
            f"MCP servers not allowed in environment '{env}': {joined}"
        )
