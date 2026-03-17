"""Tests for CORAL's central authorization policy manifest."""

import pytest

from coral.policy import (
    get_section_servers,
    get_server_policy,
    get_runtime_environment,
    is_server_authorized,
    load_policy_manifest,
    server_requires_sandbox,
    validate_configured_servers,
)


class TestPolicyManifest:
    def test_manifest_loads(self):
        manifest = load_policy_manifest()
        assert manifest["version"] == 1
        assert "data" in manifest["sections"]
        assert "viz" in manifest["servers"]

    def test_default_environment_is_local(self, monkeypatch):
        monkeypatch.delenv("CORAL_ENV", raising=False)
        assert get_runtime_environment() == "local"

    def test_invalid_environment_raises(self, monkeypatch):
        monkeypatch.setenv("CORAL_ENV", "prod")
        with pytest.raises(ValueError, match="Unsupported CORAL_ENV"):
            get_runtime_environment()

    def test_get_section_servers_returns_authorized_servers(self):
        assert get_section_servers("data") == [
            "coops", "nhc", "stofs", "recon", "erddap", "ofs",
            "adcirc", "goes", "schism", "usgs", "winds", "ww3", "netcdf",
        ]
        assert get_section_servers("code") == ["rag", "viz"]
        assert get_section_servers("workflow") == ["slurm", "ecflow", "ufs_runner", "hpc_system", "nos_workflow"]

    def test_unknown_section_raises(self):
        with pytest.raises(ValueError, match="Unknown section"):
            get_section_servers("admin")

    def test_server_authorization_is_section_scoped(self):
        assert is_server_authorized("data", "coops")
        assert not is_server_authorized("data", "viz")
        assert is_server_authorized("workflow", "slurm")
        assert not is_server_authorized("workflow", "rag")

    def test_server_policy_exposes_risk_metadata(self):
        policy = get_server_policy("viz")
        assert policy["risk"] == "code-exec"
        assert "python_execute" in policy["allowed_operations"]

    def test_sandbox_requirement_is_environment_specific(self):
        assert not server_requires_sandbox("viz", environment="local")
        assert server_requires_sandbox("viz", environment="ursa")
        assert server_requires_sandbox("viz", environment="UrSa")
        assert not server_requires_sandbox("coops", environment="ursa")

    def test_validate_configured_servers_accepts_known_servers(self):
        validate_configured_servers(["coops", "rag", "slurm"])

    def test_validate_configured_servers_rejects_unknown_servers(self):
        with pytest.raises(ValueError, match="Unapproved MCP servers"):
            validate_configured_servers(["coops", "filesystem"])
