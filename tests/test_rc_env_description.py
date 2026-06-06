"""_build_env_description: the beast-mode prompt must describe the *configured*
experiment environment, not a hardcoded Docker assumption.

Regression for the Stage-10 mismatch where the code generator was told the code
runs in "an isolated Docker container with everything pre-installed" while the
actual mode was sandbox (no network, limited packages) — which contributed to
synthetic/placeholder code.
"""

from __future__ import annotations

from pathlib import Path

from researchclaw.config import RCConfig
from researchclaw.pipeline.stage_impls._code_generation import _build_env_description


def _config(tmp_path: Path, experiment: dict) -> RCConfig:
    data = {
        "project": {"name": "rc-test", "mode": "docs-first"},
        "research": {"topic": "t", "domains": ["ml"]},
        "runtime": {"timezone": "UTC"},
        "notifications": {"channel": "local"},
        "knowledge_base": {"backend": "markdown", "root": str(tmp_path / "kb")},
        "llm": {
            "provider": "openrouter",
            "base_url": "https://openrouter.ai/api/v1",
            "api_key_env": "RC_TEST_KEY",
            "api_key": "inline",
            "primary_model": "anthropic/claude-sonnet-4.6",
        },
        "experiment": experiment,
    }
    return RCConfig.from_dict(data, project_root=tmp_path, check_paths=False)


def test_sandbox_describes_no_network_no_pip(tmp_path: Path) -> None:
    cfg = _config(tmp_path, {"mode": "sandbox", "sandbox": {"python_path": "/opt/env/bin/python"}})
    desc = _build_env_description(cfg)
    assert "sandbox" in desc.lower()
    assert "/opt/env/bin/python" in desc
    assert "NO internet" in desc and "NO pip" in desc
    # Must NOT lie about a Docker container with everything pre-installed.
    assert "Docker" not in desc


def test_docker_offline_says_no_install(tmp_path: Path) -> None:
    cfg = _config(tmp_path, {"mode": "docker", "docker": {"network_policy": "none"}})
    desc = _build_env_description(cfg)
    assert "Docker" in desc
    assert "offline" in desc.lower()
    assert "NOT installed" in desc


def test_docker_setup_only_allows_requirements(tmp_path: Path) -> None:
    cfg = _config(tmp_path, {"mode": "docker", "docker": {"network_policy": "setup_only"}})
    desc = _build_env_description(cfg)
    assert "requirements.txt" in desc


def test_ssh_and_colab_modes(tmp_path: Path) -> None:
    ssh = _build_env_description(_config(tmp_path, {"mode": "ssh_remote"}))
    assert "SSH" in ssh
    colab = _build_env_description(_config(tmp_path, {"mode": "colab_drive"}))
    assert "Colab" in colab


def test_unknown_mode_falls_back_truthfully(tmp_path: Path) -> None:
    desc = _build_env_description(_config(tmp_path, {"mode": "simulated"}))
    assert "simulated" in desc
    assert "GUIDANCE.md" in desc
