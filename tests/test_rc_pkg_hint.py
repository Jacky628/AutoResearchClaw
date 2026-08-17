"""Stage 10 pkg_hint must reflect the REAL runtime environment (缺陷1).

The old hint was a hardcoded static string ("numpy, torch, sklearn, scipy,
pandas") that both hid installed packages (transformers/peft/trl) and falsely
claimed packages that were missing (sklearn) — pushing the code generator toward
synthetic/placeholder code. These tests pin the probe-based behavior.
"""

from __future__ import annotations

import sys
from pathlib import Path
from unittest.mock import patch

from researchclaw.config import RCConfig
from researchclaw.pipeline.stage_impls._code_generation import (
    _build_pkg_hint,
    _probe_packages,
)
from researchclaw.prompts import PromptManager


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


# --- _probe_packages (integration: probes the test interpreter itself) -------

def test_probe_detects_real_packages() -> None:
    probed = _probe_packages(sys.executable, candidates=("json", "sys", "numpy", "definitely_not_a_real_pkg_xyz"))
    assert probed is not None
    assert probed["json"] is True
    assert probed["sys"] is True
    assert probed["definitely_not_a_real_pkg_xyz"] is False


def test_probe_returns_none_on_bad_interpreter() -> None:
    assert _probe_packages("/nonexistent/python/interp") is None


# --- _build_pkg_hint with probe ----------------------------------------------

def test_pkg_hint_uses_probed_truth(tmp_path: Path) -> None:
    cfg = _config(tmp_path, {"mode": "sandbox", "sandbox": {"python_path": "/opt/env/bin/python"}})
    probed = {"numpy": True, "torch": True, "transformers": True, "peft": True,
              "sklearn": False, "cadquery": False}
    with patch(
        "researchclaw.pipeline.stage_impls._code_generation._probe_packages",
        return_value=probed,
    ):
        hint = _build_pkg_hint(cfg, {"has_gpu": True, "gpu_type": "cuda", "gpu_name": "RTX 3090", "tier": "high"}, PromptManager())
    # installed ones surfaced (so the model writes real fine-tuning, not numpy fakes)
    assert "transformers" in hint and "peft" in hint
    # truly-missing ones NOT advertised as available, and flagged as ImportError risk
    assert "sklearn" not in hint.split("WILL raise ImportError")[0]
    assert "ImportError" in hint
    assert "/opt/env/bin/python" in hint
    assert "cuda" in hint


def test_pkg_hint_falls_back_to_static_when_probe_fails(tmp_path: Path) -> None:
    cfg = _config(tmp_path, {"mode": "sandbox", "sandbox": {"python_path": "/bad/python"}})
    with patch(
        "researchclaw.pipeline.stage_impls._code_generation._probe_packages",
        return_value=None,
    ):
        hint = _build_pkg_hint(cfg, None, PromptManager())
    # no-GPU + probe-fail → the static pkg_hint_sandbox block
    assert "AVAILABLE PACKAGES" in hint or "numpy" in hint


def test_pkg_hint_docker_keeps_curated_list(tmp_path: Path) -> None:
    cfg = _config(tmp_path, {"mode": "docker", "docker": {"network_policy": "setup_only"}})
    hint = _build_pkg_hint(cfg, {"has_gpu": True, "gpu_type": "cuda", "gpu_name": "A100", "tier": "high"}, PromptManager())
    assert "docker mode" in hint
    assert "transformers" in hint
    assert "requirements.txt" in hint


def test_pkg_hint_empty_for_non_sandbox_modes(tmp_path: Path) -> None:
    cfg = _config(tmp_path, {"mode": "simulated"})
    assert _build_pkg_hint(cfg, None, PromptManager()) == ""
