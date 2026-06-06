"""ExperimentSandbox.run_project provisioning wiring (P2)."""

from __future__ import annotations

import sys
from pathlib import Path

from researchclaw.config import SandboxConfig
from researchclaw.experiment.sandbox import ExperimentSandbox


def _project(tmp_path: Path) -> Path:
    proj = tmp_path / "proj"
    proj.mkdir()
    (proj / "main.py").write_text("print('validity_rate: 0.5')\n", encoding="utf-8")
    return proj


def test_no_provisioning_when_policy_none(tmp_path: Path) -> None:
    cfg = SandboxConfig(python_path=sys.executable, network_policy="none")
    sb = ExperimentSandbox(cfg, tmp_path / "work")
    res = sb.run_project(_project(tmp_path), timeout_sec=60)
    assert res.metrics.get("validity_rate") == 0.5
    sandbox_proj = next((tmp_path / "work").glob("_project_*"))
    assert not (sandbox_proj / "provision_log.txt").exists()
    assert not (sandbox_proj / ".venv").exists()


def test_provisioning_builds_venv_and_runs(tmp_path: Path) -> None:
    cfg = SandboxConfig(
        python_path=sys.executable, network_policy="setup_only", provision_timeout_sec=180
    )
    sb = ExperimentSandbox(cfg, tmp_path / "work")
    res = sb.run_project(_project(tmp_path), timeout_sec=120)
    # experiment still runs and reports its metric
    assert res.metrics.get("validity_rate") == 0.5
    sandbox_proj = next((tmp_path / "work").glob("_project_*"))
    assert (sandbox_proj / "provision_log.txt").exists()
    assert (sandbox_proj / ".venv").is_dir()
