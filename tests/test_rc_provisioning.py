"""Sandbox dependency provisioning (P2)."""

from __future__ import annotations

import sys
from pathlib import Path

from researchclaw.experiment.provisioning import ProvisionResult, provision_project


def test_disabled_when_policy_none(tmp_path: Path) -> None:
    res = provision_project(tmp_path, sys.executable, network_policy="none")
    assert res.python_path == sys.executable
    assert res.venv_created is False
    assert res.ok is True


def test_builds_venv_and_runs_setup(tmp_path: Path) -> None:
    # setup.py just writes a marker; no network needed → real subprocess.
    (tmp_path / "setup.py").write_text(
        "open('setup_ran.txt', 'w').write('ok')\n", encoding="utf-8"
    )
    res = provision_project(tmp_path, sys.executable, network_policy="setup_only", timeout_sec=120)
    assert res.venv_created is True
    assert (tmp_path / ".venv").is_dir()
    # the venv python is used, not the base interpreter
    assert res.python_path != sys.executable
    assert ".venv" in res.python_path
    assert res.setup_status == "ok"
    assert (tmp_path / "setup_ran.txt").exists()


def test_pip_not_needed_without_requirements(tmp_path: Path) -> None:
    res = provision_project(tmp_path, sys.executable, network_policy="pip_only", timeout_sec=120)
    assert res.pip_status == "not_needed"
    # pip_only does NOT run setup.py
    assert res.setup_status == "skipped"


def test_setup_failure_is_captured_not_raised(tmp_path: Path) -> None:
    (tmp_path / "setup.py").write_text("raise SystemExit(3)\n", encoding="utf-8")
    res = provision_project(tmp_path, sys.executable, network_policy="setup_only", timeout_sec=120)
    assert res.setup_status == "failed"
    assert res.ok is False
    assert res.errors


def test_pip_install_already_satisfied(tmp_path: Path) -> None:
    # Require a package guaranteed present (pip itself) → resolves offline, rc=0.
    (tmp_path / "requirements.txt").write_text("pip\n", encoding="utf-8")
    res = provision_project(tmp_path, sys.executable, network_policy="pip_only", timeout_sec=180)
    assert isinstance(res, ProvisionResult)
    assert res.pip_status == "ok"
    assert res.venv_created is True
