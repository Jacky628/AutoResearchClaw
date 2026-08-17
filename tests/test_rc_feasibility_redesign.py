"""P3: infeasible-plan → automatic redesign loop in Stage 9."""

from __future__ import annotations

import json
import sys
from pathlib import Path
from types import SimpleNamespace

import yaml

from researchclaw.config import RCConfig
from researchclaw.pipeline.stage_impls._experiment_design import (
    _feasibility_redesign,
    _resolve_environment_for_plan,
    _write_environment_resolution,
)


def _config(tmp_path: Path) -> RCConfig:
    data = {
        "project": {"name": "rc-test", "mode": "docs-first"},
        "research": {"topic": "t", "domains": ["ml"]},
        "runtime": {"timezone": "UTC"},
        "notifications": {"channel": "local"},
        "knowledge_base": {"backend": "markdown", "root": str(tmp_path / "kb")},
        "llm": {"provider": "openrouter", "base_url": "x", "api_key_env": "K",
                "api_key": "i", "primary_model": "m"},
        "experiment": {"mode": "sandbox", "sandbox": {"python_path": sys.executable}},
    }
    return RCConfig.from_dict(data, project_root=tmp_path, check_paths=False)


def _run_dir(tmp_path: Path, vram_mb: int = 24576, gpu_count: int = 2) -> Path:
    rd = tmp_path / "run"
    (rd / "stage-01").mkdir(parents=True)
    (rd / "stage-01" / "hardware_profile.json").write_text(json.dumps({
        "has_gpu": True, "gpu_type": "cuda", "gpu_name": "RTX 3090",
        "vram_mb": vram_mb, "gpu_count": gpu_count, "tier": "high",
    }))
    return rd


def _plan(min_vram_gb: int) -> dict:
    return {
        "objectives": ["o"],
        "environment": {"compute": {"gpu": "required", "min_vram_gb": min_vram_gb, "gpus": 1}},
    }


class _FakeLLM:
    def __init__(self, yaml_responses):
        self._responses = list(yaml_responses)
        self.calls = 0

    def chat(self, messages, max_tokens=0):
        idx = min(self.calls, len(self._responses) - 1)
        self.calls += 1
        return SimpleNamespace(content=self._responses[idx])


def test_no_redesign_when_feasible(tmp_path: Path) -> None:
    cfg, rd = _config(tmp_path), _run_dir(tmp_path)
    plan = _plan(24)  # 24GB ask on 24GB hw → OK
    res = _resolve_environment_for_plan(plan, rd, cfg)
    assert res.provisionable is True
    llm = _FakeLLM(["should not be called"])
    out_plan, out_res, attempts = _feasibility_redesign(plan, res, rd, cfg, llm)
    assert attempts == 0
    assert llm.calls == 0
    assert out_plan is plan


def test_redesign_fixes_infeasible_plan(tmp_path: Path) -> None:
    cfg, rd = _config(tmp_path), _run_dir(tmp_path)
    plan = _plan(40)  # 40GB ask on 24GB → INSUFFICIENT
    res = _resolve_environment_for_plan(plan, rd, cfg)
    assert res.provisionable is False
    feasible_yaml = yaml.dump(_plan(24))
    llm = _FakeLLM([feasible_yaml])
    out_plan, out_res, attempts = _feasibility_redesign(plan, res, rd, cfg, llm)
    assert attempts == 1
    assert out_res.provisionable is True
    assert out_plan["environment"]["compute"]["min_vram_gb"] == 24


def test_redesign_gives_up_after_max_attempts(tmp_path: Path) -> None:
    cfg, rd = _config(tmp_path), _run_dir(tmp_path)
    plan = _plan(40)
    res = _resolve_environment_for_plan(plan, rd, cfg)
    # LLM keeps returning an infeasible plan
    llm = _FakeLLM([yaml.dump(_plan(40))])
    out_plan, out_res, attempts = _feasibility_redesign(plan, res, rd, cfg, llm, max_attempts=2)
    assert attempts == 2
    assert out_res.provisionable is False


def test_no_redesign_without_llm(tmp_path: Path) -> None:
    cfg, rd = _config(tmp_path), _run_dir(tmp_path)
    plan = _plan(40)
    res = _resolve_environment_for_plan(plan, rd, cfg)
    out_plan, out_res, attempts = _feasibility_redesign(plan, res, rd, cfg, None)
    assert attempts == 0
    assert out_res.provisionable is False


def test_write_resolution_marks_infeasible(tmp_path: Path) -> None:
    cfg, rd = _config(tmp_path), _run_dir(tmp_path)
    stage_dir = tmp_path / "stage-09"; stage_dir.mkdir()
    res = _resolve_environment_for_plan(_plan(40), rd, cfg)
    ok = _write_environment_resolution(stage_dir, res, redesign_attempts=2)
    assert ok is True
    assert (stage_dir / "INFEASIBLE.md").exists()
    data = json.loads((stage_dir / "environment_resolution.json").read_text())
    assert data["redesign_attempts"] == 2
    assert data["provisionable"] is False


def test_write_resolution_no_marker_when_feasible(tmp_path: Path) -> None:
    cfg, rd = _config(tmp_path), _run_dir(tmp_path)
    stage_dir = tmp_path / "stage-09"; stage_dir.mkdir()
    res = _resolve_environment_for_plan(_plan(24), rd, cfg)
    _write_environment_resolution(stage_dir, res, redesign_attempts=0)
    assert not (stage_dir / "INFEASIBLE.md").exists()


def test_write_resolution_emits_operator_setup_for_system_libs(tmp_path: Path) -> None:
    cfg, rd = _config(tmp_path), _run_dir(tmp_path)
    stage_dir = tmp_path / "stage-09"; stage_dir.mkdir()
    plan = {"objectives": ["o"], "environment": {
        "pip": ["cadquery"],
        "system": ["libgl1"],
        "compute": {"gpu": "required", "min_vram_gb": 24, "gpus": 1},
    }}
    res = _resolve_environment_for_plan(plan, rd, cfg)
    _write_environment_resolution(stage_dir, res, redesign_attempts=0)
    op = stage_dir / "OPERATOR_SETUP.md"
    assert op.exists()
    body = op.read_text()
    assert "sudo apt-get install -y libgl1" in body
    # provisionable (operator-fixable) → not marked infeasible
    assert not (stage_dir / "INFEASIBLE.md").exists()
