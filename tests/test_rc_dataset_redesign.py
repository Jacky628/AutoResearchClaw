"""P8 integration: invented dataset id → redesign to a real source / gate marker."""

from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import yaml

import researchclaw.pipeline.stage_impls._experiment_design as ed


class _FakeLLM:
    def __init__(self, responses):
        self._responses = list(responses)
        self.calls = 0

    def chat(self, messages, max_tokens=0):
        idx = min(self.calls, len(self._responses) - 1)
        self.calls += 1
        return SimpleNamespace(content=self._responses[idx])


def _plan(source: str) -> dict:
    return {
        "objectives": ["o"],
        "datasets": [{"name": "DeepCAD", "source": source}],
        "environment": {"datasets": [{"name": "DeepCAD", "source": source}]},
    }


def test_missing_source_redesigned_to_real(monkeypatch):
    # verify: the invented id is MISSING (+candidates); the real one VERIFIED.
    def fake_verify(plan, *, online=True):
        src = (plan.get("datasets") or [{}])[0].get("source", "")
        if "wuchy143" in src:
            return {"DeepCAD": {"source": src, "status": "MISSING",
                                "candidates": [{"id": "wanhin/DEEPCAD-completion-sft",
                                                "downloads": 30, "gated": False}]}}
        return {"DeepCAD": {"source": src, "status": "VERIFIED", "candidates": []}}

    monkeypatch.setattr(ed, "_verify_plan_datasets", fake_verify)
    fixed_yaml = yaml.dump(_plan("wanhin/DEEPCAD-completion-sft"))
    llm = _FakeLLM([fixed_yaml])

    plan, verdicts, attempts = ed._dataset_source_redesign(
        _plan("wuchy143/deepcad"), Path("/tmp"), config=None, llm=llm
    )
    assert attempts == 1
    assert verdicts["DeepCAD"]["status"] == "VERIFIED"
    assert "wanhin" in plan["datasets"][0]["source"]


def test_no_redesign_when_already_verified(monkeypatch):
    monkeypatch.setattr(ed, "_verify_plan_datasets",
                        lambda plan, **k: {"DeepCAD": {"source": "wanhin/x", "status": "VERIFIED", "candidates": []}})
    llm = _FakeLLM(["unused"])
    plan, verdicts, attempts = ed._dataset_source_redesign(
        _plan("wanhin/x"), Path("/tmp"), config=None, llm=llm)
    assert attempts == 0 and llm.calls == 0


def test_missing_with_no_candidates_gives_up(monkeypatch):
    monkeypatch.setattr(ed, "_verify_plan_datasets",
                        lambda plan, **k: {"DeepCAD": {"source": "x/y", "status": "MISSING", "candidates": []}})
    llm = _FakeLLM(["unused"])
    plan, verdicts, attempts = ed._dataset_source_redesign(
        _plan("x/y"), Path("/tmp"), config=None, llm=llm)
    assert attempts == 0  # no candidates → nothing to suggest, don't loop


def test_dataset_unverified_marker_written(tmp_path, monkeypatch):
    # _write_environment_resolution should emit DATASET_UNVERIFIED.md on MISSING.
    res = SimpleNamespace(
        to_dict=lambda: {"declared": True, "provisionable": True},
        summary_md=lambda: "ok\n", declared=True, provisionable=True,
        needs_operator=[], compute_detail="", operator_setup_lines=lambda: [],
    )
    verdicts = {"DeepCAD": {"source": "wuchy143/deepcad", "status": "MISSING",
                            "candidates": [{"id": "wanhin/DEEPCAD-completion-sft"}]}}
    ok = ed._write_environment_resolution(tmp_path, res, 0, dataset_verdicts=verdicts)
    assert ok
    marker = tmp_path / "DATASET_UNVERIFIED.md"
    assert marker.exists()
    body = marker.read_text()
    assert "wuchy143/deepcad" in body and "wanhin/DEEPCAD-completion-sft" in body
    # and verdicts are recorded in the json
    data = json.loads((tmp_path / "environment_resolution.json").read_text())
    assert data["dataset_verification"]["DeepCAD"]["status"] == "MISSING"
