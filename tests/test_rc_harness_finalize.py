"""ExperimentHarness auto-finalize + idempotency.

Regression for the Stage-12 acceptance finding: the generated experiment
crashed in its OWN final results serializer (`_sanitize` on an int dict key)
after reporting metrics, so its results.json was never written (metrics only
survived via the sandbox's stdout parsing). The injected harness now auto-writes
results.json on ANY exit, so reported metrics persist even if the experiment
dies afterwards.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from researchclaw.experiment.harness_template import ExperimentHarness


@pytest.fixture(autouse=True)
def _no_real_atexit(monkeypatch):
    """Stop harness instances created in tests from registering a REAL atexit
    hook, which would otherwise fire at pytest shutdown and write results.json
    into the repo root. Tests that assert registration override this locally."""
    import researchclaw.experiment.harness_template as h
    monkeypatch.setattr(h.atexit, "register", lambda *a, **k: None)


def _read_results(d: Path) -> dict:
    return json.loads((d / "results.json").read_text())


def test_safe_finalize_persists_reported_metrics(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    h = ExperimentHarness(time_budget=120)
    h.report_metric("validity_rate", 0.42)
    # simulate the experiment crashing AFTER reporting → atexit hook fires
    h._safe_finalize()
    out = _read_results(tmp_path)
    assert out["metrics"]["validity_rate"] == 0.42


def test_finalize_is_idempotent(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    h = ExperimentHarness(time_budget=120)
    h.report_metric("acc", 1.0)
    h.finalize()
    # tamper, then call again — second call must NOT overwrite (idempotent)
    (tmp_path / "results.json").write_text('{"sentinel": true}')
    h.finalize()
    h._safe_finalize()
    assert _read_results(tmp_path) == {"sentinel": True}


def test_safe_finalize_never_raises(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    h = ExperimentHarness(time_budget=120)
    # force finalize to blow up; _safe_finalize must swallow it
    def boom():
        raise RuntimeError("disk gone")
    h.finalize = boom  # type: ignore[method-assign]
    h._safe_finalize()  # must not raise


def test_atexit_registered_on_construction(tmp_path, monkeypatch):
    import atexit
    monkeypatch.chdir(tmp_path)
    registered = {}
    monkeypatch.setattr(atexit, "register", lambda fn, *a, **k: registered.setdefault("fn", fn))
    h = ExperimentHarness(time_budget=60)
    assert registered.get("fn") == h._safe_finalize
