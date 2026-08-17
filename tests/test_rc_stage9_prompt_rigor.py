"""P4: Stage 9 design prompt must be academic-rigor-first, not environment-first.

Regression guard against re-introducing pre-emptive degradation wording that
tells the designer to drop scientifically-required tools just because they are
not pre-installed.
"""

from __future__ import annotations

from researchclaw.prompts import PromptManager


def _design_prompt_text() -> str:
    pm = PromptManager()
    spec = pm._stages["experiment_design"]  # raw template (system + user)
    return (spec.get("system", "") + "\n" + spec.get("user", "")).lower()


def test_no_preemptive_degradation_clause() -> None:
    t = _design_prompt_text()
    # the old environment-first clause must be gone
    assert "do not design around resources you cannot obtain" not in t
    assert "prefer a method that is feasible here rather than assuming it exists" not in t


def test_has_rigor_first_and_provisioning_promise() -> None:
    t = _design_prompt_text()
    assert "academic rigor first" in t
    # the design must be told deps will be provisioned (so it declares, not dodges)
    assert "pip install" in t and "download" in t
    assert "do not downgrade the science" in t or "do not substitute a weaker proxy" in t
    # system libs → operator prompt, not avoidance
    assert "operator" in t


def test_environment_manifest_still_documented() -> None:
    t = _design_prompt_text()
    assert "environment" in t
    assert "compute" in t and "datasets" in t
