"""Deterministic oracle-sanity backstop (_audit_metric_validity).

Regression for the Stage-12 re-run: a lenient/gamed CadQuery oracle reported a
condition at validity 1.0 (> ground-truth 83%), i.e. an artifact, not a finding.
This backstop flags such metrics without trusting the generated oracle.
"""

from __future__ import annotations

from researchclaw.pipeline.stage_impls._execution import _audit_metric_validity


def test_exceeds_ground_truth_ceiling_flagged():
    stdout = "ORACLE_CALIBRATION: ground_truth_validity=0.83\n"
    metrics = {"condA/validity_rate_mean": 1.0, "condB/validity_rate_mean": 0.5}
    w = _audit_metric_validity(stdout, metrics)
    assert any("EXCEEDS" in x and "condA" in x for x in w)
    assert not any("condB" in x for x in w)  # 0.5 < 0.83 is fine


def test_within_ceiling_no_flag():
    stdout = "ORACLE_CALIBRATION: ground_truth_validity=0.83\n"
    metrics = {"condA/validity_rate_mean": 0.40, "condB/validity_rate_mean": 0.55}
    assert _audit_metric_validity(stdout, metrics) == []


def test_floor_ceiling_polarization_flagged_without_calibration():
    # the real re-run shape: scratch 0, qwen_no 0, constraint_tokens 1.0
    metrics = {
        "scratch/validity_rate_mean": 0.0,
        "qwen_no/validity_rate_mean": 0.0,
        "constraint/validity_rate_mean": 1.0,
    }
    w = _audit_metric_validity("", metrics)
    assert w and any("exactly 0.0 or 1.0" in x for x in w)
    assert any("NOT calibrated" in x for x in w)


def test_healthy_intermediate_rates_no_flag():
    metrics = {"a/validity_rate_mean": 0.31, "b/validity_rate_mean": 0.62,
               "c/validity_rate_mean": 0.74}
    assert _audit_metric_validity("", metrics) == []


def test_std_and_nan_keys_ignored():
    metrics = {"a/validity_rate_std": 1.0, "a/validity_rate_mean": float("nan"),
               "gap_validity": 1.0}
    # only std/nan/gap → no usable validity values → no warning
    assert _audit_metric_validity("", metrics) == []


def test_no_validity_metrics_returns_empty():
    assert _audit_metric_validity("anything", {"accuracy": 0.9, "loss": 0.1}) == []
