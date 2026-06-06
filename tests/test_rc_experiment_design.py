"""Stage 9 experiment-design plan parsing robustness.

Regression for the json/yaml-truncation + wrapper-key bug that made the design
LLM output unparseable and fall back to a context-free retry (producing generic
128xA100 / 34B / GPT-4 plans that ignored the real 2x3090 hardware and the
synthesized hypotheses).
"""

from __future__ import annotations

from researchclaw.pipeline.stage_impls._experiment_design import _unwrap_plan_dict


class TestUnwrapPlanDict:
    def test_unwraps_single_parent_key(self) -> None:
        nested = {
            "experiment_plan": {
                "objectives": ["o1"],
                "baselines": ["b1"],
                "proposed_methods": ["m1"],
            }
        }
        out = _unwrap_plan_dict(nested)
        assert "objectives" in out
        assert out["baselines"] == ["b1"]

    def test_leaves_flat_plan_untouched(self) -> None:
        flat = {"objectives": ["o1"], "baselines": ["b1"]}
        assert _unwrap_plan_dict(flat) is flat

    def test_does_not_unwrap_non_plan_wrapper(self) -> None:
        # Single key but child has no plan keys -> don't unwrap arbitrarily.
        d = {"some_key": {"unrelated": 1}}
        assert _unwrap_plan_dict(d) == d

    def test_does_not_unwrap_list_child(self) -> None:
        d = {"objectives": ["o1"]}  # single key, value is a list
        assert _unwrap_plan_dict(d) is d

    def test_non_dict_passthrough(self) -> None:
        assert _unwrap_plan_dict(None) is None
        assert _unwrap_plan_dict(["a"]) == ["a"]


class TestConfiguredDomainPriority:
    """config.research.domains must override RL-heavy text inference."""

    def test_configured_domain_overrides_keyword_inference(self) -> None:
        from researchclaw.domains.detector import detect_domain

        # RL-heavy text alone would classify as ml_rl
        rl_text = (
            "reinforcement learning reward policy GRPO PPO geometric reward "
            "RL training reward signal"
        )
        inferred = detect_domain(topic=rl_text)
        assert inferred.domain_id == "ml_rl"

        # With configured domains, the declared field wins
        forced = detect_domain(topic=rl_text, configured_domains=["ml_generative", "ml_nlp"])
        assert forced.domain_id == "ml_generative"

    def test_unknown_configured_domain_falls_through(self) -> None:
        from researchclaw.domains.detector import detect_domain

        rl_text = "reinforcement learning reward policy GRPO PPO reward signal"
        # bogus configured domain -> ignored, falls back to inference
        out = detect_domain(topic=rl_text, configured_domains=["not_a_domain"])
        assert out.domain_id == "ml_rl"
