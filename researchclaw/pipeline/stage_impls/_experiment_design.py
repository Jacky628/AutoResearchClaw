"""Stage 9: Experiment design."""

from __future__ import annotations

import json
import logging
import re
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import yaml

from researchclaw.adapters import AdapterBundle
from researchclaw.config import RCConfig
from researchclaw.llm.client import LLMClient
from researchclaw.pipeline._helpers import (
    StageResult,
    _build_context_preamble,
    _chat_with_prompt,
    _extract_yaml_block,
    _get_evolution_overlay,
    _load_hardware_profile,
    _read_prior_artifact,
    _safe_json_loads,
    _utcnow_iso,
)
from researchclaw.pipeline.environment import parse_manifest, resolve_environment
from researchclaw.pipeline.stages import Stage, StageStatus
from researchclaw.prompts import PromptManager

logger = logging.getLogger(__name__)


def _scan_cached_datasets(run_dir: Path) -> set[str]:
    """Collect lowercased names of datasets already present on disk, so the
    environment resolver can mark them CACHED rather than NEEDS_DOWNLOAD."""
    tokens: set[str] = set()
    for d in (
        Path("/opt/datasets"),
        Path.home() / ".cache" / "datasets",
        run_dir / "workspace" / "data",
        run_dir / "data",
    ):
        try:
            if d.is_dir():
                for child in d.iterdir():
                    tokens.add(child.name.lower())
        except OSError:
            continue
    return tokens


# P3: max automatic redesign passes when the plan is infeasible on the hardware.
_MAX_ENV_REDESIGN = 2


def _resolve_environment_for_plan(plan: Any, run_dir: Path, config: RCConfig):
    """Parse + resolve the plan's environment manifest against the real runtime.

    Pure (no disk writes). Returns an EnvironmentResolution, or None on error.
    """
    try:
        from researchclaw.pipeline.stage_impls._code_generation import _probe_packages

        manifest = parse_manifest(plan if isinstance(plan, dict) else {})
        installed: dict[str, bool] | None = None
        if manifest.import_names:
            installed = _probe_packages(
                config.experiment.sandbox.python_path,
                candidates=manifest.import_names,
            )
        return resolve_environment(
            manifest,
            installed or {},
            hw_profile=_load_hardware_profile(run_dir),
            cached_datasets=_scan_cached_datasets(run_dir),
        )
    except Exception:  # noqa: BLE001
        logger.debug("Stage 9 environment resolution failed", exc_info=True)
        return None


def _redesign_prompt(plan: Any, constraint: str, hw_profile: Any) -> str:
    hw_str = json.dumps(hw_profile) if hw_profile else "unknown"
    return (
        "The experiment plan below is INFEASIBLE on the available hardware.\n"
        f"Hardware: {hw_str}\n"
        f"Problem: {constraint}\n\n"
        "Redesign the plan to FIT this hardware while keeping the SAME research "
        "goal and hypotheses. Levers: use a smaller open-weight model, QLoRA/4-bit "
        "quantization, gradient checkpointing, smaller batch/sequence length, fewer "
        "parameters, or drop the single component that cannot fit. Do NOT assume "
        "bigger GPUs or more VRAM than listed above.\n"
        "Re-emit the COMPLETE YAML plan (ALL keys, including the `environment` block "
        "with an updated `compute` section reflecting the smaller design). "
        "Return ONLY the YAML.\n\n"
        f"Current plan:\n```yaml\n"
        f"{yaml.dump(plan, default_flow_style=False, allow_unicode=True)}\n```"
    )


def _feasibility_redesign(
    plan: Any, resolution, run_dir: Path, config: RCConfig, llm: LLMClient | None,
    *, max_attempts: int = _MAX_ENV_REDESIGN,
):
    """P3: if the plan is infeasible on this hardware, re-prompt the design LLM
    with the constraint and regenerate, up to *max_attempts* times.

    Returns (plan, resolution, attempts). Stops as soon as the plan becomes
    provisionable, or after max_attempts (leaving an infeasible plan for the
    human gate to handle — never silently proceeds as if feasible).
    """
    attempts = 0
    while (
        resolution is not None
        and resolution.declared
        and not resolution.provisionable
        and llm is not None
        and attempts < max_attempts
    ):
        attempts += 1
        hw = _load_hardware_profile(run_dir)
        constraint = resolution.compute_detail or "the plan exceeds available hardware"
        logger.warning(
            "Stage 9 INFEASIBLE (redesign %d/%d): %s",
            attempts, max_attempts, constraint,
        )
        try:
            resp = llm.chat(
                [{"role": "user", "content": _redesign_prompt(plan, constraint, hw)}],
                max_tokens=8000,
            )
            parsed = _unwrap_plan_dict(yaml.safe_load(_extract_yaml_block(resp.content)))
            if isinstance(parsed, dict):
                plan = parsed
            else:
                logger.warning("Stage 9 redesign produced unparseable plan; stopping")
                break
        except Exception:  # noqa: BLE001
            logger.debug("Stage 9 redesign attempt failed", exc_info=True)
            break
        resolution = _resolve_environment_for_plan(plan, run_dir, config)
    return plan, resolution, attempts


def _write_environment_resolution(stage_dir: Path, resolution, redesign_attempts: int) -> bool:
    """Persist the resolution artifacts. Returns True on success."""
    if resolution is None:
        return False
    try:
        data = resolution.to_dict()
        data["redesign_attempts"] = redesign_attempts
        (stage_dir / "environment_resolution.json").write_text(
            json.dumps(data, indent=2), encoding="utf-8"
        )
        (stage_dir / "environment_resolution.md").write_text(
            resolution.summary_md(), encoding="utf-8"
        )
        if resolution.declared and not resolution.provisionable:
            # Prominent, gate-visible marker — do NOT proceed as if feasible.
            (stage_dir / "INFEASIBLE.md").write_text(
                "# ⛔ Experiment plan INFEASIBLE on this hardware\n\n"
                f"After {redesign_attempts} automatic redesign attempt(s) the plan "
                f"still does not fit:\n\n> {resolution.compute_detail}\n\n"
                "Human action required at the gate: provide bigger hardware, relax "
                "the requirement, or approve a manually-scoped-down plan.\n\n"
                + resolution.summary_md(),
                encoding="utf-8",
            )
            logger.warning(
                "Stage 9 STILL INFEASIBLE after %d redesign(s): %s",
                redesign_attempts, resolution.compute_detail,
            )
        return True
    except OSError:  # noqa: BLE001
        logger.debug("Stage 9 environment resolution write failed", exc_info=True)
        return False

# Best-of-N exploration stances for the Stage 9 design tournament. Each candidate
# plan is generated with one stance appended, giving breadth even with a single
# generator model. Domain-agnostic.
_DESIGN_ANGLES = (
    "Be ambitious: prioritize high-ceiling, novel methods that could yield a "
    "strong result, accepting higher risk.",
    "Be robust: prioritize strong, well-known baselines and a clean, defensible "
    "comparison over novelty.",
    "Be compute-efficient: design the most decisive experiment that fits a tight "
    "compute budget — minimal but conclusive.",
)


def _normalize_plan_field(value: Any) -> list:
    """Normalize a plan field (baselines, proposed_methods, ablations, datasets)
    from any shape the LLM might produce into a flat list of items.

    Handles: list[str], list[dict], dict[str, Any], str, None.
    When the input is a dict, we preserve the full structure by converting each
    key-value pair into a dict item (with at least a 'name' key), rather than
    discarding either keys or values.
    """
    if value is None:
        return []
    if isinstance(value, str):
        return [value] if value.strip() else []
    if isinstance(value, dict):
        result = []
        for k, v in value.items():
            if isinstance(v, dict):
                # e.g. {"baseline_1": {"params": ...}} -> {"name": "baseline_1", "params": ...}
                item = dict(v)
                item.setdefault("name", str(k))
                result.append(item)
            else:
                # e.g. {"baseline_1": "description"} -> {"name": "baseline_1", "description": str(v)}
                result.append({"name": str(k), "description": str(v) if v else ""})
        return result
    if isinstance(value, list):
        return list(value)
    return [value]


def _plan_field_names(items: list) -> list[str]:
    """Extract string names from a normalized plan field for display/dedup."""
    result = []
    for item in items:
        if isinstance(item, dict):
            result.append(item.get("name", str(item)))
        else:
            result.append(str(item))
    return result


_PLAN_KEYS = frozenset({
    "objectives", "datasets", "baselines", "proposed_methods",
    "ablations", "metrics", "risks", "compute_budget",
})


def _unwrap_plan_dict(parsed: Any) -> Any:
    """Unwrap a plan nested under a single parent key.

    Models sometimes emit ``{experiment_plan: {objectives: ..., ...}}`` instead
    of the keys at top level. If the parsed dict has none of the expected plan
    keys at top level but a single dict child that does, return the child.
    """
    if not isinstance(parsed, dict) or _PLAN_KEYS & parsed.keys():
        return parsed
    if len(parsed) == 1:
        only = next(iter(parsed.values()))
        if isinstance(only, dict) and (_PLAN_KEYS & only.keys()):
            return only
    return parsed


def _execute_experiment_design(
    stage_dir: Path,
    run_dir: Path,
    config: RCConfig,
    adapters: AdapterBundle,
    *,
    llm: LLMClient | None = None,
    prompts: PromptManager | None = None,
) -> StageResult:
    hypotheses = _read_prior_artifact(run_dir, "hypotheses.md") or ""
    preamble = _build_context_preamble(
        config, run_dir, include_goal=True, include_hypotheses=True
    )
    plan: dict[str, Any] | None = None

    # ── Domain detection ──────────────────────────────────────────────────
    # Detect the research domain early so we can adapt experiment design
    # and code generation. For ML domains, existing behavior is unchanged.
    _domain_profile = None
    try:
        from researchclaw.domains.detector import detect_domain as _detect_domain_adv
        _domain_profile = _detect_domain_adv(
            topic=config.research.topic,
            hypotheses=hypotheses,
            configured_domains=config.research.domains,
        )
        logger.info(
            "Domain detected: %s (%s)",
            _domain_profile.display_name,
            _domain_profile.domain_id,
        )
        # Persist domain profile for Stage 10
        import json as _json_dd
        (stage_dir / "domain_profile.json").write_text(
            _json_dd.dumps({
                "domain_id": _domain_profile.domain_id,
                "display_name": _domain_profile.display_name,
                "experiment_paradigm": _domain_profile.experiment_paradigm,
                "core_libraries": _domain_profile.core_libraries,
                "gpu_required": _domain_profile.gpu_required,
            }, indent=2),
            encoding="utf-8",
        )
    except Exception:  # noqa: BLE001
        logger.debug("Domain detection unavailable", exc_info=True)

    # --- Domain-specific experiment design context (YAML-driven overlay) ---
    # For ML and HEP, the active prompt bank is already domain-native so we
    # leave this empty. For other profiles (biology, physics, economics, …)
    # the GenericPromptAdapter injects YAML-defined guidance here.
    _domain_design_context = ""
    if _domain_profile is not None:
        try:
            from researchclaw.domains.prompt_adapter import get_adapter as _get_prompt_adapter
            _adapter = _get_prompt_adapter(_domain_profile)
            _design_blocks = _adapter.get_experiment_design_blocks(
                {"topic": config.research.topic}
            )
            if _design_blocks.experiment_design_context:
                _domain_design_context = (
                    "## Domain-Specific Experiment Guidelines\n"
                    + _design_blocks.experiment_design_context
                    + "\n\n"
                )
                if _design_blocks.statistical_test_guidance:
                    _domain_design_context += (
                        "## Statistical Analysis Guidance\n"
                        + _design_blocks.statistical_test_guidance + "\n\n"
                    )
                logger.info(
                    "ExperimentDesign: injecting YAML-driven domain context for %s",
                    _domain_profile.domain_id,
                )
        except Exception:  # noqa: BLE001
            logger.debug("Domain experiment design context unavailable", exc_info=True)

    if llm is not None:
        _pm = prompts or PromptManager()
        # Pass dataset_guidance block for experiment design
        try:
            _dg_block = _pm.block("dataset_guidance")
        except (KeyError, Exception):  # noqa: BLE001
            _dg_block = ""
        # I-08: Inject RL step guidance for RL topics
        _rl_kws = ("reinforcement learning", "ppo", "sac", "td3", "ddpg",
                    "dqn", "mujoco", "continuous control", "actor-critic",
                    "policy gradient", "exploration bonus")
        _is_rl_topic = any(kw in config.research.topic.lower() for kw in _rl_kws)
        if _is_rl_topic:
            try:
                _dg_block += _pm.block("rl_step_guidance")
            except Exception:  # noqa: BLE001
                pass
            # Improvement G: For RL with short budget, constrain to classic control
            if config.experiment.time_budget_sec <= 3600:
                _dg_block += (
                    "\n\n## RL TIME CONSTRAINT (MANDATORY):\n"
                    f"Your time budget is {config.experiment.time_budget_sec}s (≤ 3600s).\n"
                    "You MUST use ONLY classic control environments: "
                    "CartPole-v1, Pendulum-v1, MountainCar-v0, Acrobot-v1, LunarLander-v3.\n"
                    "Do NOT use MuJoCo (HalfCheetah, Hopper, Walker2d, Ant, Humanoid) — "
                    "they require >5000s for meaningful training.\n"
                )
            if config.experiment.time_budget_sec <= 1800:
                _dg_block += (
                    "Time budget ≤ 1800s: use ONLY CartPole-v1 or Pendulum-v1 "
                    "(the simplest environments).\n"
                )
        # F-01: Inject framework docs for experiment design
        try:
            from researchclaw.data import detect_frameworks, load_framework_docs
            _fw_ids = detect_frameworks(config.research.topic, hypotheses)
            if _fw_ids:
                _fw_docs = load_framework_docs(_fw_ids, max_chars=4000)
                if _fw_docs:
                    _dg_block += _fw_docs
        except Exception:  # noqa: BLE001
            pass
        # Improvement A: Compute hardware profile + per-condition budget.
        # Use the real detected profile (stage-01/hardware_profile.json) instead
        # of a hardcoded placeholder, so the design respects actual GPU count/VRAM.
        _hw = _load_hardware_profile(run_dir)
        if _hw and _hw.get("has_gpu"):
            _hw_name = _hw.get("gpu_name", "GPU")
            _hw_vram = _hw.get("vram_mb")
            _hw_count = int(_hw.get("gpu_count", 1) or 1)
            _hw_total = _hw.get("total_vram_mb")
            _vram_str = f"{_hw_vram} MB VRAM per card" if _hw_vram else "VRAM unknown"
            _total_str = f", {_hw_total} MB total" if _hw_total and _hw_count > 1 else ""
            _hw_profile_str = (
                f"- GPU: {_hw_name} ({_vram_str}{_total_str})\n"
                f"- GPU count: {_hw_count}\n"
                "- CPU: shared server"
            )
        else:
            _hw_profile_str = (
                "- GPU: none detected (CPU only)\n"
                "- GPU count: 0\n"
                "- CPU: shared server"
            )
        _per_condition_sec = int(config.experiment.time_budget_sec * 0.7 / 6)
        _tier1 = "CIFAR-10, CIFAR-100, MNIST, FashionMNIST, STL-10, SVHN"

        _overlay = _get_evolution_overlay(run_dir, "experiment_design")
        sp = _pm.for_stage(
            "experiment_design",
            evolution_overlay=_overlay,
            preamble=preamble,
            hypotheses=hypotheses,
            dataset_guidance=_dg_block,
            domain_design_context=_domain_design_context,
            time_budget_sec=config.experiment.time_budget_sec,
            metric_key=config.experiment.metric_key,
            metric_direction=config.experiment.metric_direction,
            hardware_profile=_hw_profile_str,
            per_condition_budget_sec=_per_condition_sec,
            available_tier1_datasets=_tier1,
        )
        if config.llm.tournament_enabled and config.llm.tournament_candidates >= 2:
            # --- Best-of-N tournament: generate N candidate plans from diverse
            # stances, then an independent judge picks the winner. Downstream YAML
            # parsing / normalization / caps / BenchmarkAgent run on the winner.
            from researchclaw.llm import build_panel_llms, build_reviewer_llm
            from researchclaw.pipeline.tournament import (
                effective_candidates,
                run_tournament,
            )

            _gens = build_panel_llms(config) or [llm]
            _judge = build_reviewer_llm(config) or llm
            _n = effective_candidates(config.llm.tournament_candidates)
            _cps = [
                (
                    sp.system,
                    sp.user
                    + "\n\n## Exploration stance\n"
                    + _DESIGN_ANGLES[i % len(_DESIGN_ANGLES)],
                )
                for i in range(_n)
            ]
            _winner, _ = run_tournament(
                _gens,
                _judge,
                _cps,
                rank_prompt="tournament_rank",
                out_dir=stage_dir / "tournament",
                prompts=_pm,
                author_model=getattr(llm.config, "primary_model", ""),
                label="plan",
            )
            resp = SimpleNamespace(content=_winner)
        else:
            resp = _chat_with_prompt(
                llm,
                sp.system,
                sp.user,
                json_mode=sp.json_mode,
                max_tokens=sp.max_tokens,
            )
        raw_yaml = _extract_yaml_block(resp.content)
        try:
            parsed = yaml.safe_load(raw_yaml)
        except yaml.YAMLError:
            parsed = None
        # Fallback: reasoning models sometimes emit the YAML without fences
        # or wrapped in prose. Try parsing the whole response as YAML.
        if not isinstance(parsed, dict):
            try:
                parsed = yaml.safe_load(resp.content)
            except yaml.YAMLError:
                pass
        # Last fallback: try to find any YAML-like dict in the response
        if not isinstance(parsed, dict):
            import re as _re_yaml

            # Look for lines starting with known keys
            _yaml_lines = []
            _capturing = False
            for line in resp.content.splitlines():
                if _re_yaml.match(
                    r"^(baselines|proposed_methods|ablations|datasets|"
                    r"metrics|objectives|risks|compute_budget)\s*:",
                    line,
                ):
                    _capturing = True
                if _capturing:
                    if line.strip() == "" or line.startswith("```"):
                        continue
                    if line.startswith("#") or line.startswith("**"):
                        continue
                    _yaml_lines.append(line)
            if _yaml_lines:
                try:
                    parsed = yaml.safe_load("\n".join(_yaml_lines))
                except yaml.YAMLError:
                    pass
        if isinstance(parsed, dict):
            plan = _unwrap_plan_dict(parsed)
        else:
            logger.warning(
                "Stage 09: LLM response could not be parsed as YAML "
                "(len=%d, first 200 chars: %s). Content extraction method "
                "returned: %s",
                len(resp.content),
                resp.content[:200],
                raw_yaml[:200] if raw_yaml else "<empty>",
            )
            # BUG-12: Retry with a stricter, shorter prompt
            if llm is not None:
                logger.info("Stage 09: Retrying with strict YAML-only prompt...")
                # Keep the retry GROUNDED: a context-free retry produces generic
                # plans that ignore the hardware and hypotheses (e.g. 128xA100,
                # 34B models, GPT-4 baselines). Re-inject the real constraints.
                _retry_prompt = (
                    "Output ONLY valid YAML. No prose, no markdown fences, no explanation.\n"
                    f"Topic: {config.research.topic}\n"
                    "Required keys: baselines, proposed_methods, ablations, "
                    "datasets, metrics, objectives, risks, compute_budget.\n"
                    "Each key maps to a SHORT list of one-line strings (<=7 each).\n\n"
                    "HARD CONSTRAINTS:\n"
                    f"- Hardware (use ONLY this): {_hw_profile_str}\n"
                    "- compute_budget MUST be in GPU-hours on the hardware above; "
                    "NO A100/H100 clusters or cloud dollar budgets.\n"
                    "- Open-weight models ONLY (no GPT-4 / proprietary APIs).\n"
                    "- Metrics must be automatically computable in code (no human studies).\n"
                    "- baselines/proposed_methods/ablations MUST derive from the "
                    "hypotheses below.\n\n"
                    f"Hypotheses:\n{hypotheses[:4000]}"
                )
                _retry_resp = _chat_with_prompt(
                    llm,
                    "You output ONLY valid YAML. Nothing else.",
                    _retry_prompt,
                    max_tokens=8000,
                )
                try:
                    _retry_parsed = yaml.safe_load(_retry_resp.content)
                    if isinstance(_retry_parsed, dict):
                        plan = _unwrap_plan_dict(_retry_parsed)
                        logger.info("Stage 09: Strict YAML retry succeeded.")
                except yaml.YAMLError:
                    pass

    # BUG-12: Fallback 4 — extract method/baseline names from Stage 8 hypotheses
    if plan is None:
        _hyp_text = _read_prior_artifact(run_dir, "hypotheses.md") or ""
        if _hyp_text:
            import re as _re_hyp
            # Extract method-like names from hypothesis text
            _method_candidates = _re_hyp.findall(
                r"(?:proposed|our|novel|new)\s+(?:method|approach|algorithm|framework|model)[:\s]+[\"']?([A-Za-z][\w-]+)",
                _hyp_text, _re_hyp.IGNORECASE,
            )
            _baseline_candidates = _re_hyp.findall(
                r"(?:baseline|compare|existing|standard|traditional)\s+(?:method|approach|model)?[:\s]+[\"']?([A-Za-z][\w-]+)",
                _hyp_text, _re_hyp.IGNORECASE,
            )
            if _method_candidates or _baseline_candidates:
                logger.info(
                    "Stage 09: Extracted names from hypotheses: methods=%s, baselines=%s",
                    _method_candidates[:3], _baseline_candidates[:3],
                )
                plan = {
                    "topic": config.research.topic,
                    "generated": _utcnow_iso(),
                    "objectives": ["Evaluate hypotheses with controlled experiments"],
                    "datasets": ["primary_dataset"],
                    "baselines": _baseline_candidates[:3] or ["baseline_1", "baseline_2"],
                    "proposed_methods": _method_candidates[:3] or ["proposed_method"],
                    "ablations": ["without_key_component", "simplified_version"],
                    "metrics": [config.experiment.metric_key, "secondary_metric"],
                    "risks": ["validity threats", "confounding variables"],
                    "compute_budget": {"max_gpu": 1, "max_hours": 4},
                }

    if plan is None:
        # BUG-12: Use domain-aware names instead of fully generic placeholders
        _topic_prefix = config.research.topic.split()[0] if config.research.topic else "method"
        logger.warning(
            "Stage 09: LLM failed to produce valid experiment plan YAML. "
            "Using topic-derived fallback."
        )
        plan = {
            "topic": config.research.topic,
            "generated": _utcnow_iso(),
            "objectives": ["Evaluate hypotheses with controlled experiments"],
            "datasets": ["primary_dataset", "secondary_dataset"],
            "baselines": [f"{_topic_prefix}_baseline_1", f"{_topic_prefix}_baseline_2"],
            "proposed_methods": [f"{_topic_prefix}_proposed", f"{_topic_prefix}_variant"],
            "ablations": ["without_key_component", "simplified_version"],
            "metrics": [config.experiment.metric_key, "secondary_metric"],
            "risks": ["validity threats", "confounding variables"],
            "compute_budget": {"max_gpu": 1, "max_hours": 4},
        }
    # ── BA: BenchmarkAgent — intelligent dataset/baseline selection ──────
    _benchmark_plan = None
    # BUG-40: Skip BenchmarkAgent for non-ML domains — it has no relevant
    # benchmarks for physics/chemistry/mathematics/etc. and would inject
    # wrong datasets (e.g., CIFAR-10 for PDE topics).
    _ba_domain_profile = _domain_profile
    if _ba_domain_profile is None:
        try:
            from researchclaw.domains.detector import detect_domain as _detect_domain_adv
            _ba_domain_profile = _detect_domain_adv(
                topic=config.research.topic,
                hypotheses=hypotheses,
                configured_domains=config.research.domains,
            )
        except Exception:  # noqa: BLE001
            logger.debug("BenchmarkAgent domain detection unavailable", exc_info=True)
    _ba_domain_id = (
        _ba_domain_profile.domain_id
        if _ba_domain_profile is not None
        else "generic"
    )
    _ba_domain_ok = _ba_domain_id.startswith("ml_")
    if not _ba_domain_ok:
        logger.info(
            "BenchmarkAgent skipped: domain profile '%s' is not an ML profile (topic: %s)",
            _ba_domain_id, config.research.topic[:80],
        )
    if (
        _ba_domain_ok
        and config.experiment.benchmark_agent.enabled
        and config.experiment.mode in ("sandbox", "docker")
        and llm is not None
    ):
        try:
            from researchclaw.agents.benchmark_agent import BenchmarkOrchestrator
            from researchclaw.agents.benchmark_agent.orchestrator import (
                BenchmarkAgentConfig as _BACfg,
            )

            _ba_cfg_raw = config.experiment.benchmark_agent
            _ba_cfg = _BACfg(
                enabled=_ba_cfg_raw.enabled,
                enable_hf_search=_ba_cfg_raw.enable_hf_search,
                max_hf_results=_ba_cfg_raw.max_hf_results,
                enable_web_search=_ba_cfg_raw.enable_web_search,
                max_web_results=_ba_cfg_raw.max_web_results,
                web_search_min_local=_ba_cfg_raw.web_search_min_local,
                tier_limit=_ba_cfg_raw.tier_limit,
                min_benchmarks=_ba_cfg_raw.min_benchmarks,
                min_baselines=_ba_cfg_raw.min_baselines,
                prefer_cached=_ba_cfg_raw.prefer_cached,
                max_iterations=_ba_cfg_raw.max_iterations,
            )

            _hw = _load_hardware_profile(run_dir)
            _ba = BenchmarkOrchestrator(
                llm,
                config=_ba_cfg,
                gpu_memory_mb=(
                    _hw.get("vram_mb") or _hw.get("gpu_memory_mb") or 49000
                    if _hw else 49000
                ),
                time_budget_sec=config.experiment.time_budget_sec,
                network_policy=(
                    config.experiment.docker.network_policy
                    if config.experiment.mode == "docker"
                    else "full"
                ),
                stage_dir=stage_dir / "benchmark_agent",
            )
            _benchmark_plan = _ba.orchestrate({
                "topic": config.research.topic,
                "hypothesis": hypotheses,
                "experiment_plan": plan.get("objectives", "") if isinstance(plan, dict) else "",
            })

            # Inject BenchmarkAgent selections into experiment plan
            if isinstance(plan, dict) and _benchmark_plan.selected_benchmarks:
                plan["datasets"] = [
                    b.get("name", "Unknown") for b in _benchmark_plan.selected_benchmarks
                ]
                # Normalize existing baselines — LLM may emit dict, list of
                # dicts, or list of strings.
                _baselines_from_plan = _plan_field_names(
                    _normalize_plan_field(plan.get("baselines", []))
                )
                plan["baselines"] = [
                    bl.get("name", "Unknown") for bl in _benchmark_plan.selected_baselines
                ] + _baselines_from_plan
                # Deduplicate baselines
                plan["baselines"] = list(dict.fromkeys(plan["baselines"]))

            logger.info(
                "BenchmarkAgent: %d benchmarks, %d baselines selected (%d LLM calls, %.1fs)",
                len(_benchmark_plan.selected_benchmarks),
                len(_benchmark_plan.selected_baselines),
                _benchmark_plan.total_llm_calls,
                _benchmark_plan.elapsed_sec,
            )
        except Exception as _ba_exc:
            logger.warning("BenchmarkAgent failed (non-fatal): %s", _ba_exc)

    # Save benchmark plan for code_generation stage
    if _benchmark_plan is not None:
        try:
            (stage_dir / "benchmark_plan.json").write_text(
                json.dumps(_benchmark_plan.to_dict(), indent=2, ensure_ascii=False),
                encoding="utf-8",
            )
        except Exception:  # noqa: BLE001
            pass

    plan.setdefault("topic", config.research.topic)

    # BUG-R41-09: Enforce condition count limit based on time budget.
    # Too many conditions (30+) guarantee timeouts and wasted compute.
    _time_budget = getattr(
        getattr(config, "experiment", None), "time_budget_sec", 3600
    )
    _max_conditions = 8  # default for budgets ≤ 3600s
    if _time_budget > 3600:
        _max_conditions = 12
    if _time_budget > 7200:
        _max_conditions = 20

    _baselines = _normalize_plan_field(plan.get("baselines", []))
    _proposed = _normalize_plan_field(plan.get("proposed_methods", []))
    _ablations = _normalize_plan_field(plan.get("ablations", []))
    _total = len(_baselines) + len(_proposed) + len(_ablations)

    if _total > _max_conditions:
        logger.warning(
            "Stage 9: Plan has %d conditions (limit %d for %ds budget). "
            "Trimming to fit.",
            _total, _max_conditions, _time_budget,
        )
        # Keep all proposed methods (up to max), trim baselines and ablations
        _proposed_count = min(len(_proposed), max(1, _max_conditions - 4))
        _remaining = max(0, _max_conditions - _proposed_count)
        _baseline_budget = max(1, _remaining // 2)
        _ablation_budget = max(0, _remaining - _baseline_budget)
        if len(_proposed) > _proposed_count:
            plan["proposed_methods"] = _proposed[:_proposed_count]
            logger.info(
                "Stage 9: Trimmed proposed methods %d → %d",
                len(_proposed), _proposed_count,
            )

        if len(_baselines) > _baseline_budget:
            plan["baselines"] = _baselines[:_baseline_budget]
            logger.info(
                "Stage 9: Trimmed baselines %d → %d",
                len(_baselines), _baseline_budget,
            )
        if len(_ablations) > _ablation_budget:
            plan["ablations"] = _ablations[:_ablation_budget]
            logger.info(
                "Stage 9: Trimmed ablations %d → %d",
                len(_ablations), _ablation_budget,
            )

    # --- HITL: Read human guidance if available ---
    guidance_file = stage_dir / "hitl_guidance.md"
    if guidance_file.exists():
        try:
            guidance = guidance_file.read_text(encoding="utf-8").strip()
            if guidance and llm is not None and isinstance(plan, dict):
                logger.info("Applying HITL guidance to experiment design")
                resp = llm.chat(
                    [{"role": "user", "content": (
                        f"The human researcher provided this guidance for "
                        f"the experiment design:\n\n{guidance}\n\n"
                        f"Current experiment plan:\n"
                        f"```yaml\n{yaml.dump(plan, default_flow_style=False)}\n```\n\n"
                        f"Update the YAML plan to incorporate the guidance. "
                        f"Return ONLY the updated YAML."
                    )}],
                    max_tokens=4096,
                )
                updated = _extract_yaml_block(resp.content)
                try:
                    parsed_update = yaml.safe_load(updated)
                    if isinstance(parsed_update, dict):
                        plan = parsed_update
                except yaml.YAMLError:
                    pass
        except Exception:
            logger.debug("HITL guidance application failed (non-blocking)")

    # --- HITL: Baseline Navigator data persistence ---
    try:
        from researchclaw.hitl.workshops.baseline import BaselineNavigator, BaselineCandidate

        nav = BaselineNavigator(run_dir, llm_client=llm)
        if isinstance(plan, dict):
            baselines = plan.get("baselines", [])
            if isinstance(baselines, list):
                for b in baselines:
                    if isinstance(b, dict):
                        nav.baselines.append(BaselineCandidate(
                            name=b.get("name", str(b)),
                            description=b.get("description", ""),
                        ))
                    elif isinstance(b, str):
                        nav.baselines.append(BaselineCandidate(name=b))
            metrics = plan.get("metrics", [])
            if isinstance(metrics, list):
                nav.metrics = [str(m) for m in metrics]
        nav.save()
    except Exception:
        pass

    # P1+P3: resolve the environment manifest against the real runtime; if the
    # plan is infeasible on this hardware, redesign it (bounded) BEFORE writing,
    # so feasibility is visible at the gate and codegen never gets a plan it
    # cannot run (which is what forces synthetic/placeholder code).
    _resolution = _resolve_environment_for_plan(plan, run_dir, config)
    plan, _resolution, _redesign_n = _feasibility_redesign(
        plan, _resolution, run_dir, config, llm
    )

    (stage_dir / "exp_plan.yaml").write_text(
        yaml.dump(plan, default_flow_style=False, allow_unicode=True),
        encoding="utf-8",
    )
    _env_resolved = _write_environment_resolution(stage_dir, _resolution, _redesign_n)

    _artifacts = ("exp_plan.yaml",)
    _evidence = ("stage-09/exp_plan.yaml",)
    if _env_resolved:
        _artifacts = _artifacts + ("environment_resolution.json",)
        _evidence = _evidence + ("stage-09/environment_resolution.json",)
    return StageResult(
        stage=Stage.EXPERIMENT_DESIGN,
        status=StageStatus.DONE,
        artifacts=_artifacts,
        evidence_refs=_evidence,
    )
