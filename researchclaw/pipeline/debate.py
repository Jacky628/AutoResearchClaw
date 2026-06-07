"""Multi-model debate engine (Stage 8 / 14 / 18).

Upgrades the project's shallow "multi-perspective one-shot" pattern into a real
debate:

  1. multiple *models* each play a distinct role (round-robin over the panel),
  2. an optional rebuttal round where every role sees the others' prior turn and
     pushes back, and
  3. an independent judge that scores, ranks, and synthesizes the final output.

Opt-in: callers only route here when ``build_panel_llms(config)`` returns a
non-empty panel (i.e. ``llm.debate_enabled`` is set). When the panel has a single
model this degrades to single-model multi-role, matching the legacy behaviour
plus a judging step.

The engine is provider-agnostic and offline-testable: it only calls
``client.chat(messages, *, system=...)`` on whatever clients it is handed.
"""

from __future__ import annotations

import json
import logging
import os
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)


def _model_name(client: Any) -> str:
    return getattr(getattr(client, "config", None), "primary_model", "") or "unknown"


def _resolve_rounds(config_rounds: int) -> int:
    """Effective rebuttal rounds, honoring the ARC_DEBATE_ROUNDS override."""
    env = os.environ.get("ARC_DEBATE_ROUNDS", "").strip()
    if env:
        try:
            return max(0, int(env))
        except ValueError:
            pass
    return max(0, int(config_rounds))


def run_debate(
    panel: list,
    judge: Any,
    roles: dict[str, dict[str, str]],
    variables: dict[str, str],
    *,
    rounds: int,
    synth_prompt: str,
    out_dir: Path,
    prompts: Any,
    author_model: str = "",
    gen_max_tokens: int = 8192,
) -> tuple[str, dict]:
    """Run a multi-model, multi-round debate and judge it into a final text.

    Args:
        panel: list of LLM clients (each exposes ``.chat`` and ``.config``).
            Empty list is not allowed — callers must pass at least one client.
        judge: independent judge client (scores + synthesizes). Falls back to
            ``panel[0]`` if None.
        roles: ``{role_name: {"system": ..., "user": ...}}`` (domain bank).
        variables: template variables for ``_render``.
        rounds: number of rebuttal rounds after the opening statements.
        synth_prompt: name of the judge/synthesis sub-prompt (e.g.
            ``"hypothesis_synthesize"``).
        out_dir: directory for per-role / per-round transcripts + record.
        prompts: PromptManager (for ``sub_prompt`` rendering).
        author_model: generator model name, for provenance.

    Returns:
        ``(final_text, record_dict)``. ``final_text`` is the judge's synthesis;
        ``record_dict`` is also written to ``out_dir/debate_record.json``.
    """
    from researchclaw.prompts import _render  # local import: avoid cycles

    if not panel:
        raise ValueError("run_debate requires a non-empty panel")

    rounds = _resolve_rounds(rounds)
    out_dir.mkdir(parents=True, exist_ok=True)

    # Ablation hook (shared with the legacy path): collapse to a single role.
    if os.environ.get("ARC_ABL_DISABLE_DEBATE", "").strip() == "1" and roles:
        first = next(iter(roles))
        roles = {first: roles[first]}
        logger.info("ARC_ABL_DISABLE_DEBATE=1 — debate collapsed to role %s", first)

    role_names = list(roles.keys())
    # Bind each role to a panel model round-robin so distinct models argue.
    role_model: dict[str, Any] = {
        name: panel[i % len(panel)] for i, name in enumerate(role_names)
    }

    # --- Opening statements (round 0) ---
    current: dict[str, str] = {}
    for name in role_names:
        rp = roles[name]
        try:
            system = _render(rp["system"], variables)
            user = _render(rp["user"], variables)
            resp = role_model[name].chat(
                [{"role": "user", "content": user}],
                system=system,
                max_tokens=gen_max_tokens,
            )
            text = resp.content or ""
            if not text.strip():
                # Empty output (e.g. a reasoning model starving the answer at a
                # low token budget) — drop this role rather than feed a blank
                # opening statement into the rebuttal/synthesis.
                logger.warning(
                    "Debate r0 role=%s model=%s returned empty — dropped",
                    name, _model_name(role_model[name]),
                )
                continue
            current[name] = text
            (out_dir / f"{name}.r0.md").write_text(text, encoding="utf-8")
            logger.info(
                "Debate r0 role=%s model=%s (%d chars)",
                name, _model_name(role_model[name]), len(text),
            )
        except Exception as exc:  # noqa: BLE001
            logger.warning("Debate r0 role=%s failed: %s", name, exc)

    # --- Rebuttal rounds (each role sees the others' latest turn) ---
    for r in range(1, rounds + 1):
        prev = dict(current)
        if len(prev) < 2:
            break  # nothing to push back against
        for name in role_names:
            if name not in prev:
                continue
            others = "\n\n---\n\n".join(
                f"### {other}\n{prev[other]}" for other in prev if other != name
            )
            try:
                sp = prompts.sub_prompt(
                    "debate_rebuttal",
                    role=name,
                    own_position=prev.get(name, ""),
                    others=others,
                )
                resp = role_model[name].chat(
                    [{"role": "user", "content": sp.user}],
                    system=sp.system,
                    max_tokens=gen_max_tokens,
                )
                text = resp.content or ""
                if not text.strip():
                    # Keep the prior round's non-empty position rather than
                    # overwriting it with a blank rebuttal.
                    logger.warning(
                        "Debate r%d role=%s returned empty — keeping prior turn",
                        r, name,
                    )
                    continue
                current[name] = text
                (out_dir / f"{name}.r{r}.md").write_text(
                    text, encoding="utf-8"
                )
            except Exception as exc:  # noqa: BLE001
                logger.warning("Debate r%d role=%s failed: %s", r, name, exc)

    if not current:
        raise RuntimeError("debate produced no perspectives")

    # --- Judge: score + rank + synthesize ---
    judge_client = judge or panel[0]
    judge_model = _model_name(judge_client)
    parts = [f"### Perspective: {name}\n{text}" for name, text in current.items()]
    combined = "\n\n---\n\n".join(parts)
    sp = prompts.sub_prompt(synth_prompt, perspectives=combined)
    judge_system = (
        "You are an INDEPENDENT judge, distinct from the debating models. "
        "First score each perspective 1-10 for rigor and evidence, rank them, "
        "then synthesize — take the strongest elements and preserve genuine "
        "disagreements.\n\n"
    ) + sp.system
    final_text = ""
    try:
        kwargs: dict[str, Any] = {"system": judge_system}
        if sp.max_tokens:
            kwargs["max_tokens"] = sp.max_tokens
        resp = judge_client.chat([{"role": "user", "content": sp.user}], **kwargs)
        final_text = resp.content
    except Exception as exc:  # noqa: BLE001
        logger.warning("Debate judge failed: %s — falling back to concatenation", exc)
        final_text = combined

    record = {
        "panel_models": [_model_name(c) for c in panel],
        "roles": {name: _model_name(role_model[name]) for name in role_names},
        "rounds": rounds,
        "author_model": author_model,
        "judge_model": judge_model,
        "independent_judge": bool(judge_model and judge_model != author_model),
        "perspectives_succeeded": sorted(current.keys()),
    }
    (out_dir / "debate_record.json").write_text(
        json.dumps(record, indent=2), encoding="utf-8"
    )
    return final_text, record
