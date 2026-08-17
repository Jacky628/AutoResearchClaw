# Major Overhaul 1: Review De-duplication + Multi-Model Debate Engine

> Date: 2026-06-01 (Includes two batches of changes: "Generator-Reviewer Model Separation" from 2026-05-31 and "Multi-Model Debate" from the following day).
> Revision: 2026-06-04 — Corrected the wiring scope of "Multi-Model Debate" to **Stages 14 / 18** (previously misreported as including Stage 8; the actual wiring was committed to disk today, see "Errata" at the start of Batch 2).
> Goal: To upgrade AutoResearchClaw from a circular logic of "the same model generates, reviews, and scores its own work" to a true multi-model adversarial system of "different models playing distinct roles → debating/rebutting → scored and synthesized by an independent judge."
> All changes are **opt-in, with zero behavior change by default.**

---

## Background: Why do this (Review Findings)

During an in-depth review of the project, verifying `file:line` references revealed two fundamental shortcomings:

1. **Generation = Review = Judgment (Homology)**: Stage 18 Peer Review, Stage 19 Revision, and Stage 20 Quality Gate were all performed by **the exact same LLM instance** (`executor.py` constructed only one `llm = LLMClient.from_rc_config(config)` per stage and passed it down). A generator scoring itself leads to systemic self-preference, making the review highly unreliable.
2. **Debate was shallow and fake**:
   - Stage 8 Hypothesis Generation and Stage 14 Result Analysis had "multiple perspectives," but via `_multi_perspective_generate` + `_synthesize_perspectives`—**the same model** playing multiple roles, speaking once each, **invisible to one another**, with a single synthesis step. **There was no clash, no scoring, and no selection.**
   - Stage 18 Peer Review was a "fake debate": `prompts/ml.py` simply stated *"Simulate peer review from 3 reviewers"*—**the same model pretending to be 3 reviewers in a single call.**
   - Stage 15 Research Decision was a single-model, single-call determination.

Benchmarking against industry frontiers (e.g., AI Scientist v2 tree search, Google co-scientist multi-agent tournaments, and MLR-Bench pointing out that coding agents hallucinate results 80% of the time) confirmed: **"Review de-duplication + True multi-model debate" is exactly where the cutting-edge focus lies right now.**

---

# Batch 1: Generator-Reviewer Model Separation (P0-3)

## Core Idea
Ensure that **criticism/judgment** comes from a **different** model than the author: the author generates and revises, while an independent model reviews and acts as the quality gate judge.

## What was changed

### 1. Configuration (`researchclaw/config.py`)
Added 5 fields to `LlmConfig` (default empty = disabled, backward compatible):
```python
reviewer_model: str = ""        # Empty => Reviewer reuses generator (legacy behavior)
reviewer_provider: str = ""     # Defaults to main provider
reviewer_base_url: str = ""     # Defaults to main base_url
reviewer_api_key: str = ""
reviewer_api_key_env: str = ""
```
Two usage patterns: Providing only `reviewer_model` → changes the model on the same endpoint; providing the full set → completely independent provider (e.g., Generator = GPT / Reviewer = Claude).

### 2. Client Factory (`researchclaw/llm/client.py` + `llm/__init__.py`)
- `LLMClient.reviewer_from_rc_config()` — Constructs an independent reviewer client, reusing existing provider-preset / Anthropic-adapter logic, falling back to main config for missing fields. `fallback_models=[]` (**Does NOT fall back to the author model**, preserving independence).
- `build_reviewer_llm(config)` — Convenience wrapper; returns `None` for ACP or unconfigured states (caller falls back to generator).

### 3. Integration into Review Stages (`researchclaw/pipeline/stage_impls/_review_publish.py`)
- **Stage 18 peer_review** and **Stage 20 quality_gate** now use `reviewer_llm` (fallback to generator if not configured).
- **Stage 19 paper_revision still uses the author model**—this is the core of decoupling: the author writes/revises, the independent model critiques/judges.
- Added `_INDEPENDENT_REVIEWER_PREFIX`: Prepended to the system prompt of reviews/judgments declaring "You are an independent reviewer, distinct from the author model, default to finding faults rather than endorsing" (reduces self-preference even if the same model is used).
- **Provenance Tracking**: Added comment headers to `reviews.md`, introduced `review_provenance.json` and `quality_report.json` to record `author_model` / `judge_model` / `independent` for auditing.

### 4. ARC-Bench Evaluation Independence (`experiments/arc_bench/scripts/judge.py`)
- Homology guard: Raises an error if `ARC_JUDGE_MODEL == ARC_AUTHOR_MODEL`, requiring `ARC_ALLOW_SAME_MODEL_JUDGE=1` to explicitly bypass.
- Appended "independent reviewer" declaration to the judge system prompt.

### 5. Documentation and Examples
Added reviewer field explanations and examples to `config.researchclaw.example.yaml` and `CLAUDE.md`.

## Actual Enablement (Your `config.arc.yaml`)
```yaml
llm:
  primary_model: "anthropic/claude-sonnet-4.6"   # Author: Generation + Revision
  reviewer_model: "openai/gpt-4o"                 # Reviewer + Judge (via OpenRouter to OpenAI gpt-4o)
```
Meaning: **Generation/Revision = Claude Sonnet 4.6, Review/Judgment = gpt-4o**, reusing the same `OPENROUTER_API_KEY` without extra credentials.

---

# Batch 2: Multi-Model Debate Engine (Stages 14 / 18)

> **Errata (Revised 2026-06-04)**: This batch was initially recorded as "Wiring for Stages 8 / 14 / 18". However, due to tool echo latency in the previous round, the wiring in `_synthesis.py` (Stage 8) / `_analysis.py` (Stage 14) **was not actually committed to disk**—`run_debate` was dead code at the time, and the three stages still used the legacy single-model multi-perspective approach. This was corrected on 2026-06-04:
> - **Stages 14 / 18 are now genuinely wired to `run_debate`** (see §5 below, with file:line and regression tests).
> - **Stage 8 has switched to a best-of-N tournament** (see separate document "Item 6"), **and does not connect to the debate engine**—thus the scope of this batch is narrowed to Stages 14 / 18.

## Core Idea
Upgrade the "single-model plays multiple roles, speaks once, single synthesis" approach to a unified **"Multi-Model × Multi-Perspective × Multi-Round Clash × Independent Judge"** engine, shared by Stages 14 / 18.

## What was changed

### 1. Configuration (`researchclaw/config.py`) — Reuses existing models, no new model list
```python
debate_enabled: bool = False    # Opt-in master switch, default off = zero behavior change
debate_rounds: int = 1          # Number of clash (rebuttal) rounds; 0 = parallel only + judge
```

### 2. Debate Model Panel (`build_panel_llms` in `researchclaw/llm/__init__.py`)
- If `debate_enabled=False` or using ACP → returns `[]` (takes legacy path).
- Otherwise, constructs a **deduplicated** panel pool from **`primary_model` + `reviewer_model` + `fallback_models`**. Each model is cloned into an independent single-model client (`fallback_models=[]`), sharing the main endpoint and adapter.

### 3. Debate Engine (New file `researchclaw/pipeline/debate.py`, `run_debate()`)
A single entry point shared by multiple stages. Flow:
1. **Role ↔ Model Round-Robin Binding**: `panel[i % len(panel)]`, assigning different models to different stances (wraps around if more roles than models).
2. **First Round Opening**: Independent `chat()` per role, saving to `{role}.r0.md`.
3. **Clash Round(s)** (Repeats `debate_rounds` times): Injects the **full text of all other roles' previous rounds** + the `debate_rebuttal` sub-prompt to each role, asking them to attack weaknesses and strengthen/revise their own stance. Saves to `{role}.r{n}.md`.
4. **Judgment**: An independent judge (reusing `reviewer_model`) **scores (1-10) + ranks + synthesizes** the final stances of all roles, producing the final text + `debate_record.json` (logging panel models, role bindings, rounds, judge independence, etc.).
- Fault tolerance: Skips failed single roles; falls back to legacy synthesis/default artifacts if all fail.
- Retains `ARC_ABL_DISABLE_DEBATE` ablation hook; added `ARC_DEBATE_ROUNDS` env override.

### 4. Rebuttal Prompt (`researchclaw/prompts/shared.py`)
Added `debate_rebuttal` sub-prompt (domain-agnostic): "Here is your previous stance + statements from other perspectives. Please rebut them and revise your own."

### 5. Stage Wiring (Thin changes, preserves executor dispatch signature)
Each stage starts with `panel = build_panel_llms(config)`. If non-empty, it calls `run_debate(...)`; otherwise, it follows the legacy path:
- **Stage 14 Result Analysis** (`_analysis.py:598-636`): roles=`debate_roles_analysis()` (optimist / skeptic / methodologist), synth=`analysis_synthesize`, judge=`build_reviewer_llm`. Outputs `analysis.md` + `stage-14/perspectives/debate_record.json`.
- **Stage 18 Peer Review** (`_review_publish.py:217-240`): Uses inline `_REVIEW_DEBATE_ROLES` (Methodology / Domain / Statistical Rigor), synth=new `review_synthesize` sub-prompt, judge=independent reviewer → Multiple models review independently → Clash → Synthesize to `reviews.md`. Outputs `stage-18/debate/debate_record.json` + preserves `review_provenance.json`.
  **The "fake debate" is hereby upgraded to a true multi-model debate.**
- **Stage 8 Hypothesis Gen**: **Not wired to debate in this batch**—switched to best-of-N tournament (separate document); `_synthesis.py` follows `tournament > legacy` paths without calling `run_debate`.
- Regression tests `tests/test_rc_executor.py::TestDebateWiring` (4 cases) guard the Stage 14/18 wiring to prevent degradation back to dead code.

### 6. Documentation and Examples
Added `debate_enabled` / `debate_rounds` explanations to `config.researchclaw.example.yaml` and `CLAUDE.md`.

## Actual Enablement (Your `config.arc.yaml`)
```yaml
llm:
  primary_model: "anthropic/claude-sonnet-4.6"
  reviewer_model: "openai/gpt-4o"
  fallback_models:
    - "google/gemini-2.5-pro"        # deepseek-chat deleted per your request
  debate_enabled: true
  debate_rounds: 1
```
Debate panel = **claude-sonnet-4.6 + gpt-4o + gemini-2.5-pro** (3 distinct models); Judge = gpt-4o.

---

## Behavior After Running (When Enabled)
For Stages 14 / 18 individually: 3 models play roles → First round opening → 1 round of clash/rebuttal (visible to each other) → gpt-4o judge scores, ranks, and synthesizes.
Each stage outputs `stage-XX/.../debate_record.json` and `{role}.r{n}.md` for each role/round.
(Stage 8 follows the best-of-N tournament, outputting `stage-08/tournament/tournament_record.json`, see separate document.)

---

## Verification Results (All Green)
- Added `tests/test_reviewer_llm.py`: 10 passed
- Added `tests/test_debate_engine.py`: Combined with reviewer, 18 passed (covers panel deduplication, rebuttal visibility, rounds=0, single-model degradation, empty pool error)
- Added `tests/test_rc_executor.py::TestDebateWiring` (2026-06-04): 4 passed (Asserts both paths: Stage 14/18 panel not empty → `run_debate`; panel empty → legacy)
- Full regression (rerun 2026-06-04, including tournament changes): **1275 passed**
- All modified files compile successfully; `researchclaw validate --config config.arc.yaml` passes
- End-to-end confirmation: `config.arc.yaml` parses panel(3) + judge=gpt-4o; `run_debate` is indeed called by Stages 14/18

---

## List of Affected Files

**Added**
- `researchclaw/pipeline/debate.py` — Multi-model debate engine
- `tests/test_reviewer_llm.py`, `tests/test_debate_engine.py`
- `tests/test_rc_executor.py::TestDebateWiring` (2026-06-04, wiring regression)

**Modified**
- `researchclaw/config.py` — reviewer_* + debate_enabled/debate_rounds fields and parsing
- `researchclaw/llm/client.py` — `reviewer_from_rc_config`
- `researchclaw/llm/__init__.py` — `build_reviewer_llm` + `build_panel_llms`
- `researchclaw/prompts/shared.py` — `debate_rebuttal` + `review_synthesize` (2026-06-04) sub-prompts
- `researchclaw/pipeline/stage_impls/_review_publish.py` — Stage 18/20 review separation + provenance + **Stage 18 debate wiring (Landed 2026-06-04)**
- `researchclaw/pipeline/stage_impls/_analysis.py` — **Stage 14 debate wiring (Landed 2026-06-04)**
- `experiments/arc_bench/scripts/judge.py` — Homology guard + independent review declaration
- `config.arc.yaml` (Local, gitignored) — Actual enablement of reviewer + debate
- `config.researchclaw.example.yaml`, `CLAUDE.md` — Docs and examples
- Note: `_synthesis.py` (Stage 8) is not in this batch—its multi-plan modification follows the best-of-N tournament (separate document), not the debate engine.

---

## Design Trade-offs / Notes
- **Opt-in, backward compatible**: When `reviewer_model` is empty + `debate_enabled: false` (default), all stages take the original path with zero behavior change.
- **Cost**: Enabling debate costs approx. `roles × (1 + rounds) + 1` calls per stage (≈ 7 calls/stage × 2 stages = Stages 14/18). Can be tuned via `debate_rounds` / `ARC_DEBATE_ROUNDS`, or disabled via `debate_enabled: false`.
- **Debate pool comes from existing models** (per user choice "reuse existing models"), including fallbacks; `deepseek-chat` was removed per request.
- **Not Done (Future work)**: Multi-model voting for Stage 15 Research Decision; cross-domain (hep/biology) dedicated `DEBATE_ROLES_REVIEW` (Stage 18 currently uses inline generic 3-reviewer roles `_REVIEW_DEBATE_ROLES`).

---

## Current Status
The above changes **have not been git committed**, waiting for user confirmation per protocol.
`config.arc.yaml` has been actively enabled: Generator = Claude Sonnet 4.6, Reviewer/Judge = gpt-4o, Debate pool = 3 models.
Debate acts on **Stages 14 / 18** (Stage 8 follows best-of-N tournament, default off).

> Real-world Implementation Checklist (As of 2026-06-04, all confirmed via file:line + tests):
> - ✅ Batch 1: Generator-Reviewer Separation (Stages 18/20)
> - ✅ Batch 2: Multi-Model Debate: Engine + **Stage 14 / 18 wiring** (Completed 2026-06-04, previously missed due to tool echo latency)
> - ✅ Item 6: Best-of-N Tournament (Stages 8 / 9, separate document)
