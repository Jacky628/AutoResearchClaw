# AutoResearchClaw Architecture Research Document

> **Goal**: To thoroughly explain the 23-stage state machine that "automatically produces conference-grade LaTeX papers from a single research topic." This document goes beyond listing bullet points to articulate how each mechanism **operates, how data flows, the rationale behind the design, and where the guards and boundaries lie.** All conclusions are backed by `file:line` evidence from the actual codebase.

---

## 0. System Overview: One Sentence and One Diagram

At its core, ResearchClaw is a **deterministic sequential state machine with an external "per-stage structured LLM call" and "Human-in-the-Loop (HITL) gates."** It decomposes the research workflow into 23 discrete stages concatenated into a `STAGE_SEQUENCE`. Each stage is a pure-functional execution unit:

```
Read upstream artifact files → Select and populate domain prompt templates → Call LLM → Multi-level fault-tolerant parsing of output
→ Write to disk in stage-NN/ → Validate artifacts against contracts → (If a gate) Block for human review → Write checkpoint
```

The design philosophy permeates the entire codebase:
- **Recoverable**: Every completed stage writes an atomic checkpoint; the system resumes from the next stage after a crash.
- **Human-Intervention Ready**: 3 hard gates (5/9/20) + 6 HITL strategies allow humans to pause, approve, modify artifacts, or inject guidance at any point.
- **Graceful Degradation**: Fall back to templates when LLMs are unavailable; force PROCEED with quality warnings when experiments fail repeatedly—producing a paper with warnings is preferred over an infinite loop.
- **Anti-Hallucination**: The paper writing phases constrain the LLM's "creative freedom" to actual experimental data, with citation verification and "numerical sanitization" performed before export.

---

## 1. The 23 Stages Table

**Source: `pipeline/stages.py:22-62` (`Stage` IntEnum)**; `STAGE_SEQUENCE = tuple(Stage)` (`:93`). `NEXT_STAGE` and `PREVIOUS_STAGE` are pre-calculated at build time using dictionary comprehensions for O(1) successor/predecessor retrieval during runtime (`:93-103`).

| # | Stage | Phase | Representative Artifact | Nature |
|---|-------|-------|---------|------|
| 1 | TOPIC_INIT | A: Scoping | `goal.md`, `hardware_profile.json` | Generate SMART goals + detect GPU/CPU |
| 2 | PROBLEM_DECOMPOSE | A | `problem_tree.md`, `topic_evaluation.json` | Decompose sub-problems + evaluate topic quality |
| 3 | SEARCH_STRATEGY | B: Literature | `search_plan.yaml`, `sources.json` | Multi-strategy search queries |
| 4 | LITERATURE_COLLECT | B | `candidates.jsonl` | **Non-LLM**: arXiv/Semantic Scholar crawling |
| **5** | LITERATURE_SCREEN | B | `papers_selected.jsonl` | **GATE**: Dual screening for relevance + quality |
| 6 | KNOWLEDGE_EXTRACT | B | `cards/*.md`, `knowledge_graph.json` | Extract each paper into knowledge cards |
| 7 | SYNTHESIS | C: Synthesis | `synthesis.md` | Clustering + gap identification |
| 8 | HYPOTHESIS_GEN | C | `hypotheses.md`, `perspectives/` | **Multi-role debate** to generate hypotheses |
| **9** | EXPERIMENT_DESIGN | D: Design | `exp_plan.yaml`, `domain_profile.json` | **GATE**: Experimental plan |
| 10 | CODE_GENERATION | D | `experiment.py`, `validation_log.json` | Three-layer routed code generation; **GATE** in `hep_ph` |
| 11 | RESOURCE_PLANNING | D | `schedule.json` | Scheduling of compute/dependencies/order |
| 12 | EXPERIMENT_RUN | E: Execution | `runs/`, `metrics.json` | **Sandbox execution**, parse metrics |
| 13 | ITERATIVE_REFINE | E | `refinement_log.json` | Edit-run-eval improvement loop |
| 14 | RESULT_ANALYSIS | F: Analysis | `analysis.md`, `experiment_summary.json`, `results_table.tex` | Cross-run aggregation + ablation check + paired tests |
| 15 | RESEARCH_DECISION | F | `decision.json` | Produce proceed/refine/pivot decision |
| 16 | PAPER_OUTLINE | G: Writing | `outline.md` | Outline |
| 17 | PAPER_DRAFT | G | `paper_draft.md`, `paper.tex` | **Anti-fabrication** full draft |
| 18 | PEER_REVIEW | G | `reviews.md` | Simulated triple-blind review |
| 19 | PAPER_REVISION | G | `paper_draft_revised.md` | Revision based on reviews |
| **20** | QUALITY_GATE | H: Finalization| `quality_report.json` | **GATE**: Quality scoring (non-critical) |
| 21 | KNOWLEDGE_ARCHIVE | H | `retrospective.md` | Retrospective archiving (non-critical) |
| 22 | EXPORT_PUBLISH | H | `export/paper.pdf`, `code.zip` | Export + push to GitHub/HF |
| 23 | CITATION_VERIFY | H | `citations_verified.json`, `paper_final.md` | **Citation verification + numerical sanitization** (Critical) |

---

## 2. State Machine Kernel: How `advance()` Calculates Each Step

The "brain" of the machine is `advance(stage, status, event, ...)` (`stages.py:250-359`). It is a pure function: given the **current stage, current status, and a trigger event**, it returns a `TransitionOutcome` (next stage, next status, whether a checkpoint is required, and a decision label). Valid state transitions are constrained by `TRANSITION_MAP` (`:187-201`), and `advance()` branches by event within that framework:

**① START → RUNNING** (`:266-273`): When the `START` event is received in any of the three "dormant states" (`pending`, `retrying`, `paused`), the system enters `running` with `next_stage=stage` (executing in place).

**② SUCCEED → Branching** (`:276-290`) — This is the **Gate Decision Point**:
```python
if event is SUCCEED and status is RUNNING:
    if gate_required(stage, hitl_required_stages):
        return TransitionOutcome(status=BLOCKED_APPROVAL, next_stage=stage,
                                 checkpoint_required=False, decision="block")
    return TransitionOutcome(status=DONE, next_stage=NEXT_STAGE[stage],
                             checkpoint_required=True)
```
Key design: **No checkpoint is written after a gate executes successfully** (`checkpoint_required=False`). This is because the stage might be rejected and rolled back by a human; writing to disk too early would mislead a resumption into thinking it has already passed.

**③ APPROVE → DONE** (`:293-299`): Once human approval is granted, `checkpoint_required` becomes `True`—resumption always starts from the "last human-approved milestone."

**④ REJECT → Rollback** (`:302-310`):
```python
if event is REJECT and status is BLOCKED_APPROVAL:
    return TransitionOutcome(stage=target_rollback, status=PENDING,
                             rollback_stage=target_rollback,
                             checkpoint_required=True, decision="pivot")
```
`target_rollback` is taken from `GATE_ROLLBACK[stage]`. The rejected stage is reset to `PENDING` for a rerun. The checkpoint records "rolled back to stage X" to prevent accidental reruns of rejected stages.

**⑤ FAIL/RETRY/PAUSE** (`:323-339`): Errors during `running` → `failed` (write checkpoint to remember the stuck point); `failed + retry` → `retrying`; `failed` → `paused` (hand over to human).

> **Why use a pure function + `TRANSITION_MAP`?** Decoupling state transitions from side effects (writing to disk, calling LLMs) makes the entire control flow unit-testable, reason-able, and replay-able.

---

## 3. Gates and Rollbacks: Three Hard Gates + Domain-Aware Rollbacks

**Gate Set** (`stages.py:109-115`): `{LITERATURE_SCREEN(5), EXPERIMENT_DESIGN(9), QUALITY_GATE(20)}`—situated after literature screening, experiment design, and paper quality check, respectively.

**`gate_required()` Predicate Logic** (`:214-242`) is more complex than it appears:
```python
is_gate = stage in GATE_STAGES
is_hep_ph_codegen_gate = (stage is CODE_GENERATION and profile == "hep_ph")
if not is_gate and is_hep_ph_codegen_gate: is_gate = True
if not is_gate: return False
if is_hep_ph_codegen_gate: return True          # Stage 10 is always a gate in hep_ph
if hitl_required_stages is not None:            # Can be pruned by configuration
    return int(stage) in frozenset(hitl_required_stages)
return True
```
**`hep_ph` (High Energy Physics) Exception**: Stage 10 code generation is forcibly elevated to a gate and **ignores `hitl_required_stages` pruning**. Rationale: Downstream `ColliderAgent` runs extremely expensive and irreversible collider simulations (e.g., FeynRules/MadGraph). A human must review the `collider_plan.md` before burning compute resources. This is a profile-level invariant.

**Rollback Targets** (`GATE_ROLLBACK`, `:118-123`) are **domain-aware**:
| Rejected Gate | Rollback To | Semantics |
|-----------|--------|------|
| 5 Literature Screen | 4 Literature Collect | Poor literature → Collect more |
| 9 Experiment Design | 8 Hypothesis Gen | Flawed plan → Change hypothesis |
| 10 Code (hep_ph) | 9 Experiment Design | Redesign |
| 20 Quality Gate | 16 Paper Outline | Rewrite paper |

**Non-Critical Stages** (`NONCRITICAL_STAGES = {20, 21}`, `:140-146`): Failure in quality check or archiving does not block delivery. Note: **Stage 23 Citation Verification IS critical**—hallucinated citations must block export.

---

## 4. Main Loop and PIVOT/REFINE Recursion

### 4.1 Main Loop `execute_pipeline()` (`runner.py:431-912`)

Iterates through `STAGE_SEQUENCE`. `_should_start(stage, from_stage, started)` (`:35-38`) ensures the run begins only upon reaching the starting point. After each `execute_stage()`, the system branches based on the returned status:
- **DONE** → `_write_checkpoint()`, advance.
- **BLOCKED_APPROVAL** + `stop_on_gate` → break (hand over for human review).
- **FAILED** → Skip if `skip_noncritical` is true and the stage is non-critical, otherwise break.
- **PAUSED / REJECTED** → break.

### 4.2 Decision Rollback: PROCEED / REFINE / PIVOT

After Stage 15 produces a decision, if it is `refine/pivot`, the main loop enters **recursive rollback** (`runner.py:700-813`), the most complex control flow in the system:

```python
DECISION_ROLLBACK = {"pivot": HYPOTHESIS_GEN(8), "refine": ITERATIVE_REFINE(13)}
MAX_DECISION_PIVOTS = 2
```
- **PIVOT**: The hypothesis is fundamentally wrong → Roll back to Stage 8 to regenerate hypotheses (discard old hypotheses).
- **REFINE**: The hypothesis is fine, but there is a bug in experiment execution → Roll back to Stage 13 for a rerun (keep the hypothesis). In agent modes (`collider/biology/stat`), `refine` rolls back to Stage 12.

**Version the directory before recursion** — `_version_rollback_stages()` (`:1234-1276`) has two behaviors:
```python
if incremental and stage_num >= EXPERIMENT_RUN:   # agent mode refine
    shutil.copytree(stage_dir, version_dir)        # COPY: Keep workspace for incremental improvement
else:
    stage_dir.rename(version_dir)                  # RENAME: stage-NN → stage-NN_v{n}
```
**Why distinguish between copy and rename?** A standard `refine` just needs a rename (rerun from a clean state next time). However, agent modes require copying so the redeployed `ColliderAgent` can read the CSV/KO tables from the previous round to perform **incremental improvement** rather than recalculating from scratch.

Then, **recursively call `execute_pipeline(from_stage=rollback_target)`**. After the recursion returns, call `_promote_best_stage14()` to elevate the optimal results, then break (subsequent steps are handled by the recursion).

**Prevention of Infinite Loops + Graceful Degradation**:
- When `pivot_count >= 2`, no further rollbacks occur. Call `_check_experiment_quality()` (`:1431-1510`); if quality is poor, write `quality_warning.txt` and force `PROCEED`.
- **R6-4 Guard**: If consecutive `REFINE` cycles produce empty metrics (`_consecutive_empty_metrics`), immediately force `PROCEED`—the experiment is failing silently, and further loops are useless.

### 4.3 `_promote_best_stage14()`: Why "Promote Best" is mandatory (`:1313-1429`)

After rolling back, the directory contains `stage-14/` (the latest, possibly degraded) and `stage-14_v1/`, `stage-14_v2/` (history). Paper writing stages (16-22) only read `stage-14/experiment_summary.json`; if no promotion occurs, the worst round might be used. This function:
1. Scans all `stage-14*` and sorts them by the primary metric (`metric_key`/`metric_direction`).
2. **BUG-226 Guard**: For minimization metrics, if the "best value" is more than 1000x smaller than the second best, it is judged as degraded (NaN/training crash) and discarded in favor of the second best.
3. Writes the best summary to the root directory as `experiment_summary_best.json` (BUG-223, written before any early return), then copies the best `summary/analysis/charts` into `stage-14/`.

---

## 5. Checkpoints and Resumption

**Writing** (`_write_checkpoint()`, `runner.py:78-107`) uses **atomic writes**: first write a temporary file using `mkstemp`, then use `Path(tmp).replace(target)` for an atomic rename—ensuring no half-corrupted checkpoint is left behind after a mid-process crash. The content includes `last_completed_stage/name/run_id/timestamp` and embeds `hitl_session.hitl_checkpoint_data()` (if available).

**Reading** (`read_checkpoint()`, `:126-143`) returns the **next stage to be executed**, not the completed one: if it crashes at Stage 5, the checkpoint records 5, but the read returns Stage 6. `resume_from_checkpoint()` (`:146-151`) feeds this to `from_stage`.

Additionally, there are `heartbeat.json` (PID/last stage, for liveness probing) and `pipeline_summary.json` (`:41-75`, full-process statistics).

---

## 6. Full Lifecycle of a Single Stage Execution (7 Phases)

`execute_stage()` (`executor.py:594-821`) is a "dispatcher + guard layer," wrapping each stage into a unified 7-step pipeline:

**Phase 1 — HITL Pre-Stage Hook** (`_run_hitl_pre_stage`, `:200-254`): Before any work, ask the HITL session if it should pause. Can **short-circuit**: `SKIP` → return `DONE` immediately (no execution); `ABORT` → return `FAILED` to terminate; if a human provides guidance text, it is written to `stage_dir/hitl_guidance.md` for consumption by the stage logic; otherwise, proceed normally if it returns `None`.

**Phase 2 — Input Contract Validation** (`:615-627`): Look for upstream artifacts using `_read_prior_artifact()` based on `StageContract.input_files`. If missing, return `FAILED` immediately (`decision="retry"`) **without calling the executor**—failing fast to prevent cascading errors.

**Phase 3 — Build LLMClient + PromptManager** (`:639-661`): The LLM is **optional**; if creation fails, `llm=None`, and the stage degrades to template-based fallback (e.g., using a template to generate a goal in Stage 1 when no LLM is available). `PromptManager` selects the domain bank according to `config` and is built once per stage.

**Phase 4 — Dispatch** (`:651-669`): `_STAGE_EXECUTORS[stage](stage_dir, run_dir, config, adapters, llm=llm, prompts=prompts)`. For old implementations that do not accept the `prompts` kwarg, retry without prompts after catching `TypeError` (backward compatibility).

**Phase 5 — Output Contract Validation** (`:680-705`): Check that artifacts exist and are non-empty (for directories, check they are non-empty directories) based on `output_files`. If not met, downgrade `DONE` to `FAILED`.

**Phase 6 — PRM Quality Assessment + Gate Check** (`:707-794`): If MetaClaw PRM is enabled, score stages 5/9/15/20 and write to `prm_score.json` (errors are **non-blocking**). Then evaluate `gate_required()`: if not `auto_approve_gates`, set status to `BLOCKED_APPROVAL` and send notifications.

**Phase 7 — Metadata + HITL Post-Stage Hook** (`:799-821`): Write `stage_health.json` (duration/status/artifact count) and `decision.json` (routing). `_run_hitl_post_stage` (`:257-437`) handles: cost guards (pause if over budget), SmartPause (PRM < 0.7 can trigger re-review), human actions (APPROVE/REJECT/EDIT/COLLABORATE/SKIP/ABORT), where `COLLABORATE` enters a stdin interactive chat loop (`:440-564`).

---

## 7. Shared LLM Facilities: Fault Tolerance is the Priority

### 7.1 `_chat_with_prompt()` (`_helpers.py:744-802`)
A unified wrapper for single-turn calls with several fault-tolerance points:
- **Exponential Backoff Retry**: `delay = 2**(attempt+1)` (2s, 4s, 8s).
- **HTTP 400 Auto-downgrade `json_mode`**: Some OpenAI-compatible proxies do not support `response_format`; auto-disable `json_mode` and retry upon 400 errors.
- **`strip_thinking=True`**: Strip `<think>...</think>`. Rationale: Reasoning models like o3/o4 output Chain-of-Thought; these tags break downstream JSON/YAML parsers and LaTeX compilation.

### 7.2 `_safe_json_loads()` Four-Level Fault-Tolerant Parsing (`:511-580`)
LLM output is often messy (preamble/postscript chatter, markdown wrapping, CoT), so parsing is split into four levels:
1. **Direct `json.loads`**.
2. **Extract ```json``` code fences**.
3. **Balanced Brace Matching**: Scan all `{...}` pairs and take the largest valid dict by length (LLM chatter often precedes the real data block).
4. **Balanced Bracket Matching** for arrays `[...]`.
If all fail, return the `default` provided by the caller, ensuring the process does not crash.

### 7.3 `_read_prior_artifact()` and Version De-prioritization (`:397-421`)
`_stage_sort_key`: `stage-13` → `("stage-13", 0)` (highest priority), `stage-13_v2` → `("stage-13", -2)` (lower). When scanning in reverse, **non-versioned directories win**. Design intent: Retries produce `_vN` directories; downstream stages should prioritize the original (non-versioned) artifacts and only fall back to versioned recovery versions if the original is missing, preventing a bad retry from polluting downstream stages.

### 7.4 Multi-Role Debate + Synthesis (for Stage 8)
- `_multi_perspective_generate()` (`:1678-1719`): Invoke the LLM once for each role (ML: innovator/pragmatist/contrarian; HEP: theorist/phenomenologist/experimentalist) and store outputs in `perspectives/<role>.md`. Includes an ablation hook `ARC_ABL_DISABLE_DEBATE=1` for A/B testing against single-role degradation.
- `_synthesize_perspectives()` (`:1722-1738`): Concatenate multiple perspectives into context and use a synthesizer sub-prompt to converge into a unified set of hypotheses. **Separating divergence (debate) from convergence (synthesis)** combats groupthink.

### 7.5 PromptManager (`prompts/manager.py`)
- `_load_bank()` (`:74-94`): `hep_ph` → `hep.py`, `biology_metabolic` → `biology.py`, others → `ml.py` (silent fallback to ML for unknown domains). Domain is fixed at build time and remains consistent.
- `for_stage()` (`:214-244`): `_render` uses the regex `\{(\w+)\}` for variable substitution (only matching bare identifiers so that `{...}` in JSON schemas are not mistakenly replaced), then sequentially appends `evolution_overlay` (cross-round experience) and `extra_prompts` from config. Returns `RenderedPrompt(system, user, json_mode, max_tokens)`.
- Prompt Structure: Each stage has `{system, user, json_mode, max_tokens}`. `shared.py` contains `_DEFAULT_BLOCKS` (~20 reusable snippets), `_DEFAULT_SUB_PROMPTS`, and `SECTION_WORD_TARGETS`.

---

## 8. Internal Mechanisms of Key Stages

### 8.1 Stage 8 Hypothesis Generation: Three-Layer Fallback (`_synthesis.py:92-198`)
1. Multi-role debate generates perspectives; **BUG-S2 Guard**: If all roles fail (empty perspectives), immediately fall back to `_default_hypotheses()`; never feed empty context to the LLM (to avoid pure hallucination).
2. Synthesis is only called if at least one perspective succeeds.
3. If `hitl_guidance.md` exists, refine hypotheses based on human guidance.
4. Pack as Workshop candidates for UI review (silent skip on failure).
5. **Novelty Verification**: `check_novelty()` compares against collected papers, producing `novelty_score/assessment/recommendation`—**Non-blocking** (external API timeouts do not stall this stage).

### 8.2 Stage 10 Code Generation: Complexity-Driven Three-Layer Routing (`_code_generation.py:227-800`, `code_agent.py`)
```
① Beast Mode (OpenCode) — Triggered only if complexity score > threshold (default 3.5)
   Signals: multi-file, custom algorithms, history of failures in this run; requires HITL confirmation if not in auto mode.
   Success → write files and skip ②③; Failure → fall back to ②.
② CodeAgent (Advanced LLM) — Blueprint planning → Sequential generation → Execution-in-the-loop repair.
   Phase 1: Produce YAML architecture blueprint (BUG-178: pre-process values containing ':' or '->' to avoid YAML crashes).
   Phase 2: File-by-file generation + hard validation (syntax/security/imports) + exec-fix loop.
   Phase 3: Optional tree search (parallel implementations, sandbox selection, expensive, opt-in).
   Phase 4: Coder-reviewer multi-round review (default 2 rounds).
③ Legacy Single-Shot — Fallback directly to a single LLM call.
```
**Repair Loop**: `max_repair=5`; if `validate_code()` fails, feed the issues back to the LLM for modification; break once it passes.

### 8.3 Stage 12 Execution + Stage 13 Iterative Refinement (`_execution.py:119-850`)
- **Mode Routing** (`:149-287`): `collider_agent` reads `collider_plan.md` and runs `ColliderAgentSandbox`; `sandbox/docker` runs `run_project` or `run` via `create_sandbox`.
- **Metric Extraction Guard**: **R6-2**: Even with exit code 0, judge as failed if stdout contains `FAIL:`, `NaN/divergence`, or `Traceback`; if timed out but metrics exist → `partial`; if completed in <5s → warn that the benchmark may be too easy (P1).
- **Refinement Loop** (`:843-893`): Max `min(max_iterations, 10)` rounds, with a **BUG-57 wall-clock cap** (default 1.5x per-round budget) to prevent infinite running. Each round runs the experiment, performs `_find_metric` fuzzy matching (BUG-214: exact first then substring to avoid "accuracy" mis-matching "balanced_accuracy"), and compares with baseline. **Converge after 3 rounds of no improvement**; pause if 2 consecutive rounds yield NaN/Inf; pause if best value drops below 50% of baseline (degradation).
- **BUG-58 Recovery**: When PIVOTing back to Stage 13, scan all `stage-13_v*` and select the `best_metric` refined code instead of using the original unrefined code.

### 8.4 Stage 14 Analysis + Stage 15 Decision (`_analysis.py`)
- **Stage 13 Merge (BUG-165)**: Replace Stage 12 data only if refinement results are **actually better**, preventing catastrophic regressions (e.g., 78.93% → 8.65%).
- **Condition Aggregation + P0-D Rescue**: Parse `condition/metric/seed` format; if all parsing fails, synthesize a single `default_condition` to avoid the `cond_count=0` penalty.
- **Ablation Validity (P8 / R5-BUG-03)**: Compare conditions pairwise; identical → `ABLATION FAILURE` (identifying parameters that did not take effect in code); <1% difference → `ABLATION WARNING` (trivial ablation).
- **Paired Statistical Testing (R33)**: When ≥3 common seeds exist, the pipeline calculates paired t-tests using `scipy`; if t-stats reported by experiment code are identical or too sparse, use the pipeline-calculated ones—detecting broken/duplicate statistical tests in experiment code.

### 8.5 Stage 17/23 Anti-Hallucination (`_paper_writing.py`, `_review_publish.py`)
- **Stage 17 Real Metric Injection (R4-2/BUG-29)**: `_collect_raw_experiment_metrics` collects real numbers from stage 12/13/collider, applying hard constraints in the prompt: "Every number in the Results table must come from the following data; do not fabricate, do not round arbitrarily, and do not paste raw paths into the paper." **BUG-207**: When selecting sandbox entries, use the dual criteria of "highest primary metric + richest metric set" (old bug only looked at quantity, allowing a 1.29% accuracy `sandbox_after_fix` to overshadow a 78.93% `sandbox` because it had 6 more keys).
- **Stage 23 Numerical Sanitization** (`_sanitize_fabricated_data`, `:716-1160`): Recursively collect all finite numbers from `experiment_summary_best.json` (BUG-222: use the promoted best) as the "validated set." Scan paper tables/text with **1% relative tolerance** (allowing 0.734 ↔ 73.4% percentage/decimal conversion). **BUG-175 Whitelist**: Allow common hyperparameters (learning rate, batch, small integers ≤20) and Adam betas. **Table Classification**: Skip hyperparameter and statistical test tables (t/p/effect size are not in the summary). **BUG-184 Column-wise**: Only sanitize numerical results columns, preserving method names/condition labels. Unvalidated numbers in the results area are replaced with `---` or `[value removed]`.
- **Stage 23 Citation Verification (BUG-194)**: Parse missing citation keys `(author, year, hint)` and use a three-layer strategy to resolve/complete against collected papers, removing hallucinated citations.

### 8.6 Experiment Safety Validation (`experiment/validator.py`)
`validate_code()` (`:314-349`) = AST syntax check + `_SecurityVisitor` safety scan + import blacklist:
- **DANGEROUS_CALLS** (`:67-91`): `os.system`, `subprocess.*`, `shutil.rmtree`, etc. → error.
- **DANGEROUS_BUILTINS** (`:94-101`): `eval`/`exec`/`compile`/`__import__` → error.
- **BANNED_MODULES** (`:104-117`): `subprocess`/`socket`/`http`/`urllib`/`ctypes`/`signal` → error.
- **SAFE_STDLIB** (`:120-165`): json/math/datetime etc. whitelist always allowed.
`experiment_repair.py`: Diagnose → Generate fix → Rerun; default `max_cycles=3`; elevate the `experiment_summary.json` of the best cycle to `stage-14/`.

---

## 9. Configuration System (`config.py`)

`RCConfig` is a **frozen dataclass** (`:830-865`) with 17 top-level sections (project/research/runtime/llm/security/experiment + memory/skills/kg + multi_project/mcp/overleaf + server/dashboard + trends/copilot/quality_assessor/calendar/hitl). Immutable to prevent accidental runtime modification; nested dataclasses ensure clear ownership for each subsystem.

**Loading and Profile Resolution** (`load()`, `:1007-1057`): `yaml.safe_load` → If `project.profile` or `--profile` is set, call `apply_profile_defaults()`. Profile YAML search order: `./profiles/` → `~/.researchclaw/profiles/` → builtin `researchclaw/domains/profiles/`. **User always wins**: Profiles only fill keys that the user left unset (`config[key] ?? profile_default[key]`). A single "use `hep_ph`" switches experiment mode, prompt templates, and pip packages.

**Security Section** (`:951-959`): `hitl_required_stages` defaults to `(5,9,20)`; `allow_publish_without_approval=False`; `redact_sensitive_logs=True` (wipe API keys/emails before writing to disk).

---

## 10. LLM Client (`llm/client.py`)

`LLMClient` maintains `_model_chain = [primary] + fallbacks` (`:98`). `chat()` (`:179-240`) tries models sequentially; if an exception occurs, log a warning and swap to the next one—RuntimeError is only thrown if all fail. **Transparent to user**: gpt-5.2 unavailable? Auto-try gpt-5.1.

`_call_with_retry()` (`:285-382`) uses backoff + jitter: `delay = min(base*2^attempt, 300s) + uniform(0, 30%)` to prevent thundering herds. Error classification: 403 + "not allowed to use model" → swap to fallback immediately; 400 → retry ONLY if it contains keywords like rate limit/overloaded (Azure quirk), otherwise throw as bad request; 429/5xx/529 → retry. `preflight()` (`:242-283`) distinguishes 401 (bad key)/403 (model disabled)/404 (endpoint error)/429/timeout to provide CLI feedback.

**Model Family Adaptation**: `o3`/`gpt-5.*` uses `max_completion_tokens` and ≥32768 (`:26-37, 422-426`); `o3*`/`o4-mini` does not support temperature (`:39-44`); Claude/DeepSeek/Qwen or `responses` API do not support `response_format`, so JSON instructions are injected in the system prompt (`:428-461`). MetaClaw bridge uses a proxy, falling back to direct connection on `URLError` (`:143-156, 482-504`); Anthropic provider uses `AnthropicAdapter` to connect directly to the Messages API (`:169-176`).

---

## 11. Experiment Execution Modes (`experiment/factory.py` + Various Sandboxes)

The factory `create_sandbox(config, workdir)` (`:18-108`) selects the class based on `config.mode`; all backends implement `SandboxProtocol` (`run(code)` / `run_project(dir)` → `SandboxResult{returncode, stdout, stderr, elapsed_sec, metrics, timed_out}`, `sandbox.py:281-300`).

| Mode | Execution Mechanism | Security / Features |
|------|---------|----------|
| sandbox | Local subprocess | Executed after AST validation |
| docker | Three phases: `pip install` → `setup.py` (data download) → `entry_point` | Network policies: none/setup_only/full; cut network with iptables after setup; GPU passthrough; symlink escape check |
| ssh_remote | scp upload → ssh execution | Timeouts: experiment 600s/scp 300s/setup 300s; pre- and post-entry_point validation |
| colab_drive | Async: write `pending/` → Colab worker polls and moves to `running/` → writes `done/result.json` | Decoupled: Colab session timeout kills the experiment but not the pipeline; Drive sync resists disconnects |
| agentic | Claude Code/Codex in container | Full shell access |
| collider_agent | HEP: FeynRules/MadGraph/MadAnalysis | + Magnus |
| biology_agent | FBA/pFBA/FVA | + COBRApy/BIGG |
| stat_agent | Simulation studies | + scipy/statsmodels |

**Metric Parsing** (`parse_metrics`, `sandbox.py:89-198`) supports `SUMMARY ...`, ratios `metric: N/M`, `condition=X metric: value`, and pure `metric: value`, while **rejecting NaN/Inf**.

---

## 12. HITL Human-Machine Collaboration (`hitl/`)

**6 Strategies** (`InterventionMode`, `config.py:10-18` + presets `:200-243`):
| Mode | Behavior |
|------|------|
| full-auto | No pauses |
| gate-only | Pause for 5/9/20 approval + artifact editing |
| checkpoint | Pause at phase boundaries (2/6/8/11/13/15/19/23) |
| step-by-step | Pause after every stage + streaming output |
| co-pilot | Enable collaboration for 7/8/9/17, require approval for 5/8/9/15/20 |
| custom | User-defined `StagePolicy` for each stage |

`StagePolicy` (`config.py:29-57`) provides fine-grained switches: `pause_before/after`, `require_approval`, `stream_output`, `allow_edit_output/inject_prompt`, `enable_collaboration`, `min_quality_score`, `max_auto_retries`, `human_timeout_sec`.

**`HITLSession` Runtime** (`session.py`): `pause()` → `wait_for_human()` → `resume()`. `wait_for_human()` prioritizes registered callbacks; otherwise, polls `run_dir/hitl/response.json` (allowing asynchronous reconnection from `attach`/web/MCP tools hours later); falls back to auto-approval on timeout. Each intervention is logged in `interventions` (duration/action/result). The CLI's `attach/status/approve/reject/guide` commands read/write these files to communicate with the running session.

---

## 13. Adapters, Artifacts, and CLI

**`AdapterBundle`** (`adapters.py:121-140`) uses Protocols to decouple six capabilities: `Cron` (scheduled resumption), `Message` (notifications), `Memory` (experience appending), `Sessions` (long-lived sessions), `WebFetch`, and `Browser`. Default is the **Recording stub** (logs calls for testing); `from_config()` swaps to MCP adapters when `mcp.server_enabled` is true. Significance: Stages call `message.notify()` without directly importing Slack/email libraries—loosely coupled, testable, and replaceable.

**Artifact Root** `artifacts/rc-<timestamp>-<topic_hash>/`: `checkpoint.json` / `heartbeat.json` / `pipeline_summary.json` / `stage-NN[_vM]/`; `experiment_summary.json` (`_analysis.py:552-554`), `results_table.tex` (`:556-558`), `paper.tex` (`_review_publish.py:2086`), and root-level `experiment_summary_best.json`.

**CLI** entry point `cli.py:main()` `:1458-1510` (`pyproject.toml [project.scripts] researchclaw=researchclaw.cli:main`). Commands: run/validate/doctor/init/setup/info/report/serve/dashboard/wizard/project/mcp/overleaf/trends/calendar/skills/profile + HITL (attach/status/approve/reject/guide). `run` workflow `cli.py:156-459`: Parse config → profile defaults → CLI overrides → LLM preflight → generate run_id/run_dir → create KB subdirectories → attach HITL session → `execute_pipeline()` → summary.

---

## 14. Design Summary (Why this design)

| Mechanism | Design Intent |
|------|---------|
| Gate success doesn't write checkpoint; approval does | Resumption always starts from the "human-approved milestone"; rejection doesn't leave a dirty checkpoint. |
| Stage 10 is a mandatory gate in `hep_ph` | Collider plans must be human-reviewed before burning massive compute. |
| PIVOT Rename vs. Agent-REFINE Copy | Standard reruns need a clean state; agent incremental improvement needs the workspace preserved. |
| `_promote_best_stage14` + BUG-226 | Papers use the best round's data; prevents degraded rounds from polluting results. |
| Force PROCEED + Quality Warning | Prevents infinite loops; a paper with warnings is better than no paper. |
| Atomic Checkpoint (temp + rename) | Crash doesn't leave a partially corrupted file. |
| Pre-Stage Input Validation | Fail fast to prevent cascading errors. |
| Versioned Directory De-prioritization | A bad retry doesn't pollute downstream stages. |
| Strip `<think>` / 4-Level JSON Tolerance | Resists reasoning model CoT and messy outputs. |
| Model Chain + Backoff Jitter | User-transparent high availability + prevents thundering herds. |
| Real Metric Injection + Sanitization + Citations | Three lines of defense against LLM fabrication. |
| Protocol Adapters + Recording Stubs | Loosely coupled and testable. |

---

## 15. Verification (Confirming Understanding)

```bash
# Stage Enums / Gates / Rollback Constants
python -c "from researchclaw.pipeline.stages import Stage, GATE_STAGES, GATE_ROLLBACK, DECISION_ROLLBACK; print(list(Stage)); print(GATE_STAGES); print(GATE_ROLLBACK); print(DECISION_ROLLBACK)"
# Dispatcher Table covering 23 stages
python -c "from researchclaw.pipeline.executor import _STAGE_EXECUTORS; print(len(_STAGE_EXECUTORS))"
# Security Validation Blacklist
python -c "from researchclaw.experiment.validator import validate_code; print(validate_code('import os; os.system(\"ls\")').issues)"
# Relevant Unit Tests
python -m pytest tests/ -k "stage or gate or checkpoint or contract or validator or pivot" -q
```

> **Note**: This file is a research document, not an implementation plan; it does not involve any code changes.
