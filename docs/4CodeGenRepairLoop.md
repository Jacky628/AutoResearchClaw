# In-depth Analysis of AutoResearchClaw Code Generation and Repair Loops

> **Why this document**: Transforming a "YAML experimental plan" into a "multi-file Python project capable of producing real scientific results" is the **most engineering-heavy and failure-prone** link in the pipeline. LLM-generated code often suffers from syntax errors, runtime crashes, "fake success" (printing hardcoded metrics), or degraded experimental designs (e.g., identical outputs across different ablation conditions). AutoResearchClaw addresses these challenges with a sophisticated **multi-layered generation + multi-stage validation + dual-loop repair** mechanism. This document breaks down the mechanism layer by layer, explaining **what each component does, how data flows between them, the rationale behind the design, and how various failures are handled.** All conclusions are backed by `file:line` evidence.
>
> This document complements `ArchitectureResearch.md` sections §8.2/§8.6 (Overview) by diving into executable details. For security scanning details, see `SecurityModel.md`.

---

## 0. Global Cognition: Two Independent Repair Loops

The most important thing to understand about this subsystem is that **AutoResearchClaw performs "repair" at two distinct stages using two independent mechanisms with completely different goals.**

| | **Generation-Time Repair** (Stage 10) | **Result-Time Repair** (Stages 13/14 Area) |
|---|---|---|
| **Location** | `code_agent.py` + `_code_generation.py` | `experiment_repair.py` |
| **Goal** | Make the code **runnable** (syntax, no crashes, correct APIs). | Make the results **meaningful** (non-degraded, valid ablations, quality threshold met). |
| **Trigger** | During the generation stage, before the full experiment runs. | After the experiment finishes, if results are poor or insufficient. |
| **Criteria** | `validate_code` passes + sandbox returncode == 0. | `assess_experiment_quality` judges as `sufficient`. |
| **Means** | Feedback from validation / execution errors / reviews fed back to LLM. | Diagnostic reports (`diagnose_experiment`) fed back to LLM. |

**In short**: The generation-time loop ensures "the code runs," while the result-time loop ensures "what runs looks like a real experiment."

---

## 1. Stage 10's Three-Layer Routing: Beast Mode → CodeAgent → Legacy → Fallback

`_execute_code_generation` (`_code_generation.py:227`) decides which path to use to write code based on a **routing chain with degradation** (`:489-804`). The philosophy is "best capability first, degrade to more stable paths on failure."

### 1.1 Layer 1: Beast Mode (External OpenCode Agent, Optional)

Considered only if `config.experiment.opencode.enabled` is true (`:496`). It first uses `score_complexity()` (`:505-509`) to assign a **complexity score** based on the experimental plan, topic, and **previous failures in the current run**. If the score exceeds the `complexity_threshold` (default 3.5), it recommends `beast_mode` (`:528`).

This is followed by a **Human-in-the-Loop (HITL) gate** (`:529-547`): if `opencode.auto` is false, it asks the human via an adapter whether to route to OpenCode. Once engaged, `OpenCodeBridge.generate()` runs the external agent. **Failure gracefully degrades to CodeAgent** (`:603-607`).

### 1.2 Layer 2: CodeAgent (Advanced LLM Code Agent)

The **default primary path**. It begins with domain detection. For **non-ML domains**, it runs a `CodeSearchAgent` to find real API usages and reference repositories (`:653-669`), injecting these as context. It then initializes `CodeAgent` (`:673-682`) and calls `generate()`.

### 1.3 Layer 3: Legacy Single-Shot Generation + Validation Loop

If CodeAgent is disabled but an LLM is available (`:718`), it uses the traditional single-shot path: one `_chat_with_prompt` call followed by `_extract_multi_file_blocks`. **Empty responses are retried once with a higher token limit (32768)** (`:750-766`). It then enters a `max_repair=5` validation-repair loop.

### 1.4 Layer 4: Final Fallback (Hardcoded Synthetic Experiment)

If no files are produced (`:776`), the system prevents pipeline failure by injecting a **hardcoded numpy parameter sweep experiment** (`:777-804`). This ensures Stage 12 has something to run and Stage 14 has metrics to parse, embodying the "graceful degradation" philosophy.

---

## 2. CodeAgent's Five-Phase Pipeline (The Main Path)

`CodeAgent.generate()` (`code_agent.py:192-279`) operates on a five-phase pipeline.

### 2.1 Phase 1 — Blueprint Planning (`_phase1_blueprint`, `:283-326`)

The LLM produces an **implementation blueprint** first: a YAML document listing the files to be created, their responsibilities, dependencies, and `generation_order`. Non-ML domains append domain-specific guidance and search results to the prompt (`:300-306`). Planning ahead moves architectural decisions to the cheapest phase.

### 2.2 Phase 2 — Sequential Generation + CodeMem (`_phase2_sequential_generate`, `:472-576`)

CodeAgent generates files one by one following the `generation_order` (`:496`). For each file generated, it uses `_build_code_summary` (`:608-653`) to perform **AST analysis and extract a compressed summary of signatures and imports (CodeMem)**.

When generating file N, the system injects the **CodeMem summaries** and **full code** of its declared dependencies into the prompt. This ensures the LLM "knows" the interfaces of preceding files, preventing cross-file `NameError` or `AttributeError`.

### 2.3 Phase 2.5 — Hard Validation Gate (`_hard_validate_and_repair`, `:657-705`)

After all files are generated, a round of **static AST validation** is performed. Issues are categorized into **CRITICAL** (triggers rework, max 4 times) and **WARNING** (logged only) (`_hard_validate`, `:707-799`). CRITICAL issues include:
- **Syntax errors** (always critical).
- **Class quality issues**: Identical AST to parent (fake inheritance), non-real ablations, or empty subclasses.
- **Experimental integrity**: Hardcoded metrics or trivial computations.
- **Correctness**: API mismatches or variable scope issues (e.g., `UnboundLocalError`).

### 2.4 Execution-Repair Loop (`_exec_fix_loop`, `:962-979`)

The code is **actually executed in the sandbox** for up to 3 rounds. If it crashes, `_fix_runtime_error` (`:1029-1072`) attempts a **Targeted Repair (E-05)**: it parses the Python traceback to identify the specific file and line number, sending only the affected file and a focused context window to the LLM for fixing.

During code generation, **numerical stability requirements are forcibly injected (BUG-004)**: gradient clipping, NaN loss checks (`if torch.isnan(loss): print('FAIL: NaN detected')`), and NaN/Inf guards for metrics. The `print('FAIL: ...')` statement allows Stage 12 to detect "fake successes."

### 2.5 Phase 3 — Tree Search (`_phase3_tree_search`, `:1196-1278`, Default: Off)

When enabled, the system generates `tree_search_candidates=3` candidate implementations and evaluates each in the sandbox (`_score_node`, `:1298-1312`). If the best candidate fails, it generates repair branches for the top two, exploring up to `tree_search_max_depth=2`.

### 2.6 Phase 4/5 — Coder-Reviewer Dialogue (`_phase4_review`, `:1316-1377`)

Finally, it runs up to 2 rounds of review using a `code_reviewer` sub-prompt. A separate "reviewer agent" returns a JSON verdict. If issues are found, the coder agent is asked to fix them, providing all files (including unchanged ones) in the response.

---

## 3. Validation Loop in the Legacy Path (`_code_generation.py:806-...`)

The Legacy path uses a simpler loop: it runs `validate_code` for each file and enters a `max_repair=5` loop on failure. Crucially, **only `severity==error` issues are fed back to the LLM; warnings are omitted** (`:818-822`). This prevents the LLM from over-modifying the code, such as deleting imports used at runtime.

---

## 4. Result-Time Repair Loop (`experiment_repair.py`)

This **independent second loop** occurs after the experiment finishes if results are unsatisfactory. Entry point: `run_repair_loop` (`:274`).

For each cycle (up to `MAX_REPAIR_CYCLES`), it performs:
1. **Diagnosis**: `diagnose_experiment` (`:358`) analyzes the experimental summary, plan, logs, and **accumulated history of previous diagnoses** to identify what went wrong.
2. **Repair Prompt**: `build_repair_prompt` (`:369`) assembles the diagnosis, code, plan, and budget into instructions.
3. **Repaired Code**: `_get_repaired_code` (`:375`) obtains the fix (LLM primary, OpenCode secondary).
4. **Execution**: The repaired code is saved to a versioned directory (e.g., `stage-14_repair_v1/`) and rerun in the sandbox.
5. **Evaluation**: If the new result is `sufficient`, it exits early. Otherwise, it tracks the best version found so far.

Only if a round improves the results is the best summary promoted to `experiment_summary_best.json` (`:485-487`).

---

## 5. How the Two Loops Connect (Flowchart)

```
Stage 10 Code Generation (Generation-Time Loop: Make it run)
  ├ Beast Mode? ──Success→ Use OpenCode output
  │              └Failure→↓
  ├ CodeAgent (Primary):
  │    Phase 1: Blueprint → Phase 2: Sequential Gen + CodeMem → Phase 2.5: Hard Validation
  │    → Execution-Repair Loop (Targeted/Full, Stability Injected) → Phase 4: Review
  │              └ Disabled →↓
  ├ Legacy Single-Shot + max_repair=5 loop (Errors only)
  │              └ All Fail →↓
  └ Fallback: Hardcoded numpy synthetic experiment
        │
        ▼ experiment/ + experiment_spec.md
  Stage 11/12 Scheduling and Execution → runs/metrics
        │
        ▼ If results are poor/insufficient
  Stage 13/14 Area (Result-Time Loop: Make it meaningful)
  experiment_repair.run_repair_loop:
    Each cycle: diagnose → fix → save to v{n}/ → rerun → evaluate → track best
    Stop if sufficient; Promote best to experiment_summary_best.json
```

---

## 6. Design Summary (Why this design)

| Mechanism | Origin | Intent |
|------|------|------|
| Three-layer routing with degradation | `_code_generation.py` | Prioritizes capability, ensures the pipeline never breaks. |
| Numpy fallback experiment | `:776-804` | Placeholder experiment ensuring Stage 12 always has something to run. |
| Blueprint before coding | `code_agent.py` | Catches architectural errors in the cheapest planning phase. |
| Sequential Gen + CodeMem AST summary | `:472-576` | Prevents cross-file NameErrors while avoiding token limit overflow. |
| Hard validation critical/warning levels| `:707-799` | Prevents rework oscillation by focusing only on blocking issues. |
| Intercept fake ablations/hardcoded metrics| `:738-763` | Pre-execution defense against "pretend" results. |
| Targeted repair priority (E-05) | `:1029-1072` | Cheaper and more precise than fixing all files. |
| Numerical stability + FAIL signals | `:1000-1009` | Proactive failure signaling for Stage 12 detection. |
| Diagnosis-driven Result-Time Repair | `experiment_repair.py` | Uses structured analysis to fix semantic/result issues. |
| Versioned cycles + Promotion of best | `:390-487` | Ensures the paper uses the optimal round's data. |

---

## 7. Key Configuration Items (`CodeAgentConfig`)

| Option | Default | Meaning |
|------|------|------|
| `architecture_planning` | True | Execute Phase 1 blueprint planning. |
| `sequential_generation` | True | Generate files one by one based on order. |
| `hard_validation` | True | Execute Phase 2.5 AST hard validation. |
| `hard_validation_max_repairs` | 4 | Max attempts for critical validation repairs. |
| `exec_fix_max_iterations` | 3 | Max rounds for the execution-repair loop. |
| `tree_search_enabled` | **False** | Enable parallel implementation exploration (Expensive). |
| `review_max_rounds` | 2 | Rounds of coder-reviewer dialogue. |

---

## 8. Verification

```bash
# Verify CodeAgentConfig defaults
python -c "from researchclaw.pipeline.code_agent import CodeAgentConfig as C; c=C(); print('seq:',c.sequential_generation,'hard:',c.hard_validation,'tree:',c.tree_search_enabled)"

# Check for existence of Phase methods in CodeAgent
python -c "from researchclaw.pipeline.code_agent import CodeAgent; print([m for m in dir(CodeAgent) if m.startswith('_phase')])"

# Read logic for critical issues (fake ablations/hardcoded metrics)
sed -n '737,763p' researchclaw/pipeline/code_agent.py

# Relevant Unit Tests
python -m pytest tests/ -k "code_agent or code_gen or repair or validator or complexity" -q
```

> **Note**: This file is a research document and does not involve any code changes.
