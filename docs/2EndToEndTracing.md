# AutoResearchClaw End-to-End Tracing: How a Hypothesis Travels from Stage 8 to Stage 12

> **Why this document**: `ArchitectureResearch.md` describes the static structure of the system—how the state machine rotates, how gates are guarded, and what each subsystem is responsible for. But the real "magic" of a research pipeline lies in the **hop-by-hop collapse of information forms**: how a scientific hypothesis written in natural language incrementally transforms into a structured experimental plan, thousands of lines of runnable Python code, a task dependency graph, and finally, a digital dictionary of results that can, in turn, verify the initial hypothesis. This document selects **one real hypothesis** as a sample (from artifact `rc-20260425-054156-fdedd0`, topic "Fine-tuning LLMs for Parametric CAD Modeling") and deconstructs its journey through the five hops from Stage 8 to 12: **what is read at each hop, how it is processed internally, what is written out, how the data form changes, the rationale behind the design, and the fallbacks for errors.** All conclusions are backed by `file:line` evidence and verified by actual code.
>
> This document is a dynamic supplement to `ArchitectureResearch.md`: while that document explains "what the machine is made of," this one explains "how data passes through it." For deep dives into mechanisms like "how code is generated" or "how results are analyzed," this document will point to the corresponding sections in `ArchitectureResearch.md` to avoid duplication.

---

## 0. The Tracing Subject, Overview, and Connecting Principles

### 0.1 The Hypothesis Under Trace

```
H1: Establishing "primitive-level" dedicated tokens (one token for each sketch/extrude operation)
    for CAD operations will increase the geometric validity rate of generated CAD sequences
    by ≥20 percentage points compared to standard BPE subword tokenization.
```

The key feature of this sentence is that it **carries its own measurable prediction (≥20pp) and falsification conditions**—the prerequisites for "translating" it into experimental parameters in subsequent hops. We will follow it through the pipeline to see how it is materialized.

### 0.2 The Five-Hop Overview

```
Stage 8  hypotheses.md
   │  (Semi-structured markdown: Hypothesis + Prediction + Falsification + Resource Estimate)
   ▼
Stage 9  exp_plan.yaml  (+ domain_profile.json)
   │  (Structured plan: baselines / proposed_methods / ablations / metrics / compute_budget)
   ├──────────────┬──────────────────────────────
   ▼              ▼
Stage 10        Stage 11
experiment/     schedule.json
(main.py +      (Dependency DAG of 22 tasks)
 6 modules)           │
   │                 │
   └────────┬────────┘
            ▼
        Stage 12  runs/run-1.json + results.json
            │  (Machine-readable hierarchical metrics dictionary)
            ▼
        Stage 13 / 14  Read runs/ → Verify H1 → Decision: proceed/refine/pivot
```

### 0.3 Connecting Principle: The File Name Contract

Adjacent stages in the pipeline are not connected by passing in-memory objects but are welded together through **files written to disk + a contract declaring "which files I need and which files I will produce"** (defined in `StageContract` in `pipeline/contracts.py`). Below are the contracts for the five stages involved, verified against the actual code:

| Stage | input_files (Read) | output_files (Write) | Definition of Done |
|------|------------------|--------------------|--------------------|
| 8 HYPOTHESIS_GEN | `synthesis.md` | `hypotheses.md` | ≥2 falsifiable hypotheses |
| 9 EXPERIMENT_DESIGN | `hypotheses.md` | `exp_plan.yaml` | Plan with baselines/ablations/metrics approved |
| 10 CODE_GENERATION | `exp_plan.yaml` | `experiment/`, `experiment_spec.md` | Multi-file experimental project + spec doc |
| 11 RESOURCE_PLANNING | `exp_plan.yaml` | `schedule.json` | Resource schedule with GPU/time estimates |
| 12 EXPERIMENT_RUN | `schedule.json`, `experiment/` | `runs/` | All scheduled tasks completed with artifacts |

This design is intentional. **Using pure filenames instead of memory objects yields four capabilities**: ① Any stage can be rerun individually using `--from-stage` because its inputs are ready on disk; ② The system can resume from a checkpoint after a crash; ③ HITL (Human-in-the-Loop) can **manually edit intermediate artifacts** between stages (e.g., a human edits `exp_plan.yaml` before proceeding); ④ The executor can perform mechanical contract validation before and after calling the stage logic—checking that `input_files` exist before execution (`executor.py:615-627`) and that `output_files` exist and are non-empty after execution (`executor.py:681-705`). If either fails, the stage is immediately judged as failed, stopping errors at the source.

---

## 1. Hop ① Stage 8 → 9: From a One-Sentence Hypothesis to a YAML Plan

This hop performs one of the largest semantic spans in the pipeline: **Natural Language → Structured Experimental Design**.

### 1.1 What `hypotheses.md` from Stage 8 looks like

The output of Stage 8 (`stage_impls/_synthesis.py:92-198`), `hypotheses.md`, is a **semi-structured markdown** file. It has a stable semantic skeleton but is not a strict machine schema (this is intentional: hypotheses are meant for humans to read and for the next LLM to interpret; markdown is better suited for argumentation than JSON). `_default_hypotheses()` (`:1580-1594`) provides the fallback skeleton. Each hypothesis includes:

- `### Hypothesis N: <Title>` — The hypothesis itself, a falsifiable assertion.
- `**Rationale**` — Scientific basis explaining why it's worth verifying.
- `**Measurable Prediction**` — **Quantifiable prediction**, critical for translation into metrics. For H1, this is the "≥20 percentage point improvement." It might also include diagnostic sub-assertions and scaling probes.
- `**Failure Condition**` — Falsification conditions; under what circumstances the hypothesis is refuted.
- `**Resource Requirement**` — Estimates for compute/time/engineering effort.

The file ends with two global sections: `## Unresolved Disagreements` (retaining points of contention from the multi-role debate) and `## Recommended Sequencing` (gate suggestions: e.g., "If H1 holds, proceed to H2; otherwise pivot to H3").

Additionally, Stage 8 produces `perspectives/<role>.md` (raw outputs from each role in the debate, `_helpers.py:711`) and `novelty_report.json` (`{novelty_score, assessment, recommendation}`, **non-blocking**, `:179`).

### 1.2 How Stage 9 reads hypotheses, and an easily overlooked truncation

Stage 9 (`_experiment_design.py:74-86`) reads the hypotheses immediately:

```python
hypotheses = _read_prior_artifact(run_dir, "hypotheses.md") or ""   # :83
preamble = _build_context_preamble(config, run_dir,
              include_goal=True, include_hypotheses=True)            # :85
```

A **critical and often overlooked detail**: Hypotheses enter the prompt in two places with different treatments. In the "Context Preamble" built by `_build_context_preamble`, hypotheses are truncated to `hyp[:2200]` characters (`_helpers.py:1061`); whereas the full hypothesis text is injected separately via the `{hypotheses}` placeholder. Why this split? The preamble provides a "background overview"; stuffing the entire long hypothesis document there would crowd out the output token budget for the YAML plan itself. Thus, a truncated version serves as an overview, while the full version is provided separately—a classic "overview vs. full text" trade-off in prompt engineering.

### 1.3 The Prompt is more than "Hypothesis + Instructions"—Injection of a dozen variables

Previous versions simplified this to "inject hypothesis, request YAML," but the actual prompt assembly (`:200-214`) injects **over a dozen context variables**, each shaping the LLM's output:

```python
sp = _pm.for_stage("experiment_design",
        evolution_overlay=_overlay,          # Cross-round experience (if pivot history exists)
        preamble=preamble,                   # Background preamble with truncated hypothesis
        hypotheses=hypotheses,               # Full hypothesis text
        dataset_guidance=_dg_block,          # Dataset guidance (adds RL step guidance for RL topics)
        domain_design_context=_domain_design_context,  # YAML injection for non-ML domains
        time_budget_sec=config.experiment.time_budget_sec,
        metric_key=config.experiment.metric_key,
        metric_direction=config.experiment.metric_direction,
        hardware_profile=_hw_profile_str,    # GPU model/VRAM
        per_condition_budget_sec=_per_condition_sec,  # = time_budget*0.7/6
        available_tier1_datasets=_tier1)
```

Notable dynamic behaviors:
- **RL Topic Special Handling** (`:156-180`): If the topic contains keywords like "reinforcement learning," "ppo," or "mujoco," RL step guidance is appended. If the time budget is ≤3600s, it **forces** the use of classic control environments (CartPole, etc.); ≤1800s narrows it to the simplest ones—because environments like MuJoCo cannot produce meaningful results in less than 5000s. This hardcodes "compute reality" into the prompt.
- **Framework Documentation Injection** (`:181-190`): `detect_frameworks` identifies frameworks from the topic/hypothesis, and `load_framework_docs` pulls up to 4000 characters of documentation into the prompt to ensure the plan aligns with real APIs.
- **Domain Detection First** (Next subsection).

The prompt template (`prompts.default.yaml:94-105`) finally mandates YAML output with eight specific keys: `objectives, datasets, baselines, proposed_methods, ablations, metrics, risks, compute_budget`.

### 1.4 Domain Detection: Deciding the "Research Paradigm"

Before calling the LLM, Stage 9 performs domain detection (`:89-118`):

```python
_domain_profile = detect_domain(topic=config.research.topic, hypotheses=hypotheses)
```

It determines the research domain based on the topic and hypothesis text, writing `domain_profile.json`: `{domain_id, display_name, experiment_paradigm, core_libraries, gpu_required}`. This profile is not just metadata—`experiment_paradigm` (comparison / convergence / simulation / ablation_study) influences how the experiment is organized, `core_libraries` are passed to Stage 10 for library selection, and `gpu_required` affects resource planning. For domains like ML/HEP with native prompt banks, behavior is unchanged; for others, domain guidance defined in YAML is injected via `GenericPromptAdapter` (`:124-147`).

### 1.5 The `exp_plan.yaml` produced by Stage 10: Hypotheses "Parameterized"

The LLM output, after parsing, normalization, and pruning, is dumped to `exp_plan.yaml` (`:556-559`). The real structure (root key `experiment_plan`) is as follows; note how each block traces back to a requirement in the hypothesis:

```yaml
experiment_plan:
  topic: "...";  generated: "2026-04-25T..."
  objectives:        # Translates "measurable prediction" into executable goals (primary + diagnostic)
  datasets:          # Includes factorial_design—in this case, 2 complexity × 2 param range = 4 evaluation conditions
                     # Also includes scale (2000 train / 400 val / 400 test) and parameter distributions
  baselines:         # [{name, implementation_spec{class_name, algorithm_steps,
                     #    key_hyperparameters}, training{method, epochs, lr, batch_size}}]
                     # e.g., bpe_subword_sft_qwen2 (Baseline for H1)
  proposed_methods:  # Same structure. e.g., primitive_token_vocab_sft (Protagonist for H1)
  ablations:         # [{name, variable, conditions:[label...], held_constant,
                     #    expected_outcome, compute_estimate}]
                     # Tokenization_granularity ablation directly serves H1:
                     #   bpe_default / integer_only_tokens / decimal_aware_tokens
  metrics:
    primary: [{name: sequence_validity_rate,   # The "validity rate" H1 intends to measure
               formula: "(# valid) / (# total)",
               range: "[0,1]; higher is better",
               aggregation: "mean ± std over 3 random seeds"}]   # ← 3 seeds agreed here
    secondary: [operation_type_accuracy, mean_parameter_l1_error, ...]
    statistical_testing: {method: "paired bootstrap 10000",
                          correction: "Bonferroni for 3 comparisons",
                          significance: "p < 0.05"}
  compute_budget: ["1× RTX 3090 24GB", "Total 1500s", "Per-condition ~175s"]
  hypotheses_tested: ["H1: Primitive tokens improve validity; effect size distinguishes reasoning vs retrieval"]
```

The `hypotheses_tested` field is an explicit backlink—the plan clearly declares which hypotheses it is verifying. This sample ended with **2 baselines + 2 proposed + 4 ablations = 8 conditions**, each running **3 seeds**. Note that `metrics.primary.sequence_validity_rate` is exactly what H1 needs, and the `ablations` (bpe_default vs decimal_aware_tokens) are exactly what H1 contrasts—**every requirement of the hypothesis found its corresponding executable structure in the YAML.**

### 1.6 Condition Pruning: Preventing inherently doomed experiments (BUG-R41-09)

A critical guard (`:458-503`) acts before writing to disk. LLMs often "greedily" design 30+ conditions, which will inevitably time out under a finite budget. Thus, a cap is set based on the time budget:

```python
_max_conditions = 8                  # budget ≤ 3600s
if _time_budget > 3600: _max_conditions = 12
if _time_budget > 7200: _max_conditions = 20
```

The pruning strategy is sophisticated (`:480-503`): **Priority is given to `proposed_methods`** (the protagonists, keeping up to `_max_conditions-4`), with the remaining budget split between baselines and ablations. In other words, when the plan is too large, the system would rather cut baselines and ablations to ensure "proposed new methods" finish—because the new methods are the core of the paper.

### 1.7 Robustness: Normalization + Four-Level YAML Fallback

Outputting YAML from an LLM is unreliable, so Stage 9 has layers of protection:

**Field Normalization** `_normalize_plan_field()` (`:33-60`): LLMs might write `baselines` as a list of strings, a list of dicts, or even a large dict. This function unifies all forms into `list[dict]` while **retaining the full structure**—converting `{"baseline_1": {...}}` into `{"name": "baseline_1", ...}` without losing keys or values.

**Four-Level Parsing Fallback** (`:222-344`), attempting levels sequentially until success:
1. `_extract_yaml_block` to extract ```yaml``` fences, then `yaml.safe_load`.
2. Fall back to parsing the **entire response** as YAML (reasoning models often omit fences).
3. **Line-by-line scanning**: Capture starting from a line containing known keys (`baselines:`, etc.), skipping empty lines/comments, then attempt parsing (`:234-258`).
4. **Retry with a short, strict prompt** ("Output YAML ONLY, no prose") (BUG-12, `:270-292`).

If even the retry fails, there are two final layers that produce a plan **without an LLM**: extracting method/baseline names from `hypotheses.md` using regex (`:294-324`), or using topic-derived placeholders (`:326-344`). **Regardless of the level reached, the final `plan` dictionary is guaranteed to contain the 8 mandatory keys**—the contract's hard guarantee ensuring downstream stages always receive a complete structure.

---

## 2. Hop ② Stage 9 → 10: From a YAML Plan to Runnable Code

This hop transforms "what experiments to do" (declarative YAML) into "how to run the experiments" (imperative Python project).

### 2.1 Stage 10 Inputs and Prompt Injections

```python
exp_plan = _read_prior_artifact(run_dir, "exp_plan.yaml") or ""  # :243
metric   = config.experiment.metric_key                         # :244  e.g., "validity_rate"
max_repair = 5                                                  # :245  Limit for repair loop
```

It then constructs a **hardware-aware, network-policy-aware** prompt context (`:252-346`):

- **`pkg_hint` generated by GPU tier** (`:273-296`): Reads `hardware_profile.json`. For high-performance GPUs, it suggests "GPU acceleration"; for limited VRAM/MPS, it suggests **designing lightweight experiments** (models <1M params, epoch ≤20, samples ≤10K, small batch). This translates "hardware reality" into constraints for the code generator. For Docker, it lists pre-installed packages and indicates if `pip install` is possible based on the network policy.
- **`compute_budget`** (`:300-312`): Injects the total time limit and requires the generated code to **implement a time guard—gracefully stopping at 80% budget**.
- **`extra_guidance`** (`:314-346`): Injects "Strictly Offline" guidance for `network=none`, download guidance for `full`, and setup script guidance for `setup_only`.

Finally, the prompt template (`prompts.default.yaml:72-93`) injects the full `{exp_plan}`, primary `{metric}` key, and `{pkg_hint}`, while stipulating the **output format contract**—the fundamental agreement for Stage 12's parsing:
- Entry point must be `main.py`, running experiments and printing metrics.
- **"main.py must print metrics line-by-line in `name: value` format"**—Stage 12's `parse_metrics` relies on this.
- `main.py` must also write a structured `results.json`.

### 2.2 The Multi-file Project Written by Stage 10

The contract requires `output_files=("experiment/", "experiment_spec.md")`. In this sample, `stage-10/experiment/` is a 7-file project: `main.py` (entry point), `cad_data.py`, `cad_tokenizers.py` (contrast tokenization strategies for H1), `llm_models.py` (LoRA with peft/TRL), `deepcad_model.py` (transformer baseline), `dpo_trainer_module.py`, and `requirements.txt`.

The docstring and constants in `main.py` "compile" the `exp_plan` into an execution plan:

```python
# 5 training conditions (from exp_plan baselines+proposed):
NUM_MAIN_CONDITIONS = 5
NUM_SEEDS = 3                       # ← from exp_plan "over 3 seeds"
PER_SEED_BUDGET = (TIME_GUARD - OVERHEAD) / (NUM_MAIN_CONDITIONS * NUM_SEEDS)
# Metric printing contract (parsed by Stage 12):
print(f'condition={name} validity_rate={mean:.6f}')
```

The transfer of data forms is clear: **`baselines+proposed` count → `NUM_MAIN_CONDITIONS=5`; `metrics.aggregation` "3 seeds" → `NUM_SEEDS=3`; `compute_budget` → `TIME_GUARD` constant; `metrics.primary.name` → `validity_rate` in print statements.**

The side artifact `experiment_spec.md` records the entry point, file list, primary metric key, `Topic-Experiment Alignment: ALIGNED`, and `Validated: N warning(s)`.

---

## 3. Hop ③ Stage 9 → 11: From a YAML Plan to a Task Dependency Graph

Note that the starting point for this hop is also `exp_plan.yaml`, not the Stage 10 code—**Stage 10 and Stage 11 are parallel consumers of the same plan.**

### 3.1 Stage 11 Inputs and Outputs

```python
exp_plan = _read_prior_artifact(run_dir, "exp_plan.yaml") or ""  # _execution.py:53
sp = _pm.for_stage("resource_planning", exp_plan=exp_plan)        # :58  json_mode=True
schedule = _safe_json_loads(resp.content, {})                     # Parsed as dict
```

The prompt template (`prompts.default.yaml:315-321`) forces JSON output with a specific schema: `{tasks:[{id, name, depends_on, gpu_count, estimated_minutes, priority}], total_gpu_budget, generated}`.

### 3.2 `schedule.json`: Expanding the Plan into a Dependency DAG

The LLM expands the `exp_plan` into a **task dependency graph**. In this sample, 22 tasks have a dependency structure reflecting experimental causality:

```
T01 data_generation (0 GPU) ──→ T02 oracle_unit_tests
T03/T04/T05 Train baseline+proposed (depends on T02, 1 GPU each)
T05 ──→ T06 construct_dpo_pairs ──→ T07 train_dpo   ← DPO must follow SFT
T08 eval_all (depends on T03,T04,T05,T07—Evaluation waits for all training)
T09..T20 Ablations (depend on respective parent training, reuse checkpoints)
T21 statistical_testing (depends on all ablations)
T22 results_aggregation (depends on T21)
```

Dependencies are inferred from the semantics of the `exp_plan`: DPO naturally builds on SFT; evaluation requires trained models; statistical tests compare all conditions.

> **Design Observation: Why do both 10 and 11 read `exp_plan` independently?** Because "how to implement the experiment" (Stage 10 code) and "how to schedule the experiment" (Stage 11 DAG) are **two orthogonal concerns**. Decoupling them allows parallel understanding and independent reruns.

---

## 4. Hop ④ Stage 10/11 → 12: From Code to Structured Metrics

This is the most dramatic transformation: **human-readable but hard-to-parse stdout text collapses into a machine-readable hierarchical numeric dictionary.** This is where all the preceding semantic work meets "real-world numbers."

### 4.1 Stage 12 Inputs

```python
schedule_text = _read_prior_artifact(run_dir, "schedule.json") or "{}"   # :131
exp_dir_path  = _read_prior_artifact(run_dir, "experiment/")             # Prefers multi-file project
code_text = (Path(exp_dir_path)/"main.py").read_text()                  # Entry point
```

### 4.2 Sandbox Dispatch and Isolated Execution

The backend (sandbox/docker/ssh/colab/agentic) is selected via `create_sandbox()` based on `config.experiment.mode`. Before execution, the sandbox performs three actions (`sandbox.py:314-451`): copies the project to an isolated `runs/sandbox/_project_1/` directory, **injects an immutable `experiment_harness.py`** (preventing generated code from tampering with measurement logic), and performs symlink escape checks. It then runs the subprocess with `PYTHONUNBUFFERED=1` to ensure all stdout is captured in real-time.

### 4.3 `parse_metrics()`: The Soul of this Hop (`sandbox.py:89-198`)

After the code runs, its stdout contains lines like `condition=dpo seed=0 l1_error: 21.8`. `parse_metrics` uses **five branches to match lines by priority**, collapsing text into a `dict[str, float]` with hierarchical keys using `/`.

1. **SUMMARY Line (Most reliable)**: `SUMMARY condition=X metric=Y mean=M std=S`. Produces four keys: `X/Y`, `X/Y_mean`, `X/Y_std`, and a global `Y` fallback.
2. **Condition + Ratio**: `condition=X regime=fast seed=0 validity_rate: 9/10`. Calculates `9/10` as 0.9 and creates `X/fast/validity_rate`, `X/validity_rate`, and `validity_rate`.
3. **Condition + Single Metric**: `condition=X [tags] metric: value`.
4. **Condition + Multiple Metrics**: `condition=dpo seed=0 l1_error: 21.8 param_csr: 0.88`. Uses `finditer` to extract all pairs; if `seed=` is present, creates a **per-seed key** `dpo/0/l1_error`.
5. **Pure Metric Line**: `l1_error_mean: 4.9`. The simplest fallback.

Two guards across all branches:
- **`is_metric_name()` Filtering**: Only lines matching metric naming conventions are processed, filtering out log noise.
- **NaN/Inf Rejection (R5-3)**: Non-finite values are discarded to prevent polluting "best value" logic in downstream stages.

**Why the "Key Explosion"—writing multiple keys for one number?** Because downstream stages (13/14) might know the exact `condition/metric` or only a fuzzy `metric_key`. Writing hierarchical and fallback keys ensures retrieval succeeds regardless of precision.

### 4.4 `SandboxResult` and Three-State Status Determination

Results are wrapped in `SandboxResult` and written to `runs/run-1.json` (`:367-389`):

```json
{"run_id":"run-1", "task_id":"sandbox-main", "status":"completed",
 "metrics":{...hierarchical dict...}, "elapsed_sec":1234.5,
 "stdout":"...", "stderr":"", "timed_out":false}
```

**Status Determination (R6-2 Guard, `:334-358`)** — Smarter than just checking exit codes:
- `completed` = Exit code 0 **AND** not timed out **AND** no failure signals in stdout.
- `partial` = Timed out **but metrics were already collected**—data is incomplete but usable.
- `failed` = Everything else. **Crucially, even if exit code is 0, presence of `FAIL:`, `NaN/divergence`, or `Traceback` in stdout triggers `failed`**. This guards against "fake success" where generated code catches exceptions but prints a traceback and exits with 0.

---

## 5. Beyond the Endpoint: How Stage 13/14 Consume `runs/`

- **Stage 13 Iterative Refinement** (`_execution.py:715-760`): Iterates through `runs/*.json`, using `_find_metric(metrics, metric_key)` for **fuzzy baseline matching**: exact first, then `condition/metric`, then `_mean` keys, finally pure fallback keys.
- **Stage 14 Result Analysis** (`:629-655`): Reads `runs/*.json`, detects metric directions, aggregates metrics across runs/seeds, performs ablation validity checks, and paired statistical tests. This produces `analysis.md` + `experiment_summary.json` for Stage 15 to decide **whether H1 holds.**

---

## 6. Full Lifecycle of a Hypothesis (Sample Convergence)

Following H1's journey:

```
H1: "Primitive tokens improve validity rate by ≥20pp compared to BPE"
 │
 ├ Stage 8  → hypotheses.md
 │           ### Hypothesis 1 + **Measurable Prediction** "≥20pp" + **Failure Condition**
 │
 ├ Stage 9  → exp_plan.yaml (Pruning, 8-key constraint, fallback layers)
 │           proposed_methods=[primitive_token_vocab_sft], baselines=[bpe_subword_sft],
 │           metrics.primary=[sequence_validity_rate], ablations=[tokenization_granularity]
 │
 ├ Stage 10 → experiment/main.py (Hardware-aware, time-guarded, metric contract)
 │           NUM_MAIN_CONDITIONS=5, NUM_SEEDS=3,
 │           print('condition=bpe_sft validity_rate=0.0608')
 │
 ├ Stage 11 → schedule.json (22-task DAG)
 │           T04 train_bpe_sft, T05 train_primitive_sft, T08 eval, T21 stats
 │
 ├ Stage 12 → runs/run-1.json (parse_metrics collapses stdout)
 │           metrics{"bpe_sft/validity_rate_mean": 0.0608,
 │                   "primitive_sft/validity_rate_mean": 0.82, ...}  ← H1 verified here
 │
 └ Stage 13/14 Read runs/ → 0.82 - 0.0608 ≈ 76pp ≫ 20pp → H1 Strongly Supported → Proceed
```

---

## 7. Design Summary (Why this design)

| Mechanism | Origin | Intent |
|------|------|------|
| Filename contracts, no memory objects | `contracts.py` | Decoupling: independent reruns, resumption, HITL editing. |
| 2200-char truncation + full text split | `_helpers.py:1061` | Preamble for context, full text for detail; saves output tokens. |
| Injection of hardware/budget/frameworks | `_experiment_design.py`| Hardcodes compute/domain reality into constraints; prevents doomed experiments. |
| Domain detection produces profile | `:89-118` | Determines research paradigm, library selection, and GPU needs. |
| Condition pruning (8/12/20), priority to proposed | `:458-503` | Prevents timeouts; preserves the protagonists of the paper. |
| Eight-key constraint + YAML fallbacks | `:222-344` | Downstream always gets a complete structure, even if LLM fails. |
| Stage 10/11 read `exp_plan` independently | — | Implementation vs. Scheduling are orthogonal; allows parallelism. |
| `pkg_hint` layered by GPU and network policy | `_code_generation.py`| Tailors generated code to real hardware and environment. |
| Metric printing contract in prompt | `prompts.default.yaml` | Makes stdout mechanically parsable by `parse_metrics`. |
| `parse_metrics` 5-branch + key explosion + NaN rejection | `sandbox.py` | Compatibility; hierarchical/fuzzy retrieval; prevents pollution. |
| Status three-state + R6-2 fake success detection | `_execution.py`| Usable data on timeout; detects tracebacks in exit-0 runs. |
| Immutable harness + symlink escape check | `sandbox.py` | Isolated execution; prevents tampering or unauthorized access. |

---

## 8. Verification

```bash
# 1) Verify contract chain 8->9->10->11->12
python -c "from researchclaw.pipeline.contracts import CONTRACTS; from researchclaw.pipeline.stages import Stage; [print(s.name, CONTRACTS[s].input_files, '->', CONTRACTS[s].output_files) for s in [Stage.HYPOTHESIS_GEN, Stage.EXPERIMENT_DESIGN, Stage.CODE_GENERATION, Stage.RESOURCE_PLANNING, Stage.EXPERIMENT_RUN]]"

# 2) Test parse_metrics 5-branch and "key explosion"
python -c "from researchclaw.experiment.sandbox import parse_metrics; print(sorted(parse_metrics('SUMMARY condition=dpo metric=l1 mean=2.5 std=0.1\ncondition=bpe seed=0 acc: 0.9').keys()))"
# Expected: ['acc','bpe/0/acc','bpe/acc','dpo/l1','dpo/l1_mean','dpo/l1_std','l1']

# 3) Check metric key structure in a real run (if artifact exists)
python -c "import json,glob; f=glob.glob('artifacts/*/stage-12/runs/run-1.json'); print(list(json.load(open(f[0]))['metrics'].keys())[:30]) if f else print('no artifact')"

# 4) Relevant Unit Tests
python -m pytest tests/ -k "metrics or parse or contract or experiment_run or schedule or experiment_design" -q
```

> **Note**: This file is a research document and does not involve any code changes.
