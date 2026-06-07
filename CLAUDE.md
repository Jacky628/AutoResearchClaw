# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Commands

```bash
# Install
pip install -e ".[dev]"          # dev dependencies (includes pytest)
pip install -e ".[all]"          # all optional dependencies

# Run pipeline
researchclaw run --topic "..." --auto-approve          # fully autonomous
researchclaw run --topic "..." --mode co-pilot         # HITL co-pilot mode
researchclaw run --from-stage PAPER_OUTLINE --output artifacts/rc-<id>  # resume from stage

# Diagnostics
researchclaw doctor               # environment preflight checks
researchclaw validate --config config.yaml

# Tests
python -m pytest tests/test_rc_*.py -q --tb=short     # 2700+ unit tests
python -m pytest tests/ -k "test_name" -v             # single test
python tests/e2e_real_llm.py                          # E2E (requires API key)
```

## Configuration

Copy `config.researchclaw.example.yaml` → `config.arc.yaml` (gitignored). Key fields:
- `llm.base_url` / `llm.api_key` / `llm.primary_model` — any OpenAI-compatible provider
- `llm.reviewer_model` (+ optional `reviewer_provider`/`reviewer_base_url`/`reviewer_api_key_env`) — independent reviewer/judge model for Stage 18 peer review & Stage 20 quality gate (breaks generator==judge self-preference). Empty = reuse the generator model.
- `llm.debate_enabled` / `llm.debate_rounds` — opt-in multi-model debate engine for Stage 8 (hypothesis), 14 (analysis), 18 (peer review). Panel reuses `primary_model`+`reviewer_model`+`fallback_models` (deduped); each role argues with a distinct model, then rebuttal round(s), then an independent judge (`reviewer_model`) scores + synthesizes. Off by default (multiplies LLM calls). Records `debate_record.json` per stage. Note: on Stage 8 the tournament path takes priority — debate only drives hypothesis generation when `tournament_enabled` is off (when both are on, the tournament reuses the panel for breadth instead). With no panel (debate off), Stage 8 falls back to single-model multi-perspective synthesis.
- `llm.tournament_enabled` / `llm.tournament_candidates` — opt-in best-of-N tournament for Stage 8 (hypotheses) & Stage 9 (experiment design). Generates N candidate artifacts from diverse stances (round-robin over the debate panel when available), then an independent judge (`reviewer_model`) scores/ranks and the single winner proceeds. Pipeline stays linear (one canonical artifact per stage). Off by default; `< 2` disables. Multiplies calls (~N gen + 1 judge per stage). Records `tournament_record.json` per stage. Env: `ARC_TOURNAMENT_CANDIDATES` overrides N, `ARC_ABL_DISABLE_TOURNAMENT=1` collapses to 1. Complements debate (depth) with breadth.
- `experiment.mode` — `simulated` | `sandbox` | `docker` | `ssh_remote` | `colab_drive`
- `security.hitl_required_stages` — gate stage numbers (default `[5, 9, 20]`)

## Architecture

**ResearchClaw** is a 23-stage state machine that converts a research topic into a conference-ready LaTeX paper. Each stage is an LLM call with structured I/O defined in `pipeline/contracts.py`.

### Pipeline flow

```
Phase A (1-2):  Topic scoping → problem decomposition
Phase B (3-6):  Search strategy → literature collect → GATE(5) → knowledge extract
Phase C (7-8):  Synthesis → hypothesis generation
Phase D (9-11): Experiment design → GATE(9) → code generation → resource planning
Phase E (12-13): Experiment run → iterative refinement
Phase F (14-15): Result analysis → research decision (PROCEED/PIVOT/ITERATE)
Phase G (16-19): Paper outline → draft → peer review → revision
Phase H (20-23): QUALITY_GATE(20) → knowledge archive → export → citation verify
```

Gate stages 5, 9, 20 block until approved (or `--auto-approve`); rejection rolls back to the preceding phase entry.

### Key modules

| Module | Role |
|--------|------|
| `pipeline/runner.py` (1737 lines) | State machine, checkpoint management, stage dispatch |
| `pipeline/executor.py` (794 lines) | 23 individual stage executor functions + `_STAGE_EXECUTORS` dispatch table |
| `pipeline/_helpers.py` (1834 lines) | Shared LLM call wrappers, prompt rendering, response parsing |
| `pipeline/code_agent.py` (1508 lines) | Code generation via LLM / OpenCode / Claude / Codex routing |
| `pipeline/stages.py` | `PipelineStage` IntEnum, gate definitions, rollback rules |
| `pipeline/contracts.py` | `StageContract` — required/produced keys per stage |
| `experiment/validator.py` (42K) | AST syntax check, security scan, import validation |
| `pipeline/experiment_repair.py` | Auto-fix broken generated code, re-validate loop |
| `prompts.py` (150K) | All LLM prompt templates for all 23 stages |
| `config.py` | `RCConfig` dataclass, YAML loading, provider resolution |
| `adapters.py` | `AdapterBundle` — notification stubs + OpenClaw bridge |
| `llm/client.py` | `LLMClient` wrapping OpenAI-compatible APIs |

### Experiment execution modes

Sandbox (`experiment/sandbox.py`): local subprocess. Docker (`experiment/docker_sandbox.py`): isolated container. SSH (`experiment/ssh_sandbox.py`): remote machine. Colab (`experiment/colab_sandbox.py`): Google Colab. Agentic (`experiment/agentic_sandbox.py`): LLM-driven execution.

### HITL / co-pilot

`hitl/` and `copilot/` manage the 6 human intervention strategies. `collaboration/` handles interactive commands during a live run.

### Skills system

`.claude/skills/` contains 9 domain-specific skills (ResearchClaw pipeline runner, hypothesis formulation, biology/biopython, chemistry/rdkit, scientific writing, literature search, statistical reporting, visualization, a-evolve). These are loaded by the `Skill` tool and correspond to pipeline stages.

### Output artifacts

Each run produces `artifacts/rc-<id>/` containing checkpoints per stage, `experiment_summary.json`, `results_table.tex`, charts (PNG), and final LaTeX paper.
