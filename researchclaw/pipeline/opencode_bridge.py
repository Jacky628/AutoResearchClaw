"""OpenCode 'Beast Mode' bridge — routes complex code generation to OpenCode CLI.

OpenCode (https://github.com/anomalyco/opencode) is an external AI coding agent
invoked via ``opencode run --format json "prompt"``.  This module provides:

1. **ComplexityScore / score_complexity()** — analyses an experiment plan to
   decide whether beast mode is warranted.
2. **OpenCodeBridge** — manages workspace creation, OpenCode invocation, file
   collection, and cleanup.
"""

from __future__ import annotations

import ast
import json
import logging
import os
import re
import shutil
import subprocess
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Complexity scoring
# ---------------------------------------------------------------------------

# Keywords that indicate multi-component architectures
_COMPONENT_KEYWORDS: tuple[str, ...] = (
    "encoder",
    "decoder",
    "discriminator",
    "generator",
    "critic",
    "actor",
    "teacher",
    "student",
    "backbone",
    "head",
    "neck",
    "classifier",
    "embedder",
    "attention",
    "transformer",
    "tokenizer",
    "vae",
    "autoencoder",
)

# Indicators that multi-file generation is needed
_FILE_HINT_KEYWORDS: tuple[str, ...] = (
    "model.py",
    "trainer.py",
    "dataset.py",
    "utils.py",
    "config.py",
    "multiple files",
    "modular",
    "separate module",
    "multi-file",
)

# Domain-complexity keywords
_DOMAIN_COMPLEX_KEYWORDS: tuple[str, ...] = (
    "multi-modal",
    "multimodal",
    "distributed",
    "gan",
    "diffusion",
    "nerf",
    "mixture of experts",
    "moe",
    "meta-learning",
    "meta learning",
    "maml",
    "neural ode",
    "neural sde",
    "physics-informed",
    "pinn",
    "graph neural",
    "gnn",
    "reinforcement learning",
    "multi-agent",
    "world model",
    "vision-language",
    "text-to-image",
    "image-to-text",
)

# Patterns suggesting deep dependency chains
_DEPENDENCY_KEYWORDS: tuple[str, ...] = (
    "custom layer",
    "custom loss",
    "wrapper",
    "registry",
    "hook",
    "callback",
    "scheduler",
    "custom optimizer",
    "custom dataset",
    "custom sampler",
    "custom transform",
)


@dataclass
class ComplexityScore:
    """Result of complexity analysis on an experiment plan."""

    score: float  # 0.0-1.0
    signals: dict[str, float] = field(default_factory=dict)
    recommendation: str = ""  # "beast_mode" | "code_agent" | "legacy"
    reason: str = ""


def _count_keyword_hits(text: str, keywords: tuple[str, ...]) -> int:
    text_lower = text.lower()
    return sum(1 for kw in keywords if kw in text_lower)


def score_complexity(
    exp_plan: str,
    topic: str = "",
    *,
    historical_failures: int = 0,
    threshold: float = 0.6,
) -> ComplexityScore:
    """Score the complexity of an experiment to determine if beast mode is warranted.

    Returns a ComplexityScore with score in [0.0, 1.0].
    """
    if not exp_plan and not topic:
        return ComplexityScore(
            score=0.0,
            signals={},
            recommendation="legacy",
            reason="Empty plan",
        )

    combined = f"{topic}\n{exp_plan}"

    # Signal 1: Component count (weight 0.25)
    comp_hits = _count_keyword_hits(combined, _COMPONENT_KEYWORDS)
    component_score = min(comp_hits / 5.0, 1.0)

    # Signal 2: File count hint (weight 0.20)
    file_hits = _count_keyword_hits(combined, _FILE_HINT_KEYWORDS)
    file_score = min(file_hits / 3.0, 1.0)

    # Signal 3: Domain complexity (weight 0.20)
    domain_hits = _count_keyword_hits(combined, _DOMAIN_COMPLEX_KEYWORDS)
    domain_score = min(domain_hits / 3.0, 1.0)

    # Signal 4: Condition count (weight 0.15)
    # Look for numbered conditions, ablation mentions, variant mentions
    condition_pattern = re.compile(
        r"(?:condition|ablation|variant|experiment)\s*[\-_:]?\s*\d+",
        re.IGNORECASE,
    )
    condition_matches = len(condition_pattern.findall(combined))
    # Also count bullet points in conditions/ablations sections
    condition_matches += combined.lower().count("baseline")
    condition_score = min(condition_matches / 8.0, 1.0)

    # Signal 5: Historical failures (weight 0.10)
    failure_score = min(historical_failures / 3.0, 1.0)

    # Signal 6: Dependency depth (weight 0.10)
    dep_hits = _count_keyword_hits(combined, _DEPENDENCY_KEYWORDS)
    dep_score = min(dep_hits / 3.0, 1.0)

    # Weighted sum
    weighted = (
        0.25 * component_score
        + 0.20 * file_score
        + 0.20 * domain_score
        + 0.15 * condition_score
        + 0.10 * failure_score
        + 0.10 * dep_score
    )
    final_score = min(max(weighted, 0.0), 1.0)

    signals = {
        "component_count": round(component_score, 3),
        "file_count_hint": round(file_score, 3),
        "domain_complexity": round(domain_score, 3),
        "condition_count": round(condition_score, 3),
        "historical_failure": round(failure_score, 3),
        "dependency_depth": round(dep_score, 3),
    }

    if final_score >= threshold:
        recommendation = "beast_mode"
        reason = (
            f"Complexity {final_score:.2f} >= threshold {threshold:.2f}: "
            f"top signals: "
            + ", ".join(
                f"{k}={v:.2f}"
                for k, v in sorted(signals.items(), key=lambda x: -x[1])[:3]
            )
        )
    else:
        recommendation = "code_agent"
        reason = f"Complexity {final_score:.2f} < threshold {threshold:.2f}"

    return ComplexityScore(
        score=round(final_score, 4),
        signals=signals,
        recommendation=recommendation,
        reason=reason,
    )


# ---------------------------------------------------------------------------
# OpenCode bridge
# ---------------------------------------------------------------------------

@dataclass
class OpenCodeResult:
    """Result from an OpenCode invocation."""

    success: bool
    files: dict[str, str] = field(default_factory=dict)
    opencode_log: str = ""
    elapsed_sec: float = 0.0
    error: str = ""


_MEGA_PROMPT_TEMPLATE = """\
You are implementing a complete, runnable ML/science experiment.

Read the files in the current workspace:
- EXPERIMENT_PLAN.yaml — the full experiment design
- GUIDANCE.md — topic, metric, environment constraints, domain-specific guidance

ACADEMIC RIGOR IS MANDATORY (the design's real tools/data, not stand-ins):
- The plan in EXPERIMENT_PLAN.yaml declares the REAL models, tools, libraries
  (see its `environment` block) and datasets the experiment requires. You MUST
  actually import and use them. The setup phase installs declared pip packages
  and downloads declared datasets, so they ARE available — use them.
- If the plan specifies an LLM / fine-tuning, ACTUALLY load and train it
  (e.g. transformers + peft) — do NOT replace it with a toy torch.nn model.
- If the plan specifies a real evaluator/oracle (e.g. a CAD kernel like
  cadquery), ACTUALLY execute it to compute the metric — do NOT write a
  rule-based / regex / length-check "mimic" of it.
- NEVER fabricate or fall back to synthetic/random/dummy data unless the plan
  EXPLICITLY designates synthetic data for this domain. Concretely: do NOT define
  any `generate_synthetic*` / `make_synthetic` / dummy-data function, and do NOT
  write a `try: <load real> except: <synthesize>` fallback. If the declared real
  dataset cannot be downloaded, `raise RuntimeError("<dataset> unavailable")` —
  a failed run is acceptable; fabricated data is NOT.
- These substitutions are automatically detected and the stage will be BLOCKED.
- PREFER safe library APIs: download datasets via `datasets`/`huggingface_hub`
  (not raw `requests`); import and call tools like a CAD kernel IN-PROCESS with
  try/except for crash handling. Use `subprocess` only when process isolation is
  genuinely required (e.g. guarding against a native crash).

EVALUATOR / ORACLE CORRECTNESS (CRITICAL — a lenient oracle silently voids the ENTIRE experiment):
- A validity/quality oracle must be STRICT: its except / error / parse-failure /
  empty-result branch MUST report INVALID (False). NEVER "count as success if no
  exception fired", never treat a missing or degenerate result as valid. A result
  counts as valid ONLY if it genuinely passes the check (e.g. for a CAD kernel: a
  non-degenerate solid with positive bounding-box extents AND .val().isValid()).
- ORACLE SELF-TEST FIRST: before calibration, run ONE trivially-valid case through the
  oracle (e.g. for a CAD kernel: `cq.Workplane("XY").box(1,1,1)`) — it MUST score
  VALID. If it does not, the oracle IMPLEMENTATION itself is broken (wrong API name,
  bad subprocess wiring, inverted logic): print `ORACLE_SELF_TEST_FAILED: <error>` and
  raise. Use the library's exact Python API (check attribute names carefully — e.g.
  cadquery solids use `.isValid()`, not the OCC C++ spelling `IsValid()`).
- DATA FORMAT must match the oracle's input: implement EVERY data preprocessing /
  format-conversion step the plan declares (e.g. a deterministic transpiler from a
  structured JSON sequence format to executable code). If the dataset field is
  structured data (JSON) but the oracle executes code, you MUST implement and apply
  the converter for BOTH training targets and oracle inputs — never feed raw JSON to
  an execution oracle.
- CALIBRATE the oracle against GROUND TRUTH before trusting any number: run a sample
  of REAL reference outputs (the dataset's ground-truth target sequences) through the
  SAME oracle and print exactly `ORACLE_CALIBRATION: ground_truth_validity=<val>` (one
  line). If ground-truth scores ~0, the oracle pipeline is BROKEN (oracle bug, missing
  format conversion, or wrong field) — fix it before running any condition.
  Ground-truth validity is the realistic CEILING.
- CALIBRATE THE GENERATION PATH TOO (oracle calibration alone misses it): before any
  condition runs, MEASURE the dataset's actual target lengths (tokens AND whether
  programs are single-line — never assume multi-line) and derive from them:
  (a) the prompt/completion split — split by token or character FRACTION (e.g. first
  30% of characters), NEVER by line count, and assert the prompt is a strict proper
  prefix (prompt != full target); (b) set max_new_tokens to a TIGHT cap just above
  the p95 completion length (e.g. ceil(p95 * 1.3)), NOT a large round number far
  above p95. A cap far larger than p95 is harmful, not safe: an under-trained model
  that does not emit EOS will RAMBLE past the real program end up to the cap, and that
  trailing garbage (i) corrupts the program so the execution oracle scores it INVALID
  and (ii) makes every generation take cap-many tokens, so eval/RL crawl. Then
  generate a few samples and print
  `GENERATION_CALIBRATION: truncated_frac=<f> eos_frac=<f>` — if most generations hit
  the max_new_tokens cap (truncated_frac high), completions can NEVER be complete
  programs and every validity will be a meaningless 0.0: fix the budget before
  training anything.
- ALWAYS pass eos_token_id to generate() so well-formed outputs stop early. Before
  handing a generation to an execution oracle you MUST extract the valid program
  prefix — do NOT rely on EOS alone. An under-trained model frequently emits NO EOS
  at all (eos_frac≈0, truncated_frac≈1.0); stripping "everything after EOS" then does
  nothing and the full cap-length ramble is scored INVALID. So implement a real
  program-boundary extractor that, EVEN WHEN NO EOS IS PRESENT, walks the generation
  and returns the longest leading prefix that parses/executes as a complete program
  (e.g. last syntactically-valid statement / matching close / for CAD: up to the final
  successful Workplane op). Apply this in EVERY validity-eval path, not just one
  helper. Scoring the raw full-length generation is a bug that silently zeroes a model
  whose early output was fine.
- EVAL GENERATION EFFICIENCY (eval/proxy/RL generation dominates wall-clock when the
  model does not emit EOS — every sample runs to the cap; run-4 eval was ~90s/sample,
  ~75 min for 50): (1) cap eval/collection generation at ~p95 of completion length
  (NOT p95×1.3 — the boundary extractor already salvages the valid prefix, so a
  tighter cap costs almost no validity and is ~1.5× faster); (2) load the model for
  GENERATION/eval in fp16/bf16, not 4-bit — 4-bit dequantizes every token (~2-3×
  slower) and a <=3B model fits one card in fp16 anyway (keep 4-bit QLoRA for
  TRAINING only); (3) BATCH the eval generations (pad a batch of prompts and call
  generate once) instead of one sample at a time — 2-4× on a single GPU. Combined,
  these cut eval from ~90s to ~15-25s/sample.
- HARDWARE PLACEMENT: choose placement at RUNTIME with an explicit conditional — do
  NOT hardcode either single-GPU or 'auto'. Estimate the model's memory need (param
  count × bytes/param for the chosen dtype: 4-bit≈0.5, bf16/fp16≈2, fp32≈4; add ~20%
  for optimizer/activations/a reference copy in RL) and compare it to ONE visible
  device's free memory (torch.cuda.mem_get_info / get_device_properties). Then:
    * if it fits on one device → pin to a single GPU (device_map={'': 0}) — this is
      the common case for <=3B models in 4-bit/bf16 on a 24GB card;
    * only if it does NOT fit on one device → shard with device_map='auto'.
  Print the decision, e.g. `PLACEMENT: single_gpu est_gb=3.1 free_gb=23.6` or
  `PLACEMENT: sharded est_gb=41 free_gb=23.6`. Rationale: sharding a model that fits
  one card adds a cross-GPU transfer on every decode step, making autoregressive
  generation several times slower for zero benefit; but hardcoding single-GPU would
  OOM a model that genuinely needs sharding. The conditional handles both.
  CRITICAL when pinning single-GPU: also restrict the process to ONE visible GPU by
  setting `os.environ["CUDA_VISIBLE_DEVICES"]="0"` at the VERY TOP of the entry script
  (before `import torch` / transformers). Otherwise a HuggingFace/TRL Trainer still
  sees multiple GPUs (n_gpu>1) and auto-wraps the model in torch.nn.DataParallel,
  whose input scatter then crashes with "chunk expects at least a 1-dimensional
  tensor" (a 0-dim batch scalar cannot be split across GPUs). device_map={'': 0} pins
  the MODEL but does NOT stop the Trainer's DataParallel — only limiting visible
  devices does.
- REWARD DEAD-ZONE GUARD (RL only): if the reward is constant across ALL generations
  for ~20 consecutive steps (e.g. every sample scores -1), the policy gradient is
  zero and further steps are pure waste — print
  `RL_DEAD_REWARD: steps=<n> reward=<r>` and stop that run early instead of burning
  the remaining budget.
- NO trained condition may exceed ground-truth validity. A condition scoring >=
  ground_truth_validity (especially ==1.0 with std 0) is NOT a great result — it means
  the oracle is being gamed (a lenient check, or mode-collapsed output clearing a
  trivial bar). Treat it as a BUG (tighten the oracle / check output diversity), do NOT
  report it as a finding.
- MODE-COLLAPSE CHECK: if a condition emits near-identical output across different
  inputs/regimes (e.g. uniform 1.0 in every regime), print
  `WARNING: SUSPECT_OUTPUT condition=<name> reason=<...>`. Uniform perfection across
  heterogeneous inputs is a red flag, not a success.

OBSERVABILITY / INCREMENTAL PERSISTENCE (long runs are monitored from OUTSIDE the
sandbox through files in the working directory — results that only exist in memory
or stdout are invisible until the very end and are LOST on a crash):
- progress.json heartbeat: throughout training/eval, atomically update a small
  `progress.json` in the working directory (write `progress.json.tmp`, then
  `os.replace`) at least once per epoch AND roughly every minute of training, e.g.
  {"phase": "train", "condition": "<name>", "seed": 0, "step": 120, "total_steps": 500,
   "latest_metric": 0.41, "updated_at": "<iso8601>"}.
- partial_results.jsonl: the moment ONE (condition, seed) evaluation finishes,
  append its result as one JSON line ({"condition": ..., "seed": ..., "<metric>": ...})
  and flush. Never accumulate all results only in memory until the final write.
- Key checkpoint values (e.g. the ORACLE_CALIBRATION line) must ALSO be recorded in
  progress.json (e.g. under a "calibration" key), not only printed.
- These files are IN ADDITION to stdout prints and the final results.json, never a
  replacement.
- EVERY long loop must emit progress — not just training. Eval, proxy-data
  collection, and RL rollout loops MUST also update progress.json AND print a
  line every ~10 items (e.g. `[eval <cond>] 10/50 valid_so_far=3`). A 50-sample
  eval that prints nothing for an hour is an unobservable black box from outside
  the sandbox (run-4: SFT eval ran ~75 min totally silent — the operator could
  only guess from GPU/CPU whether it was alive or hung).
- EXCEPTION HANDLERS MUST PRINT THE TRACEBACK: any `except` that records/handles
  a failure must call `traceback.print_exc()` (or log exc_info), never just
  `print(e)` / the message alone. A swallowed traceback turned a one-line
  "chunk expects at least a 1-dimensional tensor" into a forced diagnostic rerun
  before the real cause (DataParallel) was visible. Note tracebacks go to stderr.

CHECKPOINT REUSE (CRITICAL — a silent path mismatch retrains for hours and starves
later conditions out of the time budget):
- When one condition reuses another condition's trained checkpoint (e.g. RL/GRPO
  starting from an SFT model), the save path and the load path MUST come from the
  SAME single helper function or module-level constant (e.g.
  `def sft_ckpt_dir(seed): ...`) — never two hand-written string literals.
- At the start of every condition/seed, print exactly ONE of:
  `CHECKPOINT_REUSE: condition=<name> seed=<n> path=<path>` or
  `CHECKPOINT_RETRAIN: condition=<name> seed=<n> reason=<why>` (and record it in
  progress.json). If the experiment plan says the condition builds on a previous
  checkpoint, an unexpected RETRAIN is a BUG — fail loudly rather than silently
  retraining.

TIME-BUDGET GRACEFUL DEGRADATION (CRITICAL — a hard crash at the budget boundary
loses EVERY result computed before it):
- When the time guard fires mid-condition (e.g. data collection or training got
  only a partial batch), the condition must SKIP gracefully: print
  `CONDITION_SKIPPED: condition=<name> seed=<n> reason=time_budget`, record the
  skip in results, and move on to writing final outputs. NEVER let a partial
  state reach a bare `raise` (e.g. "too few samples" validation errors) — wrap
  budget-truncation paths so the run still ends with a complete results.json
  covering everything that DID finish.
- Long inner training loops (e.g. a HuggingFace Trainer call that runs for
  hours) MUST also respect the time budget — add a step/epoch callback that
  checks remaining time, or cap max_steps from the remaining budget BEFORE
  starting. Checking the budget only between conditions is NOT enough: one
  unguarded multi-hour call can blow straight through the stop line.
- STEPS ARE OPTIMIZER STEPS, NOT FORWARD PASSES: when you set HF Trainer
  `max_steps` (or compute it for N epochs), count OPTIMIZER steps, which already
  divide by gradient_accumulation_steps. One optimizer step processes
  per_device_train_batch_size × gradient_accumulation_steps × n_gpu samples. So
  `steps_per_epoch = ceil(num_samples / (batch × grad_accum × n_gpu))` and
  `max_steps = epochs × steps_per_epoch`. Do NOT use num_samples/batch (ignoring
  grad_accum) — that over-counts steps by the grad_accum factor and silently
  trains for grad_accum× too many epochs (run-4: a "3-epoch" SFT ran 24 epochs,
  4h instead of ~30min, and over-fit). Also: max_steps OVERRIDES num_train_epochs
  in HF Trainer, so if you pass both, the epoch count you intend must already be
  baked into max_steps by the formula above.
- RUNTIME ETA CALIBRATION: planned step counts (RL steps, eval set sizes,
  collection sample counts) are guesses until measured on THIS hardware. After
  the first few steps/samples of each long loop, measure the per-step time,
  print `ETA_CALIBRATION: phase=<name> sec_per_step=<s> projected_total=<s>`,
  and if the projection exceeds that phase's share of the remaining budget,
  SCALE DOWN the step/sample count proportionally (and print the new count)
  rather than running blind into the stop line.
- Write/refresh results.json incrementally (after EVERY condition completes),
  so even a hard kill preserves all finished conditions.

Your task:
1. Design the file structure (main.py is the required entry point).
2. Implement ALL files with complete, runnable code. No placeholders or TODOs.
3. main.py must be the entry point and print the primary metric as:
   {metric}: <value>
4. Include numerical stability guards (gradient clipping, NaN detection, etc.).
5. Use the EXACT seed count from the experiment plan's compute_budget (each condition's
   `seeds:` field) — do NOT default to 3 seeds. If the plan says 2 seeds, use seeds [0, 1];
   if 1, use [0]. Report mean ± std over exactly those seeds.
6. Each ablation/condition MUST be genuinely different — not copy-paste with a renamed variable.
7. Implement a time guard: stop gracefully at 80% of the time budget ({time_budget_sec} seconds).
   But DO NOT UNDER-TRAIN to save budget: a model trained too few epochs never learns to
   emit EOS and never converges, so it rambles to the token cap on every sample
   (eos_frac≈0) and scores ~0 validity — wasting the whole run. Use enough epochs to
   converge (>=3 is typical for SFT at these sizes), and follow any epoch count the plan
   specifies. If budget is tight, cut eval/collection SAMPLE COUNTS or drop a CONDITION,
   never shave training epochs below convergence.
8. Write requirements.txt listing ONLY the top-level packages your code directly imports
   (e.g. cadquery, transformers, trl, peft). Do NOT pin or declare transitive dependencies
   that a top-level package installs automatically (e.g. do NOT add OCP / OCP.Core for
   cadquery, or nvidia-* CUDA wheels for torch) — guessing their version numbers (e.g.
   `OCP>=7.7`) breaks `pip install` because that version does not exist on PyPI. List only
   what you `import` directly, at conservative minimum versions.
9. If the experiment needs dataset downloads, write a setup.py that downloads
   the REAL declared dataset (no synthetic fallback).

IMPORTANT CONSTRAINTS:
- The code will run in the experiment environment configured in the project config: {env_description}
- Do NOT use argparse or CLI arguments — hardcode all configuration.
- All metric/log output must go to stdout (print statements), in addition to the
  observability files (progress.json / partial_results.jsonl / results.json).
- Keep the experiment feasible within {time_budget_sec} seconds total.
- SMOKE-TEST HOOK: at the very top of main(), check `if os.environ.get("ARC_SMOKE"):`
  and shrink EVERY cost knob to the minimum that still exercises each code path —
  all conditions but seeds=[0] only, train steps/epochs → 1, eval/proxy/collection
  sample counts → 2, RL steps → 1, dataset subset → a few rows. The goal: run ALL
  conditions end-to-end (train 1 step, eval 2 samples, GRPO 1 step, oracle) in 1-2
  minutes so a pre-flight catches runtime crashes (import errors, NoneType, shape
  mismatches, Trainer misconfig) before the multi-hour real run. ARC_SMOKE must
  still touch every condition and the real tools/oracle — it only shrinks sizes.
"""


def _render_mega_prompt(
    metric: str, time_budget_sec: int, env_description: str
) -> str:
    """Render the beast-mode prompt.

    Uses ``str.replace`` (not ``.format``) so a metric containing curly braces
    like ``F{1}`` does not raise KeyError.
    """
    return (
        _MEGA_PROMPT_TEMPLATE
        .replace("{metric}", metric)
        .replace("{time_budget_sec}", str(time_budget_sec))
        .replace("{env_description}", env_description)
    )


class OpenCodeBridge:
    """Manages OpenCode CLI invocations for beast mode code generation."""

    def __init__(
        self,
        *,
        model: str = "",
        llm_base_url: str = "",
        api_key_env: str = "",
        llm_provider: str = "openai-compatible",
        timeout_sec: int = 600,
        max_retries: int = 1,
        workspace_cleanup: bool = True,
    ) -> None:
        self._model = model
        self._llm_base_url = llm_base_url
        self._api_key_env = api_key_env
        self._llm_provider = llm_provider
        self._timeout_sec = timeout_sec
        self._max_retries = max_retries
        self._workspace_cleanup = workspace_cleanup

    # -- availability check ---------------------------------------------------

    @staticmethod
    def check_available() -> bool:
        """Return True if the ``opencode`` CLI is installed and callable."""
        opencode_cmd = shutil.which("opencode")
        if not opencode_cmd:
            return False
            
        try:
            result = subprocess.run(
                [opencode_cmd, "--version"],
                capture_output=True,
                text=True,
                timeout=15,
            )
            return result.returncode == 0
        except FileNotFoundError:
            return False
        except subprocess.TimeoutExpired:
            return False
        except Exception:  # noqa: BLE001
            return False

    # -- workspace preparation ------------------------------------------------

    def _prepare_workspace(
        self,
        stage_dir: Path,
        topic: str,
        exp_plan: str,
        metric: str,
        pkg_hint: str,
        extra_guidance: str,
        time_budget_sec: int,
    ) -> Path:
        """Create a temporary workspace directory with context files."""
        ws = stage_dir / f"opencode_beast_{int(time.time())}_{time.monotonic_ns() % 100000}"
        ws.mkdir(parents=True, exist_ok=True)

        # Write experiment plan
        (ws / "EXPERIMENT_PLAN.yaml").write_text(
            exp_plan or "# No experiment plan provided\n",
            encoding="utf-8",
        )

        # Write guidance document
        guidance_parts = [
            f"# Experiment Guidance\n",
            f"## Topic\n{topic}\n",
            f"## Primary Metric\n{metric}\n",
            f"## Time Budget\n{time_budget_sec} seconds\n",
        ]
        if pkg_hint:
            guidance_parts.append(f"## Environment\n{pkg_hint}\n")
        if extra_guidance:
            guidance_parts.append(f"## Additional Guidance\n{extra_guidance}\n")
        (ws / "GUIDANCE.md").write_text(
            "\n".join(guidance_parts), encoding="utf-8",
        )

        # Write opencode.json config
        opencode_cfg = self._build_opencode_config()
        (ws / "opencode.json").write_text(
            json.dumps(opencode_cfg, indent=2), encoding="utf-8",
        )

        # OpenCode requires a git repository — initialise one with
        # a single commit so that ``opencode run`` doesn't hang.
        # BUG-OB-01/OB-02: Check return codes and catch TimeoutExpired.
        try:
            r = subprocess.run(
                ["git", "init"],
                cwd=str(ws), capture_output=True, timeout=10,
            )
            if r.returncode != 0:
                raise OSError(f"git init failed: {r.stderr}")
            r = subprocess.run(
                ["git", "add", "-A"],
                cwd=str(ws), capture_output=True, timeout=10,
            )
            if r.returncode != 0:
                raise OSError(f"git add failed: {r.stderr}")
            r = subprocess.run(
                ["git", "-c", "user.email=beast@researchclaw",
                 "-c", "user.name=BeastMode",
                 "commit", "-m", "init workspace"],
                cwd=str(ws), capture_output=True, timeout=10,
            )
            if r.returncode != 0:
                raise OSError(f"git commit failed: {r.stderr}")
        except subprocess.TimeoutExpired as exc:
            raise OSError(f"git workspace init timed out: {exc}") from exc

        return ws

    def _is_azure(self) -> bool:
        """Detect Azure OpenAI from base URL or provider string."""
        return (
            "azure" in (self._llm_base_url or "").lower()
            or "azure" in (self._llm_provider or "").lower()
        )

    def _is_openrouter(self) -> bool:
        """Detect OpenRouter from base URL or provider string.

        OpenRouter must NOT be routed through opencode's "openai" provider:
        that provider speaks the OpenAI *Responses API* (``/responses``), and
        while OpenRouter accepts trivial single-turn Responses requests, its
        implementation rejects the multi-turn agentic payloads opencode emits
        (tool results / reasoning items) with ``Invalid Responses API request``
        — so beast mode writes zero files. opencode ships a *native* openrouter
        provider (chat/completions) that handles the full agentic loop, so we
        target that instead.
        """
        return (
            "openrouter" in (self._llm_provider or "").lower()
            or "openrouter" in (self._llm_base_url or "").lower()
        )

    def _build_opencode_config(self) -> dict[str, Any]:
        """Build the opencode.json configuration.

        Always uses the "openai" provider — this works for both standard
        OpenAI endpoints and Azure OpenAI (which accepts Bearer token auth
        on the ``/openai/v1`` path and now supports the Responses API).
        """
        cfg: dict[str, Any] = {
            "$schema": "https://opencode.ai/config.json",
        }

        if self._is_openrouter():
            # Native openrouter provider (chat/completions). Keep the full
            # vendor-prefixed model id — the native provider knows OpenRouter's
            # model catalog, so no per-model registration is needed.
            if self._model:
                cfg["model"] = f"openrouter/{self._model}"
            cfg["provider"] = {
                "openrouter": {
                    "options": {
                        "apiKey": f"{{env:{self._api_key_env}}}"
                        if self._api_key_env
                        else "",
                    },
                }
            }
            return cfg

        if self._llm_base_url:
            # The model is always registered under the "openai" provider keyed
            # by its basename (vendor prefixes like "anthropic/" are an upstream
            # routing convention, not an opencode provider). cfg["model"] must
            # therefore reference "openai/<basename>" — passing the raw vendor
            # prefix sends opencode to its built-in provider (needs that
            # vendor's own API key) and the call hangs/fails.
            if self._model:
                cfg["model"] = f"openai/{self._model.split('/')[-1]}"
            cfg["provider"] = {
                "openai": {
                    "options": {
                        "baseURL": self._llm_base_url,
                        "apiKey": f"{{env:{self._api_key_env}}}"
                        if self._api_key_env
                        else "",
                    },
                    "models": {},
                }
            }
            # Register the model so OpenCode knows it exists
            if self._model:
                model_name = self._model.split("/")[-1]
                cfg["provider"]["openai"]["models"] = {
                    model_name: {
                        "name": model_name,
                        "modalities": {
                            "input": ["text"],
                            "output": ["text"],
                        },
                    }
                }
        elif self._model:
            cfg["model"] = (
                self._model if "/" in self._model
                else f"openai/{self._model}"
            )

        return cfg

    # -- model resolution -------------------------------------------------------

    def _resolve_opencode_model(self) -> str:
        """Resolve the model identifier for OpenCode CLI's ``-m`` flag.

        Resolution order:
        1. No model → default "anthropic/claude-sonnet-4-6".
        2. OpenRouter → native "openrouter/<full-id>" provider (the openai
           Responses-API provider breaks on multi-turn agentic requests; see
           _is_openrouter). Keep the vendor prefix — OpenRouter needs the full
           "anthropic/claude-sonnet-4.6" model id.
        3. Other custom OpenAI-compatible endpoint (Azure/standard OpenAI) →
           the model is registered under the "openai" provider keyed by
           basename, so strip any vendor prefix and return "openai/<basename>"
           to match _build_opencode_config().
        4. No custom endpoint → honor an explicit provider prefix as-is (lets
           users target opencode's built-in providers), else "openai/{model}".
        """
        if not self._model:
            return "anthropic/claude-sonnet-4-6"
        if self._is_openrouter():
            return f"openrouter/{self._model}"
        if self._llm_base_url:
            return f"openai/{self._model.split('/')[-1]}"
        if "/" in self._model:
            return self._model
        return f"openai/{self._model}"

    # -- invocation ------------------------------------------------------------

    def _invoke_opencode(
        self,
        workspace: Path,
        prompt: str,
    ) -> tuple[bool, str, float]:
        """Run ``opencode run`` in the workspace. Returns (success, log, elapsed)."""
        env = os.environ.copy()
        # Pass API key via environment if configured. The env var name must
        # match the provider opencode resolves the model under: the native
        # openrouter provider reads OPENROUTER_API_KEY; the openai provider
        # (standard OpenAI / Azure via Bearer auth) reads OPENAI_API_KEY.
        if self._api_key_env:
            api_key = os.environ.get(self._api_key_env, "")
            if api_key:
                if self._is_openrouter():
                    env["OPENROUTER_API_KEY"] = api_key
                else:
                    env["OPENAI_API_KEY"] = api_key

        # Use -m flag to specify model (more reliable than opencode.json)
        resolved_model = self._resolve_opencode_model()
        opencode_cmd = shutil.which("opencode") or "opencode"
        cmd = [opencode_cmd, "run", "-m", resolved_model, "--format", "json", prompt]

        t0 = time.monotonic()
        try:
            result = subprocess.run(
                cmd,
                cwd=str(workspace),
                capture_output=True,
                text=True,
                encoding="utf-8",
                errors="replace",
                timeout=self._timeout_sec,
                env=env,
            )
            elapsed = time.monotonic() - t0
            log = result.stdout + "\n" + result.stderr
            return result.returncode == 0, log, elapsed
        except subprocess.TimeoutExpired as exc:
            elapsed = time.monotonic() - t0
            log = f"TIMEOUT after {elapsed:.1f}s"
            if exc.stdout:
                log += f"\nstdout: {exc.stdout[:2000] if isinstance(exc.stdout, str) else exc.stdout.decode(errors='replace')[:2000]}"
            return False, log, elapsed
        except FileNotFoundError:
            return False, "opencode CLI not found", 0.0
        except Exception as exc:  # noqa: BLE001
            elapsed = time.monotonic() - t0
            return False, f"Unexpected error: {exc}", elapsed

    # -- file collection -------------------------------------------------------

    @staticmethod
    def _collect_files(workspace: Path) -> dict[str, str]:
        """Collect generated Python files, requirements.txt, and setup.py.

        File names are flattened to basenames (e.g. ``src/main.py`` → ``main.py``)
        because the downstream executor expects a flat file dict.  If two files
        share the same basename, the one closer to the workspace root wins.
        """
        files: dict[str, str] = {}
        # Sort by depth (fewer parts first) so root-level files take priority
        py_files = sorted(
            workspace.rglob("*.py"),
            key=lambda p: len(p.relative_to(workspace).parts),
        )
        for py_file in py_files:
            rel = py_file.relative_to(workspace)
            parts = rel.parts
            if any(p.startswith("__pycache__") or p.startswith(".") for p in parts):
                continue
            # Flatten to basename — executor expects flat structure
            basename = rel.name
            if basename not in files:
                try:
                    files[basename] = py_file.read_text(encoding="utf-8", errors="replace")
                except OSError as exc:
                    logger.warning("Beast mode: failed to read %s: %s", py_file, exc)

        # Also collect requirements.txt and setup.py at root
        for extra in ("requirements.txt", "setup.py"):
            p = workspace / extra
            if p.exists() and extra not in files:
                files[extra] = p.read_text(encoding="utf-8", errors="replace")

        return files

    # -- entry-point validation ------------------------------------------------

    @staticmethod
    def _has_main_guard(source: str) -> bool:
        """Return True if *source* contains ``if __name__ == "__main__":``."""
        try:
            tree = ast.parse(source)
        except SyntaxError:
            return False
        for node in ast.walk(tree):
            if isinstance(node, ast.If):
                test = node.test
                if isinstance(test, ast.Compare) and isinstance(test.left, ast.Name):
                    if test.left.id == "__name__" and len(test.comparators) == 1:
                        comp = test.comparators[0]
                        if isinstance(comp, ast.Constant) and comp.value == "__main__":
                            return True
        return False

    @staticmethod
    def _ensure_main_entry_point(files: dict[str, str]) -> dict[str, str]:
        """Ensure ``main.py`` has an ``if __name__ == "__main__"`` guard.

        Beast Mode often generates multi-file projects where ``main.py`` is a
        library module and the real entry point lives in another file (e.g.
        ``run_experiment.py``).  Since the Docker sandbox always executes
        ``python3 main.py``, a library-only ``main.py`` exits immediately with
        no output.

        Strategy:
        1. If ``main.py`` already has the guard → return unchanged.
        2. Find the first other ``.py`` file that **does** have the guard.
        3. Swap: rename that file to ``main.py`` and the old ``main.py`` to a
           helper module (its original basename, or ``_lib.py``).
        4. If no file has a guard, append a minimal stub to ``main.py`` that
           calls the most likely entry function (``main()``, ``run()``, etc.).
        """
        main_code = files.get("main.py", "")
        if not main_code:
            return files

        if OpenCodeBridge._has_main_guard(main_code):
            return files

        # -- Strategy 2/3: find another file with the guard and swap -----------
        for fname, code in files.items():
            if fname == "main.py" or not fname.endswith(".py"):
                continue
            if OpenCodeBridge._has_main_guard(code):
                logger.info(
                    "Beast mode: main.py lacks __main__ guard; swapping "
                    "entry point with %s",
                    fname,
                )
                new_files = dict(files)
                # Rename original main.py → helper module
                helper_name = fname  # reuse the other file's name for old main
                new_files[helper_name] = main_code
                new_files["main.py"] = code
                return new_files

        # -- Strategy 4: inject a minimal entry point into main.py -------------
        # Look for common entry functions defined in main.py
        entry_func: str | None = None
        try:
            tree = ast.parse(main_code)
            candidates = [
                n.name
                for n in ast.walk(tree)
                if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef))
                and n.name in ("main", "run", "run_experiment", "train",
                               "run_experiments", "experiment", "run_all")
            ]
            if candidates:
                entry_func = candidates[0]
        except SyntaxError:
            pass

        if entry_func:
            logger.info(
                "Beast mode: main.py lacks __main__ guard; injecting call "
                "to %s()",
                entry_func,
            )
            new_files = dict(files)
            new_files["main.py"] = (
                main_code.rstrip()
                + "\n\n\nif __name__ == \"__main__\":\n"
                + f"    {entry_func}()\n"
            )
            return new_files

        logger.warning(
            "Beast mode: main.py lacks __main__ guard and no known entry "
            "function found — experiment may exit without producing output",
        )
        return files

    # -- main entry point ------------------------------------------------------

    def generate(
        self,
        stage_dir: Path,
        topic: str,
        exp_plan: str,
        metric: str,
        pkg_hint: str = "",
        extra_guidance: str = "",
        time_budget_sec: int = 300,
        env_description: str = "",
    ) -> OpenCodeResult:
        """Run OpenCode to generate experiment code.

        ``env_description`` describes the configured experiment environment the
        generated code will actually run in (mode, package availability, network
        policy). It is injected into the mega-prompt so the model targets the
        real environment instead of a hardcoded assumption.

        Returns an OpenCodeResult with success status and generated files.
        """
        if not env_description:
            env_description = (
                "the configured experiment environment — use only the packages "
                "listed in GUIDANCE.md; do not assume internet access or pip "
                "installs are available at runtime."
            )
        # Check availability first
        if not self.check_available():
            return OpenCodeResult(
                success=False,
                error="OpenCode CLI not installed or not callable",
            )

        workspace: Path | None = None
        last_error = ""

        for attempt in range(1 + self._max_retries):
            # Prepare workspace
            try:
                workspace = self._prepare_workspace(
                    stage_dir=stage_dir,
                    topic=topic,
                    exp_plan=exp_plan,
                    metric=metric,
                    pkg_hint=pkg_hint,
                    extra_guidance=extra_guidance,
                    time_budget_sec=time_budget_sec,
                )
            except OSError as exc:
                last_error = f"Failed to prepare workspace: {exc}"
                logger.warning("Beast mode: %s", last_error)
                continue

            # Build the mega-prompt
            prompt = _render_mega_prompt(metric, time_budget_sec, env_description)

            logger.info(
                "Beast mode: invoking OpenCode (attempt %d/%d, timeout=%ds)",
                attempt + 1,
                1 + self._max_retries,
                self._timeout_sec,
            )

            success, log, elapsed = self._invoke_opencode(workspace, prompt)

            if success:
                files = self._collect_files(workspace)
                if "main.py" not in files:
                    logger.warning(
                        "Beast mode: OpenCode succeeded but no main.py found "
                        "(files: %s)", list(files.keys()),
                    )
                    last_error = "No main.py in OpenCode output"
                    # Cleanup failed workspace
                    if self._workspace_cleanup and workspace.exists():
                        shutil.rmtree(workspace, ignore_errors=True)
                    continue

                # BUG-R52-01: Ensure main.py has an entry point
                files = self._ensure_main_entry_point(files)

                # Write log
                try:
                    (stage_dir / "opencode_log.txt").write_text(
                        log or "", encoding="utf-8",
                    )
                except OSError as _wexc:
                    logger.warning("Beast mode: failed to write log: %s", _wexc)

                # Cleanup workspace if configured
                if self._workspace_cleanup and workspace.exists():
                    shutil.rmtree(workspace, ignore_errors=True)

                return OpenCodeResult(
                    success=True,
                    files=files,
                    opencode_log=log,
                    elapsed_sec=elapsed,
                )

            last_error = log
            logger.warning(
                "Beast mode: OpenCode attempt %d failed (%.1fs): %s",
                attempt + 1,
                elapsed,
                log[:500],
            )
            # Cleanup failed workspace
            if self._workspace_cleanup and workspace and workspace.exists():
                shutil.rmtree(workspace, ignore_errors=True)

        # All attempts failed
        return OpenCodeResult(
            success=False,
            opencode_log=last_error,
            error=f"OpenCode failed after {1 + self._max_retries} attempt(s)",
        )


# ---------------------------------------------------------------------------
# Helper: count historical failures
# ---------------------------------------------------------------------------

def count_historical_failures(run_dir: Path, stage_name: str = "stage-10") -> int:
    """Count past Stage 10 failures from stage directories and logs.

    Each stage directory is counted at most once, even if multiple failure
    indicators are present.
    """
    failures = 0
    for d in run_dir.glob(f"{stage_name}*"):
        failed = False
        # Check for beast_mode_log.json
        bm_log = d / "beast_mode_log.json"
        if bm_log.exists():
            try:
                data = json.loads(bm_log.read_text(encoding="utf-8"))
                if not data.get("success", True):
                    failed = True
            except Exception:  # noqa: BLE001
                pass
        # Check for stage health failures
        if not failed:
            health = d / "stage_health.json"
            if health.exists():
                try:
                    data = json.loads(health.read_text(encoding="utf-8"))
                    if data.get("status") == "FAILED":
                        failed = True
                except Exception:  # noqa: BLE001
                    pass
        # Check for validation report with FAILED status
        if not failed:
            vr = d / "validation_report.md"
            if vr.exists():
                try:
                    content = vr.read_text(encoding="utf-8")
                    if "BLOCKED" in content or "FAILED" in content:
                        failed = True
                except Exception:  # noqa: BLE001
                    pass
        if failed:
            failures += 1
    return failures
