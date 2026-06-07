"""Academic-rigor check for generated experiment code (P5).

Stage 9 may correctly demand a rigorous design (real LLM fine-tuning, a real
CAD-kernel validity oracle, a real dataset), but Stage 10 code generation can
still quietly degrade it — produce a toy model with no `transformers`, a
rule-based "mimics what CadQuery would do" oracle, or a silent synthetic-data
fallback. Prompts alone do not prevent this (the generator already ignores
"no placeholders"). This module gives the pipeline *teeth*: it scans the
generated code against the plan/manifest and reports concrete rigor violations
so they can be repaired or block the stage — never silently shipped.

Detection is deliberately deterministic / low-false-positive: it keys off
declared dependencies that must actually be imported, and unambiguous
synthetic-fallback / mimic markers — not fuzzy semantic judgement.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Any

# Declared libraries that, if a plan declares them, the code MUST actually
# import and use — they ARE the scientific method (a CAD kernel, the LLM stack,
# a chem/bio/physics engine). Auxiliary libs (numpy, tqdm, requests, sklearn,
# jsonlines) are intentionally excluded to avoid false positives.
_MUST_USE_IF_DECLARED: frozenset[str] = frozenset({
    "cadquery", "OCP", "FreeCAD", "freecad",
    "transformers", "peft", "trl",
    "rdkit", "Bio", "ase", "pyscf", "openmm", "deepchem",
    "torch_geometric", "openbabel", "mdtraj",
})

# Unambiguous markers of fabricated-data fallback (the harmful pattern: try the
# real dataset, except → make it up). NOT triggered by plain random seeding.
_SYNTHETIC_MARKERS: tuple[str, ...] = (
    "generate_synthetic", "_generate_synthetic", "make_synthetic",
    "fallback to synthetic", "fall back to synthetic", "generating synthetic",
    "synthetic fallback", "dummy data", "mock data",
)

# Markers of a rule-based stand-in for a declared real evaluator/oracle.
_MIMIC_MARKERS: tuple[str, ...] = (
    "mimics what", "mimic what", "rule-based oracle", "rule based oracle",
    "approximates cadquery", "pretend", "stub oracle", "fake oracle",
)


@dataclass
class RigorReport:
    violations: list[str] = field(default_factory=list)

    @property
    def ok(self) -> bool:
        return not self.violations

    def as_markdown(self) -> str:
        lines = ["# ⛔ Academic-rigor violations (generated code degrades the design)\n"]
        lines.append(
            "The plan demands real tools/data but the generated code substitutes "
            "weaker stand-ins. Per academic-rigor-first this BLOCKS the stage — the "
            "experiment must use what the design declared.\n"
        )
        for v in self.violations:
            lines.append(f"- {v}")
        return "\n".join(lines) + "\n"

    def as_feedback(self) -> str:
        """Compact, imperative feedback to hand back to the code generator."""
        head = (
            "RIGOR VIOLATIONS — you MUST fix these by using the REAL tools/data the "
            "plan declares (no rule-based/synthetic/mimic substitutes):\n"
        )
        return head + "\n".join(f"{i+1}. {v}" for i, v in enumerate(self.violations))


def _all_python(files: dict[str, str]) -> str:
    return "\n".join(c for f, c in files.items() if f.endswith(".py"))


def _imported_modules(py: str) -> set[str]:
    mods: set[str] = set()
    for m in re.findall(r"^\s*(?:from|import)\s+([a-zA-Z_][\w.]*)", py, re.MULTILINE):
        top = m.split(".")[0]
        mods.add(top)
        mods.add(m)  # keep dotted form too (e.g. OCP.something)
    return mods


def _plan_text(plan: Any) -> str:
    try:
        import json
        return json.dumps(plan, default=str).lower()
    except Exception:  # noqa: BLE001
        return str(plan).lower()


def _plan_allows_synthetic(plan: Any, manifest: Any) -> bool:
    """Synthetic data is allowed ONLY when the design deliberately chose it
    (e.g. PDE/physics/toy domains) — never as an undeclared fallback. A dataset
    named/with source mentioning 'synthetic' but ALSO 'fallback' does not count.
    """
    txt = _plan_text(plan)
    if "synthetic" not in txt:
        return False
    # A declared synthetic *fallback* is exactly what we forbid.
    if "fallback" in txt or "placeholder" in txt:
        return False
    return True


def check_rigor(
    files: dict[str, str],
    plan: Any,
    manifest: Any | None = None,
) -> RigorReport:
    """Return rigor violations for *files* given the Stage-9 *plan* / *manifest*."""
    py = _all_python(files)
    if not py.strip():
        return RigorReport()  # nothing to check (no python yet)
    imports = _imported_modules(py)
    py_low = py.lower()
    violations: list[str] = []

    # 1. Declared core scientific libs that the code never imports.
    pip_reqs = getattr(manifest, "pip", ()) if manifest is not None else ()
    for req in pip_reqs:
        name = getattr(req, "import_name", None)
        spec = getattr(req, "spec", name)
        if name and name in _MUST_USE_IF_DECLARED and name not in imports:
            violations.append(
                f"Plan declares `{spec}` but no generated code imports `{name}` — "
                f"the experiment must ACTUALLY use it (e.g. real CAD-kernel "
                f"execution / real model), not a rule-based or synthetic substitute."
            )

    # 2. Undeclared synthetic-data fabrication / fallback.
    if not _plan_allows_synthetic(plan, manifest):
        for marker in _SYNTHETIC_MARKERS:
            if marker in py_low:
                violations.append(
                    f"Generated code fabricates data (`{marker}`) but the plan does "
                    f"not declare synthetic data. Use the declared real dataset and "
                    f"raise/FAIL if it cannot be obtained — do not silently synthesize."
                )
                break

    # 3. Rule-based 'mimic' stand-in for a declared real evaluator/oracle.
    for marker in _MIMIC_MARKERS:
        if marker in py_low:
            violations.append(
                f"Generated code uses a rule-based stand-in (`{marker}`) instead of "
                f"the real evaluator the plan requires. Execute the declared real "
                f"oracle/tool to compute the metric."
            )
            break

    return RigorReport(violations)
