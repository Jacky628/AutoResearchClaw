"""Environment manifest + resolution for the experiment pipeline (P1).

Stage 9 (experiment design) declares the runtime environment its plan needs —
pip packages, system libraries, datasets, compute — as an ``environment:`` block
in ``exp_plan.yaml``. This module parses that manifest and *resolves* it against
the real environment: each requirement is classified (already AVAILABLE, NEEDS
provisioning, or genuinely INSUFFICIENT) so feasibility is visible BEFORE code
generation / at the HITL gate.

P1 is report-only: nothing is installed or downloaded here. Provisioning (P2)
and the infeasibility→redesign loop (P3) build on this.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Any, Mapping

# -- status constants ---------------------------------------------------------

PKG_AVAILABLE = "AVAILABLE"
# Not pre-installed, but pip-installable → the P2 setup phase WILL install it.
# This is NOT a reason to degrade the design (academic-rigor-first).
PKG_WILL_INSTALL = "WILL_INSTALL"

DS_CACHED = "CACHED"
# Not cached, but has a source → the setup phase WILL download it.
DS_WILL_DOWNLOAD = "WILL_DOWNLOAD"

# OS/conda libs pip cannot install — need a human/operator with root. NOT
# infeasible: surfaced as an explicit "run these commands" action, so the
# scientifically-required tool is kept and the environment adapts to it.
SYS_NEEDS_OPERATOR = "NEEDS_OPERATOR"

COMPUTE_OK = "OK"
COMPUTE_INSUFFICIENT = "INSUFFICIENT"
COMPUTE_UNKNOWN = "UNKNOWN"


# -- pip name → import name ---------------------------------------------------

# Packages whose import name differs from their pip name. Anything not listed
# defaults to the pip name with '-' → '_'.
_PIP_TO_IMPORT: dict[str, str] = {
    "scikit-learn": "sklearn",
    "scikit-image": "skimage",
    "opencv-python": "cv2",
    "opencv-python-headless": "cv2",
    "pillow": "PIL",
    "pyyaml": "yaml",
    "biopython": "Bio",
    "python-dateutil": "dateutil",
    "huggingface-hub": "huggingface_hub",
    "sentence-transformers": "sentence_transformers",
    "stable-baselines3": "stable_baselines3",
    "pytorch-lightning": "pytorch_lightning",
    "scikit-optimize": "skopt",
}

_VERSION_SPLIT = re.compile(r"[<>=!~\[ @;]")


def pip_to_pip_name(spec: str) -> str:
    """Strip version/extra markers from a pip spec → bare distribution name."""
    head = _VERSION_SPLIT.split(spec.strip(), 1)[0]
    return head.strip().lower()


def pip_to_import(spec: str) -> str:
    """Best-effort import name for a pip spec (e.g. 'scikit-learn>=1.4' → 'sklearn')."""
    name = pip_to_pip_name(spec)
    if name in _PIP_TO_IMPORT:
        return _PIP_TO_IMPORT[name]
    return name.replace("-", "_")


# -- manifest dataclasses -----------------------------------------------------


@dataclass(frozen=True)
class PackageReq:
    spec: str            # original, e.g. "transformers>=4.40"
    pip_name: str        # "transformers"
    import_name: str     # "transformers"


@dataclass(frozen=True)
class DatasetReq:
    name: str
    source: str = ""
    size_gb: float | None = None
    cache: str | None = None


@dataclass(frozen=True)
class ComputeReq:
    gpu: str | None = None          # "required" | "optional" | None
    min_vram_gb: float | None = None
    gpus: int | None = None


@dataclass(frozen=True)
class EnvironmentManifest:
    pip: tuple[PackageReq, ...] = ()
    system: tuple[str, ...] = ()
    datasets: tuple[DatasetReq, ...] = ()
    compute: ComputeReq | None = None
    est_setup_sec: int | None = None
    declared: bool = False          # did the plan actually declare environment?

    @property
    def import_names(self) -> tuple[str, ...]:
        return tuple(p.import_name for p in self.pip)


def _as_list(value: Any) -> list:
    if value is None:
        return []
    if isinstance(value, (list, tuple)):
        return list(value)
    return [value]


def _as_float(value: Any) -> float | None:
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _as_int(value: Any) -> int | None:
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def parse_manifest(plan: Any) -> EnvironmentManifest:
    """Parse the ``environment:`` block from a Stage-9 plan dict.

    Tolerant of missing/partial/badly-typed input — returns an empty manifest
    (``declared=False``) when nothing usable is present.
    """
    if not isinstance(plan, Mapping):
        return EnvironmentManifest()
    env = plan.get("environment")
    if not isinstance(env, Mapping):
        return EnvironmentManifest()

    pkgs: list[PackageReq] = []
    for item in _as_list(env.get("pip")):
        spec = item if isinstance(item, str) else (
            item.get("spec") or item.get("name") if isinstance(item, Mapping) else None
        )
        if not spec or not str(spec).strip():
            continue
        spec = str(spec).strip()
        pkgs.append(
            PackageReq(spec=spec, pip_name=pip_to_pip_name(spec), import_name=pip_to_import(spec))
        )

    systems = [str(s).strip() for s in _as_list(env.get("system")) if str(s).strip()]

    datasets: list[DatasetReq] = []
    for item in _as_list(env.get("datasets")):
        if isinstance(item, str):
            if item.strip():
                datasets.append(DatasetReq(name=item.strip()))
        elif isinstance(item, Mapping):
            name = str(item.get("name", "")).strip()
            if not name:
                continue
            datasets.append(DatasetReq(
                name=name,
                source=str(item.get("source", "")).strip(),
                size_gb=_as_float(item.get("size_gb")),
                cache=(str(item.get("cache")).strip() if item.get("cache") else None),
            ))

    compute: ComputeReq | None = None
    craw = env.get("compute")
    if isinstance(craw, Mapping):
        compute = ComputeReq(
            gpu=(str(craw.get("gpu")).strip().lower() if craw.get("gpu") is not None else None),
            min_vram_gb=_as_float(craw.get("min_vram_gb")),
            gpus=_as_int(craw.get("gpus")),
        )

    return EnvironmentManifest(
        pip=tuple(pkgs),
        system=tuple(systems),
        datasets=tuple(datasets),
        compute=compute,
        est_setup_sec=_as_int(env.get("est_setup_sec")),
        declared=True,
    )


# -- resolution ---------------------------------------------------------------


@dataclass(frozen=True)
class PackageStatus:
    req: PackageReq
    status: str


@dataclass(frozen=True)
class DatasetStatus:
    req: DatasetReq
    status: str


@dataclass
class EnvironmentResolution:
    packages: list[PackageStatus] = field(default_factory=list)
    datasets: list[DatasetStatus] = field(default_factory=list)
    system: list[str] = field(default_factory=list)
    compute_status: str = COMPUTE_UNKNOWN
    compute_detail: str = ""
    declared: bool = False

    # -- derived rollups ---
    @property
    def needs_install(self) -> list[str]:
        """pip packages the setup phase will auto-install."""
        return [p.req.spec for p in self.packages if p.status == PKG_WILL_INSTALL]

    @property
    def needs_download(self) -> list[str]:
        """datasets the setup phase will auto-download."""
        return [d.req.name for d in self.datasets if d.status == DS_WILL_DOWNLOAD]

    @property
    def needs_operator(self) -> list[str]:
        """system libraries a human operator must install (pip cannot)."""
        return list(self.system)

    @property
    def compute_infeasible(self) -> bool:
        return self.compute_status == COMPUTE_INSUFFICIENT

    @property
    def runnable_as_is(self) -> bool:
        """True if it could run offline right now — nothing to install/download,
        no operator action, compute sufficient."""
        return (
            not self.needs_install
            and not self.needs_download
            and not self.needs_operator
            and self.compute_status != COMPUTE_INSUFFICIENT
        )

    @property
    def provisionable(self) -> bool:
        """True if the design CAN be made to run on this machine — pip installs
        and downloads are automatic, system libs need an operator but are still
        obtainable. Only insufficient compute makes it genuinely infeasible.

        Per academic-rigor-first: needing to install a package or prompt the
        operator is NEVER a reason to degrade the design — only truly
        unobtainable hardware is.
        """
        return self.compute_status != COMPUTE_INSUFFICIENT

    def operator_setup_lines(self) -> list[str]:
        """Shell commands a human/root operator should run to satisfy system-lib
        requirements pip cannot install. Best-effort apt mapping (declared name)."""
        lines: list[str] = []
        for lib in self.system:
            lines.append(f"sudo apt-get install -y {lib}")
        return lines

    def to_dict(self) -> dict[str, Any]:
        return {
            "declared": self.declared,
            "runnable_as_is": self.runnable_as_is,
            "provisionable": self.provisionable,
            "packages": [
                {"spec": p.req.spec, "import_name": p.req.import_name, "status": p.status}
                for p in self.packages
            ],
            "datasets": [
                {"name": d.req.name, "source": d.req.source,
                 "size_gb": d.req.size_gb, "status": d.status}
                for d in self.datasets
            ],
            "system": self.system,
            "compute": {"status": self.compute_status, "detail": self.compute_detail},
            "needs_install": self.needs_install,
            "needs_download": self.needs_download,
            "needs_operator": self.needs_operator,
            "operator_setup": self.operator_setup_lines(),
        }

    def summary_md(self) -> str:
        if not self.declared:
            return (
                "### Environment resolution\n"
                "_No `environment:` block declared in the plan — feasibility not "
                "assessed. Consider declaring required packages/datasets/compute._\n"
            )
        lines = ["### Environment resolution\n"]
        verdict = (
            "✅ runnable as-is (offline)" if self.runnable_as_is
            else ("⚙️ provisionable (auto-install/download"
                  + ("; operator action needed for system libs" if self.needs_operator else "")
                  + ")" if self.provisionable
                  else "❌ INFEASIBLE on this hardware")
        )
        lines.append(f"**Verdict:** {verdict}\n")
        if self.packages:
            lines.append("**Packages:**")
            for p in self.packages:
                mark = "✓ installed" if p.status == PKG_AVAILABLE else "⤓ auto-install"
                lines.append(f"- `{p.req.spec}` → {mark}")
        if self.datasets:
            lines.append("\n**Datasets:**")
            for d in self.datasets:
                mark = "✓ cached" if d.status == DS_CACHED else "↓ auto-download"
                sz = f" (~{d.req.size_gb} GB)" if d.req.size_gb else ""
                lines.append(f"- {d.req.name}{sz} → {mark}")
        if self.system:
            lines.append("\n**System libs — OPERATOR must install (pip cannot):**")
            for cmd in self.operator_setup_lines():
                lines.append(f"- `{cmd}`")
        lines.append(f"\n**Compute:** {self.compute_status}"
                     + (f" — {self.compute_detail}" if self.compute_detail else ""))
        return "\n".join(lines) + "\n"


def _resolve_compute(
    compute: ComputeReq | None, hw_profile: Mapping[str, Any] | None
) -> tuple[str, str]:
    if compute is None:
        return COMPUTE_UNKNOWN, "no compute requirement declared"
    if not hw_profile:
        return COMPUTE_UNKNOWN, "hardware profile unavailable"

    has_gpu = bool(hw_profile.get("has_gpu"))
    gpu_required = compute.gpu == "required"
    if gpu_required and not has_gpu:
        return COMPUTE_INSUFFICIENT, "GPU required but none detected"

    detail_bits: list[str] = []
    if compute.min_vram_gb is not None and has_gpu:
        per_card_mb = hw_profile.get("vram_mb") or 0
        per_card_gb = per_card_mb / 1024 if per_card_mb else 0
        if per_card_gb and compute.min_vram_gb > per_card_gb + 0.5:
            return (
                COMPUTE_INSUFFICIENT,
                f"needs {compute.min_vram_gb} GB/card but only "
                f"{per_card_gb:.0f} GB available",
            )
        detail_bits.append(f"{per_card_gb:.0f} GB/card available")
    if compute.gpus is not None and has_gpu:
        have = int(hw_profile.get("gpu_count") or 1)
        if compute.gpus > have:
            detail_bits.append(f"wants {compute.gpus} GPUs, have {have} (will scale down)")
        else:
            detail_bits.append(f"{have} GPU(s) available")
    return COMPUTE_OK, "; ".join(detail_bits)


def resolve_environment(
    manifest: EnvironmentManifest,
    installed: Mapping[str, bool] | None,
    *,
    hw_profile: Mapping[str, Any] | None = None,
    cached_datasets: set[str] | None = None,
) -> EnvironmentResolution:
    """Classify each manifest requirement against the real environment.

    ``installed`` maps import-name → bool (from probing the runtime interpreter);
    ``cached_datasets`` is a set of lowercased dataset-name tokens found on disk.
    Report-only: nothing is installed or downloaded.
    """
    installed = installed or {}
    cached = {c.lower() for c in (cached_datasets or set())}

    pkgs: list[PackageStatus] = []
    for req in manifest.pip:
        ok = bool(installed.get(req.import_name, False))
        pkgs.append(PackageStatus(req, PKG_AVAILABLE if ok else PKG_WILL_INSTALL))

    dss: list[DatasetStatus] = []
    for req in manifest.datasets:
        token = req.name.lower()
        is_cached = any(token in c or c in token for c in cached) if cached else False
        dss.append(DatasetStatus(req, DS_CACHED if is_cached else DS_WILL_DOWNLOAD))

    compute_status, compute_detail = _resolve_compute(manifest.compute, hw_profile)

    return EnvironmentResolution(
        packages=pkgs,
        datasets=dss,
        system=list(manifest.system),
        compute_status=compute_status,
        compute_detail=compute_detail,
        declared=manifest.declared,
    )
