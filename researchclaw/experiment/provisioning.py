"""Sandbox dependency provisioning (P2).

Local subprocess sandbox runs the generated experiment in the pre-existing
Python environment with no setup step — so anything the code needs that is not
already installed fails (or, worse, the code is written to fake it). This module
adds an opt-in *setup phase* for sandbox mode, mirroring the docker sandbox:

  Phase 0: build a per-run venv (``--system-site-packages`` so the base env's
           heavy packages like torch/transformers are inherited) and
           ``pip install -r requirements.txt`` into it.
  Phase 1: run ``setup.py`` (dataset downloads / label generation).

The venv overlay keeps the base environment untouched (non-destructive). System
libraries that need apt/conda (e.g. cadquery's OpenGL backend) are NOT installed
here — they are an operator/image responsibility; surface them via the Stage-9
environment manifest instead.

Enabled only when ``SandboxConfig.network_policy != "none"`` (default off).
"""

from __future__ import annotations

import logging
import re
import subprocess
from dataclasses import dataclass, field
from pathlib import Path

logger = logging.getLogger(__name__)

# pip emits these when a requirement cannot be resolved on the index:
#   "ERROR: Could not find a version that satisfies the requirement OCP>=7.7 ..."
#   "ERROR: No matching distribution found for OCP>=7.7"
_PIP_UNRESOLVABLE_RE = re.compile(
    r"(?:satisfies the requirement|No matching distribution found for)\s+"
    r"([A-Za-z0-9][A-Za-z0-9._-]*)",
)


def _normalize_pkg(name: str) -> str:
    """PyPI-normalize a distribution name (case-insensitive, -/_/. fold)."""
    return re.sub(r"[-_.]+", "-", name.strip().lower())


def _req_pkg_name(line: str) -> str | None:
    """Extract the distribution name from one requirements.txt line, or None
    for blanks/comments/options."""
    s = line.split("#", 1)[0].strip()
    if not s or s.startswith("-"):
        return None
    m = re.match(r"[A-Za-z0-9][A-Za-z0-9._-]*", s)
    return m.group(0) if m else None


def _unresolvable_pkgs(pip_output: str) -> set[str]:
    """Normalized names of requirements pip reported as non-existent."""
    return {_normalize_pkg(m) for m in _PIP_UNRESOLVABLE_RE.findall(pip_output or "")}

# Policies that perform pip install (requirements.txt).
_PIP_POLICIES = {"pip_only", "setup_only", "full"}
# Policies that run setup.py (dataset downloads).
_SETUP_POLICIES = {"setup_only", "full"}


@dataclass
class ProvisionResult:
    """Outcome of a provisioning attempt.

    ``python_path`` is the interpreter the experiment should run with — the
    per-run venv when one was built, else the unchanged base interpreter.
    """

    python_path: str
    venv_created: bool = False
    pip_status: str = "skipped"     # skipped | ok | failed | not_needed
    setup_status: str = "skipped"   # skipped | ok | failed | not_needed
    log: str = ""
    errors: list[str] = field(default_factory=list)

    @property
    def ok(self) -> bool:
        """True if nothing that was attempted failed."""
        return self.pip_status != "failed" and self.setup_status != "failed"


def _venv_python(venv_dir: Path) -> Path:
    """Return the python interpreter path inside *venv_dir* (POSIX/Windows)."""
    win = venv_dir / "Scripts" / "python.exe"
    return win if win.exists() else venv_dir / "bin" / "python"


def provision_project(
    project_dir: Path,
    base_python: str,
    *,
    network_policy: str,
    timeout_sec: int = 900,
) -> ProvisionResult:
    """Provision dependencies for the experiment in *project_dir*.

    Returns a :class:`ProvisionResult`; never raises — provisioning failures are
    captured so the caller can decide whether to run anyway (the experiment may
    still work with the base env, or fail loudly, which beats silent faking).
    """
    if network_policy == "none":
        return ProvisionResult(python_path=base_python, log="provisioning disabled (network_policy=none)")

    do_pip = network_policy in _PIP_POLICIES
    do_setup = network_policy in _SETUP_POLICIES
    req = project_dir / "requirements.txt"
    setup = project_dir / "setup.py"
    logs: list[str] = []
    errors: list[str] = []
    result = ProvisionResult(python_path=base_python)

    # -- Phase 0a: build per-run venv (overlay on base env) --------------------
    venv_dir = project_dir / ".venv"
    py = base_python
    try:
        completed = subprocess.run(
            [base_python, "-m", "venv", "--system-site-packages", str(venv_dir)],
            capture_output=True, text=True, timeout=300,
        )
        if completed.returncode == 0 and _venv_python(venv_dir).exists():
            py = str(_venv_python(venv_dir))
            result.venv_created = True
            result.python_path = py
            logs.append(f"[venv] created at {venv_dir}")
        else:
            errors.append("venv creation failed; using base interpreter")
            logs.append(f"[venv] FAILED rc={completed.returncode}: {(completed.stderr or '')[:300]}")
    except (OSError, subprocess.TimeoutExpired) as exc:
        errors.append(f"venv creation error: {exc}")
        logs.append(f"[venv] ERROR: {exc}")

    # -- Phase 0b: pip install -r requirements.txt -----------------------------
    if do_pip and req.is_file():
        # Defense-in-depth: a generated requirements.txt sometimes pins an
        # invented transitive dependency (run-4 saw `OCP>=7.7`, which does not
        # exist on PyPI) that fails the WHOLE install. On such a failure, drop
        # the package(s) pip reports as unresolvable and retry, so a real top-
        # level dep set still installs. Bounded retries in case several surface
        # one at a time.
        active_req = req
        dropped_all: list[str] = []
        try:
            attempt = 0
            while True:
                attempt += 1
                completed = subprocess.run(
                    [py, "-m", "pip", "install", "--no-input", "-r", str(active_req)],
                    capture_output=True, text=True, timeout=timeout_sec, cwd=str(project_dir),
                )
                out = (completed.stdout or "") + (completed.stderr or "")
                tail = (completed.stdout or "")[-500:] + (completed.stderr or "")[-500:]
                if completed.returncode == 0:
                    result.pip_status = "ok"
                    if dropped_all:
                        logs.append(
                            "[pip] requirements.txt installed after dropping "
                            f"unresolvable: {', '.join(dropped_all)}"
                        )
                        errors.append(
                            "pip: dropped non-existent requirement(s) "
                            f"{', '.join(dropped_all)} (likely invented/transitive)"
                        )
                    else:
                        logs.append("[pip] requirements.txt installed")
                    break

                bad = _unresolvable_pkgs(out)
                if not bad or attempt > 3:
                    result.pip_status = "failed"
                    errors.append("pip install failed")
                    logs.append(f"[pip] FAILED rc={completed.returncode}: {tail[-500:]}")
                    break

                # Rewrite a filtered requirements file without the bad packages.
                kept: list[str] = []
                removed: list[str] = []
                for line in active_req.read_text(encoding="utf-8").splitlines():
                    name = _req_pkg_name(line)
                    if name and _normalize_pkg(name) in bad:
                        removed.append(line.strip())
                    else:
                        kept.append(line)
                if not removed:
                    # pip blamed a package not present as a top-level line (e.g.
                    # a transitive dep) — cannot fix by filtering; stop.
                    result.pip_status = "failed"
                    errors.append("pip install failed (unresolvable transitive dep)")
                    logs.append(f"[pip] FAILED rc={completed.returncode}: {tail[-500:]}")
                    break
                dropped_all.extend(removed)
                logs.append(f"[pip] retry {attempt}: dropping unresolvable {removed}")
                active_req = project_dir / f".requirements.filtered{attempt}.txt"
                active_req.write_text("\n".join(kept) + "\n", encoding="utf-8")
        except (OSError, subprocess.TimeoutExpired) as exc:
            result.pip_status = "failed"
            errors.append(f"pip install error: {exc}")
            logs.append(f"[pip] ERROR: {exc}")
    elif do_pip:
        result.pip_status = "not_needed"
        logs.append("[pip] no requirements.txt")

    # -- Phase 1: setup.py (dataset downloads) ---------------------------------
    if do_setup and setup.is_file():
        try:
            completed = subprocess.run(
                [py, "setup.py"],
                capture_output=True, text=True, timeout=timeout_sec, cwd=str(project_dir),
            )
            tail = (completed.stdout or "")[-500:] + (completed.stderr or "")[-500:]
            if completed.returncode == 0:
                result.setup_status = "ok"
                logs.append("[setup] setup.py completed")
            else:
                result.setup_status = "failed"
                errors.append("setup.py failed")
                logs.append(f"[setup] FAILED rc={completed.returncode}: {tail[-500:]}")
        except (OSError, subprocess.TimeoutExpired) as exc:
            result.setup_status = "failed"
            errors.append(f"setup.py error: {exc}")
            logs.append(f"[setup] ERROR: {exc}")
    elif do_setup:
        result.setup_status = "not_needed"
        logs.append("[setup] no setup.py")

    result.log = "\n".join(logs)
    result.errors = errors
    if errors:
        logger.warning("Sandbox provisioning issues: %s", "; ".join(errors))
    return result
