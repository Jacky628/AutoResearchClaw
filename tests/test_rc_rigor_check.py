"""Academic-rigor check for generated code (P5)."""

from __future__ import annotations

from researchclaw.experiment.rigor_check import check_rigor
from researchclaw.pipeline.environment import parse_manifest


def _manifest(pip):
    return parse_manifest({"environment": {"pip": pip}})


CAD_PLAN = {
    "objectives": ["Test CadQuery execution validity of generated sequences"],
    "datasets": [{"name": "DeepCAD", "source": "hf://SadilKhan/DeepCAD"}],
}


def test_declared_lib_not_imported_is_violation():
    files = {
        "main.py": "import torch\n\ndef is_valid(ops):\n    return len(ops) >= 2\n",
        "setup.py": "print('setup')\n",
    }
    rep = check_rigor(files, CAD_PLAN, _manifest(["cadquery>=2.4", "transformers>=4.40"]))
    assert not rep.ok
    joined = " ".join(rep.violations)
    assert "cadquery" in joined
    assert "transformers" in joined


def test_real_use_passes():
    files = {
        "main.py": (
            "import cadquery as cq\nimport transformers\n"
            "from transformers import AutoModelForCausalLM\n"
            "def oracle(seq):\n    r = cq.Workplane('XY').box(1,1,1)\n    return r.val().isValid()\n"
        ),
    }
    rep = check_rigor(files, CAD_PLAN, _manifest(["cadquery>=2.4", "transformers>=4.40"]))
    assert rep.ok, rep.violations


def test_synthetic_fallback_flagged_when_not_declared():
    files = {"main.py": (
        "import torch\nimport cadquery\n"
        "def load():\n    try:\n        return real()\n    except Exception:\n"
        "        return _generate_synthetic(500)\n"
    )}
    rep = check_rigor(files, CAD_PLAN, _manifest(["cadquery"]))
    assert any("synthetic" in v.lower() for v in rep.violations)


def test_synthetic_violation_names_the_offending_file():
    files = {
        "main.py": "import cadquery\nx=1\n",
        "setup.py": "def generate_synthetic_deepcad_data():\n    return []\n",
    }
    rep = check_rigor(files, CAD_PLAN, _manifest(["cadquery"]))
    syn = [v for v in rep.violations if "synthetic" in v.lower()]
    assert syn and "setup.py" in syn[0]  # surgical: points at the right file


def test_synthetic_allowed_when_plan_declares_it():
    pde_plan = {"objectives": ["solve Burgers"],
                "datasets": [{"name": "synthetic Burgers PDE", "source": "generated"}]}
    files = {"main.py": "import torch\ndata = generate_synthetic(1000)\n"}
    rep = check_rigor(files, pde_plan, _manifest([]))
    # synthetic is the declared design here → no violation from synthetic marker
    assert all("synthetic" not in v.lower() for v in rep.violations)


def test_synthetic_fallback_in_plan_is_not_an_allowance():
    # A plan that declares a synthetic *fallback* must NOT license synthetic code.
    plan = {"objectives": ["real CAD"],
            "datasets": [{"name": "DeepCAD"}, {"name": "Synthetic CAD fallback"}]}
    files = {"main.py": "import cadquery\ndata = _generate_synthetic(100)\n"}
    rep = check_rigor(files, plan, _manifest(["cadquery"]))
    assert any("synthetic" in v.lower() for v in rep.violations)


def test_mimic_oracle_flagged():
    files = {"main.py": (
        "import cadquery\n"
        "def check_validity_oracle(ops):\n"
        "    # Mimics what CadQuery execution would do\n"
        "    return len(ops) >= 2\n"
    )}
    rep = check_rigor(files, CAD_PLAN, _manifest(["cadquery"]))
    assert any("rule-based stand-in" in v.lower() or "mimic" in v.lower() for v in rep.violations)


def test_no_python_no_violation():
    rep = check_rigor({"requirements.txt": "cadquery\n"}, CAD_PLAN, _manifest(["cadquery"]))
    assert rep.ok


def test_auxiliary_libs_not_required_to_import():
    # numpy/tqdm/jsonlines/scikit-learn declared but unused → NOT a violation.
    files = {"main.py": "import cadquery\nimport transformers\nx=1\n"}
    rep = check_rigor(files, CAD_PLAN,
                      _manifest(["cadquery", "transformers", "numpy", "tqdm", "scikit-learn", "jsonlines"]))
    assert rep.ok, rep.violations


# -- B: codegen prompts carry the rigor directives ----------------------------

def test_mega_prompt_has_rigor_directives():
    from researchclaw.pipeline.opencode_bridge import _MEGA_PROMPT_TEMPLATE
    t = _MEGA_PROMPT_TEMPLATE.lower()
    assert "academic rigor" in t
    assert "mimic" in t and "synthetic" in t
    assert "toy" in t  # do not replace declared LLM with a toy model
    assert "block" in t  # violations block the stage


def test_code_generation_prompt_has_rigor_directives():
    from researchclaw.prompts import PromptManager
    u = PromptManager()._stages["code_generation"]["user"].lower()
    assert "academic rigor" in u
    assert "synthetic" in u and "mimic" in u
