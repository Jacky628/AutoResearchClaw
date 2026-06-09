"""validate_json_literals: flag bare null/true/false (JSON literals) in Python.

Regression for the Stage-12 acceptance root cause: a generated CadQuery eval
harness printed `{"valid": True, "error": null}` — `null` is a NameError at
runtime, so EVERY valid solid was caught by the except branch and reported
invalid → the experiment's primary metric was silently 0.0. This static check
catches that class at Stage 10 (codegen) so it gets repaired before running.
"""

from __future__ import annotations

from researchclaw.experiment.validator import validate_code, validate_json_literals


def _errs(code):
    return [i for i in validate_json_literals(code).issues if i.severity == "error"]


def test_flags_bare_null():
    errs = _errs("import json\nx = json.dumps({'error': null})\n")
    assert errs and "null" in errs[0].message and "None" in errs[0].message


def test_flags_true_false():
    assert _errs("a = true\n")
    assert _errs("b = false\n")


def test_clean_python_passes():
    assert not _errs("x = None\ny = True\nz = False\nimport json\njson.dumps({'e': None})\n")


def test_allows_explicitly_bound_shim():
    # an explicit `null = None` shim makes later use of `null` legitimate
    assert not _errs("null = None\nimport json\njson.dumps({'error': null})\n")


def test_replays_the_eval_harness_bug():
    # the exact pattern from the broken generated cadquery_eval.py
    code = (
        "import json, sys\n"
        "def report(result):\n"
        "    if result is not None:\n"
        "        print(json.dumps({'valid': True, 'error': null}))\n"
        "        sys.exit(0)\n"
    )
    errs = _errs(code)
    assert errs and errs[0].category == "syntax"


def test_wired_into_validate_code():
    v = validate_code("import json\nprint(json.dumps({'error': null}))\n")
    assert not v.ok
    assert any("JSON literal" in i.message for i in v.issues)
