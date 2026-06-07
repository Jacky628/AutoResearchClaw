"""P6: context-aware security profiles for generated code.

strict (default) keeps the full bans; provisioned/setup relax download /
subprocess / exec needs (real experiments download data & run real tools) while
keeping a high-risk core banned in EVERY profile.
"""

from __future__ import annotations

import pytest

from researchclaw.experiment.validator import validate_code, validate_security


def _errs(code, profile):
    return [
        (i.message)
        for i in validate_security(code, profile).issues
        if i.severity == "error"
    ]


# -- relaxable in provisioned/setup, banned in strict --------------------------

RELAXABLE = [
    "import subprocess\nsubprocess.run(['ls'])\n",
    "import requests\nrequests.get('http://x')\n",
    "import shutil\nshutil.rmtree('/tmp/x')\n",
    "import urllib.request\n",
    "exec('import cadquery as cq; cq.Workplane(\"XY\").box(1,1,1)')\n",
    "import os\nos.unlink('/tmp/x')\n",
]


@pytest.mark.parametrize("code", RELAXABLE)
def test_relaxable_banned_in_strict(code):
    assert _errs(code, "strict"), f"expected strict to ban: {code!r}"


@pytest.mark.parametrize("code", RELAXABLE)
def test_relaxable_allowed_in_provisioned(code):
    assert _errs(code, "provisioned") == [], f"provisioned should allow: {code!r}"


@pytest.mark.parametrize("code", RELAXABLE)
def test_relaxable_allowed_in_setup(code):
    assert _errs(code, "setup") == [], f"setup should allow: {code!r}"


# -- always banned in every profile -------------------------------------------

ALWAYS_BANNED = [
    "import os\nos.system('rm -rf /')\n",
    "eval('1+1')\n",
    "import socket\n",
    "import ctypes\n",
    "__import__('os')\n",
]


@pytest.mark.parametrize("code", ALWAYS_BANNED)
@pytest.mark.parametrize("profile", ["strict", "provisioned", "setup"])
def test_always_banned_in_all_profiles(code, profile):
    assert _errs(code, profile), f"{profile} must still ban: {code!r}"


# -- default is strict (back-compat) ------------------------------------------

def test_default_profile_is_strict():
    assert _errs("import subprocess\n", "strict")  # sanity
    # validate_security default arg
    assert any(
        i.severity == "error"
        for i in validate_security("import subprocess\n").issues
    )


def test_validate_code_threads_profile():
    code = "import requests\nimport cadquery\nx=1\n"
    assert not validate_code(code).ok  # strict default → requests banned
    assert validate_code(code, security_profile="provisioned").ok
    assert validate_code(code, security_profile="setup").ok


def test_setup_download_script_passes():
    # representative of the real generated setup.py (HF + GitHub-release fallback)
    setup_code = (
        "import os, shutil, requests\n"
        "from datasets import load_dataset\n"
        "def main():\n"
        "    try:\n"
        "        ds = load_dataset('wuchy143/deepcad')\n"
        "    except Exception:\n"
        "        r = requests.get('https://github.com/.../release.zip')\n"
        "        open('d.zip','wb').write(r.content)\n"
        "        shutil.unpack_archive('d.zip')\n"
        "main()\n"
    )
    assert validate_code(setup_code, security_profile="setup").ok


def test_cadquery_oracle_subprocess_passes_provisioned():
    oracle = (
        "import subprocess, json\n"
        "def oracle(seq):\n"
        "    code = 'import cadquery as cq\\nr = cq.Workplane(\"XY\").box(1,1,1)'\n"
        "    p = subprocess.run(['python','-c',code], capture_output=True)\n"
        "    return p.returncode == 0\n"
    )
    assert validate_code(oracle, security_profile="provisioned").ok
    assert not validate_code(oracle).ok  # strict blocks subprocess
