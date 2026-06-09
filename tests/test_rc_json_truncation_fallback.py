"""json_mode truncation fallback for Stage 11 (resource_planning) & 20 (quality_gate).

These stages call the LLM in json_mode with no max_tokens, so a long response is
truncated by the provider default (~4096) → unclosed ```json fence →
`_safe_json_loads` returns `{}`. Because `{}` is a dict (just empty), the old
`if isinstance(parsed, dict): use it` accepted it and the `is None` fallback
guard never fired → an empty/degraded artifact was written silently. Fix: require
a NON-EMPTY dict + warn + bump max_tokens. These tests lock the fallback.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from researchclaw.adapters import AdapterBundle
from researchclaw.config import RCConfig
from researchclaw.llm.client import LLMResponse
from researchclaw.prompts import PromptManager
from researchclaw.pipeline.stage_impls._execution import _execute_resource_planning


class _FakeLLM:
    def __init__(self, content: str):
        self._content = content

    def chat(self, messages, **kwargs):
        return LLMResponse(content=self._content, model="fake-model")


def _config(tmp_path: Path) -> RCConfig:
    return RCConfig.from_dict(
        {
            "project": {"name": "rc-test", "mode": "docs-first"},
            "research": {"topic": "t", "domains": ["ml"], "quality_threshold": 8.0},
            "runtime": {"timezone": "UTC"},
            "notifications": {"channel": "local"},
            "knowledge_base": {"backend": "markdown", "root": str(tmp_path / "kb")},
            "llm": {"provider": "openai-compatible", "base_url": "http://x/v1",
                    "api_key_env": "RC_TEST_KEY", "api_key": "k",
                    "primary_model": "fake-model", "fallback_models": []},
            "experiment": {"mode": "sandbox"},
        },
        project_root=tmp_path,
        check_paths=False,
    )


def _run_rp(tmp_path: Path, content: str) -> dict:
    run_dir = tmp_path / "run"
    stage_dir = run_dir / "stage-11"
    stage_dir.mkdir(parents=True)
    _execute_resource_planning(
        stage_dir, run_dir, _config(tmp_path), AdapterBundle(),
        llm=_FakeLLM(content), prompts=PromptManager(),
    )
    return json.loads((stage_dir / "schedule.json").read_text())


def test_empty_response_falls_back_to_template(tmp_path):
    # "" → _safe_json_loads → {} → must NOT be accepted; template used instead.
    sched = _run_rp(tmp_path, "")
    tasks = sched.get("tasks", [])
    assert tasks and any(t.get("id") == "baseline" for t in tasks)


def test_truncated_unclosed_fence_falls_back(tmp_path):
    # an unclosed ```json fence (the real truncation symptom) → {} → fallback
    sched = _run_rp(tmp_path, '```json\n{"tasks": [{"id": "real_task"')
    tasks = sched.get("tasks", [])
    assert tasks and any(t.get("id") == "baseline" for t in tasks)


def test_valid_json_still_used(tmp_path):
    # happy path must survive the `and parsed` guard
    sched = _run_rp(tmp_path, '{"tasks": [{"id": "custom"}], "total_gpu_budget": 4}')
    assert sched.get("total_gpu_budget") == 4
    assert any(t.get("id") == "custom" for t in sched.get("tasks", []))
