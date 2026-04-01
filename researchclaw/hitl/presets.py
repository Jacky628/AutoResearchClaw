"""HITL preset configurations for common use cases."""

from __future__ import annotations

from researchclaw.hitl.config import HITLConfig, StagePolicy


# ---------------------------------------------------------------------------
# Named presets
# ---------------------------------------------------------------------------

def copilot_preset() -> HITLConfig:
    """Co-pilot mode: deep collaboration at critical stages, auto elsewhere.

    Recommended for most users — balances quality with efficiency.
    """
    return HITLConfig.from_dict({
        "enabled": True,
        "mode": "co-pilot",
        "notifications": {
            "on_pause": True,
            "on_quality_drop": True,
            "on_error": True,
            "channels": ["terminal"],
        },
        "collaboration": {
            "max_chat_turns": 50,
            "save_chat_history": True,
        },
        "timeouts": {
            "default_human_timeout_sec": 86400,
            "auto_proceed_on_timeout": False,
        },
    })


def express_preset() -> HITLConfig:
    """Express mode: only 3 critical gates.

    For experienced users who want minimal interruption.
    """
    return HITLConfig.from_dict({
        "enabled": True,
        "mode": "custom",
        "stage_policies": {
            "8": {"require_approval": True},
            "9": {"require_approval": True, "allow_edit_output": True},
            "20": {"require_approval": True},
        },
        "timeouts": {
            "default_human_timeout_sec": 43200,
            "auto_proceed_on_timeout": False,
        },
    })


def thorough_preset() -> HITLConfig:
    """Thorough mode: review every phase boundary.

    For critical research where every stage matters.
    """
    return HITLConfig.from_dict({
        "enabled": True,
        "mode": "checkpoint",
        "notifications": {
            "on_pause": True,
            "on_quality_drop": True,
            "on_error": True,
            "channels": ["terminal"],
        },
        "collaboration": {
            "max_chat_turns": 100,
            "save_chat_history": True,
        },
    })


def learning_preset() -> HITLConfig:
    """Learning mode: pause at every stage for educational walkthrough.

    For new users learning the pipeline.
    """
    return HITLConfig.from_dict({
        "enabled": True,
        "mode": "step-by-step",
    })


def autonomous_preset() -> HITLConfig:
    """Autonomous mode: full auto with HITL disabled.

    Equivalent to the original pipeline behavior.
    """
    return HITLConfig(enabled=False, mode="full-auto")


# ---------------------------------------------------------------------------
# Preset registry
# ---------------------------------------------------------------------------

PRESETS: dict[str, callable] = {
    "co-pilot": copilot_preset,
    "copilot": copilot_preset,
    "express": express_preset,
    "thorough": thorough_preset,
    "learning": learning_preset,
    "autonomous": autonomous_preset,
    "full-auto": autonomous_preset,
}


def get_preset(name: str) -> HITLConfig | None:
    """Get a preset HITL config by name. Returns None if not found."""
    factory = PRESETS.get(name.lower())
    if factory is None:
        return None
    return factory()


def list_presets() -> list[str]:
    """Return available preset names."""
    return sorted(set(PRESETS.keys()))
