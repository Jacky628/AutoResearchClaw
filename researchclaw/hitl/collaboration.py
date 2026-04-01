"""HITL collaboration session: deep human-AI co-creation for a stage."""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from researchclaw.hitl.chat import ChatMessage, ChatSession, build_stage_context

logger = logging.getLogger(__name__)


@dataclass
class CollaborationSession:
    """Deep collaboration between human and AI on a specific stage.

    Extends ChatSession with:
    - Shared artifact editing (both human and AI can modify files)
    - Revision tracking
    - Structured output finalization
    """

    chat: ChatSession = field(default_factory=ChatSession)
    stage_num: int = 0
    stage_name: str = ""
    run_dir: Path | None = None

    # Shared workspace
    shared_artifacts: dict[str, str] = field(default_factory=dict)
    revision_history: list[dict[str, str]] = field(default_factory=list)
    finalized: bool = False

    def initialize(
        self,
        stage_num: int,
        stage_name: str,
        topic: str,
        run_dir: Path,
        artifacts: tuple[str, ...] = (),
        llm_client: Any = None,
    ) -> None:
        """Set up the collaboration with stage context."""
        self.stage_num = stage_num
        self.stage_name = stage_name
        self.run_dir = run_dir

        self.chat.stage_num = stage_num
        self.chat.stage_name = stage_name
        self.chat.topic = topic

        # Build and inject context
        context = build_stage_context(
            stage_num, stage_name, topic, run_dir, artifacts
        )
        self.chat.add_system_message(context)

        # Load current artifacts into shared workspace
        stage_dir = run_dir / f"stage-{stage_num:02d}"
        for fname in artifacts:
            fpath = stage_dir / fname
            if fpath.is_file():
                try:
                    self.shared_artifacts[fname] = fpath.read_text(
                        encoding="utf-8"
                    )
                except (OSError, UnicodeDecodeError):
                    pass

    def human_says(self, message: str) -> None:
        """Record human message."""
        self.chat.add_human_message(message)

    def ai_responds(self, llm_client: Any) -> str:
        """Get AI response to the current conversation."""
        return self.chat.get_ai_response(llm_client)

    def human_edits_artifact(self, filename: str, content: str) -> None:
        """Human directly edits a shared artifact."""
        old_content = self.shared_artifacts.get(filename, "")
        self.shared_artifacts[filename] = content
        self.revision_history.append({
            "action": "human_edit",
            "file": filename,
            "old_length": str(len(old_content)),
            "new_length": str(len(content)),
        })

        # Write to disk
        if self.run_dir:
            stage_dir = self.run_dir / f"stage-{self.stage_num:02d}"
            stage_dir.mkdir(parents=True, exist_ok=True)
            (stage_dir / filename).write_text(content, encoding="utf-8")

    def ai_proposes_edit(
        self, filename: str, content: str
    ) -> None:
        """AI proposes an edit to a shared artifact."""
        old_content = self.shared_artifacts.get(filename, "")
        self.shared_artifacts[filename] = content
        self.revision_history.append({
            "action": "ai_proposal",
            "file": filename,
            "old_length": str(len(old_content)),
            "new_length": str(len(content)),
        })

    def finalize(self) -> dict[str, str]:
        """End collaboration and write all artifacts to disk.

        Returns the final shared artifacts dict.
        """
        if self.run_dir:
            stage_dir = self.run_dir / f"stage-{self.stage_num:02d}"
            stage_dir.mkdir(parents=True, exist_ok=True)
            for fname, content in self.shared_artifacts.items():
                (stage_dir / fname).write_text(content, encoding="utf-8")

            # Save chat history
            hitl_dir = self.run_dir / "hitl"
            hitl_dir.mkdir(parents=True, exist_ok=True)
            self.chat.save(
                hitl_dir / f"chat_stage_{self.stage_num:02d}.jsonl"
            )

            # Save revision history
            (hitl_dir / f"revisions_stage_{self.stage_num:02d}.json").write_text(
                json.dumps(self.revision_history, indent=2),
                encoding="utf-8",
            )

        self.finalized = True
        return dict(self.shared_artifacts)

    def to_dict(self) -> dict[str, Any]:
        return {
            "stage_num": self.stage_num,
            "stage_name": self.stage_name,
            "chat": self.chat.to_dict(),
            "artifacts": list(self.shared_artifacts.keys()),
            "revision_count": len(self.revision_history),
            "finalized": self.finalized,
        }
