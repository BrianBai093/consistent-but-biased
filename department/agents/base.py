"""Shared plumbing for department roles.

Each role is its own class with its own prompt, output schema, and model client;
this base only carries what every role needs regardless of who it is. It
deliberately does not take a role parameter -- a class parameterised by role
would make "give this one role a different personality, model, or library" a
change to shared code, which is the opposite of what this simulation needs.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Sequence, TypeVar

from pydantic import BaseModel

from agents.base_agent import BaseAgent
from department.knowledge import KnowledgeSource
from exceptions import ModelInvocationError
from schemas import ChatMessage
from services.llm_client import StructuredModelClient

OutputT = TypeVar("OutputT", bound=BaseModel)
PROMPT_ROOT = Path(__file__).resolve().parent.parent / "prompts"


class DepartmentAgent(BaseAgent[OutputT]):
    """One participant in the department discussion.

    Subclasses set ``role``, ``display_name``, ``prompt_filename``,
    ``output_model`` and ``knowledge_base``, then expose whatever ``run``
    signature their part of the protocol actually needs.
    """

    #: Identifiers of the sources this role is supposed to consult. Declared
    #: now, consulted once retrieval exists.
    knowledge_base: tuple[str, ...] = ()

    def __init__(
        self,
        client: StructuredModelClient,
        *,
        knowledge_source: KnowledgeSource | None = None,
    ) -> None:
        super().__init__(client)
        self.knowledge_source = knowledge_source

    def _load_prompt(self) -> str:
        return self.load_department_prompt(self.prompt_filename)

    def load_department_prompt(self, filename: str) -> str:
        """Read a prompt from this package rather than from the v5 prompt tree."""

        path = PROMPT_ROOT / filename
        try:
            prompt = path.read_text(encoding="utf-8").strip()
        except OSError as exc:
            raise ModelInvocationError(
                f"Department prompt {filename!r} could not be loaded."
            ) from exc
        if not prompt:
            raise ModelInvocationError(f"Department prompt {filename!r} is empty.")
        return prompt

    def build_user_prompt(self, **kwargs: Any) -> str:
        """Not used: each role builds the payload its own part needs."""

        raise NotImplementedError(f"{type(self).__name__} builds its own prompt")

    @staticmethod
    def payload(**fields: Any) -> str:
        return json.dumps(fields, ensure_ascii=False)


def transcript(messages: Sequence[ChatMessage]) -> list[dict]:
    """Public discussion as every role sees it: who said what, in order."""

    return [
        {
            "sequence": message.sequence,
            "speaker": message.speaker_name,
            "role": message.speaker_role.value,
            "content": message.content,
        }
        for message in messages
    ]
