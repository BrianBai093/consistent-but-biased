"""Deputy director: professional judgement plus managing upwards (Agent 6).

The style is injected here as well as into the director. Changing only the
director's personality would not reproduce the self-censorship that a
subordinate performs under an authoritarian chief, which is the effect the
experiment is trying to observe.

What is injected is not the director's prefix. That one is addressed to the
person who rules, and giving it to the deputy produced a deputy that ruled.
The deputy reads its own switches: the same three styles, seen from below.
"""

from __future__ import annotations

from typing import Sequence

from department.agents.base import DepartmentAgent, transcript
from department.agents.style import DEPUTY_STYLE_PROMPT_FILENAMES, StyledAgent
from department.schemas import ViceDirectorOpinion
from schemas import AgentRole, ChatMessage, DirectorStyle


class ViceDirectorAgent(StyledAgent, DepartmentAgent[ViceDirectorOpinion]):
    role = AgentRole.VICE_DIRECTOR
    display_name = "Deputy Director of Spine Surgery"
    prompt_filename = "vice_director.md"
    output_model = ViceDirectorOpinion
    style_prompt_filenames = DEPUTY_STYLE_PROMPT_FILENAMES
    knowledge_base = ("spine-surgery-operative-atlas", "departmental-procedure-preferences")

    async def run(
        self,
        *,
        style: DirectorStyle,
        chat_log: Sequence[ChatMessage],
        round_number: int,
    ):
        return await self._invoke(
            user_prompt=self.payload(
                task="Give your professional judgement on this case to the director.",
                discussion=transcript(chat_log),
                round_number=round_number,
            ),
            images=(),
            system_prompt=self.styled_system_prompt(style),
        )
