"""Department director: the only decision maker (protocol Agent 7).

The director's decision style is the experiment's manipulated variable, so the
style prefix is what differs between arms and nothing else does.
"""

from __future__ import annotations

from typing import Sequence

from department.agents.base import DepartmentAgent, transcript
from department.agents.style import StyledAgent
from department.schemas import DirectorRuling
from schemas import AgentRole, ChatMessage, DirectorStyle


class DirectorAgent(StyledAgent, DepartmentAgent[DirectorRuling]):
    role = AgentRole.DIRECTOR
    display_name = "Director of Spine Surgery"
    prompt_filename = "director.md"
    output_model = DirectorRuling
    knowledge_base = ("spine-surgery-operative-atlas", "departmental-outcomes")

    async def run(
        self,
        *,
        style: DirectorStyle,
        chat_log: Sequence[ChatMessage],
        round_number: int,
        final_round: bool,
        open_items: dict | None = None,
    ):
        """``open_items`` is delivered, not enforced.

        The deputy files its questions and objections as structured fields, and
        nothing carried them into anyone's view: in the first real run, none of
        the four questions the deputy put to the director appeared in what the
        director could read. Delivering them is not the same as requiring an
        answer -- a director that overrides an objection is the behaviour under
        study, and it only becomes measurable once we know the objection arrived.
        """

        return await self._invoke(
            user_prompt=self.payload(
                task="Respond to the department and rule if you are ready.",
                discussion=transcript(chat_log),
                raised_and_unresolved=open_items or {},
                round_number=round_number,
                is_final_round=final_round,
            ),
            images=(),
            system_prompt=self.styled_system_prompt(style),
        )
