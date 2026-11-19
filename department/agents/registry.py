"""The seven roles of one department, each an independent instance.

The type check in ``__post_init__`` is not ceremony. These roles are meant to
diverge -- their own prompt, their own model, later their own library -- and the
one mistake that would quietly undo that is wiring the same object into two
slots. This makes that impossible rather than merely discouraged.
"""

from __future__ import annotations

from dataclasses import dataclass

from department.agents.anesthesiologist import AnesthesiaAgent
from department.agents.director import DirectorAgent
from department.agents.neurologist import NeurologyAgent
from department.agents.physiatrist import PhysiatristAgent
from department.agents.radiologist import RadiologyAgent
from department.agents.resident import ResidentAgent
from department.agents.vice_director import ViceDirectorAgent


@dataclass(frozen=True, slots=True)
class DepartmentRegistry:
    resident: ResidentAgent
    radiologist: RadiologyAgent
    neurologist: NeurologyAgent
    anesthesiologist: AnesthesiaAgent
    physiatrist: PhysiatristAgent
    vice_director: ViceDirectorAgent
    director: DirectorAgent

    def __post_init__(self) -> None:
        expected = {
            "resident": (self.resident, ResidentAgent),
            "radiologist": (self.radiologist, RadiologyAgent),
            "neurologist": (self.neurologist, NeurologyAgent),
            "anesthesiologist": (self.anesthesiologist, AnesthesiaAgent),
            "physiatrist": (self.physiatrist, PhysiatristAgent),
            "vice_director": (self.vice_director, ViceDirectorAgent),
            "director": (self.director, DirectorAgent),
        }
        for name, (agent, agent_type) in expected.items():
            if not isinstance(agent, agent_type):
                raise TypeError(f"{name} must be an independent {agent_type.__name__}")

    def all_agents(self) -> tuple:
        return (
            self.resident,
            self.radiologist,
            self.neurologist,
            self.anesthesiologist,
            self.physiatrist,
            self.vice_director,
            self.director,
        )
