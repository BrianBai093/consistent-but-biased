"""The seven speaking roles of the department discussion."""

from department.agents.anesthesiologist import AnesthesiaAgent
from department.agents.director import DirectorAgent
from department.agents.neurologist import NeurologyAgent
from department.agents.physiatrist import PhysiatristAgent
from department.agents.radiologist import RadiologyAgent
from department.agents.resident import ResidentAgent
from department.agents.vice_director import ViceDirectorAgent
from department.agents.registry import DepartmentRegistry

__all__ = [
    "AnesthesiaAgent",
    "DirectorAgent",
    "DepartmentRegistry",
    "NeurologyAgent",
    "PhysiatristAgent",
    "RadiologyAgent",
    "ResidentAgent",
    "ViceDirectorAgent",
]
