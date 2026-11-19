"""Build one department, giving every role its own client and its own prompt."""

from __future__ import annotations

from config import Settings
from department.agents import (
    AnesthesiaAgent,
    DepartmentRegistry,
    DirectorAgent,
    NeurologyAgent,
    PhysiatristAgent,
    RadiologyAgent,
    ResidentAgent,
    ViceDirectorAgent,
)
from department.model_config import RoleModel, resolve_role_models
from department.schemas import RoleModelAssignment
from schemas import AgentRole
from services.llm_client import DeepSeekTextClient
from services.multimodal_client import OpenAIResponsesClient


def _client(role_model: RoleModel, settings: Settings):
    """One client per role: they are configured, and may be changed, separately.

    ``supports_images`` is only opened for the radiologist, and only because
    someone deliberately pointed it at a DeepSeek-compatible endpoint. The stock
    endpoint is text-only; the guard it lifts is the thing that would otherwise
    catch a misconfiguration before it reaches the provider.
    """

    if role_model.provider == "deepseek":
        return DeepSeekTextClient(
            role_model.config,
            thinking=settings.deepseek_thinking,
            supports_images=role_model.role is AgentRole.RADIOLOGIST,
            image_detail=settings.openai_image_detail,
        )
    return OpenAIResponsesClient(
        role_model.config,
        image_detail=settings.openai_image_detail,
    )


def create_department(
    settings: Settings,
) -> tuple[DepartmentRegistry, list[RoleModelAssignment]]:
    """Return the department and the record of which model served each role."""

    role_models = resolve_role_models(settings)
    clients = {role: _client(model, settings) for role, model in role_models.items()}
    registry = DepartmentRegistry(
        resident=ResidentAgent(clients[AgentRole.RESIDENT]),
        radiologist=RadiologyAgent(
            clients[AgentRole.RADIOLOGIST],
            batch_max_images=settings.radiology_batch_max_images,
            batch_max_bytes=settings.radiology_batch_max_bytes,
            batch_concurrency=settings.radiology_batch_concurrency,
        ),
        neurologist=NeurologyAgent(clients[AgentRole.NEUROLOGIST]),
        anesthesiologist=AnesthesiaAgent(clients[AgentRole.ANESTHESIOLOGIST]),
        physiatrist=PhysiatristAgent(clients[AgentRole.PHYSIATRIST]),
        vice_director=ViceDirectorAgent(
            clients[AgentRole.VICE_DIRECTOR],
            style_variant=settings.department_style_variant,
        ),
        director=DirectorAgent(
            clients[AgentRole.DIRECTOR],
            style_variant=settings.department_style_variant,
        ),
    )
    assignments = [role_models[role].assignment() for role in role_models]
    return registry, assignments
