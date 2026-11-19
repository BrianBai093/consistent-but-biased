"""Per-role model assignment for the department workflow.

Every role is an independent individual: its own class, its own prompt, and its
own model client. Roles may legitimately run on different models -- that stays
constant across the three arms and so cannot explain an arm difference -- but
changing one role's model between batches makes those batches incomparable.
Which model actually served each role is therefore recorded with the run.

The default keeps all roles on one model so that nothing has to be configured
until someone deliberately wants a role to differ.
"""

from __future__ import annotations

from dataclasses import dataclass

from config import ProviderConfig, Settings
from department.schemas import RoleModelAssignment
from exceptions import ConfigurationError
from schemas import AgentRole

DEPARTMENT_ROLES: tuple[AgentRole, ...] = (
    AgentRole.RESIDENT,
    AgentRole.RADIOLOGIST,
    AgentRole.NEUROLOGIST,
    AgentRole.ANESTHESIOLOGIST,
    AgentRole.PHYSIATRIST,
    AgentRole.VICE_DIRECTOR,
    AgentRole.DIRECTOR,
)

_PROVIDERS = ("openai", "deepseek")


@dataclass(frozen=True, slots=True)
class RoleModel:
    """One role's resolved provider configuration plus its recordable label."""

    role: AgentRole
    provider: str
    config: ProviderConfig

    def assignment(self) -> RoleModelAssignment:
        return RoleModelAssignment(
            role=self.role,
            provider=self.provider,
            model=self.config.model,
        )


def parse_override(value: str) -> tuple[str, str]:
    """Parse a ``provider:model`` override, e.g. ``deepseek:deepseek-v4-pro``."""

    provider, separator, model = value.partition(":")
    provider = provider.strip().lower()
    model = model.strip()
    if not separator or not model:
        raise ConfigurationError(
            "A role model override must be written as 'provider:model'."
        )
    if provider not in _PROVIDERS:
        raise ConfigurationError(
            f"Unknown model provider {provider!r}; expected one of {_PROVIDERS}."
        )
    return provider, model


def resolve_role_models(
    settings: Settings,
    *,
    roles: tuple[AgentRole, ...] = DEPARTMENT_ROLES,
) -> dict[AgentRole, RoleModel]:
    """Resolve every role's model, applying per-role overrides over the default.

    An override names an unknown role loudly rather than being ignored: a typo
    that silently left a role on the default model would be invisible in the
    results and would only surface as unexplained variance.
    """

    overrides: dict[AgentRole, tuple[str, str]] = {}
    for raw_role, raw_value in (settings.department_role_models or {}).items():
        try:
            role = AgentRole(raw_role)
        except ValueError as exc:
            raise ConfigurationError(
                f"Unknown role {raw_role!r} in the department model overrides."
            ) from exc
        if role not in roles:
            raise ConfigurationError(
                f"Role {raw_role!r} does not take part in the department workflow."
            )
        overrides[role] = parse_override(raw_value)

    resolved: dict[AgentRole, RoleModel] = {}
    for role in roles:
        provider, model = overrides.get(role, (_default_provider(settings, role), ""))
        base = (
            settings.require_deepseek()
            if provider == "deepseek"
            else settings.require_openai()
        )
        update: dict[str, object] = {"model": model} if model else {}
        if role is AgentRole.RADIOLOGIST:
            # Reading a dozen photographed films is a different failure profile
            # from a text turn, so it keeps the radiology timeout and retry
            # budget rather than the general one.
            update["timeout_seconds"] = settings.radiology_timeout_seconds
            update["max_retries"] = settings.radiology_max_retries
        config = base.model_copy(update=update) if update else base
        resolved[role] = RoleModel(role=role, provider=provider, config=config)
    return resolved


def _default_provider(settings: Settings, role: AgentRole) -> str:
    """The radiologist follows ``image_provider``; everyone else the text default.

    The stock DeepSeek endpoint is text-only, so leaving the radiologist on the
    text default would send image parts to an endpoint that rejects them -- and
    the rejection is not retryable, so the run would simply fail.
    """

    if role is AgentRole.RADIOLOGIST:
        return settings.image_provider
    return settings.department_default_provider



