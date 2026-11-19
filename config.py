"""Environment-backed application configuration."""

from functools import lru_cache
from pathlib import Path
from typing import Literal

from pydantic import BaseModel, Field, SecretStr
from pydantic_settings import BaseSettings, SettingsConfigDict

from exceptions import ConfigurationError


def _secret(value: SecretStr | None) -> str:
    """A blank credential is a missing credential.

    ``OPENAI_API_KEY=`` in a .env file reads as an empty string rather than as
    an absent variable, so a ``is None`` check lets it through and the failure
    surfaces as a rejected request from the provider after the work is already
    done. Treating blank as absent moves that to the first line of the run.
    """

    return value.get_secret_value().strip() if value is not None else ""


class ProviderConfig(BaseModel):
    """Validated configuration used to instantiate a live provider adapter."""

    api_key: SecretStr
    model: str
    base_url: str | None = None
    timeout_seconds: float
    max_retries: int


class Settings(BaseSettings):
    """Application settings that remain importable without provider secrets."""

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
        case_sensitive=False,
    )

    openai_api_key: SecretStr | None = None
    openai_base_url: str | None = None
    openai_model: str = "gpt-5.6-terra"
    openai_critical_model: str = "gpt-5.6-sol"
    openai_image_detail: Literal["original", "high"] = "original"

    # DeepSeek provider: default for the terra text roles. The base URL and
    # model can be pointed at any OpenAI-compatible endpoint.
    deepseek_api_key: SecretStr | None = None
    deepseek_base_url: str = "https://api.deepseek.com"
    deepseek_model: str = "deepseek-v4-pro"
    deepseek_thinking: Literal["off", "max"] = "max"
    deepseek_timeout_seconds: float = Field(default=300.0, gt=0, le=1800)
    deepseek_max_retries: int = Field(default=3, ge=0, le=10)

    # Provider routing for the terra roles. Text roles (case summary, moderator,
    # and the four text specialists) default to DeepSeek; image roles (document
    # transcription and the radiologist) default to OpenAI because the stock
    # DeepSeek endpoint is text-only. Both are overridable here.
    text_provider: Literal["openai", "deepseek"] = "deepseek"
    image_provider: Literal["openai", "deepseek"] = "openai"

    model_timeout_seconds: float = Field(default=120.0, gt=0)
    model_max_retries: int = Field(default=3, ge=0, le=10)
    document_timeout_seconds: float = Field(default=180.0, gt=0, le=1800)
    radiology_timeout_seconds: float = Field(default=300.0, gt=0, le=1800)
    radiology_max_retries: int = Field(default=1, ge=0, le=5)
    radiology_batch_max_images: int = Field(default=4, ge=1, le=20)
    radiology_batch_max_bytes: int = Field(default=12 * 1024 * 1024, gt=0)
    radiology_batch_concurrency: int = Field(default=2, ge=1, le=4)
    radiology_cache_enabled: bool = True
    radiology_cache_dir: Path = Path(".data/radiology-cache")
    # A baseline freezes one case's prepared evidence so that repeated runs of
    # the same case read identical documents instead of regenerating them.
    baseline_store_enabled: bool = True
    baseline_store_dir: Path = Path(".data/baselines")
    document_previous_page_tail_chars: int = Field(default=1200, ge=0, le=10_000)

    # Deterministic case-record transcription. Weights are pinned local files so
    # that the same page bytes always produce the same Markdown; see
    # models/ocr/SOURCES.md.
    ocr_model_dir: Path = Path("models/ocr")
    ocr_low_confidence_threshold: float = Field(default=0.95, ge=0, le=1)
    page_slice_max_height: int = Field(default=2400, ge=200, le=20_000)
    page_slice_min_blank_run: int = Field(default=8, ge=1, le=500)

    # Department workflow: every role is an independent individual and may run
    # on its own model. The default keeps them all on one model so nothing has
    # to be configured until a role is deliberately made to differ. Overrides
    # are written as {"director": "openai:gpt-5.6-sol"}. The radiologist follows
    # image_provider instead, because it is the only role that sends images.
    department_default_provider: Literal["openai", "deepseek"] = "deepseek"
    department_role_models: dict[str, str] = Field(default_factory=dict)
    #: Which wording of the style prefixes the director and deputy read. ``b``
    #: says the same rules in other words; it exists only to show a result does
    #: not hang on the wording (experiment plan, milestone M6). ``t`` describes
    #: the three directors' character only -- no rules, no treatment direction
    #: -- to test whether the effects survive without being instructed (M9).
    department_style_variant: Literal["a", "b", "t"] = "a"
    critical_timeout_seconds: float = Field(default=300.0, gt=0, le=1800)
    critical_max_retries: int = Field(default=1, ge=0, le=5)
    checkpoint_db_path: Path = Path(".data/checkpoints.sqlite3")
    run_store_db_path: Path = Path(".data/runs.sqlite3")
    upload_dir: Path = Path(".data/uploads")
    log_level: str = "INFO"
    max_upload_bytes: int = Field(default=20 * 1024 * 1024, gt=0)
    max_images_per_case: int = Field(default=50, gt=0, le=500)
    sse_heartbeat_seconds: float = Field(default=15.0, gt=0, le=300)
    max_additional_input_chars: int = Field(default=20_000, gt=0, le=1_000_000)

    def require_openai(self) -> ProviderConfig:
        if not _secret(self.openai_api_key):
            raise ConfigurationError("OPENAI_API_KEY is required to use the OpenAI provider.")
        return ProviderConfig(
            api_key=self.openai_api_key,
            model=self.openai_model,
            base_url=self.openai_base_url,
            timeout_seconds=self.model_timeout_seconds,
            max_retries=self.model_max_retries,
        )

    def require_openai_critical(self) -> ProviderConfig:
        base = self.require_openai()
        return base.model_copy(
            update={
                "model": self.openai_critical_model,
                "timeout_seconds": self.critical_timeout_seconds,
                "max_retries": self.critical_max_retries,
            }
        )

    def require_deepseek(self) -> ProviderConfig:
        if not _secret(self.deepseek_api_key):
            raise ConfigurationError(
                "DEEPSEEK_API_KEY is required to use the DeepSeek provider."
            )
        return ProviderConfig(
            api_key=self.deepseek_api_key,
            model=self.deepseek_model,
            base_url=self.deepseek_base_url,
            timeout_seconds=self.deepseek_timeout_seconds,
            max_retries=self.deepseek_max_retries,
        )

    def configured_secrets(self) -> tuple[str, ...]:
        secrets = [
            secret
            for secret in (
                self.openai_api_key,
                self.deepseek_api_key,
            )
            if secret is not None and secret.get_secret_value()
        ]
        return tuple(secret.get_secret_value() for secret in secrets)


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    return Settings()
