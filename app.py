"""FastAPI application entry point and service lifespan."""

from __future__ import annotations

import logging
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Callable

from fastapi import FastAPI, Request
from fastapi.exceptions import RequestValidationError
from fastapi.openapi.utils import get_openapi
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

from agents import AgentRegistry
from api_routes import router as api_router
from api_schemas import ErrorResponse
from config import Settings, get_settings
from exceptions import ConfigurationError, ServiceError
from services.agent_factory import create_live_agent_registry
from services.checkpoint import open_sqlite_checkpointer
from services.run_logger import configure_logging
from services.run_manager import RunManager
from services.run_store import RunStore
from services.baseline_store import BaselineStore
from services.radiology_cache import RadiologyCache
from services.upload_store import UploadStore


logger = logging.getLogger(__name__)
WEB_DIR = Path(__file__).resolve().parent / "web"


class HealthResponse(BaseModel):
    status: str
    service: str


def create_app(
    *,
    settings: Settings | None = None,
    registry_factory: Callable[[], AgentRegistry] | None = None,
) -> FastAPI:
    application_settings = settings or get_settings()
    configure_logging(
        application_settings.log_level,
        application_settings.configured_secrets(),
    )
    effective_registry_factory = registry_factory or (
        lambda: create_live_agent_registry(application_settings)
    )

    @asynccontextmanager
    async def lifespan(application: FastAPI):
        run_store = await RunStore.open(application_settings.run_store_db_path)
        try:
            async with open_sqlite_checkpointer(
                application_settings.checkpoint_db_path
            ) as checkpointer:
                upload_store = UploadStore(
                    application_settings.upload_dir,
                    max_upload_bytes=application_settings.max_upload_bytes,
                    max_images_per_case=application_settings.max_images_per_case,
                    page_slice_max_height=application_settings.page_slice_max_height,
                    page_slice_min_blank_run=(
                        application_settings.page_slice_min_blank_run
                    ),
                )
                run_manager = RunManager(
                    checkpointer=checkpointer,
                    run_store=run_store,
                    registry_factory=effective_registry_factory,
                    max_additional_input_chars=(
                        application_settings.max_additional_input_chars
                    ),
                    radiology_cache=RadiologyCache(
                        application_settings.radiology_cache_dir,
                        enabled=application_settings.radiology_cache_enabled,
                    ),
                    baseline_store=BaselineStore(
                        application_settings.baseline_store_dir,
                        enabled=application_settings.baseline_store_enabled,
                    ),
                )
                application.state.settings = application_settings
                application.state.upload_store = upload_store
                application.state.run_manager = run_manager
                await run_manager.reconcile_startup()
                try:
                    yield
                finally:
                    await run_manager.shutdown()
        finally:
            await run_store.close()

    application = FastAPI(
        title="Spine MDT API",
        version="0.1.0",
        description="Preoperative spine surgery MDT simulation API.",
        lifespan=lifespan,
    )
    application.state.settings = application_settings

    @application.exception_handler(ServiceError)
    async def service_error_handler(
        request: Request,
        exc: ServiceError,
    ) -> JSONResponse:
        body = ErrorResponse(
            code=exc.code,
            message=exc.public_message,
            thread_id=exc.thread_id,
        )
        return JSONResponse(status_code=exc.status_code, content=body.model_dump())

    @application.exception_handler(ConfigurationError)
    async def configuration_error_handler(
        request: Request,
        exc: ConfigurationError,
    ) -> JSONResponse:
        body = ErrorResponse(
            code="PROVIDER_NOT_CONFIGURED",
            message="Required model provider configuration is unavailable.",
        )
        return JSONResponse(status_code=503, content=body.model_dump())

    @application.exception_handler(RequestValidationError)
    async def request_validation_error_handler(
        request: Request,
        exc: RequestValidationError,
    ) -> JSONResponse:
        body = ErrorResponse(
            code="REQUEST_VALIDATION_ERROR",
            message="The request does not satisfy the API schema.",
        )
        return JSONResponse(status_code=422, content=body.model_dump())

    @application.get("/health", response_model=HealthResponse)
    async def health() -> HealthResponse:
        return HealthResponse(status="ok", service="spine-mdt")

    @application.get("/", include_in_schema=False, response_class=FileResponse)
    async def web_ui() -> FileResponse:
        return FileResponse(WEB_DIR / "index.html")

    application.mount(
        "/static",
        StaticFiles(directory=WEB_DIR),
        name="static",
    )
    application.include_router(api_router)

    def custom_openapi() -> dict:
        """Preserve browser file pickers for all multipart image fields.

        FastAPI/Pydantic currently describe ``list[UploadFile]`` with the OAS
        3.1 ``contentMediaType`` keyword.  Swagger UI does not render that
        representation as a picker in all supported browser bundles, whereas
        the widely supported ``format: binary`` representation does.
        """
        if application.openapi_schema:
            return application.openapi_schema
        openapi_schema = get_openapi(
            title=application.title,
            version=application.version,
            description=application.description,
            routes=application.routes,
        )
        try:
            body_schema = openapi_schema["components"]["schemas"][
                "Body_create_case_v1_cases_post"
            ]
            for field in ("case_files", "test_files", "image_files"):
                file_items = body_schema["properties"][field]["items"]
                file_items["format"] = "binary"
                file_items.pop("contentMediaType", None)
        except KeyError:
            # Do not make schema generation fail if FastAPI changes its
            # generated component name in a future upgrade.
            logger.warning("Could not normalize the multipart upload schema")
        application.openapi_schema = openapi_schema
        return application.openapi_schema

    application.openapi = custom_openapi  # type: ignore[method-assign]
    return application


app = create_app()
