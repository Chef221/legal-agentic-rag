"""FastAPI application factory with fail-fast runtime lifespan."""

from __future__ import annotations

from collections.abc import Callable
from contextlib import asynccontextmanager
import logging

from fastapi import FastAPI, Request
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse, RedirectResponse

from legal_agentic_rag.configuration import ApplicationConfig
from legal_agentic_rag.exceptions import (
    ArtifactCompatibilityError,
    BackendInitializationError,
    ConfigurationError,
    DataValidationError,
    ExternalServiceError,
    InvalidUserInputError,
    LegalAgenticRAGError,
    ModelError,
    OperationTimeoutError,
    RetrievalError,
)
from legal_agentic_rag.observability import configure_logging
from legal_agentic_rag.runtime import OnlineRuntime, OnlineRuntimeFactory
from legal_agentic_rag.schemas import (
    AnswerResponse,
    ApiErrorDetail,
    ApiErrorResponse,
    HealthResponse,
    LegalQuestionRequest,
    RetrievalResponse,
)
from legal_agentic_rag.serving.query_service import ServingService
from legal_agentic_rag.serving.ui import mount_gradio_ui

RuntimeLoader = Callable[[], OnlineRuntime]
_LOGGER = logging.getLogger(__name__)


def create_app(
    config: ApplicationConfig,
    *,
    runtime_loader: RuntimeLoader | None = None,
) -> FastAPI:
    """Create an API/UI process that loads one immutable runtime at startup."""
    configure_logging(config.logging)
    load_runtime = runtime_loader or (
        lambda: OnlineRuntimeFactory(config).build()
    )

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        _LOGGER.info("serving_runtime_starting")
        runtime = load_runtime()
        app.state.serving_service = ServingService(
            runtime,
            config.serving,
            config.online,
        )
        _LOGGER.info(
            "serving_runtime_ready",
            extra={
                "artifact_count": len(runtime.manifests),
                "tool_count": len(runtime.tool_descriptors()),
            },
        )
        yield
        del app.state.serving_service
        _LOGGER.info("serving_runtime_stopped")

    docs_url = "/docs" if config.serving.docs_enabled else None
    redoc_url = "/redoc" if config.serving.docs_enabled else None
    app = FastAPI(
        title=config.serving.title,
        version=_service_version(),
        docs_url=docs_url,
        redoc_url=redoc_url,
        lifespan=lifespan,
    )
    _register_exception_handlers(app)
    prefix = config.serving.api_prefix

    @app.get("/", include_in_schema=False)
    async def root() -> RedirectResponse:
        destination = (
            config.serving.ui_path
            if config.serving.ui_enabled
            else docs_url or f"{prefix}/health"
        )
        return RedirectResponse(destination)

    @app.get(
        f"{prefix}/health",
        response_model=HealthResponse,
        responses={503: {"model": ApiErrorResponse}},
    )
    async def health(request: Request) -> HealthResponse:
        return _service(request).health()

    @app.post(
        f"{prefix}/retrieve",
        response_model=RetrievalResponse,
        responses={
            400: {"model": ApiErrorResponse},
            422: {"model": ApiErrorResponse},
            503: {"model": ApiErrorResponse},
            504: {"model": ApiErrorResponse},
        },
    )
    async def retrieve(
        payload: LegalQuestionRequest,
        request: Request,
    ) -> RetrievalResponse:
        return _service(request).retrieve(payload)

    @app.post(
        f"{prefix}/answer",
        response_model=AnswerResponse,
        responses={
            400: {"model": ApiErrorResponse},
            422: {"model": ApiErrorResponse},
            503: {"model": ApiErrorResponse},
            504: {"model": ApiErrorResponse},
        },
    )
    async def answer(
        payload: LegalQuestionRequest,
        request: Request,
    ) -> AnswerResponse:
        return _service(request).answer(payload)

    if config.serving.ui_enabled:
        app = mount_gradio_ui(
            app,
            service_provider=lambda: _service_from_app(app),
            path=config.serving.ui_path,
            title=config.serving.title,
        )
    return app


def _service(request: Request) -> ServingService:
    return _service_from_app(request.app)


def _service_from_app(app: FastAPI) -> ServingService:
    service = getattr(app.state, "serving_service", None)
    if not isinstance(service, ServingService):
        raise BackendInitializationError("Serving runtime is not ready")
    return service


def _register_exception_handlers(app: FastAPI) -> None:
    @app.exception_handler(RequestValidationError)
    async def request_validation_handler(
        request: Request,
        error: RequestValidationError,
    ) -> JSONResponse:
        _ = (request, error)
        return _error_response(
            422,
            "invalid_request",
            "The request does not match the API contract.",
        )

    @app.exception_handler(LegalAgenticRAGError)
    async def domain_error_handler(
        request: Request,
        error: LegalAgenticRAGError,
    ) -> JSONResponse:
        _ = request
        status_code, error_type, message = _domain_error(error)
        _LOGGER.warning(
            "serving_request_failed",
            extra={"error_type": error_type},
        )
        return _error_response(status_code, error_type, message)

    @app.exception_handler(Exception)
    async def unexpected_error_handler(
        request: Request,
        error: Exception,
    ) -> JSONResponse:
        _ = request
        _LOGGER.exception(
            "serving_unexpected_error",
            exc_info=(type(error), error, error.__traceback__),
            extra={"error_type": "internal_error"},
        )
        return _error_response(
            500,
            "internal_error",
            "The service could not complete the request.",
        )


def _domain_error(
    error: LegalAgenticRAGError,
) -> tuple[int, str, str]:
    mappings: tuple[
        tuple[type[LegalAgenticRAGError], int, str, str],
        ...,
    ] = (
        (
            InvalidUserInputError,
            400,
            "invalid_user_input",
            "The question or retrieval limits are invalid.",
        ),
        (
            OperationTimeoutError,
            504,
            "timeout",
            "The request exceeded its configured time budget.",
        ),
        (
            ArtifactCompatibilityError,
            503,
            "artifact_compatibility_error",
            "A required legal artifact is unavailable or incompatible.",
        ),
        (
            BackendInitializationError,
            503,
            "backend_initialization_error",
            "A required serving backend is unavailable.",
        ),
        (
            RetrievalError,
            503,
            "retrieval_error",
            "Legal evidence retrieval could not be completed.",
        ),
        (
            ModelError,
            503,
            "model_error",
            "Model inference could not be completed.",
        ),
        (
            ExternalServiceError,
            503,
            "external_service_error",
            "An explicitly configured external service is unavailable.",
        ),
        (
            ConfigurationError,
            500,
            "configuration_error",
            "The service configuration is invalid.",
        ),
        (
            DataValidationError,
            500,
            "data_validation_error",
            "Internal legal data failed validation.",
        ),
    )
    for error_class, status_code, error_type, message in mappings:
        if isinstance(error, error_class):
            return status_code, error_type, message
    return 500, "internal_error", "The service could not complete the request."


def _error_response(
    status_code: int,
    error_type: str,
    message: str,
) -> JSONResponse:
    payload = ApiErrorResponse(
        error=ApiErrorDetail(error_type=error_type, message=message)
    )
    return JSONResponse(
        status_code=status_code,
        content=payload.model_dump(mode="json"),
    )


def _service_version() -> str:
    from legal_agentic_rag import __version__

    return __version__
