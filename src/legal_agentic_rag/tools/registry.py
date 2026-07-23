"""Closed tool registration, schema discovery, and safe execution."""

from __future__ import annotations

import logging
from time import perf_counter
from collections.abc import Iterable

from pydantic import BaseModel, ValidationError

from legal_agentic_rag.exceptions import (
    ArtifactCompatibilityError,
    BackendInitializationError,
    ConfigurationError,
    DatasetSchemaError,
    DataValidationError,
    ExternalServiceError,
    InvalidUserInputError,
    LegalAgenticRAGError,
    ModelError,
    OperationTimeoutError,
    RetrievalError,
)
from legal_agentic_rag.schemas.tools import (
    ToolDescriptor,
    ToolError,
    ToolErrorType,
    ToolInvocationRequest,
    ToolInvocationResult,
    ToolName,
)
from legal_agentic_rag.tools.contracts import TypedTool

_LOGGER = logging.getLogger(__name__)


class ToolRegistry:
    """Hold only explicitly registered tools and normalize known failures."""

    def __init__(self, tools: Iterable[TypedTool] = ()) -> None:
        self._tools: dict[ToolName, TypedTool] = {}
        for tool in tools:
            self.register(tool)

    def register(self, tool: TypedTool) -> None:
        """Register one typed tool and reject duplicate capability names."""
        if not isinstance(tool, TypedTool):
            raise ConfigurationError("Registered object does not satisfy TypedTool")
        if tool.name in self._tools:
            raise ConfigurationError("Tool name is already registered")
        if tool.timeout_seconds <= 0:
            raise ConfigurationError("Tool timeout must be positive")
        self._tools[tool.name] = tool

    def get(self, name: ToolName) -> TypedTool:
        """Return one registered tool or raise a configuration error."""
        try:
            return self._tools[name]
        except KeyError as error:
            raise ConfigurationError("Requested tool is not registered") from error

    def descriptors(self) -> list[ToolDescriptor]:
        """Return deterministic schemas and descriptions for registered tools."""
        return [
            ToolDescriptor(
                name=tool.name,
                description=tool.description,
                input_schema=tool.input_model.model_json_schema(),
                output_schema=tool.output_model.model_json_schema(),
                timeout_seconds=tool.timeout_seconds,
            )
            for tool in sorted(self._tools.values(), key=lambda item: item.name.value)
        ]

    def execute(self, request: ToolInvocationRequest) -> ToolInvocationResult:
        """Validate, execute, time, and safely package one registered invocation."""
        tool = self._tools.get(request.tool_name)
        if tool is None:
            return self._failure(
                request,
                ToolError(
                    error_type=ToolErrorType.TOOL_NOT_REGISTERED,
                    message="The requested tool is not registered.",
                ),
                0.0,
            )
        started = perf_counter()
        try:
            payload = tool.input_model.model_validate(request.payload)
        except ValidationError:
            latency_ms = (perf_counter() - started) * 1000
            return self._failure(
                request,
                ToolError(
                    error_type=ToolErrorType.INVALID_INPUT,
                    message="The tool input does not match its schema.",
                ),
                latency_ms,
            )
        try:
            output = tool.invoke(payload)
            if not isinstance(output, tool.output_model):
                latency_ms = (perf_counter() - started) * 1000
                return self._failure(
                    request,
                    ToolError(
                        error_type=ToolErrorType.TOOL_CONTRACT_ERROR,
                        message="The tool returned an incompatible output contract.",
                    ),
                    latency_ms,
                )
            latency_ms = (perf_counter() - started) * 1000
            if latency_ms > tool.timeout_seconds * 1000:
                return self._failure(
                    request,
                    self._domain_error(
                        OperationTimeoutError("Tool time budget exceeded")
                    ),
                    latency_ms,
                )
            result = ToolInvocationResult(
                invocation_id=request.invocation_id,
                tool_name=request.tool_name,
                success=True,
                output=output.model_dump(mode="json"),
                latency_ms=latency_ms,
            )
            self._log(result)
            return result
        except ValidationError:
            latency_ms = (perf_counter() - started) * 1000
            return self._failure(
                request,
                ToolError(
                    error_type=ToolErrorType.TOOL_CONTRACT_ERROR,
                    message="The tool produced data outside its output schema.",
                ),
                latency_ms,
            )
        except LegalAgenticRAGError as error:
            latency_ms = (perf_counter() - started) * 1000
            return self._failure(
                request,
                self._domain_error(error),
                latency_ms,
            )

    @staticmethod
    def _domain_error(error: LegalAgenticRAGError) -> ToolError:
        mappings: tuple[
            tuple[type[LegalAgenticRAGError], ToolErrorType, str, bool],
            ...,
        ] = (
            (
                InvalidUserInputError,
                ToolErrorType.INVALID_INPUT,
                "The tool input is not valid for this capability.",
                False,
            ),
            (
                ConfigurationError,
                ToolErrorType.CONFIGURATION_ERROR,
                "The tool is not configured correctly.",
                False,
            ),
            (
                DatasetSchemaError,
                ToolErrorType.DATASET_SCHEMA_ERROR,
                "The dataset schema is incompatible.",
                False,
            ),
            (
                DataValidationError,
                ToolErrorType.DATA_VALIDATION_ERROR,
                "The supplied domain data is invalid.",
                False,
            ),
            (
                ArtifactCompatibilityError,
                ToolErrorType.ARTIFACT_COMPATIBILITY_ERROR,
                "A required artifact is incompatible.",
                False,
            ),
            (
                BackendInitializationError,
                ToolErrorType.BACKEND_INITIALIZATION_ERROR,
                "A required backend is unavailable.",
                False,
            ),
            (
                RetrievalError,
                ToolErrorType.RETRIEVAL_ERROR,
                "Legal evidence retrieval failed.",
                True,
            ),
            (
                ModelError,
                ToolErrorType.MODEL_ERROR,
                "Model inference failed.",
                True,
            ),
            (
                OperationTimeoutError,
                ToolErrorType.TIMEOUT,
                "The tool exceeded its configured time budget.",
                True,
            ),
            (
                ExternalServiceError,
                ToolErrorType.EXTERNAL_SERVICE_ERROR,
                "An explicitly configured external service failed.",
                True,
            ),
        )
        for error_class, error_type, message, retryable in mappings:
            if isinstance(error, error_class):
                return ToolError(
                    error_type=error_type,
                    message=message,
                    retryable=retryable,
                )
        return ToolError(
            error_type=ToolErrorType.TOOL_CONTRACT_ERROR,
            message="The tool failed with an unsupported domain error.",
        )

    def _failure(
        self,
        request: ToolInvocationRequest,
        error: ToolError,
        latency_ms: float,
    ) -> ToolInvocationResult:
        result = ToolInvocationResult(
            invocation_id=request.invocation_id,
            tool_name=request.tool_name,
            success=False,
            error=error,
            latency_ms=latency_ms,
        )
        self._log(result)
        return result

    @staticmethod
    def _log(result: ToolInvocationResult) -> None:
        level = logging.INFO if result.success else logging.WARNING
        _LOGGER.log(
            level,
            "tool_invocation_completed",
            extra={
                "invocation_id": result.invocation_id,
                "tool_name": result.tool_name.value,
                "success": result.success,
                "error_type": (
                    result.error.error_type.value
                    if result.error is not None
                    else None
                ),
                "latency_ms": result.latency_ms,
            },
        )
