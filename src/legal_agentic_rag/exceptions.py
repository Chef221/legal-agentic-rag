"""Domain-specific exception taxonomy for the application boundary."""


class LegalAgenticRAGError(Exception):
    """Base exception for errors intentionally exposed by this package."""


class ConfigurationError(LegalAgenticRAGError):
    """Raised when application configuration is invalid."""


class DatasetSchemaError(LegalAgenticRAGError):
    """Raised when an input dataset does not match its expected boundary."""


class DataValidationError(LegalAgenticRAGError):
    """Raised when normalized or processed data violates a contract."""


class ArtifactCompatibilityError(LegalAgenticRAGError):
    """Raised when a persisted artifact is incompatible with the runtime."""


class BackendInitializationError(LegalAgenticRAGError):
    """Raised when a configured backend cannot be initialized."""


class RetrievalError(LegalAgenticRAGError):
    """Raised when a retrieval operation fails."""


class ModelError(LegalAgenticRAGError):
    """Raised when a model provider fails."""


class StructuredGenerationError(ModelError):
    """Raised for a sanitized, classified model-output contract failure."""

    def __init__(self, message: str, *, failure_code: str) -> None:
        super().__init__(message)
        self.failure_code = failure_code


class OperationTimeoutError(LegalAgenticRAGError):
    """Raised when a bounded operation exceeds its timeout."""


class ExternalServiceError(LegalAgenticRAGError):
    """Raised when an explicitly configured external service fails."""


class InvalidUserInputError(LegalAgenticRAGError):
    """Raised when an online request is invalid."""
