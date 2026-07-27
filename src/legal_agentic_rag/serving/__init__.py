"""FastAPI API and diagnostic UI over the immutable online runtime."""

from legal_agentic_rag.serving.api import create_app
from legal_agentic_rag.serving.config_loader import load_application_config
from legal_agentic_rag.serving.query_service import ServingService

__all__ = ["ServingService", "create_app", "load_application_config"]
