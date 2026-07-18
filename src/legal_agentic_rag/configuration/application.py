"""Top-level composition of typed application configuration."""

from pydantic import BaseModel, ConfigDict, Field

from legal_agentic_rag.configuration.artifacts import ArtifactConfig
from legal_agentic_rag.configuration.observability import LoggingConfig
from legal_agentic_rag.configuration.offline import OfflineConfig
from legal_agentic_rag.configuration.online import OnlineConfig


class ApplicationConfig(BaseModel):
    """Framework-neutral composition of all Milestone 1 config schemas."""

    model_config = ConfigDict(extra="forbid")

    artifacts: ArtifactConfig
    offline: OfflineConfig
    online: OnlineConfig
    logging: LoggingConfig = Field(default_factory=LoggingConfig)
