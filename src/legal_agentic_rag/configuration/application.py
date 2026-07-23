"""Top-level composition of typed application configuration."""

from pydantic import BaseModel, ConfigDict, Field, model_validator

from legal_agentic_rag.configuration.artifacts import ArtifactConfig
from legal_agentic_rag.configuration.evaluation import EvaluationConfig
from legal_agentic_rag.configuration.observability import LoggingConfig
from legal_agentic_rag.configuration.offline import OfflineConfig
from legal_agentic_rag.configuration.online import OnlineConfig
from legal_agentic_rag.configuration.serving import ServingConfig


class ApplicationConfig(BaseModel):
    """Framework-neutral composition of offline, online, and serving config."""

    model_config = ConfigDict(extra="forbid")

    artifacts: ArtifactConfig
    offline: OfflineConfig
    online: OnlineConfig
    logging: LoggingConfig = Field(default_factory=LoggingConfig)
    serving: ServingConfig = Field(default_factory=ServingConfig)
    evaluation: EvaluationConfig = Field(default_factory=EvaluationConfig)

    @model_validator(mode="after")
    def validate_serving_retrieval_limits(self) -> "ApplicationConfig":
        """Ensure default online requests fit the public serving policy."""
        retrieval = self.online.retrieval
        if retrieval.top_k > self.serving.max_top_k:
            raise ValueError("online top_k exceeds serving max_top_k")
        maximum_candidate_k = min(
            self.serving.max_candidate_k,
            self.online.reranker.max_candidates,
        )
        if retrieval.candidate_k > maximum_candidate_k:
            raise ValueError(
                "online candidate_k exceeds the serving or reranker limit"
            )
        if (
            self.evaluation.strategy.value in {"hybrid_rerank", "graph"}
            and self.evaluation.candidate_k
            > self.online.reranker.max_candidates
        ):
            raise ValueError(
                "evaluation candidate_k exceeds the reranker limit"
            )
        return self
