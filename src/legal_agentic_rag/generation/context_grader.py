"""Transparent structural context grading for the fixed baseline."""

from __future__ import annotations

from collections.abc import Sequence

from legal_agentic_rag.configuration.online import ContextGradingConfig
from legal_agentic_rag.schemas.answering import ContextGrade, Evidence
from legal_agentic_rag.schemas.retrieval import RetrievalQuery


class RuleBasedContextGrader:
    """Grade minimum evidence structure without claiming semantic verification."""

    def __init__(self, config: ContextGradingConfig | None = None) -> None:
        self._config = config or ContextGradingConfig()

    def grade(
        self,
        query: RetrievalQuery,
        evidence: Sequence[Evidence],
    ) -> ContextGrade:
        """Return deterministic structural sufficiency and explicit limitations."""
        values = list(evidence)
        evidence_ids = [item.evidence_id for item in values]
        chunk_ids = [item.chunk_id for item in values]
        unique = (
            len(evidence_ids) == len(set(evidence_ids))
            and len(chunk_ids) == len(set(chunk_ids))
        )
        missing_aspects: list[str] = []
        if len(values) < self._config.minimum_evidence_count:
            missing_aspects.append("minimum_evidence_count")
        if self._config.require_document_number and any(
            item.document_number is None for item in values
        ):
            missing_aspects.append("document_number")
        if self._config.require_article_number and any(
            item.article_number is None for item in values
        ):
            missing_aspects.append("article_number")
        if not unique:
            missing_aspects.append("unique_evidence_identity")
        relevance_score = 1.0 if values else 0.0
        coverage_score = min(
            len(values) / self._config.minimum_evidence_count,
            1.0,
        )
        consistency_score = 1.0 if unique else 0.0
        score = min(relevance_score, coverage_score, consistency_score)
        warnings = ["semantic_relevance_not_verified"]
        warnings.extend(
            f"effect_status_unknown:{item.evidence_id}"
            for item in values
            if item.effect_status is None
        )
        return ContextGrade(
            is_sufficient=not missing_aspects,
            score=score,
            relevance_score=relevance_score,
            coverage_score=coverage_score,
            consistency_score=consistency_score,
            missing_aspects=missing_aspects,
            warnings=warnings,
            metadata={
                "policy": "structural_rule_v1",
                "semantic_relevance_checked": False,
                "query_id": query.query_id,
                "evidence_count": len(values),
            },
        )
