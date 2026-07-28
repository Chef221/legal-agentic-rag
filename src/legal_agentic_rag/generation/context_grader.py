"""Transparent structural context grading for the fixed baseline."""

from __future__ import annotations

from collections.abc import Sequence

from legal_agentic_rag.configuration.online import ContextGradingConfig
from legal_agentic_rag.schemas.answering import (
    ContextGrade,
    Evidence,
    EvidenceApplicability,
)
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
        reference_coverage = self._reference_coverage(query, values)
        if reference_coverage["document"] is False:
            missing_aspects.append("document_reference_match")
        if reference_coverage["article"] is False:
            missing_aspects.append("article_reference_match")
        relevance_score = self._relevance_score(values)
        coverage_score = min(
            len(values) / self._config.minimum_evidence_count,
            1.0,
        )
        required_reference_scores = [
            float(value)
            for value in reference_coverage.values()
            if value is not None
        ]
        if required_reference_scores:
            coverage_score = min(
                coverage_score,
                sum(required_reference_scores)
                / len(required_reference_scores),
            )
        consistency_score = 1.0 if unique else 0.0
        applicability_score, applicability_counts = (
            self._applicability_score(values)
        )
        if values and applicability_score == 0:
            missing_aspects.append("applicable_evidence")
        score = min(
            coverage_score,
            consistency_score,
            applicability_score,
        )
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
            applicability_score=applicability_score,
            missing_aspects=missing_aspects,
            warnings=warnings,
            metadata={
                "policy": "structural_applicability_rule_v2",
                "semantic_relevance_checked": False,
                "legal_applicability_interpreted": False,
                "query_id": query.query_id,
                "evidence_count": len(values),
                "reference_coverage": reference_coverage,
                "applicability_counts": applicability_counts,
            },
        )

    @staticmethod
    def _reference_coverage(
        query: RetrievalQuery,
        evidence: list[Evidence],
    ) -> dict[str, bool | None]:
        analysis = query.query_analysis
        if analysis is None:
            return {"document": None, "article": None}
        return {
            "document": RuleBasedContextGrader._any_selection_match(
                evidence,
                "document_reference_match",
            )
            if analysis.document_numbers
            else None,
            "article": RuleBasedContextGrader._any_selection_match(
                evidence,
                "article_reference_match",
            )
            if analysis.article_numbers
            else None,
        }

    @staticmethod
    def _any_selection_match(
        evidence: list[Evidence],
        field_name: str,
    ) -> bool:
        return any(
            RuleBasedContextGrader._selection(item).get(field_name) is True
            for item in evidence
        )

    @staticmethod
    def _relevance_score(evidence: list[Evidence]) -> float:
        if not evidence:
            return 0.0
        scores = [
            value
            for item in evidence
            if isinstance(
                value := RuleBasedContextGrader._selection(item).get(
                    "lexical_overlap_score"
                ),
                (int, float),
            )
            and not isinstance(value, bool)
        ]
        return max(scores) if scores else 1.0

    @staticmethod
    def _applicability_score(
        evidence: list[Evidence],
    ) -> tuple[float, dict[str, int]]:
        if not evidence:
            return 0.0, {}
        statuses = [
            value
            for item in evidence
            if isinstance(
                value := RuleBasedContextGrader._selection(item).get(
                    "applicability"
                ),
                str,
            )
        ]
        if not statuses:
            return 1.0, {}
        counts = {
            status: statuses.count(status) for status in sorted(set(statuses))
        }
        rejected = {
            EvidenceApplicability.INACTIVE.value,
            EvidenceApplicability.REFERENCE_MISMATCH.value,
        }
        accepted_count = sum(status not in rejected for status in statuses)
        return accepted_count / len(statuses), counts

    @staticmethod
    def _selection(evidence: Evidence) -> dict[str, object]:
        value = evidence.metadata.get("evidence_selection")
        return value if isinstance(value, dict) else {}
