"""Dependency-free grounded extractive answer generator."""

from __future__ import annotations

from collections.abc import Sequence

from legal_agentic_rag.exceptions import DataValidationError
from legal_agentic_rag.schemas.answering import AnswerResponse, Citation, Evidence
from legal_agentic_rag.schemas.retrieval import RetrievalQuery, RetrievalStrategy

ABSTENTION_TEXT = (
    "Hệ thống chưa tìm thấy căn cứ pháp luật đủ rõ trong dữ liệu hiện có "
    "để trả lời chắc chắn."
)


class ExtractiveAnswerGenerator:
    """Present selected evidence verbatim instead of synthesizing unsupported law."""

    def generate(
        self,
        query: RetrievalQuery,
        evidence: Sequence[Evidence],
        retrieval_strategy: RetrievalStrategy,
        trace_id: str,
    ) -> AnswerResponse:
        """Return cited evidence excerpts or an explicit empty-context abstention."""
        values = list(evidence)
        self._validate_evidence(values)
        if not values:
            return AnswerResponse(
                question=query.original_question,
                answer=ABSTENTION_TEXT,
                insufficient_evidence=True,
                warnings=["insufficient_evidence"],
                retrieval_strategy=retrieval_strategy,
                trace_id=trace_id,
                metadata={
                    "generator_backend": "extractive_v1",
                    "semantic_synthesis": False,
                },
            )
        excerpts = "\n\n".join(
            f"[{item.evidence_id}] {item.text}" for item in values
        )
        answer = (
            "Các căn cứ được truy xuất có nội dung như sau:\n\n"
            f"{excerpts}\n\n"
            "Cần đối chiếu tình trạng hiệu lực và hoàn cảnh áp dụng cụ thể."
        )
        warnings = [
            f"effect_status_unknown:{item.evidence_id}"
            for item in values
            if item.effect_status is None
        ]
        return AnswerResponse(
            question=query.original_question,
            answer=answer,
            citations=[self._citation(item) for item in values],
            insufficient_evidence=False,
            warnings=warnings,
            retrieval_strategy=retrieval_strategy,
            trace_id=trace_id,
            metadata={
                "generator_backend": "extractive_v1",
                "semantic_synthesis": False,
            },
        )

    @staticmethod
    def _validate_evidence(evidence: list[Evidence]) -> None:
        evidence_ids = [item.evidence_id for item in evidence]
        chunk_ids = [item.chunk_id for item in evidence]
        if (
            len(evidence_ids) != len(set(evidence_ids))
            or len(chunk_ids) != len(set(chunk_ids))
        ):
            raise DataValidationError(
                "Answer generation requires unique evidence identities"
            )

    @staticmethod
    def _citation(evidence: Evidence) -> Citation:
        return Citation(
            evidence_id=evidence.evidence_id,
            chunk_id=evidence.chunk_id,
            document_id=evidence.document_id,
            document_title=evidence.document_title,
            document_number=evidence.document_number,
            article_number=evidence.article_number,
            source_url=evidence.source_url,
        )
