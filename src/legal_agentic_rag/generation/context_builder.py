"""Convert ranked retrieval hits into bounded, traceable legal evidence."""

from __future__ import annotations

import re

from pydantic import ValidationError

from legal_agentic_rag.configuration.online import GenerationConfig
from legal_agentic_rag.exceptions import DataValidationError
from legal_agentic_rag.schemas.answering import ContextBuildResult, Evidence
from legal_agentic_rag.schemas.retrieval import RetrievalHit, RetrievalResponse

_TOKEN_PATTERN = re.compile(r"\w+|[^\w\s]", flags=re.UNICODE)


class ContextBuilder:
    """Select complete legal chunks without truncating their legal text."""

    def __init__(self, config: GenerationConfig | None = None) -> None:
        self._config = config or GenerationConfig()

    def build(self, response: RetrievalResponse) -> ContextBuildResult:
        """Build deterministic evidence within count and context-token limits."""
        unique_hits, duplicate_count = self._deduplicate(response.hits)
        selected: list[Evidence] = []
        omitted_count = 0
        estimated_tokens = 0
        warnings: list[str] = []
        for hit in sorted(
            unique_hits,
            key=lambda item: (
                self._is_inactive(item),
                item.rank,
                item.chunk_id,
            ),
        ):
            token_count = self._token_count(hit)
            if (
                len(selected) >= self._config.max_evidence
                or estimated_tokens + token_count
                > self._config.max_context_tokens
            ):
                omitted_count += 1
                continue
            evidence = self._evidence(hit, len(selected) + 1)
            selected.append(evidence)
            estimated_tokens += token_count
            if evidence.effect_status is None:
                warnings.append(f"effect_status_unknown:{evidence.evidence_id}")
            elif (
                evidence.effect_status.casefold()
                in self._config.inactive_effect_statuses
            ):
                warnings.append(f"inactive_effect_status:{evidence.evidence_id}")
        if omitted_count:
            warnings.append("context_budget_exhausted")
        if duplicate_count:
            warnings.append(f"duplicate_retrieval_hits_removed:{duplicate_count}")
        if not selected:
            warnings.append("no_selected_evidence")
        return ContextBuildResult(
            evidence=selected,
            input_hit_count=len(response.hits),
            selected_count=len(selected),
            omitted_hit_count=omitted_count,
            duplicate_hit_count=duplicate_count,
            estimated_token_count=estimated_tokens,
            truncated=bool(omitted_count),
            warnings=warnings,
        )

    @staticmethod
    def _deduplicate(
        hits: list[RetrievalHit],
    ) -> tuple[list[RetrievalHit], int]:
        by_chunk_id: dict[str, RetrievalHit] = {}
        duplicate_count = 0
        for hit in hits:
            existing = by_chunk_id.get(hit.chunk_id)
            if existing is None:
                by_chunk_id[hit.chunk_id] = hit
                continue
            if (
                existing.document_id != hit.document_id
                or existing.text != hit.text
                or existing.metadata != hit.metadata
            ):
                raise DataValidationError(
                    "Duplicate retrieval chunk has inconsistent legal payload"
                )
            duplicate_count += 1
        return list(by_chunk_id.values()), duplicate_count

    @staticmethod
    def _token_count(hit: RetrievalHit) -> int:
        value = hit.metadata.get("token_count")
        if isinstance(value, int) and not isinstance(value, bool) and value > 0:
            return value
        return max(1, len(_TOKEN_PATTERN.findall(hit.text)))

    def _is_inactive(self, hit: RetrievalHit) -> bool:
        status = hit.metadata.get("effect_status")
        return (
            isinstance(status, str)
            and status.casefold() in self._config.inactive_effect_statuses
        )

    @staticmethod
    def _evidence(hit: RetrievalHit, index: int) -> Evidence:
        structure = hit.metadata.get("structure")
        hierarchy = structure if isinstance(structure, dict) else {}
        try:
            return Evidence(
                evidence_id=f"E{index}",
                chunk_id=hit.chunk_id,
                document_id=hit.document_id,
                text=hit.text,
                article_number=hierarchy.get("article_number"),
                article_title=hierarchy.get("article_title"),
                document_title=hit.metadata.get("document_title"),
                document_number=hit.metadata.get("document_number"),
                document_type=hit.metadata.get("document_type"),
                effective_date=hit.metadata.get("effective_date"),
                expiry_date=hit.metadata.get("expiry_date"),
                effect_status=hit.metadata.get("effect_status"),
                source_url=hit.metadata.get("source_url"),
                metadata={
                    "retrieval_rank": hit.rank,
                    "retrieval_score": hit.score,
                    "retrieval_strategy": hit.strategy.value,
                    "retrieval_trace": hit.retrieval_trace.model_dump(mode="json"),
                    "chunk_metadata": hit.metadata,
                },
            )
        except ValidationError as error:
            raise DataValidationError(
                "Retrieval hit cannot be converted into legal evidence"
            ) from error
