"""Convert ranked retrieval hits into bounded, traceable legal evidence."""

from __future__ import annotations

import re

from pydantic import ValidationError

from legal_agentic_rag.configuration.online import (
    EvidenceSelectionConfig,
    GenerationConfig,
)
from legal_agentic_rag.exceptions import DataValidationError
from legal_agentic_rag.generation.evidence_selector import (
    EvidenceSelector,
    ScoredEvidenceCandidate,
)
from legal_agentic_rag.schemas.answering import (
    ContextBuildResult,
    Evidence,
    EvidenceApplicability,
    EvidenceSelectionReason,
    EvidenceSelectionTrace,
)
from legal_agentic_rag.schemas.retrieval import RetrievalHit, RetrievalResponse

_TOKEN_PATTERN = re.compile(r"\w+|[^\w\s]", flags=re.UNICODE)


class ContextBuilder:
    """Select complete legal chunks without truncating their legal text."""

    def __init__(
        self,
        config: GenerationConfig | None = None,
        selection_config: EvidenceSelectionConfig | None = None,
    ) -> None:
        self._config = config or GenerationConfig()
        self._selection_config = selection_config or EvidenceSelectionConfig()
        self._selector = EvidenceSelector(self._selection_config, self._config)

    def build(self, response: RetrievalResponse) -> ContextBuildResult:
        """Build deterministic evidence within count and context-token limits."""
        unique_hits, duplicate_count = self._deduplicate(response.hits)
        selected: list[Evidence] = []
        omitted_count = 0
        estimated_tokens = 0
        warnings: list[str] = []
        selection_trace: list[EvidenceSelectionTrace] = []
        document_counts: dict[str, int] = {}
        article_counts: dict[tuple[str, str], int] = {}
        diversity_omitted = 0
        max_evidence_omitted = 0
        token_budget_omitted = 0
        for candidate in self._selector.score(response.query, unique_hits):
            hit = candidate.hit
            token_count = self._token_count(hit)
            article_key = self._article_key(hit)
            if (
                document_counts.get(hit.document_id, 0)
                >= self._selection_config.max_per_document
                or (
                    article_key is not None
                    and article_counts.get(article_key, 0)
                    >= self._selection_config.max_per_article
                )
            ):
                omitted_count += 1
                diversity_omitted += 1
                selection_trace.append(
                    self._trace(
                        candidate,
                        reason=EvidenceSelectionReason.DIVERSITY_LIMIT,
                    )
                )
                continue
            if len(selected) >= self._config.max_evidence:
                omitted_count += 1
                max_evidence_omitted += 1
                selection_trace.append(
                    self._trace(
                        candidate,
                        reason=EvidenceSelectionReason.MAX_EVIDENCE,
                    )
                )
                continue
            if (
                estimated_tokens + token_count
                > self._config.max_context_tokens
            ):
                omitted_count += 1
                token_budget_omitted += 1
                selection_trace.append(
                    self._trace(
                        candidate,
                        reason=EvidenceSelectionReason.TOKEN_BUDGET,
                    )
                )
                continue
            trace = self._trace(
                candidate,
                reason=EvidenceSelectionReason.SELECTED,
                selection_rank=len(selected) + 1,
            )
            evidence = self._evidence(hit, len(selected) + 1, trace)
            selected.append(evidence)
            selection_trace.append(trace)
            estimated_tokens += token_count
            document_counts[hit.document_id] = (
                document_counts.get(hit.document_id, 0) + 1
            )
            if article_key is not None:
                article_counts[article_key] = (
                    article_counts.get(article_key, 0) + 1
                )
            if evidence.effect_status is None:
                warnings.append(f"effect_status_unknown:{evidence.evidence_id}")
            elif (
                evidence.effect_status.casefold()
                in self._config.inactive_effect_statuses
            ):
                warnings.append(f"inactive_effect_status:{evidence.evidence_id}")
            if (
                trace.applicability
                == EvidenceApplicability.REFERENCE_MISMATCH
            ):
                warnings.append(
                    f"explicit_reference_mismatch:{evidence.evidence_id}"
                )
        if (
            response.query.query_analysis is not None
            and response.query.query_analysis.has_explicit_legal_reference
            and not any(
                item.applicability == EvidenceApplicability.EXPLICIT_MATCH
                and item.selected
                for item in selection_trace
            )
        ):
            warnings.append("explicit_reference_not_selected")
        if token_budget_omitted:
            warnings.append("context_budget_exhausted")
        if max_evidence_omitted:
            warnings.append(
                f"max_evidence_omissions:{max_evidence_omitted}"
            )
        if duplicate_count:
            warnings.append(f"duplicate_retrieval_hits_removed:{duplicate_count}")
        if diversity_omitted:
            warnings.append(f"diversity_limit_omissions:{diversity_omitted}")
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
            selection_trace=selection_trace,
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

    @staticmethod
    def _article_key(hit: RetrievalHit) -> tuple[str, str] | None:
        """Return a stable document/article identity when metadata provides one."""
        structure = hit.metadata.get("structure")
        if not isinstance(structure, dict):
            return None
        article_number = structure.get("article_number")
        if not isinstance(article_number, str) or not article_number.strip():
            return None
        return hit.document_id, article_number.strip().casefold()

    @staticmethod
    def _evidence(
        hit: RetrievalHit,
        index: int,
        trace: EvidenceSelectionTrace,
    ) -> Evidence:
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
                    "evidence_selection": trace.model_dump(mode="json"),
                    "chunk_metadata": hit.metadata,
                },
            )
        except ValidationError as error:
            raise DataValidationError(
                "Retrieval hit cannot be converted into legal evidence"
            ) from error

    @staticmethod
    def _trace(
        candidate: ScoredEvidenceCandidate,
        *,
        reason: EvidenceSelectionReason,
        selection_rank: int | None = None,
    ) -> EvidenceSelectionTrace:
        selected = reason == EvidenceSelectionReason.SELECTED
        return EvidenceSelectionTrace(
            chunk_id=candidate.hit.chunk_id,
            source_rank=candidate.hit.rank,
            selection_rank=selection_rank,
            applicability=candidate.applicability,
            document_reference_match=(
                candidate.document_reference_match
            ),
            article_reference_match=candidate.article_reference_match,
            lexical_overlap_score=candidate.lexical_overlap_score,
            selection_score=candidate.selection_score,
            selected=selected,
            reason=reason,
        )
