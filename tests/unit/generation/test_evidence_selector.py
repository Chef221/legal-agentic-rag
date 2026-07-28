"""Tests for deterministic evidence applicability scoring."""

from legal_agentic_rag.configuration import (
    EvidenceSelectionConfig,
    GenerationConfig,
)
from legal_agentic_rag.generation import EvidenceSelector
from legal_agentic_rag.schemas import (
    EvidenceApplicability,
    QueryAnalysis,
    RetrievalHit,
    RetrievalQuery,
    RetrievalStrategy,
)


def _query() -> RetrievalQuery:
    return RetrievalQuery(
        query_id="selection-query",
        original_question=(
            "Điều 113 Luật số 45/2019/QH14 quy định nghỉ hằng năm thế nào?"
        ),
        normalized_question=(
            "Điều 113 Luật số 45/2019/QH14 quy định nghỉ hằng năm thế nào?"
        ),
        query_analysis=QueryAnalysis(
            document_numbers=["45/2019/QH14"],
            article_numbers=["113"],
        ),
        top_k=2,
        candidate_k=2,
    )


def _hit(
    chunk_id: str,
    rank: int,
    *,
    document_number: str,
    article_number: str,
    effect_status: str | None = "còn hiệu lực",
) -> RetrievalHit:
    return RetrievalHit(
        chunk_id=chunk_id,
        document_id=f"doc-{chunk_id}",
        rank=rank,
        score=float(10 - rank),
        strategy=RetrievalStrategy.HYBRID_RERANK,
        text="Người lao động được nghỉ hằng năm.",
        metadata={
            "document_title": "Bộ luật Lao động",
            "document_number": document_number,
            "effect_status": effect_status,
            "structure": {"article_number": article_number},
        },
    )


def test_explicit_document_and_article_match_outrank_raw_rank() -> None:
    """User-supplied legal references can promote the matching provision."""
    selector = EvidenceSelector()

    scored = selector.score(
        _query(),
        [
            _hit(
                "wrong",
                1,
                document_number="145/2020/NĐ-CP",
                article_number="113",
            ),
            _hit(
                "matching",
                2,
                document_number="45/2019/QH14",
                article_number="113",
            ),
        ],
    )

    assert [item.hit.chunk_id for item in scored] == ["matching", "wrong"]
    assert scored[0].applicability == EvidenceApplicability.EXPLICIT_MATCH
    assert (
        scored[1].applicability
        == EvidenceApplicability.REFERENCE_MISMATCH
    )
    assert scored[0].document_reference_match is True
    assert scored[0].article_reference_match is True


def test_configured_inactive_status_is_penalized_without_guessing_labels() -> None:
    """Only an explicitly configured inactive label changes selection score."""
    selector = EvidenceSelector(
        EvidenceSelectionConfig(inactive_penalty=2.0),
        GenerationConfig(
            inactive_effect_statuses=frozenset({"hết hiệu lực"})
        ),
    )

    scored = selector.score(
        _query().model_copy(update={"query_analysis": None}),
        [
            _hit(
                "inactive",
                1,
                document_number="1/2020/QH",
                article_number="1",
                effect_status="Hết hiệu lực",
            ),
            _hit(
                "active",
                2,
                document_number="2/2020/QH",
                article_number="2",
            ),
        ],
    )

    assert [item.hit.chunk_id for item in scored] == ["active", "inactive"]
    assert scored[1].applicability == EvidenceApplicability.INACTIVE


def test_selector_is_deterministic_for_equal_candidates() -> None:
    """Stable source rank and chunk ID break equal-score ties."""
    query = _query().model_copy(update={"query_analysis": None})
    hit_b = _hit(
        "b",
        1,
        document_number="1/2020/QH",
        article_number="1",
    )
    hit_a = hit_b.model_copy(update={"chunk_id": "a"})

    first = EvidenceSelector().score(query, [hit_b, hit_a])
    second = EvidenceSelector().score(query, [hit_a, hit_b])

    assert [item.hit.chunk_id for item in first] == ["a", "b"]
    assert [item.hit.chunk_id for item in second] == ["a", "b"]
