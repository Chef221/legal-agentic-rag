"""Tests for extractive answers and referential citation verification."""

from legal_agentic_rag.generation import (
    ExtractiveAnswerGenerator,
    RuleBasedCitationVerifier,
)
from legal_agentic_rag.contracts import AnswerGenerator, CitationVerifier
from legal_agentic_rag.schemas import (
    AnswerResponse,
    Citation,
    Evidence,
    RetrievalQuery,
    RetrievalStrategy,
)


def _query() -> RetrievalQuery:
    return RetrievalQuery(
        query_id="answer-query",
        original_question="Điều kiện áp dụng là gì?",
        normalized_question="điều kiện áp dụng",
        top_k=1,
        candidate_k=1,
    )


def _evidence() -> Evidence:
    return Evidence(
        evidence_id="E1",
        chunk_id="chunk-1",
        document_id="doc-1",
        text="Chỉ áp dụng trong trường hợp được pháp luật quy định.",
        article_number="10",
        document_title="Luật thử nghiệm",
        document_number="01/2026/QH",
        source_url="https://example.test/doc-1",
    )


def test_extractive_generator_uses_verbatim_evidence_and_exact_citation() -> None:
    """The reference answer adds no legal proposition beyond supplied evidence."""
    evidence = _evidence()
    response = ExtractiveAnswerGenerator().generate(
        _query(),
        [evidence],
        RetrievalStrategy.HYBRID_RERANK,
        "answer-query",
    )
    verification = RuleBasedCitationVerifier().verify(response, [evidence])

    assert f"[E1] {evidence.text}" in response.answer
    assert response.citations[0].article_number == "10"
    assert response.metadata["semantic_synthesis"] is False
    assert response.insufficient_evidence is False
    assert verification.is_valid is True
    assert verification.valid_citations == response.citations
    assert "semantic_claim_verification_not_performed" in verification.warnings
    assert isinstance(ExtractiveAnswerGenerator(), AnswerGenerator)
    assert isinstance(RuleBasedCitationVerifier(), CitationVerifier)


def test_extractive_generator_abstains_without_evidence() -> None:
    """Empty context produces no invented answer or citation."""
    response = ExtractiveAnswerGenerator().generate(
        _query(),
        [],
        RetrievalStrategy.HYBRID,
        "answer-query",
    )

    assert response.insufficient_evidence is True
    assert response.citations == []
    assert RuleBasedCitationVerifier().verify(response, []).is_valid is True


def test_rule_verifier_rejects_wrong_identity_and_uncited_grounded_answer() -> None:
    """Citation metadata must match evidence exactly and grounded text must cite."""
    evidence = _evidence()
    wrong_citation = Citation(
        evidence_id="E1",
        chunk_id="invented",
        document_id="doc-1",
    )
    wrong = AnswerResponse(
        question=_query().original_question,
        answer="Sai",
        citations=[wrong_citation],
        insufficient_evidence=False,
        retrieval_strategy=RetrievalStrategy.HYBRID,
        trace_id="answer-query",
    )
    missing = wrong.model_copy(update={"citations": []})

    wrong_result = RuleBasedCitationVerifier().verify(wrong, [evidence])
    missing_result = RuleBasedCitationVerifier().verify(missing, [evidence])

    assert wrong_result.is_valid is False
    assert wrong_result.invalid_citations == [wrong_citation]
    assert missing_result.is_valid is False
    assert missing_result.errors == ["grounded_answer_requires_citation"]
