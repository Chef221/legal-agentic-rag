"""Tests for deterministic claim-to-evidence grounding."""

from legal_agentic_rag.configuration import ClaimVerificationConfig
from legal_agentic_rag.generation import RuleBasedClaimGroundingVerifier
from legal_agentic_rag.schemas import (
    AnswerResponse,
    Citation,
    ClaimSupportStatus,
    Evidence,
    RetrievalStrategy,
)


def _evidence() -> Evidence:
    return Evidence(
        evidence_id="E1",
        chunk_id="chunk-1",
        document_id="doc-1",
        text=(
            "Người lao động có đủ 12 tháng làm việc thì được nghỉ hằng năm "
            "12 ngày làm việc."
        ),
        article_number="113",
        document_number="45/2019/QH14",
    )


def _response(answer: str) -> AnswerResponse:
    return AnswerResponse(
        question="Người lao động được nghỉ hằng năm bao nhiêu ngày?",
        answer=answer,
        citations=[
            Citation(
                evidence_id="E1",
                chunk_id="chunk-1",
                document_id="doc-1",
                document_number="45/2019/QH14",
                article_number="113",
            )
        ],
        insufficient_evidence=False,
        retrieval_strategy=RetrievalStrategy.HYBRID_RERANK,
        trace_id="claim-trace",
        metadata={"semantic_synthesis": True},
    )


def test_supported_claim_preserves_marker_mapping_and_numeric_grounding() -> None:
    """An inline-cited claim supported by evidence receives a passing record."""
    claims, coverage, errors, warnings = (
        RuleBasedClaimGroundingVerifier().verify(
            _response(
                "Người lao động làm đủ 12 tháng được nghỉ hằng năm "
                "12 ngày làm việc. [E1]"
            ),
            [_evidence()],
        )
    )

    assert errors == []
    assert coverage == 1.0
    assert claims[0].status == ClaimSupportStatus.SUPPORTED
    assert claims[0].evidence_ids == ["E1"]
    assert claims[0].numeric_match is True
    assert "semantic_entailment_not_verified" in warnings


def test_uncited_claim_and_unused_response_citation_fail_closed() -> None:
    """A citation object alone cannot ground a claim without an inline marker."""
    claims, coverage, errors, _ = RuleBasedClaimGroundingVerifier().verify(
        _response("Người lao động được nghỉ hằng năm 12 ngày làm việc."),
        [_evidence()],
    )

    assert coverage == 0.0
    assert claims[0].status == ClaimSupportStatus.UNSUPPORTED
    assert "missing_inline_evidence" in claims[0].errors
    assert "unsupported_claim:C1" in errors
    assert "citation_not_used_in_answer:E1" in errors


def test_numeric_and_negation_changes_are_not_accepted_as_supported() -> None:
    """Changed quantities or newly introduced negation fail deterministic checks."""
    verifier = RuleBasedClaimGroundingVerifier()

    numeric, _, _, _ = verifier.verify(
        _response("Người lao động được nghỉ hằng năm 99 ngày. [E1]"),
        [_evidence()],
    )
    negated, _, _, _ = verifier.verify(
        _response("Người lao động không được nghỉ hằng năm. [E1]"),
        [_evidence()],
    )

    assert "numeric_mismatch" in numeric[0].errors
    assert numeric[0].numeric_match is False
    assert "negation_mismatch" in negated[0].errors
    assert negated[0].negation_match is False


def test_each_synthesized_legal_sentence_requires_its_own_marker() -> None:
    """A marker on the last sentence cannot silently cover an earlier claim."""
    claims, coverage, errors, _ = RuleBasedClaimGroundingVerifier().verify(
        _response(
            "Người lao động được nghỉ hằng năm. "
            "Thời gian nghỉ là 12 ngày. [E1]"
        ),
        [_evidence()],
    )

    assert len(claims) == 2
    assert claims[0].status == ClaimSupportStatus.UNSUPPORTED
    assert claims[1].status == ClaimSupportStatus.SUPPORTED
    assert coverage == 0.5
    assert errors == ["unsupported_claim:C1"]


def test_claim_limits_and_thresholds_are_configurable_but_bounded() -> None:
    """The verifier honors explicit lexical and claim-count policy."""
    verifier = RuleBasedClaimGroundingVerifier(
        ClaimVerificationConfig(
            minimum_lexical_support=1.0,
            max_claims=1,
        )
    )

    claims, _, errors, _ = verifier.verify(
        _response(
            "Người lao động được nghỉ 12 ngày. [E1] "
            "Quy định này áp dụng. [E1]"
        ),
        [_evidence()],
    )

    assert len(claims) == 1
    assert "claim_count_exceeded" in errors


# =============================================================================
# Step 11D: Minimal Legal-Reference Identity Gate Tests
# =============================================================================

def test_legal_reference_identity_gate_correct_statute_passes() -> None:
    """Claim naming correct legal statute matching linked evidence passes identity gate."""
    ev = Evidence(
        evidence_id="E1",
        chunk_id="chunk-1",
        document_id="doc-1",
        text="Công trình xây dựng nhà ở riêng lẻ phải phù hợp quy hoạch.",
        article_number="93",
        document_title="Luật Xây dựng",
        document_number="50/2014/QH13",
    )
    resp = _response("Theo Luật Xây dựng, công trình phải phù hợp quy hoạch. [E1]")
    claims, coverage, errors, _ = RuleBasedClaimGroundingVerifier().verify(resp, [ev])

    assert errors == []
    assert coverage == 1.0
    assert claims[0].status == ClaimSupportStatus.SUPPORTED
    assert "legal_reference_mismatch" not in claims[0].errors


def test_legal_reference_identity_gate_wrong_named_statute_rejected() -> None:
    """Claim naming wrong legal statute ('Bộ luật Dân sự' instead of 'Luật Xây dựng') fails closed."""
    ev = Evidence(
        evidence_id="E1",
        chunk_id="chunk-1",
        document_id="doc-1",
        text="Công trình xây dựng nhà ở riêng lẻ phải phù hợp quy hoạch.",
        article_number="93",
        document_title="Luật Xây dựng",
        document_number="50/2014/QH13",
    )
    resp = _response("Theo Bộ luật Dân sự, công trình phải phù hợp quy hoạch. [E1]")
    claims, coverage, errors, _ = RuleBasedClaimGroundingVerifier().verify(resp, [ev])

    assert coverage == 0.0
    assert claims[0].status == ClaimSupportStatus.UNSUPPORTED
    assert "legal_reference_mismatch" in claims[0].errors
    assert "unsupported_claim:C1" in errors


def test_legal_reference_identity_gate_step11c_q2_regression_rejected() -> None:
    """Step 11C Q2 hallucinated 'Điều 93 của Bộ luật Dân sự năm 2014' is rejected."""
    ev = Evidence(
        evidence_id="E1",
        chunk_id="chunk-1",
        document_id="doc-1",
        text="Điều 93. Điều kiện cấp giấy phép xây dựng đối với nhà ở riêng lẻ tại đô thị gồm phù hợp với quy hoạch chi tiết xây dựng.",
        article_number="93",
        document_title="Luật Xây dựng",
        document_number="50/2014/QH13",
    )
    resp = _response(
        "Theo Điều 93 của Bộ luật Dân sự năm 2014, điều kiện cấp giấy phép xây dựng nhà ở riêng lẻ tại đô thị gồm phù hợp với quy hoạch chi tiết xây dựng. [E1]"
    )
    claims, coverage, errors, _ = RuleBasedClaimGroundingVerifier().verify(resp, [ev])

    assert coverage == 0.0
    assert claims[0].status == ClaimSupportStatus.UNSUPPORTED
    assert "legal_reference_mismatch" in claims[0].errors
    assert "unsupported_claim:C1" in errors


def test_legal_reference_identity_gate_generic_legal_phrase_not_overmatched() -> None:
    """Generic phrase 'theo quy định của pháp luật' does not trigger identity gate."""
    ev = Evidence(
        evidence_id="E1",
        chunk_id="chunk-1",
        document_id="doc-1",
        text="Người lao động được nghỉ hằng năm theo quy định của pháp luật 12 ngày.",
        article_number="113",
        document_title="Bộ luật Lao động",
        document_number="45/2019/QH14",
    )
    resp = _response("Theo quy định của pháp luật, người lao động được nghỉ hằng năm 12 ngày. [E1]")
    claims, coverage, errors, _ = RuleBasedClaimGroundingVerifier().verify(resp, [ev])

    assert errors == []
    assert coverage == 1.0
    assert claims[0].status == ClaimSupportStatus.SUPPORTED
    assert "legal_reference_mismatch" not in claims[0].errors


def test_legal_reference_identity_gate_bare_article_number_not_overmatched() -> None:
    """Bare article reference 'Theo Điều 93' without named instrument does not trigger mismatch."""
    ev = Evidence(
        evidence_id="E1",
        chunk_id="chunk-1",
        document_id="doc-1",
        text="Điều 93. Điều kiện cấp giấy phép xây dựng gồm phù hợp quy hoạch.",
        article_number="93",
        document_title="Luật Xây dựng",
        document_number="50/2014/QH13",
    )
    resp = _response("Theo Điều 93, điều kiện cấp giấy phép xây dựng gồm phù hợp quy hoạch. [E1]")
    claims, coverage, errors, _ = RuleBasedClaimGroundingVerifier().verify(resp, [ev])

    assert errors == []
    assert coverage == 1.0
    assert claims[0].status == ClaimSupportStatus.SUPPORTED
    assert "legal_reference_mismatch" not in claims[0].errors


def test_legal_reference_identity_gate_numbered_document_mismatch_rejected() -> None:
    """Explicit document number mismatch (e.g. Nghị định 45/2022/NĐ-CP vs 155/2016/NĐ-CP) is rejected."""
    ev = Evidence(
        evidence_id="E1",
        chunk_id="chunk-1",
        document_id="doc-1",
        text="Thời hiệu xử phạt vi phạm hành chính trong lĩnh vực bảo vệ môi trường là 02 năm.",
        article_number="5",
        document_title="Nghị định quy định về xử phạt vi phạm hành chính trong lĩnh vực bảo vệ môi trường",
        document_number="155/2016/NĐ-CP",
    )
    resp = _response("Theo Nghị định 45/2022/NĐ-CP, thời hiệu xử phạt vi phạm hành chính là 02 năm. [E1]")
    claims, coverage, errors, _ = RuleBasedClaimGroundingVerifier().verify(resp, [ev])

    assert coverage == 0.0
    assert claims[0].status == ClaimSupportStatus.UNSUPPORTED
    assert "legal_reference_mismatch" in claims[0].errors
