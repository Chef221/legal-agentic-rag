"""Unit tests for M54 Preprocessing V2 retrieval unit materialization."""

from hashlib import sha256

from legal_agentic_rag.offline.preprocessing_v2.parser import parse_provisions_from_document
from legal_agentic_rag.offline.preprocessing_v2.retrieval_units import (
    materialize_retrieval_units,
    materialize_retrieval_units_v2,
)
from legal_agentic_rag.schemas.preprocessing_v2 import (
    CanonicalDocumentV2,
    DocumentIdentityV2,
    DocumentRawV2,
    DocumentSourceV2,
    PreprocessingV2SegmentationProfile,
)


def test_parent_preamble_and_unverified_doc_number():
    text = """Điều 1. Quy định chung
Các trường hợp sau đây được áp dụng quy định:
1. Trường hợp thứ nhất áp dụng theo luật.
2. Trường hợp thứ hai áp dụng theo nghị định.
"""
    doc_id = "doc:uitdsc2026:ambiguous_doc"
    raw_sha = sha256(text.encode("utf-8")).hexdigest()
    doc = CanonicalDocumentV2(
        document_id=doc_id,
        source=DocumentSourceV2(context_id="100", member_name="context_100.json", raw_passage_sha256=raw_sha),
        raw=DocumentRawV2(text=text),
        identity=DocumentIdentityV2(status="AMBIGUOUS", document_number=None),
        authority_text=text,
        authority_text_sha256=raw_sha,
    )
    provs, _ = parse_provisions_from_document(doc_id, text)
    rus = materialize_retrieval_units([doc], provs, PreprocessingV2SegmentationProfile())

    # Verify preamble was materialized
    preambles = [r for r in rus if r.strategy == "ARTICLE_PREAMBLE"]
    assert len(preambles) == 1
    assert "Các trường hợp sau đây được áp dụng" in preambles[0].authority_text

    # Verify unverified document number is omitted in header
    for r in rus:
        assert "Số ký hiệu:" not in r.retrieval_text


def test_duplicate_article_preamble_materialization():
    text = """Điều 1. Lần một
Lời dẫn điều một lần một:
1. Khoản 1 lần 1.

Điều 1. Lần hai
Lời dẫn điều một lần hai:
1. Khoản 1 lần 2.
"""
    doc_id = "doc:uitdsc2026:dup_doc"
    raw_sha = sha256(text.encode("utf-8")).hexdigest()
    doc = CanonicalDocumentV2(
        document_id=doc_id,
        source=DocumentSourceV2(context_id="101", member_name="context_101.json", raw_passage_sha256=raw_sha),
        raw=DocumentRawV2(text=text),
        identity=DocumentIdentityV2(status="EXPLICIT", document_number="01/2020/TT-BTTTT"),
        authority_text=text,
        authority_text_sha256=raw_sha,
    )
    provs, _ = parse_provisions_from_document(doc_id, text)
    rus = materialize_retrieval_units([doc], provs, PreprocessingV2SegmentationProfile())

    ru_ids = [r.retrieval_unit_id for r in rus]
    assert "doc:uitdsc2026:dup_doc::art:1::preamble" in ru_ids
    assert "doc:uitdsc2026:dup_doc::art:1~2::preamble" in ru_ids
    assert "doc:uitdsc2026:dup_doc::art:1::cl:1" in ru_ids
    assert "doc:uitdsc2026:dup_doc::art:1::cl:1~2" in ru_ids
