"""Unit tests for M54 Preprocessing V2 legal reference resolution."""

from hashlib import sha256
from legal_agentic_rag.offline.preprocessing_v2.parser import parse_provisions_from_document
from legal_agentic_rag.offline.preprocessing_v2.references import extract_legal_references
from legal_agentic_rag.schemas.preprocessing_v2 import (
    CanonicalDocumentV2,
    DocumentIdentityEvidenceV2,
    DocumentIdentityV2,
    DocumentRawV2,
    DocumentSourceV2,
)


def test_reference_deterministic_two_index_resolution():
    text1 = "Căn cứ Nghị định số 15/2020/NĐ-CP quy định xử phạt vi phạm hành chính."
    text2 = "Căn cứ Nghị định số 99/2023/NĐ-CP quy định chi tiết thi hành."

    sha1 = sha256(text1.encode("utf-8")).hexdigest()
    sha2 = sha256(text2.encode("utf-8")).hexdigest()

    doc1 = CanonicalDocumentV2(
        document_id="doc:uitdsc2026:1",
        source=DocumentSourceV2(context_id="1", member_name="context_1.json", raw_passage_sha256=sha1),
        raw=DocumentRawV2(text=text1),
        identity=DocumentIdentityV2(
            status="EXPLICIT",
            document_number="15/2020/NĐ-CP",
            candidate_document_numbers=["15/2020/NĐ-CP"],
            evidence=[DocumentIdentityEvidenceV2(source="opening_text", matched_text="15/2020/NĐ-CP")],
        ),
        authority_text=text1,
        authority_text_sha256=sha1,
    )
    doc2 = CanonicalDocumentV2(
        document_id="doc:uitdsc2026:2",
        source=DocumentSourceV2(context_id="2", member_name="context_2.json", raw_passage_sha256=sha2),
        raw=DocumentRawV2(text=text2),
        identity=DocumentIdentityV2(
            status="AMBIGUOUS",
            document_number=None,
            candidate_document_numbers=["99/2023/NĐ-CP"],
            evidence=[DocumentIdentityEvidenceV2(source="raw_name", matched_text="99/2023/NĐ-CP")],
        ),
        authority_text=text2,
        authority_text_sha256=sha2,
    )

    provs1, _ = parse_provisions_from_document("doc:uitdsc2026:1", text1)
    provs2, _ = parse_provisions_from_document("doc:uitdsc2026:2", text2)

    refs = extract_legal_references([doc1, doc2], provs1 + provs2)

    # Ref to 15/2020/NĐ-CP should be RESOLVED_UNIQUE to doc:uitdsc2026:1
    ref1 = next(r for r in refs if r.target.document_number_normalized == "15/2020/NĐ-CP")
    assert ref1.resolution.status == "RESOLVED_UNIQUE"
    assert ref1.resolution.target_document_id == "doc:uitdsc2026:1"

    # Ref to 99/2023/NĐ-CP matches ambiguous candidate index -> RESOLVED_AMBIGUOUS
    ref2 = next(r for r in refs if r.target.document_number_normalized == "99/2023/NĐ-CP")
    assert ref2.resolution.status == "RESOLVED_AMBIGUOUS"
    assert ref2.resolution.target_document_id is None
    assert "doc:uitdsc2026:2" in ref2.resolution.candidate_document_ids
