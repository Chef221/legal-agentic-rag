"""Unit tests for M54 Preprocessing V2 Pydantic schemas."""

from hashlib import sha256
import pytest
from pydantic import ValidationError

from legal_agentic_rag.schemas.preprocessing_v2 import (
    CanonicalDocumentV2,
    DocumentIdentityEvidenceV2,
    DocumentIdentityV2,
    DocumentRawV2,
    DocumentSourceV2,
    HeadingPathItemV2,
    LegalProvisionV2,
    LegalReferenceResolutionV2,
    LegalReferenceTargetV2,
    LegalReferenceV2,
    RetrievalUnitDocumentIdentityV2,
    RetrievalUnitHierarchyV2,
    RetrievalUnitV2,
    TextSpanV2,
)


def test_document_identity_invariants():
    # EXPLICIT with doc num is valid
    ident = DocumentIdentityV2(
        status="EXPLICIT",
        document_number="12/2024/NĐ-CP",
    )
    assert ident.document_number == "12/2024/NĐ-CP"

    # AMBIGUOUS with doc num must fail
    with pytest.raises(ValidationError):
        DocumentIdentityV2(
            status="AMBIGUOUS",
            document_number="12/2024/NĐ-CP",
        )

    # UNRESOLVED with doc num must fail
    with pytest.raises(ValidationError):
        DocumentIdentityV2(
            status="UNRESOLVED",
            document_number="12/2024/NĐ-CP",
        )


def test_canonical_document_hashes_validation():
    raw_text = "Nghị định số 12/2024/NĐ-CP quy định chi tiết..."
    auth_text = "Nghị định số 12/2024/NĐ-CP quy định chi tiết..."
    raw_sha = sha256(raw_text.encode("utf-8")).hexdigest()
    auth_sha = sha256(auth_text.encode("utf-8")).hexdigest()

    doc = CanonicalDocumentV2(
        document_id="doc:uitdsc2026:1",
        source=DocumentSourceV2(
            context_id="1",
            member_name="context_1.json",
            raw_passage_sha256=raw_sha,
        ),
        raw=DocumentRawV2(text=raw_text),
        identity=DocumentIdentityV2(status="EXPLICIT", document_number="12/2024/NĐ-CP"),
        authority_text=auth_text,
        authority_text_sha256=auth_sha,
    )
    assert doc.document_id == "doc:uitdsc2026:1"

    # Mismatched raw hash must fail
    with pytest.raises(ValidationError):
        CanonicalDocumentV2(
            document_id="doc:uitdsc2026:1",
            source=DocumentSourceV2(
                context_id="1",
                member_name="context_1.json",
                raw_passage_sha256="wrong_sha",
            ),
            raw=DocumentRawV2(text=raw_text),
            identity=DocumentIdentityV2(status="EXPLICIT", document_number="12/2024/NĐ-CP"),
            authority_text=auth_text,
            authority_text_sha256=auth_sha,
        )


def test_text_span_validation():
    # Valid span
    span = TextSpanV2(start=0, end=10)
    assert span.start == 0 and span.end == 10

    # Negative start fails
    with pytest.raises(ValidationError):
        TextSpanV2(start=-1, end=5)

    # start > end fails
    with pytest.raises(ValidationError):
        TextSpanV2(start=10, end=5)


def test_legal_reference_resolution_invariants():
    # RESOLVED_UNIQUE with target_document_id is valid
    res = LegalReferenceResolutionV2(
        status="RESOLVED_UNIQUE",
        target_document_id="doc:uitdsc2026:1",
        candidate_document_ids=["doc:uitdsc2026:1"],
    )
    assert res.target_document_id == "doc:uitdsc2026:1"

    # RESOLVED_AMBIGUOUS with target_document_id must fail
    with pytest.raises(ValidationError):
        LegalReferenceResolutionV2(
            status="RESOLVED_AMBIGUOUS",
            target_document_id="doc:uitdsc2026:1",
            candidate_document_ids=["doc:uitdsc2026:1", "doc:uitdsc2026:2"],
        )

    # UNRESOLVED with target_document_id must fail
    with pytest.raises(ValidationError):
        LegalReferenceResolutionV2(
            status="UNRESOLVED",
            target_document_id="doc:uitdsc2026:1",
        )
