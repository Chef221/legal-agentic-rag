"""Focused checks for M54 V2 production validation and equivalence."""

from __future__ import annotations

from hashlib import sha256
import json
from pathlib import Path

from legal_agentic_rag.offline.preprocessing_v2.validation import (
    CANONICAL_REVISION,
    ZIP_AUTHORITY_SHA256,
    compare_preprocessing_v2_equivalence,
    validate_preprocessing_v2,
)
from legal_agentic_rag.schemas.preprocessing_v2 import (
    CanonicalDocumentV2,
    DocumentIdentityV2,
    DocumentRawV2,
    DocumentSourceV2,
    LegalProvisionV2,
    RetrievalUnitDocumentIdentityV2,
    RetrievalUnitHierarchyV2,
    RetrievalUnitV2,
    TextSpanV2,
)


def _write_records(path: Path, records: list[dict[str, object]]) -> None:
    path.write_text(
        "".join(
            json.dumps(record, ensure_ascii=False, sort_keys=True, separators=(",", ":")) + "\n"
            for record in records
        ),
        encoding="utf-8",
    )


def test_production_validator_computes_all_gates(tmp_path: Path) -> None:
    text = "Nội dung pháp luật."
    text_sha = sha256(text.encode("utf-8")).hexdigest()
    document = CanonicalDocumentV2(
        document_id="doc:uitdsc2026:1",
        source=DocumentSourceV2(
            context_id="1",
            member_name="context_1.json",
            raw_passage_sha256=text_sha,
            corpus_revision=CANONICAL_REVISION,
        ),
        raw=DocumentRawV2(text=text),
        identity=DocumentIdentityV2(status="UNRESOLVED"),
        authority_text=text,
        authority_text_sha256=text_sha,
    )
    provision = LegalProvisionV2(
        provision_id="doc:uitdsc2026:1::doc_fallback",
        document_id=document.document_id,
        provision_type="DOCUMENT_FALLBACK",
        authority_span=TextSpanV2(start=0, end=len(text)),
        parse_status="CONTROLLED_FALLBACK",
        parse_rule="DOCUMENT_FALLBACK",
        authority_text=text,
    )
    retrieval = RetrievalUnitV2(
        retrieval_unit_id=provision.provision_id,
        document_id=document.document_id,
        provision_id=provision.provision_id,
        authority_span_in_provision=TextSpanV2(start=0, end=len(text)),
        authority_text=text,
        retrieval_text=text,
        document_identity=RetrievalUnitDocumentIdentityV2(),
        hierarchy=RetrievalUnitHierarchyV2(),
        strategy="DOCUMENT_FALLBACK",
    )
    paths = {name: tmp_path / f"{name}.jsonl" for name in ("docs", "provs", "rus", "refs", "unrec")}
    _write_records(paths["docs"], [document.model_dump(mode="json")])
    _write_records(paths["provs"], [provision.model_dump(mode="json")])
    _write_records(paths["rus"], [retrieval.model_dump(mode="json")])
    _write_records(paths["refs"], [])
    _write_records(paths["unrec"], [])

    result = validate_preprocessing_v2(
        source=None,
        docs_path=paths["docs"],
        provs_path=paths["provs"],
        rus_path=paths["rus"],
        refs_path=paths["refs"],
        unrec_path=paths["unrec"],
        val_out_path=tmp_path / "production_v2_validation.json",
        source_identity={
            "raw_zip_sha256": ZIP_AUTHORITY_SHA256,
            "canonical_revision": CANONICAL_REVISION,
            "source_members": 1,
        },
    )

    assert result["gates"]["RAW_ZIP_SHA256_MATCH"] is True, result["source"]
    assert result["gates"]["CANONICAL_REVISION_MATCH"] is True, result["source"]
    assert result["failed_gates"] == []
    assert result["overall_pass"] is True
    assert result["gates"]["DOCUMENTS"] == 1
    assert result["gates"]["RETRIEVAL_UNITS"] == 1


def test_streaming_equivalence_reports_first_mismatch(tmp_path: Path) -> None:
    accepted = tmp_path / "accepted.jsonl"
    production = tmp_path / "production.jsonl"
    _write_records(accepted, [{"document_id": "doc:1", "value": "accepted"}])
    _write_records(production, [{"document_id": "doc:1", "value": "production"}])

    result = compare_preprocessing_v2_equivalence(
        accepted_paths={"documents": accepted},
        production_paths={"documents": production},
        report_path=tmp_path / "equivalence_report.json",
    )

    assert result["overall_pass"] is False
    assert result["artifacts"]["documents"]["first_mismatch"]["row"] == 1
