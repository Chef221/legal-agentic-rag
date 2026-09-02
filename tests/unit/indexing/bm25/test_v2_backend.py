"""Unit tests for V2 SQLite FTS5 BM25 backend build, search, filtering, and artifact management."""

import json
from pathlib import Path
import pytest

from legal_agentic_rag.exceptions import (
    ArtifactCompatibilityError,
    DataValidationError,
    RetrievalError,
)
from legal_agentic_rag.indexing.bm25.v2_backend import (
    EXPECTED_ANALYZER,
    EXPECTED_BACKEND,
    EXPECTED_SCHEMA,
    V2SQLiteFTS5BM25Backend,
)
from legal_agentic_rag.schemas.retrieval import (
    RetrievalFilters,
    RetrievalQuery,
    RetrievalStrategy,
)


def _create_unit_record(
    unit_id: str,
    doc_id: str,
    auth_text: str,
    retrieval_text: str,
    doc_num: str = "01/2024/TT-BTTTT",
) -> dict:
    return {
        "schema_version": "m54-preprocessing-v2.1",
        "retrieval_unit_id": unit_id,
        "document_id": doc_id,
        "provision_id": f"{doc_id}::art:1",
        "segment_index": 1,
        "segment_count": 1,
        "authority_span_in_provision": {"start": 0, "end": len(auth_text)},
        "authority_text": auth_text,
        "retrieval_text": retrieval_text,
        "document_identity": {
            "document_number": doc_num,
            "title": "Thông tư thử nghiệm",
        },
        "hierarchy": {
            "article_label": "1",
            "clause_label": None,
            "point_label": None,
            "heading_path": [{"type": "CHAPTER", "label": "I", "title": "QUY ĐỊNH CHUNG"}],
        },
        "strategy": "WHOLE_PROVISION",
        "token_count_authority": len(auth_text.split()),
        "token_count_retrieval": len(retrieval_text.split()),
        "quality_flags": [],
    }


@pytest.fixture
def sample_v2_corpus(tmp_path: Path):
    units = [
        _create_unit_record(
            "doc:1::art:1",
            "doc:1",
            "Nội dung quy định xử phạt môi trường.",
            "Văn bản: Luật Bảo vệ Môi trường\n---\nNội dung quy định xử phạt vi phạm hành chính môi trường.",
            "01/2020/QH14",
        ),
        _create_unit_record(
            "doc:1::art:2",
            "doc:1",
            "Thủ tục cấp phép môi trường tại địa phương.",
            "Văn bản: Luật Bảo vệ Môi trường\n---\nThủ tục hồ sơ cấp giấy phép môi trường.",
            "01/2020/QH14",
        ),
        _create_unit_record(
            "doc:2::art:1",
            "doc:2",
            "Thời giờ làm việc của người lao động.",
            "Văn bản: Bộ luật Lao động\n---\nThời giờ làm việc bình thường của người lao động.",
            "45/2019/QH14",
        ),
    ]
    units_file = tmp_path / "records.jsonl"
    with open(units_file, "w", encoding="utf-8") as f:
        for u in units:
            f.write(json.dumps(u, ensure_ascii=False) + "\n")

    return units_file, units


def test_v2_bm25_tiny_build_succeeds(tmp_path: Path, sample_v2_corpus):
    units_file, _ = sample_v2_corpus
    artifact_dir = tmp_path / "bm25_artifact_1"

    manifest = V2SQLiteFTS5BM25Backend.build(
        source_units_path=units_file,
        destination_dir=artifact_dir,
        source_sha256="test_sha",
    )

    assert manifest["schema"] == EXPECTED_SCHEMA
    assert manifest["backend"] == EXPECTED_BACKEND
    assert manifest["record_count"] == 3
    assert (artifact_dir / "bm25_v2.sqlite3").is_file()
    assert (artifact_dir / "bm25_v2_manifest_v1.json").is_file()
    assert (artifact_dir / "SUCCESS.json").is_file()


def test_v2_bm25_search_returns_lexical_match_and_metadata(tmp_path: Path, sample_v2_corpus):
    units_file, _ = sample_v2_corpus
    artifact_dir = tmp_path / "bm25_artifact_2"

    V2SQLiteFTS5BM25Backend.build(units_file, artifact_dir)
    backend = V2SQLiteFTS5BM25Backend.load(artifact_dir, units_file, verify_db_sha=True)

    query = RetrievalQuery(
        query_id="q1",
        original_question="xử phạt vi phạm hành chính",
        normalized_question="xu phat vi pham hanh chinh",
        top_k=2,
    )

    response = backend.search(query)
    assert response.strategy == RetrievalStrategy.BM25
    assert len(response.hits) >= 1

    top_hit = response.hits[0]
    # Requirements 3, 4, 5, 6
    assert top_hit.chunk_id == "doc:1::art:1"
    assert top_hit.document_id == "doc:1"
    assert top_hit.text == "Nội dung quy định xử phạt môi trường."  # returned text is authority_text
    assert "Luật Bảo vệ Môi trường" in top_hit.metadata["retrieval_text"]  # indexed text is retrieval_text
    assert top_hit.metadata["provision_id"] == "doc:1::art:1"
    assert top_hit.metadata["strategy"] == "WHOLE_PROVISION"
    assert top_hit.retrieval_trace.bm25_rank == 1
    assert top_hit.retrieval_trace.bm25_score is not None

    backend.close()


def test_v2_bm25_document_ids_filter(tmp_path: Path, sample_v2_corpus):
    units_file, _ = sample_v2_corpus
    artifact_dir = tmp_path / "bm25_artifact_3"

    V2SQLiteFTS5BM25Backend.build(units_file, artifact_dir)
    backend = V2SQLiteFTS5BM25Backend.load(artifact_dir, units_file)

    query = RetrievalQuery(
        query_id="q2",
        original_question="người lao động",
        normalized_question="nguoi lao dong",
        filters=RetrievalFilters(document_ids=["doc:2"]),
        top_k=5,
    )

    response = backend.search(query)
    assert len(response.hits) == 1
    assert response.hits[0].document_id == "doc:2"

    backend.close()


def test_v2_bm25_rejects_unsupported_legacy_filters(tmp_path: Path, sample_v2_corpus):
    units_file, _ = sample_v2_corpus
    artifact_dir = tmp_path / "bm25_artifact_4"

    V2SQLiteFTS5BM25Backend.build(units_file, artifact_dir)
    backend = V2SQLiteFTS5BM25Backend.load(artifact_dir, units_file)

    # Unsupported document_types
    with pytest.raises(RetrievalError, match="legacy filters"):
        backend.search(RetrievalQuery(
            query_id="q_bad1",
            original_question="test",
            normalized_question="test",
            filters=RetrievalFilters(document_types=["LUAT"]),
        ))

    # Unsupported legal_fields
    with pytest.raises(RetrievalError, match="legacy filters"):
        backend.search(RetrievalQuery(
            query_id="q_bad2",
            original_question="test",
            normalized_question="test",
            filters=RetrievalFilters(legal_fields=["CIVIL"]),
        ))

    # Unsupported effect_statuses
    with pytest.raises(RetrievalError, match="legacy filters"):
        backend.search(RetrievalQuery(
            query_id="q_bad3",
            original_question="test",
            normalized_question="test",
            filters=RetrievalFilters(effect_statuses=["CON_HIEU_LUC"]),
        ))

    backend.close()


def test_v2_bm25_rejects_duplicate_ids(tmp_path: Path):
    units = [
        _create_unit_record("doc:1::art:1", "doc:1", "Text 1", "Search 1"),
        _create_unit_record("doc:1::art:1", "doc:1", "Text 2", "Search 2"),
    ]
    units_file = tmp_path / "records_dup.jsonl"
    with open(units_file, "w", encoding="utf-8") as f:
        for u in units:
            f.write(json.dumps(u, ensure_ascii=False) + "\n")

    with pytest.raises(DataValidationError, match="Duplicate retrieval_unit_id"):
        V2SQLiteFTS5BM25Backend.build(units_file, tmp_path / "out_dup")


def test_v2_bm25_rejects_malformed_record(tmp_path: Path):
    units_file = tmp_path / "records_bad.jsonl"
    with open(units_file, "w", encoding="utf-8") as f:
        f.write('{"retrieval_unit_id": "doc:1::art:1", "document_id": "doc:1"}\n')

    with pytest.raises(DataValidationError, match="Malformed RetrievalUnitV2"):
        V2SQLiteFTS5BM25Backend.build(units_file, tmp_path / "out_bad")


def test_v2_bm25_persist_load_round_trip(tmp_path: Path, sample_v2_corpus):
    units_file, _ = sample_v2_corpus
    artifact_dir = tmp_path / "bm25_artifact_roundtrip"

    manifest_built = V2SQLiteFTS5BM25Backend.build(units_file, artifact_dir)
    backend = V2SQLiteFTS5BM25Backend.load(artifact_dir, units_file, verify_db_sha=True)

    assert backend.record_count == manifest_built["record_count"]
    assert backend.manifest["database_sha256"] == manifest_built["database_sha256"]
    backend.close()


def test_v2_bm25_manifest_schema_mismatch_rejected(tmp_path: Path, sample_v2_corpus):
    units_file, _ = sample_v2_corpus
    artifact_dir = tmp_path / "bm25_artifact_bad_schema"

    V2SQLiteFTS5BM25Backend.build(units_file, artifact_dir)

    manifest_p = artifact_dir / "bm25_v2_manifest_v1.json"
    manifest = json.loads(manifest_p.read_text(encoding="utf-8"))
    manifest["schema"] = "wrong_schema"
    manifest_p.write_text(json.dumps(manifest), encoding="utf-8")

    with pytest.raises(ArtifactCompatibilityError, match="Unsupported BM25 schema"):
        V2SQLiteFTS5BM25Backend.load(artifact_dir, units_file, strict_manifest=True)


def test_v2_bm25_db_checksum_mismatch_rejected(tmp_path: Path, sample_v2_corpus):
    units_file, _ = sample_v2_corpus
    artifact_dir = tmp_path / "bm25_artifact_bad_checksum"

    V2SQLiteFTS5BM25Backend.build(units_file, artifact_dir)

    manifest_p = artifact_dir / "bm25_v2_manifest_v1.json"
    manifest = json.loads(manifest_p.read_text(encoding="utf-8"))
    manifest["database_sha256"] = "bad_checksum_hash"
    manifest_p.write_text(json.dumps(manifest), encoding="utf-8")

    with pytest.raises(ArtifactCompatibilityError, match="Database SHA256 mismatch"):
        V2SQLiteFTS5BM25Backend.load(artifact_dir, units_file, verify_db_sha=True)


def test_v2_bm25_row_id_mismatch_rejected(tmp_path: Path, sample_v2_corpus):
    units_file, raw_units = sample_v2_corpus
    artifact_dir = tmp_path / "bm25_artifact_tampered"

    V2SQLiteFTS5BM25Backend.build(units_file, artifact_dir)

    # Tampered units file where row 0 has a mismatched unit_id
    tampered_units = list(raw_units)
    tampered_units[0] = dict(tampered_units[0])
    tampered_units[0]["retrieval_unit_id"] = "doc:1::art:TAMPERED"

    tampered_file = tmp_path / "records_tampered.jsonl"
    with open(tampered_file, "w", encoding="utf-8") as f:
        for u in tampered_units:
            f.write(json.dumps(u, ensure_ascii=False) + "\n")

    backend = V2SQLiteFTS5BM25Backend.load(artifact_dir, tampered_file)

    query = RetrievalQuery(
        query_id="q_tampered",
        original_question="xử phạt vi phạm",
        normalized_question="xu phat vi pham",
    )

    with pytest.raises(ArtifactCompatibilityError, match="Row resolution mismatch"):
        backend.search(query)

    backend.close()
