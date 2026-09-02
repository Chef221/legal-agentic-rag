"""Unit tests for V2PrecomputedDenseBackend memory-mapped cosine search and RetrievalHit conversion."""

import json
from pathlib import Path
import numpy as np
import pytest

from legal_agentic_rag.exceptions import ArtifactCompatibilityError, DataValidationError
from legal_agentic_rag.indexing.vector.v2_precomputed_backend import (
    EXPECTED_DIMENSION,
    EXPECTED_MODEL,
    EXPECTED_RECORD_COUNT,
    EXPECTED_REVISION,
    EXPECTED_SCHEMA,
    EXPECTED_SOURCE_SHA256,
    V2PrecomputedDenseBackend,
)
from legal_agentic_rag.schemas.retrieval import RetrievalQuery, RetrievalStrategy


def _create_synthetic_fixture(tmp_path: Path, num_records: int = 4, dim: int = 8):
    matrix_dir = tmp_path / "matrix"
    matrix_dir.mkdir(parents=True, exist_ok=True)

    # 1. Generate normalized vectors
    rng = np.random.default_rng(42)
    raw_vecs = rng.standard_normal((num_records, dim)).astype(np.float32)
    norms = np.linalg.norm(raw_vecs, axis=1, keepdims=True)
    norm_vecs = raw_vecs / norms

    vectors_file = matrix_dir / "vectors.npy"
    np.save(vectors_file, norm_vecs)

    # 2. Generate unit IDs and records
    unit_ids = [f"doc:1::art:{i+1}" for i in range(num_records)]
    ids_file = matrix_dir / "retrieval_unit_ids.jsonl"
    with open(ids_file, "w", encoding="utf-8") as f:
        for uid in unit_ids:
            f.write(json.dumps(uid) + "\n")

    units_file = tmp_path / "records.jsonl"
    with open(units_file, "w", encoding="utf-8") as f:
        for i, uid in enumerate(unit_ids):
            rec = {
                "schema_version": "m54-preprocessing-v2.1",
                "retrieval_unit_id": uid,
                "document_id": "doc:1",
                "provision_id": f"doc:1::art:{i+1}",
                "segment_index": 1,
                "segment_count": 1,
                "authority_span_in_provision": {"start": 0, "end": 20},
                "authority_text": f"Nội dung điều {i+1} quy định quyền và nghĩa vụ.",
                "retrieval_text": f"Văn bản: Luật\n---\nNội dung điều {i+1}",
                "document_identity": {
                    "document_number": "10/2020/QH14",
                    "title": "Luật Thử Nghiệm",
                },
                "hierarchy": {
                    "article_label": str(i + 1),
                    "clause_label": None,
                    "point_label": None,
                    "heading_path": [{"type": "CHAPTER", "label": "I", "title": "QUY ĐỊNH CHUNG"}],
                },
                "strategy": "WHOLE_PROVISION",
                "token_count_authority": 10,
                "token_count_retrieval": 15,
                "quality_flags": [],
            }
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")

    # 3. Generate manifest
    manifest = {
        "schema": EXPECTED_SCHEMA,
        "record_count": num_records,
        "dimension": dim,
        "dtype": "float32",
        "distance_metric": "cosine",
        "normalized": True,
        "model_name": EXPECTED_MODEL,
        "model_revision": EXPECTED_REVISION,
        "source_retrieval_units_sha256": EXPECTED_SOURCE_SHA256,
        "vectors_filename": "vectors.npy",
        "ids_filename": "retrieval_unit_ids.jsonl",
    }
    manifest_file = matrix_dir / "index_manifest_v1.json"
    manifest_file.write_text(json.dumps(manifest, indent=2), encoding="utf-8")

    return matrix_dir, units_file, norm_vecs, unit_ids


def test_v2_backend_exact_cosine_search(tmp_path: Path):
    matrix_dir, units_file, norm_vecs, unit_ids = _create_synthetic_fixture(tmp_path, num_records=4, dim=8)

    backend = V2PrecomputedDenseBackend.load(
        matrix_dir,
        units_file,
        verify_integrity=True,
        strict_manifest=False,
    )

    assert backend.record_count == 4
    assert backend.dimension == 8

    # Query vector matching row 2 exactly
    target_row = 2
    query_vec = norm_vecs[target_row].copy()

    results = backend.search_vector(query_vec, top_k=3)
    assert len(results) == 3
    # Top 1 must be target row with score ~1.0
    top_idx, top_score = results[0]
    assert top_idx == target_row
    assert np.isclose(top_score, 1.0, atol=1e-5)


def test_v2_backend_retrieve_full_response(tmp_path: Path):
    matrix_dir, units_file, norm_vecs, unit_ids = _create_synthetic_fixture(tmp_path, num_records=4, dim=8)

    backend = V2PrecomputedDenseBackend.load(
        matrix_dir,
        units_file,
        verify_integrity=True,
        strict_manifest=False,
    )

    query = RetrievalQuery(
        query_id="q1",
        original_question="Quy định điều 1",
        normalized_question="quy dinh dieu 1",
        top_k=2,
    )

    query_vec = norm_vecs[0].copy()
    response = backend.retrieve(query, query_vec)

    assert response.strategy == RetrievalStrategy.DENSE
    assert len(response.hits) == 2

    hit0 = response.hits[0]
    assert hit0.chunk_id == unit_ids[0]
    assert hit0.document_id == "doc:1"
    assert hit0.rank == 1
    assert np.isclose(hit0.score, 1.0, atol=1e-5)
    assert hit0.text == "Nội dung điều 1 quy định quyền và nghĩa vụ."
    assert hit0.metadata["provision_id"] == "doc:1::art:1"
    assert hit0.metadata["strategy"] == "WHOLE_PROVISION"
    assert hit0.retrieval_trace.dense_rank == 1
    assert np.isclose(hit0.retrieval_trace.dense_score, 1.0, atol=1e-5)


def test_v2_backend_query_validation(tmp_path: Path):
    matrix_dir, units_file, norm_vecs, _ = _create_synthetic_fixture(tmp_path, num_records=4, dim=8)

    backend = V2PrecomputedDenseBackend.load(
        matrix_dir,
        units_file,
        strict_manifest=False,
    )

    # Wrong dimension
    with pytest.raises(DataValidationError, match="shape must be"):
        backend.search_vector(np.ones(10, dtype=np.float32))

    # NaN / Inf
    with pytest.raises(DataValidationError, match="contains NaN or infinite"):
        bad_vec = norm_vecs[0].copy()
        bad_vec[0] = np.nan
        backend.search_vector(bad_vec)

    # Zero vector
    with pytest.raises(DataValidationError, match="norm is zero"):
        backend.search_vector(np.zeros(8, dtype=np.float32))


def test_v2_backend_manifest_rejection(tmp_path: Path):
    matrix_dir, units_file, _, _ = _create_synthetic_fixture(tmp_path, num_records=4, dim=8)

    # Invalid metric
    m_path = matrix_dir / "index_manifest_v1.json"
    manifest = json.loads(m_path.read_text(encoding="utf-8"))
    manifest["distance_metric"] = "l2"
    m_path.write_text(json.dumps(manifest), encoding="utf-8")

    with pytest.raises(ArtifactCompatibilityError, match="distance metric"):
        V2PrecomputedDenseBackend.load(matrix_dir, units_file, strict_manifest=False)
