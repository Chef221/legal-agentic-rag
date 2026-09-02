"""Unit tests for V2RetrievalUnitStore memory-bounded random access and alignment."""

import json
from pathlib import Path
import pytest

from legal_agentic_rag.exceptions import ArtifactCompatibilityError, DataValidationError
from legal_agentic_rag.indexing.vector.v2_retrieval_unit_store import V2RetrievalUnitStore
from legal_agentic_rag.schemas.preprocessing_v2 import RetrievalUnitV2


def _create_unit_record(unit_id: str, doc_id: str, text: str) -> dict:
    return {
        "schema_version": "m54-preprocessing-v2.1",
        "retrieval_unit_id": unit_id,
        "document_id": doc_id,
        "provision_id": f"{doc_id}::art:1",
        "segment_index": 1,
        "segment_count": 1,
        "authority_span_in_provision": {"start": 0, "end": len(text)},
        "authority_text": text,
        "retrieval_text": f"Văn bản: Test\n---\n{text}",
        "document_identity": {
            "document_number": "01/2024/TT-BTTTT",
            "title": "Thông tư thử nghiệm",
        },
        "hierarchy": {
            "article_label": "1",
            "clause_label": None,
            "point_label": None,
            "heading_path": [{"type": "CHAPTER", "label": "I", "title": "QUY ĐỊNH CHUNG"}],
        },
        "strategy": "WHOLE_PROVISION",
        "token_count_authority": len(text.split()),
        "token_count_retrieval": len(text.split()) + 5,
        "quality_flags": [],
    }


@pytest.fixture
def sample_aligned_data(tmp_path: Path):
    units = [
        _create_unit_record("doc:1::art:1", "doc:1", "Nội dung điều 1"),
        _create_unit_record("doc:1::art:2", "doc:1", "Nội dung điều 2"),
        _create_unit_record("doc:2::art:1", "doc:2", "Nội dung văn bản 2"),
    ]
    units_file = tmp_path / "records.jsonl"
    with open(units_file, "w", encoding="utf-8") as f:
        for u in units:
            f.write(json.dumps(u, ensure_ascii=False) + "\n")

    ids_file = tmp_path / "retrieval_unit_ids.jsonl"
    with open(ids_file, "w", encoding="utf-8") as f:
        for u in units:
            f.write(json.dumps(u["retrieval_unit_id"]) + "\n")

    return units_file, ids_file, units


def test_v2_store_load_and_random_access(sample_aligned_data):
    units_file, ids_file, expected_units = sample_aligned_data
    store = V2RetrievalUnitStore.load(
        units_file,
        ids_path=ids_file,
        expected_count=3,
        verify_alignment=True,
    )

    assert len(store) == 3

    # Test single get
    u0 = store.get(0)
    assert isinstance(u0, RetrievalUnitV2)
    assert u0.retrieval_unit_id == "doc:1::art:1"
    assert u0.authority_text == "Nội dung điều 1"
    assert u0.document_identity.document_number == "01/2024/TT-BTTTT"

    # Test get_many
    many = store.get_many([2, 0])
    assert len(many) == 2
    assert many[0].retrieval_unit_id == "doc:2::art:1"
    assert many[1].retrieval_unit_id == "doc:1::art:1"

    # Test slicing
    slice_units = store[1:3]
    assert len(slice_units) == 2
    assert slice_units[0].retrieval_unit_id == "doc:1::art:2"
    assert slice_units[1].retrieval_unit_id == "doc:2::art:1"

    # Out of bounds
    with pytest.raises(IndexError):
        store.get(3)
    with pytest.raises(IndexError):
        store.get(-1)


def test_v2_store_rejects_id_mismatch(tmp_path: Path):
    units = [
        _create_unit_record("doc:1::art:1", "doc:1", "Text 1"),
        _create_unit_record("doc:1::art:2", "doc:1", "Text 2"),
    ]
    units_file = tmp_path / "records.jsonl"
    with open(units_file, "w", encoding="utf-8") as f:
        for u in units:
            f.write(json.dumps(u, ensure_ascii=False) + "\n")

    # Mismatched IDs file
    ids_file = tmp_path / "retrieval_unit_ids.jsonl"
    with open(ids_file, "w", encoding="utf-8") as f:
        f.write(json.dumps("doc:1::art:1") + "\n")
        f.write(json.dumps("doc:1::art:WRONG") + "\n")

    with pytest.raises(DataValidationError, match="Alignment mismatch at row 1"):
        V2RetrievalUnitStore.load(units_file, ids_path=ids_file, verify_alignment=True)


def test_v2_store_rejects_count_mismatch(sample_aligned_data):
    units_file, ids_file, _ = sample_aligned_data
    with pytest.raises(DataValidationError, match="Expected 5 retrieval units, found 3"):
        V2RetrievalUnitStore.load(
            units_file,
            ids_path=ids_file,
            expected_count=5,
            verify_alignment=True,
        )


def test_v2_store_rejects_missing_file(tmp_path: Path):
    with pytest.raises(ArtifactCompatibilityError):
        V2RetrievalUnitStore.load(tmp_path / "nonexistent.jsonl")
