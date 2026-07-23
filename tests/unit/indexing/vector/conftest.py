"""Small unified chunks and vectors for exact dense retrieval tests."""

from datetime import UTC, datetime

import pytest

from legal_agentic_rag.schemas import (
    ArtifactManifest,
    ArtifactType,
    LegalChunk,
    LegalStructure,
)


def make_chunk(
    chunk_id: str,
    text: str,
    *,
    document_id: str,
    legal_field: str,
    effect_status: str = "Còn hiệu lực",
) -> LegalChunk:
    return LegalChunk(
        chunk_id=chunk_id,
        document_id=document_id,
        chunk_index=0,
        text=text,
        search_text=f"Nội dung: {text}",
        token_count=max(1, len(text.split())),
        structure=LegalStructure(article_number="1", structure_path=["Điều 1"]),
        document_title=f"Văn bản {document_id}",
        document_number=f"{document_id}/2026/QH",
        document_type="Luật",
        effect_status=effect_status,
        legal_field=legal_field,
        source_dataset="aio",
        metadata={"source_block_ids": [f"block-{chunk_id}"]},
    )

@pytest.fixture
def vector_chunks() -> list[LegalChunk]:
    return [
        make_chunk(
            "chunk-speed",
            "Người lái xe chạy quá tốc độ.",
            document_id="doc-speed",
            legal_field="Giao thông",
        ),
        make_chunk(
            "chunk-license",
            "Người điều khiển phải có giấy phép lái xe.",
            document_id="doc-license",
            legal_field="Giao thông",
        ),
        make_chunk(
            "chunk-tax",
            "Doanh nghiệp phải nộp thuế đúng thời hạn.",
            document_id="doc-tax",
            legal_field="Thuế",
            effect_status="Hết hiệu lực",
        ),
    ]


@pytest.fixture
def vectors() -> list[list[float]]:
    return [[1.0, 0.0], [0.8, 0.2], [0.0, 1.0]]


@pytest.fixture
def vector_source_manifest(vector_chunks: list[LegalChunk]) -> ArtifactManifest:
    return ArtifactManifest(
        schema_version="1.0",
        artifact_type=ArtifactType.LEGAL_CHUNKS,
        artifact_version="1.0",
        dataset_name="fixture",
        dataset_revision="fixture-revision",
        created_at=datetime(2026, 7, 22, tzinfo=UTC),
        record_count=len(vector_chunks),
        processing_config_hash="chunk-hash",
        code_version="0.7.0",
    )
