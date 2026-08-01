"""Small unified-schema fixtures for SQLite FTS5 BM25 tests."""

from datetime import UTC, date, datetime

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
    document_type: str = "Luật",
    legal_field: str = "Giao thông",
    effect_status: str = "Còn hiệu lực",
    article_number: str = "1",
) -> LegalChunk:
    """Create one minimal but provenance-complete legal chunk."""
    return LegalChunk(
        chunk_id=chunk_id,
        document_id=document_id,
        chunk_index=0,
        text=text,
        search_text=f"Điều {article_number}\nNội dung: {text}",
        token_count=max(1, len(text.split())),
        structure=LegalStructure(
            article_number=article_number,
            structure_path=[f"Điều {article_number}"],
        ),
        document_title=f"Văn bản {document_id}",
        document_number=f"{document_id}/2026/QH",
        document_type=document_type,
        issuance_date=date(2026, 1, 1),
        effect_status=effect_status,
        legal_field=legal_field,
        source_url=f"https://example.test/{document_id}",
        source_dataset="fixture-corpus",
        metadata={"source_block_ids": [f"block-{chunk_id}"]},
    )

@pytest.fixture
def legal_chunks() -> list[LegalChunk]:
    """Return a small corpus with predictable Vietnamese lexical overlap."""
    return [
        make_chunk(
            "chunk-speed",
            "Người điều khiển xe chạy quá tốc độ bị phạt tiền.",
            document_id="doc-traffic",
            article_number="5",
        ),
        make_chunk(
            "chunk-license",
            "Không có giấy phép lái xe thì bị xử phạt.",
            document_id="doc-license",
            article_number="21",
        ),
        make_chunk(
            "chunk-tax",
            "Thời hạn nộp thuế thu nhập doanh nghiệp là ba mươi ngày.",
            document_id="doc-tax",
            legal_field="Thuế",
            effect_status="Hết hiệu lực",
            article_number="10",
        ),
    ]


@pytest.fixture
def chunk_manifest(legal_chunks: list[LegalChunk]) -> ArtifactManifest:
    """Describe the complete legal-chunk fixture corpus."""
    return ArtifactManifest(
        schema_version="1.0",
        artifact_type=ArtifactType.LEGAL_CHUNKS,
        artifact_version="1.0",
        dataset_name="fixture-corpus",
        dataset_revision="fixture-revision",
        created_at=datetime(2026, 7, 22, tzinfo=UTC),
        record_count=len(legal_chunks),
        processing_config_hash="chunk-config-hash",
        code_version="0.6.0",
    )
