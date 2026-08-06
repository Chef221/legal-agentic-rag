"""Official context ingestion into the existing legal structure pipeline."""

import json
from datetime import UTC, datetime
from pathlib import Path

from legal_agentic_rag.competition import UitDsc2026CorpusIngestor
from legal_agentic_rag.offline.chunking import LegalChunker
from legal_agentic_rag.offline.parsing import LegalStructureParser
from legal_agentic_rag.schemas import ArtifactType


def test_official_context_flows_to_legal_chunks(tmp_path: Path) -> None:
    """A documented BTC context becomes searchable article-level chunks."""
    context_source = tmp_path / "contexts"
    context_source.mkdir()
    (context_source / "context_0001.json").write_text(
        json.dumps(
            {
                "id": 740,
                "link": "https://example.invalid/document-740",
                "passage": (
                    "<p>CHƯƠNG I</p>\r\n"
                    "QUY ĐỊNH CHUNG\n"
                    "Điều 1. Phạm vi điều chỉnh\n"
                    "1. Quy định này áp dụng cho doanh nghiệp.\n"
                    "2. Không áp dụng cho trường hợp được miễn."
                ),
            },
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )
    now = datetime(2026, 8, 1, 8, 30, tzinfo=UTC)
    ingestion = UitDsc2026CorpusIngestor(clock=lambda: now).ingest(
        context_source
    )

    parsing = LegalStructureParser(clock=lambda: now).parse(
        documents=ingestion.cleaned_documents,
        source_manifest=ingestion.cleaned_manifest,
    )
    chunking = LegalChunker(clock=lambda: now).chunk(
        documents=ingestion.cleaned_documents,
        blocks=parsing.blocks,
        source_manifest=parsing.manifest,
    )

    assert parsing.manifest.artifact_type == ArtifactType.LEGAL_BLOCKS
    assert chunking.manifest.artifact_type == ArtifactType.LEGAL_CHUNKS
    article = next(
        chunk for chunk in chunking.chunks if chunk.structure.article_number == "1"
    )
    assert article.document_id == "740"
    assert "Không áp dụng" in article.text
    assert article.source_dataset == ingestion.dataset_manifest.dataset_name
