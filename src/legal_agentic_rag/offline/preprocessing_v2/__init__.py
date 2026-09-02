"""M54 Preprocessing V2 package."""

from legal_agentic_rag.offline.preprocessing_v2.builder import PreprocessingV2Builder
from legal_agentic_rag.offline.preprocessing_v2.parser import (
    parse_document_structure_v2,
    parse_provisions_from_document,
)
from legal_agentic_rag.offline.preprocessing_v2.references import (
    extract_and_resolve_references_v2,
    extract_and_resolve_references_v2 as extract_legal_references,
)
from legal_agentic_rag.offline.preprocessing_v2.retrieval_units import (
    materialize_retrieval_units_v2,
    materialize_retrieval_units_v2 as materialize_retrieval_units,
)
from legal_agentic_rag.offline.preprocessing_v2.validation import validate_preprocessing_v2

__all__ = [
    "PreprocessingV2Builder",
    "parse_document_structure_v2",
    "parse_provisions_from_document",
    "materialize_retrieval_units_v2",
    "materialize_retrieval_units",
    "extract_and_resolve_references_v2",
    "extract_legal_references",
    "validate_preprocessing_v2",
]
