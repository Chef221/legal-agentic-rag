"""SQLite FTS5 reference implementation of the BM25 backend contract."""

from legal_agentic_rag.indexing.bm25.analyzer import UnicodeBM25Analyzer
from legal_agentic_rag.indexing.bm25.backend import SQLiteFTS5BM25Backend
from legal_agentic_rag.indexing.bm25.v2_backend import V2SQLiteFTS5BM25Backend

__all__ = [
    "SQLiteFTS5BM25Backend",
    "UnicodeBM25Analyzer",
    "V2SQLiteFTS5BM25Backend",
]
