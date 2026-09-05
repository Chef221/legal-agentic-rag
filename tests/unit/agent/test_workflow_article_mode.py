"""Unit tests for DeterministicAgentWorkflow in M55 Article answering mode."""

import hashlib
import json
from pathlib import Path
import pytest

from legal_agentic_rag.agent import DeterministicAgentWorkflow
from legal_agentic_rag.configuration import AgentConfig, ContextGradingConfig
from legal_agentic_rag.exceptions import ConfigurationError
from legal_agentic_rag.generation.article_authority import (
    ArticleAuthorityStore,
    FirstKFullArticleAnswerAssembler,
)
from legal_agentic_rag.generation import RuleBasedContextGrader
from legal_agentic_rag.schemas import (
    AgentStopReason,
    RetrievalHit,
    RetrievalQuery,
    RetrievalResponse,
    RetrievalStrategy,
    ToolName,
)
from legal_agentic_rag.tools import (
    ContextGradingTool,
    RetrievalTool,
    ToolRegistry,
    build_retrieval_grading_tool_registry,
)


class _DummyRetriever:
    def search(self, query: RetrievalQuery) -> RetrievalResponse:
        hits = [
            RetrievalHit(
                chunk_id="chunk-1",
                document_id="doc-100",
                rank=1,
                score=0.9,
                strategy=query.requested_strategy or RetrievalStrategy.HYBRID_RERANK,
                text="Dieu 1 text fragment",
                metadata={
                    "document_number": "100/QH",
                    "structure": {"article_number": "1"},
                },
            )
        ]
        return RetrievalResponse(
            query=query,
            strategy=query.requested_strategy or RetrievalStrategy.HYBRID_RERANK,
            hits=hits,
            warnings=[],
        )


def _make_store(tmp_path: Path) -> ArticleAuthorityStore:
    records = [
        {"document_id": "doc-100", "article_identity": "1", "article_text": "Dieu 1 Full Authority Text"},
    ]
    p = tmp_path / "lookup.jsonl"
    hasher = hashlib.sha256()
    with open(p, "w", encoding="utf-8", newline="\n") as f:
        for r in records:
            line = json.dumps(r, ensure_ascii=False) + "\n"
            hasher.update(line.encode("utf-8"))
            f.write(line)
    return ArticleAuthorityStore.from_jsonl(p, expected_sha256=hasher.hexdigest(), expected_record_count=1)


def test_article_mode_workflow_runs_without_generation_or_verification_tools(tmp_path: Path) -> None:
    store = _make_store(tmp_path)
    assembler = FirstKFullArticleAnswerAssembler(store)

    retriever = _DummyRetriever()
    grader = RuleBasedContextGrader(ContextGradingConfig(minimum_evidence_count=1))
    registry = build_retrieval_grading_tool_registry(retriever=retriever, context_grader=grader)

    # Tool registry must have retrieval tools and context grading, but NOT generation or verifier
    descriptor_names = {d.name for d in registry.descriptors()}
    assert ToolName.CONTEXT_GRADING in descriptor_names
    assert ToolName.RERANK_SEARCH in descriptor_names
    assert ToolName.ANSWER_GENERATION not in descriptor_names
    assert ToolName.CITATION_VERIFICATION not in descriptor_names

    wf = DeterministicAgentWorkflow(
        registry,
        article_answerer=assembler,
        agent_config=AgentConfig(max_retry=1),
    )

    query = RetrievalQuery(
        query_id="q-test",
        original_question="Hỏi về Điều 1",
        normalized_question="hoi ve dieu 1",
        requested_strategy=RetrievalStrategy.HYBRID_RERANK,
    )

    result = wf.run(query)

    assert result.stop_reason == AgentStopReason.AUTHORITY_ANSWER_ASSEMBLED
    assert result.response.answer == "Dieu 1 Full Authority Text"
    assert result.response.insufficient_evidence is False
    assert len(result.response.citations) == 1
    assert result.response.citations[0].document_id == "doc-100"
    assert result.response.citations[0].article_number == "1"
    assert result.response.metadata["answer_path"] == "m55_first_k_full_article"


def test_article_mode_missing_context_grading_rejected(tmp_path: Path) -> None:
    store = _make_store(tmp_path)
    assembler = FirstKFullArticleAnswerAssembler(store)

    # Registry with only retrieval tools, missing context grading
    from legal_agentic_rag.tools.retrieval import fixed_retrieval_tools
    retriever = _DummyRetriever()
    tools = fixed_retrieval_tools(retriever)
    registry = ToolRegistry(tools)

    with pytest.raises(ConfigurationError, match="missing context grading tool"):
        DeterministicAgentWorkflow(
            registry,
            article_answerer=assembler,
        )


def test_legacy_mode_without_article_answerer_requires_all_tools() -> None:
    retriever = _DummyRetriever()
    grader = RuleBasedContextGrader(ContextGradingConfig(minimum_evidence_count=1))
    registry = build_retrieval_grading_tool_registry(retriever=retriever, context_grader=grader)

    # Legacy mode (article_answerer=None) must fail if ANSWER_GENERATION / CITATION_VERIFICATION missing
    with pytest.raises(ConfigurationError, match="missing required online tools"):
        DeterministicAgentWorkflow(
            registry,
            article_answerer=None,
        )