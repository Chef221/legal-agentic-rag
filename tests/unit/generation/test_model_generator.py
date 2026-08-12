"""Tests for evidence-only model-backed answer generation."""

from collections.abc import Sequence
import json

import pytest

from legal_agentic_rag.contracts import AnswerGenerator, ChatModelProvider
from legal_agentic_rag.exceptions import DataValidationError, ModelError
from legal_agentic_rag.generation import ModelBackedAnswerGenerator
from legal_agentic_rag.schemas import Evidence, RetrievalQuery, RetrievalStrategy


class _FixtureProvider:
    provider_name = "fixture-chat"
    provider_version = "1.0"
    model_name = "fixture-legal-model"
    model_revision = "fixture-revision"

    def __init__(self, completion: str) -> None:
        self.completion = completion
        self.calls: list[tuple[str, str]] = []

    def complete(
        self,
        *,
        system_instruction: str,
        user_prompt: str,
    ) -> str:
        self.calls.append((system_instruction, user_prompt))
        return self.completion


class _SequenceProvider(_FixtureProvider):
    def __init__(self, completions: list[str]) -> None:
        super().__init__(completions[-1])
        self._completions = list(completions)

    def complete(
        self,
        *,
        system_instruction: str,
        user_prompt: str,
    ) -> str:
        self.calls.append((system_instruction, user_prompt))
        return self._completions.pop(0)


def _query() -> RetrievalQuery:
    return RetrievalQuery(
        query_id="model-answer-query",
        original_question="Doanh nghiệp phải nộp thuế khi nào?",
        normalized_question="doanh nghiệp nộp thuế",
        top_k=2,
        candidate_k=2,
    )


def _evidence(
    evidence_id: str = "E1",
    chunk_id: str = "chunk-1",
) -> Evidence:
    return Evidence(
        evidence_id=evidence_id,
        chunk_id=chunk_id,
        document_id="doc-1",
        text="Doanh nghiệp phải nộp thuế đúng thời hạn.",
        article_number="10",
        document_title="Luật mẫu",
        document_number="01/2026/QH",
        source_url="https://example.test/doc-1",
    )


def _completion(**updates: object) -> str:
    payload: dict[str, object] = {
        "answer": "Doanh nghiệp phải nộp thuế đúng thời hạn [E1].",
        "cited_evidence_ids": ["E1"],
        "insufficient_evidence": False,
        "warnings": [],
    }
    payload.update(updates)
    return json.dumps(payload, ensure_ascii=False)


def test_model_generator_builds_grounded_prompt_and_trusted_citation() -> None:
    """Only evidence IDs selected by the model become system-built citations."""
    provider = _FixtureProvider(_completion())
    generator = ModelBackedAnswerGenerator(provider)

    response = generator.generate(
        _query(),
        [_evidence()],
        RetrievalStrategy.HYBRID_RERANK,
        "model-answer-query",
    )

    assert response.answer.endswith("[E1].")
    assert response.citations[0].chunk_id == "chunk-1"
    assert response.citations[0].document_number == "01/2026/QH"
    assert response.metadata["semantic_synthesis"] is True
    assert response.metadata["model_revision"] == "fixture-revision"
    assert "effect_status_unknown:E1" in response.warnings
    system_instruction, user_prompt = provider.calls[0]
    assert "không dùng kiến thức bên ngoài" in system_instruction
    assert _query().original_question in user_prompt
    assert _evidence().text in user_prompt
    assert isinstance(provider, ChatModelProvider)
    assert isinstance(generator, AnswerGenerator)


def test_model_generator_rejects_unknown_or_missing_citations() -> None:
    """A model cannot attach an invented evidence ID or omit all grounding."""
    unknown = ModelBackedAnswerGenerator(
        _FixtureProvider(_completion(cited_evidence_ids=["E9"]))
    )
    missing = ModelBackedAnswerGenerator(
        _FixtureProvider(_completion(cited_evidence_ids=[]))
    )

    with pytest.raises(ModelError, match="not supplied"):
        unknown.generate(
            _query(),
            [_evidence()],
            RetrievalStrategy.HYBRID,
            "model-answer-query",
        )
    with pytest.raises(ModelError, match="schema"):
        missing.generate(
            _query(),
            [_evidence()],
            RetrievalStrategy.HYBRID,
            "model-answer-query",
        )


def test_model_generator_appends_verified_declared_markers_when_missing() -> None:
    """Known declared citations become visible without inventing an identity."""
    provider = _FixtureProvider(
        _completion(answer="Doanh nghiệp phải nộp thuế đúng thời hạn.")
    )

    response = ModelBackedAnswerGenerator(provider).generate(
        _query(),
        [_evidence()],
        RetrievalStrategy.HYBRID,
        "model-answer-query",
    )

    assert response.answer.endswith("[E1]")
    assert response.citations[0].evidence_id == "E1"


def test_model_generator_accepts_combined_bracket_markers() -> None:
    """Common combined marker syntax resolves to verified evidence identities."""
    provider = _FixtureProvider(
        _completion(
            answer="Hai căn cứ cùng hỗ trợ nhận định [E2, E1].",
            cited_evidence_ids=["E1", "E2"],
        )
    )

    response = ModelBackedAnswerGenerator(provider).generate(
        _query(),
        [
            _evidence("E1", "chunk-1"),
            _evidence("E2", "chunk-2"),
        ],
        RetrievalStrategy.HYBRID,
        "model-answer-query",
    )

    assert [item.evidence_id for item in response.citations] == ["E2", "E1"]


def test_model_generator_uses_verified_markers_as_citation_order() -> None:
    """Visible markers canonically select citations from the supplied allowlist."""
    provider = _FixtureProvider(
        _completion(
            answer="Nghĩa vụ thứ hai [E2], sau đó nghĩa vụ thứ nhất [E1].",
            cited_evidence_ids=["E1", "E2"],
        )
    )
    evidence = [
        _evidence("E1", "chunk-1"),
        _evidence("E2", "chunk-2"),
    ]

    response = ModelBackedAnswerGenerator(provider).generate(
        _query(),
        evidence,
        RetrievalStrategy.HYBRID,
        "model-answer-query",
    )

    assert [item.evidence_id for item in response.citations] == ["E2", "E1"]
    assert [item.chunk_id for item in response.citations] == [
        "chunk-2",
        "chunk-1",
    ]


def test_model_generator_rejects_unknown_visible_marker() -> None:
    """A marker not present in selected evidence remains a hard failure."""
    provider = _FixtureProvider(
        _completion(
            answer="Nhận định không có căn cứ [E9].",
            cited_evidence_ids=["E1"],
        )
    )

    with pytest.raises(ModelError, match="unknown evidence marker"):
        ModelBackedAnswerGenerator(provider).generate(
            _query(),
            [_evidence()],
            RetrievalStrategy.HYBRID,
            "model-answer-query",
        )


def test_model_generator_abstains_without_calling_provider() -> None:
    """Empty context never reaches the model and produces no citation."""
    provider = _FixtureProvider(_completion())
    response = ModelBackedAnswerGenerator(provider).generate(
        _query(),
        [],
        RetrievalStrategy.HYBRID,
        "model-answer-query",
    )

    assert response.insufficient_evidence is True
    assert response.citations == []
    assert provider.calls == []


def test_model_generator_accepts_json_fence_and_enforces_model_abstention() -> None:
    """A common JSON fence is parsed, but insufficient output stays fail-closed."""
    completion = _completion(
        answer="Không đủ căn cứ.",
        cited_evidence_ids=[],
        insufficient_evidence=True,
        warnings=["thiếu phạm vi áp dụng"],
    )
    provider = _FixtureProvider(f"```json\n{completion}\n```")

    response = ModelBackedAnswerGenerator(provider).generate(
        _query(),
        [_evidence()],
        RetrievalStrategy.HYBRID,
        "model-answer-query",
    )

    assert response.insufficient_evidence is True
    assert response.citations == []
    assert "model_reported_insufficient_evidence" in response.warnings
    assert "thiếu phạm vi áp dụng" in response.warnings


def test_model_generator_extracts_valid_json_after_model_preamble() -> None:
    """A harmless model preamble does not discard an otherwise strict draft."""
    provider = _FixtureProvider(
        f"Đây là JSON theo yêu cầu:\n```json\n{_completion()}\n```"
    )

    response = ModelBackedAnswerGenerator(provider).generate(
        _query(),
        [_evidence()],
        RetrievalStrategy.HYBRID,
        "model-answer-query",
    )

    assert response.answer.endswith("[E1].")
    assert len(provider.calls) == 1


def test_model_generator_retries_one_invalid_structured_completion() -> None:
    """One bounded correction attempt repairs format without weakening citations."""
    provider = _SequenceProvider(["không phải JSON", _completion()])

    response = ModelBackedAnswerGenerator(provider).generate(
        _query(),
        [_evidence()],
        RetrievalStrategy.HYBRID,
        "model-answer-query",
    )

    assert response.answer.endswith("[E1].")
    assert len(provider.calls) == 2
    assert "OUTPUT TRƯỚC KHÔNG HỢP LỆ" in provider.calls[1][1]


def test_model_generator_retries_partial_inline_citation_coverage() -> None:
    """A marker on one list item cannot silently ground the other claims."""
    provider = _SequenceProvider(
        [
            _completion(
                answer=(
                    "First legal claim [E1]; "
                    "second legal claim."
                )
            ),
            _completion(
                answer=(
                    "First legal claim [E1]; "
                    "second legal claim [E1]."
                )
            ),
        ]
    )

    response = ModelBackedAnswerGenerator(provider).generate(
        _query(),
        [_evidence()],
        RetrievalStrategy.HYBRID,
        "model-answer-query",
    )

    assert response.answer.endswith("[E1].")
    assert len(provider.calls) == 2


def test_model_generator_rejects_partial_inline_citation_without_retry() -> None:
    """Strict single-attempt mode surfaces incomplete claim grounding."""
    provider = _FixtureProvider(
        _completion(
            answer="First legal claim [E1]; second legal claim."
        )
    )

    with pytest.raises(ModelError, match="without inline evidence"):
        ModelBackedAnswerGenerator(
            provider,
            max_structured_output_retries=0,
        ).generate(
            _query(),
            [_evidence()],
            RetrievalStrategy.HYBRID,
            "model-answer-query",
        )


def test_model_generator_can_disable_structured_output_retry() -> None:
    """Configuration can preserve single-attempt behavior for strict consumers."""
    provider = _SequenceProvider(["không phải JSON", _completion()])

    with pytest.raises(ModelError, match="schema"):
        ModelBackedAnswerGenerator(
            provider,
            max_structured_output_retries=0,
        ).generate(
            _query(),
            [_evidence()],
            RetrievalStrategy.HYBRID,
            "model-answer-query",
        )
    assert len(provider.calls) == 1


def test_model_generator_rejects_duplicate_evidence_before_inference() -> None:
    """Ambiguous evidence identity fails before a model call."""
    provider = _FixtureProvider(_completion())
    values: Sequence[Evidence] = [_evidence(), _evidence()]

    with pytest.raises(DataValidationError, match="unique"):
        ModelBackedAnswerGenerator(provider).generate(
            _query(),
            values,
            RetrievalStrategy.HYBRID,
            "model-answer-query",
        )
    assert provider.calls == []
