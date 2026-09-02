"""Tests for evidence-only model-backed answer generation."""

from collections.abc import Sequence
import json

import pytest

from legal_agentic_rag.contracts import AnswerGenerator, ChatModelProvider
from legal_agentic_rag.exceptions import DataValidationError, ModelError
from legal_agentic_rag.generation import ModelBackedAnswerGenerator
from legal_agentic_rag.generation import RuleBasedCitationVerifier
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


class _ErrorThenCompletionProvider(_FixtureProvider):
    """Raise one bounded model error before returning a valid completion."""

    def __init__(self, completion: str) -> None:
        super().__init__(completion)
        self._failed = False

    def complete(
        self,
        *,
        system_instruction: str,
        user_prompt: str,
    ) -> str:
        self.calls.append((system_instruction, user_prompt))
        if not self._failed:
            self._failed = True
            raise ModelError("transient fixture failure")
        return self.completion


class _AlwaysErrorProvider(_FixtureProvider):
    """Raise a model error for every bounded provider attempt."""

    def complete(
        self,
        *,
        system_instruction: str,
        user_prompt: str,
    ) -> str:
        self.calls.append((system_instruction, user_prompt))
        raise ModelError("persistent fixture failure")


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
    assert "điều kiện" in system_instruction
    assert "Không lặp lại" in system_instruction
    assert _query().original_question in user_prompt
    assert _evidence().text in user_prompt
    assert isinstance(provider, ChatModelProvider)
    assert isinstance(generator, AnswerGenerator)


def test_model_generator_supports_reference_complete_answer_style() -> None:
    """Reference-complete style asks for bounded grounded coverage."""
    provider = _FixtureProvider(_completion())
    generator = ModelBackedAnswerGenerator(
        provider,
        answer_style="reference_complete",
    )

    response = generator.generate(
        _query(),
        [_evidence()],
        RetrievalStrategy.HYBRID_RERANK,
        "model-answer-query",
    )

    system_instruction, user_prompt = provider.calls[0]
    assert "cấu trúc tra cứu pháp luật" in system_instruction
    assert "phải đầy đủ các ý trực tiếp" in user_prompt
    assert response.metadata["answer_style"] == "reference_complete"


def test_model_generator_supports_competition_reference_compact_prompt() -> None:
    """M48 derives presentation only from the question and uses compact JSON."""
    provider = _FixtureProvider(_completion())
    generator = ModelBackedAnswerGenerator(
        provider,
        answer_style="competition_reference",
        prompt_schema_mode="compact_example",
    )

    response = generator.generate(
        _query(),
        [_evidence()],
        RetrievalStrategy.HYBRID_RERANK,
        "model-answer-query",
    )

    system_instruction, user_prompt = provider.calls[0]
    assert "tập train chính thức" in system_instruction
    assert "OUTPUT_JSON_COMPACT_EXAMPLE" in user_prompt
    assert "OUTPUT_JSON_SCHEMA" not in user_prompt
    assert "trả lời đúng phạm vi câu hỏi" in user_prompt
    assert response.metadata["prompt_schema_mode"] == "compact_example"


def test_model_generator_accepts_grounded_plain_text_for_sft_runtime() -> None:
    """M49.1 aligns inference with answer-only SFT while preserving citations."""
    provider = _FixtureProvider(
        "Doanh nghiệp phải nộp thuế đúng thời hạn [E1]."
    )
    generator = ModelBackedAnswerGenerator(
        provider,
        answer_style="competition_reference",
        prompt_schema_mode="plain_text_markers",
    )

    response = generator.generate(
        _query(),
        [_evidence()],
        RetrievalStrategy.HYBRID_RERANK,
        "model-answer-query",
    )

    system_instruction, user_prompt = provider.calls[0]
    assert "không JSON" in system_instruction
    assert "OUTPUT_PLAIN_TEXT_WITH_MARKERS" in user_prompt
    assert "slug" in user_prompt
    assert response.citations[0].evidence_id == "E1"
    assert "plain_text_marker_recovery" in response.warnings
    assert response.metadata["prompt_schema_mode"] == "plain_text_markers"


def test_model_generator_rejects_plain_text_without_trusted_marker() -> None:
    """Plain SFT recovery cannot silently assign evidence to uncited prose."""
    generator = ModelBackedAnswerGenerator(
        _FixtureProvider("Doanh nghiệp phải nộp thuế đúng thời hạn."),
        prompt_schema_mode="plain_text_markers",
        max_structured_output_retries=0,
    )

    with pytest.raises(ModelError, match="schema"):
        generator.generate(
            _query(),
            [_evidence()],
            RetrievalStrategy.HYBRID_RERANK,
            "model-answer-query",
        )


def test_model_generator_repairs_one_failed_grounding_draft() -> None:
    """The generator can repair one numeric mismatch without weakening checks."""
    invalid = _completion(
        answer="Doanh nghiệp phải nộp thuế trong năm 2025 [E1]."
    )
    provider = _SequenceProvider([invalid, _completion()])
    generator = ModelBackedAnswerGenerator(
        provider,
        grounding_verifier=RuleBasedCitationVerifier(),
        max_grounding_repair_retries=1,
    )

    response = generator.generate(
        _query(),
        [_evidence()],
        RetrievalStrategy.HYBRID_RERANK,
        "model-answer-query",
    )

    assert len(provider.calls) == 2
    assert "numeric_mismatch" in provider.calls[1][1]
    assert "grounding_repair_attempted" in response.warnings
    assert response.answer.endswith("[E1].")


def test_model_generator_salvages_only_verified_claims_after_failed_repair() -> None:
    """Supported-claim salvage drops unsupported claims and verifies the result."""
    invalid = _completion(
        answer=(
            "Doanh nghiệp phải nộp thuế đúng thời hạn [E1]. "
            "Doanh nghiệp phải nộp thuế trong năm 2025 [E1]."
        )
    )
    provider = _SequenceProvider([invalid, invalid])
    verifier = RuleBasedCitationVerifier()
    generator = ModelBackedAnswerGenerator(
        provider,
        grounding_verifier=verifier,
        max_grounding_repair_retries=1,
        grounding_failure_policy="supported_claims",
    )

    response = generator.generate(
        _query(),
        [_evidence()],
        RetrievalStrategy.HYBRID_RERANK,
        "model-answer-query",
    )

    assert len(provider.calls) == 2
    assert "2025" not in response.answer
    assert "supported_claim_salvage_applied" in response.warnings
    assert verifier.verify(response, [_evidence()]).is_valid is True


def test_model_generator_retries_one_provider_model_error() -> None:
    """The retained policy retries one transient failure with the same prompt."""
    provider = _ErrorThenCompletionProvider(_completion())
    generator = ModelBackedAnswerGenerator(
        provider,
        max_model_error_retries=1,
    )

    response = generator.generate(
        _query(),
        [_evidence()],
        RetrievalStrategy.HYBRID_RERANK,
        "model-answer-query",
    )

    assert len(provider.calls) == 2
    assert provider.calls[0] == provider.calls[1]
    assert "generator_model_error_retried" in response.warnings


def test_model_generator_uses_verified_top_evidence_after_model_error() -> None:
    """M48 replaces a zero-information model fallback with bounded verbatim text."""
    provider = _AlwaysErrorProvider(_completion())
    verifier = RuleBasedCitationVerifier()
    generator = ModelBackedAnswerGenerator(
        provider,
        max_model_error_retries=1,
        model_failure_policy="top_evidence",
    )

    response = generator.generate(
        _query(),
        [_evidence()],
        RetrievalStrategy.HYBRID_RERANK,
        "model-answer-query",
    )

    assert len(provider.calls) == 2
    assert response.insufficient_evidence is False
    assert response.metadata["semantic_synthesis"] is False
    assert response.metadata["fallback_backend"] == "extractive_top_evidence_v1"
    assert "generator_model_error_fallback" in response.warnings
    assert _evidence().text in response.answer
    assert verifier.verify(response, [_evidence()]).is_valid is True


def test_model_generator_falls_back_after_unresolved_grounding() -> None:
    """M48 keeps fail-closed identity checks while avoiding generic abstention."""
    invalid = _completion(
        answer="Doanh nghiệp phải nộp thuế trong năm 2025 [E1]."
    )
    provider = _SequenceProvider([invalid, invalid])
    verifier = RuleBasedCitationVerifier()
    generator = ModelBackedAnswerGenerator(
        provider,
        grounding_verifier=verifier,
        max_grounding_repair_retries=1,
        grounding_failure_policy="supported_claims_or_top_evidence",
    )

    response = generator.generate(
        _query(),
        [_evidence()],
        RetrievalStrategy.HYBRID_RERANK,
        "model-answer-query",
    )

    assert response.metadata["semantic_synthesis"] is False
    assert "extractive_fallback_applied" in response.warnings
    assert "2025" not in response.answer
    assert verifier.verify(response, [_evidence()]).is_valid is True


def test_model_generator_standalone_salvage_removes_orphan_enumeration() -> None:
    """M48 salvage removes list counters and capitalizes surviving fragments."""
    evidence = _evidence().model_copy(
        update={
            "text": (
                "Nhiệm vụ gồm: 1. im lặng, phụ họa theo quan điểm trái pháp luật."
            )
        }
    )
    invalid = _completion(
        answer=(
            "Nhiệm vụ gồm: 1 [E1]. "
            "im lặng, phụ họa theo quan điểm trái pháp luật [E1]. "
            "Nhiệm vụ phải hoàn thành trong năm 2025 [E1]."
        )
    )
    provider = _SequenceProvider([invalid, invalid])
    verifier = RuleBasedCitationVerifier()
    generator = ModelBackedAnswerGenerator(
        provider,
        grounding_verifier=verifier,
        max_grounding_repair_retries=1,
        grounding_failure_policy="supported_claims",
        salvage_rendering="standalone",
    )

    response = generator.generate(
        _query(),
        [evidence],
        RetrievalStrategy.HYBRID_RERANK,
        "model-answer-query",
    )

    assert ": 1" not in response.answer
    assert "Im lặng" in response.answer
    assert "2025" not in response.answer
    assert verifier.verify(response, [evidence]).is_valid is True


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


def test_model_generator_plain_text_canonical_e_markers_preserved() -> None:
    """Canonical [E1] markers are accepted unchanged without numeric recovery warning."""
    provider = _FixtureProvider("Doanh nghiệp phải nộp thuế đúng hạn [E1].")
    generator = ModelBackedAnswerGenerator(
        provider,
        answer_style="competition_reference",
        prompt_schema_mode="plain_text_markers",
    )
    response = generator.generate(
        _query(),
        [_evidence("E1")],
        RetrievalStrategy.HYBRID_RERANK,
        "model-answer-query",
    )
    assert response.answer == "Doanh nghiệp phải nộp thuế đúng hạn [E1]."
    assert len(response.citations) == 1
    assert response.citations[0].evidence_id == "E1"
    assert "plain_text_marker_recovery" in response.warnings
    assert "plain_text_numeric_marker_recovery" not in response.warnings


def test_model_generator_plain_text_accepts_single_numeric_alias() -> None:
    """Numeric marker [1] is canonicalized to [E1] when E1 is supplied."""
    provider = _FixtureProvider("Doanh nghiệp phải nộp thuế đúng hạn [1].")
    generator = ModelBackedAnswerGenerator(
        provider,
        answer_style="competition_reference",
        prompt_schema_mode="plain_text_markers",
    )
    response = generator.generate(
        _query(),
        [_evidence("E1")],
        RetrievalStrategy.HYBRID_RERANK,
        "model-answer-query",
    )
    assert response.answer == "Doanh nghiệp phải nộp thuế đúng hạn [E1]."
    assert len(response.citations) == 1
    assert response.citations[0].evidence_id == "E1"
    assert "plain_text_marker_recovery" in response.warnings
    assert "plain_text_numeric_marker_recovery" in response.warnings


def test_model_generator_plain_text_accepts_other_numeric_alias() -> None:
    """Numeric marker [2] is canonicalized to [E2] when E2 is supplied."""
    provider = _FixtureProvider("Nghĩa vụ nộp thuế được quy định tại [2].")
    generator = ModelBackedAnswerGenerator(
        provider,
        answer_style="competition_reference",
        prompt_schema_mode="plain_text_markers",
    )
    response = generator.generate(
        _query(),
        [_evidence("E1"), _evidence("E2", chunk_id="chunk-2")],
        RetrievalStrategy.HYBRID_RERANK,
        "model-answer-query",
    )
    assert response.answer == "Nghĩa vụ nộp thuế được quy định tại [E2]."
    assert len(response.citations) == 1
    assert response.citations[0].evidence_id == "E2"
    assert "plain_text_numeric_marker_recovery" in response.warnings


def test_model_generator_plain_text_mixed_canonical_and_numeric_preserves_order() -> None:
    """Mixed [E1] ... [2] canonicalizes to [E1] ... [E2] preserving citation order."""
    provider = _FixtureProvider("Theo [E1], doanh nghiệp nộp thuế. Theo [2], có ngoại lệ.")
    generator = ModelBackedAnswerGenerator(
        provider,
        answer_style="competition_reference",
        prompt_schema_mode="plain_text_markers",
    )
    response = generator.generate(
        _query(),
        [_evidence("E1"), _evidence("E2", chunk_id="chunk-2")],
        RetrievalStrategy.HYBRID_RERANK,
        "model-answer-query",
    )
    assert response.answer == "Theo [E1], doanh nghiệp nộp thuế. Theo [E2], có ngoại lệ."
    assert [c.evidence_id for c in response.citations] == ["E1", "E2"]
    assert "plain_text_numeric_marker_recovery" in response.warnings


def test_model_generator_plain_text_rejects_unknown_numeric_alias_fail_closed() -> None:
    """Unknown numeric alias [9] raises ModelError when E9 was not supplied."""
    provider = _FixtureProvider("Thời hạn nộp thuế là 30 ngày [9].")
    generator = ModelBackedAnswerGenerator(
        provider,
        answer_style="competition_reference",
        prompt_schema_mode="plain_text_markers",
        max_structured_output_retries=0,
    )
    with pytest.raises(ModelError, match="supplied"):
        generator.generate(
            _query(),
            [_evidence("E1")],
            RetrievalStrategy.HYBRID_RERANK,
            "model-answer-query",
        )


def test_model_generator_plain_text_rejects_zero_marker() -> None:
    """[0] is not a valid numeric alias and fails schema parsing."""
    provider = _FixtureProvider("Thời hạn nộp thuế là 30 ngày [0].")
    generator = ModelBackedAnswerGenerator(
        provider,
        answer_style="competition_reference",
        prompt_schema_mode="plain_text_markers",
        max_structured_output_retries=0,
    )
    with pytest.raises(ModelError, match="schema"):
        generator.generate(
            _query(),
            [_evidence("E1")],
            RetrievalStrategy.HYBRID_RERANK,
            "model-answer-query",
        )


def test_model_generator_plain_text_rejects_leading_zero_marker() -> None:
    """[01] is not a valid numeric alias and fails schema parsing."""
    provider = _FixtureProvider("Thời hạn nộp thuế là 30 ngày [01].")
    generator = ModelBackedAnswerGenerator(
        provider,
        answer_style="competition_reference",
        prompt_schema_mode="plain_text_markers",
        max_structured_output_retries=0,
    )
    with pytest.raises(ModelError, match="schema"):
        generator.generate(
            _query(),
            [_evidence("E1")],
            RetrievalStrategy.HYBRID_RERANK,
            "model-answer-query",
        )


def test_model_generator_plain_text_rejects_uncited_prose() -> None:
    """Prose with no markers still fails schema parsing as before."""
    provider = _FixtureProvider("Thời hạn nộp thuế là 30 ngày kể từ ngày nhận thông báo.")
    generator = ModelBackedAnswerGenerator(
        provider,
        answer_style="competition_reference",
        prompt_schema_mode="plain_text_markers",
        max_structured_output_retries=0,
    )
    with pytest.raises(ModelError, match="schema"):
        generator.generate(
            _query(),
            [_evidence("E1")],
            RetrievalStrategy.HYBRID_RERANK,
            "model-answer-query",
        )


def test_model_generator_json_mode_does_not_use_numeric_recovery() -> None:
    """JSON schema mode does not gain numeric alias recovery."""
    provider = _FixtureProvider(
        json.dumps({
            "answer": "Doanh nghiệp nộp thuế đúng hạn [1].",
            "cited_evidence_ids": ["1"],
            "insufficient_evidence": False,
        })
    )
    generator = ModelBackedAnswerGenerator(
        provider,
        prompt_schema_mode="json_schema",
        max_structured_output_retries=0,
    )
    with pytest.raises(ModelError, match="schema"):
        generator.generate(
            _query(),
            [_evidence("E1")],
            RetrievalStrategy.HYBRID,
            "model-answer-query",
        )


def test_model_generator_numeric_recovery_does_not_mutate_input_evidence() -> None:
    """Input evidence list and individual Evidence instances remain immutable."""
    ev = _evidence("E1")
    ev_copy = ev.model_copy()
    evidence_list = [ev]

    provider = _FixtureProvider("Nộp thuế đúng hạn [1].")
    generator = ModelBackedAnswerGenerator(
        provider,
        prompt_schema_mode="plain_text_markers",
    )
    generator.generate(
        _query(),
        evidence_list,
        RetrievalStrategy.HYBRID_RERANK,
        "model-answer-query",
    )

    assert ev == ev_copy
    assert len(evidence_list) == 1
    assert evidence_list[0] is ev


def test_model_generator_forensic_shape_regression() -> None:
    """Exact forensic Q1 shape with trailing [1] recovers semantic synthesis."""
    forensic_completion = (
        "Theo khoản 1 Điều 5 Nghị định số 45/2022/NĐ-CP, "
        "thời hiệu xử phạt vi phạm hành chính trong lĩnh vực bảo vệ môi trường là 02 năm.\n"
        "[1]"
    )
    provider = _FixtureProvider(forensic_completion)
    generator = ModelBackedAnswerGenerator(
        provider,
        answer_style="competition_reference",
        prompt_schema_mode="plain_text_markers",
    )
    ev = Evidence(
        evidence_id="E1",
        chunk_id="chunk-env-1",
        document_id="doc-45-2022",
        text="Thời hiệu xử phạt vi phạm hành chính trong lĩnh vực bảo vệ môi trường là 02 năm.",
        article_number="5",
        document_title="Nghị định 45/2022/NĐ-CP",
        document_number="45/2022/NĐ-CP",
    )
    response = generator.generate(
        RetrievalQuery(
            query_id="q1-forensic",
            original_question="Thời hiệu xử phạt vi phạm hành chính trong lĩnh vực bảo vệ môi trường là bao lâu?",
            normalized_question="thời hiệu xử phạt môi trường",
            top_k=1,
            candidate_k=1,
        ),
        [ev],
        RetrievalStrategy.HYBRID_RERANK,
        "q1-forensic",
    )
    assert response.metadata.get("semantic_synthesis") is True
    assert "generator_model_error_fallback" not in response.warnings
    assert "plain_text_numeric_marker_recovery" in response.warnings
    assert "[E1]" in response.answer
    assert "[1]" not in response.answer
    assert len(response.citations) == 1
    assert response.citations[0].evidence_id == "E1"
