"""Tests for evidence-only model-backed answer generation."""

from collections.abc import Sequence
import json

import pytest

from legal_agentic_rag.contracts import AnswerGenerator, ChatModelProvider
from legal_agentic_rag.exceptions import (
    DataValidationError,
    ModelError,
    StructuredGenerationError,
)
from legal_agentic_rag.generation import ModelBackedAnswerGenerator
from legal_agentic_rag.generation.claim_grounding import (
    extract_inline_evidence_ids,
    split_answer_claims,
)
from legal_agentic_rag.schemas import (
    AnswerGenerationCorrectionSignal,
    Evidence,
    RetrievalQuery,
    RetrievalStrategy,
    StructuredGenerationFailureCode,
)


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
    answer = updates.pop(
        "answer",
        "Doanh nghiệp phải nộp thuế đúng thời hạn.",
    )
    cited_evidence_ids = updates.pop("cited_evidence_ids", ["E1"])
    insufficient_evidence = updates.pop("insufficient_evidence", False)
    claims = updates.pop(
        "claims",
        []
        if insufficient_evidence
        else [
            {
                "text": answer,
                "evidence_ids": cited_evidence_ids,
            }
        ],
    )
    payload: dict[str, object] = {
        "claims": claims,
        "insufficient_evidence": insufficient_evidence,
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
    assert "VÍ DỤ OUTPUT ĐỦ CĂN CỨ" in user_prompt
    assert "Doanh nghiệp phải nộp thuế đúng thời hạn." in user_prompt
    assert "Có tối đa 4 phần tử claims" in user_prompt
    assert "chép nguyên văn" in system_instruction
    assert "Không đổi cách viết chữ/số" in user_prompt
    assert "OUTPUT_JSON_SCHEMA" not in user_prompt
    assert '"$defs"' not in user_prompt
    assert isinstance(provider, ChatModelProvider)
    assert isinstance(generator, AnswerGenerator)


def test_model_generator_numeric_repair_prompt_is_content_free_about_the_draft() -> None:
    """Numeric repair regenerates from evidence and a typed signal only."""
    provider = _FixtureProvider(_completion())
    rejected_draft = "Từ 22 đến 35 tuổi"

    ModelBackedAnswerGenerator(provider).generate(
        _query(),
        [_evidence()],
        RetrievalStrategy.HYBRID,
        "model-answer-query",
        AnswerGenerationCorrectionSignal.NUMERIC_MISMATCH,
    )

    prompt = provider.calls[0][1]
    assert "YÊU CẦU SỬA NUMERIC_MISMATCH" in prompt
    assert "tạo lại toàn bộ JSON từ đầu" in prompt
    assert rejected_draft not in prompt


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

    assert response.answer.endswith("[E1].")
    assert response.citations[0].evidence_id == "E1"


def test_model_generator_renders_multiple_claim_evidence_ids() -> None:
    """One claim can explicitly link multiple supplied evidence records."""
    provider = _FixtureProvider(
        _completion(
            claims=[
                {
                    "text": "Hai căn cứ cùng hỗ trợ nhận định.",
                    "evidence_ids": ["E2", "E1"],
                }
            ],
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
    assert response.answer.endswith("[E2] [E1].")


def test_model_generator_uses_verified_markers_as_citation_order() -> None:
    """Claim links canonically select citations from the supplied allowlist."""
    provider = _FixtureProvider(
        _completion(
            claims=[
                {
                    "text": "Nghĩa vụ thứ hai.",
                    "evidence_ids": ["E2"],
                },
                {
                    "text": "Sau đó là nghĩa vụ thứ nhất.",
                    "evidence_ids": ["E1"],
                },
            ],
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
    rendered_claims = split_answer_claims(response.answer)
    assert len(rendered_claims) == 2
    assert [extract_inline_evidence_ids(value) for value in rendered_claims] == [
        ["E2"],
        ["E1"],
    ]


def test_model_generator_rejects_unknown_visible_marker() -> None:
    """The model cannot bypass claim links by writing a marker in claim text."""
    provider = _FixtureProvider(
        _completion(
            answer="Nhận định không có căn cứ [E1].",
            cited_evidence_ids=["E1"],
        )
    )

    with pytest.raises(ModelError, match="must not contain evidence markers"):
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
    assert "json_decode_error" in provider.calls[1][1]


def test_model_generator_distinguishes_schema_validation_feedback() -> None:
    """Valid JSON with forbidden fields receives schema-specific correction."""
    invalid = json.dumps(
        {
            "claims": [],
            "insufficient_evidence": True,
            "warnings": [],
            "unexpected": True,
        }
    )
    provider = _SequenceProvider([invalid, _completion()])

    response = ModelBackedAnswerGenerator(provider).generate(
        _query(),
        [_evidence()],
        RetrievalStrategy.HYBRID,
        "model-answer-query",
    )

    assert response.answer.endswith("[E1].")
    assert "schema_validation_error" in provider.calls[1][1]


def test_model_generator_treats_json_null_as_schema_failure() -> None:
    """Syntactically valid JSON null is not mislabeled as a decoder failure."""
    provider = _SequenceProvider(["null", _completion()])

    ModelBackedAnswerGenerator(provider).generate(
        _query(),
        [_evidence()],
        RetrievalStrategy.HYBRID,
        "model-answer-query",
    )

    assert "schema_validation_error" in provider.calls[1][1]


def test_model_generator_retries_long_non_vietnamese_claim() -> None:
    """Long English drift is rejected without weakening evidence identity."""
    provider = _SequenceProvider(
        [
            _completion(
                answer=(
                    "Organizations must submit a complete annual report to the "
                    "competent authority where they operate."
                )
            ),
            _completion(),
        ]
    )

    response = ModelBackedAnswerGenerator(provider).generate(
        _query(),
        [_evidence()],
        RetrievalStrategy.HYBRID,
        "model-answer-query",
    )

    assert response.answer.endswith("[E1].")
    assert "non_vietnamese_claim" in provider.calls[1][1]
    assert "tiếng Việt có dấu" in provider.calls[1][1]


def test_model_generator_retries_multi_claim_item_with_specific_feedback() -> None:
    """One schema item cannot hide multiple claims behind one evidence link."""
    provider = _SequenceProvider(
        [
            _completion(answer="First legal claim; second legal claim."),
            _completion(
                claims=[
                    {
                        "text": "First legal claim.",
                        "evidence_ids": ["E1"],
                    },
                    {
                        "text": "Second legal claim.",
                        "evidence_ids": ["E1"],
                    },
                ]
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
    assert "claim_boundary_mismatch" in provider.calls[1][1]
    assert "một phần tử claims riêng" in provider.calls[1][1]


def test_model_generator_rejects_multi_claim_item_without_retry() -> None:
    """Strict single-attempt mode surfaces ambiguous claim-level grounding."""
    provider = _FixtureProvider(
        _completion(answer="First legal claim; second legal claim.")
    )

    with pytest.raises(ModelError, match="exactly one legal claim"):
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

    with pytest.raises(ModelError, match="JSON"):
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


def test_model_generator_exposes_only_a_closed_structured_failure_code() -> None:
    """A terminal malformed completion retains its typed reason without content."""
    provider = _FixtureProvider("not-json")

    with pytest.raises(StructuredGenerationError) as raised:
        ModelBackedAnswerGenerator(
            provider,
            max_structured_output_retries=0,
        ).generate(
            _query(),
            [_evidence()],
            RetrievalStrategy.HYBRID,
            "model-answer-query",
        )

    assert raised.value.failure_code == StructuredGenerationFailureCode.JSON_DECODE_ERROR


def test_model_generator_recovers_only_terminal_safe_schema_shape_errors() -> None:
    """One local repair avoids another model call and preserves trusted checks."""
    completion_payload = json.loads(_completion())
    completion_payload["claims"] = {
        "text": completion_payload["claims"][0]["text"],
        "evidence_ids": "E1",
        "discard": "private",
    }
    completion_payload["discard"] = "private"
    completion = json.dumps(completion_payload, ensure_ascii=False)
    provider = _FixtureProvider(completion)

    response = ModelBackedAnswerGenerator(
        provider,
        max_structured_output_retries=0,
        max_schema_recovery_attempts=1,
    ).generate(
        _query(),
        [_evidence()],
        RetrievalStrategy.HYBRID,
        "model-answer-query",
    )

    assert len(provider.calls) == 1
    assert response.answer.endswith("[E1].")
    assert response.metadata["schema_recovery"] == {
        "attempted": True,
        "count": 1,
        "outcome": "succeeded",
        "issue_codes": [
            "top_level_extra_fields",
            "claims_object_instead_of_list",
            "claim_extra_fields",
            "claim_evidence_id_scalar",
        ],
        "repair_codes": [
            "removed_top_level_extra_fields",
            "wrapped_single_claim",
            "removed_claim_extra_fields",
            "wrapped_scalar_evidence_id",
        ],
    }
    assert "private" not in json.dumps(response.metadata)


def test_model_generator_terminal_schema_failure_exposes_only_closed_diagnostics() -> None:
    """Nonrecoverable terminal schema output stays fail-closed without text leak."""
    private_completion = json.dumps(
        {
            "claims": [{"text": "PRIVATE_DRAFT_TEXT", "evidence_ids": "bad"}],
            "insufficient_evidence": False,
            "warnings": [],
        }
    )

    with pytest.raises(StructuredGenerationError) as raised:
        ModelBackedAnswerGenerator(
            _FixtureProvider(private_completion),
            max_structured_output_retries=0,
            max_schema_recovery_attempts=1,
        ).generate(
            _query(),
            [_evidence()],
            RetrievalStrategy.HYBRID,
            "model-answer-query",
        )

    assert raised.value.failure_code == StructuredGenerationFailureCode.SCHEMA_VALIDATION_ERROR
    assert raised.value.schema_issue_codes == ("invalid_claim_evidence_ids",)
    assert raised.value.schema_recovery_outcome == "not_recoverable"
    assert "PRIVATE_DRAFT_TEXT" not in str(raised.value)


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


def test_model_generator_missing_field_correction_triggers_one_extra_call_and_succeeds() -> None:
    """Eligible terminal missing-field failure makes exactly one correction call and succeeds."""
    missing_top_level = json.dumps(
        {
            "claims": [
                {
                    "text": "Doanh nghiệp phải nộp thuế đúng thời hạn.",
                    "evidence_ids": ["E1"],
                }
            ],
            "warnings": [],
        }
    )
    valid_completion = _completion()
    provider = _SequenceProvider([missing_top_level, valid_completion])

    generator = ModelBackedAnswerGenerator(
        provider,
        max_structured_output_retries=0,
        max_schema_recovery_attempts=1,
        max_missing_field_corrections=1,
    )
    response = generator.generate(
        _query(),
        [_evidence()],
        RetrievalStrategy.HYBRID,
        "model-answer-query",
    )

    assert len(provider.calls) == 2
    assert response.answer.endswith("[E1].")
    assert response.metadata["missing_field_correction"] == {
        "attempted": True,
        "count": 1,
        "outcome": "succeeded",
    }
    correction_prompt = provider.calls[1][1]
    assert "OUTPUT TRƯỚC BỊ THIẾU TRƯỜNG SCHEMA BẮT BUỘC" in correction_prompt
    assert "ValidationError" not in correction_prompt


def test_model_generator_missing_field_correction_disabled_by_default() -> None:
    """Disabled missing-field correction preserves exact M49.5 behavior and makes no extra call."""
    missing_top_level = json.dumps(
        {
            "claims": [
                {
                    "text": "Doanh nghiệp phải nộp thuế đúng thời hạn.",
                    "evidence_ids": ["E1"],
                }
            ],
            "warnings": [],
        }
    )
    provider = _FixtureProvider(missing_top_level)

    with pytest.raises(StructuredGenerationError) as raised:
        ModelBackedAnswerGenerator(
            provider,
            max_structured_output_retries=0,
            max_schema_recovery_attempts=1,
            max_missing_field_corrections=0,
        ).generate(
            _query(),
            [_evidence()],
            RetrievalStrategy.HYBRID,
            "model-answer-query",
        )

    assert len(provider.calls) == 1
    assert raised.value.failure_code == StructuredGenerationFailureCode.SCHEMA_VALIDATION_ERROR
    assert "missing_top_level_field" in raised.value.schema_issue_codes
    assert raised.value.missing_field_correction_attempted is False


def test_model_generator_missing_field_correction_fails_closed() -> None:
    """Failed final missing-field correction terminates fail-closed without looping."""
    missing_top_level = json.dumps(
        {
            "claims": [
                {
                    "text": "Doanh nghiệp phải nộp thuế đúng thời hạn.",
                    "evidence_ids": ["E1"],
                }
            ],
            "warnings": [],
        }
    )
    still_broken = json.dumps({"claims": "not-valid"})
    provider = _SequenceProvider([missing_top_level, still_broken])

    with pytest.raises(StructuredGenerationError) as raised:
        ModelBackedAnswerGenerator(
            provider,
            max_structured_output_retries=0,
            max_schema_recovery_attempts=1,
            max_missing_field_corrections=1,
        ).generate(
            _query(),
            [_evidence()],
            RetrievalStrategy.HYBRID,
            "model-answer-query",
        )

    assert len(provider.calls) == 2
    assert raised.value.missing_field_correction_attempted is True
    assert raised.value.missing_field_correction_outcome == "failed"


def test_model_generator_m495_local_repair_precedes_missing_field_correction() -> None:
    """M49.5 local structural repair avoids triggering M49.6 model correction."""
    completion_payload = json.loads(_completion())
    completion_payload["claims"] = {
        "text": completion_payload["claims"][0]["text"],
        "evidence_ids": "E1",
    }
    provider = _FixtureProvider(json.dumps(completion_payload, ensure_ascii=False))

    response = ModelBackedAnswerGenerator(
        provider,
        max_structured_output_retries=0,
        max_schema_recovery_attempts=1,
        max_missing_field_corrections=1,
    ).generate(
        _query(),
        [_evidence()],
        RetrievalStrategy.HYBRID,
        "model-answer-query",
    )

    assert len(provider.calls) == 1
    assert response.metadata["schema_recovery"]["outcome"] == "succeeded"
    assert "missing_field_correction" not in response.metadata


def test_model_generator_ineligible_errors_do_not_trigger_missing_field_correction() -> None:
    """Ineligible errors such as unknown evidence IDs or JSON decode errors do not trigger M49.6."""
    unknown_evidence_completion = json.dumps(
        {
            "claims": [
                {
                    "text": "Doanh nghiệp phải nộp thuế đúng thời hạn.",
                    "evidence_ids": ["E999"],
                }
            ],
            "insufficient_evidence": False,
            "warnings": [],
        }
    )
    provider = _FixtureProvider(unknown_evidence_completion)

    with pytest.raises(StructuredGenerationError) as raised:
        ModelBackedAnswerGenerator(
            provider,
            max_structured_output_retries=0,
            max_schema_recovery_attempts=1,
            max_missing_field_corrections=1,
        ).generate(
            _query(),
            [_evidence()],
            RetrievalStrategy.HYBRID,
            "model-answer-query",
        )

    assert len(provider.calls) == 1
    assert raised.value.failure_code == StructuredGenerationFailureCode.UNKNOWN_EVIDENCE_ID
    assert raised.value.missing_field_correction_attempted is False


def test_model_generator_missing_claim_field_correction_succeeds() -> None:
    """Missing claim-level field triggers one extra model correction call and succeeds."""
    missing_claim_field = json.dumps(
        {
            "claims": [
                {
                    "evidence_ids": ["E1"],
                }
            ],
            "insufficient_evidence": False,
            "warnings": [],
        }
    )
    valid_completion = _completion()
    provider = _SequenceProvider([missing_claim_field, valid_completion])

    generator = ModelBackedAnswerGenerator(
        provider,
        max_structured_output_retries=0,
        max_schema_recovery_attempts=1,
        max_missing_field_corrections=1,
    )
    response = generator.generate(
        _query(),
        [_evidence()],
        RetrievalStrategy.HYBRID,
        "model-answer-query",
    )

    assert len(provider.calls) == 2
    assert response.answer.endswith("[E1].")
    assert response.metadata["missing_field_correction"] == {
        "attempted": True,
        "count": 1,
        "outcome": "succeeded",
    }


def test_model_generator_generic_missing_required_field_only_is_ineligible() -> None:
    """Generic MISSING_REQUIRED_FIELD alone without top-level or claim classification is ineligible."""
    generic_only_error = StructuredGenerationError(
        "Generic missing field",
        failure_code=StructuredGenerationFailureCode.SCHEMA_VALIDATION_ERROR.value,
        schema_issue_codes=("missing_required_field",),
        schema_recovery_outcome="not_recoverable",
    )
    assert not ModelBackedAnswerGenerator._is_missing_field_correction_eligible(
        generic_only_error
    )

    top_level_error = StructuredGenerationError(
        "Missing top-level field",
        failure_code=StructuredGenerationFailureCode.SCHEMA_VALIDATION_ERROR.value,
        schema_issue_codes=("missing_top_level_field", "missing_required_field"),
        schema_recovery_outcome="not_recoverable",
    )
    assert ModelBackedAnswerGenerator._is_missing_field_correction_eligible(
        top_level_error
    )

    claim_level_error = StructuredGenerationError(
        "Missing claim-level field",
        failure_code=StructuredGenerationFailureCode.SCHEMA_VALIDATION_ERROR.value,
        schema_issue_codes=("missing_claim_field", "missing_required_field"),
        schema_recovery_outcome="not_recoverable",
    )
    assert ModelBackedAnswerGenerator._is_missing_field_correction_eligible(
        claim_level_error
    )


def test_model_generator_rejects_invalid_max_missing_field_corrections() -> None:
    """Bound validation enforces max_missing_field_corrections in {0, 1}."""
    provider = _FixtureProvider(_completion())
    with pytest.raises(ValueError, match="max_missing_field_corrections"):
        ModelBackedAnswerGenerator(provider, max_missing_field_corrections=2)
    with pytest.raises(ValueError, match="max_missing_field_corrections"):
        ModelBackedAnswerGenerator(provider, max_missing_field_corrections=-1)
