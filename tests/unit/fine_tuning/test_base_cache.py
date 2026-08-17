"""Tests for BASE direct QA screening cache persistence and fail-closed validation."""

from __future__ import annotations

from datetime import UTC, datetime
from pathlib import Path
import pytest

from legal_agentic_rag.exceptions import ArtifactCompatibilityError
from legal_agentic_rag.fine_tuning.screening import (
    load_and_validate_base_direct_qa_cache,
    save_base_direct_qa_cache,
)
from legal_agentic_rag.schemas import DirectQACaseResult


def test_base_direct_qa_cache_roundtrip(tmp_path: Path) -> None:
    screen_holdout = tmp_path / "screen_holdout.json"
    screen_holdout.write_text('{"1": {"question": "Q1?", "answer": "A1."}}', encoding="utf-8")

    jsonl_path = tmp_path / "m50-screen-base-results.jsonl"
    manifest_path = tmp_path / "m50-screen-base-manifest.json"

    results = [
        DirectQACaseResult(
            question_id="1",
            question="Q1?",
            generated_answer="Generated answer 1.",
            generated_token_count=10,
            model_id="Qwen/Qwen2.5-3B-Instruct",
            model_revision="a1d308dfcc03e09da285d49d912439a655a571e8",
            created_at=datetime.now(UTC),
        )
    ]

    manifest = save_base_direct_qa_cache(
        jsonl_output_path=jsonl_path,
        manifest_output_path=manifest_path,
        results=results,
        screen_holdout_path=screen_holdout,
        base_model_id="Qwen/Qwen2.5-3B-Instruct",
        base_model_revision="a1d308dfcc03e09da285d49d912439a655a571e8",
        tokenizer_revision="a1d308dfcc03e09da285d49d912439a655a571e8",
    )

    assert manifest.record_count == 1
    assert jsonl_path.exists()
    assert manifest_path.exists()

    loaded = load_and_validate_base_direct_qa_cache(
        jsonl_path=jsonl_path,
        manifest_path=manifest_path,
        screen_holdout_path=screen_holdout,
        expected_record_count=1,
    )
    assert len(loaded) == 1
    assert loaded[0].question_id == "1"


def test_base_direct_qa_cache_fails_closed_on_corrupted_data(tmp_path: Path) -> None:
    screen_holdout = tmp_path / "screen_holdout.json"
    screen_holdout.write_text('{"1": {"question": "Q1?", "answer": "A1."}}', encoding="utf-8")

    jsonl_path = tmp_path / "m50-screen-base-results.jsonl"
    manifest_path = tmp_path / "m50-screen-base-manifest.json"

    results = [
        DirectQACaseResult(
            question_id="1",
            question="Q1?",
            generated_answer="Generated answer 1.",
            generated_token_count=10,
            model_id="Qwen/Qwen2.5-3B-Instruct",
            model_revision="a1d308dfcc03e09da285d49d912439a655a571e8",
            created_at=datetime.now(UTC),
        )
    ]

    save_base_direct_qa_cache(
        jsonl_output_path=jsonl_path,
        manifest_output_path=manifest_path,
        results=results,
        screen_holdout_path=screen_holdout,
        base_model_id="Qwen/Qwen2.5-3B-Instruct",
        base_model_revision="a1d308dfcc03e09da285d49d912439a655a571e8",
        tokenizer_revision="a1d308dfcc03e09da285d49d912439a655a571e8",
    )

    # Modify JSONL data
    jsonl_path.write_text(jsonl_path.read_text(encoding="utf-8") + "\ncorrupted", encoding="utf-8")

    with pytest.raises(ArtifactCompatibilityError, match="corrupted"):
        load_and_validate_base_direct_qa_cache(
            jsonl_path=jsonl_path,
            manifest_path=manifest_path,
            screen_holdout_path=screen_holdout,
        )


class _MockAuditTokenizer:
    def apply_chat_template(
        self,
        messages: list[dict[str, str]],
        tokenize: bool = False,
        add_generation_prompt: bool = False,
    ) -> str:
        text = ""
        for m in messages:
            text += f"<|im_start|>{m['role']}\n{m['content']}<|im_end|>\n"
        if add_generation_prompt:
            text += "<|im_start|>assistant\n"
        return text

    def encode(self, text: str, add_special_tokens: bool = False) -> list[int]:
        return [1] * len(text.split())


def test_screen_token_audit_roundtrip_and_validation(tmp_path: Path) -> None:
    from legal_agentic_rag.fine_tuning.screening import (
        create_and_save_screen_token_audit,
        load_and_validate_screen_token_audit,
    )
    from legal_agentic_rag.schemas import CompetitionQuestion

    screen_holdout = tmp_path / "screen_holdout.json"
    screen_holdout.write_text('{"1": {"question": "Q1?", "answer": "A1."}}', encoding="utf-8")

    audit_path = tmp_path / "m50-screen-token-audit.json"
    questions = [
        CompetitionQuestion(question_id="1", question="Q1?", reference_answer="A1.")
    ]

    report = create_and_save_screen_token_audit(
        output_path=audit_path,
        screen_holdout_path=screen_holdout,
        questions=questions,
        tokenizer=_MockAuditTokenizer(),
    )

    assert report.question_count == 1
    assert report.selected_max_new_tokens == 1536
    assert audit_path.exists()

    # Successful validation
    cap = load_and_validate_screen_token_audit(
        audit_path=audit_path,
        screen_holdout_path=screen_holdout,
        expected_question_count=1,
    )
    assert cap == 1536

    # Modify screen_holdout -> fails closed on hash mismatch
    screen_holdout.write_text('{"1": {"question": "Q1?", "answer": "Changed."}}', encoding="utf-8")
    with pytest.raises(ArtifactCompatibilityError, match="Screen token audit mismatch"):
        load_and_validate_screen_token_audit(
            audit_path=audit_path,
            screen_holdout_path=screen_holdout,
            expected_question_count=1,
        )


def test_base_direct_qa_cache_rejects_mismatched_generation_config(tmp_path: Path) -> None:
    screen_holdout = tmp_path / "screen_holdout.json"
    screen_holdout.write_text('{"1": {"question": "Q1?", "answer": "A1."}}', encoding="utf-8")

    jsonl_path = tmp_path / "m50-screen-base-results.jsonl"
    manifest_path = tmp_path / "m50-screen-base-manifest.json"

    results = [
        DirectQACaseResult(
            question_id="1",
            question="Q1?",
            generated_answer="Generated answer 1.",
            generated_token_count=10,
            model_id="Qwen/Qwen2.5-3B-Instruct",
            model_revision="a1d308dfcc03e09da285d49d912439a655a571e8",
            created_at=datetime.now(UTC),
        )
    ]

    save_base_direct_qa_cache(
        jsonl_output_path=jsonl_path,
        manifest_output_path=manifest_path,
        results=results,
        screen_holdout_path=screen_holdout,
        base_model_id="Qwen/Qwen2.5-3B-Instruct",
        base_model_revision="a1d308dfcc03e09da285d49d912439a655a571e8",
        tokenizer_revision="a1d308dfcc03e09da285d49d912439a655a571e8",
        generation_config={"max_new_tokens": 1536, "do_sample": False},
    )

    # Rejection on max_new_tokens mismatch
    with pytest.raises(ArtifactCompatibilityError, match="max_new_tokens mismatch"):
        load_and_validate_base_direct_qa_cache(
            jsonl_path=jsonl_path,
            manifest_path=manifest_path,
            screen_holdout_path=screen_holdout,
            expected_max_new_tokens=2048,
            expected_record_count=1,
        )

    # Rejection on model revision mismatch
    with pytest.raises(ArtifactCompatibilityError, match="model revision mismatch"):
        load_and_validate_base_direct_qa_cache(
            jsonl_path=jsonl_path,
            manifest_path=manifest_path,
            screen_holdout_path=screen_holdout,
            expected_base_model_revision="wrong_revision",
            expected_record_count=1,
        )

    # Rejection on prompt mismatch
    with pytest.raises(ArtifactCompatibilityError, match="system prompt mismatch"):
        load_and_validate_base_direct_qa_cache(
            jsonl_path=jsonl_path,
            manifest_path=manifest_path,
            screen_holdout_path=screen_holdout,
            expected_system_prompt="Different system prompt",
            expected_record_count=1,
        )
