"""Tests for exact Codabench submission packaging."""

import json
from pathlib import Path
from zipfile import ZipFile

import pytest
from pydantic import ValidationError

from legal_agentic_rag.competition.uit_dsc_2026 import (
    CodabenchSubmissionFormatter,
    CompetitionBatchRunner,
)
from legal_agentic_rag.exceptions import (
    ArtifactCompatibilityError,
    DataValidationError,
)
from legal_agentic_rag.schemas import (
    AnswerResponse,
    Citation,
    CompetitionSubmissionItem,
    RetrievalStrategy,
)


class _Answerer:
    def answer(self, request):  # type: ignore[no-untyped-def]
        return AnswerResponse(
            question=request.question,
            answer=f"Trả lời cho: {request.question}",
            insufficient_evidence=True,
            retrieval_strategy=RetrievalStrategy.HYBRID,
            trace_id=f"trace-{len(request.question)}",
        )


class _CitingAnswerer:
    def answer(self, request):  # type: ignore[no-untyped-def]
        return AnswerResponse(
            question=request.question,
            answer="Người lao động được nghỉ 12 ngày. [E1]",
            citations=[
                Citation(
                    evidence_id="E1",
                    chunk_id="chunk-1",
                    document_id="document-1",
                )
            ],
            insufficient_evidence=False,
            retrieval_strategy=RetrievalStrategy.HYBRID,
            trace_id="trace-citation",
        )


def _questions(path: Path) -> None:
    path.write_text(
        json.dumps(
            {
                "q_000001": {
                    "question": "Người lao động có quyền gì?",
                    "answer": "Đáp án tham chiếu không được sao chép.",
                },
                "q_000002": {"question": "Thời hạn là bao lâu?"},
            },
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )


def _complete_batch(tmp_path: Path) -> tuple[Path, Path]:
    questions = tmp_path / "public-official.json"
    batch = tmp_path / "batch"
    _questions(questions)
    CompetitionBatchRunner(
        _Answerer(),
        application_config_hash="a" * 64,
    ).run(questions, batch)
    return questions, batch


def test_formatter_writes_only_exact_utf8_submission_contract(
    tmp_path: Path,
) -> None:
    questions, batch = _complete_batch(tmp_path)
    output = tmp_path / "submission.zip"

    result = CodabenchSubmissionFormatter().format(questions, batch, output)

    assert result.question_count == 2
    assert len(result.archive_sha256) == 64
    assert len(result.submission_json_sha256) == 64
    with ZipFile(output) as archive:
        assert archive.namelist() == ["submission.json"]
        raw = archive.read("submission.json")
    assert "Người lao động" in raw.decode("utf-8")
    payload = json.loads(raw)
    assert payload == {
        "q_000001": {
            "answer": "Trả lời cho: Người lao động có quyền gì?",
        },
        "q_000002": {
            "answer": "Trả lời cho: Thời hạn là bao lâu?",
        },
    }
    # This is the exact projection executed by the organizer scorer.
    assert {key: value["answer"] for key, value in payload.items()} == {
        "q_000001": "Trả lời cho: Người lao động có quyền gì?",
        "q_000002": "Trả lời cho: Thời hạn là bao lâu?",
    }
    assert "Đáp án tham chiếu" not in raw.decode("utf-8")


def test_submission_archive_bytes_are_reproducible(tmp_path: Path) -> None:
    questions, batch = _complete_batch(tmp_path)
    first = tmp_path / "first" / "submission.zip"
    second = tmp_path / "second" / "submission.zip"

    first_result = CodabenchSubmissionFormatter().format(questions, batch, first)
    second_result = CodabenchSubmissionFormatter().format(questions, batch, second)

    assert first.read_bytes() == second.read_bytes()
    assert first_result.archive_sha256 == second_result.archive_sha256


def test_formatter_rejects_wrong_filename_and_existing_output(
    tmp_path: Path,
) -> None:
    questions, batch = _complete_batch(tmp_path)
    formatter = CodabenchSubmissionFormatter()

    with pytest.raises(DataValidationError, match="submission.zip"):
        formatter.format(questions, batch, tmp_path / "result.zip")

    output = tmp_path / "submission.zip"
    output.write_bytes(b"existing")
    with pytest.raises(ArtifactCompatibilityError, match="already exists"):
        formatter.format(questions, batch, output)


def test_formatter_rejects_incomplete_or_tampered_batch(tmp_path: Path) -> None:
    questions, batch = _complete_batch(tmp_path)
    records = (batch / "results.jsonl").read_text(encoding="utf-8").splitlines()
    (batch / "results.jsonl").write_text(
        records[0] + "\n",
        encoding="utf-8",
    )

    with pytest.raises(ArtifactCompatibilityError, match="every official question"):
        CodabenchSubmissionFormatter().format(
            questions,
            batch,
            tmp_path / "submission.zip",
        )


def test_submission_item_requires_exact_string_fields() -> None:
    with pytest.raises(ValidationError):
        CompetitionSubmissionItem.model_validate(
            {"id": "q_000001", "answer": ["not", "a", "string"]}
        )

    with pytest.raises(ValidationError):
        CompetitionSubmissionItem.model_validate(
            {"id": "q_000001", "answer": "ok", "citation": "E1"}
        )


def test_formatter_removes_only_verified_internal_citation_markers(
    tmp_path: Path,
) -> None:
    questions = tmp_path / "public.json"
    _questions(questions)
    batch = tmp_path / "batch"
    CompetitionBatchRunner(
        _CitingAnswerer(),
        application_config_hash="b" * 64,
    ).run(questions, batch)

    output = tmp_path / "submission.zip"
    CodabenchSubmissionFormatter().format(questions, batch, output)

    with ZipFile(output) as archive:
        payload = json.loads(archive.read("submission.json"))
    assert payload["q_000001"]["answer"] == "Người lao động được nghỉ 12 ngày."
    assert "[E1]" not in payload["q_000001"]["answer"]


def test_submission_loader_rejects_documented_legacy_array(tmp_path: Path) -> None:
    output = tmp_path / "submission.zip"
    with ZipFile(output, "w") as archive:
        archive.writestr(
            "submission.json",
            json.dumps([{"id": "q_000001", "answer": "answer"}]),
        )

    from legal_agentic_rag.competition.uit_dsc_2026.submission import (
        load_submission_archive,
    )

    with pytest.raises(DataValidationError, match="root must be an object"):
        load_submission_archive(output)


def test_submission_loader_rejects_extra_value_fields(tmp_path: Path) -> None:
    output = tmp_path / "submission.zip"
    with ZipFile(output, "w") as archive:
        archive.writestr(
            "submission.json",
            json.dumps(
                {
                    "q_000001": {
                        "answer": "answer",
                        "citation": "not allowed",
                    }
                }
            ),
        )

    from legal_agentic_rag.competition.uit_dsc_2026.submission import (
        load_submission_archive,
    )

    with pytest.raises(DataValidationError, match="contain only 'answer'"):
        load_submission_archive(output)
