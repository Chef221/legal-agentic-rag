"""Tests for checkpointed competition batch inference."""

import json
from pathlib import Path

import pytest

from legal_agentic_rag.competition.uit_dsc_2026.batch_inference import (
    CompetitionBatchRunner,
)
from legal_agentic_rag.competition.uit_dsc_2026 import batch_inference
from legal_agentic_rag.exceptions import ArtifactCompatibilityError
from legal_agentic_rag.schemas import AnswerResponse, RetrievalStrategy


class _Answerer:
    def __init__(self, *, fail_on_call: int | None = None) -> None:
        self.questions: list[str] = []
        self._fail_on_call = fail_on_call

    def answer(self, request):  # type: ignore[no-untyped-def]
        self.questions.append(request.question)
        if self._fail_on_call == len(self.questions):
            raise RuntimeError("interrupted")
        return AnswerResponse(
            question=request.question,
            answer=f"Dự đoán: {request.question}",
            insufficient_evidence=True,
            retrieval_strategy=RetrievalStrategy.HYBRID,
            trace_id=f"trace-{len(self.questions)}",
        )


def _questions(path: Path) -> None:
    path.write_text(
        json.dumps(
            {
                "q1": {"question": "Câu hỏi một?", "answer": "Gold một"},
                "q2": {"question": "Câu hỏi hai?", "answer": "Gold hai"},
            },
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )


def test_batch_is_complete_reusable_and_does_not_copy_gold(tmp_path: Path) -> None:
    questions = tmp_path / "warmup.json"
    output = tmp_path / "batch"
    _questions(questions)
    answerer = _Answerer()
    runner = CompetitionBatchRunner(answerer, application_config_hash="a" * 64)

    manifest = runner.run(questions, output)
    second = runner.run(questions, output)

    assert manifest == second
    assert manifest.record_count == 2
    assert answerer.questions == ["Câu hỏi một?", "Câu hỏi hai?"]
    text = (output / "results.jsonl").read_text(encoding="utf-8")
    assert "Gold một" not in text
    assert "Gold hai" not in text


def test_batch_resumes_only_missing_ordered_questions(tmp_path: Path) -> None:
    questions = tmp_path / "public.json"
    output = tmp_path / "batch"
    _questions(questions)
    interrupted = CompetitionBatchRunner(
        _Answerer(fail_on_call=2), application_config_hash="b" * 64
    )
    with pytest.raises(RuntimeError, match="interrupted"):
        interrupted.run(questions, output)

    resumed_answerer = _Answerer()
    manifest = CompetitionBatchRunner(
        resumed_answerer, application_config_hash="b" * 64
    ).run(questions, output)

    assert manifest.record_count == 2
    assert resumed_answerer.questions == ["Câu hỏi hai?"]


def test_batch_rejects_changed_question_bytes(tmp_path: Path) -> None:
    questions = tmp_path / "public.json"
    output = tmp_path / "batch"
    _questions(questions)
    runner = CompetitionBatchRunner(_Answerer(), application_config_hash="c" * 64)
    runner.run(questions, output)
    questions.write_text('{"q1":{"question":"Đã đổi?"}}', encoding="utf-8")

    with pytest.raises(ArtifactCompatibilityError, match="incompatible"):
        runner.run(questions, output)


def test_batch_logs_each_durable_record_when_requested(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    questions = tmp_path / "public.json"
    output = tmp_path / "batch"
    _questions(questions)

    progress_updates: list[int] = []

    def record_info(message: str, *, extra: dict[str, object]) -> None:
        if message == "competition_batch_progress":
            progress_updates.append(int(extra["completed_question_count"]))

    monkeypatch.setattr(batch_inference._LOGGER, "info", record_info)
    CompetitionBatchRunner(
        _Answerer(),
        application_config_hash="d" * 64,
        progress_interval=1,
    ).run(questions, output)

    assert progress_updates == [1, 2]


def test_batch_rejects_non_positive_progress_interval() -> None:
    with pytest.raises(ValueError, match="positive"):
        CompetitionBatchRunner(
            _Answerer(), application_config_hash="e" * 64, progress_interval=0
        )
