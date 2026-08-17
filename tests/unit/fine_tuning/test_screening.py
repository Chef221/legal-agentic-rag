"""Tests for DirectQAScreeningRunner and cached results persistence."""

from __future__ import annotations

from pathlib import Path
import pytest

from legal_agentic_rag.fine_tuning.screening import (
    DirectQAScreeningRunner,
    load_cached_direct_qa_results,
)
from legal_agentic_rag.schemas import CompetitionQuestion


def test_direct_qa_screening_runner_and_cache(tmp_path: Path) -> None:
    questions = [
        CompetitionQuestion(question_id="101", question="Luật lao động có mấy chương?"),
        CompetitionQuestion(question_id="102", question="Thời gian thử việc là bao lâu?"),
    ]

    def mock_generator(prompt: str) -> tuple[str, int, bool]:
        return f"Generated answer for: {prompt}", 15, False

    output_path = tmp_path / "screen_results.jsonl"
    runner = DirectQAScreeningRunner(max_new_tokens=256)

    results = runner.run(
        questions=questions,
        generator_fn=mock_generator,
        output_path=output_path,
        model_id="Qwen/Qwen2.5-3B-Instruct",
        model_revision="a1d308dfcc03e09da285d49d912439a655a571e8",
    )

    assert len(results) == 2
    assert output_path.exists()

    cached = load_cached_direct_qa_results(output_path)
    assert len(cached) == 2
    assert cached[0].question_id == "101"
    assert cached[0].generated_token_count == 15
    assert cached[0].status == "success"
    assert cached[1].question_id == "102"
