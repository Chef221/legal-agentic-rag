"""Tests for leakage-aware official development partitioning."""

import json
from pathlib import Path

import pytest

from legal_agentic_rag.competition.uit_dsc_2026.development_split import (
    CompetitionDevelopmentSplitter,
)
from legal_agentic_rag.exceptions import ArtifactCompatibilityError


def _write(path: Path, payload: dict[str, object]) -> None:
    path.write_text(
        json.dumps(payload, ensure_ascii=False), encoding="utf-8"
    )


def test_split_quarantines_holdout_overlap_and_preserves_records(
    tmp_path: Path,
) -> None:
    train = tmp_path / "train.json"
    public = tmp_path / "public.json"
    output = tmp_path / "split"
    original = {
        "q1": {"question": "Người lao động được nghỉ bao nhiêu ngày?", "answer": "12 ngày"},
        "q2": {"question": "Doanh nghiệp phải nộp thuế khi nào?", "answer": "Theo luật"},
        "q3": {"question": "Hợp đồng có hiệu lực từ lúc nào?", "answer": "Khi ký"},
        "q4": {"question": "Ai được thành lập công ty?", "answer": "Cá nhân"},
    }
    _write(train, original)
    _write(
        public,
        {"p1": {"question": "  NGƯỜI lao động được nghỉ bao nhiêu ngày ?"}},
    )

    manifest = CompetitionDevelopmentSplitter().split(
        train, [public], output, dev_fraction=0.34, seed=7
    )

    quarantined = json.loads(
        (output / "quarantined.json").read_text(encoding="utf-8")
    )
    assert quarantined == {"q1": original["q1"]}
    partitions = [
        json.loads((output / name).read_text(encoding="utf-8"))
        for name in ("training.json", "development.json", "quarantined.json")
    ]
    assert set().union(*(set(partition) for partition in partitions)) == set(original)
    assert sum(len(partition) for partition in partitions) == len(original)
    assert manifest.training_source.question_count == 4
    assert len(manifest.partitions) == 3


def test_split_is_deterministic_and_refuses_overwrite(tmp_path: Path) -> None:
    train = tmp_path / "train.json"
    _write(
        train,
        {
            f"q{index}": {"question": f"Câu hỏi pháp luật số {index}", "answer": str(index)}
            for index in range(10)
        },
    )
    splitter = CompetitionDevelopmentSplitter()
    first = tmp_path / "first"
    second = tmp_path / "second"
    splitter.split(train, [], first, seed=99)
    splitter.split(train, [], second, seed=99)

    for filename in ("training.json", "development.json", "quarantined.json"):
        assert (first / filename).read_bytes() == (second / filename).read_bytes()
    with pytest.raises(ArtifactCompatibilityError, match="already exists"):
        splitter.split(train, [], first, seed=99)
