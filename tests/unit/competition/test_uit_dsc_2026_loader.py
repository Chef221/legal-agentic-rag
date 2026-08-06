"""Tests for strict UIT DSC 2026 Task 2 JSON boundaries."""

import json
from pathlib import Path
from zipfile import ZipFile

import pytest

from legal_agentic_rag.competition import UitDsc2026DataLoader
from legal_agentic_rag.exceptions import DatasetSchemaError


def test_loader_reads_answer_records_without_modifying_official_text(
    tmp_path: Path,
) -> None:
    """Warm-up order and exact question/answer text survive raw mapping."""
    source = tmp_path / "warmup.json"
    source.write_text(
        json.dumps(
            {
                "201": {
                    "question": "  Câu hỏi pháp luật?  ",
                    "answer": "Theo căn cứ được cung cấp.",
                },
                "305": {
                    "question": "Câu hỏi thứ hai?",
                    "answer": "Câu trả lời thứ hai.",
                },
            },
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )

    records = UitDsc2026DataLoader().load_questions(
        source,
        require_reference_answers=True,
    )

    assert [record.question_id for record in records] == ["201", "305"]
    assert records[0].question == "  Câu hỏi pháp luật?  "
    assert records[0].reference_answer == "Theo căn cứ được cung cấp."


def test_loader_accepts_official_question_only_split(tmp_path: Path) -> None:
    """Public/private inputs may omit answers without inventing labels."""
    source = tmp_path / "public-official.json"
    source.write_text(
        '{"q-1":{"question":"Câu hỏi chưa có gold?"}}',
        encoding="utf-8",
    )

    records = UitDsc2026DataLoader().load_questions(
        source,
        require_reference_answers=False,
    )

    assert records[0].reference_answer is None


def test_loader_rejects_duplicate_json_keys(tmp_path: Path) -> None:
    """JSON parsing cannot silently overwrite an official record."""
    source = tmp_path / "warmup.json"
    source.write_text(
        '{"201":{"question":"Một?","answer":"Một."},'
        '"201":{"question":"Hai?","answer":"Hai."}}',
        encoding="utf-8",
    )

    with pytest.raises(DatasetSchemaError, match="Duplicate JSON key"):
        UitDsc2026DataLoader().load_questions(
            source,
            require_reference_answers=True,
        )


def test_loader_rejects_unknown_question_fields(tmp_path: Path) -> None:
    """Format drift remains explicit instead of entering the core as metadata."""
    source = tmp_path / "warmup.json"
    source.write_text(
        '{"201":{"question":"Một?","answer":"Một.","context":"raw"}}',
        encoding="utf-8",
    )

    with pytest.raises(DatasetSchemaError, match="unknown fields"):
        UitDsc2026DataLoader().load_questions(
            source,
            require_reference_answers=True,
        )


def test_loader_reads_documented_context_files(tmp_path: Path) -> None:
    """Organizer context names map once into stable competition fields."""
    first = {
        "id": "ctx-1",
        "name": "Văn bản kiểm thử",
        "link": "https://example.invalid/context-1",
        "passage": "Điều 1. Nội dung pháp luật kiểm thử.",
    }
    second = {
        "id": "ctx-2",
        "name": "Văn bản thứ hai",
        "link": "https://example.invalid/context-2",
        "passage": "Điều 2. Nội dung thứ hai.",
    }
    (tmp_path / "context_0002.json").write_text(
        json.dumps(second, ensure_ascii=False), encoding="utf-8"
    )
    (tmp_path / "context_0001.json").write_text(
        json.dumps(first, ensure_ascii=False), encoding="utf-8"
    )

    records = UitDsc2026DataLoader().load_contexts(tmp_path)

    assert [record.context_id for record in records] == ["ctx-1", "ctx-2"]
    assert records[0].title == "Văn bản kiểm thử"
    assert records[0].passage.startswith("Điều 1")


def test_loader_accepts_audited_numeric_id_optional_name_and_blank_passage(
    tmp_path: Path,
) -> None:
    """The released corpus variants map without leaking numeric IDs into core."""
    records = [
        {
            "id": 740,
            "link": "https://example.invalid/context-740",
            "passage": "",
        },
        {
            "id": " 741 ",
            "name": "Văn bản có tên",
            "link": "https://example.invalid/context-741",
            "passage": "Điều 1. Nội dung.",
        },
    ]
    for index, record in enumerate(records, start=1):
        (tmp_path / f"context_{index:04d}.json").write_text(
            json.dumps(record, ensure_ascii=False),
            encoding="utf-8",
        )

    loaded = UitDsc2026DataLoader().load_contexts(tmp_path)

    assert [item.context_id for item in loaded] == ["740", "741"]
    assert loaded[0].title is None
    assert loaded[0].passage == ""


@pytest.mark.parametrize("invalid_id", [True, -1, 1.5, None, "   "])
def test_loader_rejects_invalid_raw_context_ids(
    tmp_path: Path,
    invalid_id: object,
) -> None:
    """Only audited integer/string identities can cross the raw boundary."""
    (tmp_path / "context_0001.json").write_text(
        json.dumps(
            {
                "id": invalid_id,
                "link": "https://example.invalid/context",
                "passage": "Điều 1. Nội dung.",
            },
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )

    with pytest.raises(DatasetSchemaError, match="invalid context ID"):
        UitDsc2026DataLoader().load_contexts(tmp_path)


def test_loader_rejects_canonical_context_id_collision(tmp_path: Path) -> None:
    """Integer 740 and string '740' cannot become two unified documents."""
    for index, raw_id in enumerate((740, "740"), start=1):
        (tmp_path / f"context_{index:04d}.json").write_text(
            json.dumps(
                {
                    "id": raw_id,
                    "link": f"https://example.invalid/context-{index}",
                    "passage": f"Điều {index}. Nội dung.",
                },
                ensure_ascii=False,
            ),
            encoding="utf-8",
        )

    with pytest.raises(DatasetSchemaError, match="Duplicate competition context"):
        UitDsc2026DataLoader().load_contexts(tmp_path)


def test_loader_rejects_unknown_context_fields(tmp_path: Path) -> None:
    """The audited optional name does not permit arbitrary raw schema drift."""
    (tmp_path / "context_0001.json").write_text(
        json.dumps(
            {
                "id": 740,
                "link": "https://example.invalid/context",
                "passage": "Điều 1. Nội dung.",
                "relationship": "external",
            },
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )

    with pytest.raises(DatasetSchemaError, match="unknown fields"):
        UitDsc2026DataLoader().load_contexts(tmp_path)


def test_loader_rejects_duplicate_context_identity(tmp_path: Path) -> None:
    """Two files cannot silently describe the same official context ID."""
    record = {
        "id": "ctx-1",
        "name": "Văn bản",
        "link": "https://example.invalid/context",
        "passage": "Điều 1. Nội dung.",
    }
    for index in (1, 2):
        (tmp_path / f"context_{index:04d}.json").write_text(
            json.dumps(record, ensure_ascii=False), encoding="utf-8"
        )

    with pytest.raises(DatasetSchemaError, match="Duplicate competition context"):
        UitDsc2026DataLoader().load_contexts(tmp_path)


def test_loader_rejects_context_list_root(tmp_path: Path) -> None:
    """The official contract is one context object per JSON member."""
    record = {
        "id": "ctx-1",
        "name": "Văn bản",
        "link": "https://example.invalid/context",
        "passage": "Điều 1. Nội dung.",
    }
    (tmp_path / "context_0001.json").write_text(
        json.dumps([record], ensure_ascii=False),
        encoding="utf-8",
    )

    with pytest.raises(DatasetSchemaError, match="must contain one object"):
        UitDsc2026DataLoader().load_contexts(tmp_path)


def test_zip_and_directory_have_identical_canonical_revision(
    tmp_path: Path,
) -> None:
    """Packaging metadata cannot change official corpus lineage."""
    source = tmp_path / "contexts"
    source.mkdir()
    payloads = {
        "context_0001.json": {
            "id": "ctx-1",
            "name": "Văn bản một",
            "link": "https://example.invalid/one",
            "passage": "Điều 1. Nội dung thứ nhất.",
        },
        "context_0002.json": {
            "id": "ctx-2",
            "name": "Văn bản hai",
            "link": "https://example.invalid/two",
            "passage": "Điều 2. Nội dung thứ hai.",
        },
    }
    for name, payload in payloads.items():
        (source / name).write_text(
            json.dumps(payload, ensure_ascii=False),
            encoding="utf-8",
        )
    archive_path = tmp_path / "selected-contexts.zip"
    with ZipFile(archive_path, "w") as archive:
        for name in reversed(list(payloads)):
            archive.writestr(
                f"selected-contexts/{name}",
                (source / name).read_bytes(),
            )

    loader = UitDsc2026DataLoader()
    directory_identity = loader.inspect_context_source(source)
    archive_identity = loader.inspect_context_source(archive_path)

    assert directory_identity.revision == archive_identity.revision
    assert directory_identity.member_count == archive_identity.member_count == 2
    assert [item.context_id for item in loader.load_contexts(archive_path)] == [
        "ctx-1",
        "ctx-2",
    ]


def test_loader_rejects_duplicate_zip_member_basename(tmp_path: Path) -> None:
    """Nested archive folders cannot hide two records with one official name."""
    archive_path = tmp_path / "selected-contexts.zip"
    payload = json.dumps(
        {
            "id": "ctx-1",
            "name": "Văn bản",
            "link": "https://example.invalid/context",
            "passage": "Điều 1. Nội dung.",
        },
        ensure_ascii=False,
    )
    with ZipFile(archive_path, "w") as archive:
        archive.writestr("first/context_0001.json", payload)
        archive.writestr("second/context_0001.json", payload)

    with pytest.raises(
        DatasetSchemaError,
        match="Duplicate competition context member",
    ):
        UitDsc2026DataLoader().load_contexts(archive_path)
