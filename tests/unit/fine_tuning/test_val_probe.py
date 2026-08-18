"""Unit tests for deterministic VAL probe extraction and BASE cache validation."""

from __future__ import annotations

import json
from pathlib import Path
import pytest

from legal_agentic_rag.exceptions import (
    ArtifactCompatibilityError,
    DataValidationError,
)
from legal_agentic_rag.fine_tuning.val_probe import (
    VAL_PROBE_BASE_MANIFEST_FILENAME,
    VAL_PROBE_BASE_RESULTS_FILENAME,
    VAL_PROBE_FILENAME,
    VAL_PROBE_MANIFEST_FILENAME,
    compute_duplicate_line_ratio,
    compute_repeat_ngram_ratio,
    create_deterministic_val_probe,
    load_and_validate_val_probe_base_cache,
)
from legal_agentic_rag.schemas import CompetitionQuestion, ValProbeCaseResult


def _create_val_fixture(tmp_path: Path, count: int = 30) -> Path:
    val_file = tmp_path / "sft_val.json"
    data = {}
    for i in range(1, count + 1):
        data[str(i)] = {
            "question": f"Cau hoi so {i} ve phap luat?",
            "answer": f"Can cu dieu {i} quy dinh tra loi so {i}.",
        }
    val_file.write_text(json.dumps(data, indent=2), encoding="utf-8")
    return val_file


def test_val_probe_deterministic_selection(tmp_path: Path) -> None:
    val_path = _create_val_fixture(tmp_path, count=30)
    out_dir = tmp_path / "probe_out"

    questions1, manifest1 = create_deterministic_val_probe(
        sft_val_path=val_path,
        output_directory=out_dir,
        probe_count=20,
    )

    assert len(questions1) == 20
    assert manifest1.question_count == 20
    assert len(manifest1.selected_question_ids) == 20
    assert (out_dir / VAL_PROBE_FILENAME).exists()
    assert (out_dir / VAL_PROBE_MANIFEST_FILENAME).exists()

    # Re-running on reversed file content produces identical question IDs in identical order
    reversed_val_path = tmp_path / "sft_val_reversed.json"
    data = json.loads(val_path.read_text(encoding="utf-8"))
    rev_data = {k: data[k] for k in reversed(list(data.keys()))}
    # Name must still contain sft_val to pass isolation check
    reversed_val_path = tmp_path / "custom_sft_val_rev.json"
    reversed_val_path.write_text(json.dumps(rev_data, indent=2), encoding="utf-8")

    out_dir2 = tmp_path / "probe_out2"
    questions2, manifest2 = create_deterministic_val_probe(
        sft_val_path=reversed_val_path,
        output_directory=out_dir2,
        probe_count=20,
    )

    ids1 = [q.question_id for q in questions1]
    ids2 = [q.question_id for q in questions2]
    assert ids1 == ids2
    assert manifest1.selected_question_ids == manifest2.selected_question_ids


def test_val_probe_strict_holdout_isolation(tmp_path: Path) -> None:
    screen_path = tmp_path / "screen_holdout.json"
    screen_path.write_text('{"1": {"question": "Q?", "answer": "A."}}', encoding="utf-8")

    with pytest.raises(DataValidationError, match="Strict holdout violation"):
        create_deterministic_val_probe(sft_val_path=screen_path, output_directory=tmp_path / "out")

    train_path = tmp_path / "sft_train.json"
    train_path.write_text('{"1": {"question": "Q?", "answer": "A."}}', encoding="utf-8")
    with pytest.raises(DataValidationError, match="Strict dataset isolation violation"):
        create_deterministic_val_probe(sft_val_path=train_path, output_directory=tmp_path / "out")


def test_val_probe_repetition_metrics() -> None:
    # 1. Distinct short text
    text_normal = "Hội đồng nhân dân cấp tỉnh có thẩm quyền ban hành nghị quyết theo quy định của pháp luật."
    rep8_normal = compute_repeat_ngram_ratio(text_normal, n=8)
    assert rep8_normal == 0.0

    # 2. Pathological repetition loop
    loop = "pháp luật Việt Nam quy định cụ thể rõ ràng " * 20
    rep8_loop = compute_repeat_ngram_ratio(loop, n=8)
    assert rep8_loop > 0.5

    # 3. Duplicate line ratio
    lines_clean = "Dòng 1\nDòng 2\nDòng 3\nDòng 4"
    assert compute_duplicate_line_ratio(lines_clean) == 0.0

    lines_dup = "Dòng 1\nDòng 1\nDòng 1\nDòng 1"
    assert compute_duplicate_line_ratio(lines_dup) == 0.75


def test_base_cache_validation_and_tampering(tmp_path: Path) -> None:
    val_path = _create_val_fixture(tmp_path, count=25)
    probe_dir = tmp_path / "probe_dir"
    questions, manifest = create_deterministic_val_probe(val_path, probe_dir, probe_count=20)

    # Manually construct valid BASE cache files for testing validation
    cache_dir = tmp_path / "cache_dir"
    cache_dir.mkdir(parents=True, exist_ok=True)
    results_path = cache_dir / VAL_PROBE_BASE_RESULTS_FILENAME
    manifest_path = cache_dir / VAL_PROBE_BASE_MANIFEST_FILENAME

    cases: list[ValProbeCaseResult] = []
    lines: list[str] = []
    from datetime import UTC, datetime
    from hashlib import sha256

    for q in questions:
        case = ValProbeCaseResult(
            question_id=q.question_id,
            question=q.question,
            generated_answer=f"BASE answer for {q.question_id}",
            generated_token_count=50,
            reached_cap=False,
            eos_emitted=True,
            cap_without_eos=False,
            repeat_8gram_ratio=0.0,
            duplicate_line_ratio=0.0,
            status="success",
            latency_ms=120.0,
            created_at=datetime.now(UTC),
        )
        cases.append(case)
        lines.append(case.model_dump_json())

    content_bytes = ("\n".join(lines) + "\n").encode("utf-8")
    results_path.write_bytes(content_bytes)
    results_sha = sha256(content_bytes).hexdigest()

    base_manifest_payload = {
        "schema_version": "1.0",
        "created_at": datetime.now(UTC).isoformat(),
        "code_version": "0.50.3",
        "val_probe_sha256": manifest.probe_sha256,
        "base_model_id": "Qwen/Qwen2.5-3B-Instruct",
        "base_model_revision": "a1d308dfcc03e09da285d49d912439a655a571e8",
        "tokenizer_revision": "a1d308dfcc03e09da285d49d912439a655a571e8",
        "system_prompt": "Bạn là một trợ lý AI hữu ích và chuyên gia pháp luật Việt Nam.",
        "generation_config": {
            "do_sample": False,
            "max_new_tokens": 512,
            "pad_token_id": 151643,
            "eos_token_id": 151643,
        },
        "results_sha256": results_sha,
        "record_count": 20,
        "unique_question_id_count": 20,
        "summary_health": {},
        "warnings": [],
    }
    manifest_path.write_text(json.dumps(base_manifest_payload, indent=2), encoding="utf-8")

    # 1. Valid load succeeds
    loaded_cases, loaded_manifest = load_and_validate_val_probe_base_cache(
        results_path=results_path,
        manifest_path=manifest_path,
        expected_val_probe_sha256=manifest.probe_sha256,
    )
    assert len(loaded_cases) == 20
    assert loaded_manifest.results_sha256 == results_sha

    # 2. Corrupted results content fails closed
    results_path.write_bytes(content_bytes + b"tampered")
    with pytest.raises(ArtifactCompatibilityError, match="BASE results SHA corrupted"):
        load_and_validate_val_probe_base_cache(
            results_path=results_path,
            manifest_path=manifest_path,
            expected_val_probe_sha256=manifest.probe_sha256,
        )

    # 3. Wrong base revision fails closed
    results_path.write_bytes(content_bytes)  # restore valid results
    with pytest.raises(ArtifactCompatibilityError, match="BASE revision mismatch"):
        load_and_validate_val_probe_base_cache(
            results_path=results_path,
            manifest_path=manifest_path,
            expected_val_probe_sha256=manifest.probe_sha256,
            expected_base_revision="wrong_revision_sha",
        )

    # 4. Wrong generation max_new_tokens fails closed
    with pytest.raises(ArtifactCompatibilityError, match="max_new_tokens mismatch"):
        load_and_validate_val_probe_base_cache(
            results_path=results_path,
            manifest_path=manifest_path,
            expected_val_probe_sha256=manifest.probe_sha256,
            expected_max_new_tokens=256,
        )

    # 5. Duplicate question IDs in results fail closed
    dup_cases = [cases[0], cases[0]]  # duplicate ID "1"
    dup_bytes = ("\n".join(c.model_dump_json() for c in dup_cases) + "\n").encode("utf-8")
    dup_res_path = tmp_path / "dup_results.jsonl"
    dup_man_path = tmp_path / "dup_manifest.json"
    dup_res_path.write_bytes(dup_bytes)
    dup_payload = dict(base_manifest_payload)
    dup_payload["results_sha256"] = sha256(dup_bytes).hexdigest()
    dup_payload["record_count"] = 2
    dup_man_path.write_text(json.dumps(dup_payload), encoding="utf-8")
    with pytest.raises(ArtifactCompatibilityError, match="Duplicate question ID"):
        load_and_validate_val_probe_base_cache(
            results_path=dup_res_path,
            manifest_path=dup_man_path,
            expected_val_probe_sha256=manifest.probe_sha256,
            expected_record_count=2,
        )



def test_val_probe_content_level_holdout_rejection(tmp_path: Path) -> None:
    from legal_agentic_rag.fine_tuning.val_probe import (
        CANONICAL_M50_SCREEN_HOLDOUT_SHA256,
        CANONICAL_M50_SFT_TRAIN_SHA256,
        CANONICAL_M50_SFT_VAL_SHA256,
    )

    # 1. SCREEN bytes saved under file named sft_val.json must fail closed
    fake_val_path = tmp_path / "sft_val.json"
    # Write a file whose SHA matches CANONICAL_M50_SCREEN_HOLDOUT_SHA256
    # (or mock _file_sha256 behavior / check content hash)
    import unittest.mock as mock

    with mock.patch("legal_agentic_rag.fine_tuning.val_probe._file_sha256", return_value=CANONICAL_M50_SCREEN_HOLDOUT_SHA256):
        with pytest.raises(DataValidationError, match="Strict holdout violation: file contents match canonical screen_holdout"):
            create_deterministic_val_probe(
                sft_val_path=fake_val_path,
                output_directory=tmp_path / "out",
                enforce_canonical_split=True,
            )

    # 2. TRAIN bytes saved under file named sft_val.json must fail closed
    with mock.patch("legal_agentic_rag.fine_tuning.val_probe._file_sha256", return_value=CANONICAL_M50_SFT_TRAIN_SHA256):
        with pytest.raises(DataValidationError, match="Strict dataset isolation violation: file contents match canonical sft_train"):
            create_deterministic_val_probe(
                sft_val_path=fake_val_path,
                output_directory=tmp_path / "out",
                enforce_canonical_split=True,
            )

    # 3. Arbitrary JSON fails when expected_sft_val_sha256 is required
    val_path = _create_val_fixture(tmp_path, count=25)
    with pytest.raises(DataValidationError, match="Validation partition SHA256 mismatch"):
        create_deterministic_val_probe(
            sft_val_path=val_path,
            output_directory=tmp_path / "out",
            expected_sft_val_sha256=CANONICAL_M50_SFT_VAL_SHA256,
        )

    # 4. Matching expected SHA succeeds
    from legal_agentic_rag.competition.uit_dsc_2026.development_split import _file_sha256

    actual_sha = _file_sha256(val_path)
    q_res, man_res = create_deterministic_val_probe(
        sft_val_path=val_path,
        output_directory=tmp_path / "out",
        expected_sft_val_sha256=actual_sha,
        probe_count=20,
    )
    assert len(q_res) == 20
    assert man_res.source_val_sha256 == actual_sha


def test_create_m50_c2_canonical_val_probe_mandatory_checks(tmp_path: Path) -> None:
    import unittest.mock as mock
    from legal_agentic_rag.fine_tuning.val_probe import (
        CANONICAL_M50_SCREEN_HOLDOUT_SHA256,
        CANONICAL_M50_SFT_TRAIN_SHA256,
        CANONICAL_M50_SFT_VAL_SHA256,
        CANONICAL_M50_SPLIT_MANIFEST_SHA256,
        create_m50_c2_canonical_val_probe,
    )
    from legal_agentic_rag.schemas import M50SplitManifest

    val_path = tmp_path / "sft_val.json"
    val_data = {str(i): {"question": f"Q{i}?", "answer": f"A{i}."} for i in range(1, 30)}
    val_path.write_text(json.dumps(val_data, indent=2), encoding="utf-8")

    out_dir = tmp_path / "probe_canonical_out"

    # 1. Arbitrary 500-record JSON renamed as sft_val.json fails closed
    with pytest.raises(DataValidationError, match="Validation partition SHA256 mismatch"):
        create_m50_c2_canonical_val_probe(
            sft_val_path=val_path,
            output_directory=out_dir,
        )

    # 2. SCREEN bytes renamed as sft_val.json fails closed
    with mock.patch("legal_agentic_rag.fine_tuning.val_probe._file_sha256", return_value=CANONICAL_M50_SCREEN_HOLDOUT_SHA256):
        with pytest.raises(DataValidationError, match="Strict holdout violation: file contents match canonical screen_holdout"):
            create_m50_c2_canonical_val_probe(
                sft_val_path=val_path,
                output_directory=out_dir,
            )

    # 3. TRAIN bytes renamed as sft_val.json fails closed
    with mock.patch("legal_agentic_rag.fine_tuning.val_probe._file_sha256", return_value=CANONICAL_M50_SFT_TRAIN_SHA256):
        with pytest.raises(DataValidationError, match="Strict dataset isolation violation: file contents match canonical sft_train"):
            create_m50_c2_canonical_val_probe(
                sft_val_path=val_path,
                output_directory=out_dir,
            )

    # 4. Correct canonical VAL passes with no manifest
    with mock.patch("legal_agentic_rag.fine_tuning.val_probe._file_sha256", return_value=CANONICAL_M50_SFT_VAL_SHA256):
        q_pass, m_pass = create_m50_c2_canonical_val_probe(
            sft_val_path=val_path,
            output_directory=out_dir,
            probe_count=20,
        )
        assert len(q_pass) == 20
        assert m_pass.source_val_sha256 == CANONICAL_M50_SFT_VAL_SHA256

    # 5. Correct canonical VAL with wrong split manifest partition fails closed
    wrong_manifest_p = tmp_path / "wrong_manifest.json"
    wrong_manifest_payload = {
        "schema_version": "1.0",
        "created_at": "2026-08-18T00:00:00Z",
        "code_version": "0.50.3",
        "clean_training_source": {
            "filename": "train.json",
            "question_count": 5617,
            "sha256": "0834091ea06dce76d45b693b679b92002c6cf17f82fc8e23f6d413d5155a38c3",
        },
        "seed": 2026,
        "near_duplicate_threshold": 0.85,
        "exact_duplicate_pair_count": 0,
        "near_duplicate_pair_count": 0,
        "val_target": 500,
        "screen_target": 617,
        "partitions": [
            {
                "filename": "sft_train.json",
                "question_count": 4500,
                "sha256": CANONICAL_M50_SFT_TRAIN_SHA256,
                "question_ids": [f"t_{i}" for i in range(4500)],
            },
            {
                "filename": "sft_val.json",
                "question_count": 500,
                "sha256": "1111111111111111111111111111111111111111111111111111111111111111",
                "question_ids": [f"v_{i}" for i in range(500)],
            },
            {
                "filename": "screen_holdout.json",
                "question_count": 617,
                "sha256": CANONICAL_M50_SCREEN_HOLDOUT_SHA256,
                "question_ids": [f"s_{i}" for i in range(617)],
            },
        ],
        "warnings": [],
    }
    wrong_manifest_p.write_text(json.dumps(wrong_manifest_payload), encoding="utf-8")

    with mock.patch("legal_agentic_rag.fine_tuning.val_probe._file_sha256", return_value=CANONICAL_M50_SFT_VAL_SHA256):
        with pytest.raises(DataValidationError, match="Validation partition SHA mismatch against split manifest"):
            create_m50_c2_canonical_val_probe(
                sft_val_path=val_path,
                output_directory=out_dir,
                split_manifest_path=wrong_manifest_p,
            )

    # 6. Correct canonical VAL with correct canonical manifest passes
    correct_manifest_p = tmp_path / "correct_manifest.json"
    correct_manifest_payload = {
        "schema_version": "1.0",
        "created_at": "2026-08-18T00:00:00Z",
        "code_version": "0.50.3",
        "clean_training_source": {
            "filename": "train.json",
            "question_count": 5617,
            "sha256": "0834091ea06dce76d45b693b679b92002c6cf17f82fc8e23f6d413d5155a38c3",
        },
        "seed": 2026,
        "near_duplicate_threshold": 0.85,
        "exact_duplicate_pair_count": 0,
        "near_duplicate_pair_count": 0,
        "val_target": 500,
        "screen_target": 617,
        "partitions": [
            {
                "filename": "sft_train.json",
                "question_count": 4500,
                "sha256": CANONICAL_M50_SFT_TRAIN_SHA256,
                "question_ids": [f"t_{i}" for i in range(4500)],
            },
            {
                "filename": "sft_val.json",
                "question_count": 500,
                "sha256": CANONICAL_M50_SFT_VAL_SHA256,
                "question_ids": [f"v_{i}" for i in range(500)],
            },
            {
                "filename": "screen_holdout.json",
                "question_count": 617,
                "sha256": CANONICAL_M50_SCREEN_HOLDOUT_SHA256,
                "question_ids": [f"s_{i}" for i in range(617)],
            },
        ],
        "warnings": [],
    }
    correct_manifest_p.write_text(json.dumps(correct_manifest_payload), encoding="utf-8")

    with mock.patch("legal_agentic_rag.fine_tuning.val_probe._file_sha256", return_value=CANONICAL_M50_SFT_VAL_SHA256):
        q_pass_m, m_pass_m = create_m50_c2_canonical_val_probe(
            sft_val_path=val_path,
            output_directory=out_dir,
            split_manifest_path=correct_manifest_p,
            probe_count=20,
        )
        assert len(q_pass_m) == 20
        assert m_pass_m.source_val_sha256 == CANONICAL_M50_SFT_VAL_SHA256
