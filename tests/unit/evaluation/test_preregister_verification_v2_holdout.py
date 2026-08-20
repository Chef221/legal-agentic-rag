"""Unit tests for Fresh V2 Holdout Pre-Registration and Sealed Materialization."""

from __future__ import annotations

from collections import Counter
from collections.abc import Sequence
from datetime import UTC, datetime
from hashlib import sha256
import io
import json
from pathlib import Path
import sys
import zipfile

import pytest

from legal_agentic_rag.exceptions import ArtifactCompatibilityError, DataValidationError
from legal_agentic_rag.generation.citation_verifier import RuleBasedCitationVerifier
from legal_agentic_rag.schemas.answering import (
    AnswerResponse,
    Citation,
    CitationVerificationResult,
    ClaimVerification,
    Evidence,
)
from legal_agentic_rag.schemas.manifests import ArtifactManifest, ArtifactType
from scripts.preregister_verification_v2_holdout import (
    CANONICAL_CONTAMINATION_EXCLUSION_SET,
    CANONICAL_DEVELOPMENT_SHA256,
    CANONICAL_PHASE_A_RESULTS_SHA256,
    CANONICAL_PHASE_A_ZIP_SHA256,
    CANONICAL_SELECTION_SALT,
    CANONICAL_SERVING_DATASET_NAME,
    CANONICAL_SERVING_DATASET_REVISION,
    CANONICAL_SERVING_RECORD_COUNT,
    HISTORICAL_B1A_RELATIONSHIP_QIDS,
    POSITIVE_CONTROL_PRIMARY_QIDS,
    POSITIVE_CONTROL_RESERVE_QIDS,
    SUSPICIOUS_FORENSIC_QIDS,
    HoldoutStratum,
    SelectedHoldoutCandidate,
    V2HoldoutPreRegistrar,
    has_negation_tokens,
    has_numeric_tokens,
    sha256_file,
    sha256_text,
)


def _compute_sha256(data: str | bytes) -> str:
    if isinstance(data, str):
        return sha256(data.encode("utf-8")).hexdigest()
    return sha256(data).hexdigest()


def _make_mock_chunk(chunk_id: str, doc_id: str, text: str = "Nội dung điều luật") -> dict:
    return {
        "chunk_id": chunk_id,
        "document_id": doc_id,
        "text": text,
        "document_title": f"Luật {doc_id}",
        "document_number": f"01/2020/{doc_id}",
        "document_type": "Luật",
        "structure": {
            "article_number": "1",
            "article_title": "Phạm vi điều chỉnh",
        },
        "metadata": {
            "document_title": f"Luật {doc_id}",
            "document_number": f"01/2020/{doc_id}",
            "document_type": "Luật",
        },
    }


def _make_mock_phase_a_record(
    qid: str,
    *,
    stop_reason: str = "answer_verified",
    is_valid: bool = True,
    claim_count: int = 1,
    claim_type: str = "clean_single",  # "clean_single", "clean_multi", "numeric", "negation"
    chunk_id: str = "chunk_001",
    doc_id: str = "doc_001",
) -> dict:
    if claim_type == "negation":
        claims = [
            {
                "claim_id": "C1",
                "claim_text": "Không áp dụng đối với người dưới 18 tuổi .",
                "evidence_ids": ["E1"],
                "status": "supported",
                "numeric_match": True,
                "negation_match": True,
                "lexical_support_score": 1.0,
                "errors": [],
            }
        ]
        answer_text = "Không áp dụng đối với người dưới 18 tuổi [E1]."
    elif claim_type == "numeric":
        claims = [
            {
                "claim_id": "C1",
                "claim_text": "Thời hạn giải quyết là 15 ngày làm việc .",
                "evidence_ids": ["E1"],
                "status": "supported",
                "numeric_match": True,
                "negation_match": True,
                "lexical_support_score": 1.0,
                "errors": [],
            }
        ]
        answer_text = "Thời hạn giải quyết là 15 ngày làm việc [E1]."
    elif claim_type == "clean_multi":
        claims = [
            {
                "claim_id": "C1",
                "claim_text": "Hồ sơ bao gồm đơn đề nghị .",
                "evidence_ids": ["E1"],
                "status": "supported",
                "numeric_match": True,
                "negation_match": True,
                "lexical_support_score": 1.0,
                "errors": [],
            },
            {
                "claim_id": "C2",
                "claim_text": "Cơ quan tiếp nhận là Ủy ban nhân dân cấp xã .",
                "evidence_ids": ["E1"],
                "status": "supported",
                "numeric_match": True,
                "negation_match": True,
                "lexical_support_score": 1.0,
                "errors": [],
            },
        ]
        answer_text = "Hồ sơ bao gồm đơn đề nghị [E1]. Cơ quan tiếp nhận là Ủy ban nhân dân cấp xã [E1]."
    else:  # clean_single
        claims = [
            {
                "claim_id": "C1",
                "claim_text": "Hồ sơ bao gồm đơn đề nghị theo mẫu .",
                "evidence_ids": ["E1"],
                "status": "supported",
                "numeric_match": True,
                "negation_match": True,
                "lexical_support_score": 1.0,
                "errors": [],
            }
        ]
        answer_text = "Hồ sơ bao gồm đơn đề nghị theo mẫu [E1]."

    return {
        "question_id": qid,
        "response": {
            "question": f"Câu hỏi {qid}?",
            "answer": answer_text,
            "insufficient_evidence": False,
            "retrieval_strategy": "hybrid_rerank",
            "citations": [
                {
                    "evidence_id": "E1",
                    "chunk_id": chunk_id,
                    "document_id": doc_id,
                    "document_title": f"Luật {doc_id}",
                    "document_number": f"01/2020/{doc_id}",
                    "article_number": "1",
                    "source_url": None,
                }
            ],
            "warnings": [],
            "trace_id": f"trace_{qid}",
            "metadata": {
                "agent": {
                    "stop_reason": stop_reason,
                    "attempts": 1,
                },
                "selected_evidence": [
                    {
                        "evidence_id": "E1",
                        "chunk_id": chunk_id,
                    }
                ],
                "context": {
                    "selection_trace": [
                        {
                            "chunk_id": chunk_id,
                            "selected": True,
                            "selection_rank": 1,
                        }
                    ]
                },
                "citation_verification": {
                    "is_valid": is_valid,
                    "valid_citations": [
                        {
                            "evidence_id": "E1",
                            "chunk_id": chunk_id,
                        }
                    ] if is_valid else [],
                    "invalid_citations": [] if is_valid else [
                        {
                            "evidence_id": "E1",
                            "chunk_id": chunk_id,
                        }
                    ],
                    "claim_verifications": claims,
                    "errors": [] if is_valid else ["Uncited claim"],
                    "warnings": ["semantic_entailment_not_verified"],
                },
            },
        },
    }


def test_contamination_exclusion_set_constants():
    """Verify contamination exclusion set contains all expected categories and is deduplicated."""
    assert len(SUSPICIOUS_FORENSIC_QIDS) == 4
    assert len(POSITIVE_CONTROL_PRIMARY_QIDS) == 16
    assert len(POSITIVE_CONTROL_RESERVE_QIDS) == 8
    assert len(HISTORICAL_B1A_RELATIONSHIP_QIDS) == 22

    # Suspicious forensic QIDs are a subset of B1A relationship QIDs
    assert SUSPICIOUS_FORENSIC_QIDS.issubset(HISTORICAL_B1A_RELATIONSHIP_QIDS)

    # Union size: 22 + 16 + 8 = 46
    assert len(CANONICAL_CONTAMINATION_EXCLUSION_SET) == 46

    # Verify no overlap between positive-control primaries and reserves
    assert POSITIVE_CONTROL_PRIMARY_QIDS.isdisjoint(POSITIVE_CONTROL_RESERVE_QIDS)
    assert POSITIVE_CONTROL_PRIMARY_QIDS.isdisjoint(HISTORICAL_B1A_RELATIONSHIP_QIDS)
    assert POSITIVE_CONTROL_RESERVE_QIDS.isdisjoint(HISTORICAL_B1A_RELATIONSHIP_QIDS)


def test_selection_salt_exact():
    """Verify pre-registered selection salt is exact."""
    assert CANONICAL_SELECTION_SALT == "verification-v2-holdout-gen-v1:"


def test_token_helpers():
    """Verify numeric and negation token detectors."""
    assert has_numeric_tokens("thời hạn 15 ngày")
    assert has_numeric_tokens("mức phạt 5.000.000 đồng")
    assert has_numeric_tokens("tỷ lệ 100%")
    assert not has_numeric_tokens("không có số")

    assert has_negation_tokens("Không được phép")
    assert has_negation_tokens("Ngoại trừ trường hợp")
    assert has_negation_tokens("Chưa hoàn thành")
    assert has_negation_tokens("Bị bãi bỏ")
    assert has_negation_tokens("Nghiêm cấm hành vi")
    assert not has_negation_tokens("Được phép thực hiện đầy đủ")


def test_stratum_precedence_and_stratification():
    """Verify stratum assignment follows strict precedence D -> C -> B -> A."""
    registrar = V2HoldoutPreRegistrar(
        phase_a_evidence_path=Path("dummy_phase_a.zip"),
        serving_root=Path("dummy_serving"),
        development_path=Path("dummy_dev.json"),
        output_dir=Path("dummy_out"),
    )

    # 1. Negation (Stratum D) takes top precedence even if numeric
    rec_d = _make_mock_phase_a_record("q_d", claim_type="negation")
    rec_d["response"]["metadata"]["citation_verification"]["claim_verifications"][0]["claim_text"] = "Không được quá 30 ngày."
    rec_d["response"]["metadata"]["citation_verification"]["claim_verifications"][0]["numeric_match"] = True
    rec_d["response"]["metadata"]["citation_verification"]["claim_verifications"][0]["negation_match"] = True

    # 2. Numeric (Stratum C)
    rec_c = _make_mock_phase_a_record("q_c", claim_type="numeric")

    # 3. Multi-claim clean (Stratum B)
    rec_b = _make_mock_phase_a_record("q_b", claim_type="clean_multi")

    # 4. Single-claim clean (Stratum A)
    rec_a = _make_mock_phase_a_record("q_a", claim_type="clean_single")

    strata = registrar._stratify_records([rec_d, rec_c, rec_b, rec_a])
    assert len(strata[HoldoutStratum.D_NEGATION_MODALITY]) == 1
    assert strata[HoldoutStratum.D_NEGATION_MODALITY][0]["question_id"] == "q_d"

    assert len(strata[HoldoutStratum.C_NUMERIC]) == 1
    assert strata[HoldoutStratum.C_NUMERIC][0]["question_id"] == "q_c"

    assert len(strata[HoldoutStratum.B_MULTI_CLAIM_CLEAN]) == 1
    assert strata[HoldoutStratum.B_MULTI_CLAIM_CLEAN][0]["question_id"] == "q_b"

    assert len(strata[HoldoutStratum.A_SINGLE_CLAIM_CLEAN]) == 1
    assert strata[HoldoutStratum.A_SINGLE_CLAIM_CLEAN][0]["question_id"] == "q_a"


def test_stratified_sampling_quotas_and_disjoint():
    """Verify deterministic sampling selects exactly 16 primary + 8 reserve candidates (4/4/4/4 and 2/2/2/2)."""
    registrar = V2HoldoutPreRegistrar(
        phase_a_evidence_path=Path("dummy_phase_a.zip"),
        serving_root=Path("dummy_serving"),
        development_path=Path("dummy_dev.json"),
        output_dir=Path("dummy_out"),
    )

    # Generate 10 records per stratum (40 total)
    records_by_stratum = {}
    for stratum, ctype in [
        (HoldoutStratum.D_NEGATION_MODALITY, "negation"),
        (HoldoutStratum.C_NUMERIC, "numeric"),
        (HoldoutStratum.B_MULTI_CLAIM_CLEAN, "clean_multi"),
        (HoldoutStratum.A_SINGLE_CLAIM_CLEAN, "clean_single"),
    ]:
        records_by_stratum[stratum] = [
            _make_mock_phase_a_record(f"qid_{stratum.value}_{i:02d}", claim_type=ctype)
            for i in range(10)
        ]

    primary, reserve = registrar._sample_strata(records_by_stratum)

    assert len(primary) == 16
    assert len(reserve) == 8

    primary_qids = {c.question_id for c in primary}
    reserve_qids = {c.question_id for c in reserve}

    # Primary and reserve sets must be strictly disjoint
    assert primary_qids.isdisjoint(reserve_qids)

    # Check stratum distribution
    primary_counts = Counter(c.stratum for c in primary)
    reserve_counts = Counter(c.stratum for c in reserve)

    for s in HoldoutStratum:
        assert primary_counts[s.value] == 4
        assert reserve_counts[s.value] == 2


def test_insufficient_stratum_records_raises_error():
    """Verify error raised if a stratum has fewer than 6 records (4 primary + 2 reserve)."""
    registrar = V2HoldoutPreRegistrar(
        phase_a_evidence_path=Path("dummy_phase_a.zip"),
        serving_root=Path("dummy_serving"),
        development_path=Path("dummy_dev.json"),
        output_dir=Path("dummy_out"),
    )

    records_by_stratum = {
        HoldoutStratum.D_NEGATION_MODALITY: [_make_mock_phase_a_record(f"q_d_{i}", claim_type="negation") for i in range(5)],  # < 6
        HoldoutStratum.C_NUMERIC: [_make_mock_phase_a_record(f"q_c_{i}", claim_type="numeric") for i in range(10)],
        HoldoutStratum.B_MULTI_CLAIM_CLEAN: [_make_mock_phase_a_record(f"q_b_{i}", claim_type="clean_multi") for i in range(10)],
        HoldoutStratum.A_SINGLE_CLAIM_CLEAN: [_make_mock_phase_a_record(f"q_a_{i}", claim_type="clean_single") for i in range(10)],
    }

    with pytest.raises(DataValidationError, match="insufficient for required 6"):
        registrar._sample_strata(records_by_stratum)


def test_contamination_exclusion_filters_out_records():
    """Verify all excluded QIDs are filtered out from eligible pool."""
    excluded = ["102047", "75171", "27503"]
    registrar = V2HoldoutPreRegistrar(
        phase_a_evidence_path=Path("dummy_phase_a.zip"),
        serving_root=Path("dummy_serving"),
        development_path=Path("dummy_dev.json"),
        output_dir=Path("dummy_out"),
        excluded_qids=excluded,
    )

    records_map = {
        "102047": _make_mock_phase_a_record("102047"),
        "75171": _make_mock_phase_a_record("75171"),
        "27503": _make_mock_phase_a_record("27503"),
        "99999": _make_mock_phase_a_record("99999"),
    }

    eligible = registrar._filter_eligible_pool(records_map)
    eligible_qids = {r["question_id"] for r in eligible}

    assert "99999" in eligible_qids
    assert not (set(excluded) & eligible_qids)


def test_development_json_validation(tmp_path: Path):
    """Verify development.json validation requires exact SHA and 991 records."""
    dev_path = tmp_path / "development.json"
    dev_data = {str(i): {"question": f"Q{i}", "answer": f"A{i}"} for i in range(991)}
    dev_path.write_text(json.dumps(dev_data), encoding="utf-8")

    registrar = V2HoldoutPreRegistrar(
        phase_a_evidence_path=Path("dummy_phase_a.zip"),
        serving_root=Path("dummy_serving"),
        development_path=dev_path,
        output_dir=tmp_path / "out",
    )

    # Fails because hash does not match canonical
    with pytest.raises(DataValidationError, match="development.json SHA mismatch"):
        registrar._load_and_validate_development(dev_path)


def test_public_commitment_contains_zero_qids(tmp_path: Path):
    """Verify public commitment JSON report does not leak any question IDs."""
    registrar = V2HoldoutPreRegistrar(
        phase_a_evidence_path=Path("dummy_phase_a.zip"),
        serving_root=Path("dummy_serving"),
        development_path=Path("dummy_dev.json"),
        output_dir=tmp_path / "out",
    )

    mock_manifest = ArtifactManifest(
        schema_version="1.0",
        artifact_type=ArtifactType.LEGAL_CHUNKS,
        artifact_version="1.0",
        dataset_name=CANONICAL_SERVING_DATASET_NAME,
        dataset_revision=CANONICAL_SERVING_DATASET_REVISION,
        record_count=CANONICAL_SERVING_RECORD_COUNT,
        created_at=datetime.now(UTC),
        processing_config_hash="dummy_hash",
        code_version="0.50.7",
        metadata={"payload_sha256": "dummy_payload_sha"},
    )

    primary_candidates = [
        SelectedHoldoutCandidate(f"q_p_{i}", "A_SINGLE_CLAIM_CLEAN", f"key_p_{i}", "primary", 1, "answer_verified")
        for i in range(16)
    ]
    reserve_candidates = [
        SelectedHoldoutCandidate(f"q_r_{i}", "A_SINGLE_CLAIM_CLEAN", f"key_r_{i}", "reserve", 1, "answer_verified")
        for i in range(8)
    ]
    primary_summaries = [
        {
            "question_id": f"q_p_{i}",
            "selected_chunk_lookup_pass": True,
            "source_mapping_pass": True,
            "metadata_crosscheck_pass": True,
            "rule_verifier_replay_pass": True,
        }
        for i in range(16)
    ]

    sel_artifact_path = tmp_path / "selection.json"
    sel_artifact_path.write_text("{}", encoding="utf-8")

    commitment = registrar._build_public_commitment_report(
        verdict="V2_HOLDOUT_PRE_REGISTERED",
        source_kind="canonical_zip",
        archive_filename="test.zip",
        archive_sha="test_sha",
        results_sha="results_sha",
        dev_sha="dev_sha",
        chunk_manifest=mock_manifest,
        eligible_count=760,
        stratum_assignments={s: [] for s in HoldoutStratum},
        primary_candidates=primary_candidates,
        reserve_candidates=reserve_candidates,
        primary_summaries=primary_summaries,
        selection_artifact_path=sel_artifact_path,
    )

    commitment_json = json.dumps(commitment, ensure_ascii=False)

    # Assert no candidate question IDs appear in the public commitment
    for c in primary_candidates:
        assert c.question_id not in commitment_json
    for c in reserve_candidates:
        assert c.question_id not in commitment_json

    assert commitment["verdict"] == "V2_HOLDOUT_PRE_REGISTERED"
    assert commitment["primary_validation"]["holdout_sealed"] is True
    assert commitment["primary_validation"]["human_labels_populated"] is False
    assert commitment["selection_commitment"]["primary_count"] == 16
    assert commitment["selection_commitment"]["reserve_count"] == 8


def test_deterministic_selection_hash_stability():
    """Verify that multiple stratified sampling runs on identical inputs yield 100% identical outputs."""
    registrar = V2HoldoutPreRegistrar(
        phase_a_evidence_path=Path("dummy_phase_a.zip"),
        serving_root=Path("dummy_serving"),
        development_path=Path("dummy_dev.json"),
        output_dir=Path("dummy_out"),
    )

    records_by_stratum = {}
    for stratum, ctype in [
        (HoldoutStratum.D_NEGATION_MODALITY, "negation"),
        (HoldoutStratum.C_NUMERIC, "numeric"),
        (HoldoutStratum.B_MULTI_CLAIM_CLEAN, "clean_multi"),
        (HoldoutStratum.A_SINGLE_CLAIM_CLEAN, "clean_single"),
    ]:
        records_by_stratum[stratum] = [
            _make_mock_phase_a_record(f"qid_{stratum.value}_{i:02d}", claim_type=ctype)
            for i in range(10)
        ]

    prim1, res1 = registrar._sample_strata(records_by_stratum)
    prim2, res2 = registrar._sample_strata(records_by_stratum)

    assert [c.question_id for c in prim1] == [c.question_id for c in prim2]
    assert [c.selection_key for c in prim1] == [c.selection_key for c in prim2]
    assert [c.question_id for c in res1] == [c.question_id for c in res2]
    assert [c.selection_key for c in res1] == [c.selection_key for c in res2]


def test_process_primary_case_reconstruction_and_replay():
    """Verify evidence reconstruction, verifier replay, and sealed review packet generation."""
    registrar = V2HoldoutPreRegistrar(
        phase_a_evidence_path=Path("dummy_phase_a.zip"),
        serving_root=Path("dummy_serving"),
        development_path=Path("dummy_dev.json"),
        output_dir=Path("dummy_out"),
    )

    candidate = SelectedHoldoutCandidate(
        question_id="test_qid_001",
        stratum="A_SINGLE_CLAIM_CLEAN",
        selection_key="key_001",
        pool_type="primary",
        claim_count=1,
        historical_stop_reason="answer_verified",
    )

    chunk_data = _make_mock_chunk("chunk_001", "doc_001", "Hồ sơ bao gồm đơn đề nghị theo mẫu.")
    chunks_by_id = {"chunk_001": chunk_data}

    record = _make_mock_phase_a_record("test_qid_001", chunk_id="chunk_001", doc_id="doc_001")

    mock_manifest = ArtifactManifest(
        schema_version="1.0",
        artifact_type=ArtifactType.LEGAL_CHUNKS,
        artifact_version="1.0",
        dataset_name=CANONICAL_SERVING_DATASET_NAME,
        dataset_revision=CANONICAL_SERVING_DATASET_REVISION,
        record_count=CANONICAL_SERVING_RECORD_COUNT,
        created_at=datetime.now(UTC),
        processing_config_hash="dummy_hash",
        code_version="0.50.7",
        metadata={"payload_sha256": "dummy_payload_sha"},
    )

    verifier = RuleBasedCitationVerifier()

    packet, summary = registrar._process_primary_case(
        candidate=candidate,
        record=record,
        question_text="Hồ sơ gồm những gì?",
        reference_answer="Hồ sơ gồm đơn đề nghị.",
        chunks_by_id=chunks_by_id,
        chunk_manifest=mock_manifest,
        verifier=verifier,
        dev_sha="dev_sha_123",
        archive_filename="archive.zip",
        archive_sha="archive_sha_123",
        results_sha="results_sha_123",
        source_kind="canonical_zip",
    )

    # Verification passes
    assert summary["selected_chunk_lookup_pass"] is True
    assert summary["source_mapping_pass"] is True
    assert summary["metadata_crosscheck_pass"] is True
    assert summary["rule_verifier_replay_pass"] is True

    # Sealed status assertions
    assert packet["human_forensic_review"]["review_status"] == "sealed_unreviewed"
    assert packet["human_forensic_review"]["claim_labels"] is None
    assert packet["human_forensic_review"]["reviewer_notes"] is None
    assert packet["human_forensic_review"]["root_cause_classification"] is None


def test_process_primary_case_missing_chunk_fails_lookup():
    """Verify that missing chunk in serving store causes lookup pass to fail."""
    registrar = V2HoldoutPreRegistrar(
        phase_a_evidence_path=Path("dummy_phase_a.zip"),
        serving_root=Path("dummy_serving"),
        development_path=Path("dummy_dev.json"),
        output_dir=Path("dummy_out"),
    )

    candidate = SelectedHoldoutCandidate(
        question_id="test_qid_002",
        stratum="A_SINGLE_CLAIM_CLEAN",
        selection_key="key_002",
        pool_type="primary",
        claim_count=1,
        historical_stop_reason="answer_verified",
    )

    # Chunks store is empty -> chunk_001 missing
    chunks_by_id = {}
    record = _make_mock_phase_a_record("test_qid_002", chunk_id="chunk_001")

    mock_manifest = ArtifactManifest(
        schema_version="1.0",
        artifact_type=ArtifactType.LEGAL_CHUNKS,
        artifact_version="1.0",
        dataset_name=CANONICAL_SERVING_DATASET_NAME,
        dataset_revision=CANONICAL_SERVING_DATASET_REVISION,
        record_count=CANONICAL_SERVING_RECORD_COUNT,
        created_at=datetime.now(UTC),
        processing_config_hash="dummy_hash",
        code_version="0.50.7",
        metadata={"payload_sha256": "dummy_payload_sha"},
    )

    _, summary = registrar._process_primary_case(
        candidate=candidate,
        record=record,
        question_text="Q?",
        reference_answer="A.",
        chunks_by_id=chunks_by_id,
        chunk_manifest=mock_manifest,
        verifier=RuleBasedCitationVerifier(),
        dev_sha="dev_sha",
        archive_filename="archive.zip",
        archive_sha="archive_sha",
        results_sha="results_sha",
        source_kind="canonical_zip",
    )

    assert summary["selected_chunk_lookup_pass"] is False


def test_process_primary_case_trace_mismatch_fails_mapping():
    """Verify that context selection trace mismatch causes source mapping pass to fail."""
    registrar = V2HoldoutPreRegistrar(
        phase_a_evidence_path=Path("dummy_phase_a.zip"),
        serving_root=Path("dummy_serving"),
        development_path=Path("dummy_dev.json"),
        output_dir=Path("dummy_out"),
    )

    candidate = SelectedHoldoutCandidate(
        question_id="test_qid_003",
        stratum="A_SINGLE_CLAIM_CLEAN",
        selection_key="key_003",
        pool_type="primary",
        claim_count=1,
        historical_stop_reason="answer_verified",
    )

    chunk_data = _make_mock_chunk("chunk_001", "doc_001")
    chunks_by_id = {"chunk_001": chunk_data}

    record = _make_mock_phase_a_record("test_qid_003", chunk_id="chunk_001")
    # Mutate selection trace to point to different chunk
    record["response"]["metadata"]["context"]["selection_trace"][0]["chunk_id"] = "chunk_DIFFERENT"

    mock_manifest = ArtifactManifest(
        schema_version="1.0",
        artifact_type=ArtifactType.LEGAL_CHUNKS,
        artifact_version="1.0",
        dataset_name=CANONICAL_SERVING_DATASET_NAME,
        dataset_revision=CANONICAL_SERVING_DATASET_REVISION,
        record_count=CANONICAL_SERVING_RECORD_COUNT,
        created_at=datetime.now(UTC),
        processing_config_hash="dummy_hash",
        code_version="0.50.7",
        metadata={"payload_sha256": "dummy_payload_sha"},
    )

    _, summary = registrar._process_primary_case(
        candidate=candidate,
        record=record,
        question_text="Q?",
        reference_answer="A.",
        chunks_by_id=chunks_by_id,
        chunk_manifest=mock_manifest,
        verifier=RuleBasedCitationVerifier(),
        dev_sha="dev_sha",
        archive_filename="archive.zip",
        archive_sha="archive_sha",
        results_sha="results_sha",
        source_kind="canonical_zip",
    )

    assert summary["source_mapping_pass"] is False


def test_write_outputs_file_structure(tmp_path: Path):
    """Verify that _write_outputs creates all expected files in execution/, results/, and holdout_packets/."""
    out_dir = tmp_path / "v2_holdout_test_out"
    registrar = V2HoldoutPreRegistrar(
        phase_a_evidence_path=Path("dummy_phase_a.zip"),
        serving_root=Path("dummy_serving"),
        development_path=Path("dummy_dev.json"),
        output_dir=out_dir,
    )

    primary_candidates = [
        SelectedHoldoutCandidate("q1", "A_SINGLE_CLAIM_CLEAN", "k1", "primary", 1, "answer_verified")
    ]
    reserve_candidates = [
        SelectedHoldoutCandidate("q2", "A_SINGLE_CLAIM_CLEAN", "k2", "reserve", 1, "answer_verified")
    ]
    primary_packets = {
        "q1": {"schema_version": "1.0", "question_id": "q1", "human_forensic_review": {"review_status": "sealed_unreviewed"}}
    }
    full_report = {"schema_version": "1.0", "verdict": "V2_HOLDOUT_PRE_REGISTERED"}

    written = registrar._write_outputs(
        primary_packets=primary_packets,
        full_selection_report=full_report,
        primary_candidates=primary_candidates,
        reserve_candidates=reserve_candidates,
        source_kind="canonical_zip",
        archive_filename="archive.zip",
        archive_sha="sha1",
        results_sha="sha2",
        dev_sha="sha3",
    )

    assert written["selection_artifact"].is_file()
    assert (out_dir / "execution" / "holdout_source_identity.json").is_file()
    assert (out_dir / "results" / "holdout_selection_commitment.json").is_file()
    assert (out_dir / "results" / "primary_holdout_identity.json").is_file()
    assert (out_dir / "results" / "fresh_reserve_identity.json").is_file()
    assert (out_dir / "holdout_packets" / "q1.json").is_file()

