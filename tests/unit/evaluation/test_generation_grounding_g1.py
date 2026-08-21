"""Unit tests for G1 generation grounding development evaluation harness."""

from __future__ import annotations

import json
from pathlib import Path
import tempfile
import zipfile

import pytest

from legal_agentic_rag.configuration.online import GenerationConfig
from legal_agentic_rag.contracts.chat_model_provider import ChatModelProvider
from legal_agentic_rag.generation.model_generator import (
    GROUNDING_PROFILE_BASELINE,
    GROUNDING_PROFILE_MATERIAL_FIDELITY_V1,
)
from scripts.evaluate_generation_grounding_g1 import (
    DiagnosticQuestionPacket,
    GenerationGroundingG1Evaluator,
    KNOWN_MATERIAL_ERROR_QUESTIONS,
    KNOWN_POSITIVE_LIST_QUESTIONS,
    TelemetryLoggingChatProviderProxy,
    compute_deterministic_blinding,
    generate_pairwise_worksheet,
    load_diagnostic_packets,
)


class _MockChatProvider(ChatModelProvider):
    provider_name = "mock-qwen"
    provider_version = "4.47.1"
    model_name = "Qwen/Qwen2.5-3B-Instruct"
    model_revision = "a1d308dfcc03e09da285d49d912439a655a571e8"

    def __init__(self, response_by_arm: dict[str, str] | None = None) -> None:
        self.calls: list[tuple[str, str]] = []
        self.response_by_arm = response_by_arm or {}

    def complete(
        self,
        *,
        system_instruction: str,
        user_prompt: str,
    ) -> str:
        self.calls.append((system_instruction, user_prompt))
        if "MATERIAL LEGAL FIDELITY" in system_instruction:
            return self.response_by_arm.get(
                "material_fidelity_v1",
                json.dumps(
                    {
                        "claims": [
                            {
                                "text": "Người bảo vệ quyền lợi có quyền tham gia từ khi khởi kiện.",
                                "evidence_ids": ["E1"],
                            }
                        ],
                        "insufficient_evidence": False,
                        "warnings": [],
                    }
                ),
            )
        return self.response_by_arm.get(
            "baseline",
            json.dumps(
                {
                    "claims": [
                        {
                            "text": "Đương sự có quyền tham gia từ khi khởi kiện.",
                            "evidence_ids": ["E1"],
                        }
                    ],
                    "insufficient_evidence": False,
                    "warnings": [],
                }
            ),
        )


def _create_mock_packet(qid: str) -> dict[str, object]:
    return {
        "question_id": qid,
        "question": f"Câu hỏi pháp lý mẫu {qid}?",
        "retrieved_evidence": [
            {
                "evidence_id": "E1",
                "chunk_id": f"chunk-{qid}-1",
                "document_id": f"doc-{qid}",
                "document_title": "Luật Tố tụng mẫu",
                "document_number": "01/2026/QH",
                "article_number": "55",
                "article_title": "Quyền của đương sự",
                "effect_status": "active",
                "text": "Người bảo vệ quyền và lợi ích hợp pháp của đương sự có quyền tham gia tố tụng từ khi khởi kiện.",
                "source_url": f"https://example.test/{qid}",
            }
        ],
        "reference_answer_context": {
            "text": "Người bảo vệ quyền lợi tham gia từ khi khởi kiện.",
            "ground_truth_status": "context_only",
        },
    }


def test_compute_deterministic_blinding_is_stable_and_balanced() -> None:
    """Blinding key generation must be deterministic and assign profiles to options."""
    qids = [f"Q{i}" for i in range(16)]
    key1 = compute_deterministic_blinding(qids)
    key2 = compute_deterministic_blinding(qids)
    assert key1 == key2

    opt1_profiles = [v["option_1"] for v in key1.values()]
    assert GROUNDING_PROFILE_BASELINE in opt1_profiles
    assert GROUNDING_PROFILE_MATERIAL_FIDELITY_V1 in opt1_profiles


def test_load_diagnostic_packets_from_zip(tmp_path: Path) -> None:
    """Packets loader must extract and parse 16 packet JSONs correctly."""
    zip_path = tmp_path / "test_packets.zip"
    with zipfile.ZipFile(zip_path, "w") as zf:
        for i in range(16):
            qid = str(100000 + i)
            data = _create_mock_packet(qid)
            zf.writestr(f"holdout_packets/{qid}.json", json.dumps(data))

    packets = load_diagnostic_packets(zip_path)
    assert len(packets) == 16
    assert packets[0].question_id == "100000"
    assert len(packets[0].retrieved_evidence) == 1
    assert packets[0].reference_answer is not None


def test_telemetry_logging_proxy_records_calls_safely() -> None:
    """Proxy must record call index, arm, timing, and SHA256 hashes without raw prompt."""
    raw_provider = _MockChatProvider()
    proxy = TelemetryLoggingChatProviderProxy(raw_provider)
    proxy.set_context("material_fidelity_v1", "125893")

    completion = proxy.complete(
        system_instruction="System instruction text",
        user_prompt="User prompt with question and evidence",
    )
    assert "claims" in completion
    assert len(proxy.records) == 1
    rec = proxy.records[0]
    assert rec.call_index == 1
    assert rec.arm == "material_fidelity_v1"
    assert rec.question_id == "125893"
    assert rec.status == "SUCCESS"
    assert rec.prompt_character_count > 0
    assert len(rec.prompt_sha256) == 64
    assert len(rec.completion_sha256) == 64


def test_generation_grounding_g1_evaluator_preflight_only(tmp_path: Path) -> None:
    """Preflight mode must validate source integrity and write source identity without inference."""
    zip_path = tmp_path / "test_packets.zip"
    with zipfile.ZipFile(zip_path, "w") as zf:
        for i in range(16):
            qid = str(100000 + i)
            data = _create_mock_packet(qid)
            zf.writestr(f"holdout_packets/{qid}.json", json.dumps(data))

    out_dir = tmp_path / "out_preflight"
    config = GenerationConfig(
        backend="transformers",
        model_name="Qwen/Qwen2.5-3B-Instruct",
        model_revision="a1d308dfcc03e09da285d49d912439a655a571e8",
        device="cpu",
    )

    evaluator = GenerationGroundingG1Evaluator(
        diagnostic_packets_path=zip_path,
        output_dir=out_dir,
        generation_config=config,
        preflight_only=True,
    )
    identity = evaluator.run()
    assert identity["preflight_ready"] is True
    assert identity["provider_constructor_contract_verified"] is True
    assert (out_dir / "execution" / "generation_g1_source_identity.json").exists()


def test_generation_grounding_g1_evaluator_executes_ab_experiment(tmp_path: Path) -> None:
    """Evaluator must run both Baseline and G1 arms, scoring reference metrics and generating worksheet."""
    zip_path = tmp_path / "test_packets.zip"
    with zipfile.ZipFile(zip_path, "w") as zf:
        for i in range(16):
            qid = str(100000 + i)
            data = _create_mock_packet(qid)
            zf.writestr(f"holdout_packets/{qid}.json", json.dumps(data))

    out_dir = tmp_path / "out_ab"
    config = GenerationConfig(
        backend="transformers",
        model_name="Qwen/Qwen2.5-3B-Instruct",
        model_revision="a1d308dfcc03e09da285d49d912439a655a571e8",
        device="cpu",
    )
    mock_provider = _MockChatProvider()

    evaluator = GenerationGroundingG1Evaluator(
        diagnostic_packets_path=zip_path,
        output_dir=out_dir,
        generation_config=config,
        provider=mock_provider,
        preflight_only=False,
    )
    report = evaluator.run()

    assert report["experiment_id"] == "GENERATION-G1-AB-DEVELOPMENT"
    assert report["total_questions"] == 16
    assert report["baseline_summary"]["successful_generations"] == 16
    assert report["g1_summary"]["successful_generations"] == 16
    assert report["criteria_evaluation"]["criterion_a_zero_execution_errors"]["status"] == "PASS"
    assert report["criteria_evaluation"]["criterion_d_bounded_abstention"]["status"] == "PASS"

    # Verify generated artifacts
    assert (out_dir / "results" / "generation_g1_ab_report.json").exists()
    assert (out_dir / "results" / "generation_g1_ab_predictions.jsonl").exists()
    assert (out_dir / "results" / "generation_g1_blinding_key.json").exists()
    assert (out_dir / "results" / "generation_g1_human_review_worksheet.md").exists()
    assert (out_dir / "telemetry" / "provider_calls.jsonl").exists()
    assert (out_dir / "execution" / "generation_g1_source_identity.json").exists()

    worksheet_text = (out_dir / "results" / "generation_g1_human_review_worksheet.md").read_text(encoding="utf-8")
    assert "Blinded Pairwise Review Worksheet" in worksheet_text
    assert "Option 1" in worksheet_text
    assert "Option 2" in worksheet_text
    assert "QID: 100000" in worksheet_text
