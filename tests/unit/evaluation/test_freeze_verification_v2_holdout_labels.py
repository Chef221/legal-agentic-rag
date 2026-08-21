"""Unit tests for Fresh Holdout Label Freezing Harness (Phase H-LABEL)."""

from __future__ import annotations

from hashlib import sha256
import json
from pathlib import Path
import zipfile

import pytest

from legal_agentic_rag.exceptions import DataValidationError
from scripts.freeze_verification_v2_holdout_labels import (
    CANONICAL_HOLDOUT_REVIEW_ZIP_SHA256,
    CANONICAL_HOLDOUT_SELECTION_SHA256,
    HoldoutEntailmentLabel,
    freeze_holdout_labels,
    sha256_text,
)


@pytest.fixture
def mock_holdout_packets_and_selection(tmp_path: Path) -> dict[str, Path]:
    """Create minimal synthetic holdout review packets ZIP and selection file."""
    packets_zip = tmp_path / "mock_holdout_packets.zip"
    selection_file = tmp_path / "mock_holdout_selection.json"

    packet_1 = {
        "question_id": "SYNTH_Q1",
        "stratum": "A_SINGLE_CLAIM_CLEAN",
        "question_text": "Sample synthetic question 1?",
        "arms": {
            "BASE": {
                "historical_stop_reason": "answer_verified",
                "claims": [
                    {
                        "claim_id": "C1",
                        "claim_text": "First synthetic claim text.",
                        "entailment_label": "SUPPORTED",
                    },
                    {
                        "claim_id": "C2",
                        "claim_text": "Second synthetic claim text.",
                        "entailment_label": "INSUFFICIENT",
                    },
                ],
            }
        },
    }

    packet_2 = {
        "question_id": "SYNTH_Q2",
        "stratum": "D_NEGATION_MODALITY",
        "question_text": "Sample synthetic question 2?",
        "arms": {
            "PRIMARY": {
                "historical_stop_reason": "answer_verified",
                "claims": [
                    {
                        "claim_id": "C1",
                        "claim_text": "Third synthetic claim text.",
                        "entailment_label": "CONTRADICTED",
                    }
                ],
            }
        },
    }

    with zipfile.ZipFile(packets_zip, "w") as zf:
        zf.writestr("packets/SYNTH_Q1.json", json.dumps(packet_1))
        zf.writestr("packets/SYNTH_Q2.json", json.dumps(packet_2))

    selection_file.write_text(
        json.dumps({"schema_version": "1.0", "holdout_count": 2}),
        encoding="utf-8",
    )

    return {
        "packets": packets_zip,
        "selection": selection_file,
    }


def test_freeze_holdout_labels_success(mock_holdout_packets_and_selection, tmp_path):
    """Test successful label freeze generating labels and content-free commitment."""
    packets_path = mock_holdout_packets_and_selection["packets"]
    selection_path = mock_holdout_packets_and_selection["selection"]
    out_labels = tmp_path / "out_labels.json"
    out_commitment = tmp_path / "out_commitment.json"

    reviewed_input = {
        "questions": {
            "SYNTH_Q1": {
                "arms": {
                    "BASE": {
                        "claims": {
                            "C1": {
                                "entailment_label": "SUPPORTED",
                                "claim_text_sha256": sha256_text("First synthetic claim text."),
                                "error_tags": [],
                            },
                            "C2": {
                                "entailment_label": "INSUFFICIENT",
                                "claim_text_sha256": sha256_text("Second synthetic claim text."),
                                "error_tags": ["SCOPE_OVERGENERALIZED"],
                                "diagnostic_note": "Scope too broad",
                            },
                        }
                    }
                }
            },
            "SYNTH_Q2": {
                "arms": {
                    "PRIMARY": {
                        "claims": {
                            "C1": {
                                "entailment_label": "CONTRADICTED",
                                "claim_text_sha256": sha256_text("Third synthetic claim text."),
                                "error_tags": ["NEGATION_INVERTED"],
                            }
                        }
                    }
                }
            },
        }
    }
    input_file = tmp_path / "reviewed_input.json"
    input_file.write_text(json.dumps(reviewed_input), encoding="utf-8")

    labels_data, commitment_data = freeze_holdout_labels(
        holdout_packets_path=packets_path,
        holdout_selection_path=selection_path,
        reviewed_input_path=input_file,
        output_labels_path=out_labels,
        commitment_output_path=out_commitment,
        bypass_source_checksums=True,
    )

    assert out_labels.is_file()
    assert out_commitment.is_file()

    assert labels_data["review_status"] == "frozen_human_reviewed"
    assert labels_data["total_questions"] == 2
    assert labels_data["total_arms"] == 2
    assert labels_data["total_claims"] == 3
    assert labels_data["class_counts"] == {
        "SUPPORTED": 1,
        "CONTRADICTED": 1,
        "INSUFFICIENT": 1,
    }

    # Verify commitment content safety: NO QIDs or question texts
    commitment_str = json.dumps(commitment_data)
    assert "SYNTH_Q1" not in commitment_str
    assert "SYNTH_Q2" not in commitment_str
    assert "synthetic claim text" not in commitment_str
    assert commitment_data["labels_sha256"] == sha256(out_labels.read_bytes()).hexdigest()
    assert commitment_data["labels_size_bytes"] == out_labels.stat().st_size
    assert commitment_data["reviewer_governance_status"] == "FROZEN_PENDING_EXTERNAL_REVIEW"


def test_freeze_holdout_labels_duplicate_claim_in_list_fails(mock_holdout_packets_and_selection, tmp_path):
    """Test that duplicate claims in list-format review fail closed."""
    packets_path = mock_holdout_packets_and_selection["packets"]
    selection_path = mock_holdout_packets_and_selection["selection"]
    out_labels = tmp_path / "out_labels.json"

    reviewed_input = [
        {"question_id": "SYNTH_Q1", "arm_id": "BASE", "claim_id": "C1", "entailment_label": "SUPPORTED"},
        {"question_id": "SYNTH_Q1", "arm_id": "BASE", "claim_id": "C2", "entailment_label": "INSUFFICIENT"},
        {"question_id": "SYNTH_Q2", "arm_id": "PRIMARY", "claim_id": "C1", "entailment_label": "CONTRADICTED"},
        # Duplicate review item
        {"question_id": "SYNTH_Q1", "arm_id": "BASE", "claim_id": "C1", "entailment_label": "CONTRADICTED"},
    ]
    input_file = tmp_path / "duplicate_list_input.json"
    input_file.write_text(json.dumps(reviewed_input), encoding="utf-8")

    with pytest.raises(DataValidationError, match="HOLD_OUT_LABEL_DUPLICATE"):
        freeze_holdout_labels(
            holdout_packets_path=packets_path,
            holdout_selection_path=selection_path,
            reviewed_input_path=input_file,
            output_labels_path=out_labels,
            bypass_source_checksums=True,
        )


def test_freeze_holdout_labels_duplicate_claim_in_dict_list_fails(mock_holdout_packets_and_selection, tmp_path):
    """Test that duplicate claims in arm-level claims list fail closed."""
    packets_path = mock_holdout_packets_and_selection["packets"]
    selection_path = mock_holdout_packets_and_selection["selection"]
    out_labels = tmp_path / "out_labels.json"

    reviewed_input = {
        "questions": {
            "SYNTH_Q1": {
                "arms": {
                    "BASE": {
                        "claims": [
                            {"claim_id": "C1", "entailment_label": "SUPPORTED"},
                            {"claim_id": "C1", "entailment_label": "CONTRADICTED"},  # Duplicate
                            {"claim_id": "C2", "entailment_label": "INSUFFICIENT"},
                        ]
                    }
                }
            },
            "SYNTH_Q2": {
                "arms": {
                    "PRIMARY": {
                        "claims": [
                            {"claim_id": "C1", "entailment_label": "CONTRADICTED"}
                        ]
                    }
                }
            },
        }
    }
    input_file = tmp_path / "duplicate_dict_list_input.json"
    input_file.write_text(json.dumps(reviewed_input), encoding="utf-8")

    with pytest.raises(DataValidationError, match="HOLD_OUT_LABEL_DUPLICATE"):
        freeze_holdout_labels(
            holdout_packets_path=packets_path,
            holdout_selection_path=selection_path,
            reviewed_input_path=input_file,
            output_labels_path=out_labels,
            bypass_source_checksums=True,
        )


def test_freeze_holdout_labels_duplicate_json_key_fails(mock_holdout_packets_and_selection, tmp_path):
    """Test that duplicate JSON keys fail closed."""
    packets_path = mock_holdout_packets_and_selection["packets"]
    selection_path = mock_holdout_packets_and_selection["selection"]
    out_labels = tmp_path / "out_labels.json"

    raw_json_with_duplicate = """{
        "questions": {
            "SYNTH_Q1": {
                "arms": {
                    "BASE": {
                        "claims": {
                            "C1": {"entailment_label": "SUPPORTED"},
                            "C1": {"entailment_label": "CONTRADICTED"},
                            "C2": {"entailment_label": "INSUFFICIENT"}
                        }
                    }
                }
            },
            "SYNTH_Q2": {
                "arms": {
                    "PRIMARY": {
                        "claims": {
                            "C1": {"entailment_label": "CONTRADICTED"}
                        }
                    }
                }
            }
        }
    }"""
    input_file = tmp_path / "dup_key_input.json"
    input_file.write_text(raw_json_with_duplicate, encoding="utf-8")

    with pytest.raises(DataValidationError, match="Duplicate JSON key detected"):
        freeze_holdout_labels(
            holdout_packets_path=packets_path,
            holdout_selection_path=selection_path,
            reviewed_input_path=input_file,
            output_labels_path=out_labels,
            bypass_source_checksums=True,
        )


def test_freeze_holdout_labels_missing_claim_fails(mock_holdout_packets_and_selection, tmp_path):
    """Test that missing claim in human review fails closed."""
    packets_path = mock_holdout_packets_and_selection["packets"]
    selection_path = mock_holdout_packets_and_selection["selection"]
    out_labels = tmp_path / "out_labels.json"

    # Incomplete review (missing SYNTH_Q2)
    reviewed_input = {
        "questions": {
            "SYNTH_Q1": {
                "arms": {
                    "BASE": {
                        "claims": {
                            "C1": {"entailment_label": "SUPPORTED"},
                            "C2": {"entailment_label": "INSUFFICIENT"},
                        }
                    }
                }
            }
        }
    }
    input_file = tmp_path / "incomplete_input.json"
    input_file.write_text(json.dumps(reviewed_input), encoding="utf-8")

    with pytest.raises(DataValidationError, match="packet claims missing"):
        freeze_holdout_labels(
            holdout_packets_path=packets_path,
            holdout_selection_path=selection_path,
            reviewed_input_path=input_file,
            output_labels_path=out_labels,
            bypass_source_checksums=True,
        )


def test_freeze_holdout_labels_extra_claim_fails(mock_holdout_packets_and_selection, tmp_path):
    """Test that extra claim in human review fails closed."""
    packets_path = mock_holdout_packets_and_selection["packets"]
    selection_path = mock_holdout_packets_and_selection["selection"]
    out_labels = tmp_path / "out_labels.json"

    reviewed_input = {
        "questions": {
            "SYNTH_Q1": {
                "arms": {
                    "BASE": {
                        "claims": {
                            "C1": {"entailment_label": "SUPPORTED"},
                            "C2": {"entailment_label": "INSUFFICIENT"},
                            "C99_EXTRA": {"entailment_label": "SUPPORTED"},
                        }
                    }
                }
            },
            "SYNTH_Q2": {
                "arms": {
                    "PRIMARY": {
                        "claims": {
                            "C1": {"entailment_label": "CONTRADICTED"}
                        }
                    }
                }
            },
        }
    }
    input_file = tmp_path / "extra_input.json"
    input_file.write_text(json.dumps(reviewed_input), encoding="utf-8")

    with pytest.raises(DataValidationError, match="extra claims in human review"):
        freeze_holdout_labels(
            holdout_packets_path=packets_path,
            holdout_selection_path=selection_path,
            reviewed_input_path=input_file,
            output_labels_path=out_labels,
            bypass_source_checksums=True,
        )


def test_freeze_holdout_labels_invalid_entailment_label(mock_holdout_packets_and_selection, tmp_path):
    """Test that invalid label string fails closed."""
    packets_path = mock_holdout_packets_and_selection["packets"]
    selection_path = mock_holdout_packets_and_selection["selection"]
    out_labels = tmp_path / "out_labels.json"

    reviewed_input = {
        "questions": {
            "SYNTH_Q1": {
                "arms": {
                    "BASE": {
                        "claims": {
                            "C1": {"entailment_label": "INVALID_LABEL_VALUE"},
                            "C2": {"entailment_label": "INSUFFICIENT"},
                        }
                    }
                }
            },
            "SYNTH_Q2": {
                "arms": {
                    "PRIMARY": {
                        "claims": {
                            "C1": {"entailment_label": "CONTRADICTED"}
                        }
                    }
                }
            },
        }
    }
    input_file = tmp_path / "invalid_label_input.json"
    input_file.write_text(json.dumps(reviewed_input), encoding="utf-8")

    with pytest.raises(DataValidationError, match="Invalid entailment_label"):
        freeze_holdout_labels(
            holdout_packets_path=packets_path,
            holdout_selection_path=selection_path,
            reviewed_input_path=input_file,
            output_labels_path=out_labels,
            bypass_source_checksums=True,
        )


def test_freeze_holdout_labels_text_sha_mismatch(mock_holdout_packets_and_selection, tmp_path):
    """Test that mismatched claim text SHA-256 fails closed."""
    packets_path = mock_holdout_packets_and_selection["packets"]
    selection_path = mock_holdout_packets_and_selection["selection"]
    out_labels = tmp_path / "out_labels.json"

    reviewed_input = {
        "questions": {
            "SYNTH_Q1": {
                "arms": {
                    "BASE": {
                        "claims": {
                            "C1": {
                                "entailment_label": "SUPPORTED",
                                "claim_text_sha256": "wrong_sha256_digest_00000000000000000000000000000000000000",
                            },
                            "C2": {"entailment_label": "INSUFFICIENT"},
                        }
                    }
                }
            },
            "SYNTH_Q2": {
                "arms": {
                    "PRIMARY": {
                        "claims": {
                            "C1": {"entailment_label": "CONTRADICTED"}
                        }
                    }
                }
            },
        }
    }
    input_file = tmp_path / "sha_mismatch_input.json"
    input_file.write_text(json.dumps(reviewed_input), encoding="utf-8")

    with pytest.raises(DataValidationError, match="Claim text SHA mismatch"):
        freeze_holdout_labels(
            holdout_packets_path=packets_path,
            holdout_selection_path=selection_path,
            reviewed_input_path=input_file,
            output_labels_path=out_labels,
            bypass_source_checksums=True,
        )


def test_freeze_holdout_labels_canonical_checksum_enforcement(mock_holdout_packets_and_selection, tmp_path):
    """Test that canonical outer checksum verification fails on non-canonical files when not bypassed."""
    packets_path = mock_holdout_packets_and_selection["packets"]
    selection_path = mock_holdout_packets_and_selection["selection"]
    out_labels = tmp_path / "out_labels.json"
    input_file = tmp_path / "input.json"
    input_file.write_text(json.dumps({"questions": {}}), encoding="utf-8")

    with pytest.raises(DataValidationError, match="Holdout review packets SHA mismatch"):
        freeze_holdout_labels(
            holdout_packets_path=packets_path,
            holdout_selection_path=selection_path,
            reviewed_input_path=input_file,
            output_labels_path=out_labels,
            bypass_source_checksums=False,
        )
