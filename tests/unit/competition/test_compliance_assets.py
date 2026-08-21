"""Static safety checks for competition compliance artifacts."""

from pathlib import Path


_ROOT = Path(__file__).resolve().parents[3]


def test_docker_context_excludes_data_models_artifacts_and_secrets() -> None:
    """Large or sensitive local state must not enter the image context."""
    ignored = set(
        (_ROOT / ".dockerignore").read_text(encoding="utf-8").splitlines()
    )

    assert {".env", "artifacts", "data", "models", "submission.zip"} <= ignored


def test_dockerfile_runs_as_non_root_and_does_not_copy_local_state() -> None:
    """The reproduction scaffold installs source and drops root privileges."""
    dockerfile = (_ROOT / "Dockerfile").read_text(encoding="utf-8")

    assert "USER legalrag" in dockerfile
    assert "COPY src ./src" in dockerfile
    assert "COPY data" not in dockerfile
    assert "COPY artifacts" not in dockerfile
    assert "COPY models" not in dockerfile


def test_required_compliance_templates_are_present_and_actionable() -> None:
    """Organizer-required evidence templates must not be empty placeholders."""
    expected = {
        "DATA-STATEMENT.md": "Official data used",
        "MODEL-CARD.md": "BTC registration evidence",
        "PRIVATE-SUBMISSION-CHECKLIST.md": "fewer than 3",
        "SUBMISSION-LEDGER.csv": "archive_sha256",
    }

    for filename, marker in expected.items():
        content = (_ROOT / "docs" / "templates" / filename).read_text(
            encoding="utf-8"
        )
        assert marker in content


def test_model_inventory_records_current_identities_and_registration_gate() -> None:
    """Retained model identities stay exact and require organizer evidence."""
    compliance = (_ROOT / "docs" / "11-COMPETITION-COMPLIANCE.md").read_text(
        encoding="utf-8"
    )

    for revision in (
        "97b0c614be4d77ee51c0cef4e5f07c00f9eb65b3",
        "e61197ed45024b0ed8a2d74b80b4d909f1255473",
        "15852e8c16360a2fea060d615a32b45270f8a8fc",
        "e6f163aa4f094ac5d943893009a78ba2c62798ed6432eb637887a9843944304b",
    ):
        assert revision in compliance
    assert "must be verified against team/BTC records" in compliance
    assert "confirm whether a new registration entry is required" in compliance
    assert "below 4B" in compliance


def test_team_onboarding_covers_all_pipeline_boundaries() -> None:
    """The team guide must remain an actionable end-to-end system map."""
    guide = (_ROOT / "docs" / "12-TEAM-ONBOARDING.md").read_text(
        encoding="utf-8"
    )

    for marker in (
        "Pipeline offline",
        "Pipeline online",
        "API và UI",
        "Batch inference",
        "Evaluation và chọn model",
        "Quy trình làm việc cho thành viên",
    ):
        assert marker in guide


def test_team_handoff_documents_keep_current_candidate_and_next_work() -> None:
    """The repository entry point exposes current evidence and next work."""
    readme = (_ROOT / "README.md").read_text(encoding="utf-8")
    start = (_ROOT / "docs" / "00-START-HERE.md").read_text(encoding="utf-8")
    handoff = (_ROOT / "HANDOFF.md").read_text(encoding="utf-8")
    result = (_ROOT / "docs" / "18-M491-PUBLIC-RESULT.md").read_text(
        encoding="utf-8"
    )

    for filename in (
        "docs/00-START-HERE.md",
        "HANDOFF.md",
        "docs/18-M491-PUBLIC-RESULT.md",
    ):
        assert filename in readme
    assert "M49.1" in start
    assert "M50" in handoff
    assert "0.382772249" in result
    assert "900" in result
