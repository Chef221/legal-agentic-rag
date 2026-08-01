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


def test_candidate_models_remain_blocked_until_registration() -> None:
    """Documented candidates cannot be mistaken for registered competition use."""
    compliance = (_ROOT / "docs" / "11-COMPETITION-COMPLIANCE.md").read_text(
        encoding="utf-8"
    )

    assert "Không được dùng cho official run" in compliance
    assert "Chờ Form" in compliance


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
