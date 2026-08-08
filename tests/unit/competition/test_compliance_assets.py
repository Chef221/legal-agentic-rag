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


def test_model_inventory_records_approval_without_replacing_primary_evidence() -> None:
    """Approved identities stay exact and still require organizer evidence."""
    compliance = (_ROOT / "docs" / "11-COMPETITION-COMPLIANCE.md").read_text(
        encoding="utf-8"
    )

    for revision in (
        "614241f622f53c4eeff9890bdc4f31cfecc418b3",
        "1427fd652930e4ba29e8149678df786c240d8825",
        "a1d308dfcc03e09da285d49d912439a655a571e8",
    ):
        assert revision in compliance
    assert compliance.count("Người dùng xác nhận BTC đã duyệt") == 3
    assert "không thay thế email/Form/spreadsheet approval gốc" in compliance
    assert "trạng thái approval" in compliance
    assert "`pending` hoặc `unknown`" in compliance


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


def test_team_handoff_documents_keep_current_baseline_and_workstreams() -> None:
    """The repository entry point must expose evidence and actionable ownership."""
    readme = (_ROOT / "README.md").read_text(encoding="utf-8")
    start = (_ROOT / "docs" / "00-START-HERE.md").read_text(encoding="utf-8")
    postmortem = (
        _ROOT / "docs" / "16-M43-BASELINE-POSTMORTEM.md"
    ).read_text(encoding="utf-8")
    backlog = (
        _ROOT / "docs" / "17-TEAM-IMPROVEMENT-BACKLOG.md"
    ).read_text(encoding="utf-8")

    for filename in (
        "docs/00-START-HERE.md",
        "docs/16-M43-BASELINE-POSTMORTEM.md",
        "docs/17-TEAM-IMPROVEMENT-BACKLOG.md",
    ):
        assert filename in readme
    assert "0.07862292376534387" in start
    assert "425" in postmortem
    assert "384" in postmortem
    for workstream in (
        "WS-A",
        "WS-B",
        "WS-C",
        "WS-D",
        "WS-E",
        "WS-F",
        "WS-G",
        "WS-H",
    ):
        assert workstream in backlog
