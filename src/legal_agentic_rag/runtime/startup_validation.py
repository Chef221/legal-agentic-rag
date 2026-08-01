"""Fast startup trust boundary for a previously deep-validated artifact set."""

from __future__ import annotations

from pathlib import Path

from pydantic import ValidationError

from legal_agentic_rag.configuration.competition import CompetitionConfig
from legal_agentic_rag.exceptions import ArtifactCompatibilityError
from legal_agentic_rag.schemas import (
    ArtifactManifest,
    ArtifactType,
    BuildValidationReport,
)

_REQUIRED_CHECKS: dict[ArtifactType, frozenset[str]] = {
    ArtifactType.LEGAL_CHUNKS: frozenset({"payload_checksum", "record_count"}),
    ArtifactType.BM25_INDEX: frozenset(
        {"payload_checksum", "sqlite_integrity", "record_count"}
    ),
    ArtifactType.VECTOR_INDEX: frozenset(
        {"payload_checksum", "vector_shape", "record_count"}
    ),
    ArtifactType.GRAPH_INDEX: frozenset({"payload_checksum", "record_count"}),
}


def validate_competition_artifact_lineage(
    manifests: tuple[ArtifactManifest, ...],
    policy: CompetitionConfig,
) -> None:
    """Reject artifacts that do not originate from the approved BTC corpus."""
    identities = {
        (manifest.dataset_name, manifest.dataset_revision)
        for manifest in manifests
    }
    if len(identities) != 1:
        raise ArtifactCompatibilityError(
            "Runtime artifacts originate from different datasets"
        )
    if any(
        manifest.dataset_name != policy.corpus_dataset_name
        for manifest in manifests
    ):
        raise ArtifactCompatibilityError(
            "Runtime artifacts do not originate from the approved "
            "competition corpus"
        )


def validate_startup_report(
    report_path: Path,
    manifests: tuple[ArtifactManifest, ...],
) -> BuildValidationReport:
    """Require a valid deep report matching every current online manifest."""
    try:
        report = BuildValidationReport.model_validate_json(
            report_path.read_text(encoding="utf-8")
        )
    except (OSError, ValidationError, ValueError) as error:
        raise ArtifactCompatibilityError(
            "Validated-report startup requires a valid build validation report"
        ) from error
    if not report.is_valid:
        raise ArtifactCompatibilityError(
            "Build validation report does not approve the artifact set"
        )

    for manifest in manifests:
        result = report.artifact_results.get(manifest.artifact_type.value)
        required_checks = _REQUIRED_CHECKS.get(manifest.artifact_type)
        if (
            result is None
            or required_checks is None
            or not result.is_valid
            or result.manifest != manifest
            or not required_checks.issubset(result.passed_checks)
        ):
            raise ArtifactCompatibilityError(
                "Build validation report is incompatible with online artifacts"
            )
    return report
