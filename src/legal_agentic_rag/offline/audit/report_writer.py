"""Persist a dataset audit report and focused CSV issue extracts."""

import csv
import json
from pathlib import Path

from legal_agentic_rag.exceptions import ArtifactCompatibilityError
from legal_agentic_rag.schemas.auditing import AuditIssue, DatasetAuditReport


class DatasetAuditReportWriter:
    """Write the Milestone 2 JSON and CSV outputs without silent overwrite."""

    _FILENAMES = (
        "data_audit.json",
        "missing_content.csv",
        "orphan_content.csv",
        "invalid_relationships.csv",
        "duplicate_records.csv",
    )

    def write(
        self,
        report: DatasetAuditReport,
        output_directory: Path,
        *,
        overwrite: bool = False,
    ) -> dict[str, Path]:
        """Persist all audit outputs and return their paths by filename."""
        paths = {name: output_directory / name for name in self._FILENAMES}
        existing = [path.name for path in paths.values() if path.exists()]
        if existing and not overwrite:
            raise ArtifactCompatibilityError(
                "Audit output already exists: " + ", ".join(sorted(existing))
            )
        output_directory.mkdir(parents=True, exist_ok=True)
        paths["data_audit.json"].write_text(
            json.dumps(
                report.model_dump(mode="json"),
                ensure_ascii=False,
                indent=2,
                sort_keys=True,
            )
            + "\n",
            encoding="utf-8",
        )
        self._write_join_issues(
            paths["missing_content.csv"],
            report.issues,
            "missing_content",
        )
        self._write_join_issues(
            paths["orphan_content.csv"],
            report.issues,
            "orphan_content",
        )
        self._write_relationship_issues(
            paths["invalid_relationships.csv"], report.issues
        )
        self._write_duplicate_issues(
            paths["duplicate_records.csv"], report.issues
        )
        return paths

    @staticmethod
    def _write_join_issues(
        path: Path, issues: list[AuditIssue], issue_type: str
    ) -> None:
        with path.open("w", encoding="utf-8-sig", newline="") as stream:
            writer = csv.DictWriter(stream, fieldnames=["document_id", "issue_type"])
            writer.writeheader()
            for issue in issues:
                if issue.issue_type == issue_type:
                    writer.writerow(
                        {"document_id": issue.record_id, "issue_type": issue_type}
                    )

    @staticmethod
    def _write_relationship_issues(path: Path, issues: list[AuditIssue]) -> None:
        selected = {
            "invalid_relationship_source",
            "invalid_relationship_target",
            "empty_relationship_endpoint",
            "malformed_relationship_endpoint",
            "empty_relationship_label",
        }
        with path.open("w", encoding="utf-8-sig", newline="") as stream:
            fields = [
                "source_document_id",
                "target_document_id",
                "relationship",
                "reason",
            ]
            writer = csv.DictWriter(stream, fieldnames=fields)
            writer.writeheader()
            for issue in issues:
                if issue.issue_type in selected:
                    writer.writerow(
                        {
                            "source_document_id": issue.record_id,
                            "target_document_id": issue.metadata.get("target_id"),
                            "relationship": issue.metadata.get("relationship"),
                            "reason": issue.issue_type,
                        }
                    )

    @staticmethod
    def _write_duplicate_issues(path: Path, issues: list[AuditIssue]) -> None:
        selected = {"duplicate_id", "duplicate_content", "duplicate_edge"}
        with path.open("w", encoding="utf-8-sig", newline="") as stream:
            fields = ["component", "record_id", "duplicate_type", "count"]
            writer = csv.DictWriter(stream, fieldnames=fields)
            writer.writeheader()
            for issue in issues:
                if issue.issue_type in selected:
                    writer.writerow(
                        {
                            "component": issue.metadata.get("component"),
                            "record_id": issue.record_id,
                            "duplicate_type": issue.issue_type,
                            "count": issue.metadata.get("count"),
                        }
                    )
