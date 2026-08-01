"""Typed policy for validating one complete offline artifact set."""

from pathlib import Path

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator


class BuildValidationConfig(BaseModel):
    """Reproducibility and completeness requirements applied after a build."""

    model_config = ConfigDict(extra="forbid", str_strip_whitespace=True)

    require_pinned_dataset_revision: bool = False
    require_full_corpus: bool = False
    expected_record_counts: dict[str, int] = Field(default_factory=dict)
    report_filename: str = "build_validation.json"

    @field_validator("expected_record_counts")
    @classmethod
    def validate_expected_record_counts(
        cls,
        values: dict[str, int],
    ) -> dict[str, int]:
        """Require named, positive component counts without dataset assumptions."""
        normalized: dict[str, int] = {}
        for component, count in values.items():
            name = component.strip()
            if not name or count <= 0:
                raise ValueError(
                    "expected record counts require non-empty names and positive values"
                )
            if name in normalized:
                raise ValueError(
                    "expected record counts contain duplicate component names"
                )
            normalized[name] = count
        return normalized

    @field_validator("report_filename")
    @classmethod
    def validate_report_filename(cls, value: str) -> str:
        """Keep the report as one JSON file directly under the artifact root."""
        path = Path(value)
        if (
            path.is_absolute()
            or len(path.parts) != 1
            or path.suffix.casefold() != ".json"
            or value in {".", ".."}
        ):
            raise ValueError("validation report must be one relative JSON filename")
        return value

    @model_validator(mode="after")
    def validate_full_corpus_policy(self) -> "BuildValidationConfig":
        """A full-corpus claim requires pinned provenance and expected counts."""
        if self.require_full_corpus and not self.require_pinned_dataset_revision:
            raise ValueError("full-corpus validation requires a pinned revision")
        if self.require_full_corpus and not self.expected_record_counts:
            raise ValueError("full-corpus validation requires expected record counts")
        return self
