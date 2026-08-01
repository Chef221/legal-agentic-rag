"""Typed policy for UIT DSC 2026 Task 2 data provenance."""

from typing import Literal

from pydantic import BaseModel, ConfigDict, Field


OFFICIAL_CORPUS_DATASET_NAME = "uit-dsc-2026-task2-selected-contexts"
OFFICIAL_QA_DATASET_NAME = "uit-dsc-2026-task2-legalqa"


class CompetitionConfig(BaseModel):
    """Fail-closed identity policy for competition data and artifacts."""

    model_config = ConfigDict(extra="forbid", str_strip_whitespace=True)

    competition_id: Literal["uit-dsc-2026-task2"] = "uit-dsc-2026-task2"
    corpus_dataset_name: Literal[
        "uit-dsc-2026-task2-selected-contexts"
    ] = OFFICIAL_CORPUS_DATASET_NAME
    qa_dataset_name: Literal["uit-dsc-2026-task2-legalqa"] = (
        OFFICIAL_QA_DATASET_NAME
    )
    data_policy: Literal["competition_only"] = "competition_only"
    allow_external_data: Literal[False] = False
    require_official_artifact_lineage: Literal[True] = True
    source_authority: str = Field(
        default="UIT Data Science Challenge 2026 Task 2 organizers",
        min_length=1,
    )
