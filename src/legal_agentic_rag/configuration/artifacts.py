"""Typed configuration for artifact location and compatibility policy."""

from pathlib import Path

from pydantic import BaseModel, ConfigDict


class ArtifactConfig(BaseModel):
    """Backend-neutral artifact storage and load policy."""

    model_config = ConfigDict(extra="forbid")

    root_path: Path
    require_compatible: bool = True
    allow_overwrite: bool = False
