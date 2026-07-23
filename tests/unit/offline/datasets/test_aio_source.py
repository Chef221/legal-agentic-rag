"""Unit tests for the bounded Hugging Face AIO source."""

from datetime import UTC, datetime
from typing import Any

import pytest

from legal_agentic_rag.configuration import DatasetSourceConfig
from legal_agentic_rag.contracts.dataset_source import DatasetComponent, DatasetSource
from legal_agentic_rag.exceptions import ConfigurationError, ExternalServiceError
from legal_agentic_rag.offline.datasets.aio import AioDatasetSource


def _config(**overrides: object) -> DatasetSourceConfig:
    values: dict[str, object] = {
        "dataset_name": "th1nhng0/vietnamese-legal-documents",
        "dataset_revision": "fixture-revision",
        "sample_limit": 2,
        "streaming": True,
    }
    values.update(overrides)
    return DatasetSourceConfig(**values)


def test_source_satisfies_contract_and_passes_backend_options() -> None:
    """Logical components map to the configured HF config and split."""
    calls: list[dict[str, object]] = []

    def load_dataset(**kwargs: object) -> list[dict[str, object]]:
        calls.append(kwargs)
        return [{"id": "one"}, {"id": "two"}, {"id": "three"}]

    source = AioDatasetSource(_config(), load_dataset=load_dataset)

    assert isinstance(source, DatasetSource)
    assert list(source.iter_records(DatasetComponent.METADATA)) == [
        {"id": "one"},
        {"id": "two"},
    ]
    assert calls == [
        {
            "logical_component": DatasetComponent.METADATA,
            "path": "th1nhng0/vietnamese-legal-documents",
            "name": "metadata",
            "split": "data",
            "revision": "fixture-revision",
            "streaming": True,
        }
    ]


def test_call_limit_never_exceeds_configured_sample_limit() -> None:
    """Explicit calls cannot accidentally escape sample mode."""
    source = AioDatasetSource(
        _config(sample_limit=2),
        load_dataset=lambda **_: ({"id": index} for index in range(10)),
    )

    records = list(source.iter_records(DatasetComponent.CONTENT, limit=8))

    assert len(records) == 2


def test_source_copies_raw_records_and_records_manifest_counts() -> None:
    """Consumers cannot mutate the backend record through the yielded mapping."""
    original = {"id": "doc-1", "title": "unchanged"}
    fixed_time = datetime(2026, 7, 18, 12, 0, tzinfo=UTC)
    source = AioDatasetSource(
        _config(sample_limit=None),
        load_dataset=lambda **_: [original],
        clock=lambda: fixed_time,
    )
    loaded = list(source.iter_records(DatasetComponent.METADATA))
    loaded[0]["title"] = "changed"

    manifest = source.dataset_manifest()

    assert original["title"] == "unchanged"
    assert manifest.loaded_at == fixed_time
    assert manifest.record_counts["metadata"] == 1
    assert manifest.record_counts["content"] == 0
    assert "Components not iterated" in manifest.warnings[0]
    assert len(manifest.processing_config_hash) == 64


def test_source_rejects_unapproved_dataset_and_invalid_limit() -> None:
    """The concrete AIO source cannot silently load another corpus."""
    with pytest.raises(ConfigurationError):
        AioDatasetSource(_config(dataset_name="another/dataset"))

    source = AioDatasetSource(_config(), load_dataset=lambda **_: [])
    with pytest.raises(ConfigurationError):
        list(source.iter_records(DatasetComponent.METADATA, limit=0))


def test_backend_failures_are_classified_without_exposing_details() -> None:
    """External loading errors cross the boundary as a domain exception."""
    def fail(**_: Any) -> object:
        raise RuntimeError("sensitive backend detail")

    source = AioDatasetSource(_config(), load_dataset=fail)

    with pytest.raises(ExternalServiceError, match="metadata") as raised:
        list(source.iter_records(DatasetComponent.METADATA))
    assert "sensitive" not in str(raised.value)
