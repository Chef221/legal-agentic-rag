"""Hugging Face source for the approved AIO legal dataset."""

from collections.abc import Callable, Iterable, Mapping
from datetime import UTC, datetime
from hashlib import sha256
from itertools import islice
import json

from legal_agentic_rag import __version__
from legal_agentic_rag.configuration.offline import DatasetSourceConfig
from legal_agentic_rag.contracts.dataset_source import DatasetComponent
from legal_agentic_rag.exceptions import (
    BackendInitializationError,
    ConfigurationError,
    DatasetSchemaError,
    ExternalServiceError,
)
from legal_agentic_rag.offline.datasets.aio.raw_schema import AIO_DATASET_NAME
from legal_agentic_rag.schemas.manifests import DatasetManifest

LoadDataset = Callable[..., object]
Clock = Callable[[], datetime]


def _load_hugging_face_dataset(**kwargs: object) -> object:
    """Import the optional backend only when a dataset is actually loaded."""
    try:
        from datasets import Features, Value, load_dataset, load_dataset_builder
    except ImportError as exc:
        raise BackendInitializationError(
            "The 'datasets' package is required to load the AIO dataset"
        ) from exc
    component = kwargs.pop("logical_component")
    if component == DatasetComponent.CONTENT:
        # The current dataset card declares string while its Parquet payload uses
        # large_string. An explicit feature prevents Arrow from narrowing a table
        # whose combined content exceeds the 32-bit string offset limit.
        builder = load_dataset_builder(
            path=kwargs["path"],
            name=kwargs["name"],
            revision=kwargs.get("revision"),
        )
        if builder.info.features is None:
            raise DatasetSchemaError("AIO content config does not declare features")
        features = Features(dict(builder.info.features))
        features["content_html"] = Value("large_string")
        kwargs["features"] = features
    return load_dataset(**kwargs)


class AioDatasetSource:
    """Yield unmodified records from the three approved AIO configs."""

    def __init__(
        self,
        config: DatasetSourceConfig,
        *,
        load_dataset: LoadDataset = _load_hugging_face_dataset,
        clock: Clock | None = None,
    ) -> None:
        if config.dataset_name.strip() != AIO_DATASET_NAME:
            raise ConfigurationError(
                f"AioDatasetSource only supports '{AIO_DATASET_NAME}'"
            )
        self._config = config
        self._load_dataset = load_dataset
        self._clock = clock or (lambda: datetime.now(UTC))
        self._loaded_at = self._clock()
        self._record_counts: dict[str, int] = {}

    @property
    def dataset_name(self) -> str:
        """Return the approved Hugging Face dataset identifier."""
        return self._config.dataset_name

    @property
    def dataset_revision(self) -> str | None:
        """Return the requested revision, which may be unpinned."""
        return self._config.dataset_revision

    def iter_records(
        self, component: DatasetComponent, limit: int | None = None
    ) -> Iterable[Mapping[str, object]]:
        """Yield copied raw records from one component with a bounded limit."""
        effective_limit = self._effective_limit(limit)
        config_name = self._component_configs()[component]

        def generate() -> Iterable[Mapping[str, object]]:
            yielded = 0
            try:
                loaded = self._load_dataset(
                    logical_component=component,
                    path=self.dataset_name,
                    name=config_name,
                    split=self._config.split,
                    revision=self.dataset_revision,
                    streaming=self._config.streaming,
                )
                if not isinstance(loaded, Iterable):
                    raise DatasetSchemaError(
                        f"Dataset component '{component.value}' is not iterable"
                    )
                records = iter(loaded)
                selected = (
                    records if effective_limit is None else islice(records, effective_limit)
                )
                for raw_record in selected:
                    if not isinstance(raw_record, Mapping):
                        raise DatasetSchemaError(
                            f"Dataset component '{component.value}' returned a non-mapping record"
                        )
                    yielded += 1
                    yield dict(raw_record)
            except (BackendInitializationError, DatasetSchemaError):
                raise
            except Exception as exc:
                raise ExternalServiceError(
                    f"Failed to load dataset component '{component.value}'"
                ) from exc
            finally:
                self._record_counts[component.value] = yielded

        return generate()

    def dataset_manifest(self) -> DatasetManifest:
        """Return provenance and the counts observed by completed iterations."""
        component_configs = self._component_configs()
        warnings: list[str] = []
        if self.dataset_revision is None:
            warnings.append("Dataset revision is not pinned")
        if self._config.sample_limit is not None:
            warnings.append(
                f"Sample mode limited each component to {self._config.sample_limit} records"
            )
        unloaded = [
            component.value
            for component in DatasetComponent
            if component.value not in self._record_counts
        ]
        if unloaded:
            warnings.append(
                "Components not iterated: " + ", ".join(sorted(unloaded))
            )
        return DatasetManifest(
            schema_version="1.0",
            dataset_name=self.dataset_name,
            dataset_revision=self.dataset_revision,
            loaded_at=self._loaded_at,
            configs=[component_configs[component] for component in DatasetComponent],
            record_counts={
                component.value: self._record_counts.get(component.value, 0)
                for component in DatasetComponent
            },
            processing_config_hash=self._config_hash(),
            code_version=__version__,
            warnings=warnings,
        )

    def _component_configs(self) -> dict[DatasetComponent, str]:
        return {
            DatasetComponent.METADATA: self._config.metadata_config,
            DatasetComponent.CONTENT: self._config.content_config,
            DatasetComponent.RELATIONSHIPS: self._config.relationships_config,
        }

    def _effective_limit(self, limit: int | None) -> int | None:
        if limit is not None and limit <= 0:
            raise ConfigurationError("limit must be greater than zero")
        configured = self._config.sample_limit
        if limit is None:
            return configured
        if configured is None:
            return limit
        return min(limit, configured)

    def _config_hash(self) -> str:
        serialized = json.dumps(
            self._config.model_dump(mode="json"),
            ensure_ascii=False,
            separators=(",", ":"),
            sort_keys=True,
        )
        return sha256(serialized.encode("utf-8")).hexdigest()
