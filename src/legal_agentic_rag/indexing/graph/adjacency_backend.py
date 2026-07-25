"""Persisted standard-library adjacency graph for bounded legal traversal."""

from __future__ import annotations

from collections import defaultdict, deque
from collections.abc import Iterable, Sequence
from datetime import UTC, datetime
from hashlib import sha256
import json
import logging
from pathlib import Path
import shutil
import tempfile
from typing import Callable

from pydantic import TypeAdapter, ValidationError

from legal_agentic_rag import __version__
from legal_agentic_rag.configuration.hashing import canonical_sha256
from legal_agentic_rag.configuration.offline import GraphIndexConfig
from legal_agentic_rag.exceptions import (
    ArtifactCompatibilityError,
    BackendInitializationError,
    DataValidationError,
    RetrievalError,
)
from legal_agentic_rag.schemas.legal_documents import LegalDocument
from legal_agentic_rag.schemas.legal_relationships import LegalRelationship
from legal_agentic_rag.schemas.manifests import ArtifactManifest, ArtifactType
from legal_agentic_rag.schemas.retrieval import GraphPathStep

GRAPH_FILENAME = "graph.json"
MANIFEST_FILENAME = "manifest.json"
Clock = Callable[[], datetime]
_RELATIONSHIPS_ADAPTER = TypeAdapter(list[LegalRelationship])
_LOGGER = logging.getLogger(__name__)


class AdjacencyGraphBackend:
    """Build, persist, reload, and traverse a directed adjacency graph."""

    def __init__(
        self,
        config: GraphIndexConfig | None = None,
        *,
        clock: Clock | None = None,
    ) -> None:
        self._config = config or GraphIndexConfig()
        self._clock = clock or (lambda: datetime.now(UTC))
        self._document_ids: tuple[str, ...] = ()
        self._relationships: tuple[LegalRelationship, ...] = ()
        self._adjacency: dict[str, tuple[LegalRelationship, ...]] = {}
        self._manifest: ArtifactManifest | None = None

    @property
    def manifest(self) -> ArtifactManifest:
        """Return the manifest of the ready graph."""
        if self._manifest is None:
            raise BackendInitializationError("Graph index has not been built or loaded")
        return self._manifest

    def build(
        self,
        documents: Iterable[LegalDocument],
        relationships: Iterable[LegalRelationship],
        *,
        document_manifest: ArtifactManifest,
        relationship_manifest: ArtifactManifest,
    ) -> ArtifactManifest:
        """Build an immutable directed graph from compatible unified artifacts."""
        document_list = list(documents)
        relationship_list = list(relationships)
        self._validate_build_inputs(
            document_list,
            relationship_list,
            document_manifest,
            relationship_manifest,
        )
        self._document_ids = tuple(
            sorted(document.document_id for document in document_list)
        )
        self._relationships = tuple(sorted(relationship_list, key=self._edge_key))
        self._adjacency = self._build_adjacency(
            self._document_ids, self._relationships
        )
        self._manifest = ArtifactManifest(
            schema_version="1.0",
            artifact_type=ArtifactType.GRAPH_INDEX,
            artifact_version=self._config.artifact_version,
            dataset_name=document_manifest.dataset_name,
            dataset_revision=document_manifest.dataset_revision,
            created_at=self._clock(),
            record_count=len(self._relationships),
            processing_config_hash=self._config_hash(),
            code_version=__version__,
            backend=self._config.backend_name,
            metadata={
                "node_count": len(self._document_ids),
                "edge_count": len(self._relationships),
                "direction": "directed",
                "source_document_artifact_version": (
                    document_manifest.artifact_version
                ),
                "source_document_processing_config_hash": (
                    document_manifest.processing_config_hash
                ),
                "source_relationship_artifact_version": (
                    relationship_manifest.artifact_version
                ),
                "source_relationship_processing_config_hash": (
                    relationship_manifest.processing_config_hash
                ),
            },
        )
        _LOGGER.info(
            "graph_index_built",
            extra={
                "document_count": len(self._document_ids),
                "relationship_count": len(self._relationships),
            },
        )
        return self._manifest

    def persist(self, destination: Path) -> ArtifactManifest:
        """Persist graph payload and checksum without overwriting artifacts."""
        manifest = self.manifest
        destination = destination.resolve()
        if destination.exists():
            raise BackendInitializationError(
                "Graph artifact destination already exists"
            )
        if not destination.parent.exists():
            raise BackendInitializationError(
                "Graph artifact parent directory does not exist"
            )
        temporary = Path(
            tempfile.mkdtemp(prefix=f".{destination.name}-", dir=destination.parent)
        )
        destination_created = False
        try:
            graph_path = temporary / GRAPH_FILENAME
            graph_path.write_text(
                json.dumps(
                    {
                        "document_ids": list(self._document_ids),
                        "relationships": [
                            item.model_dump(mode="json")
                            for item in self._relationships
                        ],
                    },
                    ensure_ascii=False,
                    separators=(",", ":"),
                    sort_keys=True,
                )
                + "\n",
                encoding="utf-8",
            )
            metadata = dict(manifest.metadata)
            metadata.update(
                {
                    "graph_filename": GRAPH_FILENAME,
                    "graph_sha256": self._sha256_file(graph_path),
                    "manifest_filename": MANIFEST_FILENAME,
                }
            )
            final_manifest = manifest.model_copy(update={"metadata": metadata})
            (temporary / MANIFEST_FILENAME).write_text(
                json.dumps(
                    final_manifest.model_dump(mode="json"),
                    ensure_ascii=False,
                    indent=2,
                    sort_keys=True,
                )
                + "\n",
                encoding="utf-8",
            )
            destination.mkdir(exist_ok=False)
            destination_created = True
            for staged_file in temporary.iterdir():
                staged_file.replace(destination / staged_file.name)
            temporary.rmdir()
            self._manifest = final_manifest
            return final_manifest
        except (OSError, ValueError) as error:
            shutil.rmtree(temporary, ignore_errors=True)
            if destination_created:
                shutil.rmtree(destination, ignore_errors=True)
            raise BackendInitializationError(
                "Graph artifact could not be persisted"
            ) from error
        except Exception:
            shutil.rmtree(temporary, ignore_errors=True)
            if destination_created:
                shutil.rmtree(destination, ignore_errors=True)
            raise

    def load(self, source: Path, manifest: ArtifactManifest) -> None:
        """Load and validate a persisted graph artifact."""
        source = source.resolve()
        graph_path = source / GRAPH_FILENAME
        manifest_path = source / MANIFEST_FILENAME
        if not source.is_dir() or not all(
            path.is_file() for path in (graph_path, manifest_path)
        ):
            raise ArtifactCompatibilityError("Graph artifact files are incomplete")
        try:
            stored_manifest = ArtifactManifest.model_validate_json(
                manifest_path.read_text(encoding="utf-8")
            )
        except (OSError, ValidationError, ValueError) as error:
            raise ArtifactCompatibilityError("Graph manifest is invalid") from error
        if stored_manifest != manifest:
            raise ArtifactCompatibilityError(
                "Supplied graph manifest does not match persisted manifest"
            )
        self._validate_manifest(stored_manifest)
        expected_checksum = stored_manifest.metadata.get("graph_sha256")
        if (
            not isinstance(expected_checksum, str)
            or self._sha256_file(graph_path) != expected_checksum
        ):
            raise ArtifactCompatibilityError("Graph artifact checksum does not match")
        try:
            payload = json.loads(graph_path.read_text(encoding="utf-8"))
            document_ids = tuple(payload["document_ids"])
            relationships = tuple(
                _RELATIONSHIPS_ADAPTER.validate_python(payload["relationships"])
            )
        except (
            KeyError,
            OSError,
            TypeError,
            ValidationError,
            ValueError,
            json.JSONDecodeError,
        ) as error:
            raise ArtifactCompatibilityError(
                "Graph artifact payload is invalid"
            ) from error
        self._validate_loaded_payload(
            document_ids, relationships, stored_manifest
        )
        self._document_ids = document_ids
        self._relationships = relationships
        self._adjacency = self._build_adjacency(document_ids, relationships)
        self._manifest = stored_manifest

    def traverse(
        self,
        seed_document_ids: Sequence[str],
        max_hops: int,
        relationship_types: Sequence[str] | None = None,
    ) -> Sequence[GraphPathStep]:
        """Return one deterministic BFS discovery edge per reached document."""
        _ = self.manifest
        if not 1 <= max_hops <= 2:
            raise RetrievalError("Graph traversal max_hops must be between 1 and 2")
        seeds = list(dict.fromkeys(seed_document_ids))
        if any(not seed.strip() for seed in seeds):
            raise RetrievalError("Graph seed document IDs must not be empty")
        unknown = set(seeds) - set(self._document_ids)
        if unknown:
            raise RetrievalError("Graph seeds are absent from the graph artifact")
        allowed_types = self._relationship_type_filter(relationship_types)
        visited = set(seeds)
        queue = deque((seed, 0) for seed in seeds)
        steps: list[GraphPathStep] = []
        while queue:
            source_id, source_hop = queue.popleft()
            if source_hop >= max_hops:
                continue
            for relationship in self._adjacency[source_id]:
                effective_type = self._effective_type(relationship)
                if allowed_types is not None and effective_type not in allowed_types:
                    continue
                target_id = relationship.target_document_id
                if target_id in visited:
                    continue
                hop = source_hop + 1
                visited.add(target_id)
                queue.append((target_id, hop))
                steps.append(
                    GraphPathStep(
                        source_document_id=source_id,
                        target_document_id=target_id,
                        relationship_type=effective_type,
                        hop=hop,
                    )
                )
        return steps

    def _validate_build_inputs(
        self,
        documents: list[LegalDocument],
        relationships: list[LegalRelationship],
        document_manifest: ArtifactManifest,
        relationship_manifest: ArtifactManifest,
    ) -> None:
        if document_manifest.artifact_type != ArtifactType.NORMALIZED_DOCUMENTS:
            raise ArtifactCompatibilityError(
                "Graph build requires normalized documents"
            )
        if relationship_manifest.artifact_type != ArtifactType.RELATIONSHIP_MAPPING:
            raise ArtifactCompatibilityError(
                "Graph build requires a relationship mapping"
            )
        if (
            document_manifest.dataset_name != relationship_manifest.dataset_name
            or document_manifest.dataset_revision
            != relationship_manifest.dataset_revision
        ):
            raise ArtifactCompatibilityError(
                "Graph source artifacts originate from different datasets"
            )
        source_hash = relationship_manifest.metadata.get(
            "source_processing_config_hash"
        )
        if source_hash != document_manifest.processing_config_hash:
            raise ArtifactCompatibilityError(
                "Relationship mapping does not originate from these documents"
            )
        document_ids = [document.document_id for document in documents]
        if (
            document_manifest.record_count != len(documents)
            or len(document_ids) != len(set(document_ids))
        ):
            raise DataValidationError(
                "Graph document payload is incompatible with its manifest"
            )
        if relationship_manifest.record_count != len(relationships):
            raise DataValidationError(
                "Graph relationship payload is incompatible with its manifest"
            )
        known_documents = set(document_ids)
        identities = [self._edge_key(item) for item in relationships]
        if len(identities) != len(set(identities)):
            raise DataValidationError("Graph relationships must be unique")
        for relationship in relationships:
            if not relationship.is_directed:
                raise DataValidationError(
                    "Reference graph only accepts directed relationships"
                )
            if (
                relationship.source_document_id not in known_documents
                or relationship.target_document_id not in known_documents
                or relationship.source_document_id
                == relationship.target_document_id
            ):
                raise DataValidationError(
                    "Graph relationship contains an invalid endpoint"
                )

    def _validate_manifest(self, manifest: ArtifactManifest) -> None:
        if (
            manifest.artifact_type != ArtifactType.GRAPH_INDEX
            or manifest.artifact_version != self._config.artifact_version
            or manifest.backend != self._config.backend_name
            or manifest.metadata.get("direction") != "directed"
        ):
            raise ArtifactCompatibilityError("Graph manifest is incompatible")

    def _validate_loaded_payload(
        self,
        document_ids: tuple[str, ...],
        relationships: tuple[LegalRelationship, ...],
        manifest: ArtifactManifest,
    ) -> None:
        node_count = manifest.metadata.get("node_count")
        if (
            not all(isinstance(item, str) and item.strip() for item in document_ids)
            or document_ids != tuple(sorted(document_ids))
            or len(document_ids) != len(set(document_ids))
            or node_count != len(document_ids)
            or manifest.record_count != len(relationships)
            or tuple(sorted(relationships, key=self._edge_key)) != relationships
        ):
            raise ArtifactCompatibilityError(
                "Graph payload ordering or counts are incompatible"
            )
        known_documents = set(document_ids)
        identities = [self._edge_key(item) for item in relationships]
        if len(identities) != len(set(identities)):
            raise ArtifactCompatibilityError("Graph payload contains duplicate edges")
        if any(
            not relationship.is_directed
            or relationship.source_document_id not in known_documents
            or relationship.target_document_id not in known_documents
            or relationship.source_document_id == relationship.target_document_id
            for relationship in relationships
        ):
            raise ArtifactCompatibilityError(
                "Graph payload contains an invalid directed edge"
            )

    @staticmethod
    def _build_adjacency(
        document_ids: Sequence[str],
        relationships: Sequence[LegalRelationship],
    ) -> dict[str, tuple[LegalRelationship, ...]]:
        adjacency: defaultdict[str, list[LegalRelationship]] = defaultdict(list)
        for relationship in relationships:
            adjacency[relationship.source_document_id].append(relationship)
        return {
            document_id: tuple(
                sorted(
                    adjacency[document_id],
                    key=lambda item: (
                        item.target_document_id,
                        item.relationship_type or "",
                        item.raw_relationship,
                    ),
                )
            )
            for document_id in document_ids
        }

    @staticmethod
    def _edge_key(
        relationship: LegalRelationship,
    ) -> tuple[str, str, str, str]:
        return (
            relationship.source_document_id,
            relationship.target_document_id,
            relationship.raw_relationship,
            relationship.relationship_type or "",
        )

    @staticmethod
    def _effective_type(relationship: LegalRelationship) -> str:
        return relationship.relationship_type or relationship.raw_relationship

    @staticmethod
    def _relationship_type_filter(
        relationship_types: Sequence[str] | None,
    ) -> set[str] | None:
        if relationship_types is None:
            return None
        normalized = [item.strip() for item in relationship_types]
        if any(not item for item in normalized):
            raise RetrievalError("Graph relationship filters must not be empty")
        return set(normalized)

    def _config_hash(self) -> str:
        return canonical_sha256(self._config)

    @staticmethod
    def _sha256_file(path: Path) -> str:
        digest = sha256()
        with path.open("rb") as stream:
            for block in iter(lambda: stream.read(1024 * 1024), b""):
                digest.update(block)
        return digest.hexdigest()
