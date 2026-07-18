"""Small integration tests across schema, configuration, and contract boundaries."""

from legal_agentic_rag.schemas import (
    AnswerResponse,
    ArtifactManifest,
    DatasetManifest,
    LegalChunk,
    LegalDocument,
)


def test_persisted_schema_samples_roundtrip_through_json(
    load_schema_sample: object,
) -> None:
    """Persisted Milestone 1 schemas retain values through JSON serialization."""
    sample_models = (
        LegalDocument.model_validate(
            load_schema_sample("valid_legal_document.json")  # type: ignore[operator]
        ),
        LegalChunk.model_validate(
            load_schema_sample("valid_legal_chunk.json")  # type: ignore[operator]
        ),
        DatasetManifest.model_validate(
            load_schema_sample("valid_dataset_manifest.json")  # type: ignore[operator]
        ),
        ArtifactManifest.model_validate(
            load_schema_sample("valid_artifact_manifest.json")  # type: ignore[operator]
        ),
        AnswerResponse.model_validate(
            load_schema_sample("valid_answer_response.json")  # type: ignore[operator]
        ),
    )

    for model in sample_models:
        model_type = type(model)
        restored = model_type.model_validate_json(model.model_dump_json())
        assert restored == model
