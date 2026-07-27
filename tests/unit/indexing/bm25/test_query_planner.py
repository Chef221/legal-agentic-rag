"""Unit tests for bounded corpus-aware BM25 query planning."""

from legal_agentic_rag.configuration import BM25RuntimeConfig
from legal_agentic_rag.indexing.bm25.query_planner import BM25QueryPlanner


def test_planner_prefers_rare_terms_and_preserves_original_order() -> None:
    planner = BM25QueryPlanner(
        BM25RuntimeConfig(
            max_query_terms=3,
            max_document_frequency_ratio=0.25,
        )
    )

    plan = planner.plan(
        ["người", "lao", "động", "nghỉ", "12", "tháng"],
        document_frequencies={
            "người": 900,
            "lao": 500,
            "động": 500,
            "nghỉ": 40,
            "12": 10,
            "tháng": 100,
        },
        document_count=1_000,
    )

    assert plan.terms == ("nghỉ", "12", "tháng")
    assert plan.original_unique_term_count == 6
    assert plan.known_term_count == 6
    assert plan.was_limited is True


def test_planner_retains_negation_and_numeric_legal_meaning() -> None:
    planner = BM25QueryPlanner(BM25RuntimeConfig(max_query_terms=3))

    plan = planner.plan(
        ["người", "không", "được", "nghỉ", "12"],
        document_frequencies={
            "người": 900,
            "không": 800,
            "được": 950,
            "nghỉ": 50,
            "12": 10,
        },
        document_count=1_000,
    )

    assert plan.terms == ("không", "nghỉ", "12")


def test_planner_falls_back_when_every_term_is_common_or_unknown() -> None:
    planner = BM25QueryPlanner(
        BM25RuntimeConfig(
            max_query_terms=2,
            max_document_frequency_ratio=0.01,
        )
    )

    common = planner.plan(
        ["lao", "động", "người"],
        document_frequencies={"lao": 500, "động": 400, "người": 900},
        document_count=1_000,
    )
    unknown = planner.plan(
        ["khôngtồntại", "xyz"],
        document_frequencies={},
        document_count=1_000,
    )

    assert common.terms == ("lao", "động")
    assert unknown.terms == ("khôngtồntại", "xyz")
