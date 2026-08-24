"""Comprehensive unit tests for JinaNativeReranker and configuration contract."""

from __future__ import annotations

import copy
import hashlib
import json
import math
from pathlib import Path
from typing import Any
from unittest.mock import MagicMock, patch

import pytest

from legal_agentic_rag.configuration.application import ApplicationConfig
from legal_agentic_rag.configuration.online import RerankerConfig
from legal_agentic_rag.exceptions import (
    BackendInitializationError,
    ConfigurationError,
    ModelError,
    RetrievalError,
)
from legal_agentic_rag.reranking.jina_native import JinaNativeReranker
from legal_agentic_rag.schemas.retrieval import (
    RetrievalHit,
    RetrievalQuery,
    RetrievalStrategy,
    RetrievalTrace,
)


def _make_dummy_hit(
    chunk_id: str,
    doc_id: str = "doc1",
    rank: int = 1,
    score: float = 0.5,
    text: str = "passage text",
) -> RetrievalHit:
    return RetrievalHit(
        chunk_id=chunk_id,
        document_id=doc_id,
        rank=rank,
        score=score,
        strategy=RetrievalStrategy.HYBRID,
        text=text,
        metadata={"article_number": "1", "document_title": "Luat A"},
        retrieval_trace=RetrievalTrace(rrf_score=score),
    )


# --- RESTORED ORIGINAL PHASE-A TESTS ---

def test_jina_config_defaults_and_validation() -> None:
    """1. Test Jina configuration defaults and validation policy."""
    cfg = RerankerConfig(
        backend="jina_native_listwise",
        model_name="jinaai/jina-reranker-v3.5",
        model_revision="e8a93f33f0b22108f8c2364f8484ce3422552fbc",
        device="cpu",
        torch_dtype="float32",
    )
    assert cfg.backend == "jina_native_listwise"
    assert cfg.native_context_cap == 12288
    assert cfg.max_length == 512


def test_jina_reranker_properties() -> None:
    """2. Test Jina reranker provider and model identity properties."""
    cfg = RerankerConfig(
        backend="jina_native_listwise",
        model_name="jinaai/jina-reranker-v3.5",
        model_revision="e8a93f33f0b22108f8c2364f8484ce3422552fbc",
        device="cpu",
        torch_dtype="float32",
    )
    reranker = JinaNativeReranker(cfg)
    assert reranker.provider_name == "transformers-jina"
    assert reranker.model_name == "jinaai/jina-reranker-v3.5"
    assert reranker.model_revision == "e8a93f33f0b22108f8c2364f8484ce3422552fbc"


def test_jina_rerank_scoring_and_ranking_flow() -> None:
    """3. Test Jina reranker basic scoring, sorting, and trace preservation."""
    mock_model = MagicMock()
    mock_model.rerank.return_value = [
        {"index": 0, "relevance_score": 0.10},
        {"index": 1, "relevance_score": 0.90},
    ]

    cfg = RerankerConfig(
        backend="jina_native_listwise",
        device="cpu",
        torch_dtype="float32",
    )
    reranker = JinaNativeReranker(cfg, model_loader=lambda c: mock_model)

    candidates = [
        _make_dummy_hit("chunk_a", rank=1, score=0.5),
        _make_dummy_hit("chunk_b", rank=2, score=0.4),
    ]
    query = RetrievalQuery(
        query_id="q1",
        original_question="test query",
        normalized_question="test query",
        top_k=2,
        candidate_k=2,
    )

    response = reranker.rerank(query, candidates)
    assert len(response.hits) == 2
    assert response.hits[0].chunk_id == "chunk_b"
    assert response.hits[0].rank == 1
    assert response.hits[0].score == 0.90
    assert response.hits[1].chunk_id == "chunk_a"
    assert response.hits[1].rank == 2
    assert response.hits[1].score == 0.10


def test_jina_empty_candidates_handling() -> None:
    """4. Test that empty candidate sequence returns empty response with warning."""
    cfg = RerankerConfig(backend="jina_native_listwise", device="cpu", torch_dtype="float32")
    reranker = JinaNativeReranker(cfg)
    query = RetrievalQuery(
        query_id="q1",
        original_question="test",
        normalized_question="test",
        top_k=5,
        candidate_k=5,
    )
    response = reranker.rerank(query, [])
    assert len(response.hits) == 0
    assert "no_rerank_candidates" in response.warnings


def test_jina_candidate_limit_violation() -> None:
    """5. Test that exceeding max_candidates raises RetrievalError."""
    cfg = RerankerConfig(backend="jina_native_listwise", device="cpu", torch_dtype="float32", max_candidates=2)
    reranker = JinaNativeReranker(cfg)
    candidates = [
        _make_dummy_hit("c1", rank=1),
        _make_dummy_hit("c2", rank=2),
        _make_dummy_hit("c3", rank=3),
    ]
    query = RetrievalQuery(
        query_id="q1",
        original_question="test",
        normalized_question="test",
        top_k=5,
        candidate_k=5,
    )
    with pytest.raises(RetrievalError, match="exceeds reranker limit"):
        reranker.rerank(query, candidates)


def test_jina_non_finite_score_rejected() -> None:
    """6. Test that non-finite scores produced by model raise ModelError."""
    reranker = JinaNativeReranker()
    with pytest.raises(ModelError, match="Non-finite candidate score"):
        reranker._parse_native_results([{"index": 0, "score": float("nan")}], expected_count=1)


def test_jina_missing_or_duplicate_indices_rejected() -> None:
    """7. Test that missing, out-of-bounds, or duplicate indices raise ModelError."""
    reranker = JinaNativeReranker()

    # Duplicate index
    with pytest.raises(ModelError, match="Duplicate candidate index"):
        reranker._parse_native_results(
            [{"index": 0, "score": 0.5}, {"index": 0, "score": 0.6}],
            expected_count=2,
        )

    # Wrong count / Incomplete indices
    with pytest.raises(ModelError, match="returned 1 items|Incomplete coverage"):
        reranker._parse_native_results(
            [{"index": 0, "score": 0.5}],
            expected_count=2,
        )


def test_jina_model_loader_missing_rerank_method() -> None:
    """8. Test that loaded model without callable .rerank() raises BackendInitializationError."""
    cfg = RerankerConfig(backend="jina_native_listwise", device="cpu", torch_dtype="float32")
    reranker = JinaNativeReranker(cfg)

    class _InvalidModel:
        def to(self, d: str) -> Any: return self
        def eval(self) -> Any: return self

    with patch("transformers.AutoModel.from_pretrained", return_value=_InvalidModel()):
        with pytest.raises(BackendInitializationError, match="does not expose a callable .rerank"):
            reranker._load_jina_model(cfg)


def test_jina_parameter_budget_accounting() -> None:
    """9. Test parameter budget verification during model load."""
    cfg = RerankerConfig(
        backend="jina_native_listwise",
        device="cpu",
        torch_dtype="float32",
        expected_parameter_count=596836352,
    )
    reranker = JinaNativeReranker(cfg)

    param_mock = MagicMock()
    param_mock.numel.return_value = 100

    class _ModelWrongParams:
        def to(self, d: str) -> Any: return self
        def eval(self) -> Any: return self
        def rerank(self, *a: Any, **k: Any) -> Any: pass
        def _ensure_tokenizer(self) -> None: pass
        _tokenizer = MagicMock(model_max_length=12288)
        def parameters(self) -> Any: return iter([param_mock])

    with patch("transformers.AutoModel.from_pretrained", return_value=_ModelWrongParams()):
        with pytest.raises(ModelError, match="Jina parameter gate violation"):
            reranker._load_jina_model(cfg)


def test_config_isolation_qwen_vs_jina() -> None:
    """10. Test config isolation outside reranker block."""
    repo_root = Path(__file__).resolve().parents[3]
    ctrl_path = repo_root / "configs" / "uit-dsc-2026-task2-m491-qwen3-dev.example.json"
    jina_path = repo_root / "configs" / "uit-dsc-2026-task2-m491-jina35.example.json"

    ctrl = json.loads(ctrl_path.read_text(encoding="utf-8"))
    jina = json.loads(jina_path.read_text(encoding="utf-8"))

    ctrl_copy = copy.deepcopy(ctrl)
    jina_copy = copy.deepcopy(jina)
    del ctrl_copy["online"]["reranker"]
    del jina_copy["online"]["reranker"]
    assert ctrl_copy == jina_copy


# --- PHASE A.1 / A.2 HARDENING TESTS ---

def test_control_config_git_object_sha() -> None:
    """Test A: Exact control config LF SHA is immutable a38bc642f0..."""
    repo_root = Path(__file__).resolve().parents[3]
    ctrl_path = repo_root / "configs" / "uit-dsc-2026-task2-m491-qwen3-dev.example.json"
    ctrl_bytes = ctrl_path.read_bytes()
    lf_sha = hashlib.sha256(ctrl_bytes.replace(b"\r\n", b"\n")).hexdigest()
    assert lf_sha == "a38bc642f0e4bf006d624ccb1f56721775c5d9aa4a4b24cf82abe5ed52046be6"


def test_config_recursive_equality_and_json_path_diff() -> None:
    """Test B & C & X: JSON path diff contains only online.reranker paths."""
    repo_root = Path(__file__).resolve().parents[3]
    ctrl_path = repo_root / "configs" / "uit-dsc-2026-task2-m491-qwen3-dev.example.json"
    jina_path = repo_root / "configs" / "uit-dsc-2026-task2-m491-jina35.example.json"

    ctrl = json.loads(ctrl_path.read_text(encoding="utf-8"))
    jina = json.loads(jina_path.read_text(encoding="utf-8"))

    def diff_json(p1: Any, p2: Any, path: str = "") -> list[str]:
        diffs = []
        if isinstance(p1, dict) and isinstance(p2, dict):
            for k in set(p1.keys()).union(set(p2.keys())):
                child_path = f"{path}.{k}" if path else k
                if k not in p1 or k not in p2:
                    diffs.append(child_path)
                else:
                    diffs.extend(diff_json(p1[k], p2[k], child_path))
        elif p1 != p2:
            diffs.append(path)
        return diffs

    all_diffs = diff_json(ctrl, jina)
    assert len(all_diffs) == 7
    for p in all_diffs:
        assert p.startswith("online.reranker")


def test_jina_device_and_dtype_contract() -> None:
    """Test N & O: CUDA requires float16, CPU requires float32, unsupported device fails."""
    cfg_cuda = RerankerConfig(backend="jina_native_listwise", device="cuda:0", torch_dtype="float16")
    assert cfg_cuda.torch_dtype == "float16"

    with pytest.raises(ValueError, match="CUDA Jina reranking requires float16"):
        RerankerConfig(backend="jina_native_listwise", device="cuda", torch_dtype="float32")

    cfg_cpu = RerankerConfig(backend="jina_native_listwise", device="cpu", torch_dtype="float32")
    assert cfg_cpu.torch_dtype == "float32"

    with pytest.raises(ValueError, match="CPU Jina reranking requires float32"):
        RerankerConfig(backend="jina_native_listwise", device="cpu", torch_dtype="float16")

    with pytest.raises(ValueError, match="Unsupported device family"):
        RerankerConfig(backend="jina_native_listwise", device="mps", torch_dtype="float32")


def test_jina_native_v4_result_mapping() -> None:
    """Test J & T: Exact native result parser matches V4 fixture with list and object formats."""
    reranker = JinaNativeReranker()

    # V4 fixture 1: list of dicts with relevance_score
    v4_list_dicts = [
        {"index": 0, "relevance_score": 0.95},
        {"index": 1, "relevance_score": 0.42},
    ]
    scores = reranker._parse_native_results(v4_list_dicts, expected_count=2)
    assert scores == [0.95, 0.42]

    # V4 fixture 2: list of objects with index and relevance_score
    class _Item:
        def __init__(self, index: int, relevance_score: float) -> None:
            self.index = index
            self.relevance_score = relevance_score

    v4_list_objs = [_Item(1, 0.12), _Item(0, 0.88)]
    scores_obj = reranker._parse_native_results(v4_list_objs, expected_count=2)
    assert scores_obj == [0.88, 0.12]

    # V4 fixture 3: wrapper object with .results attribute
    class _Wrapper:
        def __init__(self, results: list[Any]) -> None:
            self.results = results

    wrapper = _Wrapper([_Item(0, 0.77), _Item(1, 0.33)])
    scores_wrap = reranker._parse_native_results(wrapper, expected_count=2)
    assert scores_wrap == [0.77, 0.33]


def test_jina_native_strict_output_validation() -> None:
    """Test P, Q, R, S: Strict rejection of malformed index, duplicate index, missing score, non-integer index."""
    reranker = JinaNativeReranker()

    # Non-integer index (boolean)
    with pytest.raises(ModelError, match="Invalid non-integer candidate index"):
        reranker._parse_native_results([{"index": True, "score": 0.5}], expected_count=1)

    # Missing index key
    with pytest.raises(ModelError, match="Missing 'index'"):
        reranker._parse_native_results([{"score": 0.5}], expected_count=1)

    # Missing score key
    with pytest.raises(ModelError, match="Missing relevance_score/score"):
        reranker._parse_native_results([{"index": 0}], expected_count=1)

    # Out of range index
    with pytest.raises(ModelError, match="Out-of-range candidate index"):
        reranker._parse_native_results([{"index": 5, "score": 0.5}], expected_count=1)


def test_jina_loader_tokenizer_and_cap_fail_closed() -> None:
    """Test L & M: Missing _ensure_tokenizer or context cap mismatch fails closed."""
    cfg = RerankerConfig(
        backend="jina_native_listwise",
        device="cpu",
        torch_dtype="float32",
        native_context_cap=12288,
    )

    reranker = JinaNativeReranker(cfg)

    # Missing _ensure_tokenizer
    class _ModelNoTokenizer:
        def to(self, dev: str) -> Any: return self
        def eval(self) -> Any: return self
        def rerank(self, *args: Any, **kwargs: Any) -> Any: pass
        def parameters(self) -> Any: return iter([])

    with patch("transformers.AutoModel.from_pretrained", return_value=_ModelNoTokenizer()):
        with pytest.raises(BackendInitializationError, match="_ensure_tokenizer"):
            reranker._load_jina_model(cfg)

    # Tokenizer model_max_length mismatch
    class _TokenizerBroken:
        @property
        def model_max_length(self) -> int: return 512
        @model_max_length.setter
        def model_max_length(self, val: Any) -> None: pass

    class _ModelBrokenCap:
        def __init__(self) -> None:
            self._tokenizer = _TokenizerBroken()
        def to(self, dev: str) -> Any: return self
        def eval(self) -> Any: return self
        def _ensure_tokenizer(self) -> None: pass
        def rerank(self, *args: Any, **kwargs: Any) -> Any: pass
        def parameters(self) -> Any: return iter([])

    with patch("transformers.AutoModel.from_pretrained", return_value=_ModelBrokenCap()):
        with pytest.raises(BackendInitializationError, match="model_max_length"):
            reranker._load_jina_model(cfg)


def test_jina_rerank_scoring_flow_and_no_grad() -> None:
    """Test K & P: Scoring flow executes with torch.no_grad, logs complete telemetry including parameter count and actual device."""
    mock_model = MagicMock()
    mock_model.rerank.return_value = [
        {"index": 0, "relevance_score": 0.10},
        {"index": 1, "relevance_score": 0.90},
    ]

    cfg = RerankerConfig(
        backend="jina_native_listwise",
        device="cpu",
        torch_dtype="float32",
        expected_parameter_count=596836352,
    )
    reranker = JinaNativeReranker(cfg, model_loader=lambda c: mock_model)
    reranker._actual_device = "cpu"
    reranker._actual_parameter_count = 596836352

    candidates = [
        _make_dummy_hit("chunk_a", rank=1, score=0.5),
        _make_dummy_hit("chunk_b", rank=2, score=0.4),
    ]
    query = RetrievalQuery(
        query_id="q1",
        original_question="cau hoi gi",
        normalized_question="cau hoi gi",
        top_k=2,
        candidate_k=2,
    )

    with patch("legal_agentic_rag.reranking.jina_native._LOGGER.info") as mock_info:
        response = reranker.rerank(query, candidates)

    assert len(response.hits) == 2
    assert response.hits[0].chunk_id == "chunk_b"
    assert response.hits[0].rank == 1
    assert response.hits[0].score == 0.90
    assert response.hits[1].chunk_id == "chunk_a"
    assert response.hits[1].rank == 2
    assert response.hits[1].score == 0.10

    # Test P: Telemetry includes parameter_count and actual_device
    assert mock_info.called
    call_args = mock_info.call_args
    assert call_args[0][0] == "jina_reranking_completed"
    extra = call_args[1].get("extra", {})
    assert extra.get("actual_parameter_count") == 596836352
    assert extra.get("actual_parameter_device") == "cpu"
    assert extra.get("backend") == "jina_native_listwise"
    assert "cau hoi gi" not in str(extra)
