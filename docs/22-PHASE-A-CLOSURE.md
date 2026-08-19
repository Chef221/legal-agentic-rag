# Phase-A Current-System Census Closure Report

## 1. Executive Summary and Phase Status

- **Status**: **PHASE A: CLOSED**
- **Next Authorized Step**: **PHASE B1A: PAIRED GRAPH-ROUTING BEHAVIORAL ABLATION (NOT YET IMPLEMENTED)**
- **Historical Reference Baseline**: M49.6 (`9b0cd0b`) remains the frozen reliability baseline.
- **Repository Package Version**: Bumped to `0.50.6` following permanent ToolError contract hardening and Phase-A formal closure.

This document formally records the complete results, diagnostics, scoring benchmarks, and architecture findings from the Phase-A Current-System Census executed on Kaggle using the official competition data.

---

## 2. Census Authority, Runtime Environment, and Evidence Integrity

### 2.1 Code and Benchmark Authority
- **Census Source Snapshot**: `923557c3a01654aca979ec917a32993af01ad0d0` ("Add Phase A current-system census runbook")
- **Package Version during Census**: `0.50.5` + temporary ToolError contract patch
- **Engineering Benchmark Set**: `development.json` (canonical M44 development partition)
- **Record Count**: 991 questions
- **Benchmark Question SHA-256**: `8678791de5194cbac073732a59541cbba8336aad74ff384410e2025c92bd0bd8`
- **Important Scientific Caveat**: The 991-question development set is a historical engineering benchmark, not an untouched holdout. Metrics recorded here represent current-system baseline performance and census telemetry, not generalizable unseen-data claims.

### 2.2 Actual Kaggle Hardware Placement
During initial Kaggle diagnostics on dual T4 GPUs, running Qwen structured generation on GPU0 alongside vector search and the cross-encoder reranker produced a CUDA illegal memory access. Isolated diagnostic tests confirmed:
- GPU0 + SDPA / eager: FAIL (CUDA illegal memory access under shared memory pressure)
- GPU1 + SDPA / eager: PASS (clean execution)

Consequently, hardware placement was partitioned:
- **GPU0 (`cuda:0`)**: Dense vector search, Multilingual-E5-Small embedding provider, cross-encoder reranker (`mmarco-mMiniLMv2-L12-H384-v1`).
- **GPU1 (`cuda:1`)**: Qwen/Qwen2.5-3B-Instruct generator.

> [!NOTE]
> This was a hardware-placement-only configuration change (`online.generation.device: "cuda:1"`). No algorithms, prompt structures, model revisions, or pipeline parameters were modified.

### 2.3 Evidence Archive and Config Hashes
- **Evidence Archive Filename**: `phase-a-current-system-census-final-evidence.zip`
- **Evidence Archive SHA-256**: `df05a401599c43a28e39136d72b225841b242d10a40dc5bc475b9be6ed86be8b`
- **Baseline Runtime Config SHA-256**: `23d154feafa46300215e8498e9738d345c48122739e377dcab43e9e5475b1a31`
- **Actual GPU-Split Config SHA-256**: `03a32009c0dc9a68ac93538710ec741b7a7e68a8ef9e160116ecc6bcb76d64fc`
- **Submission ZIP SHA-256**: `040143457941c23156d655d694d60d3199621630bead23268c1ee5760e0dc22d`
- **Official Scorer SHA-256**: `4fac914203d325445a666c0c566530c962ba95b843e1988e4f37057c47447891`

---

## 3. ToolError Reliability Bug and Permanent Hardening

### 3.1 Root Cause Analysis
During the initial census execution on Kaggle, the batch processor halted at ordinal 73 (question ID `111779`) due to an uncaught Pydantic `ValidationError` at the tool boundary:
1. Model generation produced an initial malformed draft missing required fields.
2. Missing-field correction was attempted, but the subsequent regeneration failed with a terminal `JSON_DECODE_ERROR`.
3. The generator preserved earlier diagnostic issue codes (`MISSING_REQUIRED_FIELD`).
4. `ToolRegistry` attempted to construct a `ToolError` containing both `generation_failure_code = JSON_DECODE_ERROR` and `generation_schema_issue_codes = [MISSING_REQUIRED_FIELD]`.
5. The `ToolError` model validator required that schema issue diagnostics be accompanied strictly by `SCHEMA_VALIDATION_ERROR`, rejecting the valid correction-chain diagnostic envelope and crashing the entire batch process.

### 3.2 Permanent Validation Logic
The validator in `src/legal_agentic_rag/schemas/tools.py` was hardened to explicitly permit earlier schema diagnostics when they originate from an attempted missing-field correction chain (`generation_missing_field_correction_attempted = True`):
- **Original `tools.py` LF SHA-256**: `c4788daee3438a51973dde76138a2372fb53ba551b82ffa6b88cc60a2067b666`
- **Patched `tools.py` LF SHA-256**: `c67025e0d6f9a4643adc6ea153ac402176f186e3e76de0045babba004045cdaf`
- **Permanent Regression Coverage**: `tests/unit/schemas/test_tools.py` covers valid correction chains, free-floating schema error rejection, non-model error validation, and outcome consistency.

---

## 4. Phase-A Official-Compatible Evaluation Scores

Scored against canonical `development.json` (991 questions) using the official competition scoring contract:

| Metric | Authoritative Value | Display Value (Rounded) |
|---|---|---|
| **Exact Match** | `0.0` | `0.000000` |
| **METEOR** (Primary Metric) | `0.0980790959` | `0.098079` |
| **ROUGE-L** (Secondary Metric) | `0.1871225729` | `0.187123` |
| **Question Count** | `991` | `991` |
| **Evaluated Records** | `991` | `991` |
| **NLTK Version** | `3.7` | `3.7` |

---

## 5. Phase-A Reliability, Latency, and Diagnostics

### 5.1 Agent Stop Reasons and Outcome Distribution
Out of 991 records evaluated end-to-end:
- **`answer_verified`**: 806 (81.33%)
- **`generation_failed`**: 177 (17.86%)
- **`citation_verification_failed`**: 7 (0.71%)
- **`max_retry_reached`**: 1 (0.10%)
- **`insufficient_evidence_count`**: 185 (18.67%)
- **`generator_model_error_count`**: 10 (1.01%)
- **`generator_model_error_unclassified`**: 0
- **`retrieval_model_error_count`**: 0

### 5.2 Agent Execution Latency
- **Count**: 991
- **Mean Latency**: 15,732.10 ms (~15.73 s)
- **Median (p50) Latency**: 14,820.32 ms (~14.82 s)
- **95th Percentile (p95) Latency**: 24,604.78 ms (~24.60 s)
- **Maximum Latency**: 54,748.44 ms (~54.75 s)

---

## 6. Granular Component Diagnostics

### 6.1 Structured Generation Failures and Recovery
- **Terminal Generation Failure Codes** (10 total model errors):
  - `claim_boundary_mismatch`: 4
  - `json_decode_error`: 5
  - `schema_validation_error`: 1
- **Generation Schema Issue Codes**:
  - `claim_limit_exceeded`: 2
  - `missing_claim_field`: 4
  - `missing_required_field`: 4
- **Deterministic Schema Recovery**: Attempted 5, Succeeded 1, Failed 4 (outcomes: `not_recoverable` = 3, `revalidation_failed` = 1, `succeeded` = 1)
- **Missing-Field Correction**: Attempted 7, Succeeded 4, Failed 3 (outcomes: `succeeded` = 4, `failed` = 3)

### 6.2 Citation and Numeric Verification
- **Citation Verification Present**: 813 / 991
- **Citation Verification Failed**: 7
- **Claim Errors Detected**:
  - `numeric_mismatch`: 45
  - `negation_mismatch`: 3
  - `claim_too_short`: 1
  - `insufficient_lexical_support`: 1
  - `missing_inline_evidence`: 1
  - `no_linked_evidence`: 1
- **Numeric Repair**: Attempted 44, Succeeded 18, Failed 26
- **Supported-Claim Salvage**: Attempted 1, Succeeded 1

### 6.3 Evidence Context Selection
- **Context Selection Trace Present**: 991 / 991
- **Selected Evidence Chunks**: 4,677 total
- **Selection Reasons**:
  - `selected`: 4,677
  - `document_cap` (capped at 2 per document): 1,180
  - `max_evidence` (capped at 5 total evidence items): 2,071
- **Top Warning Frequencies**:
  - `effect_status_unknown:E1..E5`: 4,677 (expected because UIT context lacks structured effect metadata)
  - `context_max_evidence_reached:3`: 586
  - `generator:insufficient_evidence`: 167
  - `model_reported_insufficient_evidence`: 145
  - `numeric_repair_failed`: 26
  - `numeric_repair_succeeded`: 18
  - `generator:model_error`: 10

---

## 7. Graph Retrieval & Adaptive Routing Empirical Census

### 7.1 Empirical Routing Behavior on 991 Benchmark Records
- **Relationship Heuristic Keyword Matches**: 22 / 991 (2.22%)
- **`GRAPH_SEARCH` Invocations**: 22 / 991
- **Final `retrieval_strategy == "graph"`**: 22 / 991
- **Relationship Match Without Graph Attempt**: 0
- **Graph Attempt Without Relationship Cue**: 0
- **Fallback to `hybrid_rerank`**: 0 / 22 (100% of graph attempts terminated on Attempt 1)

### 7.2 Substring Cue Breakdown
- `"bổ sung"`: 10
- `"hướng dẫn"`: 6
- `"sửa đổi"`: 6
- `"thay thế"`: 2

### 7.3 Observational Subgroup Breakdown
- **Graph-Routed Group (22 records)**:
  - `answer_verified`: 18 (81.82%)
  - `generation_failed`: 4 (18.18%)
  - Mean Latency: 9,402.79 ms
- **Non-Graph Group (969 records)**:
  - `answer_verified`: 788 (81.32%)
  - `generation_failed`: 173 (17.85%)
  - `citation_verification_failed`: 7 (0.72%)
  - `max_retry_reached`: 1 (0.10%)
  - Mean Latency: 15,875.87 ms

> [!WARNING]
> **Non-Causal Scientific Interpretation**:
> The 81.82% verified rate in the graph group versus 81.32% in the non-graph group does NOT prove that graph retrieval improves quality. These are observational, unpaired subgroups with distinct legal queries. The true causal effect of graph routing versus direct `hybrid_rerank` will be measured exclusively via the Phase B1A paired counterfactual ablation.

---

## 8. Static Architecture Audit & Verdict on Graph Path

### 8.1 Confirmed Static Findings
1. **Empty Official Relationships**: `relationships.jsonl` has `record_count = 0` by design (`src/legal_agentic_rag/runtime/competition_offline.py:246-285`).
2. **Zero-Edge Graph**: `graph.json` contains 8,532 nodes and 0 edges; build validation strictly asserts `record_count == 0` (`src/legal_agentic_rag/runtime/competition_offline.py:476-477`).
3. **Unconditional Registration**: `GRAPH_SEARCH` is registered as an active tool regardless of edge count (`src/legal_agentic_rag/tools/retrieval.py:116-125`).
4. **Classifier Priority**: `QueryIntent.RELATIONSHIP` is evaluated first in `QueryUnderstandingService._intent()` (`src/legal_agentic_rag/retrieval/query_understanding.py:163-164`).
5. **Adaptive Route Prepending**: Router prepends `[GRAPH, HYBRID_RERANK, HYBRID]` ahead of `strategy_order` (`src/legal_agentic_rag/agent/router.py:85-103`).
6. **Candidate Pool Truncation Bug**: `graph_search` requests `min(graph_seed_chunk_k, candidate_k - 1) = min(20, 39) = 20` hybrid seeds. Because traversal returns 0 edges, the reranker evaluates **only 20 candidates instead of 40** (`src/legal_agentic_rag/retrieval/graph.py:63-104`).
7. **Early Termination**: `RuleBasedContextGrader` evaluates the 8 reranked seed chunks as `is_sufficient = True`, terminating Attempt 1 and never executing `hybrid_rerank` (`src/legal_agentic_rag/agent/workflow.py:236-256`).

### 8.2 Architecture Verdict
- **Generic Graph Capability**: **`KEEP_GENERIC_ONLY`** (retain `GraphBackend`, `AdjacencyGraphBackend`, and `GraphExpandedRetriever` as reusable library components).
- **UIT Competition Graph Integration**: **`REMOVE_COMPETITION_PATH CANDIDATE`** (decoupling competition pipeline from 0-edge graph artifact, subject to Phase B1A experimental confirmation).

---

## 9. Phase-B Entry Boundary

- **Phase A**: **FORMALLY CLOSED**
- **Phase B**: **NOT YET IMPLEMENTED**
- **Next Step**: **PHASE B1A — PAIRED GRAPH-ROUTING BEHAVIORAL ABLATION**
  - Evaluate the exact 22 graph-routed questions using `hybrid_rerank` directly at `candidate_k = 40`.
  - Compare paired $\Delta\text{METEOR}$, $\Delta\text{ROUGE-L}$, win/tie/loss rates, generation failure rates, and latency under strictly identical model and generation parameters.
