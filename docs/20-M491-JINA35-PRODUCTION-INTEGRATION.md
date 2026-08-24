# M49.1-JINA35 Production Integration Record

## 1. Status
- **Current State:** `PHASE_A_IMPLEMENTATION_COMPLETE`
- **Research Validation:** `PASS` (Clean100 unexposed benchmark $+0.037$ ROUGE-L, $+0.039$ METEOR)
- **Production Integration:** `AUTHORIZED`
- **Production Promotion:** `PENDING_MECHANICAL_INTEGRATION_GATE`
- **Execution Date:** 2026-08-24

---

## 2. Exact Baseline Provenance
- **Repository Root:** `lkey07/legal-agentic-rag`
- **Base Commit:** `10681c8c05008432cd1c7170cd3917f4317c1c69`
- **Control Config:** `configs/uit-dsc-2026-task2-m491-qwen3-dev.example.json`
  - **Raw Checksum:** `becedefa3d9e86887bd6435cc7c3b6daffd6a76aa64af06e2c3376e19cdeab19`
  - **LF Checksum:** `b7d495f6adbec5626dace689c4b1444b3a480c78ef2b11f0abc4dd95025b841c`
- **Control Status:** 100% byte-identical, immutable control baseline.

---

## 3. Exact Candidate Identity
- **Model Identifier:** `jinaai/jina-reranker-v3.5`
- **Revision:** `e8a93f33f0b22108f8c2364f8484ce3422552fbc`
- **Parameter Count:** `596,836,352`
- **Loader:** `transformers.AutoModel.from_pretrained(..., trust_remote_code=True)`
- **Native Context Cap:** `12,288` tokens
- **Input Serialization:** `build_legal_rerank_text(hit)`

---

## 4. Files Changed / Created

### New Source Files Created
- `src/legal_agentic_rag/reranking/jina_native.py` (Implements `JinaNativeReranker` adhering to `Reranker` protocol)
- `src/legal_agentic_rag/reranking/factory.py` (Implements `build_reranker()` dispatch)
- `configs/uit-dsc-2026-task2-m491-jina35.example.json` (Selectable Jina production configuration)
- `scripts/m491_jina35_mechanical_validation.py` (Two-gate non-gold validation runner)

### Modified Source Files
- `src/legal_agentic_rag/configuration/online.py` (Added `backend`, `native_context_cap`, and `expected_parameter_count` to `RerankerConfig`)
- `src/legal_agentic_rag/reranking/__init__.py` (Exported `JinaNativeReranker` and `build_reranker`)
- `src/legal_agentic_rag/runtime/online.py` (Updated `OnlineRuntimeFactory` to instantiate reranker via `build_reranker()`)

### New Tests Created
- `tests/unit/reranking/test_factory.py`
- `tests/unit/reranking/test_jina_native.py`
- `tests/test_m491_jina35_mechanical_validation.py`

### Documentation Created / Updated
- `docs/19-M491-RERANKER-RESEARCH-STORY.md`
- `docs/20-M491-JINA35-PRODUCTION-INTEGRATION.md`
- `docs/00-START-HERE.md`
- `docs/08-DESIGN-DECISIONS.md`

---

## 5. Files Explicitly Unchanged
- `src/legal_agentic_rag/reranking/cross_encoder.py` (Untouched, exact legacy/Qwen cross-encoder)
- `src/legal_agentic_rag/reranking/legal_context.py` (Untouched, identical legal text formatting)
- `src/legal_agentic_rag/generation/*` (Untouched, identical evidence selector & generator)
- `src/legal_agentic_rag/retrieval/*` (Untouched, identical BM25, Dense, Hybrid, and RRF fusion)
- `configs/uit-dsc-2026-task2-m491-qwen3-dev.example.json` (Untouched, byte-identical control config)

---

## 6. Config Diff
Comparing `uit-dsc-2026-task2-m491-qwen3-dev.example.json` vs `uit-dsc-2026-task2-m491-jina35.example.json`:
```json
<<<< Control Config (Qwen3)
    "reranker": {
      "model_name": "Qwen/Qwen3-Reranker-0.6B",
      "model_revision": "e61197ed45024b0ed8a2d74b80b4d909f1255473",
      "device": "cuda",
      "torch_dtype": "float16",
      "batch_size": 4,
      "max_length": 2048,
      "max_candidates": 40,
      "relationship_candidate_k": 20,
      "local_files_only": false,
      "input_mode": "legal_context",
      "prompt_name": "vietnamese_legal_retrieval",
      "instruction": "Given a Vietnamese legal question, rank authoritative legal passages by whether they directly answer the requested relation, scope, conditions, exceptions, authority, deadlines and amounts."
    }
====
>>>> Jina Config (Jina v3.5)
    "reranker": {
      "backend": "jina_native_listwise",
      "model_name": "jinaai/jina-reranker-v3.5",
      "model_revision": "e8a93f33f0b22108f8c2364f8484ce3422552fbc",
      "device": "cuda",
      "torch_dtype": "float16",
      "batch_size": 4,
      "max_length": 2048,
      "max_candidates": 40,
      "relationship_candidate_k": 20,
      "local_files_only": false,
      "input_mode": "legal_context",
      "native_context_cap": 12288,
      "expected_parameter_count": 596836352
    }
```
Outside the `reranker` block, both JSON configurations are byte-for-byte identical.

---

## 7. Runtime Contract

| Dimension | Control Reranker (Qwen3) | Candidate Reranker (Jina v3.5) |
|---|---|---|
| **Backend Name** | `sentence_transformers_cross_encoder` | `jina_native_listwise` |
| **Loader Method** | `sentence_transformers.CrossEncoder` | `transformers.AutoModel.from_pretrained` (`trust_remote_code=True`) |
| **Inference Call** | `model.predict(pairs, batch_size=...)` | `model.rerank(query, documents, top_n=..., return_embeddings=False)` |
| **Input Structure** | List of `(query, doc_text)` pairs | Single query string + list of 40 doc strings |
| **Context Limit** | `max_length = 2048` | `_tokenizer.model_max_length = 12288` |
| **Device Execution** | Pointwise batching | Single-pass native listwise cross-attention |
| **VRAM Policy** | Default PyTorch allocation | `gc.collect() + torch.cuda.empty_cache()` per QID |

---

## 8. Parameter Budget

| Component | Model Identifier | Parameter Count |
|---|---|---|
| **Embedding** | `Qwen/Qwen3-Embedding-0.6B` | 600,000,000 |
| **Candidate Reranker** | `jinaai/jina-reranker-v3.5` | 596,836,352 |
| **Generator** | `Qwen/Qwen2.5-3B-Instruct` | 2,114,914,432 |
| **Candidate Total** | Full Pipeline Stack | **3,311,750,784** |
| **Control Total** | Full Control Pipeline Stack | 3,466,000,000 |
| **Competition Limit** | UIT DSC 2026 Hard Cap | **< 4,000,000,000** |
| **Headroom Under Cap** | Remaining Allowable Parameters | **688,249,216** (17.2%) |

---

## 9. Unit & Integration Tests
- `tests/unit/reranking/test_factory.py`: 3/3 passed
- `tests/unit/reranking/test_jina_native.py`: 10/10 passed
- `tests/unit/reranking/test_cross_encoder.py`: 9/9 passed
- `tests/test_m491_jina35_mechanical_validation.py`: 1/1 passed
- Full repository test suite: **710 passed, 1 skipped, 4 warnings in 65s**

---

## 10. Mechanical Parity Gate (Gate A) — NOT RUN YET
- **Script:** `scripts/m491_jina35_mechanical_validation.py --gate A`
- **Inputs:** `clean100_shared_candidate_pools.json`, `clean100_jina_reranked.json`
- **Objective:** Verify production `JinaNativeReranker` produces 100/100 top-1 matches and $<10^{-3}$ score delta against frozen Phase-1 outputs.
- **Status:** Prepared for execution on Kaggle environment in Phase B.

---

## 11. Full M49.1 Runtime T4 Smoke (Gate B) — NOT RUN YET
- **Script:** `scripts/m491_jina35_mechanical_validation.py --gate B`
- **Inputs:** `configs/uit-dsc-2026-task2-m491-jina35.example.json`, `clean100_questions_only.json`
- **Objective:** Verify un-quantized float16 coexistence of Embedding (600M) + Jina (596.8M) + Generator (2.115B) on 16GB Tesla T4 GPU.
- **Status:** Prepared for execution on Kaggle environment in Phase B.

---

## 12. Promotion Decision
- **Decision:** `PENDING_MECHANICAL_INTEGRATION_GATE`
- The candidate is fully integrated as a selectable backend. Production promotion (making it the default configuration or submitting to official evaluation) will occur ONLY after successful execution of Gates A and B.

---

## 13. Evidence & Artifact Ledger

| Artifact | Path / Identifier | SHA-256 Checksum |
|---|---|---|
| **Clean100 Validation Evidence** | `m491_jina35_clean100_validation_evidence.zip` | `f8ee98878b8cf6d6268879957e4ca4b95f3d21e7bdec9c474f4c71a03d8b6f39` |
| **Authority Bundle v4** | `m491_jina35_clean100_authority_bundle_v4.zip` | `b3b02ecdb8851adb8f73cce8d7985cdadf99afdcf02a8d9fc1a50debd666fa97` |
| **Control Config** | `configs/uit-dsc-2026-task2-m491-qwen3-dev.example.json` | `b7d495f6adbec5626dace689c4b1444b3a480c78ef2b11f0abc4dd95025b841c` |
| **Jina Config** | `configs/uit-dsc-2026-task2-m491-jina35.example.json` | `0507309ea83685e13d9876797a151b72e022dfca4b9101d2da4ca1078652ecab` |

---

## 14. Next Action & Future Execution Ledger Template

### Execution Instruction Rule
Any script executing longer than 30 seconds must expose progress telemetry:
- Stage name
- Processed / total count
- Percentage complete
- Elapsed time & ETA
- Current VRAM allocation
- Checkpoint / last meaningful event

### Future Execution Ledger Entries (Append Below Upon Kaggle Run)
```text
Date/Time: <UTC timestamp>
Execution Environment: Kaggle GPU (Tesla T4)
Commit: <git rev-parse HEAD>
Config SHA: <SHA-256 of config used>
Model Identity/Revision: jinaai/jina-reranker-v3.5 @ e8a93f33f0b22108f8c2364f8484ce3422552fbc
Input Population: Clean100 frozen candidate pools (Gate A) / Clean100 questions only (Gate B)
Reference Status: UNEXPOSED (Zero references)
Command: python scripts/m491_jina35_mechanical_validation.py --gate ALL
Log Artifact: reports/mechanical_validation/mechanical_validation_summary.json
Output Hashes:
Gate Result: GATE A = <PASS/FAIL>, GATE B = <PASS/FAIL>
Interpretation:
Decision:
Next Action:
```


---

## 15. Phase A.1 Contract Hardening & Verification Ledger

- **Status:** COMPLETED / PRE-GPU GATES HARDENED
- **Worktree:** `C:\legal-agentic-rag-m491-jina35`
- **Branch:** `m491/jina35-production-integration`
- **Root Commit:** `10681c8c05008432cd1c7170cd3917f4317c1c69`
- **Control Config SHA:** `a38bc642f0e4bf006d624ccb1f56721775c5d9aa4a4b24cf82abe5ed52046be6` (100% byte-identical to 10681c8 Git object).
- **Candidate Config Isolation:** 100% verified (0 differing paths outside `online.reranker.*`).
- **Gate A JSONL Schema:** Aligned to `clean100_shared_candidate_pools.jsonl` (SHA `45a9bd97...`) and `clean100_jina_reranked.jsonl` (SHA `eaafc39d...`).
- **Gate B API Contract:** Strict `RetrievalQuery` request validation against `OnlineRuntime.answer()`.
- **Runtime Safety:** Explicit `torch.no_grad()`, fail-closed dtype validation, context cap 12288 check, durable disk logging via `--log-path`.
- **Pytest Status:** 451 passed, 1 skipped, 0 failed.


---

## 16. Phase A.2 Pre-GPU Mechanical Closure Ledger

- **Status:** COMPLETED / MECHANICAL HARNESS HARDENED
- **Worktree:** `C:\legal-agentic-rag-m491-jina35`
- **Branch:** `m491/jina35-production-integration`
- **Root Commit:** `10681c8c05008432cd1c7170cd3917f4317c1c69`
- **Gate A Full-K Call:** `top_k=len(candidate_hits)` (40 candidates), evaluated across full-K, top-10, top-1, and scores.
- **Gate B Result Mapping:** Strict `AgentRunResult.response` and `AgentRunResult.state` attribute extraction.
- **Strict Authority Validation:** Mandatory `clean100_phase1_manifest.json` SHA verification, line-by-line field validation, unique QIDs/chunk IDs, 40-candidate parity.
- **Strict Native Result Parser:** Rejects missing/duplicate/non-integer indices, non-finite scores, unexpected item formats, and incomplete coverage.
- **Test Suite Status:** 464 passed, 1 skipped (0 failed). All 10 original Phase A tests restored and retained.


---

## 17. Phase A.3 Pre-GPU Checkpoint & Authority Freeze Ledger

- **Status:** IMMUTABLE PRE-GPU CHECKPOINT FROZEN
- **Worktree:** `C:\legal-agentic-rag-m491-jina35`
- **Branch:** `m491/jina35-production-integration`
- **Pre-GPU Code Authority Commit:** `90de9a9d813df87432bc9183f8edebd4ed1f0b24`
- **Root Control Commit:** `10681c8c05008432cd1c7170cd3917f4317c1c69`
- **Control Config Status against 10681c8:** UNCHANGED (`a38bc642f0e4bf006d624ccb1f56721775c5d9aa4a4b24cf82abe5ed52046be6`)
- **Exact Parameter Total:** `3,311,750,784` (Compliant under 4B cap)
- **Execution Authority Bundle:** `m491_jina35_production_gate_v1.zip` (SHA `be32b11284fd627750d0afa17723e625522d1cf5c26dac5f58715e128d8ca711`)
- **Reference Material in Bundle:** NO (0 reference answers)
- **Kaggle Execution Cell File:** `kaggle/m491_jina35_production_gate_v1_cells.md`
