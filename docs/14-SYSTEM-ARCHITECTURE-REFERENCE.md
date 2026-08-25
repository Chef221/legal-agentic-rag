# System architecture and I/O reference

## 1. Current snapshot

| Item | Current value |
|---|---|
| Official contexts | 8,532 |
| M45 legal chunks | 385,962 |
| Sparse index | SQLite FTS5 BM25 |
| Dense model | Qwen3-Embedding-0.6B, 1,024 dimensions (595,776,512 params) |
| Retained branch reranker | `Qwen/Qwen3-Reranker-0.6B` (`e61197ed45024b0ed8a2d74b80b4d909f1255473`) |
| Evaluated candidate reranker | `jinaai/jina-reranker-v3.5` (`e8a93f33f0b22108f8c2364f8484ce3422552fbc`, 596,836,352 params) |
| Generator | Qwen3.5-2B; M49 official-only merged revision (2,213,241,664 params) |
| Total Active Learned Stack (M49.1-JINA35) | 3,405,854,528 params (< 4.0B cap; +594,145,472 headroom) |
| Retained control | M48 (METEOR 0.2685876695, ROUGE-L 0.3631401334) |
| Prior baseline | M49.1 Qwen3-Reranker (METEOR 0.382772249, ROUGE-L 0.473653736) |
| **Current canonical baseline** | **M49.1-JINA35** (reconciled repository baseline) |
| **Best measured Codabench score** | **METEOR 0.406858976, ROUGE-L 0.496260842** |
| **Evaluated submission.zip SHA256** | `f11af3c9a4571ff8e8997716b39484bcf69f636b54af7c815ba44756ac2d9200` |

Raw datasets and large artifacts are intentionally absent from Git.

## 2. High-level architecture

```text
Official corpus
  -> competition adapter
  -> unified legal documents
  -> cleaning/parsing/chunking
  -> BM25/vector/graph artifact stores

Official question
  -> query understanding
  -> sparse+dense retrieval
  -> RRF fusion and reranking (Qwen3 / Jina v3.5)
  -> bounded graph/retry workflow
  -> evidence/context builder
  -> generator
  -> claim/citation verification
  -> AnswerResponse
  -> competition renderer
  -> submission.json
```

The core depends on contracts, not concrete backends. Dataset, embedding, vector,
reranker, graph, generator, verifier and tracing implementations are composed at
runtime boundaries.

## 3. Offline I/O

### Input

The UIT DSC adapter accepts the official ZIP/directory whose context records have
raw `id`, `link`, `passage` and optional `name`. Raw names do not escape the
competition adapter.

### Main stages

1. duplicate/schema audit;
2. canonical revision computation;
3. unified document mapping;
4. normalization and controlled HTML cleaning;
5. legal hierarchy parsing;
6. article-first chunking with clause/token fallback;
7. BM25 persistence;
8. dense embedding/vector persistence;
9. relationship/graph persistence;
10. vector-serving metadata and full validation.

### Output root

```text
dataset_manifest.json
audit/corpus_audit.json
normalized_documents/
cleaned_documents/
legal_blocks/
legal_chunks/manifest.json
bm25/index.sqlite3
bm25/manifest.json
vector/vectors.npy
vector/chunks.jsonl
vector/manifest.json
vector_serving/metadata.sqlite3
vector_serving/manifest.json
graph/
relationships/
build_validation_full_corpus.json
```

Every persisted artifact has lineage/config/model metadata. Online startup rejects
missing or incompatible artifacts.

## 4. Online I/O

### Query input

The online runtime receives a validated `LegalQuery`/competition question. It does
not receive raw corpus records and cannot mutate offline artifacts.

### Retrieval output

Retrieval returns ranked unified chunks with document/article metadata, branch
scores, RRF contributions, reranker score, graph path when applicable and artifact
identity.

### Generation input

The context builder receives selected ranked evidence and constructs a bounded
prompt. It preserves legal identifiers, numbers, negation, exceptions and source
provenance.

### Answer output

Internal `AnswerResponse` contains answer text, evidence/citations, warnings,
insufficient-evidence state and metadata. The competition renderer converts it to
the score-facing answer string and removes verified internal `[E#]` markers.

## 5. Retained profiles

### M45 offline foundation

M45 builds the Qwen3 dense index and serving metadata. Rebuild is required after a
corpus, chunking, embedding-model or embedding-revision change.

### M48 control

M48 uses base Qwen3.5-2B with competition-reference prompting, compact structured
output, bounded repair/salvage and deterministic top-evidence fallback. It is the
retained non-SFT control.

### M49 training lineage

M49 uses the group-safe official train partition to train QLoRA for one epoch with
response-only loss, then merges the adapter into the base fp16 generator. It does
not fine-tune retrieval/reranking and does not rebuild M45.

### M49.1 baseline

M49.1 uses the same M49 weights with plain-text evidence markers, deterministic
repetition controls, exact claim deduplication and the M48 bounded recovery path.
Its public batch is the prior measured baseline.

### M49.1-JINA35 evaluated candidate

M49.1-JINA35 upgraded reranking to `jinaai/jina-reranker-v3.5`, validated Hotfix V1 raw-identity preservation and Hotfix V2 dual-source anchored explicit-document matching on Kaggle hardware, producing the latest evaluated Codabench submission (METEOR `0.406858976`, ROUGE-L `0.496260842`). The Public-1000 execution is **CLOSED** (1000/1000 strict valid). Source code reconciliation into the repository canonical baseline is **COMPLETE**.

## 6. Model inventory & Parameter compliance

| Role | Identity | Revision | Parameters | Provenance |
|---|---|---|---:|---|
| Embedding | `Qwen/Qwen3-Embedding-0.6B` | `97b0c614be4d77ee51c0cef4e5f07c00f9eb65b3` | 595,776,512 | `docs/artifacts/m491-jina35-parameter-budget-authority.json` (Safetensors header) |
| Reranker (Evaluated) | `jinaai/jina-reranker-v3.5` | `e8a93f33f0b22108f8c2364f8484ce3422552fbc` | 596,836,352 | `docs/artifacts/m491-jina35-parameter-budget-authority.json` (Safetensors header) |
| Generator (Merged) | `Qwen/Qwen3.5-2B` (M49 SFT) | `e6f163aa4f094ac5d943893009a78ba2c62798ed6432eb637887a9843944304b` | 2,213,241,664 | `docs/artifacts/m491-jina35-parameter-budget-authority.json` (Instantiated model numel) |
| **Total Stack** | — | — | **3,405,854,528** | `< 4,000,000,000` competition cap (+594,145,472 headroom) |

## 7. Package ownership

| Package | Responsibility |
|---|---|
| `competition/uit_dsc_2026` | raw adapters, official split, batch/submission boundary |
| `schemas`, `contracts` | stable typed I/O between modules |
| `configuration` | validated paths, model identities and bounds |
| `offline` | normalize, clean, parse, chunk |
| `indexing` | persistent BM25/vector/graph stores |
| `embeddings` | document/query embedding provider |
| `retrieval` | sparse/dense retrieval, fusion and strategies |
| `reranking` | bounded cross-encoder / listwise reranking |
| `generation` | context, generation, grounding and citations |
| `agent`, `tools` | closed bounded orchestration |
| `runtime` | offline/online composition |
| `serving` | CLI, API, UI and config loading |
| `evaluation` | metrics, comparison and regression gates |

## 8. Batch and submission

The batch directory contains `results.jsonl`, `manifest.json` and state needed to
resume. Each record is written after one question completes. A batch can resume
only when source/config/code identity is compatible.

`legal-rag-submit` requires complete exact ID coverage and writes a deterministic
ZIP containing only UTF-8 `submission.json` with shape:

```json
{
  "question-id": {"answer": "Vietnamese prose"}
}
```

## 9. CLI surface

```text
legal-rag-build-competition
legal-rag-prepare-serving
legal-rag-batch
legal-rag-submit
legal-rag-score-warmup
legal-rag-validate
legal-rag-evaluate
legal-rag-compare
legal-rag-serve
```

## 10. Configuration and errors

Paths, top-k, candidate-k, token limits, retry limits, models, revisions, device
and timeouts live in validated configuration. Production modules classify config,
data, artifact, retrieval, model and timeout errors and log structured trace fields.

## 11. Known technical risks

1. M49.1 still falls back to top evidence for most questions because SFT output and
   evidence-marker parsing are not fully aligned.
2. The top-evidence fallback is a strong lexical-metric baseline; removing it may
   reduce METEOR.
3. Public runtime is slow because generation dominates median latency.
4. Exact organizer model-registration and final environment limits must be
   reconfirmed before private submission.
5. The official scorer dependency/resource environment may change and must be
   checked by checksum rather than metric name.

## 12. Definition of a reproducible candidate

A candidate is reproducible only when it records:

- source commit;
- config hash;
- official data hashes and canonical corpus revision;
- artifact manifests;
- exact model names/revisions and total parameters;
- split seed and dev ID hash;
- scorer checksum/environment;
- batch/result/submission hashes;
- metrics, warning counts and latency summary.
