# M49.1 Public Result

## Result

The retained public candidate is M49.1. The official-compatible scoring run over
1,000 public questions produced:

| Candidate | ROUGE-L | METEOR |
|---|---:|---:|
| M48 control | 0.3631401334440235 | 0.2685876695455311 |
| M49.1 | 0.473653736 | 0.382772249 |
| Absolute delta | +0.110513602556 | +0.114184579454 |

METEOR improved by about 42.5% relative to M48. ROUGE-L improved by about
30.4% relative to M48.

## Reproducibility identity

- public question SHA-256:
  `5f68ca901cb20798559538bef60fa7c32bd7d0df59f5bf31a37eb220c9e00df5`;
- M49.1 result JSONL SHA-256:
  `e444bd7cc7c10cfbd568f38bd3eb33bdabdd7213d13a837fd66f9a76670d4523`;
- M49.1 config hash:
  `160c49dab9f595b2a0d49e9d2b1aa56ce0a0c49885fe7615d3c96bc86831ff16`;
- merged M49 generator tree SHA-256:
  `e6f163aa4f094ac5d943893009a78ba2c62798ed6432eb637887a9843944304b`;
- validated submission ZIP SHA-256:
  `fe226ea3d56d2d11623910ab3f52f05463c0fd88f8e13ca66568fbc877a911d0`;
- code version reported by the batch: `0.45.0`.

The submission contained only UTF-8 `submission.json`, exactly 1,000 official
question IDs in source order, no empty answer and no internal `[E#]` marker.

## Batch audit

- 996 responses ended as `answer_verified`; four ended at the bounded retry
  limit with `insufficient_evidence=true`;
- insufficient-evidence IDs: `17789`, `160333`, `118413`, `119219`;
- retrieval strategy: 974 `hybrid_rerank`, 22 `graph`, four `hybrid`;
- 88 responses used successful semantic synthesis;
- 908 responses used the deterministic top-evidence backend;
- 900 responses carried `generator_model_error_fallback`;
- 80 responses used supported-claim salvage;
- 90 grounding repairs were attempted and eight repair model calls failed;
- zero final citation-verification failures were present.

Summed per-record latency was about 23.81 hours. Median total latency was about
80.7 seconds and median generation latency about 70.5 seconds. These sums include
several resumed Kaggle sessions and should not be interpreted as one uninterrupted
wall-clock run.

## Interpretation

M49 was trained with response-only loss on official question-to-answer pairs.
M49.1 asks the model for grounded plain text containing `[E#]` markers. A model
answer without valid markers is rejected by the parser and goes through bounded
recovery, normally ending at the verbatim top-evidence fallback. This contract
mismatch explains the 900 generator fallbacks.

The fallback is not automatically a quality failure: the M49.1 public score shows
that strong retrieval plus official-text overlap is highly competitive for the
current METEOR/ROUGE scorer. A successor must therefore preserve M49.1 as an
immutable control and test generator-contract changes on the frozen dev split.
Do not remove the fallback merely to improve a warning count.

Raw questions, answers, batch JSONL, model weights, indexes and submission files
are intentionally not committed. They must be restored from official inputs or
the team's private artifact storage and checked against the hashes above.
