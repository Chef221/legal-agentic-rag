# M49→M53 ARCHITECTURAL RESET — CLOSURE / HANDOFF MEMO

**Project:** `legal-agentic-rag`  
**Closure date:** 2026-09-02  
**Authority role:** Strategic closure + handoff for future chats / Gemini / implementation agents  
**Repository state at reset:** `db0e85b1ce4fe5e0e770cf5d7f71bacade15bbf5` (`origin/main` at time of reset)  
**Purpose:** Preserve only decisions, frozen baselines, closed-work boundaries, and lessons required to start M54 without re-learning or rerunning M49–M53.

---

## 0. How to use this memo

This file is **not** a chronological reconstruction of every experiment, traceback, Kaggle/Modal run, or implementation patch.

It is the compact authority for:

1. what M49→M53 established;
2. what is permanently CLOSED;
3. what artifacts/metrics remain useful as baselines;
4. what architectural lessons were learned;
5. why the project is resetting its retrieval/data foundation;
6. what work is authorized next.

If older chat history conflicts with this memo on strategic status, prefer this memo unless a newer explicit authority supersedes it.

---

# 1. CURRENT STATE

M49→M53 is now considered a **closed architectural lineage**.

It remains valuable as:

- production/reference baseline;
- source of frozen artifacts;
- empirical evidence;
- implementation lessons;
- comparison point for M54.

It is **not** the architecture to continue incrementally patching.

### NEXT AUTHORIZED WORK

> **M54.0 — Raw Legal Corpus Forensics + Preprocessing V2 Design**

M54 starts from the **raw legal corpus/data foundation**, not from another selector/generator patch.

No new full-corpus embedding, retraining, INTERNAL_TEST, Public, or Holdout work is authorized until M54.0 establishes the actual corpus anatomy and a frozen preprocessing design.

---

# 2. KEY DATA AUTHORITY

## 2.1 Official `train.json`

Canonical local authority used throughout the closed lineage:

```text
C:\Users\Nguyen\Downloads\train.json
bytes  = 16,078,892
SHA256 = 2a52501cc065d266f2f832475950bcf1e7c75c386efa9b2f568f251d745f5988
```

Canonical schema:

```json
{
  "<qid>": {
    "question": "...",
    "answer": "..."
  }
}
```

Total organizer records:

```text
7000
```

Frozen project split:

```text
M52_TRAIN                    5300
M52_DEV                       700
M52_INTERNAL_TEST_FROM_TRAIN 1000
```

Frozen TRAIN QID authority:

```text
train_qids_v1_r6.json
rows   = 5300
bytes  = 65,408
SHA256 = 71aa07413572006a59677108e8af4bd4241af996d0aa9af68cdf7327ecc6bec1
```

---

# 3. CLOSED LINEAGE SUMMARY

## 3.1 M49 — canonical production lineage

M49 established the production-style legal RAG architecture that later work inherited:

```text
question
  ↓
hybrid retrieval
  ↓
dense + lexical candidate generation
  ↓
cross-encoder reranking
  ↓
selected evidence
  ↓
generator / production answer pipeline
```

The important surviving lesson is not a single M49 metric; it is that M49 provided the stable production reference against which later retrieval and generator experiments were evaluated.

M49 artifacts remain useful for regression/reference comparison only.

---

## 3.2 M50 O2 — retrieval improvement accepted

M50 O2 improved the retrieval/reranking side sufficiently to be integrated into the canonical repository state.

At architectural reset time, Git was bound to:

```text
HEAD        db0e85b1ce4fe5e0e770cf5d7f71bacade15bbf5
origin/main db0e85b1ce4fe5e0e770cf5d7f71bacade15bbf5
```

Relevant merge history:

```text
db0e85b Merge pull request #1 from Chef221/m50/o2-production-integration-clean
75f2b9a feat: integrate clean-validated M50 O2 Jina projector
```

O2 demonstrated that retrieval quality could be improved, but did **not** eliminate the remaining end-to-end ceiling.

### M50/O2 lesson

> Better retrieval helped, but the post-O2 system still had substantial evidence-ranking/composition headroom.

Do not interpret M50 O2 as proof that the current corpus representation is optimal.

---

# 4. M52 — FULL GENERATOR SFT

## 4.1 What was trained

M52 performed a real full-parameter generator fine-tune on frozen Stage4 contexts.

Base model authority:

```text
ntphuc149/ViLegalQwen3-1.7B-Base
revision = 258c56ed40529cced26fa7fcc3ecc0663e914c18
architecture = Qwen3ForCausalLM
parameters ≈ 1,720,574,976
```

Training population:

```text
5300 TRAIN records
3 epochs
996 optimizer steps
```

Final permanent candidate:

```text
Volume:
m52-stage5-training-checkpoints-v1

Path:
/executions/m52_stage5_full_training_production_canonical_v17_retry02/candidates/step_000996/model.pt
```

Checkpoint semantics:

```text
full model.state_dict()
```

## 4.2 Critical M52 result

Fixed DEV100:

```text
M52 step996 raw METEOR ≈ 0.396707854
E1-only baseline       ≈ 0.430779176
```

Paired diagnostic:

```text
M52 wins vs E1 : 38/100
E1 wins vs M52 : 62/100
oracle max(M52,E1) ≈ 0.483062040
```

### M52 verdict

> **CLOSED — full target-only generator SFT does not beat the evidence baseline.**

### M52 lesson

The generator can sometimes synthesize useful information, but free-form generation frequently **destroys value already present in good evidence** through:

- paraphrasing;
- omission;
- numeric/legal-reference drift;
- over-generation;
- distractor use;
- loss of lexical overlap with reference answers.

This was a core empirical reason to stop treating the generator as the only missing component.

---

# 5. M53 PHASE 0 — EVIDENCE / TARGET ALIGNMENT CENSUS

M53 Phase 0 used frozen TRAIN evidence and official TRAIN answers to measure answer-utility headroom.

Frozen TRAIN artifacts:

```text
train_questions_only.json
bytes  = 984,586
SHA256 = 7de3079ed58a58168f3948cb1b59cdf6ce69dfe0499442c0677e49cb8f556101

train_contexts.jsonl
bytes  = 67,433,585
SHA256 = 0e9c103481eb6be8651a040b6b64ced8882e48f029c2bbfe3eda000f2329959e

train_traces.jsonl
bytes  = 298,329,286
SHA256 = b8b8bd8ec3a17f6aa5d93e4143bdd2e298245f8bfd68a6c2da8eb912a3b38160
```

Phase0 census artifact:

```text
train_5300_census_records_v1.jsonl
bytes  = 94,228,658
SHA256 = 8d80132e04b11a4a52156e91c93d1e9df8aac356ea6f5b3aa095417e578ba673
```

Important finding:

```text
current/frozen E1 mean       ≈ 0.4219 on TRAIN-5300 OOF population
best-single evidence oracle ≈ 0.5200
```

Approximate single-evidence headroom:

```text
~ +0.098
```

### Phase0 lesson

The frozen candidate/evidence set contains much more answer utility than current top-1 selection captures.

However, later M53 experiments showed that this oracle headroom is **not trivially learnable by a small post-hoc selector**.

---

# 6. M53.1 — CLASSICAL REFERENCE-BLIND SELECTOR

Goal:

```text
frozen candidate/evidence features
          ↓
small classical selector
          ↓
choose Ei instead of always E1
```

Result:

```text
E1 baseline             ≈ 0.4219
best classical selector ≈ 0.4299
gain                    ≈ +0.008
```

### Verdict

> **CLOSED / WEAK**

The gain is real enough to be diagnostically interesting, but too small relative to the ~0.098 oracle headroom.

### Lesson

Rank/lexical/metadata signals recover only a small portion of answer-utility headroom.

Do not rescue this branch with more tiny classical models.

---

# 7. M53.2B — SEMANTIC UTILITY SELECTOR

Frozen Qwen embedding run:

```text
model:
Qwen/Qwen3-Embedding-0.6B

unique document/search passages: 36,669
questions:                       5,300
candidate instances:            43,714
```

Frozen embedding outputs:

```text
m53_2b_v2_1_embeddings.npz
bytes  = 337,389,100
SHA256 = dfa6a45f048786034a9201939ee7fb80105200ab82e1c70c5016fda6dfb9f050

m53_2b_v2_1_embedding_index.json
bytes  = 2,301,231
SHA256 = ba992fd40155c5eea8a7aa6d5e0b883716606117ae6ec14c775b7ee34b4e1a69
```

5-fold OOF result:

```text
E1_ALWAYS                         ≈ 0.4219
Semantic Scalar Ridge diagnostic ≈ 0.4299
Semantic Interaction MLP         ≈ 0.3929

MLP delta vs E1                  ≈ -0.029
```

Fold-by-fold primary MLP:

```text
Fold 1: E1 0.4238 → MLP 0.3984
Fold 2: E1 0.4181 → MLP 0.3911
Fold 3: E1 0.4210 → MLP 0.3782
Fold 4: E1 0.4230 → MLP 0.3968
Fold 5: E1 0.4236 → MLP 0.4000
```

### Verdict

> **CLOSED / NO_SEMANTIC_UTILITY_SELECTOR**

### Lesson

Adding frozen semantic similarity/interactions did not unlock the oracle headroom.

Do not create:

```text
M53.2C
another hidden size
another MLP
XGBoost rescue
embedding fine-tune rescue
```

for this closed hypothesis.

---

# 8. M53.3 — SUPPORT-FIRST GENERATOR REPAIR

Hypothesis:

Instead of ordinary target-only SFT:

```text
question + E1..En → answer
```

force the generator to emit an explicit primary support before answering:

```text
question + E1..En
      ↓
E{primary_support}
TRẢ LỜI:
<answer>
```

Starting point:

```text
M52 step996
```

Training:

```text
LoRA r=16
alpha=32
dropout=0.05
q/k/v/o projections only
1 epoch
5300 TRAIN
332 optimizer steps
LR=5e-5
```

Training completed successfully:

```text
332/332 optimizer steps
final partial group = 4 records
training time ≈ 2376.3 s
```

DEV100 generation also completed successfully.

Result:

```text
M53.3 Support-First METEOR : 0.382519
M52 step996                 : 0.396708
E1 baseline                 : 0.430779

delta vs M52                : -0.014189
delta vs E1                 : -0.048260

wins vs E1                  : 40/100
losses vs E1                : 60/100
ties                        : 0
catastrophic losses <=-0.20 : 18/100
big wins >=+0.20            : 5/100
support parse rate          : 100/100
```

Frozen verdict:

```text
M53_3_NO_SIGNAL
```

### Verdict

> **CLOSED / NO_SIGNAL**

### Important interpretation

100% support parse rate proves the model learned the output format.

It does **not** prove that predicted support IDs were correct or that support prediction improved reasoning.

The answer quality became worse than both M52 and E1.

Do not rescue with:

```text
LoRA r=32
different LR
2 epochs
different support-token format
M53.3b
```

---

# 9. WHAT `train.json` REVEALED ABOUT THE ACTUAL TASK

A later direct analysis of the real `train.json` materially changed the architectural interpretation.

## 9.1 This is not ordinary short-answer QA

Typical TRAIN question:

```text
~19 words median
```

Typical reference answer:

```text
~311 words median
```

Reference answers are often long legal compositions rather than concise fact strings.

Common structure:

```text
LEGAL BASIS
   ↓
VERBATIM / NEAR-VERBATIM STATUTORY MATERIAL
   ↓
APPLICATION / CONCLUSION
```

## 9.2 Reference answers are strongly extractive

Approximate TRAIN-5300 observations from direct target-side anatomy:

```text
~88% contain "Điều <number>"
~61% contain "khoản <number>"
~91% show list/clause/statutory-enumeration structure
~67% use "Căn cứ"
~63% contain conclusion markers such as:
     "Theo đó", "Như vậy", "Do đó", "Vì vậy"
```

Many answers largely reproduce statutory material and then add a relatively short conclusion.

### Architectural implication

Free-form regeneration of long legal text is intrinsically risky and often unnecessary.

A strong system should prefer:

```text
retrieve exact legal material
→ select exact spans
→ preserve/copy canonical statutory text
→ generate only the minimal application/conclusion when needed
```

---

## 9.3 Multi-provision / multi-document composition is common

Approximate TRAIN-5300 target-side census:

```text
>=2 distinct Article numbers       ~36.7%
>=3 distinct Article numbers       ~14.5%

>=2 explicit legal instruments     ~38.9%
>=3 explicit legal instruments     ~15.5%
```

Therefore:

> `retrieve one best evidence` is a useful diagnostic but is not sufficient as the final architecture.

The final system likely needs **evidence-set closure / composition**, not merely better top-1 ranking.

---

## 9.4 Temporal / amendment reasoning matters

A meaningful fraction of answers explicitly involve:

```text
sửa đổi
bổ sung
thay thế
hết hiệu lực
effective dates
```

Therefore the legal representation should eventually model relations such as:

```text
AMENDS
REPLACES
REPEALS
EFFECTIVE_FROM
EFFECTIVE_TO
```

not only:

```text
Document → Chapter → Article → Clause → Point
```

---

## 9.5 Raw reference answers contain editorial noise

Observed patterns include:

```text
(Hình từ Internet)
"Tải ... tại đây"
ellipsis placeholders
related-question text appended after the answer
website/editorial residue
```

### Architectural/training implication

Raw full-answer SFT teaches both legal content **and source-site noise**.

Future training targets should not automatically assume the raw reference is an ideal generation style, even though the raw reference remains the official scoring authority.

---

# 10. CORE LESSONS FROM M49→M53

## Lesson A — retrieval improved, but corpus representation may still be the ceiling

O2 improved retrieval without removing the gap between current E1 and the answer-utility oracle.

This suggests that the retrieval **data foundation itself** may be limiting:

- chunk boundaries;
- document identity;
- article/clause structure;
- cross-references;
- amendment/version relationships;
- multi-document composition.

---

## Lesson B — current answerer is too destructive

Empirical ordering on fixed DEV100:

```text
E1 evidence baseline     ≈ 0.430779
M52 full generator SFT   ≈ 0.396708
M53.3 support-first      ≈ 0.382519
```

Granting the generator more freedom to rewrite statutory material did not improve the final metric.

Future architectures should strongly consider:

```text
copy/extract first
generate minimally
```

rather than unrestricted rewriting.

---

## Lesson C — post-hoc tiny selectors cannot recover oracle headroom

M53.1 and M53.2B showed:

```text
~ +0.008 classical gain
~ -0.029 semantic-MLP delta
```

against an oracle headroom near +0.098.

Therefore the missing capability is probably **not** a simple selector over the frozen current representation.

---

## Lesson D — "support before answer" is not the fix

Explicit support-token supervision was mechanically learned but answer quality worsened.

Evidence selection and answer synthesis should be treated as separate architectural functions rather than forcing both into one answer target.

---

## Lesson E — single-evidence oracle is diagnostic, not the final objective

Because many official answers require multiple provisions/documents, future work should study:

```text
minimal evidence sets
coverage of reference legal basis
cross-reference closure
temporal/version closure
```

not only best-single rank.

---

# 11. ARCHITECTURAL RESET DECISION

M49→M53 largely optimized components on top of the same underlying corpus/index representation.

The project will now test whether the primary ceiling is located **below** those components:

> **raw legal corpus preprocessing and legal representation**

Therefore M54 will not begin by:

```text
fine-tuning another selector
training another generator
swapping embeddings on the same old chunks
patching O2
```

M54 begins by understanding and redesigning the legal data layer.

---

# 12. M54 DIRECTION — NOT YET IMPLEMENTATION AUTHORITY

Current working direction, subject to M54.0 corpus forensics:

```text
RAW LEGAL DOCUMENTS
        ↓
canonical document normalization
        ↓
legal structure reconstruction
Document
 └ Chapter
    └ Article
       └ Clause
          └ Point
        ↓
legal relation graph
parent/child
citation
amendment
replacement
repeal
effective dates
        ↓
hierarchical / multi-view legal chunks
        ↓
new lexical + dense index
        ↓
full re-embedding
        ↓
hybrid + graph-aware retrieval
        ↓
reranking / evidence-set closure
        ↓
extractive/compositional answer construction
        ↓
minimal constrained generation where required
```

### Important

This is a **research direction**, not yet a frozen implementation spec.

M54.0 must first inspect the real raw corpus and determine:

- what structure already exists;
- what information has been lost by current preprocessing;
- whether amendments/citations are recoverable;
- duplicate/version prevalence;
- current chunk-boundary failure rate;
- document-level retrieval mismatch;
- evidence-set coverage of official TRAIN answers.

---

# 13. CLOSED-WORK FIREWALL

Unless a future explicit authority reopens them, do **not**:

```text
rerun M52 full SFT
continue M52 step997+
rescue M53.1
rescue M53.2B
create M53.2C
rescue/tune M53.3
change M53.3 LoRA rank/LR/epochs
blindly rerun O2
rerun closed M49 validation work
```

Do **not** spend protected evaluation populations merely to explore M54 ideas.

Specifically:

```text
NO INTERNAL_TEST
NO PUBLIC
NO HOLDOUT
```

until a later explicit gate authorizes them.

DEV exposure must also remain controlled; use TRAIN-only/offline diagnostics first whenever possible.

---

# 14. ARTIFACTS WORTH PRESERVING

Future agents do not need every intermediate file, but should preserve access to at least:

### Data / splits

```text
train.json
train_qids_v1_r6.json
dev_qids_v1_r6.json
internal_test_qids_v1_r6.json
split_manifest_v1_r6.json
```

### Frozen M52/M53 evidence

```text
train_questions_only.json
train_contexts.jsonl
train_traces.jsonl
train_5300_census_records_v1.jsonl
```

### M50/O2 production reference

```text
canonical M50 O2 production config
O2 projector
closed validation/promotion authorities
```

### Generator baseline

```text
M52 step996 model.pt
M52 DEV100 reports
```

### Closed M53 reports

```text
M53.1 selector summary
M53.2B OOF report / embeddings
M53.3 DEV100 report
```

These are baseline/forensic assets.

They are not default starting points for M54 implementation.

---

# 15. RULES FOR FUTURE CHATS / AGENTS

When a new chat or coding agent takes over:

1. Read this memo first.
2. Treat M49→M53 as closed baseline research.
3. Do not propose another M53 rescue experiment before understanding M54.
4. Do not confuse:
   - official reference answer,
   - legal authority text,
   - retrieval representation,
   - generated answer.
5. Synthetic/document summaries may be used as **retrieval representations**, but must never silently become legal authority.
6. Preserve canonical statutory text separately from embedding/search text.
7. Long-running scripts must print visible progress:
   - stage;
   - current/total;
   - percentage;
   - elapsed;
   - ETA when practical.
8. Prefer cheap TRAIN-only/offline gates before GPU or protected evaluation.
9. Do not allow Gemini/implementation agents to launch Modal/Kaggle unless explicitly authorized by the human operator.
10. Do not overengineer speculative infrastructure before an experiment demonstrates signal.

---

# 16. CURRENT AUTHORITY IN ONE SCREEN

```text
STATUS
------
M49     CLOSED baseline lineage
M50 O2  ACCEPTED retrieval improvement / production reference
M52     CLOSED — full generator SFT failed to beat E1
M53.1   CLOSED — weak selector (~+0.008)
M53.2B  CLOSED — semantic selector failed
M53.3   CLOSED — NO_SIGNAL (0.382519)

KEY FIXED DEV100 NUMBERS
------------------------
E1-only                ~0.430779176
M52 step996            ~0.396707854
M53.3 Support-First    ~0.382519
oracle max(M52,E1)     ~0.483062040

TRAIN DIAGNOSTIC
----------------
frozen E1             ~0.4219
best-single oracle    ~0.5200
headroom              ~+0.098

STRATEGIC CONCLUSION
--------------------
The project should stop local optimization of the current stack and investigate
the raw legal corpus / preprocessing / legal representation as the next likely
source of step-change improvement.

NEXT AUTHORIZED WORK
--------------------
M54.0 — Raw Legal Corpus Forensics + Preprocessing V2 Design

DO NOT YET
----------
re-embed full corpus
train new models
burn INTERNAL_TEST
burn Public
burn Holdout
```

---

# 17. FINAL CLOSURE STATEMENT

M49→M53 was not wasted work.

It established, with actual experiments, that:

- retrieval improvements alone do not fully solve the task;
- the current candidate pool contains substantial unused answer utility;
- tiny post-hoc selectors do not recover that utility;
- unrestricted generator fine-tuning can reduce quality below raw evidence;
- support-first generation does not repair the problem;
- the official dataset is strongly extractive/compositional and often multi-provision;
- legal structure, document identity, amendment relations, evidence-set composition, and preservation of exact statutory text are likely first-class requirements.

The next phase should therefore investigate and, if justified by forensics, rebuild the **legal corpus foundation** before further large-scale training.

**M49→M53 CLOSED.  
M54 ARCHITECTURAL RESET AUTHORIZED AT FORENSICS/DESIGN LEVEL ONLY.**
