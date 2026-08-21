"""UIT Data Science Challenge 2026 Task 2 input boundary."""

from legal_agentic_rag.competition.uit_dsc_2026.context_adapter import (
    UitDsc2026ContextAdapter,
)
from legal_agentic_rag.competition.uit_dsc_2026.answer_rendering import (
    render_competition_answer,
)
from legal_agentic_rag.competition.uit_dsc_2026.corpus_ingestion import (
    UitDsc2026CorpusIngestor,
)
from legal_agentic_rag.competition.uit_dsc_2026.batch_inference import (
    CompetitionBatchRunner,
)
from legal_agentic_rag.competition.uit_dsc_2026.loader import (
    ContextSourceIdentity,
    UitDsc2026DataLoader,
)
from legal_agentic_rag.competition.uit_dsc_2026.passage_cleaner import (
    PassageCleaningResult,
    UitDsc2026PassageCleaner,
)
from legal_agentic_rag.competition.uit_dsc_2026.submission import (
    CodabenchSubmissionFormatter,
    load_submission_archive,
)
from legal_agentic_rag.competition.uit_dsc_2026.warmup_scoring import (
    CompetitionWarmupScorer,
)
from legal_agentic_rag.competition.uit_dsc_2026.generator_training import (
    GeneratorSupervisionSplit,
    fixed_dev_sample,
    normalize_supervision_question,
    question_id_digest,
    split_generator_supervision,
)

__all__ = [
    "ContextSourceIdentity",
    "UitDsc2026ContextAdapter",
    "UitDsc2026CorpusIngestor",
    "CompetitionBatchRunner",
    "CodabenchSubmissionFormatter",
    "render_competition_answer",
    "load_submission_archive",
    "CompetitionWarmupScorer",
    "GeneratorSupervisionSplit",
    "fixed_dev_sample",
    "normalize_supervision_question",
    "question_id_digest",
    "split_generator_supervision",
    "PassageCleaningResult",
    "UitDsc2026DataLoader",
    "UitDsc2026PassageCleaner",
]
