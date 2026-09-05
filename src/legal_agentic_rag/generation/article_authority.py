"""Deterministic full-Article authority store and answer assembler for M55."""

from __future__ import annotations

import hashlib
import json
import logging
from pathlib import Path
from typing import Any

from legal_agentic_rag.configuration.online import ArticleAnswerConfig
from legal_agentic_rag.exceptions import DataValidationError
from legal_agentic_rag.schemas.answering import (
    AnswerResponse,
    Citation,
    ContextBuildResult,
)
from legal_agentic_rag.schemas.retrieval import (
    RetrievalQuery,
    RetrievalStrategy,
)

_LOGGER = logging.getLogger(__name__)

DEFAULT_ABSTENTION_TEXT = (
    "Hệ thống chưa tìm thấy căn cứ pháp luật đủ rõ trong dữ liệu hiện có "
    "để trả lời chắc chắn."
)


class ArticleAuthorityStore:
    """Read-only exact Article authority lookup store verified against SHA256 and count."""

    def __init__(
        self,
        records: dict[tuple[str, str], str],
        *,
        sha256: str,
        record_count: int,
    ) -> None:
        self._records = records
        self._sha256 = sha256
        self._record_count = record_count

    @classmethod
    def from_jsonl(
        cls,
        path: str | Path,
        *,
        expected_sha256: str,
        expected_record_count: int,
    ) -> ArticleAuthorityStore:
        """Load, parse, hash-verify, and count-verify a full-Article JSONL lookup file."""
        file_path = Path(path)
        if not file_path.is_file():
            raise FileNotFoundError(f"Article authority lookup file not found: {file_path}")

        hasher = hashlib.sha256()
        records: dict[tuple[str, str], str] = {}
        count = 0

        with open(file_path, "r", encoding="utf-8") as f:
            for line_number, line in enumerate(f, start=1):
                count += 1
                # Normalized LF hashing for cross-platform newline consistency
                normalized_line = line.rstrip("\r\n") + "\n"
                hasher.update(normalized_line.encode("utf-8"))

                try:
                    data = json.loads(line)
                except Exception as exc:
                    raise DataValidationError(
                        f"Malformed JSON at line {line_number} of {file_path}: {exc}"
                    ) from exc

                if not isinstance(data, dict):
                    raise DataValidationError(
                        f"Record at line {line_number} is not a JSON object"
                    )

                doc_id = data.get("document_id")
                if not isinstance(doc_id, str) or not doc_id.strip():
                    raise DataValidationError(
                        f"Empty or missing document_id at line {line_number}"
                    )

                art_id = data.get("article_identity")
                if not isinstance(art_id, str) or not art_id.strip():
                    raise DataValidationError(
                        f"Empty or missing article_identity at line {line_number}"
                    )

                text = data.get("article_text")
                if not isinstance(text, str) or not text.strip():
                    raise DataValidationError(
                        f"Empty or missing article_text at line {line_number}"
                    )

                key = (doc_id.strip(), art_id.strip().lower())
                if key in records:
                    raise DataValidationError(
                        f"Duplicate exact article key {key} at line {line_number}"
                    )

                records[key] = text

        if count != expected_record_count:
            raise DataValidationError(
                f"Article authority record count mismatch: expected {expected_record_count}, "
                f"got {count}"
            )

        computed_sha = hasher.hexdigest()
        if computed_sha != expected_sha256.strip().lower():
            raise DataValidationError(
                f"Article authority SHA256 mismatch: expected {expected_sha256}, "
                f"got {computed_sha}"
            )

        _LOGGER.info(
            "article_authority_store_loaded",
            extra={
                "record_count": count,
                "sha256": computed_sha,
                "path": str(file_path),
            },
        )
        return cls(records, sha256=computed_sha, record_count=count)

    def get(self, document_id: str, article_number: str) -> str | None:
        """Resolve exact Article text by document and normalized article identity."""
        key = (str(document_id).strip(), str(article_number).strip().lower())
        return self._records.get(key)

    def has(self, document_id: str, article_number: str) -> bool:
        """Check whether an exact Article is present in the authority store."""
        return self.get(document_id, article_number) is not None

    def __len__(self) -> int:
        return len(self._records)

    @property
    def sha256(self) -> str:
        return self._sha256

    @property
    def record_count(self) -> int:
        return self._record_count


class FirstKFullArticleAnswerAssembler:
    """Deterministic assembler for First-K full-Article answers with structural fallback."""

    def __init__(
        self,
        store: ArticleAuthorityStore,
        *,
        config: ArticleAnswerConfig | None = None,
    ) -> None:
        self._store = store
        self._config = config or ArticleAnswerConfig(enabled=True)

    def assemble(
        self,
        *,
        query: RetrievalQuery,
        strategy: RetrievalStrategy,
        context: ContextBuildResult,
    ) -> AnswerResponse:
        """Assemble First-K full Articles or apply deterministic top-3 structural fallback."""
        ordered_articles: list[dict[str, Any]] = []
        seen_keys: set[tuple[str, str]] = set()
        warnings: list[str] = []

        for ev in context.evidence:
            doc_id = ev.document_id
            art_num = ev.article_number
            if not doc_id or not art_num:
                warnings.append(f"evidence_missing_identity:{ev.evidence_id}")
                continue

            auth_text = self._store.get(doc_id, art_num)
            if auth_text is None:
                warnings.append(f"article_resolution_failed:{doc_id}:{art_num}")
                continue

            key = (doc_id.strip(), str(art_num).strip().lower())
            if key not in seen_keys:
                seen_keys.add(key)
                ordered_articles.append(
                    {
                        "key": key,
                        "document_id": doc_id,
                        "article_number": art_num,
                        "text": auth_text,
                        "first_evidence": ev,
                    }
                )
                if len(ordered_articles) >= self._config.max_articles:
                    break

        if ordered_articles:
            answer_text = "\n\n".join(item["text"] for item in ordered_articles)
            citations = [
                Citation(
                    evidence_id=item["first_evidence"].evidence_id,
                    chunk_id=item["first_evidence"].chunk_id,
                    document_id=item["first_evidence"].document_id,
                    document_title=item["first_evidence"].document_title,
                    document_number=item["first_evidence"].document_number,
                    article_number=item["first_evidence"].article_number,
                    source_url=item["first_evidence"].source_url,
                )
                for item in ordered_articles
            ]
            metadata = {
                "answer_path": "m55_first_k_full_article",
                "included_articles": [
                    {
                        "document_id": item["document_id"],
                        "article_identity": item["key"][1],
                    }
                    for item in ordered_articles
                ],
                "source_evidence_ids": [
                    item["first_evidence"].evidence_id for item in ordered_articles
                ],
                "requested_max_articles": self._config.max_articles,
                "resolved_article_count": len(ordered_articles),
                "lookup_sha256": self._store.sha256,
                "structural_fallback_used": False,
            }
            validated_response = AnswerResponse(
                question=query.original_question,
                answer=answer_text,
                citations=citations,
                insufficient_evidence=False,
                warnings=list(dict.fromkeys([*context.warnings, *warnings])),
                retrieval_strategy=strategy,
                trace_id=query.query_id,
                metadata=metadata,
            )
            # Article authority text is already hash/count verified; exact A4 semantics
            # require preservation of raw whitespace. This path intentionally overrides
            # only the already-validated response's answer value via model_copy.
            if validated_response.answer != answer_text:
                return validated_response.model_copy(update={"answer": answer_text})
            return validated_response

        # Zero reconstructable Articles: structural fallback or abstention
        if context.evidence:
            fallback_k = self._config.structural_fallback_max_evidence
            fallback_evidence = context.evidence[:fallback_k]
            fallback_texts = [
                ev.text.strip() for ev in fallback_evidence if ev.text.strip()
            ]
            answer_text = "\n\n".join(fallback_texts)
            citations = [
                Citation(
                    evidence_id=ev.evidence_id,
                    chunk_id=ev.chunk_id,
                    document_id=ev.document_id,
                    document_title=ev.document_title,
                    document_number=ev.document_number,
                    article_number=ev.article_number,
                    source_url=ev.source_url,
                )
                for ev in fallback_evidence
            ]
            metadata = {
                "answer_path": "m55_first_k_full_article",
                "included_articles": [],
                "source_evidence_ids": [ev.evidence_id for ev in fallback_evidence],
                "requested_max_articles": self._config.max_articles,
                "resolved_article_count": 0,
                "lookup_sha256": self._store.sha256,
                "structural_fallback_used": True,
            }
            validated_response = AnswerResponse(
                question=query.original_question,
                answer=answer_text,
                citations=citations,
                insufficient_evidence=False,
                warnings=list(
                    dict.fromkeys(
                        [*context.warnings, *warnings, "structural_fallback_used"]
                    )
                ),
                retrieval_strategy=strategy,
                trace_id=query.query_id,
                metadata=metadata,
            )
            # Article authority / structural fallback text is already verified; exact A4
            # semantics require preservation of raw whitespace. Override only the
            # already-validated response's answer value via model_copy.
            if validated_response.answer != answer_text:
                return validated_response.model_copy(update={"answer": answer_text})
            return validated_response

        # Completely empty evidence: explicit abstention
        return AnswerResponse(
            question=query.original_question,
            answer=DEFAULT_ABSTENTION_TEXT,
            citations=[],
            insufficient_evidence=True,
            warnings=list(
                dict.fromkeys(
                    [*context.warnings, *warnings, "no_evidence_available"]
                )
            ),
            retrieval_strategy=strategy,
            trace_id=query.query_id,
            metadata={
                "answer_path": "m55_first_k_full_article",
                "included_articles": [],
                "source_evidence_ids": [],
                "requested_max_articles": self._config.max_articles,
                "resolved_article_count": 0,
                "lookup_sha256": self._store.sha256,
                "structural_fallback_used": False,
            },
        )