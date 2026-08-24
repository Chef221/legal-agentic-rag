"""Multi-session crash-safe Public-1000 inference runner with 12-hour session survival.

Implements:
1. Level 1: Per-question durable append + flush + fsync + atomic manifest replace.
2. Level 2: Cross-session checkpoint bundling and fail-closed resume validation.
3. Session wall-clock budget monitoring and clean session checkpoint export.
4. Submission packaging strictly gated on 100% complete validated checkpoints.
"""

from __future__ import annotations

import datetime
import hashlib
import json
import logging
import os
from pathlib import Path
import shutil
import time
from typing import Any, Callable, Sequence
import zipfile

from legal_agentic_rag import __version__
from legal_agentic_rag.configuration.application import ApplicationConfig
from legal_agentic_rag.contracts import AgentWorkflow
from legal_agentic_rag.exceptions import ArtifactCompatibilityError, DataValidationError
from legal_agentic_rag.runtime.online import OnlineRuntime
from legal_agentic_rag.schemas.agent_state import AgentStopReason
from legal_agentic_rag.schemas.retrieval import RetrievalQuery

_LOGGER = logging.getLogger("public1000_session_runner")

CHECKPOINT_RESULTS_FILENAME = "public1000_checkpoint_results.jsonl"
CHECKPOINT_MANIFEST_FILENAME = "public1000_checkpoint_manifest.json"
CHECKPOINT_JOURNAL_FILENAME = "public1000_checkpoint_journal.jsonl"
CHECKPOINT_AUDIT_FILENAME = "public1000_checkpoint_audit.json"
CHECKPOINT_LATEST_ZIP_FILENAME = "public1000_checkpoint_latest.zip"

FROZEN_AUTHORITY_BINDINGS = {
    "execution_code_authority_commit": "52df95fe67139b27bcb7669b2888d7be52bbd80a",
    "v4_execution_bundle_sha256": "1c77240452774580bd6f8d7a0d5c075567d14cd73c8c4f3c40d38fe76619aa91",
    "generator_tree_sha256": "e6f163aa4f094ac5d943893009a78ba2c62798ed6432eb637887a9843944304b",
    "embedding_model": "Qwen/Qwen3-Embedding-0.6B",
    "embedding_revision": "97b0c614be4d77ee51c0cef4e5f07c00f9eb65b3",
    "jina_model": "jinaai/jina-reranker-v3.5",
    "jina_revision": "e8a93f33f0b22108f8c2364f8484ce3422552fbc",
    "generator_architecture": "Qwen3_5ForConditionalGeneration",
    "generator_loader": "image_text_to_text",
    "exact_runtime_parameter_total": 3405854528,
    "max_competition_params": 4000000000,
    "runtime_dependencies": {
        "transformers": "5.15.0",
        "safetensors": "0.8.0",
        "accelerate": "1.14.0",
        "tokenizers": "0.22.2",
        "huggingface-hub": "1.11.0",
    },
}


def compute_file_sha256(path: Path) -> str:
    """Compute sha256 of file bytes in chunks."""
    hasher = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            hasher.update(chunk)
    return hasher.hexdigest()


def compute_string_sha256(text: str) -> str:
    """Compute sha256 of a string."""
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def compute_qid_set_hash(qids: Sequence[str]) -> str:
    """Compute a canonical SHA256 of the sorted QID set."""
    canonical_list = sorted(str(q) for q in qids)
    joined = "\n".join(canonical_list) + "\n"
    return compute_string_sha256(joined)


class Public1000SessionRunner:
    """Multi-session durable Public-1000 execution controller."""

    def __init__(
        self,
        *,
        app_config: ApplicationConfig,
        working_dir: Path,
        questions_path: Path,
        session_budget_hours: float = 9.5,
        session_id: str | None = None,
        runtime_builder: Callable[[], OnlineRuntime] | None = None,
    ) -> None:
        self.app_config = app_config
        self.working_dir = working_dir
        self.questions_path = questions_path
        self.session_budget_seconds = float(session_budget_hours * 3600.0)
        self.session_id = session_id or f"session_{datetime.datetime.now(datetime.timezone.utc).strftime('%Y%m%d_%H%M%S')}"
        self.runtime_builder = runtime_builder
        self.runtime: OnlineRuntime | None = None

        self.results_path = self.working_dir / CHECKPOINT_RESULTS_FILENAME
        self.manifest_path = self.working_dir / CHECKPOINT_MANIFEST_FILENAME
        self.journal_path = self.working_dir / CHECKPOINT_JOURNAL_FILENAME
        self.audit_path = self.working_dir / CHECKPOINT_AUDIT_FILENAME
        self.latest_zip_path = self.working_dir / CHECKPOINT_LATEST_ZIP_FILENAME

        self.questions: list[tuple[str, str]] = self._load_questions(self.questions_path)
        self.canonical_qids = [q[0] for q in self.questions]
        self.expected_qid_set = set(self.canonical_qids)
        self.expected_total_count = len(self.canonical_qids)
        self.expected_qid_set_hash = compute_qid_set_hash(self.canonical_qids)
        self.question_source_sha256 = compute_file_sha256(self.questions_path)

        self.config_hash = self._compute_config_hash()

    def _compute_config_hash(self) -> str:
        data = self.app_config.model_dump(mode="json")
        return compute_string_sha256(json.dumps(data, sort_keys=True))

    @staticmethod
    def _load_questions(path: Path) -> list[tuple[str, str]]:
        if not path.exists():
            raise FileNotFoundError(f"Questions file not found: {path}")
        raw = json.loads(path.read_text(encoding="utf-8"))
        items: list[tuple[str, str]] = []
        if isinstance(raw, dict):
            for qid, val in raw.items():
                if isinstance(val, dict) and "question" in val:
                    items.append((str(qid), str(val["question"])))
                elif isinstance(val, str):
                    items.append((str(qid), val))
        elif isinstance(raw, list):
            for idx, item in enumerate(raw):
                if isinstance(item, dict):
                    qid = str(item.get("question_id") or item.get("id") or idx)
                    q_text = str(item.get("question") or "")
                    items.append((qid, q_text))
                elif isinstance(item, str):
                    items.append((str(idx), item))
        if not items:
            raise DataValidationError(f"Zero questions loaded from {path}")
        return items

    def _fsync_file(self, stream: Any) -> None:
        stream.flush()
        try:
            os.fsync(stream.fileno())
        except (AttributeError, OSError):
            pass

    def _write_json_atomic(self, path: Path, payload: dict[str, Any]) -> None:
        tmp_path = path.with_suffix(".tmp")
        with tmp_path.open("w", encoding="utf-8") as f:
            json.dump(payload, f, ensure_ascii=False, indent=2)
            f.write("\n")
            self._fsync_file(f)
        os.replace(tmp_path, path)

    def _append_journal_event(self, event_type: str, details: dict[str, Any]) -> None:
        event = {
            "timestamp": datetime.datetime.now(datetime.timezone.utc).isoformat(),
            "session_id": self.session_id,
            "event_type": event_type,
            "details": details,
        }
        with self.journal_path.open("a", encoding="utf-8") as f:
            f.write(json.dumps(event, ensure_ascii=False) + "\n")
            self._fsync_file(f)

    def restore_and_validate_checkpoint(self, checkpoint_archive_path: Path | None = None) -> list[dict[str, Any]]:
        """Restore previous checkpoint if present and strictly validate authority bindings."""
        if checkpoint_archive_path is not None and checkpoint_archive_path.exists():
            _LOGGER.info(f"Restoring checkpoint from archive: {checkpoint_archive_path}")
            self.working_dir.mkdir(parents=True, exist_ok=True)
            with zipfile.ZipFile(checkpoint_archive_path, "r") as zf:
                for info in zf.infolist():
                    if ".." in info.filename or info.filename.startswith("/"):
                        raise ArtifactCompatibilityError(f"Malformed path in checkpoint zip: {info.filename}")
                zf.extractall(self.working_dir)

        if not self.manifest_path.exists() and not self.results_path.exists():
            _LOGGER.info("No prior checkpoint found. Initializing fresh Session 1.")
            self._append_journal_event("FRESH_SESSION_INITIALIZED", {"session_id": self.session_id})
            return []

        if not self.manifest_path.exists() or not self.results_path.exists():
            raise ArtifactCompatibilityError("Incomplete checkpoint: manifest or results file is missing")

        manifest = json.loads(self.manifest_path.read_text(encoding="utf-8"))

        # Verify authority bindings fail-closed
        if manifest.get("question_source_sha256") != self.question_source_sha256:
            raise ArtifactCompatibilityError("Checkpoint question_source_sha256 does not match current public input")

        if manifest.get("expected_total_qid_count") != self.expected_total_count:
            raise ArtifactCompatibilityError("Checkpoint expected_total_qid_count does not match current input count")

        if manifest.get("expected_complete_qid_set_hash") != self.expected_qid_set_hash:
            raise ArtifactCompatibilityError("Checkpoint expected_complete_qid_set_hash does not match current input")

        if manifest.get("application_config_hash") != self.config_hash:
            raise ArtifactCompatibilityError("Checkpoint application_config_hash does not match current resolved config")

        if manifest.get("execution_code_authority_commit") != FROZEN_AUTHORITY_BINDINGS["execution_code_authority_commit"]:
            raise ArtifactCompatibilityError("Checkpoint execution_code_authority_commit mismatch")

        # Verify results.jsonl SHA
        actual_results_sha = compute_file_sha256(self.results_path)
        if manifest.get("results_jsonl_sha256") != actual_results_sha:
            raise ArtifactCompatibilityError(
                f"Checkpoint results JSONL SHA mismatch: {actual_results_sha} != {manifest.get('results_jsonl_sha256')}"
            )

        # Parse every existing record strictly
        existing_records: list[dict[str, Any]] = []
        seen_qids: set[str] = set()

        with self.results_path.open("r", encoding="utf-8") as f:
            for line_no, line in enumerate(f, 1):
                clean_line = line.strip()
                if not clean_line:
                    raise ArtifactCompatibilityError(f"Blank line in checkpoint results at line {line_no}")
                try:
                    record = json.loads(clean_line)
                except json.JSONDecodeError as err:
                    raise ArtifactCompatibilityError(f"Corrupt JSONL in checkpoint at line {line_no}: {err}") from err

                qid = str(record.get("question_id"))
                if not qid or qid not in self.expected_qid_set:
                    raise ArtifactCompatibilityError(f"Unexpected question ID '{qid}' at line {line_no}")
                if qid in seen_qids:
                    raise ArtifactCompatibilityError(f"Duplicate question ID '{qid}' found in checkpoint at line {line_no}")

                seen_qids.add(qid)
                existing_records.append(record)

        if len(existing_records) != manifest.get("completed_question_count"):
            raise ArtifactCompatibilityError(
                f"Checkpoint count mismatch: parsed {len(existing_records)} records, manifest reports {manifest.get('completed_question_count')}"
            )

        _LOGGER.info(
            f"Checkpoint successfully restored and validated: {len(existing_records)}/{self.expected_total_count} QIDs completed."
        )
        self._append_journal_event(
            "CHECKPOINT_RESUMED",
            {
                "session_id": self.session_id,
                "resumed_completed_count": len(existing_records),
                "manifest_sha": compute_file_sha256(self.manifest_path),
            },
        )
        return existing_records

    def _refresh_checkpoint_zip(self) -> None:
        """Create or refresh the durable checkpoint export ZIP."""
        tmp_zip = self.latest_zip_path.with_suffix(".tmp.zip")
        files_to_zip = [
            self.results_path,
            self.manifest_path,
            self.journal_path,
        ]
        if self.audit_path.exists():
            files_to_zip.append(self.audit_path)

        with zipfile.ZipFile(tmp_zip, "w", zipfile.ZIP_DEFLATED) as zf:
            for p in files_to_zip:
                if p.exists():
                    zf.write(p, arcname=p.name)

        os.replace(tmp_zip, self.latest_zip_path)

    def run_session(
        self,
        checkpoint_archive_path: Path | None = None,
        max_questions_in_session: int | None = None,
    ) -> dict[str, Any]:
        """Execute Public-1000 questions within the session wall-clock budget with per-question durability."""
        self.working_dir.mkdir(parents=True, exist_ok=True)
        existing_records = self.restore_and_validate_checkpoint(checkpoint_archive_path)
        completed_qids = {str(r["question_id"]) for r in existing_records}

        pending_items = [(qid, q_text) for qid, q_text in self.questions if qid not in completed_qids]
        if max_questions_in_session is not None:
            pending_items = pending_items[:max_questions_in_session]

        total_pending = len(pending_items)
        _LOGGER.info(f"Session {self.session_id} starting: {len(completed_qids)} completed, {total_pending} pending.")

        if not pending_items:
            _LOGGER.info("All questions already completed in checkpoint!")
            return self._finalize_session(existing_records, status="ALL_QUESTIONS_COMPLETED")

        # Lazily instantiate runtime if not already built
        if self.runtime is None and self.runtime_builder is not None:
            _LOGGER.info("Building OnlineRuntime for session...")
            self.runtime = self.runtime_builder()

        session_start_time = time.perf_counter()
        question_latencies: list[float] = []
        stopped_due_to_budget = False

        failed_qids = [r["question_id"] for r in existing_records if not r.get("success", True)]
        generation_failed_qids = [r["question_id"] for r in existing_records if r.get("stop_reason") == "generation_failed"]
        insufficient_ev_qids = [r["question_id"] for r in existing_records if r.get("insufficient_evidence", False)]

        # Open results file in append mode
        results_file = self.results_path.open("a", encoding="utf-8")

        try:
            for idx, (qid, q_text) in enumerate(pending_items, start=1):
                elapsed_session = time.perf_counter() - session_start_time
                remaining_budget = self.session_budget_seconds - elapsed_session

                avg_latency = (sum(question_latencies) / len(question_latencies)) if question_latencies else 60.0
                max_latency = max(question_latencies) if question_latencies else 180.0
                safe_margin = max(180.0, 2.5 * max_latency)

                if remaining_budget < safe_margin:
                    _LOGGER.warning(
                        f"Session wall-clock budget reached! Remaining: {remaining_budget:.1f}s, Safe Margin: {safe_margin:.1f}s. "
                        f"Stopping cleanly before QID {qid}."
                    )
                    stopped_due_to_budget = True
                    break

                q_start = time.perf_counter()
                _LOGGER.info(f"[{idx}/{total_pending}] Processing QID {qid} (Elapsed: {elapsed_session:.1f}s)...")

                query = RetrievalQuery(
                    query_id=qid,
                    original_question=q_text,
                    normalized_question=q_text,
                    top_k=self.app_config.online.retrieval.top_k,
                    candidate_k=self.app_config.online.retrieval.candidate_k,
                )

                try:
                    if self.runtime is not None:
                        run_result = self.runtime.answer(query)
                        resp = run_result.response
                        state = run_result.state
                        stop_reason = (
                            run_result.stop_reason.value
                            if hasattr(run_result.stop_reason, "value")
                            else str(run_result.stop_reason)
                        )
                        strategy = (
                            resp.retrieval_strategy.value
                            if hasattr(resp.retrieval_strategy, "value")
                            else str(resp.retrieval_strategy)
                        )
                        answer_text = resp.answer if resp else ""
                        insufficient_ev = bool(resp.insufficient_evidence) if resp else False
                        selected_evidence_count = len(state.selected_evidence) if state and state.selected_evidence else 0
                        warnings = resp.warnings if resp else []
                    else:
                        raise RuntimeError("No OnlineRuntime provided for execution")

                    q_time = time.perf_counter() - q_start
                    question_latencies.append(q_time)

                    record_payload = {
                        "question_id": qid,
                        "question": q_text,
                        "answer": answer_text,
                        "stop_reason": stop_reason,
                        "insufficient_evidence": insufficient_ev,
                        "retrieval_strategy": strategy,
                        "selected_evidence_count": selected_evidence_count,
                        "warnings": warnings,
                        "latency_seconds": q_time,
                        "success": (stop_reason != "generation_failed" and not insufficient_ev),
                        "session_id": self.session_id,
                        "completed_at": datetime.datetime.now(datetime.timezone.utc).isoformat(),
                    }

                except Exception as error:
                    q_time = time.perf_counter() - q_start
                    question_latencies.append(q_time)
                    _LOGGER.error(f"Error on QID {qid}: {error}")
                    record_payload = {
                        "question_id": qid,
                        "question": q_text,
                        "answer": "",
                        "stop_reason": "exception",
                        "insufficient_evidence": True,
                        "retrieval_strategy": "failed",
                        "selected_evidence_count": 0,
                        "warnings": [f"exception:{error}"],
                        "latency_seconds": q_time,
                        "success": False,
                        "error": str(error),
                        "session_id": self.session_id,
                        "completed_at": datetime.datetime.now(datetime.timezone.utc).isoformat(),
                    }

                # LEVEL 1 PER-QUESTION DURABLE WRITE
                results_file.write(json.dumps(record_payload, ensure_ascii=False) + "\n")
                self._fsync_file(results_file)

                existing_records.append(record_payload)
                completed_qids.add(qid)

                if not record_payload["success"]:
                    failed_qids.append(qid)
                if record_payload["stop_reason"] == "generation_failed":
                    generation_failed_qids.append(qid)
                if record_payload["insufficient_evidence"]:
                    insufficient_ev_qids.append(qid)

                # Atomically update manifest
                results_sha = compute_file_sha256(self.results_path)
                manifest_payload = {
                    "schema_version": "4.0.0",
                    "execution_code_authority_commit": FROZEN_AUTHORITY_BINDINGS["execution_code_authority_commit"],
                    "v4_execution_bundle_sha256": FROZEN_AUTHORITY_BINDINGS["v4_execution_bundle_sha256"],
                    "application_config_hash": self.config_hash,
                    "generator_tree_sha256": FROZEN_AUTHORITY_BINDINGS["generator_tree_sha256"],
                    "embedding_model": FROZEN_AUTHORITY_BINDINGS["embedding_model"],
                    "embedding_revision": FROZEN_AUTHORITY_BINDINGS["embedding_revision"],
                    "jina_model": FROZEN_AUTHORITY_BINDINGS["jina_model"],
                    "jina_revision": FROZEN_AUTHORITY_BINDINGS["jina_revision"],
                    "generator_architecture": FROZEN_AUTHORITY_BINDINGS["generator_architecture"],
                    "exact_runtime_parameter_total": FROZEN_AUTHORITY_BINDINGS["exact_runtime_parameter_total"],
                    "runtime_dependencies": FROZEN_AUTHORITY_BINDINGS["runtime_dependencies"],
                    "question_source_sha256": self.question_source_sha256,
                    "expected_total_qid_count": self.expected_total_count,
                    "expected_complete_qid_set_hash": self.expected_qid_set_hash,
                    "completed_question_count": len(existing_records),
                    "completed_qids": [r["question_id"] for r in existing_records],
                    "failed_qids": failed_qids,
                    "generation_failed_qids": generation_failed_qids,
                    "insufficient_evidence_qids": insufficient_ev_qids,
                    "results_jsonl_sha256": results_sha,
                    "session_id": self.session_id,
                    "updated_at": datetime.datetime.now(datetime.timezone.utc).isoformat(),
                }
                self._write_json_atomic(self.manifest_path, manifest_payload)

                # Refresh live checkpoint zip
                self._refresh_checkpoint_zip()

        finally:
            results_file.close()

        status = "SESSION_CHECKPOINT_COMPLETE" if stopped_due_to_budget else (
            "ALL_QUESTIONS_COMPLETED" if len(existing_records) == self.expected_total_count else "SESSION_BATCH_COMPLETE"
        )
        return self._finalize_session(existing_records, status=status)

    def _finalize_session(self, existing_records: list[dict[str, Any]], status: str) -> dict[str, Any]:
        """Produce audit report, final export zip, and prominent user banner."""
        results_sha = compute_file_sha256(self.results_path) if self.results_path.exists() else ""
        manifest_sha = compute_file_sha256(self.manifest_path) if self.manifest_path.exists() else ""

        completed_count = len(existing_records)
        remaining_count = self.expected_total_count - completed_count
        first_qid = existing_records[0]["question_id"] if existing_records else "N/A"
        last_qid = existing_records[-1]["question_id"] if existing_records else "N/A"
        completed_set_hash = compute_qid_set_hash([r["question_id"] for r in existing_records])

        audit_payload = {
            "session_id": self.session_id,
            "status": status,
            "completed_count": completed_count,
            "remaining_count": remaining_count,
            "total_expected": self.expected_total_count,
            "first_completed_qid": first_qid,
            "last_completed_qid": last_qid,
            "completed_qid_set_hash": completed_set_hash,
            "results_jsonl_sha256": results_sha,
            "manifest_sha256": manifest_sha,
            "timestamp": datetime.datetime.now(datetime.timezone.utc).isoformat(),
        }
        self._write_json_atomic(self.audit_path, audit_payload)
        self._refresh_checkpoint_zip()

        zip_sha = compute_file_sha256(self.latest_zip_path)
        zip_size = self.latest_zip_path.stat().st_size

        audit_payload["checkpoint_zip_path"] = str(self.latest_zip_path)
        audit_payload["checkpoint_zip_size"] = zip_size
        audit_payload["checkpoint_zip_sha256"] = zip_sha

        # Print prominent user export UX
        print("\n" + "=" * 80)
        print(" PUBLIC-1000 SESSION CHECKPOINT READY")
        print("=" * 80)
        print(f" Session ID:           {self.session_id}")
        print(f" Status:               {status}")
        print(f" Completed Questions:  {completed_count} / {self.expected_total_count}")
        print(f" Remaining Questions:  {remaining_count}")
        print(f" First Completed QID:  {first_qid}")
        print(f" Last Completed QID:   {last_qid}")
        print(f" Results SHA256:       {results_sha}")
        print(f" Checkpoint ZIP:       {self.latest_zip_path}")
        print(f" Checkpoint ZIP Size:  {zip_size:,} bytes")
        print(f" Checkpoint SHA256:    {zip_sha}")
        print("-" * 80)
        if status == "ALL_QUESTIONS_COMPLETED":
            print(" ALL 1000 QUESTIONS COMPLETED. READY FOR SUBMISSION PACKAGING.")
        else:
            print(" SAFE TO STOP THIS KAGGLE SESSION: YES")
            print(" NEXT SESSION ACTION:")
            print(" 1. Download 'public1000_checkpoint_latest.zip' before stopping session.")
            print(" 2. In the next session, upload it as an opaque .zip.bin input.")
            print(" 3. Resume from Cell 1.")
        print("=" * 80 + "\n")

        return audit_payload

    def package_final_submission(self, output_dir: Path) -> Path:
        """Package official Codabench submission.json and submission.zip strictly gated on 100% complete records."""
        existing_records = self.restore_and_validate_checkpoint()

        if len(existing_records) != self.expected_total_count:
            raise DataValidationError(
                f"Cannot package submission: incomplete checkpoint ({len(existing_records)}/{self.expected_total_count} completed)"
            )

        completed_qids = [str(r["question_id"]) for r in existing_records]
        if set(completed_qids) != self.expected_qid_set:
            missing = self.expected_qid_set - set(completed_qids)
            extra = set(completed_qids) - self.expected_qid_set
            raise DataValidationError(f"Cannot package submission: missing {len(missing)} QIDs, extra {len(extra)} QIDs")

        if len(completed_qids) != len(set(completed_qids)):
            raise DataValidationError("Cannot package submission: duplicate QIDs detected in checkpoint")

        output_dir.mkdir(parents=True, exist_ok=True)
        submission_json_path = output_dir / "submission.json"
        submission_zip_path = output_dir / "submission.zip"

        submission_dict: dict[str, dict[str, str]] = {}
        for r in existing_records:
            qid = str(r["question_id"])
            ans = str(r.get("answer") or "")
            submission_dict[qid] = {"answer": ans}

        submission_json_path.write_text(
            json.dumps(submission_dict, ensure_ascii=False, indent=2), encoding="utf-8"
        )

        with zipfile.ZipFile(submission_zip_path, "w", zipfile.ZIP_DEFLATED) as zf:
            zf.write(submission_json_path, arcname="submission.json")

        _LOGGER.info(f"Official submission packaged: {submission_zip_path} ({submission_zip_path.stat().st_size:,} bytes)")
        return submission_zip_path
