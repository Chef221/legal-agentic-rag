"""Strict Dual-T4 Public-1000 multi-process orchestration layer.

Implements true OS-process replicated inference parallelism across 2 physical GPUs
(CUDA_VISIBLE_DEVICES=0 and 1) with deterministic canonical partitioning (canonical_index_mod_2_v1),
per-worker durable fsync checkpoints, race-safe live progress telemetry, and fail-closed resume validation.
"""

from __future__ import annotations

import datetime
import importlib
import json
import logging
import multiprocessing as mp
import os
from pathlib import Path
import subprocess
import sys
import threading
import time
from typing import Any, Callable, Sequence
import zipfile

from legal_agentic_rag.competition.uit_dsc_2026.public1000_session_runner import (
    CHECKPOINT_AUDIT_FILENAME,
    CHECKPOINT_JOURNAL_FILENAME,
    CHECKPOINT_LATEST_ZIP_FILENAME,
    CHECKPOINT_MANIFEST_FILENAME,
    CHECKPOINT_RESULTS_FILENAME,
    FROZEN_AUTHORITY_BINDINGS,
    Public1000SessionRunner,
    compute_file_sha256,
    compute_qid_set_hash,
    compute_string_sha256,
)
from legal_agentic_rag.configuration.application import ApplicationConfig
from legal_agentic_rag.exceptions import ArtifactCompatibilityError, DataValidationError
from legal_agentic_rag.runtime.online import OnlineRuntime

_LOGGER = logging.getLogger("dual_session_runner")

DUAL_GPU_MANIFEST_FILENAME = "public1000_dual_gpu_manifest.json"
DUAL_GPU_AUDIT_FILENAME = "public1000_dual_gpu_audit.json"
DUAL_GPU_CHECKPOINT_ZIP_FILENAME = "public1000_dual_gpu_checkpoint_latest.zip"
PARTITION_STRATEGY_V1 = "canonical_index_mod_2_v1"


def get_dual_gpu_telemetry() -> dict[str, dict[str, Any]]:
    """Query live telemetry for GPU 0 and GPU 1 without creating a CUDA context."""
    stats = {
        "gpu_0": {"util_pct": "N/A", "vram_used_mb": 0.0, "vram_total_mb": 0.0, "vram_peak_mb": 0.0},
        "gpu_1": {"util_pct": "N/A", "vram_used_mb": 0.0, "vram_total_mb": 0.0, "vram_peak_mb": 0.0},
    }
    try:
        res = subprocess.run(
            ["nvidia-smi", "--query-gpu=index,utilization.gpu,memory.used,memory.total", "--format=csv,noheader,nounits"],
            capture_output=True,
            text=True,
            timeout=1.5,
        )
        if res.returncode == 0:
            lines = res.stdout.strip().splitlines()
            for line in lines:
                parts = [p.strip() for p in line.split(",")]
                if len(parts) >= 4:
                    idx_str = parts[0]
                    key = f"gpu_{idx_str}"
                    if key in stats:
                        stats[key]["util_pct"] = f"{parts[1]}%"
                        stats[key]["vram_used_mb"] = float(parts[2])
                        stats[key]["vram_total_mb"] = float(parts[3])
    except Exception:
        pass

    return stats


def _count_durable_records_safe(results_path: Path) -> int:
    """Read durable JSONL lines safely without crashing during in-progress file appends."""
    if not results_path.exists():
        return 0
    count = 0
    try:
        with results_path.open("r", encoding="utf-8", errors="replace") as f:
            for line in f:
                clean_line = line.strip()
                if clean_line:
                    try:
                        rec = json.loads(clean_line)
                        if "question_id" in rec:
                            count += 1
                    except Exception:
                        pass
    except Exception:
        pass
    return count


def _worker_process_entrypoint(
    worker_id: int,
    worker_dir_str: str,
    qfile_str: str,
    config_dict: dict[str, Any],
    session_budget_seconds: float,
    session_id: str,
    device_str: str,
    max_questions: int | None,
    checkpoint_archive_str: str | None,
    custom_builder_target: tuple[str, str] | None,
    repo_root_str: str | None,
    result_queue: Any,
) -> None:
    """True OS-process worker entrypoint.

    Sets CUDA_VISIBLE_DEVICES as the very first line before importing/initializing PyTorch.
    """
    # 1. Strict process-level CUDA isolation
    os.environ["CUDA_VISIBLE_DEVICES"] = str(device_str)

    # Ensure repository root is in child sys.path
    if repo_root_str and repo_root_str not in sys.path:
        sys.path.insert(0, repo_root_str)

    pid = os.getpid()
    _LOGGER.info(f"[WORKER {worker_id} START] PID={pid}, CUDA_VISIBLE_DEVICES={os.environ.get('CUDA_VISIBLE_DEVICES')}")

    try:
        worker_dir = Path(worker_dir_str)
        qfile_path = Path(qfile_str)
        app_cfg = ApplicationConfig.model_validate(config_dict)

        # 2. Determine runtime builder
        if custom_builder_target is not None:
            mod_name, fn_name = custom_builder_target
            mod = importlib.import_module(mod_name)
            builder_fn = getattr(mod, fn_name)
            runtime_builder = builder_fn(worker_id)
        else:
            def runtime_builder():
                from legal_agentic_rag.runtime.online import OnlineRuntimeFactory
                factory = OnlineRuntimeFactory(app_cfg)
                return factory.build()

        runner = Public1000SessionRunner(
            app_config=app_cfg,
            working_dir=worker_dir,
            questions_path=qfile_path,
            session_budget_hours=session_budget_seconds / 3600.0,
            session_id=session_id,
            runtime_builder=runtime_builder,
        )

        chk_path = Path(checkpoint_archive_str) if checkpoint_archive_str else None
        audit_res = runner.run_session(
            checkpoint_archive_path=chk_path,
            max_questions_in_session=max_questions,
        )

        if result_queue is not None:
            result_queue.put({
                "worker_id": worker_id,
                "pid": pid,
                "success": True,
                "status": audit_res.get("status"),
                "completed_count": audit_res.get("completed_question_count", 0),
                "audit": audit_res,
            })
    except Exception as err:
        _LOGGER.error(f"[WORKER {worker_id} FATAL ERROR] PID={pid}: {err}", exc_info=True)
        if result_queue is not None:
            result_queue.put({
                "worker_id": worker_id,
                "pid": pid,
                "success": False,
                "error": str(err),
            })
        sys.exit(1)


def _canonicalize_for_hash(obj: Any) -> Any:
    if isinstance(obj, dict):
        return {k: _canonicalize_for_hash(v) for k, v in sorted(obj.items())}
    elif isinstance(obj, (list, tuple, set)):
        items = [_canonicalize_for_hash(x) for x in obj]
        if all(isinstance(x, (str, int, float, bool)) for x in items):
            return sorted(items)
        return items
    return obj


class DualPublic1000SessionRunner:
    """Coordinator for 2-worker replicated Public-1000 execution via independent OS processes."""

    def __init__(
        self,
        *,
        app_config: ApplicationConfig,
        working_dir: Path,
        questions_path: Path,
        session_budget_hours: float = 9.5,
        session_id: str | None = None,
        runtime_builders: dict[int, Callable[[], OnlineRuntime]] | None = None,
        custom_builder_target: tuple[str, str] | None = None,
        worker_devices: tuple[str, str] = ("0", "1"),
    ) -> None:
        self.app_config = app_config
        self.working_dir = working_dir
        self.questions_path = questions_path
        self.session_budget_seconds = float(session_budget_hours * 3600.0)
        self.session_id = (
            session_id
            or f"dual_session_{datetime.datetime.now(datetime.timezone.utc).strftime('%Y%m%d_%H%M%S')}"
        )
        self.runtime_builders = runtime_builders or {}
        self.custom_builder_target = custom_builder_target
        self.worker_devices = worker_devices

        self.questions: list[tuple[str, str]] = Public1000SessionRunner._load_questions(self.questions_path)
        self.canonical_qids = [q[0] for q in self.questions]
        self.expected_total_count = len(self.canonical_qids)
        self.expected_qid_set = set(self.canonical_qids)
        self.expected_qid_set_hash = compute_qid_set_hash(self.canonical_qids)
        self.question_source_sha256 = compute_file_sha256(self.questions_path)
        self.config_hash = self._compute_config_hash()

        # Deterministic partition by canonical index (canonical_index_mod_2_v1)
        self.partition_0_items = [(qid, q) for idx, (qid, q) in enumerate(self.questions) if idx % 2 == 0]
        self.partition_1_items = [(qid, q) for idx, (qid, q) in enumerate(self.questions) if idx % 2 == 1]

        self.partition_0_qids = [q[0] for q in self.partition_0_items]
        self.partition_1_qids = [q[0] for q in self.partition_1_items]

        self.partition_0_set = set(self.partition_0_qids)
        self.partition_1_set = set(self.partition_1_qids)

        # File paths
        self.worker_0_dir = self.working_dir / "worker_0"
        self.worker_1_dir = self.working_dir / "worker_1"

        self.combined_manifest_path = self.working_dir / DUAL_GPU_MANIFEST_FILENAME
        self.combined_audit_path = self.working_dir / DUAL_GPU_AUDIT_FILENAME
        self.combined_zip_path = self.working_dir / DUAL_GPU_CHECKPOINT_ZIP_FILENAME

        self.previous_checkpoint_sha256: str | None = None

        # Setup worker questions files
        self.worker_0_qfile = self.worker_0_dir / "worker_questions.json"
        self.worker_1_qfile = self.worker_1_dir / "worker_questions.json"

        self.worker_0_runner: Public1000SessionRunner | None = None
        self.worker_1_runner: Public1000SessionRunner | None = None

    def _compute_config_hash(self) -> str:
        data = self.app_config.model_dump(mode="json")
        canonical = _canonicalize_for_hash(data)
        return compute_string_sha256(json.dumps(canonical, sort_keys=True))

    def _ensure_worker_question_files(self) -> None:
        self.worker_0_dir.mkdir(parents=True, exist_ok=True)
        self.worker_1_dir.mkdir(parents=True, exist_ok=True)

        if not self.worker_0_qfile.exists():
            w0_dict = {qid: {"question": q_text} for qid, q_text in self.partition_0_items}
            self.worker_0_qfile.write_text(json.dumps(w0_dict, ensure_ascii=False, indent=2), encoding="utf-8")

        if not self.worker_1_qfile.exists():
            w1_dict = {qid: {"question": q_text} for qid, q_text in self.partition_1_items}
            self.worker_1_qfile.write_text(json.dumps(w1_dict, ensure_ascii=False, indent=2), encoding="utf-8")

    def _init_worker_runners(self) -> tuple[Public1000SessionRunner, Public1000SessionRunner]:
        self._ensure_worker_question_files()

        builder_0 = self.runtime_builders.get(0)
        builder_1 = self.runtime_builders.get(1)

        runner_0 = Public1000SessionRunner(
            app_config=self.app_config,
            working_dir=self.worker_0_dir,
            questions_path=self.worker_0_qfile,
            session_budget_hours=self.session_budget_seconds / 3600.0,
            session_id=f"{self.session_id}_w0",
            runtime_builder=builder_0,
        )
        runner_1 = Public1000SessionRunner(
            app_config=self.app_config,
            working_dir=self.worker_1_dir,
            questions_path=self.worker_1_qfile,
            session_budget_hours=self.session_budget_seconds / 3600.0,
            session_id=f"{self.session_id}_w1",
            runtime_builder=builder_1,
        )
        self.worker_0_runner = runner_0
        self.worker_1_runner = runner_1
        return runner_0, runner_1

    def _write_json_atomic(self, path: Path, payload: dict[str, Any]) -> None:
        tmp_path = path.with_suffix(".tmp")
        with tmp_path.open("w", encoding="utf-8") as f:
            json.dump(payload, f, ensure_ascii=False, indent=2)
            f.write("\n")
            f.flush()
            try:
                os.fsync(f.fileno())
            except (AttributeError, OSError):
                pass
        os.replace(tmp_path, path)

    def restore_and_validate_checkpoint(
        self,
        combined_checkpoint_archive_path: Path | None = None,
        expected_previous_checkpoint_sha256: str | None = None,
    ) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
        """Restore combined dual-GPU checkpoint and strictly validate fail-closed."""
        if combined_checkpoint_archive_path is not None and combined_checkpoint_archive_path.exists():
            _LOGGER.info(f"Restoring combined dual-GPU checkpoint from: {combined_checkpoint_archive_path}")
            self.previous_checkpoint_sha256 = compute_file_sha256(combined_checkpoint_archive_path)
            self.working_dir.mkdir(parents=True, exist_ok=True)
            with zipfile.ZipFile(combined_checkpoint_archive_path, "r") as zf:
                for info in zf.infolist():
                    if ".." in info.filename or info.filename.startswith("/"):
                        raise ArtifactCompatibilityError(f"Malformed path in combined checkpoint zip: {info.filename}")
                zf.extractall(self.working_dir)

        runner_0, runner_1 = self._init_worker_runners()

        if not self.combined_manifest_path.exists() and not (self.worker_0_dir / CHECKPOINT_RESULTS_FILENAME).exists():
            _LOGGER.info("No prior dual-GPU checkpoint found. Initializing fresh Dual Session 1.")
            return [], []

        if not self.combined_manifest_path.exists():
            raise ArtifactCompatibilityError("Incomplete dual-GPU checkpoint: combined manifest is missing")

        combined_manifest = json.loads(self.combined_manifest_path.read_text(encoding="utf-8"))

        # Verify top-level authority bindings
        if combined_manifest.get("partition_strategy") != PARTITION_STRATEGY_V1:
            raise ArtifactCompatibilityError(
                f"Dual-GPU partition strategy mismatch: {combined_manifest.get('partition_strategy')} != {PARTITION_STRATEGY_V1}"
            )

        if combined_manifest.get("question_source_sha256") != self.question_source_sha256:
            raise ArtifactCompatibilityError("Dual-GPU question_source_sha256 mismatch")

        if combined_manifest.get("expected_total_qid_count") != self.expected_total_count:
            raise ArtifactCompatibilityError("Dual-GPU expected_total_qid_count mismatch")

        if combined_manifest.get("expected_complete_qid_set_hash") != self.expected_qid_set_hash:
            raise ArtifactCompatibilityError("Dual-GPU expected_complete_qid_set_hash mismatch")

        if combined_manifest.get("application_config_hash") != self.config_hash:
            raise ArtifactCompatibilityError("Dual-GPU application_config_hash mismatch")

        if combined_manifest.get("execution_code_authority_commit") != FROZEN_AUTHORITY_BINDINGS["execution_code_authority_commit"]:
            raise ArtifactCompatibilityError("Dual-GPU execution_code_authority_commit mismatch")

        if (
            combined_manifest.get("runtime_dependencies")
            and combined_manifest.get("runtime_dependencies") != FROZEN_AUTHORITY_BINDINGS["runtime_dependencies"]
        ):
            raise ArtifactCompatibilityError("Dual-GPU runtime_dependencies mismatch")

        if expected_previous_checkpoint_sha256 is not None:
            if combined_manifest.get("previous_checkpoint_sha256") != expected_previous_checkpoint_sha256:
                raise ArtifactCompatibilityError(
                    f"Dual-GPU previous_checkpoint_sha256 mismatch: {combined_manifest.get('previous_checkpoint_sha256')} != {expected_previous_checkpoint_sha256}"
                )

        # Validate worker 0 and worker 1 individual checkpoints
        records_0 = runner_0.restore_and_validate_checkpoint()
        records_1 = runner_1.restore_and_validate_checkpoint()

        qids_0 = [str(r["question_id"]) for r in records_0]
        qids_1 = [str(r["question_id"]) for r in records_1]

        # Verify partition membership integrity
        for qid in qids_0:
            if qid not in self.partition_0_set:
                raise ArtifactCompatibilityError(f"Worker 0 checkpoint contains non-partition-0 QID '{qid}'")
        for qid in qids_1:
            if qid not in self.partition_1_set:
                raise ArtifactCompatibilityError(f"Worker 1 checkpoint contains non-partition-1 QID '{qid}'")

        # Verify no cross-worker duplicates
        cross_duplicates = set(qids_0) & set(qids_1)
        if cross_duplicates:
            raise ArtifactCompatibilityError(f"Cross-worker duplicate QIDs detected: {cross_duplicates}")

        # Verify global completed count
        global_completed = len(qids_0) + len(qids_1)
        if combined_manifest.get("global_completed_count") != global_completed:
            raise ArtifactCompatibilityError(
                f"Dual-GPU count mismatch: parsed {global_completed} records, manifest reports {combined_manifest.get('global_completed_count')}"
            )

        # Verify global QID set hash
        global_qids = qids_0 + qids_1
        if combined_manifest.get("global_completed_qid_set_hash") != compute_qid_set_hash(global_qids):
            raise ArtifactCompatibilityError("Dual-GPU global_completed_qid_set_hash mismatch")

        _LOGGER.info(
            f"Dual-GPU checkpoint validated: {global_completed}/{self.expected_total_count} QIDs completed "
            f"(Worker 0: {len(qids_0)}/{len(self.partition_0_items)}, Worker 1: {len(qids_1)}/{len(self.partition_1_items)})"
        )
        return records_0, records_1

    def _refresh_combined_checkpoint_zip(
        self,
        records_0: list[dict[str, Any]],
        records_1: list[dict[str, Any]],
        status: str,
    ) -> None:
        """Create or refresh the top-level combined dual-GPU export ZIP."""
        global_qids = [str(r["question_id"]) for r in records_0] + [str(r["question_id"]) for r in records_1]
        global_completed = len(global_qids)
        global_remaining = self.expected_total_count - global_completed

        manifest_payload = {
            "schema_version": "4.0.0",
            "partition_strategy": PARTITION_STRATEGY_V1,
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
            "replicated_worker_instances": 2,
            "runtime_dependencies": FROZEN_AUTHORITY_BINDINGS["runtime_dependencies"],
            "question_source_sha256": self.question_source_sha256,
            "expected_total_qid_count": self.expected_total_count,
            "expected_complete_qid_set_hash": self.expected_qid_set_hash,
            "global_completed_count": global_completed,
            "global_remaining_count": global_remaining,
            "global_completed_qid_set_hash": compute_qid_set_hash(global_qids),
            "worker_0_completed_count": len(records_0),
            "worker_0_manifest_sha256": compute_file_sha256(self.worker_0_dir / CHECKPOINT_MANIFEST_FILENAME)
            if (self.worker_0_dir / CHECKPOINT_MANIFEST_FILENAME).exists()
            else "",
            "worker_0_results_sha256": compute_file_sha256(self.worker_0_dir / CHECKPOINT_RESULTS_FILENAME)
            if (self.worker_0_dir / CHECKPOINT_RESULTS_FILENAME).exists()
            else "",
            "worker_1_completed_count": len(records_1),
            "worker_1_manifest_sha256": compute_file_sha256(self.worker_1_dir / CHECKPOINT_MANIFEST_FILENAME)
            if (self.worker_1_dir / CHECKPOINT_MANIFEST_FILENAME).exists()
            else "",
            "worker_1_results_sha256": compute_file_sha256(self.worker_1_dir / CHECKPOINT_RESULTS_FILENAME)
            if (self.worker_1_dir / CHECKPOINT_RESULTS_FILENAME).exists()
            else "",
            "session_id": self.session_id,
            "previous_checkpoint_sha256": self.previous_checkpoint_sha256,
            "updated_at": datetime.datetime.now(datetime.timezone.utc).isoformat(),
        }
        self._write_json_atomic(self.combined_manifest_path, manifest_payload)

        audit_payload = {
            "session_id": self.session_id,
            "status": status,
            "partition_strategy": PARTITION_STRATEGY_V1,
            "global_completed_count": global_completed,
            "global_remaining_count": global_remaining,
            "expected_total_count": self.expected_total_count,
            "worker_0_completed": len(records_0),
            "worker_1_completed": len(records_1),
            "combined_manifest_sha256": compute_file_sha256(self.combined_manifest_path),
            "timestamp": datetime.datetime.now(datetime.timezone.utc).isoformat(),
        }
        self._write_json_atomic(self.combined_audit_path, audit_payload)

        tmp_zip = self.combined_zip_path.with_suffix(".tmp.zip")
        with zipfile.ZipFile(tmp_zip, "w", zipfile.ZIP_DEFLATED) as zf:
            zf.write(self.combined_manifest_path, arcname=DUAL_GPU_MANIFEST_FILENAME)
            zf.write(self.combined_audit_path, arcname=DUAL_GPU_AUDIT_FILENAME)
            for w_dir in [self.worker_0_dir, self.worker_1_dir]:
                for fpath in sorted(w_dir.rglob("*")):
                    if fpath.is_file():
                        rel = fpath.relative_to(self.working_dir).as_posix()
                        zf.write(fpath, arcname=rel)

        os.replace(tmp_zip, self.combined_zip_path)

    def run_session(
        self,
        combined_checkpoint_archive_path: Path | None = None,
        expected_previous_checkpoint_sha256: str | None = None,
        max_questions_per_worker: int | None = None,
    ) -> dict[str, Any]:
        """Run Dual-T4 session with true OS-process isolation and race-safe monitoring."""
        self.working_dir.mkdir(parents=True, exist_ok=True)
        self.restore_and_validate_checkpoint(
            combined_checkpoint_archive_path=combined_checkpoint_archive_path,
            expected_previous_checkpoint_sha256=expected_previous_checkpoint_sha256,
        )

        self._ensure_worker_question_files()
        config_dict = self.app_config.model_dump(mode="json")
        session_start_time = time.perf_counter()

        ctx = mp.get_context("spawn")
        result_queue = ctx.Queue()

        # Launch Worker 0 process
        p0 = ctx.Process(
            target=_worker_process_entrypoint,
            args=(
                0,
                str(self.worker_0_dir),
                str(self.worker_0_qfile),
                config_dict,
                self.session_budget_seconds,
                f"{self.session_id}_w0",
                self.worker_devices[0],
                max_questions_per_worker,
                None,
                self.custom_builder_target,
                str(Path(__file__).resolve().parents[4]),
                result_queue,
            ),
            name="DualWorker_0",
        )

        # Launch Worker 1 process
        p1 = ctx.Process(
            target=_worker_process_entrypoint,
            args=(
                1,
                str(self.worker_1_dir),
                str(self.worker_1_qfile),
                config_dict,
                self.session_budget_seconds,
                f"{self.session_id}_w1",
                self.worker_devices[1],
                max_questions_per_worker,
                None,
                self.custom_builder_target,
                str(Path(__file__).resolve().parents[4]),
                result_queue,
            ),
            name="DualWorker_1",
        )

        p0.start()
        p1.start()
        _LOGGER.info(f"Launched Worker 0 (PID={p0.pid}, Device={self.worker_devices[0]}) and Worker 1 (PID={p1.pid}, Device={self.worker_devices[1]})")

        stop_heartbeat = threading.Event()

        # Race-safe read-only heartbeat thread
        def _heartbeat_loop() -> None:
            while not stop_heartbeat.wait(30.0):
                elapsed = time.perf_counter() - session_start_time
                w0_done = _count_durable_records_safe(self.worker_0_dir / CHECKPOINT_RESULTS_FILENAME)
                w1_done = _count_durable_records_safe(self.worker_1_dir / CHECKPOINT_RESULTS_FILENAME)
                total_done = w0_done + w1_done
                pct = (total_done / self.expected_total_count) * 100.0 if self.expected_total_count > 0 else 0.0
                q_hr = (total_done / (elapsed / 3600.0)) if elapsed > 0 else 0.0
                eta_s = ((self.expected_total_count - total_done) / (total_done / elapsed)) if total_done > 0 else 0.0

                gpu_stats = get_dual_gpu_telemetry()
                print(
                    f"\n[DUAL-GPU HEARTBEAT {elapsed:.1f}s] GLOBAL DURABLE: {total_done}/{self.expected_total_count} ({pct:.1f}%) | "
                    f"W0 (PID {p0.pid}): {w0_done}/{len(self.partition_0_items)} | W1 (PID {p1.pid}): {w1_done}/{len(self.partition_1_items)} | "
                    f"Throughput: {q_hr:.1f} q/hr | ETA: {eta_s/60:.1f} min | "
                    f"GPU0: {gpu_stats['gpu_0']['util_pct']} ({gpu_stats['gpu_0']['vram_used_mb']:.0f}/{gpu_stats['gpu_0']['vram_total_mb']:.0f} MB) | "
                    f"GPU1: {gpu_stats['gpu_1']['util_pct']} ({gpu_stats['gpu_1']['vram_used_mb']:.0f}/{gpu_stats['gpu_1']['vram_total_mb']:.0f} MB)"
                )

        hb_thread = threading.Thread(target=_heartbeat_loop, daemon=True)
        hb_thread.start()

        # Coordinator process monitor loop
        worker_failed = False
        try:
            while p0.is_alive() or p1.is_alive():
                # Check if one process died unexpectedly
                if not p0.is_alive() and p0.exitcode != 0:
                    _LOGGER.error(f"Worker 0 process (PID={p0.pid}) exited with failure code {p0.exitcode}")
                    worker_failed = True
                    if p1.is_alive():
                        p1.terminate()
                    break

                if not p1.is_alive() and p1.exitcode != 0:
                    _LOGGER.error(f"Worker 1 process (PID={p1.pid}) exited with failure code {p1.exitcode}")
                    worker_failed = True
                    if p0.is_alive():
                        p0.terminate()
                    break

                time.sleep(0.5)

            p0.join(timeout=5.0)
            p1.join(timeout=5.0)
        finally:
            stop_heartbeat.set()
            if p0.is_alive():
                p0.terminate()
                p0.join()
            if p1.is_alive():
                p1.terminate()
                p1.join()

        if p0.exitcode != 0 or p1.exitcode != 0:
            worker_failed = True

        # Drain result queue
        queue_results = []
        while not result_queue.empty():
            try:
                queue_results.append(result_queue.get_nowait())
            except Exception:
                break

        for q_res in queue_results:
            if not q_res.get("success"):
                worker_failed = True

        # Read whatever durable records exist on disk
        runner_0, runner_1 = self._init_worker_runners()
        try:
            res_records_0 = runner_0.restore_and_validate_checkpoint()
        except Exception:
            res_records_0 = []
            if (self.worker_0_dir / CHECKPOINT_RESULTS_FILENAME).exists():
                with (self.worker_0_dir / CHECKPOINT_RESULTS_FILENAME).open("r", encoding="utf-8") as f:
                    for line in f:
                        if line.strip():
                            res_records_0.append(json.loads(line))

        try:
            res_records_1 = runner_1.restore_and_validate_checkpoint()
        except Exception:
            res_records_1 = []
            if (self.worker_1_dir / CHECKPOINT_RESULTS_FILENAME).exists():
                with (self.worker_1_dir / CHECKPOINT_RESULTS_FILENAME).open("r", encoding="utf-8") as f:
                    for line in f:
                        if line.strip():
                            res_records_1.append(json.loads(line))

        total_completed = len(res_records_0) + len(res_records_1)

        if worker_failed:
            status = "DUAL_GPU_WORKER_FAILURE_CHECKPOINT_READY"
        elif total_completed == self.expected_total_count:
            status = "ALL_QUESTIONS_COMPLETED"
        else:
            status = "SESSION_CHECKPOINT_COMPLETE"

        self._refresh_combined_checkpoint_zip(res_records_0, res_records_1, status=status)

        zip_sha = compute_file_sha256(self.combined_zip_path)
        zip_size = self.combined_zip_path.stat().st_size

        print("\n" + "=" * 80)
        print(" DUAL-GPU PUBLIC-1000 SESSION CHECKPOINT READY")
        print("=" * 80)
        print(f" Session ID:           {self.session_id}")
        print(f" Status:               {status}")
        print(f" Global Completed:     {total_completed} / {self.expected_total_count}")
        print(f" Worker 0 Completed:   {len(res_records_0)} / {len(self.partition_0_items)}")
        print(f" Worker 1 Completed:   {len(res_records_1)} / {len(self.partition_1_items)}")
        print(f" Checkpoint ZIP:       {self.combined_zip_path}")
        print(f" Checkpoint ZIP Size:  {zip_size:,} bytes")
        print(f" Checkpoint SHA256:    {zip_sha}")
        print("-" * 80)
        if status == "ALL_QUESTIONS_COMPLETED":
            print(" ALL 1000 QUESTIONS COMPLETED. READY FOR SUBMISSION PACKAGING.")
        else:
            print(" SAFE TO STOP THIS KAGGLE SESSION: YES")
            print(" NEXT SESSION ACTION:")
            print(" 1. Download 'public1000_dual_gpu_checkpoint_latest.zip' before ending session.")
            print(" 2. Attach as input in the next Kaggle session and resume.")
        print("=" * 80 + "\n")

        audit_res = json.loads(self.combined_audit_path.read_text(encoding="utf-8"))
        audit_res["checkpoint_zip_path"] = str(self.combined_zip_path)
        audit_res["checkpoint_zip_size"] = zip_size
        audit_res["checkpoint_zip_sha256"] = zip_sha
        return audit_res

    def package_final_submission(self, output_dir: Path) -> Path:
        """Merge both worker records into official canonical ordering and package Codabench submission."""
        runner_0, runner_1 = self._init_worker_runners()
        records_0 = runner_0.restore_and_validate_checkpoint()
        records_1 = runner_1.restore_and_validate_checkpoint()

        total_count = len(records_0) + len(records_1)
        if total_count != self.expected_total_count:
            raise DataValidationError(
                f"Cannot package submission: incomplete dual-GPU checkpoint ({total_count}/{self.expected_total_count} completed)"
            )

        dict_0 = {str(r["question_id"]): r for r in records_0}
        dict_1 = {str(r["question_id"]): r for r in records_1}

        # Verify disjoint partitions and exact coverage
        if set(dict_0.keys()) & set(dict_1.keys()):
            raise DataValidationError("Cross-worker duplicate QIDs detected")

        all_qids = set(dict_0.keys()) | set(dict_1.keys())
        if all_qids != self.expected_qid_set:
            missing = self.expected_qid_set - all_qids
            extra = all_qids - self.expected_qid_set
            raise DataValidationError(f"Submission set mismatch: missing {len(missing)}, extra {len(extra)}")

        output_dir.mkdir(parents=True, exist_ok=True)
        submission_json_path = output_dir / "submission.json"
        submission_zip_path = output_dir / "submission.zip"

        # Canonical ordered merge
        submission_dict: dict[str, dict[str, str]] = {}
        for qid in self.canonical_qids:
            if qid in dict_0:
                ans = str(dict_0[qid].get("answer") or "")
            elif qid in dict_1:
                ans = str(dict_1[qid].get("answer") or "")
            else:
                raise DataValidationError(f"QID {qid} missing during merge")
            submission_dict[qid] = {"answer": ans}

        submission_json_path.write_text(
            json.dumps(submission_dict, ensure_ascii=False, indent=2), encoding="utf-8"
        )

        with zipfile.ZipFile(submission_zip_path, "w", zipfile.ZIP_DEFLATED) as zf:
            zf.write(submission_json_path, arcname="submission.json")

        _LOGGER.info(f"Official submission packaged: {submission_zip_path} ({submission_zip_path.stat().st_size:,} bytes)")
        return submission_zip_path
