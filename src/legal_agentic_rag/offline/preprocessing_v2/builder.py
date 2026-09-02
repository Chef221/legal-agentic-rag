"""Production entry point for M54 Preprocessing V2 offline build."""

from __future__ import annotations

import collections
import datetime
from hashlib import sha256
import json
from pathlib import Path
import re
from typing import Any
import unicodedata

from legal_agentic_rag.competition.uit_dsc_2026.loader import UitDsc2026DataLoader
from legal_agentic_rag.competition.uit_dsc_2026.passage_cleaner import UitDsc2026PassageCleaner
from legal_agentic_rag.offline.chunking.tokenizer import UnicodeWordTokenizer
from legal_agentic_rag.offline.preprocessing_v2.parser import parse_document_structure_v2
from legal_agentic_rag.offline.preprocessing_v2.references import extract_and_resolve_references_v2
from legal_agentic_rag.offline.preprocessing_v2.retrieval_units import materialize_retrieval_units_v2
from legal_agentic_rag.offline.preprocessing_v2.validation import validate_preprocessing_v2
from legal_agentic_rag.schemas.manifests import ArtifactManifest, ArtifactType
from legal_agentic_rag.schemas.preprocessing_v2 import (
    CanonicalDocumentV2,
    LegalProvisionV2,
    LegalReferenceV2,
    PreprocessingV2BuildResult,
    PreprocessingV2SegmentationProfile,
    RetrievalUnitV2,
    UnrecognizedMarkerV2,
)

CANONICAL_REVISION = "sha256:9a4441b4537ceb646b15359f470a1da0904e6c92a61e8c4c376c19e17dec395e"
ZIP_AUTHORITY_SHA256 = "ebcfc896df06087e7da532b4653f32adfaba2200c8ed92a0069e46dbfa126a97"
DATASET_NAME = "uit-dsc-2026-task2"

DOC_NUMBER_RE = re.compile(
    r"(?:số|so|số ký hiệu|so ky hieu)\s*[:.]?\s*([0-9]+(?:/[0-9]+)?(?:/[A-ZĐ0-9_\-]+(?:\.[A-ZĐ0-9_\-]+)*|[A-ZĐ0-9_\-]+(?:/[A-ZĐ0-9_\-]+)+|[A-ZĐ0-9_\-]+/[A-ZĐ0-9_\-]+))",
    re.IGNORECASE,
)

INSTRUMENT_TYPE_RE = re.compile(
    r"^(LUẬT|BỘ LUẬT|NGHỊ ĐỊNH|THÔNG TƯ LIÊN TỊCH|THÔNG TƯ|QUYẾT ĐỊNH|NGHỊ QUYẾT|CHỈ THỊ|LỆNH|HIẾP ĐỊNH|PHÁP LỆNH|BÁO CÁO|CÔNG VĂN|QUY ĐỊNH|QUY CHẾ|ĐIỀU LỆ|HƯỚNG DẪN)",
    re.IGNORECASE,
)

DATE_RE = re.compile(
    r"ngày\s+([0-9]{1,2})\s+tháng\s+([0-9]{1,2})\s+năm\s+([0-9]{4})",
    re.IGNORECASE,
)


def sha256_file(path: Path) -> str:
    h = sha256()
    with open(path, "rb") as f:
        while chunk := f.read(1024 * 1024):
            h.update(chunk)
    return h.hexdigest()


def sha256_text(text: str) -> str:
    return sha256(text.encode("utf-8")).hexdigest()


def extract_document_identity(raw_id: str, raw_name: str | None, raw_link: str | None, authority_text: str) -> dict[str, Any]:
    norm_name = unicodedata.normalize("NFC", raw_name).strip() if raw_name else None

    # 1. Instrument Type
    inst_type = None
    if norm_name:
        m_type = INSTRUMENT_TYPE_RE.search(norm_name)
        if m_type:
            inst_type = m_type.group(1).upper()

    # 2. Issue Date
    issue_date = None
    head_text = authority_text[:2000]
    m_date = DATE_RE.search(head_text)
    if m_date:
        d, m, y = m_date.groups()
        issue_date = f"{int(y):04d}-{int(m):02d}-{int(d):02d}"

    # 3. Document Number Resolution following strictly B1 & B2 priority
    name_nums: list[str] = []
    if norm_name:
        m_num = DOC_NUMBER_RE.findall(norm_name)
        for n in m_num:
            n_clean = n.strip(" /.,:;")
            if n_clean and n_clean not in name_nums:
                name_nums.append(n_clean)

    doc_number: str | None = None
    status: str = "UNRESOLVED"
    candidates: list[str] = []
    evidence: list[dict[str, Any]] = []

    # B1. Raw-name candidates first
    if len(name_nums) == 1:
        status = "DERIVED_FROM_NAME"
        doc_number = name_nums[0]
        candidates = [name_nums[0]]
        evidence.append({"source": "raw_name", "matched_text": doc_number})
    elif len(name_nums) > 1:
        status = "AMBIGUOUS"
        doc_number = None
        candidates = list(name_nums)
        for n in name_nums:
            evidence.append({"source": "raw_name_multiple", "matched_text": n})
    else:
        # B2. Opening text only when name did not resolve identity
        head_nums: list[str] = []
        m_head = DOC_NUMBER_RE.findall(head_text)
        for n in m_head:
            n_clean = n.strip(" /.,:;")
            if n_clean and n_clean not in head_nums:
                head_nums.append(n_clean)

        if len(head_nums) == 1:
            status = "EXPLICIT"
            doc_number = head_nums[0]
            candidates = [head_nums[0]]
            evidence.append({"source": "opening_text", "matched_text": doc_number})
        elif len(head_nums) > 1:
            status = "AMBIGUOUS"
            doc_number = None
            candidates = list(head_nums)
            for n in head_nums:
                evidence.append({"source": "opening_text_multiple", "matched_text": n})
        else:
            status = "UNRESOLVED"
            doc_number = None
            candidates = []

    if not inst_type:
        m_type = INSTRUMENT_TYPE_RE.search(head_text[:500])
        if m_type:
            inst_type = m_type.group(1).upper()

    return {
        "instrument_type": inst_type,
        "document_number": doc_number,
        "title": norm_name,
        "issue_date": issue_date,
        "status": status,
        "candidate_document_numbers": candidates,
        "evidence": evidence,
    }


def write_jsonl_record(fh, record: dict) -> None:
    fh.write(
        json.dumps(
            record,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
        )
        + "\n"
    )


class PreprocessingV2Builder:
    """Builds canonical V2 preprocessing artifacts from official competition zip."""

    def __init__(
        self,
        *,
        segmentation_profile: PreprocessingV2SegmentationProfile | None = None,
    ) -> None:
        self.segmentation_profile = segmentation_profile or PreprocessingV2SegmentationProfile()
        self.cleaner = UitDsc2026PassageCleaner()
        self.cleaner_version = self.cleaner.version

    def build(
        self,
        source: Path,
        destination: Path,
    ) -> PreprocessingV2BuildResult:
        """Execute full offline V2 build reproducing exact accepted shadow output."""
        destination = destination.resolve()
        destination.mkdir(parents=True, exist_ok=True)

        now = datetime.datetime.now(datetime.timezone.utc)
        processing_hash = sha256(
            f"m54_v2_{self.cleaner_version}_{self.segmentation_profile.profile_name}_{CANONICAL_REVISION}".encode("utf-8")
        ).hexdigest()

        # Directories
        docs_dir = destination / "canonical_documents_v2"
        provs_dir = destination / "legal_provisions_v2"
        rus_dir = destination / "retrieval_units_v2"
        refs_dir = destination / "legal_references_v2"
        unrec_dir = destination / "unrecognized_markers_v2"

        for d in [docs_dir, provs_dir, rus_dir, refs_dir, unrec_dir]:
            d.mkdir(parents=True, exist_ok=True)

        docs_jsonl = docs_dir / "records.jsonl"
        provs_jsonl = provs_dir / "records.jsonl"
        rus_jsonl = rus_dir / "records.jsonl"
        refs_jsonl = refs_dir / "records.jsonl"
        unrec_jsonl = unrec_dir / "records.jsonl"

        # 1. Load source contexts and verify authority
        loader = UitDsc2026DataLoader()
        source_identity = loader.inspect_context_source(source)
        actual_zip_sha = sha256_file(source) if source.is_file() else ""
        zip_sha_match = actual_zip_sha == ZIP_AUTHORITY_SHA256
        canonical_rev_match = source_identity.revision == CANONICAL_REVISION

        cleaner_name = "UitDsc2026PassageCleaner"
        cleaner_version = self.cleaner.version
        cleaner_policy_id = self.cleaner.policy_identity()

        # ================= STAGE A: DOCUMENTS =================
        documents: list[dict[str, Any]] = []
        confirmed_index: dict[str, list[str]] = collections.defaultdict(list)
        ambiguous_candidate_index: dict[str, list[str]] = collections.defaultdict(list)

        with open(docs_jsonl, "w", encoding="utf-8", newline="\n") as f_docs:
            for idx, (member_name, raw_record) in enumerate(loader._iter_zip_contexts(source)):
                rec = loader._context_record(raw_record, member_name)
                raw_id = str(rec.context_id)
                raw_link = rec.source_url
                raw_name = rec.title
                raw_passage = rec.passage or ""
                raw_passage_sha256 = sha256_text(raw_passage)

                if not raw_passage:
                    auth_text = ""
                else:
                    cleaning_result = self.cleaner.clean(raw_passage)
                    auth_text = unicodedata.normalize("NFC", cleaning_result.text)

                auth_text_sha256 = sha256_text(auth_text)
                ident = extract_document_identity(raw_id, raw_name, raw_link, auth_text)

                doc_obj = {
                    "schema_version": "m54-preprocessing-v2.1",
                    "document_id": f"doc:uitdsc2026:{raw_id}",
                    "source": {
                        "dataset": DATASET_NAME,
                        "context_id": raw_id,
                        "member_name": member_name,
                        "corpus_revision": source_identity.revision,
                        "raw_passage_sha256": raw_passage_sha256,
                    },
                    "raw": {
                        "name": raw_name,
                        "link": raw_link,
                        "text": raw_passage,
                    },
                    "identity": ident,
                    "authority_text": auth_text,
                    "authority_text_sha256": auth_text_sha256,
                    "cleaner": {
                        "name": cleaner_name,
                        "version": cleaner_version,
                        "policy_identity": cleaner_policy_id,
                    },
                    "quality_flags": (["EMPTY_RAW_PASSAGE"] if not raw_passage else []) + (["MISSING_RAW_NAME"] if not raw_name else []),
                }
                write_jsonl_record(f_docs, doc_obj)
                documents.append(doc_obj)

                d_status = ident.get("status")
                if d_status in ("EXPLICIT", "DERIVED_FROM_NAME") and ident.get("document_number"):
                    norm_num = ident["document_number"].strip().upper()
                    confirmed_index[norm_num].append(doc_obj["document_id"])
                elif d_status == "AMBIGUOUS":
                    for cand in ident.get("candidate_document_numbers") or []:
                        norm_cand = cand.strip().upper()
                        ambiguous_candidate_index[norm_cand].append(doc_obj["document_id"])

        man_docs = ArtifactManifest(
            schema_version="m54-preprocessing-v2.1",
            artifact_type=ArtifactType.CANONICAL_DOCUMENTS_V2,
            artifact_version="2.0",
            dataset_name=DATASET_NAME,
            dataset_revision=CANONICAL_REVISION,
            created_at=now,
            record_count=len(documents),
            processing_config_hash=processing_hash,
            code_version="2.0.0",
            metadata={"record_model": "CanonicalDocumentV2"},
        )
        (docs_dir / "manifest.json").write_text(
            json.dumps(man_docs.model_dump(mode="json"), indent=2, ensure_ascii=False) + "\n",
            encoding="utf-8",
        )

        # ================= STAGE B: PARSER =================
        all_provisions: list[dict[str, Any]] = []
        all_unrec_markers: list[dict[str, Any]] = []

        with open(provs_jsonl, "w", encoding="utf-8", newline="\n") as f_provs, open(
            unrec_jsonl, "w", encoding="utf-8", newline="\n"
        ) as f_unrec:
            for d in documents:
                doc_id = d["document_id"]
                auth_text = d["authority_text"]

                doc_provs, doc_unrec, _ = parse_document_structure_v2(doc_id, auth_text)

                for p in doc_provs:
                    write_jsonl_record(f_provs, p)
                    all_provisions.append(p)

                for u in doc_unrec:
                    write_jsonl_record(f_unrec, u)
                    all_unrec_markers.append(u)

        man_provs = ArtifactManifest(
            schema_version="m54-preprocessing-v2.1",
            artifact_type=ArtifactType.LEGAL_PROVISIONS_V2,
            artifact_version="2.0",
            dataset_name=DATASET_NAME,
            dataset_revision=CANONICAL_REVISION,
            created_at=now,
            record_count=len(all_provisions),
            processing_config_hash=processing_hash,
            code_version="2.0.0",
            metadata={"record_model": "LegalProvisionV2"},
        )
        (provs_dir / "manifest.json").write_text(
            json.dumps(man_provs.model_dump(mode="json"), indent=2, ensure_ascii=False) + "\n",
            encoding="utf-8",
        )

        # ================= STAGE C: RETRIEVAL UNITS =================
        retrieval_units = materialize_retrieval_units_v2(documents, all_provisions, self.segmentation_profile.max_tokens)
        with open(rus_jsonl, "w", encoding="utf-8", newline="\n") as f_rus:
            for ru in retrieval_units:
                write_jsonl_record(f_rus, ru)

        man_rus = ArtifactManifest(
            schema_version="m54-preprocessing-v2.1",
            artifact_type=ArtifactType.RETRIEVAL_UNITS_V2,
            artifact_version="2.0",
            dataset_name=DATASET_NAME,
            dataset_revision=CANONICAL_REVISION,
            created_at=now,
            record_count=len(retrieval_units),
            processing_config_hash=processing_hash,
            code_version="2.0.0",
            metadata={
                "record_model": "RetrievalUnitV2",
                "segmentation_profile": self.segmentation_profile.model_dump(),
            },
        )
        (rus_dir / "manifest.json").write_text(
            json.dumps(man_rus.model_dump(mode="json"), indent=2, ensure_ascii=False) + "\n",
            encoding="utf-8",
        )

        # ================= STAGE D: REFERENCES =================
        references = extract_and_resolve_references_v2(documents, confirmed_index, ambiguous_candidate_index)
        with open(refs_jsonl, "w", encoding="utf-8", newline="\n") as f_refs:
            for ref in references:
                write_jsonl_record(f_refs, ref)

        man_refs = ArtifactManifest(
            schema_version="m54-preprocessing-v2.1",
            artifact_type=ArtifactType.LEGAL_REFERENCES_V2,
            artifact_version="2.0",
            dataset_name=DATASET_NAME,
            dataset_revision=CANONICAL_REVISION,
            created_at=now,
            record_count=len(references),
            processing_config_hash=processing_hash,
            code_version="2.0.0",
            metadata={"record_model": "LegalReferenceV2"},
        )
        (refs_dir / "manifest.json").write_text(
            json.dumps(man_refs.model_dump(mode="json"), indent=2, ensure_ascii=False) + "\n",
            encoding="utf-8",
        )

        # ================= STAGE E: VALIDATION =================
        val_path = destination / "production_v2_validation.json"
        val_result = validate_preprocessing_v2(
            source=source,
            docs_path=docs_jsonl,
            provs_path=provs_jsonl,
            rus_path=rus_jsonl,
            refs_path=refs_jsonl,
            unrec_path=unrec_jsonl,
            val_out_path=val_path,
        )

        return PreprocessingV2BuildResult(
            root_path=destination,
            documents_path=docs_jsonl,
            provisions_path=provs_jsonl,
            retrieval_units_path=rus_jsonl,
            legal_references_path=refs_jsonl,
            unrecognized_markers_path=unrec_jsonl,
            validation_path=val_path,
            document_count=len(documents),
            provision_count=len(all_provisions),
            retrieval_unit_count=len(retrieval_units),
            legal_reference_count=len(references),
            unrecognized_marker_count=len(all_unrec_markers),
            overall_pass=val_result.get("overall_pass", False),
        )
