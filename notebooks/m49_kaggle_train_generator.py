"""Fine-tune the M49 generator using only official UIT DSC 2026 labels."""

from __future__ import annotations

import gc
import json
import os
import shutil
import subprocess
import sys
from datetime import UTC, datetime
from hashlib import sha256
from pathlib import Path
from typing import Any
from zipfile import ZipFile

INPUT_ROOT = Path("/kaggle/input")
WORKING = Path("/kaggle/working")
REPO = WORKING / "legal-agentic-rag"
OUTPUT_ROOT = WORKING / "m49-qwen3.5-2b-official-sft-v1"
TRAINER_ROOT = OUTPUT_ROOT / "trainer"
ADAPTER_ROOT = OUTPUT_ROOT / "adapter"
MERGED_ROOT = OUTPUT_ROOT / "merged"
SPLIT_MANIFEST = OUTPUT_ROOT / "split-manifest.json"
ADAPTER_MARKER = OUTPUT_ROOT / "adapter-complete.json"
TRAINING_MANIFEST = OUTPUT_ROOT / "m49-training-manifest.json"

TRAIN_SHA256 = "2a52501cc065d266f2f832475950bcf1e7c75c386efa9b2f568f251d745f5988"
BASE_MODEL = "Qwen/Qwen3.5-2B"
BASE_REVISION = "15852e8c16360a2fea060d615a32b45270f8a8fc"
EXPERIMENT = "m49-official-qa-qlora-v1"
SEED = 20260817
MAX_SEQUENCE_LENGTH = 1024
LORA_TARGET_MODULES = [
    "q_proj",
    "k_proj",
    "v_proj",
    "o_proj",
    "gate_proj",
    "up_proj",
    "down_proj",
    "in_proj_qkv",
    "in_proj_z",
    "in_proj_b",
    "in_proj_a",
    "out_proj",
]
SYSTEM_INSTRUCTION = (
    "Bạn là trợ lý trả lời câu hỏi pháp luật Việt Nam. Trả lời trực tiếp bằng "
    "văn xuôi tiếng Việt, đầy đủ chủ thể, điều kiện, ngoại lệ, thủ tục, thời hạn "
    "và số liệu có liên quan; không thêm nội dung ngoài phạm vi câu hỏi."
)


def file_sha256(path: Path) -> str:
    """Hash one file without loading it entirely into memory."""
    digest = sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def directory_sha256(path: Path) -> str:
    """Hash relative names and bytes of every regular file in a directory."""
    digest = sha256()
    files = sorted(item for item in path.rglob("*") if item.is_file())
    if not files:
        raise AssertionError(f"Thư mục artifact rỗng: {path}")
    for item in files:
        relative = item.relative_to(path).as_posix().encode("utf-8")
        digest.update(len(relative).to_bytes(8, "big"))
        digest.update(relative)
        digest.update(bytes.fromhex(file_sha256(item)))
    return digest.hexdigest()


def matching_file(filename: str, expected_sha256: str | None = None) -> Path:
    """Find one exact Kaggle input, optionally requiring its official hash."""
    matches = sorted(INPUT_ROOT.rglob(filename))
    if expected_sha256 is not None:
        matches = [item for item in matches if file_sha256(item) == expected_sha256]
    if not matches:
        raise AssertionError(f"Không thấy input hợp lệ: {filename}")
    preferred = [item for item in matches if "m49" in str(item).casefold()]
    return (preferred or matches)[0]


def restore_previous_output() -> None:
    """Restore a complete artifact or resumable Trainer checkpoints."""
    if OUTPUT_ROOT.is_dir():
        return
    candidates = [path.parent for path in INPUT_ROOT.rglob(TRAINING_MANIFEST.name)]
    for checkpoint in INPUT_ROOT.rglob("checkpoint-*"):
        if checkpoint.is_dir() and checkpoint.parent.name == TRAINER_ROOT.name:
            candidates.append(checkpoint.parent.parent)
    candidates = [path for path in candidates if path.name == OUTPUT_ROOT.name]
    if not candidates:
        return
    source = min(set(candidates))
    print("Khôi phục checkpoint M49:", source, flush=True)
    shutil.copytree(source, OUTPUT_ROOT)


def restore_project_source() -> None:
    """Restore source whether Kaggle preserves or extracts the uploaded ZIP."""
    if REPO.is_dir():
        return
    source_zips = sorted(INPUT_ROOT.rglob("legal-agentic-rag-m49-source.zip"))
    if source_zips:
        with ZipFile(source_zips[0]) as archive:
            archive.extractall(WORKING)
        return
    candidates: list[Path] = []
    for script in INPUT_ROOT.rglob("m49_kaggle_train_generator.py"):
        project_root = script.parent.parent
        if (
            (project_root / "pyproject.toml").is_file()
            and (project_root / "src/legal_agentic_rag").is_dir()
        ):
            candidates.append(project_root)
    if not candidates:
        raise AssertionError(
            "Khong thay source M49 o dang ZIP hoac thu muc Kaggle da giai nen"
        )
    preferred = [path for path in candidates if "m49" in str(path).casefold()]
    source = (preferred or candidates)[0]
    print("Khoi phuc source M49 tu thu muc Kaggle:", source, flush=True)
    shutil.copytree(source, REPO)


def install_source() -> Path:
    """Restore exact M49 source and install pinned training dependencies."""
    restore_project_source()
    if not (REPO / "src/legal_agentic_rag").is_dir():
        raise AssertionError("Source ZIP M49 không có project hợp lệ")
    subprocess.run(
        [
            sys.executable,
            "-m",
            "pip",
            "install",
            "-q",
            "--no-cache-dir",
            "transformers==5.15.0",
            "peft==0.19.1",
            "bitsandbytes==0.49.2",
            "accelerate==1.14.0",
        ],
        check=True,
    )
    # Kaggle currently preinstalls torchao 0.10, while PEFT 0.19 requires
    # torchao >= 0.16 whenever that optional backend is present. M49 does not
    # use TorchAO, so removing the incompatible optional package lets PEFT use
    # the standard Linear dispatcher during adapter merge.
    subprocess.run(
        [sys.executable, "-m", "pip", "uninstall", "-q", "-y", "torchao"],
        check=True,
    )
    subprocess.run(
        [sys.executable, "-m", "pip", "install", "-q", "-e", str(REPO), "--no-deps"],
        check=True,
    )
    repo_src = str(REPO / "src")
    if repo_src not in sys.path:
        sys.path.insert(0, repo_src)
    return matching_file("train.json", TRAIN_SHA256)


def prepare_supervision(train_path: Path) -> tuple[list[Any], dict[str, Any]]:
    """Load real labels and reproduce the frozen group-safe split."""
    from legal_agentic_rag.competition.uit_dsc_2026 import (
        UitDsc2026DataLoader,
        fixed_dev_sample,
        question_id_digest,
        split_generator_supervision,
    )

    questions = UitDsc2026DataLoader().load_questions(
        train_path,
        require_reference_answers=True,
    )
    if len(questions) != 7000:
        raise AssertionError(f"Train record count thay đổi: {len(questions)}")
    split = split_generator_supervision(questions)
    expected_counts = {"train": 5617, "dev": 678, "holdout": 705}
    if split.record_counts != expected_counts:
        raise AssertionError(split.record_counts)
    dev200 = fixed_dev_sample(split)
    expected_dev_digest = "694825b5961a90a284ad0364ac4f31a1a85f446519c92274a784c8e2be9a48ad"
    if question_id_digest(dev200) != expected_dev_digest:
        raise AssertionError("Dev-200 không khớp control M48")
    manifest = {
        "source_sha256": file_sha256(train_path),
        "normalization": "NFC+casefold+whitespace",
        "split_seed": split.split_seed,
        "record_counts": split.record_counts,
        "normalized_question_group_count": split.normalized_question_group_count,
        "duplicate_group_count": split.duplicate_group_count,
        "dev_sample_count": len(dev200),
        "dev_sample_ids_sha256": question_id_digest(dev200),
        "training_partition": "train",
        "training_record_count": len(split.train),
    }
    OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)
    SPLIT_MANIFEST.write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    return list(split.train), manifest


class OfficialAnswerDataset:
    """Tokenize only official question-answer pairs with response-only loss."""

    def __init__(self, records: list[Any], tokenizer: Any) -> None:
        import torch

        self._torch = torch
        self._items: list[dict[str, Any]] = []
        eos = tokenizer.eos_token or ""
        for index, record in enumerate(records, start=1):
            if record.reference_answer is None:
                raise AssertionError("Training record thiếu gold answer")
            prefix = tokenizer.apply_chat_template(
                [
                    {"role": "system", "content": SYSTEM_INSTRUCTION},
                    {"role": "user", "content": record.question},
                ],
                tokenize=False,
                add_generation_prompt=True,
            )
            prefix_ids = tokenizer(prefix, add_special_tokens=False)["input_ids"]
            answer_ids = tokenizer(
                f"{record.reference_answer}{eos}",
                add_special_tokens=False,
            )["input_ids"]
            available = MAX_SEQUENCE_LENGTH - len(prefix_ids)
            if available <= 0:
                raise AssertionError(f"Question prompt quá dài: {record.question_id}")
            answer_ids = answer_ids[:available]
            input_ids = [*prefix_ids, *answer_ids]
            labels = [-100] * len(prefix_ids) + answer_ids.copy()
            self._items.append(
                {
                    "input_ids": torch.tensor(input_ids, dtype=torch.long),
                    "attention_mask": torch.ones(len(input_ids), dtype=torch.long),
                    "labels": torch.tensor(labels, dtype=torch.long),
                }
            )
            if index % 1000 == 0:
                print(f"Đã tokenize {index}/{len(records)} mẫu thật...", flush=True)

    def __len__(self) -> int:
        return len(self._items)

    def __getitem__(self, index: int) -> dict[str, Any]:
        return self._items[index]


class ResponseOnlyCollator:
    """Pad causal-LM examples while retaining ignored prompt labels."""

    def __init__(self, pad_token_id: int) -> None:
        self._pad_token_id = pad_token_id

    def __call__(self, features: list[dict[str, Any]]) -> dict[str, Any]:
        import torch

        length = max(int(item["input_ids"].shape[0]) for item in features)
        length = ((length + 7) // 8) * 8
        batch: dict[str, list[Any]] = {
            "input_ids": [],
            "attention_mask": [],
            "labels": [],
        }
        for item in features:
            padding = length - int(item["input_ids"].shape[0])
            batch["input_ids"].append(
                torch.nn.functional.pad(
                    item["input_ids"], (0, padding), value=self._pad_token_id
                )
            )
            batch["attention_mask"].append(
                torch.nn.functional.pad(item["attention_mask"], (0, padding), value=0)
            )
            batch["labels"].append(
                torch.nn.functional.pad(item["labels"], (0, padding), value=-100)
            )
        return {name: torch.stack(values) for name, values in batch.items()}


def train_adapter(records: list[Any]) -> None:
    """Train one deterministic QLoRA adapter with resumable checkpoints."""
    if ADAPTER_MARKER.is_file() and ADAPTER_ROOT.is_dir():
        print("Adapter M49 đã hoàn tất; bỏ qua training.", flush=True)
        return

    import torch
    from peft import (
        LoraConfig,
        TaskType,
        get_peft_model,
        prepare_model_for_kbit_training,
    )
    from transformers import (
        AutoModelForMultimodalLM,
        AutoTokenizer,
        BitsAndBytesConfig,
        Trainer,
        TrainingArguments,
        set_seed,
    )
    from transformers.trainer_utils import get_last_checkpoint

    set_seed(SEED)
    tokenizer = AutoTokenizer.from_pretrained(
        BASE_MODEL,
        revision=BASE_REVISION,
        trust_remote_code=False,
    )
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    dataset = OfficialAnswerDataset(records, tokenizer)
    quantization = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_use_double_quant=True,
        bnb_4bit_compute_dtype=torch.float16,
    )
    model = AutoModelForMultimodalLM.from_pretrained(
        BASE_MODEL,
        revision=BASE_REVISION,
        trust_remote_code=False,
        local_files_only=False,
        quantization_config=quantization,
        device_map={"": 0},
        dtype=torch.float16,
    )
    model.config.use_cache = False
    model = prepare_model_for_kbit_training(
        model,
        use_gradient_checkpointing=True,
    )
    model = get_peft_model(
        model,
        LoraConfig(
            task_type=TaskType.CAUSAL_LM,
            r=16,
            lora_alpha=32,
            lora_dropout=0.05,
            bias="none",
            target_modules=LORA_TARGET_MODULES,
        ),
    )
    trainable = [name for name, value in model.named_parameters() if value.requires_grad]
    if not trainable or any("visual" in name.casefold() for name in trainable):
        raise AssertionError("LoRA target không giới hạn đúng text backbone")
    model.print_trainable_parameters()

    arguments = TrainingArguments(
        output_dir=str(TRAINER_ROOT),
        num_train_epochs=1.0,
        per_device_train_batch_size=1,
        gradient_accumulation_steps=8,
        learning_rate=1e-4,
        weight_decay=0.0,
        warmup_steps=0.03,
        lr_scheduler_type="cosine",
        logging_steps=5,
        save_steps=50,
        save_total_limit=2,
        fp16=True,
        bf16=False,
        optim="paged_adamw_8bit",
        gradient_checkpointing=True,
        gradient_checkpointing_kwargs={"use_reentrant": False},
        report_to="none",
        remove_unused_columns=False,
        dataloader_num_workers=2,
        seed=SEED,
        data_seed=SEED,
    )
    trainer = Trainer(
        model=model,
        args=arguments,
        train_dataset=dataset,
        data_collator=ResponseOnlyCollator(tokenizer.pad_token_id),
        processing_class=tokenizer,
    )
    checkpoint = get_last_checkpoint(str(TRAINER_ROOT)) if TRAINER_ROOT.is_dir() else None
    print("Bắt đầu/resume QLoRA M49 từ:", checkpoint or "đầu", flush=True)
    trainer.train(resume_from_checkpoint=checkpoint)
    ADAPTER_ROOT.mkdir(parents=True, exist_ok=True)
    trainer.save_model(str(ADAPTER_ROOT))
    tokenizer.save_pretrained(ADAPTER_ROOT)
    ADAPTER_MARKER.write_text(
        json.dumps(
            {
                "experiment": EXPERIMENT,
                "adapter_sha256": directory_sha256(ADAPTER_ROOT),
            },
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    del trainer, model, dataset
    gc.collect()
    torch.cuda.empty_cache()


def merge_adapter() -> None:
    """Merge LoRA into fp16 base weights for dependency-free M49 inference."""
    if MERGED_ROOT.is_dir() and any(MERGED_ROOT.glob("*.safetensors")):
        print("Merged generator M49 đã tồn tại; bỏ qua merge.", flush=True)
        return

    import torch
    from peft import PeftModel
    from transformers import AutoModelForMultimodalLM, AutoTokenizer

    print("Đang merge adapter M49 vào base fp16 trên CPU...", flush=True)
    base = AutoModelForMultimodalLM.from_pretrained(
        BASE_MODEL,
        revision=BASE_REVISION,
        trust_remote_code=False,
        local_files_only=False,
        dtype=torch.float16,
        device_map="cpu",
    )
    adapted = PeftModel.from_pretrained(base, ADAPTER_ROOT, is_trainable=False)
    merged = adapted.merge_and_unload(safe_merge=True, progressbar=True)
    MERGED_ROOT.mkdir(parents=True, exist_ok=True)
    merged.save_pretrained(
        MERGED_ROOT,
        safe_serialization=True,
        max_shard_size="5GB",
    )
    AutoTokenizer.from_pretrained(
        BASE_MODEL,
        revision=BASE_REVISION,
        trust_remote_code=False,
    ).save_pretrained(MERGED_ROOT)
    del adapted, merged, base
    gc.collect()


def finalize_manifest(split_manifest: dict[str, Any]) -> dict[str, Any]:
    """Pin model, dependency, data and output identities after successful merge."""
    import accelerate
    import bitsandbytes
    import peft
    import torch
    import transformers

    adapter_hash = directory_sha256(ADAPTER_ROOT)
    merged_hash = directory_sha256(MERGED_ROOT)
    manifest = {
        "schema_version": "1.0",
        "artifact_type": "merged_generator_model",
        "experiment": EXPERIMENT,
        "created_at": datetime.now(UTC).isoformat().replace("+00:00", "Z"),
        "complete": True,
        "official_data_only": True,
        "synthetic_data_used": False,
        "source_train_sha256": TRAIN_SHA256,
        "split_manifest": split_manifest,
        "base_model": BASE_MODEL,
        "base_model_revision": BASE_REVISION,
        "final_model_parameter_identity": "same architecture as merged Qwen3.5-2B base",
        "system_stack_parameter_count": 3_466_000_000,
        "adapter_sha256": adapter_hash,
        "merged_model_sha256": merged_hash,
        "training": {
            "objective": "causal_lm_response_only_on_official_answer",
            "seed": SEED,
            "epochs": 1.0,
            "max_sequence_length": MAX_SEQUENCE_LENGTH,
            "per_device_batch_size": 1,
            "gradient_accumulation_steps": 8,
            "learning_rate": 1e-4,
            "quantization": "nf4_double_quant_training_only",
            "lora_rank": 16,
            "lora_alpha": 32,
            "lora_dropout": 0.05,
            "lora_target_modules": LORA_TARGET_MODULES,
        },
        "dependencies": {
            "torch": torch.__version__,
            "transformers": transformers.__version__,
            "peft": peft.__version__,
            "bitsandbytes": bitsandbytes.__version__,
            "accelerate": accelerate.__version__,
        },
        "registration_status": "requires_organizer_confirmation_for_finetuned_revision",
    }
    TRAINING_MANIFEST.write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    return manifest


def validate_complete_artifact() -> dict[str, Any]:
    """Fail closed if a restored or newly built M49 artifact is incomplete."""
    manifest = json.loads(TRAINING_MANIFEST.read_text(encoding="utf-8"))
    if manifest.get("complete") is not True:
        raise AssertionError("M49 training manifest chưa complete")
    if manifest.get("source_train_sha256") != TRAIN_SHA256:
        raise AssertionError("M49 dùng sai train lineage")
    if directory_sha256(ADAPTER_ROOT) != manifest.get("adapter_sha256"):
        raise AssertionError("Adapter M49 checksum không khớp")
    if directory_sha256(MERGED_ROOT) != manifest.get("merged_model_sha256"):
        raise AssertionError("Merged model M49 checksum không khớp")
    return manifest


def main() -> None:
    """Prepare, train, merge and validate the official-only M49 generator."""
    os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
    import torch

    if not torch.cuda.is_available():
        raise AssertionError("Hãy bật GPU Accelerator trong Kaggle Settings")
    torch.ones(1, device="cuda")
    print("GPU training:", torch.cuda.get_device_name(0), flush=True)
    restore_previous_output()
    train_path = install_source()
    if TRAINING_MANIFEST.is_file():
        manifest = validate_complete_artifact()
        print("M49 đã hoàn tất từ output trước.", flush=True)
    else:
        records, split_manifest = prepare_supervision(train_path)
        train_adapter(records)
        merge_adapter()
        manifest = finalize_manifest(split_manifest)
        validate_complete_artifact()
    print(json.dumps(manifest, ensure_ascii=False, indent=2), flush=True)
    print("M49 OUTPUT:", OUTPUT_ROOT, flush=True)
    print("Hãy Quick Save với Always save output.", flush=True)


if __name__ == "__main__":
    main()
