"""Lightweight tests for the Kaggle M49 training data path."""

from importlib.util import module_from_spec, spec_from_file_location
from pathlib import Path
from types import SimpleNamespace


def _training_module():
    path = Path(__file__).parents[3] / "notebooks/m49_kaggle_train_generator.py"
    spec = spec_from_file_location("m49_kaggle_train_generator", path)
    assert spec is not None and spec.loader is not None
    module = module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class _Tokenizer:
    eos_token = "<eos>"
    pad_token_id = 0

    @staticmethod
    def apply_chat_template(*args, **kwargs) -> str:
        return "system user assistant"

    @staticmethod
    def __call__(text: str, **kwargs) -> dict[str, list[int]]:
        return {"input_ids": [index + 1 for index, _ in enumerate(text.split())]}


def test_m49_dataset_masks_prompt_and_keeps_real_answer_tokens() -> None:
    """Loss applies only to the unchanged official answer completion."""
    module = _training_module()
    record = SimpleNamespace(
        question_id="q1",
        question="Câu hỏi thật?",
        reference_answer="Câu trả lời thật.",
    )

    dataset = module.OfficialAnswerDataset([record], _Tokenizer())
    item = dataset[0]

    assert len(dataset) == 1
    assert item["labels"][:3].tolist() == [-100, -100, -100]
    assert all(value > 0 for value in item["labels"][3:].tolist())
    assert item["input_ids"].shape == item["labels"].shape


def test_m49_collator_uses_safe_padding_values() -> None:
    """Batch padding cannot become answer loss or an attended model token."""
    import torch

    module = _training_module()
    collator = module.ResponseOnlyCollator(pad_token_id=9)
    features = [
        {
            "input_ids": torch.tensor([1, 2]),
            "attention_mask": torch.tensor([1, 1]),
            "labels": torch.tensor([-100, 2]),
        },
        {
            "input_ids": torch.tensor([1, 2, 3]),
            "attention_mask": torch.tensor([1, 1, 1]),
            "labels": torch.tensor([-100, 2, 3]),
        },
    ]

    batch = collator(features)

    assert batch["input_ids"].shape == (2, 8)
    assert batch["input_ids"][0, -1].item() == 9
    assert batch["attention_mask"][0, -1].item() == 0
    assert batch["labels"][0, -1].item() == -100


def test_m49_restores_kaggle_extracted_source(tmp_path, monkeypatch) -> None:
    """Kaggle may expose an uploaded ZIP as an extracted project directory."""
    module = _training_module()
    input_root = tmp_path / "input"
    project = input_root / "datasets/team/legal-agentic-rag-m49-source/legal-agentic-rag"
    (project / "notebooks").mkdir(parents=True)
    (project / "src/legal_agentic_rag").mkdir(parents=True)
    (project / "pyproject.toml").write_text("[project]\nname='test'\n", encoding="utf-8")
    (project / "notebooks/m49_kaggle_train_generator.py").write_text(
        "# source marker\n", encoding="utf-8"
    )
    repo = tmp_path / "working/legal-agentic-rag"
    monkeypatch.setattr(module, "INPUT_ROOT", input_root)
    monkeypatch.setattr(module, "WORKING", tmp_path / "working")
    monkeypatch.setattr(module, "REPO", repo)

    module.restore_project_source()

    assert (repo / "pyproject.toml").is_file()
    assert (repo / "notebooks/m49_kaggle_train_generator.py").is_file()


def test_m49_uses_transformers_v5_warmup_argument() -> None:
    """Transformers v5 expresses a ratio through warmup_steps."""
    path = Path(__file__).parents[3] / "notebooks/m49_kaggle_train_generator.py"
    source = path.read_text(encoding="utf-8")

    assert "warmup_steps=0.03" in source
    assert "warmup_ratio=" not in source


def test_m49_removes_incompatible_optional_torchao() -> None:
    """Kaggle's stale optional TorchAO cannot block the standard PEFT merge."""
    path = Path(__file__).parents[3] / "notebooks/m49_kaggle_train_generator.py"
    source = path.read_text(encoding="utf-8")

    assert '"uninstall", "-q", "-y", "torchao"' in source
