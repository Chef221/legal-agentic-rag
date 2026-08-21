"""Regression checks for the cleaned retained Kaggle workflows."""

import ast
from importlib.util import module_from_spec, spec_from_file_location
from pathlib import Path
from zipfile import ZipFile


ROOT = Path(__file__).parents[3]
NOTEBOOKS = ROOT / "notebooks"


def test_retained_python_runners_parse_without_retired_imports() -> None:
    """M48/M49/M49.1 runners use neutral shared modules after cleanup."""
    retained = [
        "kaggle_candidate_dev_common.py",
        "kaggle_public_submission_common.py",
        "m48_kaggle_candidate_dev.py",
        "m48_kaggle_public_submission.py",
        "m49_kaggle_candidate_dev.py",
        "m49_kaggle_train_generator.py",
        "m491_kaggle_candidate_dev.py",
        "m491_kaggle_public_submission.py",
    ]
    combined = ""
    for filename in retained:
        source = (NOTEBOOKS / filename).read_text(encoding="utf-8")
        ast.parse(source, filename=filename)
        combined += source

    assert "import m46_kaggle" not in combined
    assert "import m47_kaggle" not in combined


def test_retired_m46_m47_executables_are_absent() -> None:
    """Historical candidates remain documentation only, not runnable profiles."""
    retired = [
        "m46_kaggle_candidate_dev.py",
        "m47_kaggle_candidate_dev.py",
        "m47_kaggle_public_submission.py",
        "M46_KAGGLE_01_DEV_EVAL.ipynb",
        "M46_KAGGLE_02_CANDIDATE_DEV.ipynb",
        "M47_KAGGLE_01_DEV_EVAL.ipynb",
        "M47_KAGGLE_02_PUBLIC_SUBMISSION.ipynb",
    ]

    assert all(not (NOTEBOOKS / filename).exists() for filename in retired)


def test_source_packager_uses_posix_root_and_stable_metadata(
    tmp_path: Path,
    monkeypatch,
) -> None:
    """Kaggle archives avoid Windows separators and untracked local data."""
    script = ROOT / "scripts/package_kaggle_source.py"
    spec = spec_from_file_location("package_kaggle_source", script)
    assert spec is not None and spec.loader is not None
    module = module_from_spec(spec)
    spec.loader.exec_module(module)

    project = tmp_path / "project"
    (project / "src").mkdir(parents=True)
    (project / "README.md").write_text("readme\n", encoding="utf-8")
    (project / "src/app.py").write_text("value = 1\n", encoding="utf-8")
    output = tmp_path / "source.zip"
    monkeypatch.setattr(module, "PROJECT_ROOT", project)
    monkeypatch.setattr(module, "require_clean_checkout", lambda: None)
    monkeypatch.setattr(
        module,
        "tracked_files",
        lambda: [Path("README.md"), Path("src/app.py")],
    )

    first_hash = module.package("m491", output)

    assert len(first_hash) == 64
    with ZipFile(output) as archive:
        assert archive.namelist() == [
            "legal-agentic-rag/README.md",
            "legal-agentic-rag/src/app.py",
        ]
        assert all(item.date_time == (1980, 1, 1, 0, 0, 0) for item in archive.infolist())
