"""Optional accelerator resource observation tests."""

from types import SimpleNamespace
import sys

from legal_agentic_rag.evaluation.runner import (
    _accelerator_usage,
    _reset_accelerator_peak_memory,
)


class _Cuda:
    def __init__(self) -> None:
        self.reset_count = 0

    @staticmethod
    def is_available() -> bool:
        return True

    def reset_peak_memory_stats(self) -> None:
        self.reset_count += 1

    @staticmethod
    def current_device() -> int:
        return 0

    @staticmethod
    def get_device_name(index: int) -> str:
        assert index == 0
        return "Fixture GPU"

    @staticmethod
    def max_memory_allocated(index: int) -> int:
        assert index == 0
        return 4096


def test_loaded_accelerator_runtime_is_observed_without_importing_it(
    monkeypatch,
) -> None:
    """Evaluation reuses optional telemetry from an already-loaded runtime."""
    cuda = _Cuda()
    monkeypatch.setitem(sys.modules, "torch", SimpleNamespace(cuda=cuda))

    _reset_accelerator_peak_memory()
    name, peak = _accelerator_usage()

    assert cuda.reset_count == 1
    assert name == "Fixture GPU"
    assert peak == 4096
