"""Evaluate the retained M48 candidate on the frozen dev-200 split."""

from __future__ import annotations

import os

os.environ["LEGAL_RAG_CANDIDATE"] = "m48"

import kaggle_candidate_dev_common as candidate


def main() -> None:
    """Run M48 against the immutable measured M47 control values."""
    candidate.CONTROL_SCORES = {
        "rouge": 0.3304165218171184,
        "meteor": 0.2341513827104324,
    }
    candidate.CONTROL_FALLBACK_COUNT = 14
    candidate.main()


if __name__ == "__main__":
    main()
