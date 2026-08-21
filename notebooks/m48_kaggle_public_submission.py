"""Run retained M48 public inference through the shared Kaggle runner."""

from __future__ import annotations

import os

os.environ["LEGAL_RAG_PUBLIC_CANDIDATE"] = "m48"

import kaggle_public_submission_common as public


if __name__ == "__main__":
    public.main()
