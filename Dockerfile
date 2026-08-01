# syntax=docker/dockerfile:1

ARG PYTHON_IMAGE=python:3.11.11-slim-bookworm
FROM ${PYTHON_IMAGE}

LABEL org.opencontainers.image.title="Vietnamese Legal Agentic RAG"
LABEL org.opencontainers.image.licenses="MIT"

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    PIP_NO_CACHE_DIR=1

RUN groupadd --system legalrag \
    && useradd --system --gid legalrag --create-home legalrag

WORKDIR /app

COPY pyproject.toml README.md LICENSE ./
COPY constraints/competition-direct.txt ./constraints/competition-direct.txt
COPY src ./src

RUN python -m pip install --upgrade "pip==24.3.1" "setuptools==75.6.0" \
    && python -m pip install \
        --constraint constraints/competition-direct.txt \
        .

USER legalrag

# The concrete command and mounted config/artifact paths are supplied by the
# organizer reproduction command. No dataset, model, artifact, or secret is
# embedded in this image.
ENTRYPOINT ["legal-rag-serve"]
CMD ["--help"]
