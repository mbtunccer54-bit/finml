# syntax=docker/dockerfile:1.7
# =============================================================================
# Multi-stage build. `runtime` is shared by the api and dashboard services;
# the command differs per service in docker-compose.yml.
# =============================================================================

FROM python:3.11-slim AS base

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1

# libgomp1 is required by LightGBM/XGBoost; curl backs the container healthchecks.
RUN apt-get update \
    && apt-get install --no-install-recommends -y libgomp1 curl \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# -----------------------------------------------------------------------------
FROM base AS builder

COPY pyproject.toml README.md ./
COPY src ./src

RUN python -m venv /opt/venv \
    && /opt/venv/bin/pip install --upgrade pip \
    && /opt/venv/bin/pip install .

# -----------------------------------------------------------------------------
FROM base AS runtime

COPY --from=builder /opt/venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH" \
    PYTHONPATH=/app/src

COPY configs ./configs
COPY src ./src

# Non-root runtime user; artifact dirs are created up front so bind mounts and
# named volumes inherit the right ownership.
RUN useradd --create-home --uid 1000 finml \
    && mkdir -p /app/artifacts /app/mlruns /app/data/feature_store \
    && chown -R finml:finml /app
USER finml

EXPOSE 8000 8501

CMD ["uvicorn", "infrastructure.api.fastapi_app:app", "--host", "0.0.0.0", "--port", "8000"]
