# --- Stage 1: Builder ---
FROM python:3.12-slim AS builder

COPY --from=ghcr.io/astral-sh/uv:latest /uv /usr/local/bin/uv

ENV UV_EXTRA_INDEX_URL=https://download.pytorch.org/whl/cpu

WORKDIR /app

# Copy dependency files first for Docker layer caching
COPY pyproject.toml uv.lock ./

# Install CPU-only deps
RUN uv sync --locked
RUN uv run python -c \
    "from sentence_transformers import SentenceTransformer; \
    SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')"

COPY app/ ./app/

# --- Stage 2: Runtime ---
FROM python:3.12-slim

# For HF Spaces
RUN useradd -m -u 1000 user

WORKDIR /app

COPY --from=builder /app /app
COPY --from=builder --chown=user /root/.cache/huggingface /home/user/.cache/huggingface

ENV DEVICE=cpu
ENV PATH="/app/.venv/bin:${PATH}"
ENV HF_HOME="/home/user/.cache/huggingface"

USER user

EXPOSE 7860

CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "7860"]
