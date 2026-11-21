# --- builder ---
    FROM python:3.11-slim AS builder
    COPY --from=ghcr.io/astral-sh/uv:latest /uv /uvx /bin/
    WORKDIR /app
    COPY pyproject.toml uv.lock README.md ./
    # No --mount cache: Cloud Build's legacy builder lacks BuildKit support
    RUN uv sync --frozen --no-dev --no-install-project
    
    # --- runtime ---
    FROM python:3.11-slim
    # Patch base OS packages (debian CVEs) + pip ecosystem. Explicit version
    # floors force pip to upgrade — CVE-2026-24049 (wheel), CVE-2026-23949
    # (jaraco.context), CVE-2025-47273 (setuptools, flagged by Trivy in the
    # stock slim image; plain `--upgrade setuptools` silently no-op'd).
    RUN apt-get update && apt-get -y upgrade && rm -rf /var/lib/apt/lists/* \
        && pip install --no-cache-dir --upgrade pip \
        && pip install --no-cache-dir --upgrade "setuptools>=78.1.1" "wheel>=0.46.2" "jaraco.context>=6.1.0"
    WORKDIR /app
    COPY --from=builder /app/.venv /app/.venv
    COPY . .
    ENV PATH="/app/.venv/bin:$PATH" PYTHONUNBUFFERED=1 PORT=8080 UV_LINK_MODE=copy
    EXPOSE 8080
    CMD ["uvicorn", "api.main:app", "--host", "0.0.0.0", "--port", "8080"]