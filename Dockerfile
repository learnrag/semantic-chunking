FROM python:3.12-slim

WORKDIR /app

# System deps sometimes needed by torch wheels on slim.
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

COPY pyproject.toml README.md ./
COPY src ./src

RUN pip install --no-cache-dir .

EXPOSE 8000

# Bind to 0.0.0.0 for container/host networking (Render-compatible PORT).
CMD ["sh", "-c", "uvicorn chunker.api:app --host 0.0.0.0 --port ${PORT:-8000}"]
