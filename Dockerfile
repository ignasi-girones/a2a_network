FROM python:3.13-slim

WORKDIR /app

# Install dependencies
COPY pyproject.toml .
RUN pip install --no-cache-dir "a2a-sdk[http,sqlite]==1.0.0a0" \
    litellm uvicorn httpx pydantic pydantic-settings \
    sse-starlette packaging mcp

# Copy source code
COPY common/ common/
COPY agents/ agents/

ENV PYTHONPATH=/app
ENV PYTHONUNBUFFERED=1
