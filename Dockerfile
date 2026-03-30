FROM python:3.13-slim

WORKDIR /app

# Install dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy source code
COPY common/ common/
COPY agents/ agents/

ENV PYTHONPATH=/app
ENV PYTHONUNBUFFERED=1
