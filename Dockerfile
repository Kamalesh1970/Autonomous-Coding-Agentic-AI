# Phase 16J — Production Dockerfile for Autonomous Coding Agent

FROM python:3.12-slim

# Set environment variables for Python runtime optimization
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1

# Set working directory
WORKDIR /app

# Install dependencies
COPY requirements.txt pyproject.toml /app/
RUN pip install --no-cache-dir -r requirements.txt

# Copy application source code and web static files
COPY app/ /app/app/
COPY web/ /app/web/
COPY README.md /app/

# Create a non-root system user and set workspace directory permissions
RUN useradd -m -u 1000 agentuser && \
    mkdir -p /app/.agent_memory && \
    chown -R agentuser:agentuser /app

# Switch to non-root execution
USER agentuser

# Expose FastAPI application port
EXPOSE 8000

# Container healthcheck targeting the existing /health endpoint
HEALTHCHECK --interval=30s --timeout=5s --start-period=5s --retries=3 \
  CMD python3 -c "import urllib.request; urllib.request.urlopen('http://localhost:8000/health')"

# FastAPI entrypoint using Uvicorn
CMD ["uvicorn", "app.web:app", "--host", "0.0.0.0", "--port", "8000"]
