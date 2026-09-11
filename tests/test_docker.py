"""Phase 16J Pytest suite for Docker Containerization."""

import os
import shutil
import subprocess
from pathlib import Path
import pytest


REPO_ROOT = Path(__file__).parent.parent
DOCKERFILE_PATH = REPO_ROOT / "Dockerfile"
DOCKERIGNORE_PATH = REPO_ROOT / ".dockerignore"


def is_docker_available() -> bool:
    """Helper checking if docker CLI and daemon are accessible in execution environment."""
    if not shutil.which("docker"):
        return False
    try:
        res = subprocess.run(["docker", "info"], capture_output=True, timeout=5)
        return res.returncode == 0
    except Exception:
        return False


# -----------------------------------------------------------------------------
# 1. test_dockerfile_exists
# -----------------------------------------------------------------------------

def test_dockerfile_exists():
    """Verify Dockerfile exists at repository root."""
    assert DOCKERFILE_PATH.exists()
    assert DOCKERFILE_PATH.is_file()


# -----------------------------------------------------------------------------
# 2. test_dockerfile_base_image
# -----------------------------------------------------------------------------

def test_dockerfile_base_image():
    """Verify Dockerfile uses Python 3.12 compatible base image."""
    content = DOCKERFILE_PATH.read_text(encoding="utf-8")
    assert "FROM python:3.12" in content or "FROM python:3.12-slim" in content


# -----------------------------------------------------------------------------
# 3. test_dockerfile_no_secrets
# -----------------------------------------------------------------------------

def test_dockerfile_no_secrets():
    """Verify Dockerfile does not contain hardcoded API keys or environment secrets."""
    content = DOCKERFILE_PATH.read_text(encoding="utf-8")
    forbidden_patterns = [
        "sk-", "AIzaSy", "ghp_", "GEMINI_API_KEY=", "OPENAI_API_KEY=", "OPENROUTER_API_KEY="
    ]
    for pattern in forbidden_patterns:
        assert pattern not in content


# -----------------------------------------------------------------------------
# 4. test_dockerfile_no_env_copy
# -----------------------------------------------------------------------------

def test_dockerfile_no_env_copy():
    """Verify Dockerfile does not explicitly copy .env files into the image."""
    content = DOCKERFILE_PATH.read_text(encoding="utf-8")
    lines = content.splitlines()
    for line in lines:
        stripped = line.strip()
        if stripped.startswith("COPY") or stripped.startswith("ADD"):
            assert ".env" not in stripped


# -----------------------------------------------------------------------------
# 5. test_dockerfile_non_root_user
# -----------------------------------------------------------------------------

def test_dockerfile_non_root_user():
    """Verify Dockerfile configures execution under a non-root USER."""
    content = DOCKERFILE_PATH.read_text(encoding="utf-8")
    assert "USER " in content
    assert "USER root" not in content


# -----------------------------------------------------------------------------
# 6. test_dockerfile_exposes_port
# -----------------------------------------------------------------------------

def test_dockerfile_exposes_port():
    """Verify Dockerfile exposes port 8000 for the web dashboard."""
    content = DOCKERFILE_PATH.read_text(encoding="utf-8")
    assert "EXPOSE 8000" in content


# -----------------------------------------------------------------------------
# 7. test_dockerfile_contains_healthcheck
# -----------------------------------------------------------------------------

def test_dockerfile_contains_healthcheck():
    """Verify Dockerfile includes a container HEALTHCHECK instruction."""
    content = DOCKERFILE_PATH.read_text(encoding="utf-8")
    assert "HEALTHCHECK" in content


# -----------------------------------------------------------------------------
# 8. test_dockerfile_healthcheck_targets_health
# -----------------------------------------------------------------------------

def test_dockerfile_healthcheck_targets_health():
    """Verify HEALTHCHECK targets the existing /health endpoint."""
    content = DOCKERFILE_PATH.read_text(encoding="utf-8")
    assert "/health" in content


# -----------------------------------------------------------------------------
# 9. test_dockerfile_startup_command_matches_entrypoint
# -----------------------------------------------------------------------------

def test_dockerfile_startup_command_matches_entrypoint():
    """Verify CMD matches real FastAPI uvicorn entrypoint (app.web:app)."""
    content = DOCKERFILE_PATH.read_text(encoding="utf-8")
    assert "uvicorn" in content
    assert "app.web:app" in content


# -----------------------------------------------------------------------------
# 10. test_dockerignore_exists
# -----------------------------------------------------------------------------

def test_dockerignore_exists():
    """Verify .dockerignore exists at repository root."""
    assert DOCKERIGNORE_PATH.exists()
    assert DOCKERIGNORE_PATH.is_file()


# -----------------------------------------------------------------------------
# 11. test_dockerignore_excludes_env
# -----------------------------------------------------------------------------

def test_dockerignore_excludes_env():
    """Verify .dockerignore excludes .env secret files."""
    content = DOCKERIGNORE_PATH.read_text(encoding="utf-8")
    lines = [l.strip() for l in content.splitlines() if l.strip() and not l.startswith("#")]
    assert ".env" in lines or "*.env" in lines or ".env.*" in lines


# -----------------------------------------------------------------------------
# 12. test_dockerignore_excludes_venv
# -----------------------------------------------------------------------------

def test_dockerignore_excludes_venv():
    """Verify .dockerignore excludes local virtual environments."""
    content = DOCKERIGNORE_PATH.read_text(encoding="utf-8")
    lines = [l.strip() for l in content.splitlines() if l.strip() and not l.startswith("#")]
    assert ".venv" in lines
    assert "venv" in lines


# -----------------------------------------------------------------------------
# 13. test_dockerignore_excludes_git
# -----------------------------------------------------------------------------

def test_dockerignore_excludes_git():
    """Verify .dockerignore excludes .git repository metadata."""
    content = DOCKERIGNORE_PATH.read_text(encoding="utf-8")
    lines = [l.strip() for l in content.splitlines() if l.strip() and not l.startswith("#")]
    assert ".git" in lines


# -----------------------------------------------------------------------------
# 14. test_docker_no_privileged_execution
# -----------------------------------------------------------------------------

def test_docker_no_privileged_execution():
    """Verify Docker configuration does not request privileged mode."""
    content = DOCKERFILE_PATH.read_text(encoding="utf-8")
    assert "--privileged" not in content
    assert "privileged" not in content.lower()


# -----------------------------------------------------------------------------
# 15. test_sandbox_preservation
# -----------------------------------------------------------------------------

def test_sandbox_preservation():
    """Verify existing application-level ExecutionSandbox is preserved."""
    from app.sandbox import ExecutionSandbox
    sandbox = ExecutionSandbox(sandbox_root=".")
    assert sandbox.sandbox_root.name != ""
    # Command allowlist policy check
    allowed, _ = sandbox.is_command_allowed(["pytest"])
    assert allowed is True
    rejected, _ = sandbox.is_command_allowed(["bash", "-c", "rm -rf /"])
    assert rejected is False


# -----------------------------------------------------------------------------
# 16. test_web_preservation
# -----------------------------------------------------------------------------

def test_web_preservation():
    """Verify FastAPI web app entrypoint and health route function correctly."""
    from fastapi.testclient import TestClient
    from app.web import app
    client = TestClient(app)
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json()["status"] == "ok"


# -----------------------------------------------------------------------------
# 17. test_docker_build_and_run_smoke_test (Optional live Docker daemon check)
# -----------------------------------------------------------------------------

def test_docker_build_and_run_smoke_test():
    """Builds and runs Docker container if Docker daemon is available."""
    if not is_docker_available():
        pytest.skip("Docker daemon is not available in the current execution environment.")

    image_tag = "autonomous-coding-agent:test"
    try:
        # Build image
        build_res = subprocess.run(
            ["docker", "build", "-t", image_tag, "."],
            cwd=str(REPO_ROOT),
            capture_output=True,
            text=True,
            timeout=120,
        )
        assert build_res.returncode == 0

        # Run container
        container_name = "test_agent_container_smoke"
        run_res = subprocess.run(
            ["docker", "run", "-d", "--name", container_name, "-p", "8005:8000", image_tag],
            capture_output=True,
            text=True,
            timeout=15,
        )
        assert run_res.returncode == 0

        # Check health endpoint inside container
        import time
        time.sleep(3)
        import urllib.request
        resp = urllib.request.urlopen("http://localhost:8005/health", timeout=5)
        assert resp.status == 200

    finally:
        # Cleanup container and image
        subprocess.run(["docker", "rm", "-f", "test_agent_container_smoke"], capture_output=True)
        subprocess.run(["docker", "rmi", "-f", image_tag], capture_output=True)
