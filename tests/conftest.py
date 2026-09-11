"""Pytest configuration and session-level setup hooks for Autonomous Coding Agent test suite."""

import sys
from pathlib import Path

# Ensure project root is in sys.path
REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts.cleanup_scratch import clean_repository_root

# Trigger cleanup on conftest import
clean_repository_root()


def pytest_sessionstart(session):
    """Pytest session start hook ensuring repository root is clean of temporary scratch artifacts."""
    clean_repository_root()
