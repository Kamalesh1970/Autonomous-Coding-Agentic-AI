"""Utility script to clean up temporary scratch files from repository root."""

import os
import shutil
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent

SCRATCH_FILES = [
    "Test gemini thought signature ·.py",
    "check_api_keys.py",
    "run_direct.py",
    "run_failing_check.py",
    "run_full_pytest.py",
    "run_pytest_check.py",
    "run_semantic_test.py",
    "run_tests_now.sh",
    "agent_run_final.log",
    "agent_run_full.log",
    "agent_run_verified.log",
    "pytest_run.log",
]

SCRATCH_DIRS = [
    "my-repo",
    "autonomous_coding_agent.egg-info",
]

def clean_repository_root():
    """Removes temporary debugging scratch files and disposable test directories."""
    for item in SCRATCH_FILES:
        target = REPO_ROOT / item
        if target.exists():
            try:
                target.unlink()
                print(f"Removed scratch file: {item}")
            except Exception as e:
                print(f"Failed to remove {item}: {e}")

    for item in SCRATCH_DIRS:
        target = REPO_ROOT / item
        if target.exists():
            try:
                shutil.rmtree(target)
                print(f"Removed scratch dir: {item}")
            except Exception as e:
                print(f"Failed to remove dir {item}: {e}")

if __name__ == "__main__":
    clean_repository_root()
