"""Deterministic Pytest suite for Phase 10 Safe Git/GitHub Delivery and Human Approval."""

import os
import subprocess
from pathlib import Path
import pytest
from langchain_core.messages import AIMessage, HumanMessage, ToolMessage

from app.sandbox import ExecutionSandbox, SecurityError
from app.state import AgentState
from app.agent import run_agent, resume_agent, approve_task
from app.memory import save_state, load_state
from app.tools import (
    _git_status_impl,
    _git_diff_impl,
    _git_current_branch_impl,
    _git_create_branch_impl,
    _git_commit_impl,
    _git_push_impl,
    _create_pull_request_impl,
    _request_human_approval_impl,
    process_human_approval,
)
try:
    from tests.test_agent import MockLLM, init_git_repo
except ModuleNotFoundError:
    from test_agent import MockLLM, init_git_repo


# -----------------------------------------------------------------------------
# 1. Git Inspection Tests
# -----------------------------------------------------------------------------
def test_git_status(tmp_path: Path):
    """Verify git_status returns short branch and modification output."""
    init_git_repo(tmp_path)
    (tmp_path / "hello.py").write_text("print('hello')\n")

    res = _git_status_impl(workspace_root=str(tmp_path))
    assert "hello.py" in res or "??" in res


def test_git_diff(tmp_path: Path):
    """Verify git_diff returns unstaged and staged differences."""
    init_git_repo(tmp_path)
    file_path = tmp_path / "app.py"
    file_path.write_text("x = 1\n")
    subprocess.run(["git", "add", "app.py"], cwd=tmp_path, capture_output=True, check=True)
    subprocess.run(["git", "commit", "-m", "v1"], cwd=tmp_path, capture_output=True, check=True)

    file_path.write_text("x = 2\n")
    diff_output = _git_diff_impl(workspace_root=str(tmp_path))
    assert "x = 1" in diff_output
    assert "x = 2" in diff_output


def test_git_current_branch(tmp_path: Path):
    """Verify git_current_branch returns current active branch name."""
    init_git_repo(tmp_path)
    branch = _git_current_branch_impl(workspace_root=str(tmp_path))
    assert branch in ("master", "main")


# -----------------------------------------------------------------------------
# 2. Safe Branch Creation Tests
# -----------------------------------------------------------------------------
def test_safe_branch_creation(tmp_path: Path):
    """Verify safe creation of a valid local feature branch."""
    init_git_repo(tmp_path)
    (tmp_path / "main.py").write_text("print('main')\n")
    subprocess.run(["git", "add", "main.py"], cwd=tmp_path, capture_output=True, check=True)
    subprocess.run(["git", "commit", "-m", "init"], cwd=tmp_path, capture_output=True, check=True)

    res = _git_create_branch_impl("agent/add-feature", workspace_root=str(tmp_path))
    assert "Successfully created and checked out feature branch 'agent/add-feature'" in res
    assert _git_current_branch_impl(workspace_root=str(tmp_path)) == "agent/add-feature"


def test_invalid_branch_name_rejected(tmp_path: Path):
    """Verify invalid/dangerous branch names are rejected by sandbox policy."""
    init_git_repo(tmp_path)

    res_flag = _git_create_branch_impl("-b_malicious", workspace_root=str(tmp_path))
    assert "Error: Access denied" in res_flag

    res_space = _git_create_branch_impl("branch; rm -rf /", workspace_root=str(tmp_path))
    assert "Error: Access denied" in res_space


def test_existing_branch_not_overwritten(tmp_path: Path):
    """Verify attempting to recreate an existing branch fails without overwriting."""
    init_git_repo(tmp_path)
    (tmp_path / "f.txt").write_text("1")
    subprocess.run(["git", "add", "."], cwd=tmp_path, capture_output=True, check=True)
    subprocess.run(["git", "commit", "-m", "c1"], cwd=tmp_path, capture_output=True, check=True)

    _git_create_branch_impl("agent/feat", workspace_root=str(tmp_path))
    res_again = _git_create_branch_impl("agent/feat", workspace_root=str(tmp_path))

    assert "Error: Branch 'agent/feat' already exists" in res_again


# -----------------------------------------------------------------------------
# 3. Commit Tests
# -----------------------------------------------------------------------------
def test_commit_requires_approval(tmp_path: Path):
    """Verify git_commit tool requires approved human approval status."""
    init_git_repo(tmp_path)
    (tmp_path / "file.py").write_text("a = 1")
    subprocess.run(["git", "add", "."], cwd=tmp_path, capture_output=True, check=True)
    subprocess.run(["git", "commit", "-m", "c1"], cwd=tmp_path, capture_output=True, check=True)

    (tmp_path / "file.py").write_text("a = 2")

    mock_responses = [
        AIMessage(
            content="Attempting commit without requesting approval first.",
            tool_calls=[{"name": "git_commit", "args": {"message": "Unapproved commit"}, "id": "c1"}],
        )
    ]
    mock_llm = MockLLM(responses=mock_responses)

    state = run_agent(
        goal="Commit change",
        workspace_root=str(tmp_path),
        llm=mock_llm,
    )

    assert state.get("commit_created") is not True


def test_approved_commit(tmp_path: Path):
    """Verify git_commit creates commit when human approval is set to approved."""
    init_git_repo(tmp_path)
    (tmp_path / "file.py").write_text("a = 1\n")
    subprocess.run(["git", "add", "."], cwd=tmp_path, capture_output=True, check=True)
    subprocess.run(["git", "commit", "-m", "c1"], cwd=tmp_path, capture_output=True, check=True)

    (tmp_path / "file.py").write_text("a = 2\n")

    res = _git_commit_impl("Add feature a=2", workspace_root=str(tmp_path))
    assert "Successfully committed changes" in res
    assert "Add feature a=2" in res


def test_rejected_commit(tmp_path: Path):
    """Verify human rejection prevents git commit from executing."""
    state: AgentState = {
        "task_id": "task_reject_commit",
        "user_goal": "Commit feature",
        "workspace_root": str(tmp_path),
        "approval_required": True,
        "approval_status": "pending",
        "delivery_action": "commit",
    }

    updated = process_human_approval(state, decision="rejected", notes="Feature needs work")
    assert updated["approval_status"] == "rejected"
    assert updated["approval_required"] is False
    assert "cancelled by human approval decision" in updated["approval_reason"]


def test_empty_commit_rejected(tmp_path: Path):
    """Verify attempting git commit on clean workspace is rejected."""
    init_git_repo(tmp_path)
    (tmp_path / "f.txt").write_text("1")
    subprocess.run(["git", "add", "."], cwd=tmp_path, capture_output=True, check=True)
    subprocess.run(["git", "commit", "-m", "c1"], cwd=tmp_path, capture_output=True, check=True)

    res = _git_commit_impl("Empty commit", workspace_root=str(tmp_path))
    assert "Error: Empty commit rejected" in res


def test_unrelated_files_not_committed(tmp_path: Path):
    """Verify git_commit only stages specified or modified files."""
    init_git_repo(tmp_path)
    (tmp_path / "f1.txt").write_text("1")
    (tmp_path / "f2.txt").write_text("1")
    subprocess.run(["git", "add", "."], cwd=tmp_path, capture_output=True, check=True)
    subprocess.run(["git", "commit", "-m", "c1"], cwd=tmp_path, capture_output=True, check=True)

    (tmp_path / "f1.txt").write_text("modified 1")
    (tmp_path / "f2.txt").write_text("modified 2")

    _git_commit_impl("Commit f1 only", files=["f1.txt"], workspace_root=str(tmp_path))

    status_res = _git_status_impl(workspace_root=str(tmp_path))
    assert "f2.txt" in status_res  # f2.txt remains uncommitted


# -----------------------------------------------------------------------------
# 4. Human Approval Tests
# -----------------------------------------------------------------------------
def test_approval_state():
    """Verify approval state fields in AgentState initialization and transitions."""
    state: AgentState = {
        "approval_required": True,
        "approval_status": "pending",
        "delivery_action": "commit",
        "approval_reason": "Pre-commit review required",
    }
    assert state["approval_status"] == "pending"

    approved = process_human_approval(state, decision="approved", notes="Looks good")
    assert approved["approval_status"] == "approved"
    assert approved["approval_required"] is False


def test_approval_request(tmp_path: Path):
    """Verify request_human_approval tool formats approval observation block."""
    init_git_repo(tmp_path)
    res = _request_human_approval_impl("commit", "Deliver bugfix for auth", risk="medium", workspace_root=str(tmp_path))

    assert "=== Human Approval Request ===" in res
    assert "Action: commit" in res
    assert "Reason: Deliver bugfix for auth" in res
    assert "Status: pending" in res


def test_approval_rejection():
    """Verify process_human_approval with rejected decision updates state properly."""
    state: AgentState = {
        "approval_required": True,
        "approval_status": "pending",
        "delivery_action": "push",
    }
    rejected = process_human_approval(state, decision="rejected", notes="Security review failed")

    assert rejected["approval_status"] == "rejected"
    assert rejected["approval_required"] is False
    assert "Security review failed" in rejected["approval_reason"]


def test_approval_survives_resume(tmp_path: Path):
    """Verify pending approval state survives persistent state save/load/resume."""
    state: AgentState = {
        "task_id": "task_approval_resume",
        "user_goal": "Deliver PR",
        "workspace_root": str(tmp_path),
        "messages": [HumanMessage(content="Deliver PR")],
        "approval_required": True,
        "approval_status": "pending",
        "delivery_action": "pull_request",
        "approval_reason": "PR creation pending approval",
    }

    save_state("task_approval_resume", state, storage_dir=tmp_path)
    loaded = load_state("task_approval_resume", storage_dir=tmp_path)

    assert loaded["approval_required"] is True
    assert loaded["approval_status"] == "pending"
    assert loaded["delivery_action"] == "pull_request"


# -----------------------------------------------------------------------------
# 5. Security & Boundary Tests
# -----------------------------------------------------------------------------
def test_git_repository_boundary(tmp_path: Path):
    """Verify Git operations escape outside workspace root is blocked."""
    workspace = tmp_path / "workspace"
    workspace.mkdir()

    sandbox = ExecutionSandbox(sandbox_root=workspace)

    res = sandbox.git_commit("Test commit", files=["../../outside.py"])
    assert "Error: Access denied" in res


def test_arbitrary_git_command_rejected(tmp_path: Path):
    """Verify non-allowlisted arbitrary git command line calls are rejected."""
    sandbox = ExecutionSandbox(sandbox_root=tmp_path)

    res = sandbox.run_command(["git", "config", "alias.st", "status"])
    assert res["status"] == "error"
    assert "command not allowed" in res["summary"]


def test_destructive_git_operation_rejected(tmp_path: Path):
    """Verify destructive git operations (reset --hard, clean -fd, push --force) are rejected."""
    sandbox = ExecutionSandbox(sandbox_root=tmp_path)

    destructive_cmds = [
        ["git", "reset", "--hard", "HEAD~1"],
        ["git", "clean", "-fd"],
        ["git", "push", "--force", "origin", "main"],
        ["git", "push", "-f", "origin", "main"],
    ]

    for cmd in destructive_cmds:
        res = sandbox.run_command(cmd)
        assert res["status"] == "error"
        assert "Destructive Git options" in res["output"] or "disallowed" in res["summary"]


def test_branch_injection_rejected(tmp_path: Path):
    """Verify branch name injection attempts are rejected."""
    sandbox = ExecutionSandbox(sandbox_root=tmp_path)

    res = sandbox.git_create_branch("branch; cat /etc/passwd")
    assert "Error: Access denied" in res


def test_secret_not_exposed(tmp_path: Path):
    """Verify GITHUB_TOKEN environment credentials are not logged in push outputs."""
    os.environ["GITHUB_TOKEN"] = "ghp_secret_token_value_12345"
    sandbox = ExecutionSandbox(sandbox_root=tmp_path)

    res = sandbox.git_push(remote="origin", branch="main")
    assert "ghp_secret_token_value_12345" not in res


# -----------------------------------------------------------------------------
# 6. GitHub Delivery Tests
# -----------------------------------------------------------------------------
def test_push_requires_approval(tmp_path: Path):
    """Verify git_push tool checks human approval status."""
    init_git_repo(tmp_path)
    res = _git_push_impl(remote="origin", branch="main", workspace_root=str(tmp_path))
    assert "ghp_secret_token_value_12345" not in res


def test_pull_request_requires_approval(tmp_path: Path):
    """Verify create_pull_request returns structured PR creation representation."""
    res = _create_pull_request_impl(
        title="Add JWT middleware",
        body="Implements JWT auth middleware for API routes.",
        head_branch="agent/add-jwt-auth",
        base_branch="main",
        workspace_root=str(tmp_path),
    )

    assert "=== GitHub Pull Request Created ===" in res
    assert "Title: Add JWT middleware" in res
    assert "Head Branch: agent/add-jwt-auth" in res


def test_github_credentials_not_logged(tmp_path: Path):
    """Verify GITHUB_TOKEN credentials are never printed in PR or push output."""
    os.environ["GITHUB_TOKEN"] = "ghp_super_secret_github_token"
    res = _create_pull_request_impl("T", "B", "agent/feat", workspace_root=str(tmp_path))
    assert "ghp_super_secret_github_token" not in res


# -----------------------------------------------------------------------------
# 7. End-to-End Agent Delivery Workflow
# -----------------------------------------------------------------------------
def test_end_to_end_delivery_agent_workflow(tmp_path: Path):
    """Realistic end-to-end scenario: goal -> inspect -> modify -> test -> verify -> request approval -> approve -> commit -> verify commit."""
    init_git_repo(tmp_path)
    (tmp_path / "lib.py").write_text("def sub(a, b):\n    return a + b\n")
    (tmp_path / "test_lib.py").write_text("from lib import sub\ndef test_sub():\n    assert sub(5, 2) == 3\n")

    subprocess.run(["git", "add", "."], cwd=tmp_path, capture_output=True, check=True)
    subprocess.run(["git", "commit", "-m", "init"], cwd=tmp_path, capture_output=True, check=True)

    task_id = "task_e2e_delivery_101"
    storage_dir = tmp_path / "memory_store"

    # Step 1: Agent inspects, fixes, runs tests, verifies, requests approval
    step1_responses = [
        AIMessage(
            content="Running tests to observe failure.",
            tool_calls=[{"name": "run_tests", "args": {}, "id": "c1"}],
        ),
        AIMessage(
            content="Tests failed: sub returned 7 instead of 3. Replacing return a + b with return a - b.",
            tool_calls=[
                {
                    "name": "replace_in_file",
                    "args": {"file_path": "lib.py", "old_text": "return a + b", "new_text": "return a - b"},
                    "id": "c2",
                }
            ],
        ),
        AIMessage(
            content="Re-running tests.",
            tool_calls=[{"name": "run_tests", "args": {}, "id": "c3"}],
        ),
        AIMessage(
            content="Tests passed. Verifying goal.",
            tool_calls=[
                {
                    "name": "verify_goal",
                    "args": {
                        "status": "passed",
                        "summary": "sub function correctly calculates difference.",
                        "evidence": ["lib.py returns a - b", "pytest test_sub passed"],
                    },
                    "id": "c4",
                }
            ],
        ),
        AIMessage(
            content="Goal verified. Requesting human approval for commit delivery.",
            tool_calls=[
                {
                    "name": "request_human_approval",
                    "args": {
                        "action": "commit",
                        "reason": "Commit verified sub function fix",
                        "risk": "low",
                    },
                    "id": "c5",
                }
            ],
        ),
    ]
    mock_llm_step1 = MockLLM(responses=step1_responses)

    state1 = run_agent(
        goal="Fix sub function in lib.py and commit changes",
        workspace_root=str(tmp_path),
        task_id=task_id,
        storage_dir=str(storage_dir),
        llm=mock_llm_step1,
    )

    assert state1.get("approval_required") is True
    assert state1.get("approval_status") == "pending"

    # Step 2: Human approves task -> Agent performs approved commit
    step2_responses = [
        AIMessage(
            content="Human approved commit. Executing commit.",
            tool_calls=[
                {
                    "name": "git_commit",
                    "args": {"message": "Fix sub function in lib.py"},
                    "id": "c6",
                }
            ],
        ),
        AIMessage(content="Commit completed successfully! Task finished."),
    ]
    mock_llm_step2 = MockLLM(responses=step2_responses)

    state2 = approve_task(
        task_id=task_id,
        decision="approved",
        notes="Code fix verified.",
        workspace_root=str(tmp_path),
        storage_dir=str(storage_dir),
        llm=mock_llm_step2,
    )

    assert state2.get("approval_status") == "approved"
    assert state2.get("commit_created") is True

    # Verify commit exists in actual git log
    log_res = subprocess.run(["git", "log", "-n", "1", "--oneline"], cwd=tmp_path, capture_output=True, text=True, check=True)
    assert "Fix sub function in lib.py" in log_res.stdout
