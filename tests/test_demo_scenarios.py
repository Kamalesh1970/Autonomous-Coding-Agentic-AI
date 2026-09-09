"""Phase 16D — Professional Demonstration Scenarios Test Suite.

Proves major autonomous capabilities of the Autonomous Coding Agent across 9 core scenarios:
1. Simple Success
2. Real Bug Fix
3. Self-Correction / Recovery
4. Repository Understanding / Retrieval
5. Multi-Agent Workflow
6. Security / Sandbox Rejection
7. Human Approval Boundary
8. Failure / Escalation
9. Final Execution Report
"""

import os
import sys
from pathlib import Path
from unittest.mock import patch, MagicMock
import pytest
from langchain_core.messages import AIMessage, HumanMessage, ToolMessage

from app.agent import main, run_agent, sanitize_log_output
from app.multi_agent import run_multi_agent
from tests.test_agent import MockLLM, init_git_repo
from tests.test_multi_agent import MockLLM as MultiAgentMockLLM


# -----------------------------------------------------------------------------
# Demo 1 — Simple Success
# -----------------------------------------------------------------------------
def test_demo_1_simple_success(tmp_path: Path):
    """Demo 1: Simple repository-level feature addition that completes successfully."""
    init_git_repo(tmp_path)
    (tmp_path / "calculator.py").write_text("def add(a, b):\n    return a + b\n")
    (tmp_path / "test_calculator.py").write_text("from calculator import add\n\ndef test_add():\n    assert add(2, 3) == 5\n")

    responses = [
        AIMessage(
            content="Planning task",
            tool_calls=[{
                "name": "create_plan",
                "args": {"goal": "Add divide function", "tasks": [{"id": "task-1", "title": "Add divide", "description": "Add divide"}]},
                "id": "c1",
            }],
        ),
        AIMessage(
            content="Writing divide function",
            tool_calls=[{
                "name": "write_file",
                "args": {
                    "file_path": "calculator.py",
                    "content": "def add(a, b):\n    return a + b\n\ndef divide(a, b):\n    return a / b\n"
                },
                "id": "c2",
            }],
        ),
        AIMessage(
            content="Writing test for divide",
            tool_calls=[{
                "name": "write_file",
                "args": {
                    "file_path": "test_calculator.py",
                    "content": "from calculator import add, divide\n\ndef test_add():\n    assert add(2, 3) == 5\n\ndef test_divide():\n    assert divide(10, 2) == 5\n"
                },
                "id": "c3",
            }],
        ),
        AIMessage(
            content="Running validation tests",
            tool_calls=[{"name": "run_tests", "args": {}, "id": "c4"}],
        ),
        AIMessage(
            content="Verifying goal satisfaction",
            tool_calls=[{
                "name": "verify_goal",
                "args": {"status": "passed", "summary": "Divide function added and tests pass", "evidence": ["test_divide passed"]},
                "id": "c5",
            }],
        ),
    ]
    mock_llm = MockLLM(responses=responses)

    state = run_agent(
        goal="Add a divide function to calculator.py and make sure the tests pass.",
        workspace_root=str(tmp_path),
        llm=mock_llm,
    )

    assert state.get("status") == "completed"
    assert state.get("final_outcome") == "SUCCESS"
    assert "calculator.py" in state.get("modified_files", [])
    assert state.get("verification_result", {}).get("status") == "passed"
    code = (tmp_path / "calculator.py").read_text()
    assert "def divide" in code


# -----------------------------------------------------------------------------
# Demo 2 — Real Bug Fix
# -----------------------------------------------------------------------------
def test_demo_2_real_bug_fix(tmp_path: Path):
    """Demo 2: Known bug investigation and resolution in isolated repository."""
    init_git_repo(tmp_path)
    (tmp_path / "calculator.py").write_text("def multiply(a, b):\n    return a + b\n")
    (tmp_path / "test_calculator.py").write_text("from calculator import multiply\n\ndef test_multiply():\n    assert multiply(4, 5) == 20\n")

    responses = [
        AIMessage(
            content="Running initial test to observe failure",
            tool_calls=[{"name": "run_tests", "args": {}, "id": "c1"}],
        ),
        AIMessage(
            content="Reading calculator.py to inspect buggy multiply implementation",
            tool_calls=[{"name": "read_file", "args": {"file_path": "calculator.py"}, "id": "c2"}],
        ),
        AIMessage(
            content="Fixing multiply implementation from addition to multiplication",
            tool_calls=[{
                "name": "replace_in_file",
                "args": {
                    "file_path": "calculator.py",
                    "old_text": "return a + b",
                    "new_text": "return a * b",
                },
                "id": "c3",
            }],
        ),
        AIMessage(
            content="Running tests to validate fix",
            tool_calls=[{"name": "run_tests", "args": {}, "id": "c4"}],
        ),
        AIMessage(
            content="Verifying goal",
            tool_calls=[{
                "name": "verify_goal",
                "args": {"status": "passed", "summary": "multiply fixed and test passes", "evidence": ["multiply(4,5)==20"]},
                "id": "c5",
            }],
        ),
    ]
    mock_llm = MockLLM(responses=responses)

    state = run_agent(
        goal="Fix the multiply bug in calculator.py so that multiply(4, 5) returns 20.",
        workspace_root=str(tmp_path),
        llm=mock_llm,
    )

    assert state.get("status") == "completed"
    assert state.get("final_outcome") == "SUCCESS"
    assert state.get("validation_result", {}).get("status") == "passed"
    code = (tmp_path / "calculator.py").read_text()
    assert "return a * b" in code


# -----------------------------------------------------------------------------
# Demo 3 — Self-Correction / Recovery
# -----------------------------------------------------------------------------
def test_demo_3_self_correction(tmp_path: Path):
    """Demo 3: First edit fails validation, agent observes traceback, retries and succeeds."""
    init_git_repo(tmp_path)
    (tmp_path / "math_utils.py").write_text("def square(x):\n    return x + x\n")
    (tmp_path / "test_math.py").write_text("from math_utils import square\n\ndef test_square():\n    assert square(4) == 16\n")

    responses = [
        AIMessage(
            content="Running initial test",
            tool_calls=[{"name": "run_tests", "args": {}, "id": "c1"}],
        ),
        # First fix attempt is buggy (returns x * 2 instead of x ** 2)
        AIMessage(
            content="Applying first fix attempt",
            tool_calls=[{
                "name": "replace_in_file",
                "args": {"file_path": "math_utils.py", "old_text": "return x + x", "new_text": "return x * 2"},
                "id": "c2",
            }],
        ),
        AIMessage(
            content="Running tests to validate first fix",
            tool_calls=[{"name": "run_tests", "args": {}, "id": "c3"}],
        ),
        # Second fix attempt (correct: x ** 2)
        AIMessage(
            content="Diagnosed failure (8 != 16). Applying correct exponent fix.",
            tool_calls=[{
                "name": "replace_in_file",
                "args": {"file_path": "math_utils.py", "old_text": "return x * 2", "new_text": "return x ** 2"},
                "id": "c4",
            }],
        ),
        AIMessage(
            content="Re-running tests",
            tool_calls=[{"name": "run_tests", "args": {}, "id": "c5"}],
        ),
        AIMessage(
            content="Verifying goal",
            tool_calls=[{
                "name": "verify_goal",
                "args": {"status": "passed", "summary": "square function corrected after recovery retry", "evidence": ["square(4)==16"]},
                "id": "c6",
            }],
        ),
    ]
    mock_llm = MockLLM(responses=responses)

    state = run_agent(
        goal="Fix square in math_utils.py",
        workspace_root=str(tmp_path),
        llm=mock_llm,
    )

    assert state.get("status") == "completed"
    assert state.get("final_outcome") == "SUCCESS"
    assert state.get("retry_count", 0) > 0
    assert (tmp_path / "math_utils.py").read_text() == "def square(x):\n    return x ** 2\n"


# -----------------------------------------------------------------------------
# Demo 4 — Repository Understanding / Retrieval
# -----------------------------------------------------------------------------
def test_demo_4_repository_understanding_retrieval(tmp_path: Path):
    """Demo 4: Search multi-file repo, retrieve relevant code snippet, edit correct file."""
    init_git_repo(tmp_path)
    (tmp_path / "calculator.py").write_text("def add(a, b): return a + b\n")
    (tmp_path / "string_utils.py").write_text("def reverse_string(s):\n    return s\n")
    (tmp_path / "config.py").write_text("APP_NAME = 'demo'\n")
    (tmp_path / "unrelated.py").write_text("def foo(): pass\n")
    (tmp_path / "test_string_utils.py").write_text("from string_utils import reverse_string\n\ndef test_reverse():\n    assert reverse_string('hello') == 'olleh'\n")

    responses = [
        AIMessage(
            content="Searching repository for string reversal code",
            tool_calls=[{
                "name": "retrieve_relevant_context",
                "args": {"query": "reverse_string string reversal"},
                "id": "c1",
            }],
        ),
        AIMessage(
            content="Modifying string_utils.py to reverse strings correctly",
            tool_calls=[{
                "name": "replace_in_file",
                "args": {
                    "file_path": "string_utils.py",
                    "old_text": "return s",
                    "new_text": "return s[::-1]",
                },
                "id": "c2",
            }],
        ),
        AIMessage(
            content="Running tests",
            tool_calls=[{"name": "run_tests", "args": {}, "id": "c3"}],
        ),
        AIMessage(
            content="Verifying goal",
            tool_calls=[{
                "name": "verify_goal",
                "args": {"status": "passed", "summary": "reverse_string fixed in string_utils.py", "evidence": ["test_reverse passed"]},
                "id": "c4",
            }],
        ),
    ]
    mock_llm = MockLLM(responses=responses)

    state = run_agent(
        goal="Fix the string reversal functionality.",
        workspace_root=str(tmp_path),
        llm=mock_llm,
    )

    assert state.get("status") == "completed"
    assert state.get("final_outcome") == "SUCCESS"
    assert state.get("retrieved_context") is not None
    assert "string_utils.py" in state.get("modified_files", [])
    assert "unrelated.py" not in state.get("modified_files", [])
    assert (tmp_path / "string_utils.py").read_text() == "def reverse_string(s):\n    return s[::-1]\n"


# -----------------------------------------------------------------------------
# Demo 5 — Multi-Agent Workflow
# -----------------------------------------------------------------------------
def test_demo_5_multi_agent_workflow(tmp_path: Path):
    """Demo 5: Analyzer -> Coder -> Reviewer -> Orchestrator multi-agent execution."""
    init_git_repo(tmp_path)
    (tmp_path / "app.py").write_text("def main(): pass\n")

    responses = [
        # Analyzer phase
        AIMessage(
            content="Analyzer: Analyzing goal and repository structure.",
            tool_calls=[{"name": "list_files", "args": {}, "id": "c1"}],
        ),
        AIMessage(content="Analysis complete: app.py needs implementation."),
        # Coder phase
        AIMessage(
            content="Coder: Implementing main logic.",
            tool_calls=[{
                "name": "write_file",
                "args": {"file_path": "app.py", "content": "def main():\n    return 'OK'\n"},
                "id": "c2",
            }],
        ),
        AIMessage(content="Implementation complete."),
        # Reviewer phase
        AIMessage(content="Reviewer: Code is clean and approved. STATUS: APPROVED"),
    ]
    mock_llm = MultiAgentMockLLM(responses=responses)

    state = run_multi_agent(
        goal="Implement main in app.py",
        workspace_root=str(tmp_path),
        llm=mock_llm,
    )

    assert state.get("mode") == "multi_agent"
    assert state.get("review_status") == "approved"
    assert state.get("final_outcome") == "SUCCESS"
    assert (tmp_path / "app.py").read_text() == "def main():\n    return 'OK'\n"


# -----------------------------------------------------------------------------
# Demo 6 — Security / Sandbox Rejection
# -----------------------------------------------------------------------------
def test_demo_6_security_sandbox_rejection(tmp_path: Path):
    """Demo 6: Agent attempts path traversal or unsafe file access; security boundary rejects it."""
    init_git_repo(tmp_path)

    responses = [
        AIMessage(
            content="Attempting to read path outside workspace boundary",
            tool_calls=[{
                "name": "read_file",
                "args": {"file_path": "../../etc/passwd"},
                "id": "c1",
            }],
        ),
        AIMessage(
            content="Observed access denied security rejection.",
        ),
    ]
    mock_llm = MockLLM(responses=responses)

    state = run_agent(
        goal="Read system passwd file at ../../etc/passwd",
        workspace_root=str(tmp_path),
        llm=mock_llm,
    )

    messages = state.get("messages", [])
    obs = [m.content for m in messages if isinstance(m, ToolMessage) or m.__class__.__name__ == "ToolMessage"]
    assert any("Access denied" in o or "escapes workspace" in o for o in obs)


# -----------------------------------------------------------------------------
# Demo 7 — Human Approval Boundary
# -----------------------------------------------------------------------------
def test_demo_7_human_approval_boundary(tmp_path: Path):
    """Demo 7: Protected Git delivery action triggers human approval gate."""
    init_git_repo(tmp_path)

    responses = [
        AIMessage(
            content="Requesting human approval for git push delivery action",
            tool_calls=[{
                "name": "request_human_approval",
                "args": {
                    "action": "push",
                    "reason": "Pushing commit to remote repository main branch",
                    "risk_assessment": "high",
                },
                "id": "c1",
            }],
        ),
    ]
    mock_llm = MockLLM(responses=responses)

    state = run_agent(
        goal="Push changes to remote repository",
        workspace_root=str(tmp_path),
        llm=mock_llm,
    )

    assert state.get("approval_required") is True
    assert state.get("approval_status") == "pending"
    assert state.get("final_outcome") == "ESCALATED"
    assert "requires human approval" in state.get("approval_reason", "").lower() or "push" in state.get("approval_reason", "").lower()


# -----------------------------------------------------------------------------
# Demo 8 — Failure / Escalation
# -----------------------------------------------------------------------------
def test_demo_8_failure_escalation(tmp_path: Path):
    """Demo 8: Persistent failing validation exhausts max retries and reports FAILED."""
    init_git_repo(tmp_path)
    (tmp_path / "calc.py").write_text("def add(a, b): return 0\n")
    (tmp_path / "test_calc.py").write_text("from calc import add\n\ndef test_add(): assert add(2, 2) == 4\n")

    responses = [
        AIMessage(content="Run test 1", tool_calls=[{"name": "run_tests", "args": {}, "id": "c1"}]),
        AIMessage(content="Fix 1", tool_calls=[{"name": "replace_in_file", "args": {"file_path": "calc.py", "old_text": "return 0", "new_text": "return 1"}, "id": "c2"}]),
        AIMessage(content="Run test 2", tool_calls=[{"name": "run_tests", "args": {}, "id": "c3"}]),
        AIMessage(content="Fix 2", tool_calls=[{"name": "replace_in_file", "args": {"file_path": "calc.py", "old_text": "return 1", "new_text": "return 2"}, "id": "c4"}]),
        AIMessage(content="Run test 3", tool_calls=[{"name": "run_tests", "args": {}, "id": "c5"}]),
        AIMessage(content="Fix 3", tool_calls=[{"name": "replace_in_file", "args": {"file_path": "calc.py", "old_text": "return 2", "new_text": "return 3"}, "id": "c6"}]),
        AIMessage(content="Run test 4", tool_calls=[{"name": "run_tests", "args": {}, "id": "c7"}]),
    ]
    mock_llm = MockLLM(responses=responses)

    state = run_agent(
        goal="Fix add in calc.py",
        workspace_root=str(tmp_path),
        llm=mock_llm,
    )

    assert state.get("status") == "failed"
    assert state.get("final_outcome") == "FAILED"
    assert state.get("retry_count", 0) >= 3


# -----------------------------------------------------------------------------
# Demo 9 — Final Execution Report
# -----------------------------------------------------------------------------
def test_demo_9_final_execution_report(tmp_path: Path, monkeypatch, capsys):
    """Demo 9: CLI execution prints the complete Phase 16C final execution report block."""
    monkeypatch.setenv("AGENT_LOG_LEVEL", "normal")
    init_git_repo(tmp_path)
    (tmp_path / "calc.py").write_text("def add(a, b): return a + b\n")

    responses = [
        AIMessage(
            content="",
            tool_calls=[{
                "name": "verify_goal",
                "args": {"status": "passed", "summary": "Verified calc.py", "evidence": ["passed"]},
                "id": "c1",
            }],
        ),
    ]
    mock_llm = MockLLM(responses=responses)

    with patch("sys.argv", ["agent.py", "Verify calc.py functionality", str(tmp_path)]):
        with patch("app.agent.get_default_llm", return_value=mock_llm):
            main()

    captured = capsys.readouterr().out
    assert "FINAL EXECUTION REPORT" in captured
    assert "Status: SUCCESS" in captured
    assert "Goal:" in captured
    assert "Verify calc.py functionality" in captured
    assert "Workspace:" in captured
    assert "Tests:" in captured
    assert "Files modified:" in captured
    assert "Tool calls:" in captured
    assert "Validation attempts:" in captured
    assert "Recovery retries:" in captured
    assert "Goal verification: PASSED" in captured
    assert "Human interventions:" in captured
    assert "Execution time:" in captured
    assert "==================================================" in captured
    assert "SUCCESS" in captured
