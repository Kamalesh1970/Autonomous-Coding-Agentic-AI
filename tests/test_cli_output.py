"""Tests for Phase 16B human-readable progress and CLI output behavior."""

import os
import sys
from pathlib import Path
from unittest.mock import patch, MagicMock
import pytest
from langchain_core.messages import AIMessage, HumanMessage, ToolMessage

from app.agent import main, run_agent, sanitize_log_output
from tests.test_agent import MockLLM, init_git_repo


def test_normal_mode_displays_human_readable_progress(tmp_path: Path, monkeypatch, capsys):
    """Requirement A: Normal mode displays human-readable progress steps."""
    monkeypatch.setenv("AGENT_LOG_LEVEL", "normal")
    init_git_repo(tmp_path)
    (tmp_path / "calc.py").write_text("def add(a, b): return a + b\n")

    responses = [
        AIMessage(
            content="",
            tool_calls=[{
                "name": "list_files",
                "args": {},
                "id": "c1",
            }],
        ),
        AIMessage(
            content="",
            tool_calls=[{
                "name": "retrieve_relevant_context",
                "args": {"query": "calc"},
                "id": "c2",
            }],
        ),
        AIMessage(
            content="",
            tool_calls=[{
                "name": "run_tests",
                "args": {},
                "id": "c3",
            }],
        ),
        AIMessage(
            content="",
            tool_calls=[{
                "name": "create_plan",
                "args": {"goal": "Fix calc", "tasks": [{"id": "task-1", "title": "Fix bug", "description": "Fix bug"}]},
                "id": "c4",
            }],
        ),
        AIMessage(
            content="",
            tool_calls=[{
                "name": "write_file",
                "args": {"path": "calc.py", "content": "def add(a, b): return a + b\n"},
                "id": "c5",
            }],
        ),
        AIMessage(
            content="",
            tool_calls=[{
                "name": "run_tests",
                "args": {},
                "id": "c6",
            }],
        ),
        AIMessage(
            content="",
            tool_calls=[{
                "name": "verify_goal",
                "args": {
                    "status": "passed",
                    "summary": "calc.py verified",
                    "evidence": ["add returns a+b"],
                },
                "id": "c7",
            }],
        ),
        AIMessage(content="Task complete."),
    ]
    mock_llm = MockLLM(responses=responses)

    with patch("sys.argv", ["agent.py", "Verify calc.py", str(tmp_path)]):
        with patch("app.agent.get_default_llm", return_value=mock_llm):
            main()

    captured = capsys.readouterr().out
    assert "Understanding repository" in captured
    assert "Retrieving relevant code" in captured
    assert "Running initial tests" in captured
    assert "Planning changes" in captured
    assert "Modifying files" in captured
    assert "Running validation" in captured
    assert "Verifying goal" in captured

    pos_repo = captured.index("Understanding repository")
    pos_retrieval = captured.index("Retrieving relevant code")
    pos_plan = captured.index("Planning changes")
    pos_modify = captured.index("Modifying files")
    pos_val = captured.index("Running validation")
    pos_verify = captured.index("Verifying goal")

    assert pos_repo < pos_retrieval < pos_plan < pos_modify < pos_val < pos_verify


def test_progress_output_does_not_expose_api_keys(monkeypatch):
    """Requirement B: Progress output does not expose API keys."""
    sensitive_gemini = "AIzaSyD-secret-key-9999999999999999"
    sensitive_openai = "sk-proj-123456789012345678901234567890"
    sensitive_openrouter = "sk-or-v1-abcdef1234567890abcdef1234567890"

    sanitized_g = sanitize_log_output(sensitive_gemini)
    sanitized_o = sanitize_log_output(sensitive_openai)
    sanitized_r = sanitize_log_output(sensitive_openrouter)

    assert sensitive_gemini not in sanitized_g
    assert sensitive_openai not in sanitized_o
    assert sensitive_openrouter not in sanitized_r
    assert "[REDACTED_API_KEY]" in sanitized_g
    assert "[REDACTED_API_KEY]" in sanitized_o
    assert "[REDACTED_API_KEY]" in sanitized_r


def test_progress_output_does_not_expose_thought_signatures(tmp_path: Path, monkeypatch, capsys):
    """Requirement C: Progress output does not expose thought signatures."""
    monkeypatch.setenv("AGENT_LOG_LEVEL", "normal")
    init_git_repo(tmp_path)

    responses = [
        AIMessage(
            content="",
            tool_calls=[{
                "name": "verify_goal",
                "args": {
                    "status": "passed",
                    "summary": "verified",
                    "evidence": [],
                },
                "id": "c1",
            }],
            additional_kwargs={"__gemini_function_call_thought_signatures__": {"c1": "secret_thought_sig_999"}}
        ),
    ]
    mock_llm = MockLLM(responses=responses)

    with patch("sys.argv", ["agent.py", "Test goal", str(tmp_path)]):
        with patch("app.agent.get_default_llm", return_value=mock_llm):
            main()

    captured = capsys.readouterr().out
    assert "secret_thought_sig_999" not in captured
    assert "__gemini_function_call_thought_signatures__" not in captured


def test_progress_output_does_not_dump_raw_aimessage(tmp_path: Path, monkeypatch, capsys):
    """Requirement D: Progress output does not dump raw AIMessage objects."""
    monkeypatch.setenv("AGENT_LOG_LEVEL", "normal")
    init_git_repo(tmp_path)

    responses = [
        AIMessage(
            content="Internal reasoning dump content",
            tool_calls=[{
                "name": "verify_goal",
                "args": {"status": "passed", "summary": "done", "evidence": []},
                "id": "c1",
            }],
            additional_kwargs={"some_internal_kwarg": "secret_data"},
            response_metadata={"raw_headers": {"x-api-key": "secret"}}
        ),
    ]
    mock_llm = MockLLM(responses=responses)

    with patch("sys.argv", ["agent.py", "Test goal", str(tmp_path)]):
        with patch("app.agent.get_default_llm", return_value=mock_llm):
            main()

    captured = capsys.readouterr().out
    assert "AIMessage(" not in captured
    assert "additional_kwargs=" not in captured
    assert "response_metadata=" not in captured
    assert "some_internal_kwarg" not in captured


def test_debug_mode_exposes_useful_diagnostic_information(tmp_path: Path, monkeypatch, capsys):
    """Requirement E: Debug mode still exposes useful diagnostic information."""
    monkeypatch.setenv("AGENT_LOG_LEVEL", "debug")
    init_git_repo(tmp_path)

    responses = [
        AIMessage(
            content="Debug verification",
            tool_calls=[{
                "name": "verify_goal",
                "args": {"status": "passed", "summary": "all good", "evidence": []},
                "id": "c1",
            }],
        ),
    ]
    mock_llm = MockLLM(responses=responses)

    with patch("sys.argv", ["agent.py", "Debug test", str(tmp_path)]):
        with patch("app.agent.get_default_llm", return_value=mock_llm):
            main()

    captured = capsys.readouterr().out
    assert "=== DEBUG: Message Metadata Trace ===" in captured
    assert "=== Agent Trace ===" in captured
    assert "[Agent]: Debug verification" in captured
    assert "→ Tool Call: verify_goal" in captured


def test_successful_execution_displays_success(tmp_path: Path, monkeypatch, capsys):
    """Requirement F: Successful execution displays SUCCESS block with metrics."""
    monkeypatch.setenv("AGENT_LOG_LEVEL", "normal")
    init_git_repo(tmp_path)

    responses = [
        AIMessage(
            content="",
            tool_calls=[{
                "name": "verify_goal",
                "args": {
                    "status": "passed",
                    "summary": "4 passed",
                    "evidence": ["4 tests passed"],
                },
                "id": "c1",
            }],
        ),
    ]
    mock_llm = MockLLM(responses=responses)

    with patch("sys.argv", ["agent.py", "Fix calc", str(tmp_path)]):
        with patch("app.agent.get_default_llm", return_value=mock_llm):
            main()

    captured = capsys.readouterr().out
    assert "==================================================" in captured
    assert "SUCCESS" in captured
    assert "Tests:" in captured
    assert "Files modified:" in captured
    assert "Recovery retries:" in captured
    assert "Goal verification: PASSED" in captured


def test_failed_execution_displays_failed(tmp_path: Path, monkeypatch, capsys):
    """Requirement G: Failed execution displays FAILED block."""
    monkeypatch.setenv("AGENT_LOG_LEVEL", "normal")
    init_git_repo(tmp_path)

    with patch("sys.argv", ["agent.py", "Fail test", str(tmp_path)]):
        with patch("app.agent.run_agent", side_effect=RuntimeError("Provider limit exhausted")):
            with pytest.raises(SystemExit):
                main()

    captured = capsys.readouterr().out
    assert "FAILED" in captured
    assert "Reason:" in captured
    assert "Provider limit exhausted" in captured


def test_escalated_execution_displays_escalated(tmp_path: Path, monkeypatch, capsys):
    """Requirement H: Escalated execution displays ESCALATED block."""
    monkeypatch.setenv("AGENT_LOG_LEVEL", "normal")
    init_git_repo(tmp_path)

    fake_state = {
        "user_goal": "Deploy feature",
        "workspace_root": str(tmp_path),
        "status": "completed",
        "final_outcome": "ESCALATED",
        "approval_required": True,
        "approval_status": "pending",
        "approval_reason": "Git push requires human approval",
        "messages": [],
    }

    with patch("sys.argv", ["agent.py", "Deploy feature", str(tmp_path)]):
        with patch("app.agent.run_agent", return_value=fake_state):
            main()

    captured = capsys.readouterr().out
    assert "==================================================" in captured
    assert "ESCALATED" in captured
    assert "Reason:" in captured
    assert "Git push requires human approval" in captured


def test_existing_cli_behavior_remains_compatible(tmp_path: Path, monkeypatch, capsys):
    """Requirement I: Existing CLI behavior remains compatible."""
    monkeypatch.setenv("AGENT_LOG_LEVEL", "normal")
    init_git_repo(tmp_path)

    responses = [
        AIMessage(
            content="Completed task",
            tool_calls=[{
                "name": "verify_goal",
                "args": {"status": "passed", "summary": "verified", "evidence": []},
                "id": "c1",
            }],
        ),
    ]
    mock_llm = MockLLM(responses=responses)

    with patch("sys.argv", ["agent.py", "Fix all failing tests", str(tmp_path)]):
        with patch("app.agent.get_default_llm", return_value=mock_llm):
            main()

    captured = capsys.readouterr().out
    assert "AUTONOMOUS CODING AGENT" in captured
    assert "Goal:" in captured
    assert "Fix all failing tests" in captured
    assert "Workspace:" in captured


def test_existing_agent_functionality_remains_unchanged(tmp_path: Path):
    """Requirement J: Existing agent functionality remains unchanged."""
    init_git_repo(tmp_path)

    responses = [
        AIMessage(
            content="Verifying goal",
            tool_calls=[{
                "name": "verify_goal",
                "args": {"status": "passed", "summary": "verified", "evidence": []},
                "id": "c1",
            }],
        ),
    ]
    mock_llm = MockLLM(responses=responses)

    state = run_agent(goal="Test goal", workspace_root=str(tmp_path), llm=mock_llm)
    assert "messages" in state
    assert "verification_result" in state
    assert state.get("verification_result", {}).get("status") == "passed"
