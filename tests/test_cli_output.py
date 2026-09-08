"""Tests for Phase 16A clean CLI output and AGENT_LOG_LEVEL behavior."""

import os
import sys
from pathlib import Path
from unittest.mock import patch, MagicMock
import pytest
from langchain_core.messages import AIMessage, HumanMessage

from app.agent import main, run_agent, sanitize_log_output
from tests.test_agent import MockLLM, init_git_repo


def test_normal_mode_hides_verbose_internal_dumps(tmp_path: Path, monkeypatch, capsys):
    """Test A: Normal mode hides verbose internal message dumps and thought signatures."""
    monkeypatch.setenv("AGENT_LOG_LEVEL", "normal")
    init_git_repo(tmp_path)
    (tmp_path / "calc.py").write_text("def add(a, b): return a + b\n")

    responses = [
        AIMessage(
            content="",
            tool_calls=[{
                "name": "verify_goal",
                "args": {
                    "status": "passed",
                    "summary": "calc.py verified",
                    "evidence": ["add returns a+b"],
                },
                "id": "c1",
            }],
            additional_kwargs={"__gemini_function_call_thought_signatures__": {"c1": "secret_thought_sig_123"}}
        ),
        AIMessage(content="Task complete."),
    ]
    mock_llm = MockLLM(responses=responses)

    with patch("sys.argv", ["agent.py", "Verify calc.py", str(tmp_path)]):
        with patch("app.agent.get_default_llm", return_value=mock_llm):
            main()

    captured = capsys.readouterr().out
    assert "AUTONOMOUS CODING AGENT" in captured
    assert "Goal:" in captured
    assert "Verify calc.py" in captured
    assert "Workspace:" in captured
    assert "SUCCESS" in captured
    assert "secret_thought_sig_123" not in captured
    assert "=== DEBUG:" not in captured
    assert "=== Message Metadata Trace ===" not in captured


def test_normal_mode_does_not_expose_api_keys(monkeypatch):
    """Test B: Normal mode sanitizes API keys and sensitive tokens."""
    sensitive = "AIzaSyD-secret-key-9999999999999999"
    sanitized = sanitize_log_output(sensitive)
    assert sensitive not in sanitized
    assert "[REDACTED_API_KEY]" in sanitized


def test_debug_mode_remains_available(tmp_path: Path, monkeypatch, capsys):
    """Test C: Debug mode displays verbose message metadata trace."""
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


def test_successful_execution_produces_clean_success_result(tmp_path: Path, monkeypatch, capsys):
    """Test D: Successful execution produces clean SUCCESS section."""
    monkeypatch.setenv("AGENT_LOG_LEVEL", "normal")
    init_git_repo(tmp_path)

    responses = [
        AIMessage(
            content="",
            tool_calls=[{
                "name": "verify_goal",
                "args": {
                    "status": "passed",
                    "summary": "All tests passed cleanly",
                    "evidence": ["test output passed"],
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
    assert "Files modified: 0" in captured
    assert "Recovery retries: 0" in captured
    assert "Goal verification: PASSED" in captured


def test_failed_execution_produces_clean_failed_result(tmp_path: Path, monkeypatch, capsys):
    """Test E: Failed execution produces clean FAILED section."""
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


def test_existing_agent_functionality_remains_unchanged(tmp_path: Path):
    """Test F: Core run_agent functionality returns complete state dict with full trace and metrics."""
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
