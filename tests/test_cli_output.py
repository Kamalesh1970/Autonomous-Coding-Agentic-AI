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


# -----------------------------------------------------------------------------
# Phase 16C — Final Execution Report Tests
# -----------------------------------------------------------------------------

def test_phase16c_successful_execution_produces_final_execution_report(tmp_path: Path, monkeypatch, capsys):
    """Requirement A: Successful execution produces a final execution report."""
    monkeypatch.setenv("AGENT_LOG_LEVEL", "normal")
    init_git_repo(tmp_path)

    responses = [
        AIMessage(
            content="",
            tool_calls=[{
                "name": "verify_goal",
                "args": {"status": "passed", "summary": "Goal completed", "evidence": []},
                "id": "c1",
            }],
        ),
    ]
    mock_llm = MockLLM(responses=responses)

    with patch("sys.argv", ["agent.py", "Fix bug in calc", str(tmp_path)]):
        with patch("app.agent.get_default_llm", return_value=mock_llm):
            main()

    captured = capsys.readouterr().out
    assert "FINAL EXECUTION REPORT" in captured
    assert "Status: SUCCESS" in captured
    assert "==================================================" in captured
    assert "SUCCESS" in captured


def test_phase16c_failed_execution_produces_failed_report(tmp_path: Path, monkeypatch, capsys):
    """Requirement B: Failed execution produces a FAILED report."""
    monkeypatch.setenv("AGENT_LOG_LEVEL", "normal")
    init_git_repo(tmp_path)

    with patch("sys.argv", ["agent.py", "Fail test", str(tmp_path)]):
        with patch("app.agent.run_agent", side_effect=RuntimeError("Provider limit exhausted")):
            with pytest.raises(SystemExit):
                main()

    captured = capsys.readouterr().out
    assert "FINAL EXECUTION REPORT" in captured
    assert "Status: FAILED" in captured
    assert "Reason:" in captured
    assert "Provider limit exhausted" in captured


def test_phase16c_escalated_execution_produces_escalated_report(tmp_path: Path, monkeypatch, capsys):
    """Requirement C: Escalated execution produces an ESCALATED report."""
    monkeypatch.setenv("AGENT_LOG_LEVEL", "normal")
    init_git_repo(tmp_path)

    fake_state = {
        "user_goal": "Deploy production release",
        "workspace_root": str(tmp_path),
        "status": "completed",
        "final_outcome": "ESCALATED",
        "approval_required": True,
        "approval_status": "pending",
        "approval_reason": "Git push requires human approval",
        "messages": [],
    }

    with patch("sys.argv", ["agent.py", "Deploy production release", str(tmp_path)]):
        with patch("app.agent.run_agent", return_value=fake_state):
            main()

    captured = capsys.readouterr().out
    assert "FINAL EXECUTION REPORT" in captured
    assert "Status: ESCALATED" in captured
    assert "Reason:" in captured
    assert "Git push requires human approval" in captured


def test_phase16c_report_contains_actual_user_goal(tmp_path: Path, monkeypatch, capsys):
    """Requirement D: Report contains the actual user goal."""
    monkeypatch.setenv("AGENT_LOG_LEVEL", "normal")
    init_git_repo(tmp_path)

    responses = [
        AIMessage(
            content="",
            tool_calls=[{
                "name": "verify_goal",
                "args": {"status": "passed", "summary": "verified", "evidence": []},
                "id": "c1",
            }],
        ),
    ]
    mock_llm = MockLLM(responses=responses)
    custom_goal = "Refactor calculation engine in math module"

    with patch("sys.argv", ["agent.py", custom_goal, str(tmp_path)]):
        with patch("app.agent.get_default_llm", return_value=mock_llm):
            main()

    captured = capsys.readouterr().out
    assert "FINAL EXECUTION REPORT" in captured
    assert custom_goal in captured


def test_phase16c_report_contains_actual_test_validation_information(tmp_path: Path, monkeypatch, capsys):
    """Requirement E: Report contains actual test/validation information."""
    monkeypatch.setenv("AGENT_LOG_LEVEL", "normal")
    init_git_repo(tmp_path)

    fake_state = {
        "user_goal": "Fix tests",
        "workspace_root": str(tmp_path),
        "status": "completed",
        "final_outcome": "SUCCESS",
        "validation_result": {"status": "passed", "summary": "12 passed, 0 failed"},
        "verification_result": {"status": "passed", "summary": "verified", "evidence": []},
        "messages": [],
    }

    with patch("sys.argv", ["agent.py", "Fix tests", str(tmp_path)]):
        with patch("app.agent.run_agent", return_value=fake_state):
            main()

    captured = capsys.readouterr().out
    assert "12 passed, 0 failed" in captured


def test_phase16c_report_contains_actual_recovery_retry_count(tmp_path: Path, monkeypatch, capsys):
    """Requirement F: Report contains actual recovery retry count."""
    monkeypatch.setenv("AGENT_LOG_LEVEL", "normal")
    init_git_repo(tmp_path)

    fake_state = {
        "user_goal": "Recover failing code",
        "workspace_root": str(tmp_path),
        "status": "completed",
        "final_outcome": "SUCCESS",
        "retry_count": 3,
        "validation_result": {"status": "passed", "summary": "passed after retries"},
        "verification_result": {"status": "passed", "summary": "verified", "evidence": []},
        "messages": [],
    }

    with patch("sys.argv", ["agent.py", "Recover failing code", str(tmp_path)]):
        with patch("app.agent.run_agent", return_value=fake_state):
            main()

    captured = capsys.readouterr().out
    assert "Recovery retries:\n3" in captured or "3" in captured


def test_phase16c_report_contains_actual_modified_file_count(tmp_path: Path, monkeypatch, capsys):
    """Requirement G: Report contains actual modified-file count where available."""
    monkeypatch.setenv("AGENT_LOG_LEVEL", "normal")
    init_git_repo(tmp_path)

    fake_state = {
        "user_goal": "Edit files",
        "workspace_root": str(tmp_path),
        "status": "completed",
        "final_outcome": "SUCCESS",
        "modified_files": ["calc.py", "utils.py", "test_calc.py"],
        "verification_result": {"status": "passed", "summary": "verified", "evidence": []},
        "messages": [],
    }

    with patch("sys.argv", ["agent.py", "Edit files", str(tmp_path)]):
        with patch("app.agent.run_agent", return_value=fake_state):
            main()

    captured = capsys.readouterr().out
    assert "Files modified:\n3" in captured


def test_phase16c_report_contains_actual_tool_call_count(tmp_path: Path, monkeypatch, capsys):
    """Requirement H: Report contains actual tool-call count where available."""
    monkeypatch.setenv("AGENT_LOG_LEVEL", "normal")
    init_git_repo(tmp_path)

    fake_state = {
        "user_goal": "Multi-tool run",
        "workspace_root": str(tmp_path),
        "status": "completed",
        "final_outcome": "SUCCESS",
        "evaluation_report": {"tool_call_count": 7},
        "verification_result": {"status": "passed", "summary": "verified", "evidence": []},
        "messages": [],
    }

    with patch("sys.argv", ["agent.py", "Multi-tool run", str(tmp_path)]):
        with patch("app.agent.run_agent", return_value=fake_state):
            main()

    captured = capsys.readouterr().out
    assert "Tool calls:\n7" in captured


def test_phase16c_report_contains_actual_goal_verification_status(tmp_path: Path, monkeypatch, capsys):
    """Requirement I: Report contains actual goal-verification status."""
    monkeypatch.setenv("AGENT_LOG_LEVEL", "normal")
    init_git_repo(tmp_path)

    fake_state = {
        "user_goal": "Verify goal status",
        "workspace_root": str(tmp_path),
        "status": "completed",
        "final_outcome": "SUCCESS",
        "verification_result": {"status": "passed", "summary": "Verified completely", "evidence": ["evidence1"]},
        "messages": [],
    }

    with patch("sys.argv", ["agent.py", "Verify goal status", str(tmp_path)]):
        with patch("app.agent.run_agent", return_value=fake_state):
            main()

    captured = capsys.readouterr().out
    assert "Goal verification: PASSED" in captured


def test_phase16c_report_does_not_fabricate_unavailable_metrics(capsys):
    """Requirement J: Report does not fabricate unavailable metrics."""
    from app.agent import _print_cli_summary

    empty_state = {
        "user_goal": None,
        "workspace_root": None,
        "status": "unknown",
    }
    _print_cli_summary(empty_state)

    captured = capsys.readouterr().out
    assert "N/A" in captured
    assert "FINAL EXECUTION REPORT" in captured


def test_phase16c_report_does_not_expose_api_keys(capsys):
    """Requirement K: Report does not expose API keys."""
    from app.agent import _print_cli_summary

    sensitive_state = {
        "user_goal": "Task with AIzaSyD-secret-key-123456789 and sk-proj-12345678901234567890",
        "workspace_root": ".",
        "status": "completed",
        "final_outcome": "SUCCESS",
    }
    _print_cli_summary(sensitive_state)

    captured = capsys.readouterr().out
    assert "AIzaSyD-secret-key-123456789" not in captured
    assert "sk-proj-12345678901234567890" not in captured
    assert "[REDACTED_API_KEY]" in captured


def test_phase16c_report_does_not_expose_thought_signatures(tmp_path: Path, monkeypatch, capsys):
    """Requirement L: Report does not expose thought signatures."""
    monkeypatch.setenv("AGENT_LOG_LEVEL", "normal")
    init_git_repo(tmp_path)

    responses = [
        AIMessage(
            content="",
            tool_calls=[{
                "name": "verify_goal",
                "args": {"status": "passed", "summary": "done", "evidence": []},
                "id": "c1",
            }],
            additional_kwargs={"__gemini_function_call_thought_signatures__": {"c1": "thought_sig_secret_123"}}
        ),
    ]
    mock_llm = MockLLM(responses=responses)

    with patch("sys.argv", ["agent.py", "Thought sig test", str(tmp_path)]):
        with patch("app.agent.get_default_llm", return_value=mock_llm):
            main()

    captured = capsys.readouterr().out
    assert "thought_sig_secret_123" not in captured
    assert "__gemini_function_call_thought_signatures__" not in captured


def test_phase16c_report_does_not_expose_raw_aimessage(tmp_path: Path, monkeypatch, capsys):
    """Requirement M: Report does not expose raw AIMessage objects."""
    monkeypatch.setenv("AGENT_LOG_LEVEL", "normal")
    init_git_repo(tmp_path)

    responses = [
        AIMessage(
            content="Internal thoughts",
            tool_calls=[{
                "name": "verify_goal",
                "args": {"status": "passed", "summary": "done", "evidence": []},
                "id": "c1",
            }],
            additional_kwargs={"raw_internal": "secret"},
        ),
    ]
    mock_llm = MockLLM(responses=responses)

    with patch("sys.argv", ["agent.py", "AIMessage test", str(tmp_path)]):
        with patch("app.agent.get_default_llm", return_value=mock_llm):
            main()

    captured = capsys.readouterr().out
    assert "AIMessage(" not in captured
    assert "raw_internal" not in captured


# -----------------------------------------------------------------------------
# Phase 16E — Production Configuration & Error Handling Tests
# -----------------------------------------------------------------------------

def test_phase16e_nonexistent_workspace_produces_failed_report(tmp_path: Path, monkeypatch, capsys):
    """Requirement A: Non-existent workspace produces a FAILED report without traceback."""
    monkeypatch.setenv("AGENT_LOG_LEVEL", "normal")
    nonexistent = str(tmp_path / "does_not_exist" / "workspace")

    with patch("sys.argv", ["agent.py", "Fix something", nonexistent]):
        with pytest.raises(SystemExit):
            main()

    captured = capsys.readouterr().out
    assert "FINAL EXECUTION REPORT" in captured
    assert "Status: FAILED" in captured
    assert "FAILED" in captured
    # Must not expose raw Python traceback
    assert "Traceback" not in captured
    assert "does not exist" in captured or "Invalid workspace" in captured


def test_phase16e_file_instead_of_directory_produces_failed_report(tmp_path: Path, monkeypatch, capsys):
    """Requirement B: Providing a file path as workspace produces a FAILED report."""
    monkeypatch.setenv("AGENT_LOG_LEVEL", "normal")
    file_path = tmp_path / "notadir.py"
    file_path.write_text("x = 1\n")

    with patch("sys.argv", ["agent.py", "Fix something", str(file_path)]):
        with pytest.raises(SystemExit):
            main()

    captured = capsys.readouterr().out
    assert "FINAL EXECUTION REPORT" in captured
    assert "Status: FAILED" in captured
    assert "Traceback" not in captured
    assert "not a directory" in captured or "Invalid workspace" in captured


def test_phase16e_workspace_error_report_contains_user_goal(tmp_path: Path, monkeypatch, capsys):
    """Requirement C: Workspace error report still contains the user's original goal."""
    monkeypatch.setenv("AGENT_LOG_LEVEL", "normal")
    nonexistent = str(tmp_path / "missing_workspace")
    custom_goal = "Deploy the production feature branch"

    with patch("sys.argv", ["agent.py", custom_goal, nonexistent]):
        with pytest.raises(SystemExit):
            main()

    captured = capsys.readouterr().out
    assert "FINAL EXECUTION REPORT" in captured
    assert custom_goal in captured


def test_phase16e_workspace_error_does_not_expose_api_keys(tmp_path: Path, monkeypatch, capsys):
    """Requirement D: Workspace validation error output does not expose API keys."""
    monkeypatch.setenv("AGENT_LOG_LEVEL", "normal")
    monkeypatch.setenv("GEMINI_API_KEY_1", "AIzaSyD-secret-key-789012345678901234567")
    nonexistent = str(tmp_path / "no_such_workspace")

    with patch("sys.argv", ["agent.py", "Test goal", nonexistent]):
        with pytest.raises(SystemExit):
            main()

    captured = capsys.readouterr().out
    assert "AIzaSyD-secret-key-789012345678901234567" not in captured
    assert "FINAL EXECUTION REPORT" in captured


def test_phase16e_provider_configuration_error_displays_clean_message(tmp_path: Path, monkeypatch, capsys):
    """Requirement E: Missing provider API key displays a clean FAILED report with readable error message."""
    monkeypatch.setenv("AGENT_LOG_LEVEL", "normal")
    init_git_repo(tmp_path)

    with patch("sys.argv", ["agent.py", "Fix tests", str(tmp_path)]):
        with patch("app.agent.run_agent", side_effect=ValueError("OPENAI_API_KEY is required when LLM_PROVIDER=openai")):
            with pytest.raises(SystemExit):
                main()

    captured = capsys.readouterr().out
    assert "FINAL EXECUTION REPORT" in captured
    assert "Status: FAILED" in captured
    assert "Reason:" in captured
    assert "Traceback" not in captured


def test_phase16e_unsupported_provider_error_displays_clean_message(tmp_path: Path, monkeypatch, capsys):
    """Requirement F: Unsupported LLM_PROVIDER value displays a clean FAILED report."""
    monkeypatch.setenv("AGENT_LOG_LEVEL", "normal")
    init_git_repo(tmp_path)

    with patch("sys.argv", ["agent.py", "Fix tests", str(tmp_path)]):
        with patch("app.agent.run_agent", side_effect=ValueError("Unsupported LLM provider: badprovider")):
            with pytest.raises(SystemExit):
                main()

    captured = capsys.readouterr().out
    assert "FINAL EXECUTION REPORT" in captured
    assert "Status: FAILED" in captured
    assert "Traceback" not in captured


def test_phase16e_tool_execution_exception_displays_clean_report(tmp_path: Path, monkeypatch, capsys):
    """Requirement G: Unexpected runtime exception during agent execution shows FAILED report, not raw traceback."""
    monkeypatch.setenv("AGENT_LOG_LEVEL", "normal")
    init_git_repo(tmp_path)

    with patch("sys.argv", ["agent.py", "Crash test", str(tmp_path)]):
        with patch("app.agent.run_agent", side_effect=RuntimeError("Unexpected internal agent crash")):
            with pytest.raises(SystemExit):
                main()

    captured = capsys.readouterr().out
    assert "FINAL EXECUTION REPORT" in captured
    assert "Status: FAILED" in captured
    assert "Reason:" in captured
    assert "Traceback" not in captured
    # Raw exception class name should NOT appear in output
    assert "RuntimeError" not in captured


def test_phase16e_provider_error_does_not_expose_api_keys_in_error_message(tmp_path: Path, monkeypatch, capsys):
    """Requirement H: Provider errors do not leak API key values in the CLI output."""
    monkeypatch.setenv("AGENT_LOG_LEVEL", "normal")
    init_git_repo(tmp_path)
    secret = "sk-proj-leakme12345678901234567890123456"

    with patch("sys.argv", ["agent.py", "Test goal", str(tmp_path)]):
        with patch("app.agent.run_agent", side_effect=ValueError(f"Invalid key: {secret}")):
            with pytest.raises(SystemExit):
                main()

    captured = capsys.readouterr().out
    assert secret not in captured
    assert "[REDACTED_API_KEY]" in captured


def test_phase16e_validate_workspace_valid_directory(tmp_path: Path):
    """Requirement I: _validate_workspace raises no exception for a valid, readable directory."""
    from app.agent import _validate_workspace
    # Should not raise any exception
    _validate_workspace(str(tmp_path))


def test_phase16e_validate_workspace_nonexistent_raises_value_error(tmp_path: Path):
    """Requirement J: _validate_workspace raises ValueError for non-existent paths."""
    from app.agent import _validate_workspace
    with pytest.raises(ValueError, match="does not exist"):
        _validate_workspace(str(tmp_path / "no_such_directory"))


def test_phase16e_validate_workspace_file_raises_value_error(tmp_path: Path):
    """Requirement K: _validate_workspace raises ValueError when path is a file, not directory."""
    from app.agent import _validate_workspace
    file_path = tmp_path / "file.py"
    file_path.write_text("x = 1\n")
    with pytest.raises(ValueError, match="not a directory"):
        _validate_workspace(str(file_path))


def test_phase16e_failed_execution_exit_code_nonzero(tmp_path: Path, monkeypatch, capsys):
    """Requirement L: Failed agent execution exits with non-zero exit code."""
    monkeypatch.setenv("AGENT_LOG_LEVEL", "normal")
    init_git_repo(tmp_path)

    with patch("sys.argv", ["agent.py", "Fail test", str(tmp_path)]):
        with patch("app.agent.run_agent", side_effect=RuntimeError("Provider exhausted")):
            with pytest.raises(SystemExit) as exc_info:
                main()
            assert exc_info.value.code != 0


def test_phase16e_workspace_validation_failure_exit_code_nonzero(tmp_path: Path, monkeypatch, capsys):
    """Requirement M: Workspace validation failure exits with non-zero exit code."""
    monkeypatch.setenv("AGENT_LOG_LEVEL", "normal")
    nonexistent = str(tmp_path / "missing")

    with patch("sys.argv", ["agent.py", "Test goal", nonexistent]):
        with pytest.raises(SystemExit) as exc_info:
            main()
        assert exc_info.value.code != 0
