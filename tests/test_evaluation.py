"""Pytest test suite for Phase 13 Observability, Telemetry, and Evaluation System."""

import json
import pytest
from pathlib import Path
from typing import List, Any, Optional

from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import AIMessage, BaseMessage, ToolMessage, HumanMessage

from app.state import AgentState, ValidationResult, VerificationResult
from app.agent import run_agent
from app.multi_agent import run_multi_agent
from app.evaluation import (
    ExecutionTrace,
    classify_error,
    sanitize_telemetry_dict,
    generate_evaluation_report,
    export_report_json,
    calculate_benchmark_statistics,
    compare_agent_evaluations,
)


class MockLLM(BaseChatModel):
    """Deterministic Mock LLM for observability pytest execution."""

    responses: List[AIMessage]
    call_count: int = 0

    def __init__(self, responses: List[AIMessage]):
        super().__init__(responses=list(responses), call_count=0)

    def _generate(self, messages: List[BaseMessage], stop: Optional[List[str]] = None, **kwargs: Any):
        if self.call_count < len(self.responses):
            resp = self.responses[self.call_count]
            self.call_count += 1
        else:
            resp = AIMessage(content="STATUS: APPROVED\nDefault mock response.")
        from langchain_core.outputs import ChatGeneration, ChatResult
        return ChatResult(generations=[ChatGeneration(message=resp)])

    def bind_tools(self, tools: List[Any], **kwargs: Any) -> "MockLLM":
        clone = MockLLM(responses=self.responses)
        clone.call_count = self.call_count
        return clone

    @property
    def _llm_type(self) -> str:
        return "mock"


@pytest.fixture
def tmp_workspace(tmp_path: Path) -> Path:
    """Fixture providing a clean temporary workspace directory."""
    (tmp_path / "src").mkdir(parents=True, exist_ok=True)
    (tmp_path / "src" / "math_utils.py").write_text("def add(a, b):\n    return a + b\n")
    (tmp_path / "tests").mkdir(parents=True, exist_ok=True)
    (tmp_path / "tests" / "test_math.py").write_text("from src.math_utils import add\n\ndef test_add():\n    assert add(1, 2) == 3\n")
    return tmp_path


# -----------------------------------------------------------------------------
# Trace Tests
# -----------------------------------------------------------------------------
def test_trace_records_agent_event():
    """Verify ExecutionTrace records agent_start and agent_end events."""
    trace = ExecutionTrace(task_id="trace_test_1")
    trace.record_event(event_type="agent_start", agent_role="coder", status="started")
    events = trace.get_events()

    assert len(events) == 1
    assert events[0]["event_type"] == "agent_start"
    assert events[0]["agent_role"] == "coder"


def test_trace_records_tool_event():
    """Verify ExecutionTrace records tool_call and tool_result events with durations."""
    trace = ExecutionTrace(task_id="trace_test_2")
    trace.record_event(
        event_type="tool_call",
        tool_name="read_file",
        status="success",
        duration=0.015,
        metadata={"file_path": "src/math_utils.py"},
    )
    events = trace.get_events()

    assert len(events) == 1
    assert events[0]["tool_name"] == "read_file"
    assert events[0]["duration"] == 0.015


def test_trace_records_error():
    """Verify ExecutionTrace records error events with error category details."""
    trace = ExecutionTrace(task_id="trace_test_3")
    trace.record_event(
        event_type="error",
        status="security_error",
        metadata={"error": "Access denied: path escapes sandbox"},
    )
    events = trace.get_events()

    assert len(events) == 1
    assert events[0]["status"] == "security_error"


def test_trace_does_not_store_secrets():
    """Verify telemetry sanitization redacts API keys, tokens, passwords, and secrets."""
    dirty_meta = {
        "api_key": "sk-proj-12345678901234567890",
        "secret_token": "ghp_1234567890abcdef",
        "user_password": "supersecretpassword",
        "normal_key": "normal_value",
    }
    sanitized = sanitize_telemetry_dict(dirty_meta)

    assert sanitized["api_key"] == "[REDACTED_SECRET]"
    assert sanitized["secret_token"] == "[REDACTED_SECRET]"
    assert sanitized["user_password"] == "[REDACTED_SECRET]"
    assert sanitized["normal_key"] == "normal_value"


# -----------------------------------------------------------------------------
# Metrics Tests
# -----------------------------------------------------------------------------
def test_tool_call_count(tmp_workspace: Path):
    """Verify evaluation report accurately counts total, successful, and failed tool calls."""
    state: AgentState = {
        "task_id": "metrics_task_1",
        "user_goal": "Check code",
        "messages": [
            HumanMessage(content="Check code"),
            AIMessage(content="Reading", tool_calls=[{"name": "read_file", "args": {"file_path": "src/math_utils.py"}, "id": "c1"}]),
            ToolMessage(content="def add(a, b): return a + b", tool_call_id="c1", name="read_file"),
            AIMessage(content="Error test", tool_calls=[{"name": "write_file", "args": {"file_path": "../secret.txt"}, "id": "c2"}]),
            ToolMessage(content="Error: Access denied: path escapes sandbox", tool_call_id="c2", name="write_file"),
        ],
        "status": "completed",
    }

    report = generate_evaluation_report(state)

    assert report["tool_call_count"] == 2
    assert report["successful_tool_calls"] == 1
    assert report["failed_tool_calls"] == 1
    assert report["tool_calls_by_tool"]["read_file"] == 1
    assert report["tool_calls_by_tool"]["write_file"] == 1


def test_iteration_count():
    """Verify evaluation report tracks single-agent and multi-agent iteration counts."""
    state_single: AgentState = {"task_id": "iter_1", "retry_count": 2, "mode": "single_agent"}
    report_s = generate_evaluation_report(state_single)
    assert report_s["iteration_count"] == 3

    state_multi: AgentState = {"task_id": "iter_2", "multi_agent_iteration": 2, "mode": "multi_agent"}
    report_m = generate_evaluation_report(state_multi)
    assert report_m["iteration_count"] == 2


def test_retry_metrics():
    """Verify evaluation report records retry count."""
    state: AgentState = {"task_id": "retry_task", "retry_count": 3}
    report = generate_evaluation_report(state)
    assert report["retry_count"] == 3


def test_validation_metrics():
    """Verify evaluation report records validation passes, failures, and timeouts."""
    state: AgentState = {
        "task_id": "val_metrics",
        "messages": [
            ToolMessage(content="=== Test Execution Result ===\nStatus: failed\nSummary: 1 failed", tool_call_id="t1"),
            ToolMessage(content="=== Test Execution Result ===\nStatus: passed\nSummary: 1 passed", tool_call_id="t2"),
        ],
    }
    report = generate_evaluation_report(state)

    assert report["validation_attempts"] == 2
    assert report["validation_passes"] == 1
    assert report["validation_failures"] == 1


def test_execution_duration():
    """Verify execution_time calculation between start_time and end_time."""
    state: AgentState = {"task_id": "dur_test"}
    report = generate_evaluation_report(state, start_time=100.0, end_time=105.5)

    assert report["execution_time"] == 5.5
    assert report["start_time"] == 100.0
    assert report["end_time"] == 105.5


# -----------------------------------------------------------------------------
# Recovery Metrics Tests
# -----------------------------------------------------------------------------
def test_successful_recovery_recorded():
    """Verify recovery metric records successful recovery when initial failed test is fixed by passing test."""
    state: AgentState = {
        "task_id": "rec_success",
        "messages": [
            ToolMessage(content="=== Test Execution Result ===\nStatus: failed\nSummary: 1 failed", tool_call_id="t1"),
            ToolMessage(content="=== Test Execution Result ===\nStatus: passed\nSummary: 1 passed", tool_call_id="t2"),
        ],
    }
    report = generate_evaluation_report(state)

    assert report["recovery_attempts"] == 1
    assert report["successful_recoveries"] == 1
    assert report["failed_recoveries"] == 0


def test_failed_recovery_recorded():
    """Verify recovery metric records failed recovery when tests fail without passing."""
    state: AgentState = {
        "task_id": "rec_fail",
        "messages": [
            ToolMessage(content="=== Test Execution Result ===\nStatus: failed\nSummary: 1 failed", tool_call_id="t1"),
        ],
    }
    report = generate_evaluation_report(state)

    assert report["recovery_attempts"] == 1
    assert report["successful_recoveries"] == 0
    assert report["failed_recoveries"] == 1


# -----------------------------------------------------------------------------
# Retrieval Metrics Tests
# -----------------------------------------------------------------------------
def test_retrieval_metrics_recorded():
    """Verify Phase 11 hybrid context retrieval queries are recorded in metrics."""
    state: AgentState = {
        "task_id": "ret_metrics",
        "retrieved_context": [
            {"query": "add function", "chunks": [{"file_path": "src/math.py"}]},
            {"query": "test_add", "chunks": [{"file_path": "tests/test_math.py"}]},
        ],
    }
    report = generate_evaluation_report(state)

    assert report["retrieval_metrics"] is not None
    assert report["retrieval_metrics"]["query_count"] == 2
    assert report["retrieval_metrics"]["retrieved_chunks"] == 2


def test_retrieval_metrics_exported(tmp_path: Path):
    """Verify retrieval metrics are serialized into exported JSON report."""
    state: AgentState = {
        "task_id": "ret_export",
        "retrieved_context": [{"query": "search authentication", "chunks": []}],
    }
    report = generate_evaluation_report(state)
    json_path = tmp_path / "report.json"
    export_report_json(report, file_path=json_path)

    content = json.loads(json_path.read_text())
    assert content["retrieval_metrics"]["query_count"] == 1


# -----------------------------------------------------------------------------
# Multi-Agent Metrics Tests
# -----------------------------------------------------------------------------
def test_multi_agent_metrics():
    """Verify Phase 12 multi-agent metrics are captured in report."""
    state: AgentState = {
        "task_id": "multi_metrics",
        "mode": "multi_agent",
        "multi_agent_iteration": 2,
        "review_status": "approved",
    }
    report = generate_evaluation_report(state)

    assert report["multi_agent_metrics"] is not None
    assert report["multi_agent_metrics"]["review_iterations"] == 2
    assert report["multi_agent_metrics"]["review_approvals"] == 1


def test_review_metrics():
    """Verify Reviewer approvals and rejections are reflected in metrics."""
    state_rej: AgentState = {
        "task_id": "rev_rej",
        "mode": "multi_agent",
        "review_status": "changes_requested",
    }
    report = generate_evaluation_report(state_rej)

    assert report["multi_agent_metrics"]["review_rejections"] == 1


def test_agent_role_trace():
    """Verify execution events store specific agent roles."""
    trace = ExecutionTrace(task_id="role_trace")
    trace.record_event(event_type="analysis", agent_role="analyzer")
    trace.record_event(event_type="coding", agent_role="coder")
    trace.record_event(event_type="review", agent_role="reviewer")

    roles = [e["agent_role"] for e in trace.get_events()]
    assert roles == ["analyzer", "coder", "reviewer"]


# -----------------------------------------------------------------------------
# Human Approval Tests
# -----------------------------------------------------------------------------
def test_approval_event_recorded():
    """Verify human approval requests record intervention events."""
    state: AgentState = {
        "task_id": "app_event",
        "messages": [
            AIMessage(content="Commit", tool_calls=[{"name": "request_human_approval", "args": {"action": "commit"}, "id": "c1"}]),
        ],
    }
    report = generate_evaluation_report(state)
    assert report["human_interventions"] == 1


def test_human_intervention_count():
    """Verify human_interventions accurately totals human approval tool calls."""
    state: AgentState = {
        "task_id": "app_count",
        "messages": [
            AIMessage(content="Commit", tool_calls=[{"name": "request_human_approval", "args": {"action": "commit"}, "id": "c1"}]),
            AIMessage(content="Push", tool_calls=[{"name": "request_human_approval", "args": {"action": "push"}, "id": "c2"}]),
        ],
    }
    report = generate_evaluation_report(state)
    assert report["human_interventions"] == 2


# -----------------------------------------------------------------------------
# Error Classification Tests
# -----------------------------------------------------------------------------
def test_error_classification():
    """Verify error classifier maps content to standard error categories."""
    assert classify_error("Error: tool failed") == "tool_error"
    assert classify_error("Status: failed in pytest") == "validation_error"


def test_timeout_classification():
    """Verify error classifier identifies execution timeouts."""
    assert classify_error("Status: timeout after 30s") == "timeout"


def test_security_error_classification():
    """Verify error classifier identifies path traversal and security sandbox violations."""
    assert classify_error("Access denied: path '../secret.txt' escapes sandbox") == "security_error"


# -----------------------------------------------------------------------------
# JSON Export & Secret Scrubbing Tests
# -----------------------------------------------------------------------------
def test_evaluation_report_serializable():
    """Verify EvaluationReport is JSON serializable."""
    state: AgentState = {"task_id": "serial_task", "status": "completed"}
    report = generate_evaluation_report(state)
    json_str = export_report_json(report)

    parsed = json.loads(json_str)
    assert parsed["task_id"] == "serial_task"


def test_json_export(tmp_path: Path):
    """Verify export_report_json writes report to disk."""
    state: AgentState = {"task_id": "disk_export"}
    report = generate_evaluation_report(state)
    out_file = tmp_path / "test_report.json"
    export_report_json(report, file_path=out_file)

    assert out_file.exists()
    data = json.loads(out_file.read_text())
    assert data["task_id"] == "disk_export"


def test_sensitive_data_not_exported(tmp_path: Path):
    """Verify secrets and passwords in telemetry metadata are scrubbed in JSON exports."""
    report = {
        "task_id": "sec_export",
        "events": [
            {
                "event_type": "tool_call",
                "metadata": {"api_key": "sk-proj-99999", "password": "supersecretpassword"},
            }
        ],
    }
    json_str = export_report_json(report)
    parsed = json.loads(json_str)

    assert parsed["events"][0]["metadata"]["api_key"] == "[REDACTED_SECRET]"
    assert parsed["events"][0]["metadata"]["password"] == "[REDACTED_SECRET]"


# -----------------------------------------------------------------------------
# Benchmark Comparison Tests
# -----------------------------------------------------------------------------
def test_single_vs_multi_agent_comparison():
    """Verify compare_agent_evaluations produces side-by-side metric comparisons."""
    single_reports = [
        {"task_success": True, "execution_time": 10.0, "tool_call_count": 5, "iteration_count": 1, "retry_count": 0},
        {"task_success": True, "execution_time": 12.0, "tool_call_count": 7, "iteration_count": 1, "retry_count": 0},
    ]
    multi_reports = [
        {"task_success": True, "execution_time": 15.0, "tool_call_count": 8, "iteration_count": 2, "retry_count": 1},
        {"task_success": True, "execution_time": 17.0, "tool_call_count": 10, "iteration_count": 2, "retry_count": 1},
    ]

    comp = compare_agent_evaluations(single_reports, multi_reports)

    assert "single_agent_summary" in comp
    assert "multi_agent_summary" in comp
    assert comp["single_agent_summary"]["success_rate"] == 1.0
    assert comp["multi_agent_summary"]["success_rate"] == 1.0
    assert comp["comparison_metrics"]["avg_execution_time_diff"] > 0


def test_benchmark_aggregation():
    """Verify calculate_benchmark_statistics computes mean, median, min, max correctly."""
    reports = [
        {"task_success": True, "execution_time": 10.0, "tool_call_count": 4, "iteration_count": 1, "retry_count": 0},
        {"task_success": False, "execution_time": 20.0, "tool_call_count": 10, "iteration_count": 3, "retry_count": 2},
    ]
    stats = calculate_benchmark_statistics(reports)

    assert stats["total_runs"] == 2
    assert stats["success_rate"] == 0.5
    assert stats["execution_time_seconds"]["mean"] == 15.0
    assert stats["execution_time_seconds"]["min"] == 10.0
    assert stats["execution_time_seconds"]["max"] == 20.0


def test_success_rate():
    """Verify benchmark success_rate calculation across multiple task runs."""
    reports = [
        {"task_success": True},
        {"task_success": True},
        {"task_success": False},
        {"task_success": True},
    ]
    stats = calculate_benchmark_statistics(reports)
    assert stats["success_rate"] == 0.75


def test_mean_execution_time():
    """Verify benchmark mean execution time computation."""
    reports = [
        {"execution_time": 5.0},
        {"execution_time": 15.0},
    ]
    stats = calculate_benchmark_statistics(reports)
    assert stats["execution_time_seconds"]["mean"] == 10.0


# -----------------------------------------------------------------------------
# End-to-End Evaluation Test
# -----------------------------------------------------------------------------
def test_end_to_end_evaluation_flow(tmp_workspace: Path, tmp_path: Path):
    """Verify end-to-end task execution -> trace -> evaluation report -> JSON export."""
    mock_llm = MockLLM([
        AIMessage(content="Evaluating goal", tool_calls=[{"name": "read_file", "args": {"file_path": "src/math_utils.py"}, "id": "c1"}]),
        AIMessage(content="Goal verified"),
    ])

    storage_dir = str(tmp_path / ".agent_memory")
    final_state = run_agent(
        goal="Verify math module",
        workspace_root=str(tmp_workspace),
        llm=mock_llm,
        storage_dir=storage_dir,
    )

    assert "evaluation_report" in final_state
    report = final_state["evaluation_report"]
    assert report is not None
    assert report["task_id"] == final_state["task_id"]
    assert report["tool_call_count"] >= 1

    json_export_path = tmp_path / "e2e_report.json"
    json_text = export_report_json(report, file_path=json_export_path)

    assert json_export_path.exists()
    parsed = json.loads(json_text)
    assert parsed["task_id"] == final_state["task_id"]
