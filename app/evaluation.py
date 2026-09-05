"""Observability, telemetry, and evaluation module for Phase 13 Autonomous Coding Agent."""

import json
import os
import re
import statistics
import time
from pathlib import Path
from typing import Any, Sequence, TypedDict

from langchain_core.messages import AIMessage, BaseMessage, ToolMessage
from app.state import AgentState, ExecutionEvent, EvaluationReport


# -----------------------------------------------------------------------------
# Security & Secret Sanitization
# -----------------------------------------------------------------------------
SENSITIVE_KEY_PATTERNS = [
    r"api[-_]?key",
    r"apikey",
    r"access[-_]?token",
    r"auth[-_]?token",
    r"token",
    r"secret",
    r"password",
    r"credential",
    r"auth",
    r"authorization",
    r"client[-_]?secret",
    r"private[-_]?key",
    r"bearer",
]

SENSITIVE_VALUE_PATTERNS = [
    r"sk-[a-zA-Z0-9_-]{5,}",
    r"ghp_[a-zA-Z0-9_-]{5,}",
    r"-----BEGIN PRIVATE KEY-----[^-----]*-----END PRIVATE KEY-----",
    r"bearer\s+[a-zA-Z0-9_.\-]+",
]


def sanitize_telemetry_value(value: Any) -> Any:
    """Sanitizes sensitive strings, credentials, or file dumps from telemetry values."""
    if isinstance(value, str):
        text = value
        # Scrub explicit secret pattern matches embedded in string
        for pat in SENSITIVE_VALUE_PATTERNS:
            text = re.sub(pat, "[REDACTED_SECRET]", text, flags=re.IGNORECASE)

        # Scrub key=val or key: val patterns inside string
        kv_pattern = r"((?:api[-_]?key|apikey|access[-_]?token|auth[-_]?token|secret|password|authorization|client[-_]?secret|private[-_]?key)\s*[:=]\s*['\"]?)([a-zA-Z0-9_.\-]{3,})"
        text = re.sub(kv_pattern, r"\1[REDACTED_SECRET]", text, flags=re.IGNORECASE)

        # Truncate overly long content strings
        if len(text) > 2000:
            return text[:2000] + " ... [truncated]"
        return text

    if isinstance(value, dict):
        return sanitize_telemetry_dict(value)

    if isinstance(value, (list, tuple)):
        return [sanitize_telemetry_value(item) for item in value]

    return value


def sanitize_telemetry_dict(data: dict[str, Any]) -> dict[str, Any]:
    """Recursively redacts sensitive keys and values from telemetry metadata dicts."""
    if not isinstance(data, dict):
        return sanitize_telemetry_value(data)

    sanitized: dict[str, Any] = {}
    for key, val in data.items():
        key_str = str(key)
        # Check key name
        is_sensitive = any(re.search(pat, key_str, re.IGNORECASE) for pat in SENSITIVE_KEY_PATTERNS)
        if is_sensitive:
            sanitized[key_str] = "[REDACTED_SECRET]"
        else:
            sanitized[key_str] = sanitize_telemetry_value(val)
    return sanitized


# -----------------------------------------------------------------------------
# Execution Trace
# -----------------------------------------------------------------------------
class ExecutionTrace:
    """In-memory lightweight telemetry execution trace builder."""

    def __init__(self, task_id: str = "task_default"):
        self.task_id = task_id
        self.start_time = time.time()
        self.events: list[ExecutionEvent] = []

    def record_event(
        self,
        event_type: str,
        agent_role: str | None = None,
        tool_name: str | None = None,
        status: str | None = None,
        duration: float | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> ExecutionEvent:
        """Appends a sanitized ExecutionEvent to the trace log."""
        sanitized_meta = sanitize_telemetry_dict(metadata) if metadata else None
        event: ExecutionEvent = {
            "timestamp": round(time.time(), 3),
            "event_type": str(event_type),
            "agent_role": str(agent_role) if agent_role else None,
            "tool_name": str(tool_name) if tool_name else None,
            "status": str(status) if status else None,
            "duration": round(float(duration), 4) if duration is not None else None,
            "metadata": sanitized_meta,
        }
        self.events.append(event)
        return event

    def get_events(self) -> list[ExecutionEvent]:
        """Returns the recorded execution events."""
        return list(self.events)


# -----------------------------------------------------------------------------
# Evaluation Report Generator
# -----------------------------------------------------------------------------
def classify_error(content: str) -> str:
    """Classifies an error message into standard security/system categories."""
    upper = content.upper()
    if "ACCESS DENIED" in upper or "SECURITY" in upper or "TRAVERSAL" in upper or "ESCAPES" in upper:
        return "security_error"
    if "STATUS: TIMEOUT" in upper or "TIMED OUT" in upper or "TIMEOUT" in upper:
        return "timeout"
    if "STATUS: FAILED" in upper or "TEST" in upper or "ASSERTIONERROR" in upper:
        return "validation_error"
    if "GOAL VERIFICATION" in upper or "VERIFICATION" in upper:
        return "verification_error"
    if "RETRIEVAL" in upper or "HYBRID" in upper:
        return "retrieval_error"
    if "APPROVAL" in upper or "HUMAN APPROVAL" in upper:
        return "approval_error"
    if "ERROR:" in upper or "TOOL" in upper:
        return "tool_error"
    return "agent_error"


def generate_evaluation_report(
    state: AgentState,
    start_time: float | None = None,
    end_time: float | None = None,
) -> EvaluationReport:
    """Calculates comprehensive evaluation metrics and builds a serializable EvaluationReport."""
    task_id = str(state.get("task_id") or "task_default")
    messages = state.get("messages", [])
    val_res = state.get("validation_result")
    ver_res = state.get("verification_result")
    retrieved_ctx = state.get("retrieved_context") or []
    review_res = state.get("review_result")
    events = list(state.get("execution_trace") or [])

    # Calculate timestamps & duration
    t_start = start_time if start_time is not None else (events[0]["timestamp"] if events else time.time())
    t_end = end_time if end_time is not None else time.time()
    exec_time = round(max(0.0, t_end - t_start), 3)

    # Task success & final outcome calculation
    ver_status = ver_res.get("status") if isinstance(ver_res, dict) else None
    val_status = val_res.get("status") if isinstance(val_res, dict) else None
    state_status = state.get("status")
    approval_req = bool(state.get("approval_required", False))
    approval_st = state.get("approval_status", "not_required")

    if approval_req and approval_st == "pending":
        task_success = False
        final_status = "escalated"
        final_outcome = "ESCALATED"
    elif state.get("review_status") == "blocked" or state_status == "blocked":
        task_success = False
        final_status = "blocked"
        final_outcome = "ESCALATED"
    elif ver_status == "passed":
        task_success = True
        final_status = "success"
        final_outcome = "SUCCESS"
    elif ver_status == "failed":
        task_success = False
        final_status = "failed"
        final_outcome = "FAILED"
    elif state.get("review_status") == "approved":
        task_success = True
        final_status = "success"
        final_outcome = "SUCCESS"
    elif state_status == "completed" and val_status in ("passed", None) and ver_status in ("passed", None):
        task_success = True
        final_status = "success"
        final_outcome = "SUCCESS"
    elif state_status == "failed" or val_status in ("failed", "error"):
        task_success = False
        final_status = "failed"
        final_outcome = "FAILED"
    else:
        task_success = False
        final_status = "uncertain"
        final_outcome = "FAILED"

    # Analyze message history for tool calls and metrics
    tool_call_count = 0
    successful_tool_calls = 0
    failed_tool_calls = 0
    tool_calls_by_tool: dict[str, int] = {}

    validation_attempts = 0
    validation_passes = 0
    validation_failures = 0
    timeouts = 0

    human_interventions = 0
    error_categories: dict[str, int] = {}
    error_count = 0

    input_tokens = None
    output_tokens = None
    total_tokens = None

    for msg in messages:
        # Check token usage metadata if available
        resp_meta = getattr(msg, "response_metadata", {}) or {}
        token_usage = resp_meta.get("token_usage") or resp_meta.get("usage")
        if isinstance(token_usage, dict):
            input_tokens = (input_tokens or 0) + token_usage.get("prompt_tokens", token_usage.get("input_tokens", 0))
            output_tokens = (output_tokens or 0) + token_usage.get("completion_tokens", token_usage.get("output_tokens", 0))
            total_tokens = (total_tokens or 0) + token_usage.get("total_tokens", 0)

        if isinstance(msg, AIMessage) and getattr(msg, "tool_calls", None):
            for tc in msg.tool_calls:
                tool_call_count += 1
                t_name = tc.get("name", "unknown_tool")
                tool_calls_by_tool[t_name] = tool_calls_by_tool.get(t_name, 0) + 1
                if t_name == "request_human_approval":
                    human_interventions += 1

        elif isinstance(msg, ToolMessage) or msg.__class__.__name__ == "ToolMessage":
            content = getattr(msg, "content", "")
            if content.startswith("Error:") or "Access denied" in content:
                failed_tool_calls += 1
                cat = classify_error(content)
                error_categories[cat] = error_categories.get(cat, 0) + 1
                error_count += 1
            else:
                successful_tool_calls += 1

            if "=== Test Execution Result ===" in content:
                validation_attempts += 1
                if "Status: passed" in content:
                    validation_passes += 1
                elif "Status: failed" in content:
                    validation_failures += 1
                elif "Status: timeout" in content:
                    timeouts += 1
                    error_categories["timeout"] = error_categories.get("timeout", 0) + 1
                    error_count += 1

    # Recovery metrics calculation
    retry_count = int(state.get("retry_count", 0))
    recovery_attempts = validation_failures
    if validation_passes > 0 and validation_failures > 0:
        successful_recoveries = 1
        failed_recoveries = max(0, validation_failures - 1)
    else:
        successful_recoveries = 0
        failed_recoveries = validation_failures

    # Retrieval metrics calculation
    retrieval_metrics = None
    if retrieved_ctx:
        retrieval_metrics = {
            "query_count": len(retrieved_ctx),
            "retrieved_chunks": sum(len(r.get("chunks", [])) for r in retrieved_ctx if isinstance(r, dict)),
            "queries": [r.get("query", "") for r in retrieved_ctx if isinstance(r, dict)],
        }

    # Multi-agent metrics calculation
    multi_agent_metrics = None
    if state.get("mode") == "multi_agent":
        multi_agent_metrics = {
            "orchestration_iterations": state.get("multi_agent_iteration", 0),
            "review_iterations": state.get("multi_agent_iteration", 0),
            "review_status": state.get("review_status", "none"),
            "review_approvals": 1 if state.get("review_status") == "approved" else 0,
            "review_rejections": 1 if state.get("review_status") == "changes_requested" else 0,
        }

    iteration_count = max(1, state.get("multi_agent_iteration", 1) if state.get("mode") == "multi_agent" else (retry_count + 1))

    return EvaluationReport(
        task_id=task_id,
        task_success=task_success,
        final_status=final_status,
        final_outcome=final_outcome,
        execution_time=exec_time,
        start_time=round(t_start, 3),
        end_time=round(t_end, 3),
        tool_call_count=tool_call_count,
        successful_tool_calls=successful_tool_calls,
        failed_tool_calls=failed_tool_calls,
        tool_calls_by_tool=tool_calls_by_tool,
        iteration_count=iteration_count,
        retry_count=retry_count,
        recovery_attempts=recovery_attempts,
        successful_recoveries=successful_recoveries,
        failed_recoveries=failed_recoveries,
        validation_attempts=validation_attempts,
        validation_passes=validation_passes,
        validation_failures=validation_failures,
        timeouts=timeouts,
        retrieval_metrics=retrieval_metrics,
        multi_agent_metrics=multi_agent_metrics,
        human_interventions=human_interventions,
        error_count=error_count,
        error_categories=error_categories,
        input_tokens=input_tokens,
        output_tokens=output_tokens,
        total_tokens=total_tokens,
        events=events,
    )


# -----------------------------------------------------------------------------
# JSON Export
# -----------------------------------------------------------------------------
def export_report_json(
    report: EvaluationReport | dict[str, Any],
    file_path: str | Path | None = None,
) -> str:
    """Converts evaluation report into a sanitized, indent-formatted JSON string or file."""
    report_dict = dict(report)
    sanitized_report = sanitize_telemetry_dict(report_dict)

    json_text = json.dumps(sanitized_report, indent=2, ensure_ascii=False)

    if file_path:
        p = Path(file_path)
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(json_text, encoding="utf-8")

    return json_text


# -----------------------------------------------------------------------------
# Benchmark Runner & Statistical Aggregation
# -----------------------------------------------------------------------------
class BenchmarkTask(TypedDict, total=False):
    """Structured representation of a benchmark evaluation task."""
    task_id: str
    description: str
    repository: str
    expected_behavior: str


def calculate_benchmark_statistics(reports: list[EvaluationReport | dict[str, Any]]) -> dict[str, Any]:
    """Calculates mean, median, min, max, and success rate statistics across multiple task reports."""
    if not reports:
        return {"total_runs": 0, "success_rate": 0.0}

    total_runs = len(reports)
    successes = sum(1 for r in reports if r.get("task_success"))
    success_rate = round(successes / total_runs, 3)

    exec_times = [float(r.get("execution_time", 0.0)) for r in reports]
    tool_counts = [int(r.get("tool_call_count", 0)) for r in reports]
    iter_counts = [int(r.get("iteration_count", 0)) for r in reports]
    retry_counts = [int(r.get("retry_count", 0)) for r in reports]

    def _stats(values: list[float | int]) -> dict[str, float]:
        if not values:
            return {"mean": 0.0, "median": 0.0, "min": 0.0, "max": 0.0}
        return {
            "mean": round(statistics.mean(values), 3),
            "median": round(statistics.median(values), 3),
            "min": round(min(values), 3),
            "max": round(max(values), 3),
        }

    return {
        "total_runs": total_runs,
        "success_rate": success_rate,
        "execution_time_seconds": _stats(exec_times),
        "tool_call_count": _stats(tool_counts),
        "iteration_count": _stats(iter_counts),
        "retry_count": _stats(retry_counts),
    }


def compare_agent_evaluations(
    single_agent_reports: list[EvaluationReport | dict[str, Any]],
    multi_agent_reports: list[EvaluationReport | dict[str, Any]],
) -> dict[str, Any]:
    """Computes side-by-side evaluation comparison metrics between single-agent and multi-agent modes."""
    single_stats = calculate_benchmark_statistics(single_agent_reports)
    multi_stats = calculate_benchmark_statistics(multi_agent_reports)

    return {
        "single_agent_summary": single_stats,
        "multi_agent_summary": multi_stats,
        "comparison_metrics": {
            "success_rate_diff": round(multi_stats["success_rate"] - single_stats["success_rate"], 3),
            "avg_execution_time_diff": round(
                multi_stats.get("execution_time_seconds", {}).get("mean", 0.0)
                - single_stats.get("execution_time_seconds", {}).get("mean", 0.0),
                3,
            ),
            "avg_tool_calls_diff": round(
                multi_stats.get("tool_call_count", {}).get("mean", 0.0)
                - single_stats.get("tool_call_count", {}).get("mean", 0.0),
                3,
            ),
        },
    }
