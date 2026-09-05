"""Pytest integration suite for Phase 15 End-to-End Autonomous Agent Evaluation & Finalization."""

import json
import os
from pathlib import Path
from typing import List, Any, Optional

import pytest
from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import AIMessage, BaseMessage, ToolMessage, HumanMessage

from app.state import AgentState, ValidationResult, VerificationResult
from app.agent import run_agent, approve_task
from app.multi_agent import run_multi_agent
from app.evaluation import (
    ExecutionTrace,
    export_report_json,
    generate_evaluation_report,
)
from app.benchmark import (
    BenchmarkTask,
    ExperimentConfig,
    run_benchmark_task,
    export_benchmark_json,
    export_benchmark_csv,
)
from app.sandbox import ExecutionSandbox, SecurityError
from app.tools import create_workspace_tools, safe_resolve_path


class MockLLM(BaseChatModel):
    """Deterministic Mock LLM for Phase 15 end-to-end testing."""

    responses: List[AIMessage]
    call_count: int = 0

    def __init__(self, responses: List[AIMessage]):
        super().__init__(responses=list(responses), call_count=0)

    def _generate(self, messages: List[BaseMessage], stop: Optional[List[str]] = None, **kwargs: Any):
        if self.call_count < len(self.responses):
            resp = self.responses[self.call_count]
            self.call_count += 1
        else:
            resp = AIMessage(content="STATUS: APPROVED\nGoal complete.")
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
    """Fixture creating an isolated temporary workspace repository."""
    (tmp_path / "src").mkdir(parents=True, exist_ok=True)
    (tmp_path / "src" / "calc.py").write_text("def add(a, b):\n    return a + b\n")
    (tmp_path / "tests").mkdir(parents=True, exist_ok=True)
    (tmp_path / "tests" / "test_calc.py").write_text("from src.calc import add\n\ndef test_add():\n    assert add(2, 3) == 5\n")
    return tmp_path


# -----------------------------------------------------------------------------
# TEST 1: Simple successful repository task
# -----------------------------------------------------------------------------
def test_e2e_simple_successful_task(tmp_workspace: Path):
    """TEST 1: Simple successful repository task producing SUCCESS outcome."""
    mock_llm = MockLLM([
        AIMessage(content="Inspecting repository"),
        AIMessage(content="Goal completed successfully"),
    ])

    state = run_agent(
        goal="Inspect calc.py and verify functions",
        workspace_root=str(tmp_workspace),
        llm=mock_llm,
    )

    assert state["final_outcome"] == "SUCCESS"
    assert state["status"] == "completed"


# -----------------------------------------------------------------------------
# TEST 2: Task requiring code modification
# -----------------------------------------------------------------------------
def test_e2e_code_modification_task(tmp_workspace: Path):
    """TEST 2: Task modifying repository code via tools."""
    mock_llm = MockLLM([
        AIMessage(
            content="Modifying code",
            tool_calls=[{
                "name": "write_file",
                "args": {"file_path": "src/calc.py", "content": "def add(a, b):\n    return a + b\n\ndef sub(a, b):\n    return a - b\n"},
                "id": "call_mod_1",
            }],
        ),
        AIMessage(content="Code modified successfully"),
    ])

    state = run_agent(
        goal="Add sub function to calc.py",
        workspace_root=str(tmp_workspace),
        llm=mock_llm,
    )

    assert "src/calc.py" in (state.get("modified_files") or [])
    assert (tmp_workspace / "src" / "calc.py").read_text().find("sub") != -1


# -----------------------------------------------------------------------------
# TEST 3: Task where initial implementation fails and agent self-corrects
# -----------------------------------------------------------------------------
def test_e2e_self_correction_recovery(tmp_workspace: Path):
    """TEST 3: Task where initial implementation fails validation and agent self-corrects."""
    mock_llm = MockLLM([
        AIMessage(
            content="Writing buggy code",
            tool_calls=[{"name": "write_file", "args": {"file_path": "src/calc.py", "content": "def add(a, b):\n    return a * b\n"}, "id": "call_1"}],
        ),
        AIMessage(
            content="Running tests",
            tool_calls=[{"name": "run_tests", "args": {}, "id": "call_2"}],
        ),
        AIMessage(
            content="Self-correcting code after test failure",
            tool_calls=[{"name": "write_file", "args": {"file_path": "src/calc.py", "content": "def add(a, b):\n    return a + b\n"}, "id": "call_3"}],
        ),
        AIMessage(
            content="Re-running tests",
            tool_calls=[{"name": "run_tests", "args": {}, "id": "call_4"}],
        ),
        AIMessage(content="Self correction passed"),
    ])

    state = run_agent(
        goal="Fix add function and make tests pass",
        workspace_root=str(tmp_workspace),
        llm=mock_llm,
    )

    report = state.get("evaluation_report", {})
    assert report.get("recovery_attempts", 0) >= 1
    assert report.get("successful_recoveries", 0) >= 1


# -----------------------------------------------------------------------------
# TEST 4: Task requiring repository retrieval
# -----------------------------------------------------------------------------
def test_e2e_retrieval_integration(tmp_workspace: Path):
    """TEST 4: Task using hybrid context retrieval connected to downstream action."""
    mock_llm = MockLLM([
        AIMessage(
            content="Retrieving code context",
            tool_calls=[{"name": "retrieve_hybrid_context", "args": {"query": "add function"}, "id": "call_ret"}],
        ),
        AIMessage(content="Context retrieved and processed"),
    ])

    state = run_agent(
        goal="Locate add function using hybrid retrieval",
        workspace_root=str(tmp_workspace),
        llm=mock_llm,
    )

    assert state.get("retrieved_context") is not None
    assert len(state.get("retrieved_context", [])) >= 1


# -----------------------------------------------------------------------------
# TEST 5: Task using multi-agent mode
# -----------------------------------------------------------------------------
def test_e2e_multi_agent_workflow(tmp_workspace: Path):
    """TEST 5: Task using Phase 12 multi-agent orchestration (Analyzer -> Coder -> Reviewer)."""
    mock_llm = MockLLM([
        AIMessage(content="Analyzer: plan created."),
        AIMessage(content="Coder: code modified."),
        AIMessage(content="STATUS: APPROVED\nReviewer: code looks great."),
    ])

    state = run_multi_agent(
        goal="Refactor calc module",
        workspace_root=str(tmp_workspace),
        llm=mock_llm,
    )

    assert state["mode"] == "multi_agent"
    assert state["review_status"] == "approved"
    assert state["final_outcome"] == "SUCCESS"


# -----------------------------------------------------------------------------
# TEST 6: Task requiring human approval (ESCALATED)
# -----------------------------------------------------------------------------
def test_e2e_human_approval_required(tmp_workspace: Path):
    """TEST 6: Task requiring human approval returns ESCALATED status when pending."""
    mock_llm = MockLLM([
        AIMessage(
            content="Requesting approval for commit",
            tool_calls=[{"name": "request_human_approval", "args": {"action": "commit", "reason": "Commit calc fix"}, "id": "c_app"}],
        ),
    ])

    state = run_agent(
        goal="Commit changes",
        workspace_root=str(tmp_workspace),
        llm=mock_llm,
    )

    assert state["approval_required"] is True
    assert state["approval_status"] == "pending"
    assert state["final_outcome"] == "ESCALATED"


# -----------------------------------------------------------------------------
# TEST 7: Task where human approval is rejected
# -----------------------------------------------------------------------------
def test_e2e_human_approval_rejected(tmp_workspace: Path, tmp_path: Path):
    """TEST 7: Task where human approval is explicitly rejected by human decision."""
    task_id = "test_e2e_reject_7"
    initial_state: AgentState = {
        "task_id": task_id,
        "user_goal": "Commit fix",
        "workspace_root": str(tmp_workspace),
        "status": "paused",
        "approval_required": True,
        "approval_status": "pending",
        "approval_reason": "Human commit request",
        "delivery_action": "commit",
        "messages": [],
    }
    from app.memory import save_state
    storage_dir = tmp_path / ".agent_memory"
    save_state(task_id, initial_state, status="paused", storage_dir=storage_dir)

    updated_state = approve_task(
        task_id=task_id,
        decision="reject",
        notes="Unsafe changes",
        workspace_root=str(tmp_workspace),
        storage_dir=storage_dir,
    )

    assert updated_state["approval_status"] == "rejected"
    assert updated_state["status"] == "paused"


# -----------------------------------------------------------------------------
# TEST 8: Task where goal verification fails
# -----------------------------------------------------------------------------
def test_e2e_goal_verification_failure(tmp_workspace: Path):
    """TEST 8: Task where goal verification explicitly fails producing FAILED outcome."""
    mock_llm = MockLLM([
        AIMessage(
            content="Evaluating goal",
            tool_calls=[{"name": "verify_goal", "args": {"status": "failed", "summary": "Goal criteria not satisfied"}, "id": "c_ver"}],
        ),
    ])

    state = run_agent(
        goal="Verify criteria",
        workspace_root=str(tmp_workspace),
        llm=mock_llm,
    )

    ver_res = state.get("verification_result")
    assert ver_res is not None
    assert ver_res.get("status") == "failed"
    assert state["final_outcome"] == "FAILED"


# -----------------------------------------------------------------------------
# TEST 9: Task exceeding retry/recovery limit
# -----------------------------------------------------------------------------
def test_e2e_exceeding_retry_limit(tmp_workspace: Path):
    """TEST 9: Task exceeding retry count limit returns FAILED outcome."""
    state: AgentState = {
        "task_id": "test_retry_limit",
        "user_goal": "Fix bug",
        "workspace_root": str(tmp_workspace),
        "status": "failed",
        "retry_count": 3,
        "max_retries": 3,
        "validation_result": {"status": "failed", "output": "Tests still failing"},
    }

    report = generate_evaluation_report(state)
    assert report["final_outcome"] == "FAILED"


# -----------------------------------------------------------------------------
# TEST 10: Security boundary regression
# -----------------------------------------------------------------------------
def test_e2e_security_boundary_regression(tmp_workspace: Path):
    """TEST 10: Path traversal and dangerous command execution remain strictly blocked in sandbox."""
    sandbox = ExecutionSandbox(sandbox_root=str(tmp_workspace))

    # Path traversal outside sandbox root blocked
    with pytest.raises(SecurityError):
        sandbox.safe_resolve_path("../../../etc/passwd")

    # Unsafe command blocked
    res = sandbox.execute_command(["rm", "-rf", "/"])
    assert "Security violation" in res["stderr"] or res["exit_code"] != 0


# -----------------------------------------------------------------------------
# TEST 11: Benchmark & evaluation metrics produced
# -----------------------------------------------------------------------------
def test_e2e_benchmark_metrics_produced(tmp_workspace: Path):
    """TEST 11: Benchmark task execution produces valid report metrics."""
    task: BenchmarkTask = {
        "task_id": "T_e2e_11",
        "name": "E2E Benchmark Test",
        "category": "bug_fix",
        "difficulty": "easy",
        "repository_setup": {"src/calc.py": "def add(a, b):\n    return a + b\n"},
        "description": "Verify add function",
        "expected_behavior": "Add passes",
        "verification": {},
    }
    config: ExperimentConfig = {"agent_mode": "single_agent", "retrieval_mode": "hybrid"}
    mock_llm = MockLLM([AIMessage(content="Verified task")])

    res = run_benchmark_task(task, config=config, run_index=1, llm=mock_llm)

    assert res["task_id"] == "T_e2e_11"
    assert res["execution_time"] > 0
    assert "tool_call_count" in res


# -----------------------------------------------------------------------------
# TEST 12: Final result JSON export sanitized
# -----------------------------------------------------------------------------
def test_e2e_final_result_export_sanitized(tmp_path: Path):
    """TEST 12: Exported report JSON scrubs all API keys and secret credentials."""
    report = {
        "task_id": "e2e_export_12",
        "final_outcome": "SUCCESS",
        "metadata": {
            "api_key": "sk-proj-99999999999999999999",
            "auth_header": "Bearer secret_token_12345",
        },
    }

    out_file = tmp_path / "e2e_sanitized.json"
    json_text = export_report_json(report, file_path=out_file)

    assert "sk-proj-99999999999999999999" not in json_text
    assert "secret_token_12345" not in json_text
    assert "[REDACTED_SECRET]" in json_text
