"""Pytest test suite for Phase 12 Multi-Agent Architecture and Specialized Role Coordination."""

import pytest
from pathlib import Path
from typing import List, Any, Optional

from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import AIMessage, BaseMessage
from langchain_core.outputs import ChatGeneration, ChatResult

from app.state import AgentState, ReviewStatus
from app.memory import save_state, load_state, delete_state
from app.multi_agent import (
    get_analyzer_tools,
    get_coder_tools,
    get_reviewer_tools,
    analyzer_node,
    coder_node,
    reviewer_node,
    orchestrator_node,
    route_multi_agent,
    build_multi_agent_graph,
    run_multi_agent,
    MultiAgentEvaluator,
)
from app.agent import run_agent


class MockLLM(BaseChatModel):
    """Deterministic Mock LLM for multi-agent pytest execution."""

    responses: List[AIMessage]
    call_count: int = 0

    def __init__(self, responses: List[AIMessage]):
        super().__init__(responses=list(responses), call_count=0)

    def _generate(self, messages: List[BaseMessage], stop: Optional[List[str]] = None, **kwargs: Any) -> ChatResult:
        if self.call_count < len(self.responses):
            resp = self.responses[self.call_count]
            self.call_count += 1
        else:
            resp = AIMessage(content="STATUS: APPROVED\nDefault mock response.")
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
    (tmp_path / "src" / "main.py").write_text("def hello():\n    return 'world'\n")
    (tmp_path / "tests").mkdir(parents=True, exist_ok=True)
    (tmp_path / "tests" / "test_main.py").write_text("from src.main import hello\n\ndef test_hello():\n    assert hello() == 'world'\n")
    return tmp_path


def test_analyzer_role_least_privilege(tmp_workspace: Path):
    """Verify Analyzer role tool access is strictly read-only retrieval."""
    tools = get_analyzer_tools(workspace_root=str(tmp_workspace))
    tool_names = {t.name for t in tools}

    assert "read_file" in tool_names
    assert "list_files" in tool_names
    assert "search_code" in tool_names
    assert "retrieve_hybrid_context" in tool_names

    # Ensure modification and delivery tools are excluded
    assert "write_file" not in tool_names
    assert "replace_in_file" not in tool_names
    assert "git_commit" not in tool_names
    assert "git_push" not in tool_names
    assert "create_pull_request" not in tool_names


def test_coder_role_least_privilege(tmp_workspace: Path):
    """Verify Coder role has modification and test access but not direct Git delivery tools."""
    tools = get_coder_tools(workspace_root=str(tmp_workspace))
    tool_names = {t.name for t in tools}

    assert "write_file" in tool_names
    assert "replace_in_file" in tool_names
    assert "run_tests" in tool_names

    # Ensure direct git delivery actions are restricted
    assert "git_commit" not in tool_names
    assert "git_push" not in tool_names
    assert "create_pull_request" not in tool_names


def test_reviewer_role_least_privilege(tmp_workspace: Path):
    """Verify Reviewer role has inspection and test verification tools but no file modification tools."""
    tools = get_reviewer_tools(workspace_root=str(tmp_workspace))
    tool_names = {t.name for t in tools}

    assert "git_diff" in tool_names
    assert "git_status" in tool_names
    assert "run_tests" in tool_names
    assert "verify_goal" in tool_names

    # Ensure modification and delivery tools are excluded
    assert "write_file" not in tool_names
    assert "replace_in_file" not in tool_names
    assert "git_commit" not in tool_names
    assert "git_push" not in tool_names


def test_multi_agent_state_fields():
    """Verify AgentState typed dict supports Phase 12 multi-agent fields."""
    state: AgentState = {
        "task_id": "test_multi_1",
        "user_goal": "Implement feature X",
        "workspace_root": ".",
        "mode": "multi_agent",
        "agent_role": "analyzer",
        "analysis_result": {
            "summary": "Analysis complete",
            "affected_files": ["src/main.py"],
            "recommended_approach": "Add function X",
            "risk_assessment": "Low risk",
        },
        "coding_result": {
            "summary": "Added function X",
            "modified_files": ["src/main.py"],
        },
        "review_result": {
            "status": "approved",
            "feedback": "LGTM",
            "score": 1.0,
            "issues": [],
        },
        "review_status": "approved",
        "review_feedback": "LGTM",
        "multi_agent_iteration": 1,
        "max_multi_agent_iterations": 3,
    }

    assert state["mode"] == "multi_agent"
    assert state["agent_role"] == "analyzer"
    assert state["review_status"] == "approved"
    assert state["multi_agent_iteration"] == 1


def test_multi_agent_memory_persistence(tmp_path: Path):
    """Verify save_state and load_state serialize and restore all Phase 12 fields."""
    task_id = "test_multi_persist_12"
    initial_state: AgentState = {
        "task_id": task_id,
        "user_goal": "Refactor auth module",
        "workspace_root": str(tmp_path),
        "status": "running",
        "mode": "multi_agent",
        "agent_role": "reviewer",
        "analysis_result": {"summary": "Analysis done"},
        "coding_result": {"summary": "Coding done"},
        "review_result": {"status": "approved", "feedback": "Code clean"},
        "review_status": "approved",
        "review_feedback": "Code clean",
        "multi_agent_iteration": 2,
        "max_multi_agent_iterations": 5,
        "messages": [],
    }

    storage_dir = tmp_path / ".agent_memory"
    save_state(task_id, initial_state, status="running", storage_dir=storage_dir)

    restored_state = load_state(task_id, storage_dir=storage_dir)

    assert restored_state["mode"] == "multi_agent"
    assert restored_state["agent_role"] == "reviewer"
    assert restored_state["review_status"] == "approved"
    assert restored_state["review_feedback"] == "Code clean"
    assert restored_state["multi_agent_iteration"] == 2
    assert restored_state["max_multi_agent_iterations"] == 5

    delete_state(task_id, storage_dir=storage_dir)


def test_orchestrator_routing():
    """Verify route_multi_agent conditional transitions."""
    # Approved -> end
    state_approved: AgentState = {
        "review_status": "approved",
        "multi_agent_iteration": 1,
        "max_multi_agent_iterations": 3,
    }
    assert route_multi_agent(state_approved) == "end"

    # Changes requested -> coder (iteration < max)
    state_changes: AgentState = {
        "review_status": "changes_requested",
        "multi_agent_iteration": 1,
        "max_multi_agent_iterations": 3,
    }
    assert route_multi_agent(state_changes) == "coder"

    # Changes requested -> end (iteration >= max)
    state_max_iters: AgentState = {
        "review_status": "changes_requested",
        "multi_agent_iteration": 3,
        "max_multi_agent_iterations": 3,
    }
    assert route_multi_agent(state_max_iters) == "end"

    # Blocked -> end
    state_blocked: AgentState = {
        "review_status": "blocked",
        "multi_agent_iteration": 1,
        "max_multi_agent_iterations": 3,
    }
    assert route_multi_agent(state_blocked) == "end"


def test_end_to_end_multi_agent_workflow(tmp_workspace: Path, tmp_path: Path):
    """Verify run_multi_agent executes full Analyzer -> Coder -> Reviewer flow successfully."""
    responses = [
        AIMessage(content="Analysis complete: recommend modifying src/main.py to update return string."),
        AIMessage(content="Code modified: src/main.py updated."),
        AIMessage(content="STATUS: APPROVED\nCode review passed cleanly."),
    ]
    mock_llm = MockLLM(responses)

    storage_dir = str(tmp_path / ".agent_memory")
    final_state = run_multi_agent(
        goal="Update hello return value",
        workspace_root=str(tmp_workspace),
        llm=mock_llm,
        storage_dir=storage_dir,
    )

    assert final_state["mode"] == "multi_agent"
    assert final_state["status"] == "completed"
    assert final_state["review_status"] == "approved"
    assert final_state["analysis_result"] is not None
    assert final_state["coding_result"] is not None
    assert final_state["review_result"] is not None


def test_multi_agent_retry_loop(tmp_workspace: Path, tmp_path: Path):
    """Verify self-correcting review retry loop (changes_requested -> coder -> review approved)."""
    responses = [
        AIMessage(content="Analysis: locate test file."),
        AIMessage(content="Coder: initial edit applied."),
        AIMessage(content="STATUS: CHANGES_REQUESTED\nMissing assertion in test file."),
        AIMessage(content="Coder: updated test file with assertion."),
        AIMessage(content="STATUS: APPROVED\nAll issues resolved."),
    ]
    mock_llm = MockLLM(responses)

    storage_dir = str(tmp_path / ".agent_memory")
    final_state = run_multi_agent(
        goal="Fix assertion in tests",
        workspace_root=str(tmp_workspace),
        llm=mock_llm,
        max_iterations=3,
        storage_dir=storage_dir,
    )

    assert final_state["status"] == "completed"
    assert final_state["review_status"] == "approved"
    assert final_state["multi_agent_iteration"] == 2


def test_multi_agent_evaluator(tmp_workspace: Path):
    """Verify MultiAgentEvaluator returns comparison metrics between single-agent and multi-agent modes."""
    evaluator = MultiAgentEvaluator(workspace_root=str(tmp_workspace))

    mock_llm = MockLLM([
        AIMessage(content="Single agent response."),
        AIMessage(content="Analyzer plan."),
        AIMessage(content="Coder edit."),
        AIMessage(content="STATUS: APPROVED\nReview ok."),
    ])

    metrics = evaluator.evaluate_comparison(
        goal="Add docs to main.py",
        llm=mock_llm,
        max_iterations=2,
    )

    assert "single_agent" in metrics
    assert "multi_agent" in metrics
    assert "execution_time_seconds" in metrics["single_agent"]
    assert "execution_time_seconds" in metrics["multi_agent"]
    assert "winner" in metrics


def test_single_agent_backward_compatibility(tmp_workspace: Path):
    """Verify default single-agent mode maintains exact existing Phase 1-11 functionality."""
    mock_llm = MockLLM([
        AIMessage(content="Single agent execution result.")
    ])

    res = run_agent(
        goal="Inspect repo",
        workspace_root=str(tmp_workspace),
        llm=mock_llm,
        mode="single_agent",
    )

    assert res.get("status") in ("completed", "paused", "running")
    assert res.get("messages") is not None
