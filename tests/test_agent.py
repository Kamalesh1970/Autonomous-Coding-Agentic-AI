"""Deterministic Pytest suite for Phase 4 Autonomous Coding Agent (Structured Retrieval & Ranking)."""

import os
import subprocess
from pathlib import Path
from typing import Any, List, Optional, Sequence

import pytest
from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import AIMessage, HumanMessage, ToolMessage, BaseMessage
from langchain_core.outputs import ChatGeneration, ChatResult

from app.agent import build_agent_graph, run_agent, sync_plan_from_messages
from app.state import (
    AgentState,
    ExecutionPlan,
    Task,
    create_plan_state,
    get_next_available_task_id,
    update_task_state,
    revise_plan_state,
)
from app.tools import (
    create_workspace_tools,
    list_files,
    read_file,
    search_code,
    git_status,
    git_diff,
    create_plan,
    update_task_status,
    revise_plan,
    retrieve_relevant_context,
    safe_resolve_path,
    _list_files_impl,
    _read_file_impl,
    _search_code_impl,
    _git_status_impl,
    _git_diff_impl,
    _retrieve_relevant_context_impl,
)


class MockLLM(BaseChatModel):
    """Deterministic Mock LLM for testing agent control flow without external API dependencies."""

    responses: List[AIMessage]
    bound_tools: Optional[List[Any]] = None

    def __init__(self, responses: List[AIMessage]):
        super().__init__(responses=list(responses))
        object.__setattr__(self, "bound_tools", None)

    @property
    def _llm_type(self) -> str:
        return "mock_chat_model"

    def _generate(
        self,
        messages: List[BaseMessage],
        stop: Optional[List[str]] = None,
        run_manager: Optional[Any] = None,
        **kwargs: Any,
    ) -> ChatResult:
        if not self.responses:
            response = AIMessage(content="Finished with no further actions.")
        else:
            response = self.responses.pop(0)

        return ChatResult(generations=[ChatGeneration(message=response)])

    def bind_tools(self, tools: Sequence[Any], **kwargs: Any) -> "MockLLM":
        mock_copy = MockLLM(responses=self.responses)
        object.__setattr__(mock_copy, "bound_tools", list(tools))
        return mock_copy


def init_git_repo(repo_path: Path):
    """Initialize a temporary git repository for testing."""
    subprocess.run(["git", "init"], cwd=repo_path, capture_output=True, check=True)
    subprocess.run(["git", "config", "user.name", "Test User"], cwd=repo_path, capture_output=True, check=True)
    subprocess.run(["git", "config", "user.email", "test@example.com"], cwd=repo_path, capture_output=True, check=True)


# -----------------------------------------------------------------------------
# 1. State & Planning Tests (Phases 1-3 Preservation)
# -----------------------------------------------------------------------------
def test_agent_state_initialization():
    """Verify AgentState can hold user goal, workspace root, messages, and plan."""
    raw_tasks = [{"id": "t1", "title": "Inspect code", "dependencies": []}]
    plan = create_plan_state("Add auth", raw_tasks)

    state: AgentState = {
        "user_goal": "Add auth",
        "workspace_root": "/tmp/test",
        "messages": [HumanMessage(content="Add auth")],
        "plan": plan,
        "retrieved_context": [],
    }

    assert state["user_goal"] == "Add auth"
    assert state["plan"]["tasks"][0]["id"] == "t1"


def test_task_statuses_and_transitions():
    """Verify tasks support pending, in_progress, completed, failed, and blocked states."""
    raw_tasks = [
        {"id": "t1", "title": "Task 1", "status": "pending", "dependencies": []},
        {"id": "t2", "title": "Task 2", "status": "pending", "dependencies": ["t1"]},
    ]
    plan = create_plan_state("Test Goal", raw_tasks)

    plan = update_task_state(plan, "t1", "in_progress")
    assert plan["tasks"][0]["status"] == "in_progress"

    plan = update_task_state(plan, "t1", "completed")
    assert plan["tasks"][0]["status"] == "completed"
    assert plan["current_task_id"] == "t2"


# -----------------------------------------------------------------------------
# 2. Phase 4 Retrieval & Ranking Tests
# -----------------------------------------------------------------------------
def test_retrieve_relevant_context_tool(tmp_path: Path):
    """Verify retrieve_relevant_context finds matching code context for a query."""
    (tmp_path / "auth.py").write_text("def authenticate_user():\n    return True\n")
    output = _retrieve_relevant_context_impl("authenticate_user", workspace_root=str(tmp_path))

    assert "Rank 1: auth.py" in output
    assert "def authenticate_user():" in output


def test_relevance_ranking_priority(tmp_path: Path):
    """Verify files with path matches, symbol definitions, and content matches rank higher."""
    (tmp_path / "app").mkdir()
    # High match: filename + AST symbol + content
    (tmp_path / "app" / "auth.py").write_text("def authenticate_user(u, p):\n    return True\n")
    # Low match: content line only
    (tmp_path / "app" / "notes.txt").write_text("Mention auth in docs\n")

    output = _retrieve_relevant_context_impl("auth", workspace_root=str(tmp_path))

    assert "Rank 1: app/auth.py" in output
    assert "Rank 2: app/notes.txt" in output


def test_surrounding_code_context_extraction(tmp_path: Path):
    """Verify retrieval extracts surrounding line window blocks with line numbers."""
    lines = [f"# Line {i}\n" for i in range(1, 30)]
    lines[14] = "SECRET_KEY = 'jwt_secret_token_key'\n"
    (tmp_path / "config.py").write_text("".join(lines))

    output = _retrieve_relevant_context_impl("jwt_secret_token_key", workspace_root=str(tmp_path))

    assert "config.py" in output
    assert "Lines 5–25:" in output or "Lines 10–20:" in output or "15:" in output
    assert "15: SECRET_KEY = 'jwt_secret_token_key'" in output


def test_retrieval_output_bounded(tmp_path: Path):
    """Verify retrieval output is bounded in total characters and files returned."""
    for i in range(10):
        (tmp_path / f"module_{i}.py").write_text(f"def common_function_{i}():\n    return '{'A'*500}'\n")

    output = _retrieve_relevant_context_impl(
        "common_function", workspace_root=str(tmp_path), max_files=2, max_total_chars=1000
    )

    assert "Rank 1:" in output
    assert "Rank 2:" in output
    assert "Rank 3:" not in output  # Bounded to max 2 files!
    assert "truncated at 1000 characters" in output or len(output) <= 1200


def test_task_driven_retrieval(tmp_path: Path):
    """Verify different task queries retrieve different relevant repository contexts."""
    (tmp_path / "auth.py").write_text("def login_user(): pass\n")
    (tmp_path / "db.py").write_text("def connect_database(): pass\n")

    auth_res = _retrieve_relevant_context_impl("login_user", workspace_root=str(tmp_path))
    assert "auth.py" in auth_res
    assert "db.py" not in auth_res

    db_res = _retrieve_relevant_context_impl("connect_database", workspace_root=str(tmp_path))
    assert "db.py" in db_res
    assert "auth.py" not in db_res


def test_retrieval_path_traversal_safety(tmp_path: Path):
    """Verify retrieve_relevant_context blocks directory traversal attempts."""
    workspace = tmp_path / "workspace"
    workspace.mkdir()

    output = _retrieve_relevant_context_impl("auth", directory="../", workspace_root=str(workspace))
    assert "Error: Access denied" in output


# -----------------------------------------------------------------------------
# 3. Read-Only Security & Tool Audit Tests
# -----------------------------------------------------------------------------
def test_path_traversal_safety(tmp_path: Path):
    """Verify attempts to escape workspace directory via ../ are safely denied."""
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    rel_result = _read_file_impl("../secret.txt", workspace_root=str(workspace))
    assert "Error: Access denied" in rel_result


def test_no_shell_execution_tools(tmp_path: Path):
    """Verify system does not expose arbitrary shell/bash execution tools."""
    tools = create_workspace_tools(workspace_root=str(tmp_path))
    tool_names = [t.name for t in tools]
    forbidden = ["shell", "bash", "exec", "terminal", "run_command", "eval", "system"]
    for name in tool_names:
        for f in forbidden:
            assert f not in name.lower()


def test_no_write_edit_delete_tools(tmp_path: Path):
    """Verify tool layer is strictly read-only + retrieval + planning tools."""
    tools = create_workspace_tools(workspace_root=str(tmp_path))
    tool_names = set(t.name for t in tools)

    expected_tools = {
        "list_files",
        "read_file",
        "search_code",
        "git_status",
        "git_diff",
        "retrieve_relevant_context",
        "create_plan",
        "update_task_status",
        "revise_plan",
    }
    assert tool_names == expected_tools

    forbidden_actions = ["write", "edit", "delete", "remove", "patch", "commit", "push", "checkout"]
    for name in tool_names:
        for action in forbidden_actions:
            assert action not in name.lower()


# -----------------------------------------------------------------------------
# 4. Multi-Step Retrieval & Planning Agent Loop Tests
# -----------------------------------------------------------------------------
def test_multi_step_retrieval_agent_loop(tmp_path: Path):
    """Verify agent loop performs contextual retrieval -> deeper file read -> plan update -> complete."""
    init_git_repo(tmp_path)
    (tmp_path / "app").mkdir()
    (tmp_path / "app" / "auth.py").write_text("def jwt_authenticate(token):\n    # Verify JWT signature\n    return True\n")

    mock_responses = [
        # Step 1: Agent creates plan
        AIMessage(
            content="Creating plan for JWT authentication analysis.",
            tool_calls=[
                {
                    "name": "create_plan",
                    "args": {
                        "tasks": [
                            {"id": "t1", "title": "Retrieve JWT auth context", "dependencies": []},
                            {"id": "t2", "title": "Inspect token verification", "dependencies": ["t1"]},
                        ]
                    },
                    "id": "c1",
                }
            ],
        ),
        # Step 2: Agent retrieves context for task t1
        AIMessage(
            content="Retrieving relevant context for JWT authentication.",
            tool_calls=[
                {"name": "update_task_status", "args": {"task_id": "t1", "status": "in_progress"}, "id": "c2_1"},
                {"name": "retrieve_relevant_context", "args": {"query": "jwt_authenticate"}, "id": "c2_2"},
            ],
        ),
        # Step 3: Agent completes t1 and reads auth.py for task t2
        AIMessage(
            content="Retrieved context found app/auth.py. Now inspecting app/auth.py for t2.",
            tool_calls=[
                {"name": "update_task_status", "args": {"task_id": "t1", "status": "completed"}, "id": "c3_1"},
                {"name": "update_task_status", "args": {"task_id": "t2", "status": "in_progress"}, "id": "c3_2"},
                {"name": "read_file", "args": {"file_path": "app/auth.py"}, "id": "c3_3"},
            ],
        ),
        # Step 4: Agent completes t2 and returns answer
        AIMessage(
            content="Task t2 complete. JWT authentication is implemented in app/auth.py via jwt_authenticate.",
            tool_calls=[
                {"name": "update_task_status", "args": {"task_id": "t2", "status": "completed"}, "id": "c4_1"}
            ],
        ),
    ]
    mock_llm = MockLLM(responses=mock_responses)

    final_state = run_agent(
        goal="Understand JWT authentication architecture",
        workspace_root=str(tmp_path),
        llm=mock_llm,
    )

    plan = final_state.get("plan")
    assert plan is not None
    assert len(plan["tasks"]) == 2
    assert plan["tasks"][0]["status"] == "completed"
    assert plan["tasks"][1]["status"] == "completed"

    retrieved = final_state.get("retrieved_context", [])
    assert len(retrieved) > 0
    assert retrieved[0]["query"] == "jwt_authenticate"
