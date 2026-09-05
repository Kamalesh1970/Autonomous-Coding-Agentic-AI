"""Deterministic Pytest suite for Phase 3 Autonomous Coding Agent (Planning & Decomposition)."""

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
    safe_resolve_path,
    _list_files_impl,
    _read_file_impl,
    _search_code_impl,
    _git_status_impl,
    _git_diff_impl,
)


class MockLLM(BaseChatModel):
    """Deterministic Mock LLM for testing agent control flow without external API dependencies."""

    responses: List[AIMessage]
    bound_tools: Optional[List[Any]] = None

    def __init__(self, responses: List[AIMessage]):
        super().__init__()
        object.__setattr__(self, "responses", list(responses))
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
# 1. State & Planning Data Structure Tests
# -----------------------------------------------------------------------------
def test_plan_state_representation():
    """Verify AgentState can hold user goal, workspace root, messages, and structured plan."""
    raw_tasks = [
        {"id": "t1", "title": "Inspect code", "description": "Check auth", "dependencies": []},
        {"id": "t2", "title": "Implement auth", "description": "Add JWT", "dependencies": ["t1"]},
    ]
    plan = create_plan_state("Add authentication", raw_tasks)

    state: AgentState = {
        "user_goal": "Add authentication",
        "workspace_root": "/tmp/test",
        "messages": [HumanMessage(content="Add authentication")],
        "plan": plan,
    }

    assert state["user_goal"] == "Add authentication"
    assert state["plan"]["goal"] == "Add authentication"
    assert len(state["plan"]["tasks"]) == 2
    assert state["plan"]["tasks"][0]["id"] == "t1"
    assert state["plan"]["tasks"][1]["dependencies"] == ["t1"]


def test_task_statuses_and_transitions():
    """Verify tasks support pending, in_progress, completed, failed, and blocked states."""
    raw_tasks = [
        {"id": "t1", "title": "Task 1", "status": "pending", "dependencies": []},
        {"id": "t2", "title": "Task 2", "status": "pending", "dependencies": ["t1"]},
    ]
    plan = create_plan_state("Test Goal", raw_tasks)
    assert plan["current_task_id"] == "t1"

    # Transition t1 -> in_progress
    plan = update_task_state(plan, "t1", "in_progress")
    assert plan["tasks"][0]["status"] == "in_progress"
    assert plan["current_task_id"] == "t1"

    # Transition t1 -> completed
    plan = update_task_state(plan, "t1", "completed")
    assert plan["tasks"][0]["status"] == "completed"
    assert plan["current_task_id"] == "t2"  # Now t2 is unblocked!

    # Transition t2 -> failed
    plan = update_task_state(plan, "t2", "failed")
    assert plan["tasks"][1]["status"] == "failed"

    # Transition t2 -> blocked
    plan = update_task_state(plan, "t2", "blocked")
    assert plan["tasks"][1]["status"] == "blocked"


def test_dependency_blocking():
    """Verify tasks are not selected if their dependencies are incomplete."""
    raw_tasks = [
        {"id": "t1", "title": "Task 1", "status": "pending", "dependencies": []},
        {"id": "t2", "title": "Task 2", "status": "pending", "dependencies": ["t1"]},
    ]
    plan = create_plan_state("Test", raw_tasks)

    # t2 depends on t1, so next available task must be t1
    assert get_next_available_task_id(plan["tasks"]) == "t1"

    # Mark t1 as failed (not completed), t2 remains blocked
    plan = update_task_state(plan, "t1", "failed")
    assert get_next_available_task_id(plan["tasks"]) is None


def test_plan_revision_state():
    """Verify plan revision updates task list and preserves revision history."""
    initial_tasks = [{"id": "t1", "title": "Initial assumption", "dependencies": []}]
    plan = create_plan_state("Goal", initial_tasks)
    assert plan["revision_count"] == 0

    revised_tasks = [
        {"id": "rt1", "title": "Revised task 1", "dependencies": []},
        {"id": "rt2", "title": "Revised task 2", "dependencies": ["rt1"]},
    ]
    revised = revise_plan_state(plan, revised_tasks, reason="Found OAuth already implemented")

    assert revised["revision_count"] == 1
    assert revised["revision_reason"] == "Found OAuth already implemented"
    assert len(revised["tasks"]) == 2
    assert revised["tasks"][0]["id"] == "rt1"


# -----------------------------------------------------------------------------
# 2. Phase 2 Tools Preservation Tests
# -----------------------------------------------------------------------------
def test_list_files_recursive(tmp_path: Path):
    """Verify list_files correctly lists repository files recursively while skipping .git."""
    (tmp_path / "app").mkdir()
    (tmp_path / "app" / "main.py").write_text("print('main')")

    output = _list_files_impl(".", workspace_root=str(tmp_path))
    assert "app/main.py" in output


def test_read_file_tool(tmp_path: Path):
    """Verify read_file correctly reads text contents of workspace file."""
    test_file = tmp_path / "sample.txt"
    test_file.write_text("Sample file content.")
    content = _read_file_impl("sample.txt", workspace_root=str(tmp_path))
    assert content == "Sample file content."


def test_search_code_tool(tmp_path: Path):
    """Verify search_code finds matching query lines."""
    (tmp_path / "auth.py").write_text("def authenticate_user(): pass\n")
    output = _search_code_impl("authenticate_user", workspace_root=str(tmp_path))
    assert "auth.py:1: def authenticate_user(): pass" in output


def test_git_status_tool(tmp_path: Path):
    """Verify git_status inspects repository branch and file modifications."""
    init_git_repo(tmp_path)
    (tmp_path / "new.py").write_text("# new")
    status = _git_status_impl(workspace_root=str(tmp_path))
    assert "new.py" in status or "??" in status


def test_git_diff_tool(tmp_path: Path):
    """Verify git_diff displays unstaged changes."""
    init_git_repo(tmp_path)
    file = tmp_path / "c.py"
    file.write_text("v1")
    subprocess.run(["git", "add", "c.py"], cwd=tmp_path, capture_output=True, check=True)
    subprocess.run(["git", "commit", "-m", "v1"], cwd=tmp_path, capture_output=True, check=True)
    file.write_text("v2")

    diff_output = _git_diff_impl(workspace_root=str(tmp_path))
    assert "c.py" in diff_output or "v1" in diff_output or "v2" in diff_output


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
    """Verify tool layer is strictly read-only + planning tools."""
    tools = create_workspace_tools(workspace_root=str(tmp_path))
    tool_names = set(t.name for t in tools)

    expected_tools = {
        "list_files",
        "read_file",
        "search_code",
        "git_status",
        "git_diff",
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
# 4. Agentic Planning & Revision Control Loop Tests
# -----------------------------------------------------------------------------
def test_deterministic_planning_scenario(tmp_path: Path):
    """Verify full agent loop: Goal -> Inspect -> Create Plan -> Task Execution -> Plan Revision -> Complete."""
    init_git_repo(tmp_path)
    (tmp_path / "routes.py").write_text("# Existing OAuth authentication present\ndef oauth_login(): pass\n")

    mock_responses = [
        # Step 1: Agent inspects repo
        AIMessage(
            content="Searching codebase for existing authentication.",
            tool_calls=[{"name": "search_code", "args": {"query": "auth"}, "id": "c1"}],
        ),
        # Step 2: Agent observes OAuth in routes.py, initializes plan
        AIMessage(
            content="Found existing OAuth in routes.py. Initializing plan.",
            tool_calls=[
                {
                    "name": "create_plan",
                    "args": {
                        "tasks": [
                            {"id": "t1", "title": "Inspect routes.py OAuth", "dependencies": []},
                            {"id": "t2", "title": "Add JWT support", "dependencies": ["t1"]},
                        ]
                    },
                    "id": "c2",
                }
            ],
        ),
        # Step 3: Agent marks t1 in_progress and reads routes.py
        AIMessage(
            content="Reading routes.py to complete t1.",
            tool_calls=[
                {"name": "update_task_status", "args": {"task_id": "t1", "status": "in_progress"}, "id": "c3_1"},
                {"name": "read_file", "args": {"file_path": "routes.py"}, "id": "c3_2"},
            ],
        ),
        # Step 4: Agent marks t1 completed and revises plan for JWT integration
        AIMessage(
            content="t1 complete. Revising plan based on OAuth observation.",
            tool_calls=[
                {"name": "update_task_status", "args": {"task_id": "t1", "status": "completed"}, "id": "c4_1"},
                {
                    "name": "revise_plan",
                    "args": {
                        "new_tasks": [
                            {"id": "t1", "title": "Inspect routes.py OAuth", "status": "completed", "dependencies": []},
                            {"id": "t2_rev", "title": "Integrate JWT bearer token into OAuth routes", "status": "pending", "dependencies": ["t1"]},
                        ],
                        "reason": "OAuth structure requires bearer token integration strategy",
                    },
                    "id": "c4_2",
                },
            ],
        ),
        # Step 5: Final completion
        AIMessage(
            content="Planning complete. Authentication architecture analyzed and plan revised."
        ),
    ]
    mock_llm = MockLLM(responses=mock_responses)

    final_state = run_agent(
        goal="Understand and improve authentication in this repository.",
        workspace_root=str(tmp_path),
        llm=mock_llm,
    )

    plan = final_state.get("plan")
    assert plan is not None
    assert plan["revision_count"] == 1
    assert "OAuth structure requires bearer token integration strategy" in plan["revision_reason"]
    assert len(plan["tasks"]) == 2
    assert plan["tasks"][0]["status"] == "completed"
    assert plan["tasks"][1]["id"] == "t2_rev"
