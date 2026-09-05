"""Deterministic Pytest suite for Phase 5 Autonomous Coding Agent (Safe Code Modification)."""

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
    retrieve_relevant_context,
    write_file,
    replace_in_file,
    create_plan,
    update_task_status,
    revise_plan,
    safe_resolve_path,
    _list_files_impl,
    _read_file_impl,
    _search_code_impl,
    _git_status_impl,
    _git_diff_impl,
    _retrieve_relevant_context_impl,
    _write_file_impl,
    _replace_in_file_impl,
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
# 1. State & Planning Tests (Phases 1-4 Preservation)
# -----------------------------------------------------------------------------
def test_agent_state_initialization():
    """Verify AgentState can hold user goal, workspace root, messages, plan, and modified files."""
    raw_tasks = [{"id": "t1", "title": "Inspect code", "dependencies": []}]
    plan = create_plan_state("Add auth", raw_tasks)

    state: AgentState = {
        "user_goal": "Add auth",
        "workspace_root": "/tmp/test",
        "messages": [HumanMessage(content="Add auth")],
        "plan": plan,
        "retrieved_context": [],
        "modified_files": [],
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
# 2. Retrieval & Read-Only Tests (Phases 1-4 Preservation)
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
    (tmp_path / "app" / "auth.py").write_text("def authenticate_user(u, p):\n    return True\n")
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
    assert "Rank 3:" not in output


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


# -----------------------------------------------------------------------------
# 3. Phase 5 Code Modification Tests
# -----------------------------------------------------------------------------
def test_write_file_create_new(tmp_path: Path):
    """Verify write_file safely creates a new file inside the repository."""
    result = _write_file_impl("new_module.py", "def new_function(): pass\n", workspace_root=str(tmp_path))
    assert "Successfully wrote file 'new_module.py'" in result
    assert (tmp_path / "new_module.py").exists()
    assert (tmp_path / "new_module.py").read_text() == "def new_function(): pass\n"


def test_write_file_modify_existing(tmp_path: Path):
    """Verify write_file safely overwrites an existing file inside the repository."""
    existing_file = tmp_path / "app.py"
    existing_file.write_text("print('old')\n")

    result = _write_file_impl("app.py", "print('new')\n", workspace_root=str(tmp_path))
    assert "Successfully wrote file 'app.py'" in result
    assert existing_file.read_text() == "print('new')\n"


def test_replace_in_file_success(tmp_path: Path):
    """Verify replace_in_file replaces unique target text accurately."""
    target_file = tmp_path / "main.py"
    target_file.write_text("def greet():\n    return 'Hello World'\n")

    result = _replace_in_file_impl(
        "main.py",
        old_text="return 'Hello World'",
        new_text="return 'Hello from autonomous agent'",
        workspace_root=str(tmp_path),
    )

    assert "Successfully replaced target text in 'main.py'" in result
    assert "return 'Hello from autonomous agent'" in target_file.read_text()


def test_replace_in_file_missing_target(tmp_path: Path):
    """Verify replace_in_file fails safely when the target text is not found."""
    target_file = tmp_path / "main.py"
    target_file.write_text("def greet(): pass\n")

    result = _replace_in_file_impl(
        "main.py",
        old_text="return 'Nonexistent String'",
        new_text="return 'New String'",
        workspace_root=str(tmp_path),
    )

    assert "Error: Target text to replace was not found in 'main.py'" in result
    assert target_file.read_text() == "def greet(): pass\n"


def test_replace_in_file_ambiguous_target(tmp_path: Path):
    """Verify replace_in_file fails safely when multiple matches exist."""
    target_file = tmp_path / "config.py"
    target_file.write_text("DEBUG = True\nLOG_LEVEL = True\n")

    result = _replace_in_file_impl(
        "config.py",
        old_text="True",
        new_text="False",
        workspace_root=str(tmp_path),
    )

    assert "Error: Ambiguous replacement target" in result
    assert "Found 2 occurrences" in result
    assert target_file.read_text() == "DEBUG = True\nLOG_LEVEL = True\n"


def test_write_tools_path_traversal_safety(tmp_path: Path):
    """Verify write_file and replace_in_file block path traversal escaping workspace."""
    workspace = tmp_path / "repo"
    workspace.mkdir()

    # Attempt write_file escape
    write_res = _write_file_impl("../outside.py", "malicious", workspace_root=str(workspace))
    assert "Error: Access denied" in write_res
    assert not (tmp_path / "outside.py").exists()

    # Attempt replace_in_file escape
    abs_outside = tmp_path / "outside_file.py"
    abs_outside.write_text("target")
    replace_res = _replace_in_file_impl(str(abs_outside), "target", "new", workspace_root=str(workspace))
    assert "Error: Access denied" in replace_res
    assert abs_outside.read_text() == "target"


def test_git_diff_reflects_modification(tmp_path: Path):
    """Verify git_diff output displays changes created by write_file or replace_in_file."""
    init_git_repo(tmp_path)
    file_path = tmp_path / "app.py"
    file_path.write_text("def hello():\n    return 'Hello'\n")

    subprocess.run(["git", "add", "app.py"], cwd=tmp_path, capture_output=True, check=True)
    subprocess.run(["git", "commit", "-m", "initial"], cwd=tmp_path, capture_output=True, check=True)

    # Modify file via replace_in_file
    _replace_in_file_impl("app.py", "return 'Hello'", "return 'Hello World'", workspace_root=str(tmp_path))

    diff_output = _git_diff_impl(workspace_root=str(tmp_path))
    assert "return 'Hello'" in diff_output
    assert "return 'Hello World'" in diff_output


# -----------------------------------------------------------------------------
# 5. Deterministic Integration Test
# -----------------------------------------------------------------------------
def test_sample_project_greeting_modification_integration(tmp_path: Path):
    """Integration Test: Goal -> Retrieve Context -> Replace Greeting -> Inspect Git Diff -> Complete."""
    init_git_repo(tmp_path)
    app_file = tmp_path / "app.py"
    app_file.write_text("def get_greeting():\n    return 'Hello'\n")

    subprocess.run(["git", "add", "app.py"], cwd=tmp_path, capture_output=True, check=True)
    subprocess.run(["git", "commit", "-m", "v1"], cwd=tmp_path, capture_output=True, check=True)

    mock_responses = [
        # Step 1: Create plan
        AIMessage(
            content="Plan created for greeting modification.",
            tool_calls=[
                {
                    "name": "create_plan",
                    "args": {
                        "tasks": [
                            {"id": "t1", "title": "Locate greeting function", "dependencies": []},
                            {"id": "t2", "title": "Update greeting return string", "dependencies": ["t1"]},
                            {"id": "t3", "title": "Inspect git diff feedback", "dependencies": ["t2"]},
                        ]
                    },
                    "id": "c1",
                }
            ],
        ),
        # Step 2: Retrieve context for t1
        AIMessage(
            content="Retrieving greeting function context.",
            tool_calls=[
                {"name": "update_task_status", "args": {"task_id": "t1", "status": "in_progress"}, "id": "c2_1"},
                {"name": "retrieve_relevant_context", "args": {"query": "get_greeting"}, "id": "c2_2"},
            ],
        ),
        # Step 3: Modify code via replace_in_file for t2
        AIMessage(
            content="Found get_greeting in app.py. Updating greeting string.",
            tool_calls=[
                {"name": "update_task_status", "args": {"task_id": "t1", "status": "completed"}, "id": "c3_1"},
                {"name": "update_task_status", "args": {"task_id": "t2", "status": "in_progress"}, "id": "c3_2"},
                {
                    "name": "replace_in_file",
                    "args": {
                        "file_path": "app.py",
                        "old_text": "return 'Hello'",
                        "new_text": "return 'Hello from the autonomous agent'",
                    },
                    "id": "c3_3",
                },
            ],
        ),
        # Step 4: Call git_diff for t3
        AIMessage(
            content="Modification complete. Inspecting git diff for feedback.",
            tool_calls=[
                {"name": "update_task_status", "args": {"task_id": "t2", "status": "completed"}, "id": "c4_1"},
                {"name": "update_task_status", "args": {"task_id": "t3", "status": "in_progress"}, "id": "c4_2"},
                {"name": "git_diff", "args": {}, "id": "c4_3"},
            ],
        ),
        # Step 5: Final completion evaluation
        AIMessage(
            content="Task t3 complete. Verified via git diff that app.py greeting was updated to 'Hello from the autonomous agent'.",
            tool_calls=[
                {"name": "update_task_status", "args": {"task_id": "t3", "status": "completed"}, "id": "c5_1"}
            ],
        ),
    ]
    mock_llm = MockLLM(responses=mock_responses)

    final_state = run_agent(
        goal="Change the greeting returned by this sample Python project.",
        workspace_root=str(tmp_path),
        llm=mock_llm,
    )

    # 1. Real repository file actually changed
    assert app_file.read_text() == "def get_greeting():\n    return 'Hello from the autonomous agent'\n"

    # 2. State tracks modified file
    modified = final_state.get("modified_files", [])
    assert "app.py" in modified

    # 3. All tasks completed
    plan = final_state.get("plan")
    assert plan is not None
    assert all(t["status"] == "completed" for t in plan["tasks"])
