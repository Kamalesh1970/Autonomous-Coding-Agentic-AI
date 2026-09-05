"""Deterministic Pytest suite for Phase 2 Autonomous Coding Agent."""

import os
import subprocess
from pathlib import Path
from typing import Any, List, Optional, Sequence

import pytest
from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import AIMessage, HumanMessage, ToolMessage, BaseMessage
from langchain_core.outputs import ChatGeneration, ChatResult

from app.agent import build_agent_graph, run_agent
from app.state import AgentState
from app.tools import (
    create_workspace_tools,
    list_files,
    read_file,
    search_code,
    git_status,
    git_diff,
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
# 1. State Tests
# -----------------------------------------------------------------------------
def test_agent_state_initialization():
    """Verify AgentState can hold user goal, workspace root, and messages."""
    state: AgentState = {
        "user_goal": "Inspect repository",
        "workspace_root": "/tmp/test",
        "messages": [HumanMessage(content="Inspect repository")],
    }
    assert state["user_goal"] == "Inspect repository"
    assert state["workspace_root"] == "/tmp/test"
    assert len(state["messages"]) == 1
    assert state["messages"][0].content == "Inspect repository"


# -----------------------------------------------------------------------------
# 2. Tool Functionality Tests (Phase 1 & Phase 2)
# -----------------------------------------------------------------------------
def test_list_files_recursive(tmp_path: Path):
    """Verify list_files correctly lists repository files recursively while skipping .git."""
    (tmp_path / "app").mkdir()
    (tmp_path / "app" / "main.py").write_text("print('main')")
    (tmp_path / "tests").mkdir()
    (tmp_path / "tests" / "test_main.py").write_text("pass")
    
    # Create git dir which should be ignored
    (tmp_path / ".git").mkdir()
    (tmp_path / ".git" / "config").write_text("git config")

    output = _list_files_impl(".", workspace_root=str(tmp_path))
    assert "app/main.py" in output
    assert "tests/test_main.py" in output
    assert ".git" not in output


def test_read_file_tool(tmp_path: Path):
    """Verify read_file correctly reads text contents of workspace file."""
    test_file = tmp_path / "sample.txt"
    test_file.write_text("Sample file content for testing.")

    content = _read_file_impl("sample.txt", workspace_root=str(tmp_path))
    assert content == "Sample file content for testing."


def test_read_file_binary_detection(tmp_path: Path):
    """Verify read_file rejects binary files safely."""
    bin_file = tmp_path / "image.png"
    bin_file.write_bytes(b"\x89PNG\r\n\x1a\n\x00\x00\x00\rIHDR")

    result = _read_file_impl("image.png", workspace_root=str(tmp_path))
    assert "appears to be binary" in result


def test_read_file_truncation(tmp_path: Path):
    """Verify read_file truncates overly large text files."""
    large_file = tmp_path / "large.txt"
    large_file.write_text("A" * 200)

    result = _read_file_impl("large.txt", workspace_root=str(tmp_path), max_bytes=50)
    assert len(result) < 200
    assert "truncated file content at 50 bytes" in result


def test_search_code_tool(tmp_path: Path):
    """Verify search_code finds matching query lines across repository text files."""
    (tmp_path / "auth.py").write_text("def authenticate_user():\n    return True\n")
    (tmp_path / "main.py").write_text("import auth\nauth.authenticate_user()\n")

    output = _search_code_impl("authenticate_user", workspace_root=str(tmp_path))
    assert "auth.py:1: def authenticate_user():" in output
    assert "main.py:2: auth.authenticate_user()" in output


def test_search_code_bounded(tmp_path: Path):
    """Verify search_code output is bounded when matches exceed max limit."""
    lines = ["target_keyword_here\n" for _ in range(50)]
    (tmp_path / "repeat.txt").write_text("".join(lines))

    output = _search_code_impl("target_keyword", workspace_root=str(tmp_path), max_matches=10)
    assert "truncated at 10 matches" in output


def test_git_status_tool(tmp_path: Path):
    """Verify git_status inspects repository branch and file modifications."""
    init_git_repo(tmp_path)
    (tmp_path / "untracked.py").write_text("# new file")

    status = _git_status_impl(workspace_root=str(tmp_path))
    assert "untracked.py" in status or "??" in status


def test_git_diff_tool(tmp_path: Path):
    """Verify git_diff displays unstaged and staged changes in test git repo."""
    init_git_repo(tmp_path)
    tracked_file = tmp_path / "code.py"
    tracked_file.write_text("version 1")

    subprocess.run(["git", "add", "code.py"], cwd=tmp_path, capture_output=True, check=True)
    subprocess.run(["git", "commit", "-m", "v1"], cwd=tmp_path, capture_output=True, check=True)

    tracked_file.write_text("version 2")

    diff_output = _git_diff_impl(workspace_root=str(tmp_path))
    assert "version 1" in diff_output or "version 2" in diff_output or "code.py" in diff_output


# -----------------------------------------------------------------------------
# 3. Security & Safety Tests
# -----------------------------------------------------------------------------
def test_path_traversal_safety(tmp_path: Path):
    """Verify attempts to escape workspace directory via ../ or absolute path are safely denied."""
    secret_dir = tmp_path / "outside"
    secret_dir.mkdir()
    secret_file = secret_dir / "secret.txt"
    secret_file.write_text("TOP SECRET")

    workspace = tmp_path / "workspace"
    workspace.mkdir()

    # Relative path traversal
    rel_result = _read_file_impl("../outside/secret.txt", workspace_root=str(workspace))
    assert "Error: Access denied" in rel_result

    # Absolute path outside workspace
    abs_result = _read_file_impl(str(secret_file), workspace_root=str(workspace))
    assert "Error: Access denied" in abs_result

    # Safe path validation directly
    with pytest.raises(ValueError, match="escapes workspace"):
        safe_resolve_path(workspace, "../outside/secret.txt")


def test_invalid_file_reading(tmp_path: Path):
    """Verify read_file safely handles missing files and directories."""
    missing_res = _read_file_impl("non_existent.txt", workspace_root=str(tmp_path))
    assert "Error: File 'non_existent.txt' does not exist" in missing_res

    dir_res = _read_file_impl(".", workspace_root=str(tmp_path))
    assert "is a directory" in dir_res


def test_no_shell_execution_tools(tmp_path: Path):
    """Verify system does not expose arbitrary shell/bash execution tools."""
    tools = create_workspace_tools(workspace_root=str(tmp_path))
    tool_names = [t.name for t in tools]

    forbidden = ["shell", "bash", "exec", "terminal", "run_command", "eval", "system"]
    for name in tool_names:
        for f in forbidden:
            assert f not in name.lower()


def test_no_write_edit_delete_tools(tmp_path: Path):
    """Verify tool layer is strictly read-only and does not expose write/edit/delete tools."""
    tools = create_workspace_tools(workspace_root=str(tmp_path))
    tool_names = set(t.name for t in tools)

    expected_read_only = {"list_files", "read_file", "search_code", "git_status", "git_diff"}
    assert tool_names == expected_read_only

    forbidden_actions = ["write", "edit", "delete", "remove", "patch", "commit", "push", "checkout"]
    for name in tool_names:
        for action in forbidden_actions:
            assert action not in name.lower()


# -----------------------------------------------------------------------------
# 4. Multi-Step Repository Understanding Agent Loop Tests
# -----------------------------------------------------------------------------
def test_multi_step_repo_investigation(tmp_path: Path):
    """Verify agent can perform sequential repository tool calls and reason over observations."""
    init_git_repo(tmp_path)
    (tmp_path / "auth.py").write_text("def login_user(user, pwd):\n    return True\n")
    
    mock_responses = [
        # Step 1: Agent searches code for login_user
        AIMessage(
            content="I will search the codebase for login_user.",
            tool_calls=[
                {
                    "name": "search_code",
                    "args": {"query": "login_user"},
                    "id": "call_search_1",
                }
            ],
        ),
        # Step 2: Agent reads auth.py based on search observation
        AIMessage(
            content="Found match in auth.py. Reading auth.py.",
            tool_calls=[
                {
                    "name": "read_file",
                    "args": {"file_path": "auth.py"},
                    "id": "call_read_1",
                }
            ],
        ),
        # Step 3: Agent checks git status
        AIMessage(
            content="Inspecting repository git status.",
            tool_calls=[
                {
                    "name": "git_status",
                    "args": {},
                    "id": "call_git_1",
                }
            ],
        ),
        # Step 4: Final response
        AIMessage(
            content="Authentication is implemented in auth.py via login_user function."
        ),
    ]
    mock_llm = MockLLM(responses=mock_responses)

    final_state = run_agent(
        goal="Understand authentication implementation",
        workspace_root=str(tmp_path),
        llm=mock_llm,
    )

    messages = final_state["messages"]
    # Messages sequence:
    # 0: User Goal (HumanMessage)
    # 1: AI (search_code)
    # 2: ToolMessage (search observation)
    # 3: AI (read_file)
    # 4: ToolMessage (file content observation)
    # 5: AI (git_status)
    # 6: ToolMessage (git status observation)
    # 7: AI (Final response)
    assert len(messages) == 8

    assert isinstance(messages[2], ToolMessage)
    assert "auth.py:1: def login_user" in messages[2].content

    assert isinstance(messages[4], ToolMessage)
    assert "def login_user(user, pwd):" in messages[4].content

    assert isinstance(messages[6], ToolMessage)

    assert isinstance(messages[7], AIMessage)
    assert "Authentication is implemented in auth.py" in messages[7].content


def test_tool_observations_influence_decisions(tmp_path: Path):
    """Verify tool observation output directly influences subsequent agent choices."""
    (tmp_path / "config.txt").write_text("FEATURE_FLAG_AUTH=enabled")

    mock_responses = [
        AIMessage(
            content="Reading config.txt first.",
            tool_calls=[
                {
                    "name": "read_file",
                    "args": {"file_path": "config.txt"},
                    "id": "call_c1",
                }
            ],
        ),
        AIMessage(
            content="Based on config observation (FEATURE_FLAG_AUTH=enabled), authentication is enabled."
        ),
    ]
    mock_llm = MockLLM(responses=mock_responses)

    final_state = run_agent(
        goal="Check if auth feature is enabled",
        workspace_root=str(tmp_path),
        llm=mock_llm,
    )

    messages = final_state["messages"]
    assert len(messages) == 4
    assert "FEATURE_FLAG_AUTH=enabled" in messages[2].content
    assert "authentication is enabled" in messages[3].content
