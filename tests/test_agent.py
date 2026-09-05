"""Deterministic Pytest suite for Phase 1 Autonomous Coding Agent."""

import os
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
    safe_resolve_path,
    _list_files_impl,
    _read_file_impl,
)


class MockLLM(BaseChatModel):
    """Deterministic Mock LLM for testing agent control flow without external API dependencies."""

    responses: List[AIMessage]
    bound_tools: Optional[List[Any]] = None

    def __init__(self, responses: List[AIMessage]):
        super().__init__()

        # Use object.__setattr__ because BaseChatModel is a Pydantic V2 model
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


# -----------------------------------------------------------------------------
# 1. State Tests
# -----------------------------------------------------------------------------
def test_agent_state_initialization():
    """Verify AgentState can hold user goal, workspace root, and messages."""
    state: AgentState = {
        "user_goal": "Inspect code",
        "workspace_root": "/tmp/test",
        "messages": [HumanMessage(content="Inspect code")],
    }
    assert state["user_goal"] == "Inspect code"
    assert state["workspace_root"] == "/tmp/test"
    assert len(state["messages"]) == 1
    assert state["messages"][0].content == "Inspect code"


# -----------------------------------------------------------------------------
# 2. Tool Functionality Tests
# -----------------------------------------------------------------------------
def test_list_files_tool(tmp_path: Path):
    """Verify list_files correctly lists workspace directory contents."""
    (tmp_path / "file1.txt").write_text("Hello")
    (tmp_path / "subdir").mkdir()
    (tmp_path / "subdir" / "file2.txt").write_text("World")

    output = _list_files_impl(".", workspace_root=str(tmp_path))
    assert "file1.txt" in output
    assert "subdir/" in output


def test_read_file_tool(tmp_path: Path):
    """Verify read_file correctly reads text contents of workspace file."""
    test_file = tmp_path / "sample.txt"
    test_file.write_text("Sample file content for testing.")

    content = _read_file_impl("sample.txt", workspace_root=str(tmp_path))
    assert content == "Sample file content for testing."


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

    # Attempt relative path traversal
    rel_result = _read_file_impl("../outside/secret.txt", workspace_root=str(workspace))
    assert "Error: Access denied" in rel_result

    # Attempt absolute path outside workspace
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
    """Verify system does not expose shell/bash/command execution tools."""
    tools = create_workspace_tools(workspace_root=str(tmp_path))
    tool_names = [t.name for t in tools]

    assert "list_files" in tool_names
    assert "read_file" in tool_names

    forbidden = ["shell", "bash", "exec", "terminal", "run_command", "eval", "system"]
    for name in tool_names:
        for f in forbidden:
            assert f not in name.lower()


# -----------------------------------------------------------------------------
# 4. Agent Invocation & Observation Loop Tests
# -----------------------------------------------------------------------------
def test_agent_invokes_tool_and_receives_observation(tmp_path: Path):
    """Verify agent can invoke a tool and receive observation in agent state."""
    (tmp_path / "info.txt").write_text("Repository details inside.")

    mock_responses = [
        AIMessage(
            content="I will list the files in the workspace.",
            tool_calls=[
                {
                    "name": "list_files",
                    "args": {"directory": "."},
                    "id": "call_list_1",
                }
            ],
        ),
        AIMessage(content="I see info.txt in the workspace. Task complete."),
    ]
    mock_llm = MockLLM(responses=mock_responses)

    final_state = run_agent(
        goal="Check files",
        workspace_root=str(tmp_path),
        llm=mock_llm,
    )

    messages = final_state["messages"]
    assert len(messages) == 4  # Human, AI(tool_call), ToolMessage(observation), AI(final)

    assert isinstance(messages[0], HumanMessage)
    assert isinstance(messages[1], AIMessage)
    assert isinstance(messages[2], ToolMessage)
    assert "info.txt" in messages[2].content
    assert isinstance(messages[3], AIMessage)
    assert "Task complete" in messages[3].content


def test_multi_step_agent_loop(tmp_path: Path):
    """Verify agent can perform multiple reasoning and action steps before terminating."""
    (tmp_path / "app.py").write_text("print('Hello Agent')")

    mock_responses = [
        # Step 1: LLM decides to list files
        AIMessage(
            content="First, I need to list the directory.",
            tool_calls=[
                {
                    "name": "list_files",
                    "args": {"directory": "."},
                    "id": "call_step1",
                }
            ],
        ),
        # Step 2: LLM inspects observation and decides to read app.py
        AIMessage(
            content="I see app.py. Now I will read its contents.",
            tool_calls=[
                {
                    "name": "read_file",
                    "args": {"file_path": "app.py"},
                    "id": "call_step2",
                }
            ],
        ),
        # Step 3: LLM receives file contents observation and produces final answer
        AIMessage(
            content="The repository contains app.py which prints 'Hello Agent'."
        ),
    ]
    mock_llm = MockLLM(responses=mock_responses)

    final_state = run_agent(
        goal="Understand repository structure and app.py content",
        workspace_root=str(tmp_path),
        llm=mock_llm,
    )

    messages = final_state["messages"]
    # Messages sequence:
    # 0: User Goal (HumanMessage)
    # 1: AI (list_files)
    # 2: Observation 1 (ToolMessage list_files result)
    # 3: AI (read_file)
    # 4: Observation 2 (ToolMessage read_file content)
    # 5: AI (Final Answer)
    assert len(messages) == 6

    assert isinstance(messages[2], ToolMessage)
    assert "app.py" in messages[2].content

    assert isinstance(messages[4], ToolMessage)
    assert "print('Hello Agent')" in messages[4].content

    assert isinstance(messages[5], AIMessage)
    assert "prints 'Hello Agent'" in messages[5].content


def test_agent_termination_without_tools(tmp_path: Path):
    """Verify agent terminates immediately when LLM provides direct answer without requesting tools."""
    mock_responses = [
        AIMessage(content="I already know the answer. No tools needed.")
    ]
    mock_llm = MockLLM(responses=mock_responses)

    final_state = run_agent(
        goal="Simple question",
        workspace_root=str(tmp_path),
        llm=mock_llm,
    )

    messages = final_state["messages"]
    assert len(messages) == 2  # HumanMessage + AIMessage
    assert messages[1].content == "I already know the answer. No tools needed."
