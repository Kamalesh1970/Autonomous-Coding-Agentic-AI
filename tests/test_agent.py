"""Deterministic Pytest suite for Phase 6 Autonomous Coding Agent (Testing, Recovery & Retry)."""

import os
import subprocess
from pathlib import Path
from typing import Any, List, Optional, Sequence

import pytest
from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import AIMessage, HumanMessage, ToolMessage, BaseMessage
from langchain_core.outputs import ChatGeneration, ChatResult

from app.agent import build_agent_graph, run_agent, resume_agent, sync_plan_from_messages
from app.memory import save_state, load_state, delete_state, safe_resolve_task_memory_path
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
    run_tests,
    create_plan,
    update_task_status,
    revise_plan,
    verify_goal,
    safe_resolve_path,
    _list_files_impl,
    _read_file_impl,
    _search_code_impl,
    _git_status_impl,
    _git_diff_impl,
    _retrieve_relevant_context_impl,
    _write_file_impl,
    _replace_in_file_impl,
    _run_tests_impl,
    _verify_goal_impl,
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
# 1. State & Planning Tests (Phases 1-5 Preservation)
# -----------------------------------------------------------------------------
def test_agent_state_initialization():
    """Verify AgentState can hold user goal, workspace root, messages, plan, and validation result."""
    raw_tasks = [{"id": "t1", "title": "Inspect code", "dependencies": []}]
    plan = create_plan_state("Add auth", raw_tasks)

    state: AgentState = {
        "user_goal": "Add auth",
        "workspace_root": "/tmp/test",
        "messages": [HumanMessage(content="Add auth")],
        "plan": plan,
        "retrieved_context": [],
        "modified_files": [],
        "validation_result": None,
        "retry_count": 0,
        "max_retries": 3,
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
# 2. Read-Only, Retrieval & Edit Tests (Phases 1-5 Preservation)
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

    db_res = _retrieve_relevant_context_impl("connect_database", workspace_root=str(tmp_path))
    assert "db.py" in db_res


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


def test_write_file_create_new(tmp_path: Path):
    """Verify write_file safely creates a new file inside the repository."""
    result = _write_file_impl("new_module.py", "def new_function(): pass\n", workspace_root=str(tmp_path))
    assert "Successfully wrote file 'new_module.py'" in result
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


def test_write_tools_path_traversal_safety(tmp_path: Path):
    """Verify write_file and replace_in_file block path traversal escaping workspace."""
    workspace = tmp_path / "repo"
    workspace.mkdir()

    write_res = _write_file_impl("../outside.py", "malicious", workspace_root=str(workspace))
    assert "Error: Access denied" in write_res


def test_git_diff_reflects_modification(tmp_path: Path):
    """Verify git_diff output displays changes created by replace_in_file."""
    init_git_repo(tmp_path)
    file_path = tmp_path / "app.py"
    file_path.write_text("def hello():\n    return 'Hello'\n")

    subprocess.run(["git", "add", "app.py"], cwd=tmp_path, capture_output=True, check=True)
    subprocess.run(["git", "commit", "-m", "initial"], cwd=tmp_path, capture_output=True, check=True)

    _replace_in_file_impl("app.py", "return 'Hello'", "return 'Hello World'", workspace_root=str(tmp_path))

    diff_output = _git_diff_impl(workspace_root=str(tmp_path))
    assert "return 'Hello'" in diff_output
    assert "return 'Hello World'" in diff_output


# -----------------------------------------------------------------------------
# 3. Phase 6 Autonomous Testing & Validation Tool Tests
# -----------------------------------------------------------------------------
def test_run_tests_success(tmp_path: Path):
    """Verify run_tests returns passed status when temporary repository tests pass."""
    (tmp_path / "test_sample.py").write_text("def test_pass():\n    assert 1 + 1 == 2\n")

    output = _run_tests_impl(".", workspace_root=str(tmp_path))

    assert "Status: passed" in output
    assert "Exit Code: 0" in output
    assert "All tests passed successfully" in output


def test_run_tests_failure(tmp_path: Path):
    """Verify run_tests returns failed status and traceback when test fails."""
    (tmp_path / "test_sample.py").write_text("def test_fail():\n    assert 1 + 1 == 99\n")

    output = _run_tests_impl(".", workspace_root=str(tmp_path))

    assert "Status: failed" in output
    assert "Exit Code: 1" in output
    assert "AssertionError" in output or "assert 1 + 1 == 99" in output


def test_run_tests_timeout(tmp_path: Path):
    """Verify run_tests catches process timeouts and returns timeout observation."""
    (tmp_path / "test_sleep.py").write_text("import time\ndef test_slow():\n    time.sleep(10)\n")

    output = _run_tests_impl(".", workspace_root=str(tmp_path), timeout_seconds=1)

    assert "Status: timeout" in output
    assert "TimeoutExpired" in output


def test_run_tests_bounded_output(tmp_path: Path):
    """Verify run_tests bounds test output to max_output_chars."""
    (tmp_path / "test_verbose.py").write_text(
        "def test_verbose():\n    for i in range(100):\n        print(f'Verbose log line {i} ' * 10)\n    assert False\n"
    )

    output = _run_tests_impl(".", workspace_root=str(tmp_path), max_output_chars=500)

    assert "Status: failed" in output
    assert "truncated at 500 characters" in output
    assert len(output) <= 800


# -----------------------------------------------------------------------------
# 4. Agent Self-Correction & Integration Tests
# -----------------------------------------------------------------------------
def test_agent_diagnoses_and_fixes_code_self_correction(tmp_path: Path):
    """Verify agent self-correction loop: Run Tests (FAIL) -> Diagnose -> Replace Code -> Run Tests (PASS)."""
    init_git_repo(tmp_path)
    (tmp_path / "math_lib.py").write_text("def multiply(a, b):\n    return a + b\n")
    (tmp_path / "test_math.py").write_text("from math_lib import multiply\ndef test_multiply():\n    assert multiply(2, 3) == 6\n")

    subprocess.run(["git", "add", "."], cwd=tmp_path, capture_output=True, check=True)
    subprocess.run(["git", "commit", "-m", "v1"], cwd=tmp_path, capture_output=True, check=True)

    mock_responses = [
        # Step 1: Create plan
        AIMessage(
            content="Plan created for fixing multiply function.",
            tool_calls=[
                {
                    "name": "create_plan",
                    "args": {
                        "tasks": [
                            {"id": "t1", "title": "Run initial validation tests", "dependencies": []},
                            {"id": "t2", "title": "Fix multiply implementation", "dependencies": ["t1"]},
                            {"id": "t3", "title": "Verify fix with pytest", "dependencies": ["t2"]},
                        ]
                    },
                    "id": "c1",
                }
            ],
        ),
        # Step 2: Run initial tests (t1) -> FAIL
        AIMessage(
            content="Running initial test suite to check failure.",
            tool_calls=[
                {"name": "update_task_status", "args": {"task_id": "t1", "status": "in_progress"}, "id": "c2_1"},
                {"name": "run_tests", "args": {}, "id": "c2_2"},
            ],
        ),
        # Step 3: Observe failure, replace code (t2)
        AIMessage(
            content="Observed test failure: multiply(2, 3) returned 5. Fixing return a + b to return a * b.",
            tool_calls=[
                {"name": "update_task_status", "args": {"task_id": "t1", "status": "completed"}, "id": "c3_1"},
                {"name": "update_task_status", "args": {"task_id": "t2", "status": "in_progress"}, "id": "c3_2"},
                {
                    "name": "replace_in_file",
                    "args": {
                        "file_path": "math_lib.py",
                        "old_text": "return a + b",
                        "new_text": "return a * b",
                    },
                    "id": "c3_3",
                },
            ],
        ),
        # Step 4: Re-run tests (t3) -> PASS
        AIMessage(
            content="Fixed math_lib.py. Re-running validation tests for t3.",
            tool_calls=[
                {"name": "update_task_status", "args": {"task_id": "t2", "status": "completed"}, "id": "c4_1"},
                {"name": "update_task_status", "args": {"task_id": "t3", "status": "in_progress"}, "id": "c4_2"},
                {"name": "run_tests", "args": {}, "id": "c4_3"},
            ],
        ),
        # Step 5: Final completion
        AIMessage(
            content="Tests passed! Fixed multiply function successfully.",
            tool_calls=[
                {"name": "update_task_status", "args": {"task_id": "t3", "status": "completed"}, "id": "c5_1"}
            ],
        ),
    ]
    mock_llm = MockLLM(responses=mock_responses)

    final_state = run_agent(
        goal="Fix the multiply function so that the project's tests pass.",
        workspace_root=str(tmp_path),
        llm=mock_llm,
    )

    # 1. Real repository file fixed
    assert (tmp_path / "math_lib.py").read_text() == "def multiply(a, b):\n    return a * b\n"

    # 2. Validation result is passed
    val_res = final_state.get("validation_result")
    assert val_res is not None
    assert val_res["status"] == "passed"

    # 3. Retry count recorded for failure recovery
    assert final_state.get("retry_count", 0) >= 1


def test_retry_limit_enforcement(tmp_path: Path):
    """Verify retry_count increments on failed validation runs in state."""
    (tmp_path / "test_fail.py").write_text("def test_f(): assert False\n")

    mock_responses = [
        AIMessage(
            content="Running test 1",
            tool_calls=[{"name": "run_tests", "args": {}, "id": "c1"}],
        ),
        AIMessage(
            content="Running test 2",
            tool_calls=[{"name": "run_tests", "args": {}, "id": "c2"}],
        ),
        AIMessage(
            content="Reached max retries. Escalating failure.",
        ),
    ]
    mock_llm = MockLLM(responses=mock_responses)

    final_state = run_agent(
        goal="Fix persistent test failure",
        workspace_root=str(tmp_path),
        llm=mock_llm,
    )

    assert final_state.get("retry_count") == 2
    assert final_state.get("validation_result")["status"] == "failed"


# -----------------------------------------------------------------------------
# 5. Phase 7 Autonomous Goal Verification Tests
# -----------------------------------------------------------------------------
def test_agent_successful_goal_verification(tmp_path: Path):
    """Verify agent goal verification: run_tests (PASS) -> inspect repository -> verify_goal (PASS)."""
    init_git_repo(tmp_path)
    (tmp_path / "math_lib.py").write_text("def multiply(a, b):\n    return a + b\n")
    (tmp_path / "test_math.py").write_text("from math_lib import multiply\ndef test_multiply():\n    assert multiply(2, 3) == 6\n")

    subprocess.run(["git", "add", "."], cwd=tmp_path, capture_output=True, check=True)
    subprocess.run(["git", "commit", "-m", "v1"], cwd=tmp_path, capture_output=True, check=True)

    mock_responses = [
        AIMessage(
            content="Running initial test suite.",
            tool_calls=[{"name": "run_tests", "args": {}, "id": "c1"}],
        ),
        AIMessage(
            content="Test failed. Replacing return a + b with return a * b.",
            tool_calls=[
                {
                    "name": "replace_in_file",
                    "args": {"file_path": "math_lib.py", "old_text": "return a + b", "new_text": "return a * b"},
                    "id": "c2",
                }
            ],
        ),
        AIMessage(
            content="Re-running tests.",
            tool_calls=[{"name": "run_tests", "args": {}, "id": "c3"}],
        ),
        AIMessage(
            content="Tests passed. Inspecting math_lib.py to verify goal.",
            tool_calls=[{"name": "read_file", "args": {"file_path": "math_lib.py"}, "id": "c4"}],
        ),
        AIMessage(
            content="Repository evidence confirms multiply returned a * b.",
            tool_calls=[
                {
                    "name": "verify_goal",
                    "args": {
                        "status": "passed",
                        "summary": "multiply function correctly calculates product a * b.",
                        "evidence": ["math_lib.py contains return a * b", "pytest test_multiply passed"],
                    },
                    "id": "c5",
                }
            ],
        ),
        AIMessage(content="Goal verified successfully!"),
    ]
    mock_llm = MockLLM(responses=mock_responses)

    final_state = run_agent(
        goal="Fix multiply function so it returns product of a and b",
        workspace_root=str(tmp_path),
        llm=mock_llm,
    )

    assert final_state.get("validation_result", {}).get("status") == "passed"
    ver_res = final_state.get("verification_result")
    assert ver_res is not None
    assert ver_res["status"] == "passed"
    assert "math_lib.py contains return a * b" in ver_res["evidence"]


def test_agent_tests_pass_but_goal_incomplete(tmp_path: Path):
    """Verify that passing existing tests does NOT automatically produce goal verification success when requirement is missing."""
    (tmp_path / "math_lib.py").write_text("def multiply(a, b):\n    return a * b\n")
    (tmp_path / "test_math.py").write_text("from math_lib import multiply\ndef test_multiply():\n    assert multiply(2, 3) == 6\n")

    mock_responses = [
        AIMessage(
            content="Running existing tests.",
            tool_calls=[{"name": "run_tests", "args": {}, "id": "c1"}],
        ),
        AIMessage(
            content="Existing tests pass, but inspecting repo for subtract(a, b).",
            tool_calls=[{"name": "read_file", "args": {"file_path": "math_lib.py"}, "id": "c2"}],
        ),
        AIMessage(
            content="Function subtract(a, b) is missing in math_lib.py.",
            tool_calls=[
                {
                    "name": "verify_goal",
                    "args": {
                        "status": "failed",
                        "summary": "Requested subtract(a, b) function was not implemented.",
                        "evidence": ["math_lib.py does not define subtract(a, b)"],
                    },
                    "id": "c3",
                }
            ],
        ),
        AIMessage(content="Goal incomplete. Verification failed."),
    ]
    mock_llm = MockLLM(responses=mock_responses)

    final_state = run_agent(
        goal="Add subtract(a, b) function to math_lib.py",
        workspace_root=str(tmp_path),
        llm=mock_llm,
    )

    assert final_state.get("validation_result", {}).get("status") == "passed"
    ver_res = final_state.get("verification_result")
    assert ver_res is not None
    assert ver_res["status"] != "passed"
    assert ver_res["status"] == "failed"


def test_verification_uses_repository_evidence(tmp_path: Path):
    """Verify that agent performs repository inspection (read_file / search_code) before executing verify_goal tool."""
    (tmp_path / "config.py").write_text("APP_NAME = 'AutonomousAgent'\n")

    mock_responses = [
        AIMessage(
            content="Reading config.py to verify APP_NAME definition.",
            tool_calls=[{"name": "read_file", "args": {"file_path": "config.py"}, "id": "c1"}],
        ),
        AIMessage(
            content="Verifying goal against retrieved evidence.",
            tool_calls=[
                {
                    "name": "verify_goal",
                    "args": {
                        "status": "passed",
                        "summary": "APP_NAME is defined in config.py.",
                        "evidence": ["config.py contains APP_NAME = 'AutonomousAgent'"],
                    },
                    "id": "c2",
                }
            ],
        ),
    ]
    mock_llm = MockLLM(responses=mock_responses)

    final_state = run_agent(
        goal="Verify config.py contains APP_NAME",
        workspace_root=str(tmp_path),
        llm=mock_llm,
    )

    messages = final_state.get("messages", [])
    tool_names_called = []
    for msg in messages:
        if isinstance(msg, AIMessage) and getattr(msg, "tool_calls", None):
            for tc in msg.tool_calls:
                tool_names_called.append(tc.get("name"))

    assert "read_file" in tool_names_called
    assert "verify_goal" in tool_names_called
    assert tool_names_called.index("read_file") < tool_names_called.index("verify_goal")


def test_verification_failure_triggers_recovery(tmp_path: Path):
    """Verify that a goal verification failure triggers agent recovery and re-verification."""
    (tmp_path / "math_lib.py").write_text("def multiply(a, b):\n    return a * b\n")
    (tmp_path / "test_math.py").write_text("from math_lib import multiply\ndef test_multiply():\n    assert multiply(2, 3) == 6\n")

    mock_responses = [
        AIMessage(
            content="Running initial tests.",
            tool_calls=[{"name": "run_tests", "args": {}, "id": "c1"}],
        ),
        AIMessage(
            content="Reading math_lib.py to verify divide function.",
            tool_calls=[{"name": "read_file", "args": {"file_path": "math_lib.py"}, "id": "c2"}],
        ),
        AIMessage(
            content="Divide function is missing.",
            tool_calls=[
                {
                    "name": "verify_goal",
                    "args": {
                        "status": "failed",
                        "summary": "divide function is missing.",
                        "evidence": ["math_lib.py lacks divide"],
                    },
                    "id": "c3",
                }
            ],
        ),
        AIMessage(
            content="Adding divide(a, b) function to math_lib.py.",
            tool_calls=[
                {
                    "name": "write_file",
                    "args": {
                        "file_path": "math_lib.py",
                        "content": "def multiply(a, b):\n    return a * b\n\ndef divide(a, b):\n    if b == 0:\n        raise ValueError('Division by zero')\n    return a / b\n",
                    },
                    "id": "c4",
                }
            ],
        ),
        AIMessage(
            content="Re-running tests.",
            tool_calls=[{"name": "run_tests", "args": {}, "id": "c5"}],
        ),
        AIMessage(
            content="Reading math_lib.py to verify divide implementation.",
            tool_calls=[{"name": "read_file", "args": {"file_path": "math_lib.py"}, "id": "c6"}],
        ),
        AIMessage(
            content="Divide function is now present and verified.",
            tool_calls=[
                {
                    "name": "verify_goal",
                    "args": {
                        "status": "passed",
                        "summary": "divide(a, b) successfully implemented.",
                        "evidence": ["math_lib.py defines divide(a, b)"],
                    },
                    "id": "c7",
                }
            ],
        ),
        AIMessage(content="Goal verified after recovery."),
    ]
    mock_llm = MockLLM(responses=mock_responses)

    final_state = run_agent(
        goal="Add divide(a, b) function to math_lib.py",
        workspace_root=str(tmp_path),
        llm=mock_llm,
    )

    ver_res = final_state.get("verification_result")
    assert ver_res is not None
    assert ver_res["status"] == "passed"
    assert final_state.get("retry_count", 0) >= 1


def test_uncertain_verification(tmp_path: Path):
    """Verify that when repository evidence is insufficient, status is uncertain rather than falsely passed."""
    mock_responses = [
        AIMessage(
            content="Listing files to find production deployment config.",
            tool_calls=[{"name": "list_files", "args": {"directory": "."}, "id": "c1"}],
        ),
        AIMessage(
            content="No deployment config found. Returning uncertain verification.",
            tool_calls=[
                {
                    "name": "verify_goal",
                    "args": {
                        "status": "uncertain",
                        "summary": "Insufficient evidence to verify cloud deployment parameters.",
                        "evidence": ["No deploy.yml or docker-compose.yml files found"],
                    },
                    "id": "c2",
                }
            ],
        ),
        AIMessage(content="Verification completed with uncertain status."),
    ]
    mock_llm = MockLLM(responses=mock_responses)

    final_state = run_agent(
        goal="Verify cloud deployment parameters",
        workspace_root=str(tmp_path),
        llm=mock_llm,
    )

    ver_res = final_state.get("verification_result")
    assert ver_res is not None
    assert ver_res["status"] == "uncertain"


def test_end_to_end_goal_verification_workflow(tmp_path: Path):
    """End-to-end test: Fix multiply function, run real pytest, inspect repo, verify goal."""
    init_git_repo(tmp_path)
    (tmp_path / "math_lib.py").write_text("def multiply(a, b):\n    return a + b\n")
    (tmp_path / "test_math.py").write_text("from math_lib import multiply\ndef test_multiply():\n    assert multiply(2, 3) == 6\n")

    subprocess.run(["git", "add", "."], cwd=tmp_path, capture_output=True, check=True)
    subprocess.run(["git", "commit", "-m", "initial"], cwd=tmp_path, capture_output=True, check=True)

    mock_responses = [
        AIMessage(
            content="Running pytest to inspect initial failure.",
            tool_calls=[{"name": "run_tests", "args": {}, "id": "c1"}],
        ),
        AIMessage(
            content="Tests failed. Replacing return a + b with return a * b.",
            tool_calls=[
                {
                    "name": "replace_in_file",
                    "args": {"file_path": "math_lib.py", "old_text": "return a + b", "new_text": "return a * b"},
                    "id": "c2",
                }
            ],
        ),
        AIMessage(
            content="Re-running tests.",
            tool_calls=[{"name": "run_tests", "args": {}, "id": "c3"}],
        ),
        AIMessage(
            content="Inspecting git diff for verification evidence.",
            tool_calls=[{"name": "git_diff", "args": {}, "id": "c4"}],
        ),
        AIMessage(
            content="Verifying goal against diff and test pass.",
            tool_calls=[
                {
                    "name": "verify_goal",
                    "args": {
                        "status": "passed",
                        "summary": "multiply function successfully fixed and verified.",
                        "evidence": ["git_diff shows return a * b", "pytest test_multiply passed"],
                    },
                    "id": "c5",
                }
            ],
        ),
        AIMessage(content="Goal verified! Complete."),
    ]
    mock_llm = MockLLM(responses=mock_responses)

    final_state = run_agent(
        goal="Fix the multiply function so that it correctly multiplies two numbers.",
        workspace_root=str(tmp_path),
        llm=mock_llm,
    )

    assert (tmp_path / "math_lib.py").read_text() == "def multiply(a, b):\n    return a * b\n"
    assert final_state.get("validation_result", {}).get("status") == "passed"
    ver_res = final_state.get("verification_result")
    assert ver_res is not None
    assert ver_res["status"] == "passed"


# -----------------------------------------------------------------------------
# 6. Phase 8 Persistent Memory & Long-Running Agent State Tests
# -----------------------------------------------------------------------------
def test_save_state(tmp_path: Path):
    """Verify save_state persists valid serializable state to JSON file."""
    state: AgentState = {
        "task_id": "task_save_1",
        "user_goal": "Add auth module",
        "workspace_root": str(tmp_path),
        "messages": [HumanMessage(content="Add auth module")],
        "plan": create_plan_state("Add auth module", [{"id": "t1", "title": "Inspect code"}]),
        "retrieved_context": [{"query": "auth"}],
        "modified_files": ["auth.py"],
        "validation_result": {"status": "passed", "exit_code": 0, "summary": "Passed"},
        "verification_result": {"status": "passed", "summary": "Verified", "evidence": ["auth.py present"]},
        "retry_count": 1,
        "max_retries": 3,
    }

    path = save_state("task_save_1", state, storage_dir=tmp_path)
    assert path.exists()
    assert path.name == "task_save_1.json"


def test_load_state(tmp_path: Path):
    """Verify load_state restores state attributes accurately from persisted storage."""
    state: AgentState = {
        "task_id": "task_load_1",
        "user_goal": "Refactor database module",
        "workspace_root": str(tmp_path),
        "messages": [HumanMessage(content="Refactor database module")],
        "plan": create_plan_state("Refactor database module", [{"id": "t1", "title": "Check DB"}]),
        "retrieved_context": [{"query": "database"}],
        "modified_files": ["db.py"],
        "validation_result": {"status": "failed", "exit_code": 1, "summary": "Failed"},
        "verification_result": {"status": "failed", "summary": "Unverified", "evidence": ["db.py missing"]},
        "retry_count": 2,
        "max_retries": 3,
    }

    save_state("task_load_1", state, storage_dir=tmp_path)
    loaded = load_state("task_load_1", storage_dir=tmp_path)

    assert loaded["task_id"] == "task_load_1"
    assert loaded["user_goal"] == "Refactor database module"
    assert loaded["modified_files"] == ["db.py"]
    assert loaded["validation_result"]["status"] == "failed"
    assert loaded["verification_result"]["status"] == "failed"
    assert loaded["retry_count"] == 2


def test_retry_count_survives_resume(tmp_path: Path):
    """Verify that retry_count is preserved when a task is resumed and not reset to zero."""
    state: AgentState = {
        "task_id": "task_retry_1",
        "user_goal": "Fix flaky test",
        "workspace_root": str(tmp_path),
        "messages": [HumanMessage(content="Fix flaky test")],
        "plan": None,
        "retrieved_context": [],
        "modified_files": ["test_flaky.py"],
        "validation_result": {"status": "failed", "exit_code": 1, "summary": "1 failed"},
        "verification_result": None,
        "retry_count": 2,
        "max_retries": 3,
    }

    save_state("task_retry_1", state, storage_dir=tmp_path)

    mock_responses = [
        AIMessage(content="Resuming task. Retry count should be preserved."),
    ]
    mock_llm = MockLLM(responses=mock_responses)

    resumed = run_agent(
        task_id="task_retry_1",
        resume=True,
        storage_dir=str(tmp_path),
        llm=mock_llm,
    )

    assert resumed.get("retry_count") == 2


def test_plan_survives_resume(tmp_path: Path):
    """Verify partially completed plan task statuses survive process pause/resume."""
    plan = create_plan_state("Task Plan", [{"id": "t1", "title": "Step 1"}, {"id": "t2", "title": "Step 2"}])
    plan = update_task_state(plan, "t1", "completed")

    state: AgentState = {
        "task_id": "task_plan_1",
        "user_goal": "Task Plan",
        "workspace_root": str(tmp_path),
        "messages": [HumanMessage(content="Task Plan")],
        "plan": plan,
        "retrieved_context": [],
        "modified_files": [],
        "validation_result": None,
        "verification_result": None,
        "retry_count": 0,
        "max_retries": 3,
    }

    save_state("task_plan_1", state, storage_dir=tmp_path)

    mock_responses = [AIMessage(content="Continuing execution from step 2.")]
    mock_llm = MockLLM(responses=mock_responses)

    resumed = run_agent(
        task_id="task_plan_1",
        resume=True,
        storage_dir=str(tmp_path),
        llm=mock_llm,
    )

    restored_plan = resumed.get("plan")
    assert restored_plan is not None
    assert restored_plan["tasks"][0]["status"] == "completed"
    assert restored_plan["current_task_id"] == "t2"


def test_goal_survives_resume(tmp_path: Path):
    """Verify original user goal is restored when resuming with task_id alone."""
    state: AgentState = {
        "task_id": "task_goal_1",
        "user_goal": "Original long-running goal",
        "workspace_root": str(tmp_path),
        "messages": [HumanMessage(content="Original long-running goal")],
        "plan": None,
        "retrieved_context": [],
        "modified_files": [],
        "validation_result": None,
        "verification_result": None,
        "retry_count": 0,
        "max_retries": 3,
    }

    save_state("task_goal_1", state, storage_dir=tmp_path)

    mock_responses = [AIMessage(content="Continuing goal.")]
    mock_llm = MockLLM(responses=mock_responses)

    resumed = resume_agent(
        task_id="task_goal_1",
        storage_dir=str(tmp_path),
        llm=mock_llm,
    )

    assert resumed.get("user_goal") == "Original long-running goal"


def test_missing_task_id_resume(tmp_path: Path):
    """Verify resuming a non-existent task ID raises FileNotFoundError."""
    with pytest.raises(FileNotFoundError):
        resume_agent("non_existent_task_xyz", storage_dir=str(tmp_path))


def test_corrupt_state_handling(tmp_path: Path):
    """Verify loading corrupted JSON state produces a controlled ValueError."""
    corrupt_file = tmp_path / "corrupt_task.json"
    corrupt_file.write_text("{ invalid_json_syntax ...", encoding="utf-8")

    with pytest.raises(ValueError):
        load_state("corrupt_task", storage_dir=tmp_path)


def test_atomic_persistence(tmp_path: Path):
    """Verify atomic state updates maintain file integrity across repeated writes."""
    state: AgentState = {
        "task_id": "atomic_task",
        "user_goal": "Atomic test",
        "workspace_root": str(tmp_path),
        "messages": [HumanMessage(content="Atomic test")],
        "plan": None,
        "retrieved_context": [],
        "modified_files": [],
        "validation_result": None,
        "verification_result": None,
        "retry_count": 0,
        "max_retries": 3,
    }

    for i in range(5):
        state["retry_count"] = i
        save_state("atomic_task", state, storage_dir=tmp_path)

    final_loaded = load_state("atomic_task", storage_dir=tmp_path)
    assert final_loaded["retry_count"] == 4


def test_completed_task_state_preservation(tmp_path: Path):
    """Verify completed task state retains status, validation, and verification results."""
    state: AgentState = {
        "task_id": "completed_task",
        "user_goal": "Completed goal",
        "workspace_root": str(tmp_path),
        "messages": [HumanMessage(content="Completed goal")],
        "plan": None,
        "retrieved_context": [],
        "modified_files": ["app.py"],
        "validation_result": {"status": "passed", "exit_code": 0, "summary": "All tests passed"},
        "verification_result": {"status": "passed", "summary": "Goal satisfied", "evidence": ["app.py updated"]},
        "retry_count": 1,
        "max_retries": 3,
    }

    save_state("completed_task", state, status="completed", storage_dir=tmp_path)
    loaded = load_state("completed_task", storage_dir=tmp_path)

    assert loaded["status"] == "completed"
    assert loaded["validation_result"]["status"] == "passed"
    assert loaded["verification_result"]["status"] == "passed"


def test_delete_state(tmp_path: Path):
    """Verify delete_state removes saved state file cleanly."""
    state: AgentState = {
        "task_id": "del_task",
        "user_goal": "Delete test",
        "workspace_root": str(tmp_path),
        "messages": [HumanMessage(content="Delete test")],
        "plan": None,
        "retrieved_context": [],
        "modified_files": [],
        "validation_result": None,
        "verification_result": None,
        "retry_count": 0,
        "max_retries": 3,
    }

    save_state("del_task", state, storage_dir=tmp_path)
    assert (tmp_path / "del_task.json").exists()

    removed = delete_state("del_task", storage_dir=tmp_path)
    assert removed is True
    assert not (tmp_path / "del_task.json").exists()


def test_security_path_traversal_prevention(tmp_path: Path):
    """Verify directory traversal attempts in task_id are safely denied."""
    with pytest.raises(ValueError):
        safe_resolve_task_memory_path(tmp_path, "../../etc/passwd")

    with pytest.raises(ValueError):
        load_state("../escaped_task", storage_dir=tmp_path)


def test_end_to_end_resume_workflow(tmp_path: Path):
    """Simulate process boundary: Process A runs step 1 & saves state -> Process B loads state & completes work."""
    init_git_repo(tmp_path)
    (tmp_path / "math_lib.py").write_text("def multiply(a, b):\n    return a + b\n")
    (tmp_path / "test_math.py").write_text("from math_lib import multiply\ndef test_multiply():\n    assert multiply(2, 3) == 6\n")

    subprocess.run(["git", "add", "."], cwd=tmp_path, capture_output=True, check=True)
    subprocess.run(["git", "commit", "-m", "v1"], cwd=tmp_path, capture_output=True, check=True)

    task_id = "e2e_persistent_task_100"
    storage_dir = tmp_path / "memory_store"

    # --- PROCESS A: Initial agent run performs replacement and saves state ---
    process_a_responses = [
        AIMessage(
            content="Process A: Running test suite.",
            tool_calls=[{"name": "run_tests", "args": {}, "id": "c1"}],
        ),
        AIMessage(
            content="Process A: Fixing math_lib.py.",
            tool_calls=[
                {
                    "name": "replace_in_file",
                    "args": {"file_path": "math_lib.py", "old_text": "return a + b", "new_text": "return a * b"},
                    "id": "c2",
                }
            ],
        ),
    ]
    process_a_llm = MockLLM(responses=process_a_responses)

    state_a = run_agent(
        goal="Fix multiply function in math_lib.py",
        workspace_root=str(tmp_path),
        task_id=task_id,
        storage_dir=str(storage_dir),
        llm=process_a_llm,
    )

    assert (tmp_path / "math_lib.py").read_text() == "def multiply(a, b):\n    return a * b\n"
    assert (storage_dir / f"{task_id}.json").exists()

    # --- PROCESS B: Process restart simulation - fresh agent invocation loads state and finishes task ---
    process_b_responses = [
        AIMessage(
            content="Process B: Re-running validation tests.",
            tool_calls=[{"name": "run_tests", "args": {}, "id": "c3"}],
        ),
        AIMessage(
            content="Process B: Verifying goal against repository evidence.",
            tool_calls=[
                {
                    "name": "verify_goal",
                    "args": {
                        "status": "passed",
                        "summary": "multiply function verified after resume.",
                        "evidence": ["math_lib.py returns a * b", "pytest test_multiply passed"],
                    },
                    "id": "c4",
                }
            ],
        ),
        AIMessage(content="Process B: Task complete."),
    ]
    process_b_llm = MockLLM(responses=process_b_responses)

    state_b = resume_agent(
        task_id=task_id,
        workspace_root=str(tmp_path),
        storage_dir=str(storage_dir),
        llm=process_b_llm,
    )

    assert state_b.get("user_goal") == "Fix multiply function in math_lib.py"
    assert state_b.get("validation_result", {}).get("status") == "passed"
    ver_res = state_b.get("verification_result")
    assert ver_res is not None
    assert ver_res["status"] == "passed"


