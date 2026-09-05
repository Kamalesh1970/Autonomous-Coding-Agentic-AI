"""Persistent execution memory and state storage module for Phase 8 Autonomous Coding Agent."""

import json
import os
import re
from pathlib import Path
from typing import Any, Sequence

from langchain_core.messages import (
    AIMessage,
    BaseMessage,
    HumanMessage,
    SystemMessage,
    ToolMessage,
)

from app.state import AgentState, ExecutionPlan, ValidationResult, VerificationResult


def validate_task_id(task_id: str) -> None:
    """Validates task_id format to prevent directory traversal and path injection attacks."""
    if not task_id or not isinstance(task_id, str):
        raise ValueError("Error: task_id must be a non-empty string.")

    if not re.match(r"^[a-zA-Z0-9_-]+$", task_id):
        raise ValueError(
            f"Error: Invalid task_id '{task_id}'. Task ID must contain only alphanumeric characters, underscores, or hyphens."
        )


def safe_resolve_task_memory_path(storage_dir: str | Path, task_id: str) -> Path:
    """Resolves task memory JSON file path safely within storage_dir directory constraints."""
    validate_task_id(task_id)

    base = Path(storage_dir).resolve()
    target = (base / f"{task_id}.json").resolve()

    try:
        target.relative_to(base)
    except ValueError:
        raise ValueError(
            f"Access denied: task_id '{task_id}' escapes memory storage directory '{base}'"
        )

    return target


def serialize_message(msg: BaseMessage) -> dict[str, Any]:
    """Converts a LangChain BaseMessage instance into a JSON-safe dictionary."""
    msg_type = msg.__class__.__name__
    content = getattr(msg, "content", "")
    tool_calls = getattr(msg, "tool_calls", None)
    tool_call_id = getattr(msg, "tool_call_id", None)
    name = getattr(msg, "name", None)

    serialized: dict[str, Any] = {"type": msg_type, "content": content}

    if tool_calls is not None:
        serialized["tool_calls"] = tool_calls
    if tool_call_id is not None:
        serialized["tool_call_id"] = tool_call_id
    if name is not None:
        serialized["name"] = name

    return serialized


def deserialize_message(data: dict[str, Any]) -> BaseMessage:
    """Reconstructs a LangChain BaseMessage instance from a serialized dictionary."""
    msg_type = data.get("type", "HumanMessage")
    content = str(data.get("content", ""))
    tool_calls = data.get("tool_calls")
    tool_call_id = data.get("tool_call_id")
    name = data.get("name")

    if msg_type == "HumanMessage":
        return HumanMessage(content=content)
    elif msg_type == "AIMessage":
        kwargs: dict[str, Any] = {"content": content}
        if tool_calls is not None:
            kwargs["tool_calls"] = tool_calls
        return AIMessage(**kwargs)
    elif msg_type == "ToolMessage":
        kwargs = {"content": content, "tool_call_id": tool_call_id or "call_default"}
        if name is not None:
            kwargs["name"] = name
        return ToolMessage(**kwargs)
    elif msg_type == "SystemMessage":
        return SystemMessage(content=content)
    else:
        return HumanMessage(content=content)


def serialize_state(state: AgentState, status: str = "running") -> dict[str, Any]:
    """Converts an AgentState typed dictionary into a JSON-safe serializable dictionary."""
    task_id = str(state.get("task_id") or "task_default")
    user_goal = str(state.get("user_goal") or "")
    workspace_root = str(state.get("workspace_root") or ".")
    plan = state.get("plan")
    retrieved_context = state.get("retrieved_context") or []
    modified_files = state.get("modified_files") or []
    validation_result = state.get("validation_result")
    verification_result = state.get("verification_result")
    retry_count = int(state.get("retry_count", 0))
    max_retries = int(state.get("max_retries", 3))

    git_status = state.get("git_status")
    git_diff = state.get("git_diff")
    current_branch = state.get("current_branch")
    target_branch = state.get("target_branch")
    delivery_action = state.get("delivery_action")
    approval_required = bool(state.get("approval_required", False))
    approval_status = str(state.get("approval_status") or "not_required")
    approval_reason = state.get("approval_reason")
    commit_message = state.get("commit_message")
    commit_created = bool(state.get("commit_created", False))
    push_requested = bool(state.get("push_requested", False))
    pr_requested = bool(state.get("pr_requested", False))

    mode = str(state.get("mode") or "single_agent")
    agent_role = state.get("agent_role")
    analysis_result = state.get("analysis_result")
    coding_result = state.get("coding_result")
    review_result = state.get("review_result")
    review_status = str(state.get("review_status") or "pending")
    review_feedback = state.get("review_feedback")
    multi_agent_iteration = int(state.get("multi_agent_iteration", 0))
    max_multi_agent_iterations = int(state.get("max_multi_agent_iterations", 3))

    raw_messages = state.get("messages", [])
    serialized_messages = [serialize_message(m) for m in raw_messages]

    return {
        "version": "1.0",
        "task_id": task_id,
        "user_goal": user_goal,
        "workspace_root": workspace_root,
        "status": status,
        "plan": plan,
        "retrieved_context": retrieved_context,
        "modified_files": modified_files,
        "validation_result": validation_result,
        "verification_result": verification_result,
        "retry_count": retry_count,
        "max_retries": max_retries,
        "git_status": git_status,
        "git_diff": git_diff,
        "current_branch": current_branch,
        "target_branch": target_branch,
        "delivery_action": delivery_action,
        "approval_required": approval_required,
        "approval_status": approval_status,
        "approval_reason": approval_reason,
        "commit_message": commit_message,
        "commit_created": commit_created,
        "push_requested": push_requested,
        "pr_requested": pr_requested,
        "mode": mode,
        "agent_role": agent_role,
        "analysis_result": analysis_result,
        "coding_result": coding_result,
        "review_result": review_result,
        "review_status": review_status,
        "review_feedback": review_feedback,
        "multi_agent_iteration": multi_agent_iteration,
        "max_multi_agent_iterations": max_multi_agent_iterations,
        "messages": serialized_messages,
    }


def deserialize_state(data: dict[str, Any]) -> AgentState:
    """Restores a serialized JSON dictionary back into a typed AgentState object."""
    if not isinstance(data, dict):
        raise ValueError("Error: State data must be a JSON dictionary object.")

    user_goal = data.get("user_goal")
    if user_goal is None:
        raise ValueError("Error: Corrupted state file missing required field 'user_goal'.")

    task_id = str(data.get("task_id") or "task_default")
    workspace_root = str(data.get("workspace_root") or ".")
    status = str(data.get("status") or "running")
    plan = data.get("plan")
    retrieved_context = list(data.get("retrieved_context") or [])
    modified_files = list(data.get("modified_files") or [])
    validation_result = data.get("validation_result")
    verification_result = data.get("verification_result")
    retry_count = int(data.get("retry_count", 0))
    max_retries = int(data.get("max_retries", 3))

    git_status = data.get("git_status")
    git_diff = data.get("git_diff")
    current_branch = data.get("current_branch")
    target_branch = data.get("target_branch")
    delivery_action = data.get("delivery_action")
    approval_required = bool(data.get("approval_required", False))
    approval_status = data.get("approval_status", "not_required")
    approval_reason = data.get("approval_reason")
    commit_message = data.get("commit_message")
    commit_created = bool(data.get("commit_created", False))
    push_requested = bool(data.get("push_requested", False))
    pr_requested = bool(data.get("pr_requested", False))

    mode = data.get("mode", "single_agent")
    agent_role = data.get("agent_role")
    analysis_result = data.get("analysis_result")
    coding_result = data.get("coding_result")
    review_result = data.get("review_result")
    review_status = data.get("review_status", "pending")
    review_feedback = data.get("review_feedback")
    multi_agent_iteration = int(data.get("multi_agent_iteration", 0))
    max_multi_agent_iterations = int(data.get("max_multi_agent_iterations", 3))

    raw_messages = data.get("messages") or []
    deserialized_messages = [deserialize_message(m) for m in raw_messages]

    return AgentState(
        task_id=task_id,
        user_goal=str(user_goal),
        workspace_root=workspace_root,
        status=status,
        plan=plan,
        retrieved_context=retrieved_context,
        modified_files=modified_files,
        validation_result=validation_result,
        verification_result=verification_result,
        retry_count=retry_count,
        max_retries=max_retries,
        git_status=git_status,
        git_diff=git_diff,
        current_branch=current_branch,
        target_branch=target_branch,
        delivery_action=delivery_action,
        approval_required=approval_required,
        approval_status=approval_status,
        approval_reason=approval_reason,
        commit_message=commit_message,
        commit_created=commit_created,
        push_requested=push_requested,
        pr_requested=pr_requested,
        mode=mode,
        agent_role=agent_role,
        analysis_result=analysis_result,
        coding_result=coding_result,
        review_result=review_result,
        review_status=review_status,
        review_feedback=review_feedback,
        multi_agent_iteration=multi_agent_iteration,
        max_multi_agent_iterations=max_multi_agent_iterations,
        messages=deserialized_messages,
    )



def save_state(
    task_id: str,
    state: AgentState,
    status: str = "running",
    storage_dir: str | Path = ".agent_memory",
) -> Path:
    """Atomically persists serializable AgentState into storage_dir/<task_id>.json."""
    validate_task_id(task_id)

    storage_path = Path(storage_dir)
    storage_path.mkdir(parents=True, exist_ok=True)

    file_path = safe_resolve_task_memory_path(storage_path, task_id)
    tmp_file_path = file_path.with_suffix(".tmp")

    serialized = serialize_state(state, status=status)
    serialized["task_id"] = task_id

    json_text = json.dumps(serialized, indent=2, ensure_ascii=False)

    try:
        tmp_file_path.write_text(json_text, encoding="utf-8")
        os.replace(tmp_file_path, file_path)
        return file_path
    except Exception as exc:
        if tmp_file_path.exists():
            try:
                tmp_file_path.unlink()
            except Exception:
                pass
        raise IOError(f"Failed to persist state for task_id '{task_id}': {exc}") from exc


def load_state(
    task_id: str,
    storage_dir: str | Path = ".agent_memory",
) -> AgentState:
    """Loads and deserializes persisted AgentState from storage_dir/<task_id>.json."""
    validate_task_id(task_id)

    storage_path = Path(storage_dir)
    file_path = safe_resolve_task_memory_path(storage_path, task_id)

    if not file_path.exists():
        raise FileNotFoundError(f"No persisted state found for task_id '{task_id}'.")

    try:
        content = file_path.read_text(encoding="utf-8")
        data = json.loads(content)
        return deserialize_state(data)
    except json.JSONDecodeError as err:
        raise ValueError(f"Corrupted state file for task_id '{task_id}': invalid JSON ({err})") from err
    except Exception as exc:
        if isinstance(exc, ValueError):
            raise
        raise IOError(f"Error loading state for task_id '{task_id}': {exc}") from exc


def delete_state(
    task_id: str,
    storage_dir: str | Path = ".agent_memory",
) -> bool:
    """Deletes persisted state file for task_id if it exists."""
    validate_task_id(task_id)

    try:
        file_path = safe_resolve_task_memory_path(storage_dir, task_id)
        if file_path.exists():
            file_path.unlink()
            return True
        return False
    except Exception:
        return False
