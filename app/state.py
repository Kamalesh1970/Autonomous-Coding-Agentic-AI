"""Agent state definition and planning structures for the Autonomous Coding Agent."""

from typing import Annotated, Literal, Sequence, TypedDict
from langchain_core.messages import BaseMessage
from langgraph.graph.message import add_messages

TaskStatus = Literal["pending", "in_progress", "completed", "failed", "blocked"]


class Task(TypedDict, total=False):
    """Structured representation of a subtask within an execution plan.

    Attributes:
        id: Unique identifier for the task (e.g. 'task-1').
        title: Short title of the task.
        description: Detailed explanation of what needs to be done.
        status: Execution status (pending, in_progress, completed, failed, blocked).
        dependencies: List of task IDs that must be completed before this task can start.
    """
    id: str
    title: str
    description: str
    status: TaskStatus
    dependencies: list[str]


class ExecutionPlan(TypedDict, total=False):
    """Structured representation of the overall agent execution plan.

    Attributes:
        goal: The high-level user goal.
        tasks: List of subtasks forming the plan.
        current_task_id: ID of the currently active task, if any.
        revision_count: Number of times the plan has been revised.
        revision_reason: Explanation of the most recent plan revision.
    """
    goal: str
    tasks: list[Task]
    current_task_id: str | None
    revision_count: int
    revision_reason: str | None


class AgentState(TypedDict, total=False):
    """Explicit state representation for the autonomous coding agent.

    Attributes:
        messages: Sequence of conversation messages (user goal, LLM responses, tool outputs).
            Annotated with `add_messages` reducer to automatically handle appending new messages.
        user_goal: The high-level objective provided by the user.
        workspace_root: Base directory path constraining file tool execution for safety.
        plan: Optional ExecutionPlan structured object tracking task decomposition and progress.
        retrieved_context: Optional list of retrieved code context snippets and query metadata.
        modified_files: Optional list of repository file paths modified by code writing tools.
    """
    messages: Annotated[Sequence[BaseMessage], add_messages]
    user_goal: str
    workspace_root: str
    plan: ExecutionPlan | None
    retrieved_context: list[dict] | None
    modified_files: list[str] | None


def create_plan_state(goal: str, raw_tasks: list[dict]) -> ExecutionPlan:
    """Helper to create a structured ExecutionPlan object from a goal and task definitions."""
    tasks: list[Task] = []
    for idx, t in enumerate(raw_tasks, start=1):
        task_id = str(t.get("id", f"task-{idx}"))
        dependencies = [str(d) for d in t.get("dependencies", [])]
        status: TaskStatus = t.get("status", "pending")

        tasks.append(
            Task(
                id=task_id,
                title=str(t.get("title", f"Task {idx}")),
                description=str(t.get("description", "")),
                status=status,
                dependencies=dependencies,
            )
        )

    current_id = get_next_available_task_id(tasks)

    return ExecutionPlan(
        goal=goal,
        tasks=tasks,
        current_task_id=current_id,
        revision_count=0,
        revision_reason=None,
    )


def get_next_available_task_id(tasks: list[Task]) -> str | None:
    """Finds the ID of the first pending task whose dependencies are all completed."""
    completed_ids = {t["id"] for t in tasks if t.get("status") == "completed"}

    # Check if there is already an in_progress task
    for t in tasks:
        if t.get("status") == "in_progress":
            return t["id"]

    for t in tasks:
        if t.get("status") == "pending":
            deps = set(t.get("dependencies", []))
            if deps.issubset(completed_ids):
                return t["id"]

    return None


def update_task_state(plan: ExecutionPlan, task_id: str, status: TaskStatus) -> ExecutionPlan:
    """Updates the status of a specific task within an execution plan."""
    updated_tasks = []
    for t in plan.get("tasks", []):
        if t["id"] == task_id:
            updated_task = dict(t)
            updated_task["status"] = status
            updated_tasks.append(updated_task)
        else:
            updated_tasks.append(t)

    current_id = get_next_available_task_id(updated_tasks)

    return ExecutionPlan(
        goal=plan.get("goal", ""),
        tasks=updated_tasks,
        current_task_id=current_id,
        revision_count=plan.get("revision_count", 0),
        revision_reason=plan.get("revision_reason"),
    )


def revise_plan_state(plan: ExecutionPlan, new_raw_tasks: list[dict], reason: str) -> ExecutionPlan:
    """Revises an existing execution plan with updated tasks and records the revision reason."""
    revised_plan = create_plan_state(goal=plan.get("goal", ""), raw_tasks=new_raw_tasks)
    revised_plan["revision_count"] = plan.get("revision_count", 0) + 1
    revised_plan["revision_reason"] = reason
    return revised_plan
