"""Agent state definition for the Autonomous Coding Agent."""

from typing import Annotated, Sequence, TypedDict
from langchain_core.messages import BaseMessage
from langgraph.graph.message import add_messages


class AgentState(TypedDict):
    """Explicit state representation for the autonomous coding agent.

    Attributes:
        messages: Sequence of conversation messages (user goal, LLM responses, tool outputs).
            Annotated with `add_messages` reducer to automatically handle appending new messages.
        user_goal: The high-level objective provided by the user.
        workspace_root: Base directory path constraining file tool execution for safety.
    """
    messages: Annotated[Sequence[BaseMessage], add_messages]
    user_goal: str
    workspace_root: str
