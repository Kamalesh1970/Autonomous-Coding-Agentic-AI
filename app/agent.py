"""LangGraph Agent Loop for Phase 1 Autonomous Coding Agent."""

import os
import sys
from typing import Literal
from dotenv import load_dotenv

from langchain_core.messages import HumanMessage, AIMessage, BaseMessage
from langchain_core.language_models import BaseChatModel
from langchain_openai import ChatOpenAI
from langgraph.graph import StateGraph, START, END
from langgraph.prebuilt import ToolNode

from app.state import AgentState
from app.tools import create_workspace_tools


def get_default_llm() -> BaseChatModel:
    """Initialize the default LLM using environment variables."""
    load_dotenv()
    api_key = os.getenv("OPENAI_API_KEY")
    model_name = os.getenv("OPENAI_MODEL_NAME", "gpt-4o-mini")

    if not api_key or api_key == "your_openai_api_key_here":
        raise ValueError(
            "OPENAI_API_KEY environment variable is not configured. "
            "Please set a valid key in .env or provide a mock LLM for testing."
        )

    return ChatOpenAI(model=model_name, temperature=0)


def build_agent_graph(llm: BaseChatModel | None = None, workspace_root: str = "."):
    """Constructs and compiles the LangGraph single-agent loop.

    Args:
        llm: Optional chat model instance. Defaults to ChatOpenAI configured via env.
        workspace_root: Base workspace directory for safe tool operations.

    Returns:
        Compiled StateGraph instance.
    """
    if llm is None:
        llm = get_default_llm()

    tools = create_workspace_tools(workspace_root=workspace_root)
    llm_with_tools = llm.bind_tools(tools)

    def reason_node(state: AgentState) -> dict:
        """Reasoning node that invokes the model with accumulated messages."""
        messages = list(state.get("messages", []))
        user_goal = state.get("user_goal", "")

        # If no messages exist yet, seed with user goal
        if not messages and user_goal:
            messages = [HumanMessage(content=user_goal)]

        response = llm_with_tools.invoke(messages)

        if not state.get("messages") and user_goal:
            return {"messages": [HumanMessage(content=user_goal), response]}
        return {"messages": [response]}

    tool_node = ToolNode(tools)

    def route_after_reason(state: AgentState) -> Literal["tools", "__end__"]:
        """Conditional edge checking if the model requested tool execution."""
        messages = state.get("messages", [])
        if not messages:
            return END

        last_message = messages[-1]

        # Check if the AI requested any tool calls
        if isinstance(last_message, AIMessage) and getattr(last_message, "tool_calls", None):
            return "tools"

        return END

    workflow = StateGraph(AgentState)
    workflow.add_node("reason", reason_node)
    workflow.add_node("tools", tool_node)

    workflow.add_edge(START, "reason")
    workflow.add_conditional_edges("reason", route_after_reason, ["tools", END])
    workflow.add_edge("tools", "reason")

    return workflow.compile()


def run_agent(goal: str, workspace_root: str = ".", llm: BaseChatModel | None = None) -> dict:
    """Executes the agent loop for a given user goal.

    Args:
        goal: High-level goal string for the agent.
        workspace_root: Directory path restricting tool execution.
        llm: Optional chat model (useful for passing mocked LLMs in tests).

    Returns:
        Final state dictionary containing the conversation messages and history.
    """
    graph = build_agent_graph(llm=llm, workspace_root=workspace_root)
    initial_state = {
        "user_goal": goal,
        "workspace_root": workspace_root,
        "messages": [HumanMessage(content=goal)],
    }

    return graph.invoke(initial_state)


def main():
    """CLI entrypoint for running the agent directly."""
    if len(sys.argv) < 2:
        print("Usage: python -m app.agent \"<user_goal>\" [workspace_root]")
        sys.exit(1)

    goal = sys.argv[1]
    workspace_root = sys.argv[2] if len(sys.argv) > 2 else "."

    print(f"Goal: {goal}")
    print(f"Workspace Root: {os.path.abspath(workspace_root)}")
    print("-" * 50)

    try:
        final_state = run_agent(goal=goal, workspace_root=workspace_root)
        messages = final_state.get("messages", [])
        print("\n=== Agent Trace ===")
        for msg in messages:
            role = msg.__class__.__name__
            content = getattr(msg, "content", "")
            tool_calls = getattr(msg, "tool_calls", None)

            if role == "HumanMessage":
                print(f"\n[User]: {content}")
            elif role == "AIMessage":
                print(f"\n[Agent]: {content}")
                if tool_calls:
                    for tc in tool_calls:
                        print(f"  → Tool Call: {tc.get('name')}({tc.get('args')})")
            elif role == "ToolMessage":
                print(f"\n[Observation]:\n{content}")

    except Exception as exc:
        print(f"\nError executing agent: {exc}")
        sys.exit(1)


if __name__ == "__main__":
    main()
