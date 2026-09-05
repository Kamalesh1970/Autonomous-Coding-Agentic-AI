"""LangGraph Agent Loop for Phase 5 Code Modification Agent."""

import os
import sys
from typing import Literal
from dotenv import load_dotenv

from langchain_core.messages import HumanMessage, AIMessage, SystemMessage, BaseMessage
from langchain_core.language_models import BaseChatModel
from langchain_openai import ChatOpenAI
from langgraph.graph import StateGraph, START, END
from langgraph.prebuilt import ToolNode

from app.state import (
    AgentState,
    create_plan_state,
    update_task_state,
    revise_plan_state,
)
from app.tools import create_workspace_tools


SYSTEM_PROMPT = (
    "You are an autonomous software engineering assistant equipped with repository inspection, "
    "structured context retrieval, dynamic planning, and safe code editing tools.\n\n"
    "When given a software engineering goal:\n"
    "1. Decompose complex goals into structured subtasks using `create_plan` with explicit dependencies.\n"
    "2. For specific subtasks, use `retrieve_relevant_context(query)` to identify relevant code context.\n"
    "3. Use `read_file(file_path)` for targeted deep dives into specific files.\n"
    "4. When modifying repository code, use `replace_in_file(file_path, old_text, new_text)` for targeted "
    "replacements, or `write_file(file_path, content)` to create/overwrite files.\n"
    "5. IMPERATIVE: After applying code modifications, call `git_diff()` to inspect and evaluate the "
    "actual repository changes before marking tasks as complete.\n"
    "6. Track progress using `update_task_status(task_id, status)` as tasks complete.\n"
    "7. Revise your plan using `revise_plan(new_tasks, reason)` if repository evidence alters strategy.\n"
    "8. Conclude with a comprehensive context-aware response when all tasks are complete."
)


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


def sync_plan_from_messages(state: AgentState) -> AgentState:
    """Helper to update state['plan'], state['retrieved_context'], and state['modified_files']."""
    messages = state.get("messages", [])
    plan = state.get("plan")
    user_goal = state.get("user_goal", "")
    retrieved_context = list(state.get("retrieved_context") or [])
    modified_files = list(state.get("modified_files") or [])

    for msg in messages:
        if isinstance(msg, AIMessage) and getattr(msg, "tool_calls", None):
            for tc in msg.tool_calls:
                name = tc.get("name")
                args = tc.get("args", {})
                if name == "create_plan":
                    raw_tasks = args.get("tasks", [])
                    plan = create_plan_state(user_goal, raw_tasks)
                elif name == "update_task_status":
                    if plan:
                        tid = args.get("task_id", "")
                        st = args.get("status", "pending")
                        plan = update_task_state(plan, tid, st)
                elif name == "revise_plan":
                    if plan:
                        raw_tasks = args.get("new_tasks", [])
                        reason = args.get("reason", "Plan revised")
                        plan = revise_plan_state(plan, raw_tasks, reason)
                elif name == "retrieve_relevant_context":
                    query = args.get("query", "")
                    if query and query not in [r.get("query") for r in retrieved_context]:
                        retrieved_context.append({"query": query})
                elif name in ("write_file", "replace_in_file"):
                    fp = args.get("file_path", "")
                    if fp and fp not in modified_files:
                        modified_files.append(fp)

    updated_state = dict(state)
    updated_state["plan"] = plan
    updated_state["retrieved_context"] = retrieved_context
    updated_state["modified_files"] = modified_files
    return updated_state  # type: ignore


def build_agent_graph(llm: BaseChatModel | None = None, workspace_root: str = "."):
    """Constructs and compiles the Phase 5 code modification agent loop.

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
        """Reasoning node that invokes the model with accumulated messages and context."""
        state = sync_plan_from_messages(state)
        messages = list(state.get("messages", []))
        user_goal = state.get("user_goal", "")

        input_messages = []
        if not any(isinstance(m, SystemMessage) for m in messages):
            input_messages.append(SystemMessage(content=SYSTEM_PROMPT))

        if not messages and user_goal:
            input_messages.append(HumanMessage(content=user_goal))
        else:
            input_messages.extend(messages)

        response = llm_with_tools.invoke(input_messages)

        res_dict = {"messages": [response]}
        if state.get("plan"):
            res_dict["plan"] = state["plan"]
        if state.get("retrieved_context"):
            res_dict["retrieved_context"] = state["retrieved_context"]
        if state.get("modified_files"):
            res_dict["modified_files"] = state["modified_files"]

        if not state.get("messages") and user_goal:
            res_dict["messages"] = [HumanMessage(content=user_goal), response]

        return res_dict

    tool_node = ToolNode(tools)

    def route_after_reason(state: AgentState) -> Literal["tools", "__end__"]:
        """Conditional edge checking if the model requested tool execution."""
        messages = state.get("messages", [])
        if not messages:
            return END

        last_message = messages[-1]
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
        Final state dictionary containing conversation history, plan, and modified files.
    """
    graph = build_agent_graph(llm=llm, workspace_root=workspace_root)
    initial_state = {
        "user_goal": goal,
        "workspace_root": workspace_root,
        "messages": [HumanMessage(content=goal)],
        "plan": None,
        "retrieved_context": [],
        "modified_files": [],
    }

    final_state = graph.invoke(initial_state)
    return sync_plan_from_messages(final_state)


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
        plan = final_state.get("plan")
        modified_files = final_state.get("modified_files", [])

        print("\n=== Agent Trace ===")
        for msg in messages:
            role = msg.__class__.__name__
            content = getattr(msg, "content", "")
            tool_calls = getattr(msg, "tool_calls", None)

            if role == "HumanMessage":
                print(f"\n[User Goal]: {content}")
            elif role == "AIMessage":
                print(f"\n[Agent]: {content}")
                if tool_calls:
                    for tc in tool_calls:
                        print(f"  → Tool Call: {tc.get('name')}({tc.get('args')})")
            elif role == "ToolMessage":
                print(f"\n[Observation]:\n{content}")

        if modified_files:
            print("\n=== Modified Files ===")
            for mf in modified_files:
                print(f"  • {mf}")

        if plan:
            print("\n=== Final Plan State ===")
            print(f"Goal: {plan.get('goal')}")
            print(f"Revision Count: {plan.get('revision_count', 0)}")
            if plan.get("revision_reason"):
                print(f"Revision Reason: {plan.get('revision_reason')}")
            print("Tasks:")
            for t in plan.get("tasks", []):
                deps = f" (deps: {t.get('dependencies')})" if t.get("dependencies") else ""
                print(f"  [{t.get('status').upper()}] {t.get('id')}: {t.get('title')}{deps}")

    except Exception as exc:
        print(f"\nError executing agent: {exc}")
        sys.exit(1)


if __name__ == "__main__":
    main()
