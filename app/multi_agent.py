"""Multi-agent architecture and specialized role coordination for Phase 12 Autonomous Coding Agent."""

import time
from typing import Any, Literal
from langchain_core.messages import (
    AIMessage,
    BaseMessage,
    HumanMessage,
    SystemMessage,
    ToolMessage,
)
from langchain_core.language_models import BaseChatModel
from langgraph.graph import StateGraph, START, END

from app.state import (
    AgentState,
    AgentRole,
    AnalysisResult,
    CodingResult,
    ReviewResult,
    ReviewStatus,
    set_active_approval_status,
)
from app.tools import create_workspace_tools
from app.memory import save_state, load_state


def get_analyzer_tools(workspace_root: str = ".") -> list:
    """Returns the restricted read-only analysis tool subset for the Analyzer agent role."""
    all_tools = create_workspace_tools(workspace_root=workspace_root)
    allowed_names = {
        "list_files",
        "read_file",
        "search_code",
        "git_status",
        "git_diff",
        "retrieve_relevant_context",
        "retrieve_hybrid_context",
        "create_plan",
        "update_task_status",
        "revise_plan",
    }
    return [t for t in all_tools if t.name in allowed_names]


def get_coder_tools(workspace_root: str = ".") -> list:
    """Returns the code editing and testing tool subset for the Coder agent role."""
    all_tools = create_workspace_tools(workspace_root=workspace_root)
    allowed_names = {
        "list_files",
        "read_file",
        "search_code",
        "git_status",
        "git_diff",
        "retrieve_relevant_context",
        "retrieve_hybrid_context",
        "write_file",
        "replace_in_file",
        "run_tests",
        "create_plan",
        "update_task_status",
        "revise_plan",
    }
    return [t for t in all_tools if t.name in allowed_names]


def get_reviewer_tools(workspace_root: str = ".") -> list:
    """Returns the inspection and test verification tool subset for the Reviewer agent role."""
    all_tools = create_workspace_tools(workspace_root=workspace_root)
    allowed_names = {
        "list_files",
        "read_file",
        "search_code",
        "git_status",
        "git_diff",
        "run_tests",
        "verify_goal",
    }
    return [t for t in all_tools if t.name in allowed_names]


def _execute_agent_subloop(
    llm: BaseChatModel,
    tools: list,
    system_prompt: str,
    user_prompt: str,
    max_steps: int = 10,
) -> tuple[str, list[BaseMessage]]:
    """Executes a bounded reasoning and tool invocation loop for a specialized agent role."""
    tool_map = {t.name: t for t in tools}
    llm_with_tools = llm.bind_tools(tools) if tools else llm

    messages: list[BaseMessage] = [
        SystemMessage(content=system_prompt),
        HumanMessage(content=user_prompt),
    ]

    for step in range(max_steps):
        if hasattr(llm_with_tools, "call_count") and hasattr(llm, "call_count"):
            setattr(llm_with_tools, "call_count", getattr(llm, "call_count"))

        response = llm_with_tools.invoke(messages)

        if hasattr(llm_with_tools, "call_count") and hasattr(llm, "call_count"):
            setattr(llm, "call_count", getattr(llm_with_tools, "call_count"))

        messages.append(response)

        tool_calls = getattr(response, "tool_calls", None)
        if not tool_calls:
            final_content = getattr(response, "content", "")
            return str(final_content), messages

        for tc in tool_calls:
            t_name = tc.get("name")
            t_args = tc.get("args", {})
            call_id = tc.get("id", f"call_{t_name}")

            if t_name in tool_map:
                try:
                    res = tool_map[t_name].invoke(t_args)
                    obs = str(res)
                except Exception as exc:
                    obs = f"Error executing tool '{t_name}': {exc}"
            else:
                obs = f"Error: Tool '{t_name}' is not permitted for this role."

            messages.append(ToolMessage(content=obs, tool_call_id=call_id, name=t_name))

    last_content = getattr(messages[-1], "content", "")
    return str(last_content), messages


def analyzer_node(state: AgentState, llm: BaseChatModel, workspace_root: str = ".") -> dict:
    """Analyzer agent node: inspects repository structure and designs an architectural implementation plan."""
    tools = get_analyzer_tools(workspace_root=workspace_root)
    user_goal = state.get("user_goal", "")

    system_prompt = (
        "You are the Analyzer Agent in a multi-agent software engineering framework.\n"
        "Your task is to analyze the codebase and create a precise, structured architectural plan for the requested goal.\n"
        "Inspect relevant files using tools, identify risk areas, and output your analysis.\n"
        "You MUST NOT edit files directly."
    )

    user_prompt = f"Goal: {user_goal}\nInspect the repository and provide a summary of affected files, recommended approach, and risks."

    final_text, history = _execute_agent_subloop(
        llm=llm,
        tools=tools,
        system_prompt=system_prompt,
        user_prompt=user_prompt,
    )

    # Extract affected files if mentioned
    affected_files = list(state.get("modified_files") or [])
    analysis_res: AnalysisResult = {
        "summary": final_text,
        "affected_files": affected_files,
        "recommended_approach": final_text,
        "risk_assessment": "Low risk analysis complete",
    }

    return {
        "agent_role": "analyzer",
        "analysis_result": analysis_res,
        "messages": [AIMessage(content=f"[Analyzer Node]: {final_text}")],
    }


def coder_node(state: AgentState, llm: BaseChatModel, workspace_root: str = ".") -> dict:
    """Coder agent node: applies targeted code modifications and runs local tests."""
    tools = get_coder_tools(workspace_root=workspace_root)
    user_goal = state.get("user_goal", "")
    analysis_res = state.get("analysis_result", {})
    analysis_summary = analysis_res.get("summary", "") if isinstance(analysis_res, dict) else ""
    review_feedback = state.get("review_feedback")

    system_prompt = (
        "You are the Coder Agent in a multi-agent software engineering framework.\n"
        "Your task is to write and modify code files using `replace_in_file` or `write_file` to accomplish the goal.\n"
        "Always execute `run_tests` to verify your changes before finishing.\n"
        "If review feedback is provided, you MUST explicitly fix the issues highlighted."
    )

    prompt_lines = [f"Goal: {user_goal}"]
    if analysis_summary:
        prompt_lines.append(f"Analyzer Recommendations:\n{analysis_summary}")
    if review_feedback:
        prompt_lines.append(f"CRITICAL REVIEW FEEDBACK TO FIX:\n{review_feedback}")

    user_prompt = "\n\n".join(prompt_lines)

    final_text, history = _execute_agent_subloop(
        llm=llm,
        tools=tools,
        system_prompt=system_prompt,
        user_prompt=user_prompt,
    )

    # Collect modified files from tool messages
    modified = set(state.get("modified_files") or [])
    for msg in history:
        if isinstance(msg, AIMessage) and getattr(msg, "tool_calls", None):
            for tc in msg.tool_calls:
                if tc.get("name") in ("write_file", "replace_in_file"):
                    fp = tc.get("args", {}).get("file_path")
                    if fp:
                        modified.add(fp)

    coding_res: CodingResult = {
        "summary": final_text,
        "modified_files": list(modified),
        "test_results": state.get("validation_result"),
    }

    return {
        "agent_role": "coder",
        "coding_result": coding_res,
        "modified_files": list(modified),
        "messages": [AIMessage(content=f"[Coder Node]: {final_text}")],
    }


def reviewer_node(state: AgentState, llm: BaseChatModel, workspace_root: str = ".") -> dict:
    """Reviewer agent node: evaluates code diffs and test results, issuing review decisions."""
    tools = get_reviewer_tools(workspace_root=workspace_root)
    user_goal = state.get("user_goal", "")
    coding_res = state.get("coding_result", {})
    coding_summary = coding_res.get("summary", "") if isinstance(coding_res, dict) else ""
    modified_files = state.get("modified_files", [])

    system_prompt = (
        "You are the Reviewer Agent in a multi-agent software engineering framework.\n"
        "Your task is to inspect code modifications (`git_diff`, `read_file`) and test outputs (`run_tests`).\n"
        "Evaluate whether the code correctly and safely meets the goal.\n"
        "You MUST conclude with a clear line:\n"
        "STATUS: APPROVED\n"
        "or\n"
        "STATUS: CHANGES_REQUESTED\n"
        "or\n"
        "STATUS: BLOCKED\n"
        "Followed by FEEDBACK: <detailed explanation>."
    )

    user_prompt = (
        f"Goal: {user_goal}\n"
        f"Modified Files: {modified_files}\n"
        f"Coder Summary: {coding_summary}\n"
        "Inspect the repository diff and test status. Provide your review decision."
    )

    final_text, history = _execute_agent_subloop(
        llm=llm,
        tools=tools,
        system_prompt=system_prompt,
        user_prompt=user_prompt,
    )

    status_val: ReviewStatus = "approved"
    upper_text = final_text.upper()
    if "STATUS: CHANGES_REQUESTED" in upper_text or "CHANGES_REQUESTED" in upper_text:
        status_val = "changes_requested"
    elif "STATUS: BLOCKED" in upper_text or "BLOCKED" in upper_text:
        status_val = "blocked"
    elif "STATUS: APPROVED" in upper_text or "APPROVED" in upper_text:
        status_val = "approved"
    else:
        # Fallback check on validation result
        val_res = state.get("validation_result")
        if val_res and val_res.get("status") == "failed":
            status_val = "changes_requested"
        else:
            status_val = "approved"

    review_res: ReviewResult = {
        "status": status_val,
        "feedback": final_text,
        "score": 1.0 if status_val == "approved" else 0.0,
        "issues": [final_text] if status_val != "approved" else [],
    }

    return {
        "agent_role": "reviewer",
        "review_result": review_res,
        "review_status": status_val,
        "review_feedback": final_text,
        "messages": [AIMessage(content=f"[Reviewer Node]: Decision = {status_val.upper()}\n{final_text}")],
    }


def orchestrator_node(state: AgentState) -> dict:
    """Orchestrator node: manages multi-agent iteration state and flow control."""
    current_iter = int(state.get("multi_agent_iteration", 0)) + 1
    return {
        "agent_role": "orchestrator",
        "multi_agent_iteration": current_iter,
    }


def route_multi_agent(state: AgentState) -> Literal["coder", "end"]:
    """Conditional routing logic for the multi-agent orchestration loop."""
    review_status = state.get("review_status", "pending")
    iteration = int(state.get("multi_agent_iteration", 0))
    max_iters = int(state.get("max_multi_agent_iterations", 3))

    if review_status == "approved":
        return "end"

    if review_status == "blocked":
        return "end"

    if iteration >= max_iters:
        return "end"

    if review_status == "changes_requested":
        return "coder"

    return "end"


def build_multi_agent_graph(llm: BaseChatModel, workspace_root: str = "."):
    """Constructs and compiles the Phase 12 specialized multi-agent orchestration graph."""
    workflow = StateGraph(AgentState)

    def _analyzer_wrapper(state: AgentState) -> dict:
        return analyzer_node(state, llm=llm, workspace_root=workspace_root)

    def _coder_wrapper(state: AgentState) -> dict:
        return coder_node(state, llm=llm, workspace_root=workspace_root)

    def _reviewer_wrapper(state: AgentState) -> dict:
        return reviewer_node(state, llm=llm, workspace_root=workspace_root)

    def _orchestrator_wrapper(state: AgentState) -> dict:
        return orchestrator_node(state)

    workflow.add_node("analyzer", _analyzer_wrapper)
    workflow.add_node("coder", _coder_wrapper)
    workflow.add_node("reviewer", _reviewer_wrapper)
    workflow.add_node("orchestrator", _orchestrator_wrapper)

    workflow.add_edge(START, "analyzer")
    workflow.add_edge("analyzer", "coder")
    workflow.add_edge("coder", "reviewer")
    workflow.add_edge("reviewer", "orchestrator")

    workflow.add_conditional_edges(
        "orchestrator",
        route_multi_agent,
        {
            "coder": "coder",
            "end": END,
        },
    )

    return workflow.compile()


def run_multi_agent(
    goal: str,
    workspace_root: str = ".",
    llm: BaseChatModel | None = None,
    task_id: str | None = None,
    max_iterations: int = 3,
    storage_dir: str = ".agent_memory",
) -> dict:
    """Top-level entrypoint for executing a goal using the Phase 12 multi-agent workflow."""
    if not goal:
        raise ValueError("Error: Goal must be provided for multi-agent execution.")

    if not task_id:
        import uuid
        task_id = f"multi_task_{uuid.uuid4().hex[:8]}"

    if llm is None:
        from app.agent import get_default_llm
        llm = get_default_llm()

    import time
    from app.evaluation import ExecutionTrace, generate_evaluation_report

    t_start = time.time()
    trace = ExecutionTrace(task_id=task_id)
    trace.record_event(event_type="agent_start", agent_role="orchestrator", status="started")

    initial_state: AgentState = {
        "task_id": task_id,
        "status": "running",
        "mode": "multi_agent",
        "user_goal": goal,
        "workspace_root": workspace_root,
        "messages": [HumanMessage(content=goal)],
        "plan": None,
        "retrieved_context": [],
        "modified_files": [],
        "validation_result": None,
        "verification_result": None,
        "retry_count": 0,
        "max_retries": 3,
        "agent_role": "analyzer",
        "analysis_result": None,
        "coding_result": None,
        "review_result": None,
        "review_status": "pending",
        "review_feedback": None,
        "multi_agent_iteration": 0,
        "max_multi_agent_iterations": max_iterations,
        "execution_trace": trace.get_events(),
        "evaluation_report": None,
    }

    set_active_approval_status(workspace_root, "not_required")

    graph = build_multi_agent_graph(llm=llm, workspace_root=workspace_root)
    final_state = graph.invoke(initial_state)

    rev_status = final_state.get("review_status")
    if rev_status == "approved":
        final_state["status"] = "completed"
    else:
        final_state["status"] = "failed"

    t_end = time.time()
    current_trace = final_state.get("execution_trace") or []
    current_trace.append({
        "timestamp": round(t_end, 3),
        "event_type": "agent_end",
        "agent_role": "orchestrator",
        "status": final_state.get("status", "completed"),
        "duration": round(t_end - t_start, 3),
        "metadata": None,
    })
    final_state["execution_trace"] = current_trace

    report = generate_evaluation_report(final_state, start_time=t_start, end_time=t_end)
    final_state["evaluation_report"] = report

    if task_id:
        try:
            save_state(task_id, final_state, status=final_state.get("status", "completed"), storage_dir=storage_dir)
        except Exception:
            pass

    return final_state


class MultiAgentEvaluator:
    """Benchmark evaluator comparing Single-Agent vs Multi-Agent performance."""

    def __init__(self, workspace_root: str = "."):
        self.workspace_root = workspace_root

    def evaluate_comparison(
        self,
        goal: str,
        llm: BaseChatModel,
        max_iterations: int = 3,
    ) -> dict[str, Any]:
        """Runs a task goal under both single-agent and multi-agent modes and returns performance metrics."""
        from app.agent import run_agent

        # Single agent run
        t0 = time.time()
        single_res = run_agent(
            goal=goal,
            workspace_root=self.workspace_root,
            llm=llm,
            mode="single_agent",
        )
        single_time = round(time.time() - t0, 3)

        # Multi agent run
        t1 = time.time()
        multi_res = run_multi_agent(
            goal=goal,
            workspace_root=self.workspace_root,
            llm=llm,
            max_iterations=max_iterations,
        )
        multi_time = round(time.time() - t1, 3)

        single_mods = len(single_res.get("modified_files") or [])
        multi_mods = len(multi_res.get("modified_files") or [])

        single_status = single_res.get("status", "unknown")
        multi_status = multi_res.get("status", "unknown")
        review_status = multi_res.get("review_status", "none")

        return {
            "goal": goal,
            "single_agent": {
                "execution_time_seconds": single_time,
                "modified_files_count": single_mods,
                "status": single_status,
            },
            "multi_agent": {
                "execution_time_seconds": multi_time,
                "modified_files_count": multi_mods,
                "status": multi_status,
                "review_status": review_status,
                "iterations": multi_res.get("multi_agent_iteration", 0),
            },
            "winner": "multi_agent" if review_status == "approved" else "single_agent",
        }
