"""LangGraph Agent Loop for Phase 6 Autonomous Testing, Recovery & Retry Agent."""

import os
import sys
from typing import Literal
from dotenv import load_dotenv

from langchain_core.messages import (
    HumanMessage,
    AIMessage,
    SystemMessage,
    BaseMessage,
    ToolMessage,
)
from langchain_core.language_models import BaseChatModel
from langchain_openai import ChatOpenAI
from langgraph.graph import StateGraph, START, END
from langgraph.prebuilt import ToolNode

from app.state import (
    AgentState,
    create_plan_state,
    update_task_state,
    revise_plan_state,
    ValidationResult,
    VerificationResult,
)
from app.tools import create_workspace_tools
from app.memory import save_state, load_state, delete_state


SYSTEM_PROMPT = (
    "You are an autonomous software engineering assistant equipped with repository inspection, "
    "context retrieval, dynamic planning, safe code editing, automated testing, goal verification, "
    "and safe Git/GitHub delivery tools.\n\n"
    "When given a software engineering goal or bug fix task:\n"
    "1. Decompose goals into structured subtasks using `create_plan` with explicit dependencies.\n"
    "2. Use `retrieve_hybrid_context(query)`, `retrieve_relevant_context(query)`, or `read_file(file_path)` to locate relevant code context.\n"
    "3. Apply code modifications using `replace_in_file(file_path, old_text, new_text)` or `write_file`.\n"
    "4. Inspect repository changes with `git_diff()` or `git_status()`.\n"
    "5. IMPERATIVE VALIDATION & SELF-CORRECTION LOOP:\n"
    "   - Execute validation tests using `run_tests()`.\n"
    "   - If tests fail (`Status: failed`), observe error tracebacks/assertions, diagnose the root cause, "
    "apply a corrective fix with `replace_in_file`, inspect `git_diff()`, and re-run `run_tests()`.\n"
    "6. AUTONOMOUS GOAL VERIFICATION:\n"
    "   - Passing tests (`run_tests`) ALONE does not mean the user's goal is complete.\n"
    "   - Inspect repository evidence (e.g. read_file, search_code, git_diff) to check whether the original "
    "user goal has actually been satisfied.\n"
    "   - Invoke `verify_goal(status, summary, evidence)` where status is 'passed', 'failed', or 'uncertain'.\n"
    "7. SAFE GIT DELIVERY & HUMAN APPROVAL:\n"
    "   - For externally impactful actions (commit, push, pull_request), you MUST request human approval first "
    "using `request_human_approval(action, reason, risk)`.\n"
    "   - Only execute `git_commit`, `git_push`, or `create_pull_request` after human approval is confirmed.\n"
    "8. Conclude with a comprehensive response when all tasks pass validation, verification passes, and delivery is complete."
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
    """Helper to update state plan, retrieved context, modified files, validation, goal verification, and Git approval state."""
    messages = state.get("messages", [])
    plan = state.get("plan")
    user_goal = state.get("user_goal", "")
    task_id = state.get("task_id", "")
    status_val = state.get("status", "running")
    retrieved_context = list(state.get("retrieved_context") or [])
    modified_files = list(state.get("modified_files") or [])
    validation_result = state.get("validation_result")
    verification_result = state.get("verification_result")
    existing_retry_count = int(state.get("retry_count", 0))
    failed_tool_messages = 0
    max_retries = state.get("max_retries", 3)

    git_status = state.get("git_status")
    git_diff = state.get("git_diff")
    current_branch = state.get("current_branch")
    target_branch = state.get("target_branch")
    delivery_action = state.get("delivery_action")
    approval_required = bool(state.get("approval_required", False))
    approval_status = state.get("approval_status", "not_required")
    approval_reason = state.get("approval_reason")
    commit_message = state.get("commit_message")
    commit_created = bool(state.get("commit_created", False))
    push_requested = bool(state.get("push_requested", False))
    pr_requested = bool(state.get("pr_requested", False))

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
                elif name in ("retrieve_relevant_context", "retrieve_hybrid_context") or (name and name.startswith("retrieve_")):
                    query = args.get("query", "")
                    if query and query not in [r.get("query") for r in retrieved_context]:
                        retrieved_context.append({"query": query})
                elif name in ("write_file", "replace_in_file"):
                    fp = args.get("file_path", "")
                    if fp and fp not in modified_files:
                        modified_files.append(fp)
                elif name == "git_create_branch":
                    b = args.get("branch_name", "")
                    if b:
                        current_branch = b
                elif name == "request_human_approval":
                    delivery_action = args.get("action", "commit")
                    approval_reason = args.get("reason", "")
                    if approval_status not in ("approved", "rejected"):
                        approval_required = True
                        approval_status = "pending"
                elif name == "git_commit":
                    cm = args.get("message", "")
                    if cm:
                        commit_message = cm
                elif name == "git_push":
                    push_requested = True
                elif name == "create_pull_request":
                    pr_requested = True
                    tb = args.get("base_branch", "main")
                    if tb:
                        target_branch = tb

        elif isinstance(msg, ToolMessage) or msg.__class__.__name__ == "ToolMessage":
            content = getattr(msg, "content", "")
            if "=== Test Execution Result ===" in content:
                status = "passed" if "Status: passed" in content else ("failed" if "Status: failed" in content else ("timeout" if "Status: timeout" in content else "error"))
                summary_line = [line for line in content.splitlines() if line.startswith("Summary:")]
                summary = summary_line[0].replace("Summary:", "").strip() if summary_line else ""

                validation_result = ValidationResult(
                    status=status,
                    exit_code=0 if status == "passed" else (1 if status == "failed" else None),
                    summary=summary,
                    output=content,
                )
                if status != "passed":
                    failed_tool_messages += 1

            elif "=== Goal Verification Result ===" in content:
                status = "passed" if "Status: passed" in content else ("failed" if "Status: failed" in content else "uncertain")
                summary_line = [line for line in content.splitlines() if line.startswith("Summary:")]
                summary = summary_line[0].replace("Summary:", "").strip() if summary_line else ""

                evidence_lines = []
                in_evidence = False
                for line in content.splitlines():
                    if line.startswith("Evidence:"):
                        in_evidence = True
                        continue
                    if in_evidence and line.startswith("  •"):
                        evidence_lines.append(line.replace("  •", "").strip())

                verification_result = VerificationResult(
                    status=status,
                    summary=summary,
                    evidence=evidence_lines,
                )
                if status != "passed":
                    failed_tool_messages += 1

            elif "=== Human Approval Request ===" in content:
                if approval_status not in ("approved", "rejected"):
                    approval_required = True
                    approval_status = "pending"

            elif "Successfully committed changes" in content:
                commit_created = True

    retry_count = max(existing_retry_count, failed_tool_messages)

    updated_state = dict(state)
    updated_state["plan"] = plan
    updated_state["retrieved_context"] = retrieved_context
    updated_state["modified_files"] = modified_files
    updated_state["validation_result"] = validation_result
    updated_state["verification_result"] = verification_result
    updated_state["retry_count"] = retry_count
    updated_state["max_retries"] = max_retries
    updated_state["git_status"] = git_status
    updated_state["git_diff"] = git_diff
    updated_state["current_branch"] = current_branch
    updated_state["target_branch"] = target_branch
    updated_state["delivery_action"] = delivery_action
    updated_state["approval_required"] = approval_required
    updated_state["approval_status"] = approval_status
    updated_state["approval_reason"] = approval_reason
    updated_state["commit_message"] = commit_message
    updated_state["commit_created"] = commit_created
    updated_state["push_requested"] = push_requested
    updated_state["pr_requested"] = pr_requested

    if task_id:
        updated_state["task_id"] = task_id
    updated_state["status"] = status_val

    from app.state import set_active_approval_status
    ws = updated_state.get("workspace_root", ".")
    set_active_approval_status(ws, approval_status)

    return updated_state  # type: ignore


def build_agent_graph(llm: BaseChatModel | None = None, workspace_root: str = "."):
    """Constructs and compiles the Phase 8 persistent agent execution graph.

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
        """Reasoning node that invokes the model with accumulated state and checkpoints persistent memory."""
        state = sync_plan_from_messages(state)
        messages = list(state.get("messages", []))
        user_goal = state.get("user_goal", "")
        task_id = state.get("task_id")
        storage_dir = getattr(state, "storage_dir", ".agent_memory")

        if task_id:
            try:
                save_state(task_id, state, status=state.get("status", "running"), storage_dir=storage_dir)
            except Exception:
                pass

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
        if state.get("validation_result"):
            res_dict["validation_result"] = state["validation_result"]
        if state.get("verification_result"):
            res_dict["verification_result"] = state["verification_result"]
        res_dict["retry_count"] = state.get("retry_count", 0)
        if state.get("task_id"):
            res_dict["task_id"] = state["task_id"]
        if state.get("status"):
            res_dict["status"] = state["status"]

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


def run_agent(
    goal: str = "",
    workspace_root: str = ".",
    llm: BaseChatModel | None = None,
    task_id: str | None = None,
    resume: bool = False,
    storage_dir: str = ".agent_memory",
    mode: Literal["single_agent", "multi_agent"] = "single_agent",
) -> dict:
    """Executes or resumes the agent loop for a software engineering goal with persistent memory.

    Args:
        goal: High-level goal string for the agent (required for new tasks).
        workspace_root: Directory path restricting tool execution.
        llm: Optional chat model instance.
        task_id: Optional unique task identifier for persistence and resume.
        resume: Set to True to resume an existing task from storage_dir/<task_id>.json.
        storage_dir: Directory path for persistent JSON memory storage.
        mode: Execution mode ('single_agent' default or 'multi_agent').

    Returns:
        Final state dictionary containing conversation history, plan, validation, verification results, and task_id.
    """
    import time
    from app.evaluation import ExecutionTrace, generate_evaluation_report

    t_start = time.time()
    trace = ExecutionTrace(task_id=task_id if task_id else "task_default")
    trace.record_event(event_type="agent_start", agent_role="single_agent", status="started")

    if mode == "multi_agent" and not resume:
        from app.multi_agent import run_multi_agent
        return run_multi_agent(
            goal=goal,
            workspace_root=workspace_root,
            llm=llm,
            task_id=task_id,
            storage_dir=storage_dir,
        )
    if resume:
        if not task_id:
            raise ValueError("Error: task_id must be provided when resume=True.")
        initial_state = load_state(task_id, storage_dir=storage_dir)
        if workspace_root != ".":
            initial_state["workspace_root"] = workspace_root
        existing_trace = initial_state.get("execution_trace") or []
        for ev in trace.get_events():
            existing_trace.append(ev)
        initial_state["execution_trace"] = existing_trace
    else:
        if not task_id:
            import uuid
            task_id = f"task_{uuid.uuid4().hex[:8]}"

        if not goal:
            raise ValueError("Error: Goal must be provided for new agent task execution.")

        initial_state = {
            "task_id": task_id,
            "status": "running",
            "mode": "single_agent",
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
            "execution_trace": trace.get_events(),
            "evaluation_report": None,
        }

    from app.state import set_active_approval_status
    set_active_approval_status(workspace_root, initial_state.get("approval_status", "pending"))

    graph = build_agent_graph(llm=llm, workspace_root=workspace_root)
    final_state = graph.invoke(initial_state)
    synced_state = sync_plan_from_messages(final_state)

    ver_res = synced_state.get("verification_result")
    val_res = synced_state.get("validation_result")
    app_req = bool(synced_state.get("approval_required", False))
    app_st = synced_state.get("approval_status", "not_required")

    if app_req and app_st == "pending":
        synced_state["status"] = "paused"
    elif ver_res and ver_res.get("status") == "passed":
        synced_state["status"] = "completed"
    elif ver_res and ver_res.get("status") == "failed":
        synced_state["status"] = "failed"
    elif val_res and val_res.get("status") in ("failed", "error") and synced_state.get("retry_count", 0) >= synced_state.get("max_retries", 3):
        synced_state["status"] = "failed"
    else:
        synced_state["status"] = "completed"

    t_end = time.time()
    current_trace = synced_state.get("execution_trace") or []
    current_trace.append({
        "timestamp": round(t_end, 3),
        "event_type": "agent_end",
        "agent_role": "single_agent",
        "status": synced_state.get("status", "completed"),
        "duration": round(t_end - t_start, 3),
        "metadata": None,
    })
    synced_state["execution_trace"] = current_trace

    report = generate_evaluation_report(synced_state, start_time=t_start, end_time=t_end)
    synced_state["evaluation_report"] = report
    synced_state["final_outcome"] = report.get("final_outcome", "FAILED")

    if task_id:
        try:
            save_state(task_id, synced_state, status=synced_state.get("status", "completed"), storage_dir=storage_dir)
        except Exception:
            pass

    return synced_state


def resume_agent(
    task_id: str,
    workspace_root: str = ".",
    llm: BaseChatModel | None = None,
    storage_dir: str = ".agent_memory",
) -> dict:
    """Resumes an existing agent task from persistent memory using its task_id."""
    return run_agent(
        goal="",
        workspace_root=workspace_root,
        llm=llm,
        task_id=task_id,
        resume=True,
        storage_dir=storage_dir,
    )


def approve_task(
    task_id: str,
    decision: str,
    notes: str = "",
    workspace_root: str = ".",
    llm: BaseChatModel | None = None,
    storage_dir: str = ".agent_memory",
) -> dict:
    """Processes a human approval decision for a persistent task and resumes if approved."""
    from app.tools import process_human_approval

    initial_state = load_state(task_id, storage_dir=storage_dir)
    updated_state = process_human_approval(initial_state, decision=decision, notes=notes)

    if updated_state.get("approval_status") == "rejected":
        updated_state["status"] = "paused"
        save_state(task_id, updated_state, status="paused", storage_dir=storage_dir)
        return updated_state

    save_state(task_id, updated_state, status="running", storage_dir=storage_dir)
    return resume_agent(
        task_id=task_id,
        workspace_root=workspace_root,
        llm=llm,
        storage_dir=storage_dir,
    )



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
        task_id = final_state.get("task_id", "")
        exec_status = final_state.get("status", "completed")
        messages = final_state.get("messages", [])
        plan = final_state.get("plan")
        modified_files = final_state.get("modified_files", [])
        val_result = final_state.get("validation_result")
        ver_result = final_state.get("verification_result")
        retry_count = final_state.get("retry_count", 0)

        print(f"Task ID: {task_id}")
        print(f"Execution Status: {exec_status}")

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

        if val_result:
            print("\n=== Validation Result ===")
            print(f"Status: {val_result.get('status')}")
            print(f"Summary: {val_result.get('summary')}")
            print(f"Recovery Retries Performed: {retry_count}")

        if ver_result:
            print("\n=== Goal Verification Result ===")
            print(f"Status: {ver_result.get('status')}")
            print(f"Summary: {ver_result.get('summary')}")
            if ver_result.get("evidence"):
                print("Evidence:")
                for ev in ver_result.get("evidence", []):
                    print(f"  • {ev}")

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
