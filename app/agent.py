"""LangGraph Agent Loop for Phase 6 Autonomous Testing, Recovery & Retry Agent."""

import os
import sys
from typing import Any, Literal
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
from langchain_google_genai import ChatGoogleGenerativeAI
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


def is_retryable_error(error: Exception) -> bool:
    """Classify whether an exception is a retryable provider error."""
    err_str = str(error).lower()
    err_type = type(error).__name__.lower()

    # Non-retryable configuration, schema, or account billing errors (402 insufficient credits)
    non_retryable_patterns = [
        "404", "not_found", "no longer available", "invalid model",
        "402", "insufficient credits", "insufficient_credits", "payment_required", "upgrade to a paid account",
        "invalid_argument", "malformed request", "thought_signature"
    ]
    for pattern in non_retryable_patterns:
        if pattern in err_str:
            return False

    # Retryable provider errors (quota, rate limits, server errors, auth key failure)
    retryable_patterns = [
        "429", "quota", "rate limit", "ratelimit", "resource_exhausted", "resourceexhausted",
        "500", "502", "503", "504", "unavailable", "overloaded",
        "invalid_api_key", "unauthorized", "401", "403"
    ]
    for pattern in retryable_patterns:
        if pattern in err_str or pattern in err_type:
            return True

    if "400" in err_str and ("bad request" in err_str or "status" in err_str or "invalid" in err_str):
        return False

    return False


class GeminiChatOpenAI(ChatOpenAI):
    """ChatOpenAI variant tailored for Google Gemini's OpenAI-compatible endpoint that preserves additional_kwargs and satisfies schema rules (no null strings)."""

    def _get_request_payload(self, messages: list[BaseMessage], **kwargs: Any) -> dict[str, Any]:
        payload = super()._get_request_payload(messages, **kwargs)
        payload_messages = payload.get("messages", [])
        for orig_msg, msg_dict in zip(messages, payload_messages):
            if isinstance(orig_msg, AIMessage):
                sig = getattr(orig_msg, "additional_kwargs", {}).get("thought_signature") if hasattr(orig_msg, "additional_kwargs") and orig_msg.additional_kwargs else None
                if not sig and getattr(orig_msg, "response_metadata", None):
                    sig = orig_msg.response_metadata.get("thought_signature")
                if not sig and getattr(orig_msg, "tool_calls", None):
                    for tc in orig_msg.tool_calls:
                        if isinstance(tc, dict):
                            if tc.get("thought_signature"):
                                sig = tc.get("thought_signature")
                                break
                            elif isinstance(tc.get("args"), dict) and tc["args"].get("thought_signature"):
                                sig = tc["args"].get("thought_signature")
                                break

                if sig:
                    msg_dict["thought_signature"] = sig
                    if not getattr(orig_msg, "additional_kwargs", None):
                        orig_msg.additional_kwargs = {}
                    orig_msg.additional_kwargs["thought_signature"] = sig

                if getattr(orig_msg, "additional_kwargs", None):
                    for k, v in orig_msg.additional_kwargs.items():
                        if k not in msg_dict:
                            msg_dict[k] = v
                if orig_msg.tool_calls and "tool_calls" in msg_dict:
                    for orig_tc, dict_tc in zip(orig_msg.tool_calls, msg_dict["tool_calls"]):
                        if isinstance(orig_tc, dict):
                            for k, v in orig_tc.items():
                                if k not in ("name", "args", "id", "type"):
                                    dict_tc[k] = v
                                    if "function" in dict_tc and isinstance(dict_tc["function"], dict):
                                        dict_tc["function"][k] = v
                        if sig and isinstance(dict_tc, dict):
                            dict_tc["thought_signature"] = sig
                            if "function" in dict_tc and isinstance(dict_tc["function"], dict):
                                dict_tc["function"]["thought_signature"] = sig

        for msg in payload_messages:
            if msg.get("content") is None:
                msg["content"] = ""
            keys_to_remove = [k for k, v in msg.items() if v is None and k != "content"]
            for k in keys_to_remove:
                del msg[k]
            if "tool_calls" in msg and isinstance(msg["tool_calls"], list):
                for tc in msg["tool_calls"]:
                    if isinstance(tc, dict):
                        tc_none_keys = [k for k, v in tc.items() if v is None]
                        for k in tc_none_keys:
                            del tc[k]
                        if "function" in tc and isinstance(tc["function"], dict):
                            fn_none_keys = [k for k, v in tc["function"].items() if v is None]
                            for k in fn_none_keys:
                                del tc["function"][k]

        return payload


def sanitize_log_output(text: str) -> str:
    """Sanitizes text so sensitive API keys are never leaked in logs or error output."""
    if not text:
        return ""
    import re
    text = re.sub(r"AIzaSy[A-Za-z0-9_-]{10,}", "[REDACTED_API_KEY]", text)
    text = re.sub(r"sk-(proj-)?[A-Za-z0-9_-]{15,}", "[REDACTED_API_KEY]", text)
    return text


class FailoverChatModel(BaseChatModel):
    """ChatModel wrapper that manages key failover and provider fallback transparently."""

    candidates: list[Any]

    def __init__(self, candidates: list[Any]):
        super().__init__(candidates=candidates)

    def _generate(self, messages, stop=None, run_manager=None, **kwargs):
        last_error = None
        for i, candidate in enumerate(self.candidates):
            try:
                if hasattr(candidate, "_generate"):
                    return candidate._generate(messages, stop=stop, run_manager=run_manager, **kwargs)
                elif hasattr(candidate, "invoke"):
                    res = candidate.invoke(messages, config=run_manager, **kwargs)
                    from langchain_core.outputs import ChatGeneration, ChatResult
                    if isinstance(res, AIMessage):
                        return ChatResult(generations=[ChatGeneration(message=res)])
                    return res
            except Exception as e:
                last_error = e
                if is_retryable_error(e) and i < len(self.candidates) - 1:
                    clean_err = sanitize_log_output(str(e))
                    print(f"Provider attempt {i + 1} failed with retryable error ({type(e).__name__}: {clean_err}); trying next configured option.")
                    continue
                raise e
        if last_error:
            raise last_error
        raise RuntimeError("No candidate models configured for FailoverChatModel.")

    def invoke(self, input, config=None, **kwargs):
        last_error = None
        for i, candidate in enumerate(self.candidates):
            try:
                return candidate.invoke(input, config=config, **kwargs)
            except Exception as e:
                last_error = e
                if is_retryable_error(e) and i < len(self.candidates) - 1:
                    clean_err = sanitize_log_output(str(e))
                    print(f"Provider attempt {i + 1} failed with retryable error ({type(e).__name__}: {clean_err}); trying next configured option.")
                    continue
                raise e
        if last_error:
            raise last_error
        raise RuntimeError("No candidate models configured for FailoverChatModel.")

    def bind_tools(self, tools, **kwargs):
        bound_candidates = [c.bind_tools(tools, **kwargs) for c in self.candidates]
        return FailoverChatModel(candidates=bound_candidates)

    @property
    def _llm_type(self) -> str:
        return "failover_chat_model"



def get_default_llm() -> BaseChatModel:
    """Initialize the configured LLM using environment variables based on LLM_PROVIDER with failover support."""
    load_dotenv()
    raw_provider = os.getenv("LLM_PROVIDER", "openai").strip()
    provider = raw_provider.lower()

    if provider == "gemini":
        gemini_keys = []
        for key_name in ["GEMINI_API_KEY_1", "GEMINI_API_KEY_2", "GEMINI_API_KEY_3"]:
            val = os.getenv(key_name)
            if val and val != "your_gemini_api_key_here":
                gemini_keys.append(val)
        
        # Fallback to GEMINI_API_KEY if none of the numbered keys exist
        if not gemini_keys:
            val = os.getenv("GEMINI_API_KEY")
            if val and val != "your_gemini_api_key_here":
                gemini_keys.append(val)

        model_name = os.getenv("LLM_MODEL") or os.getenv("GEMINI_MODEL_NAME") or "gemini-3.6-flash"
        if "gemini-2.0" in model_name:
            model_name = "gemini-3.6-flash"

        import warnings
        warnings.filterwarnings("ignore", message=".*uses fixed sampling defaults.*")

        candidates = []
        for k in gemini_keys:
            candidates.append(
                ChatGoogleGenerativeAI(
                    model=model_name,
                    google_api_key=k,
                    max_retries=1,
                )
            )

        # OpenRouter fallback if configured
        openrouter_key = os.getenv("OPENROUTER_API_KEY")
        if openrouter_key and openrouter_key != "your_openrouter_api_key_here":
            openrouter_model = os.getenv("OPENROUTER_MODEL") or os.getenv("LLM_MODEL") or "openai/gpt-4o-mini"
            openrouter_max_tokens_val = os.getenv("OPENROUTER_MAX_TOKENS", "8192").strip()
            openrouter_max_tokens = int(openrouter_max_tokens_val) if openrouter_max_tokens_val.isdigit() else 8192

            extra_headers = {}
            site_url = os.getenv("OPENROUTER_SITE_URL")
            app_name = os.getenv("OPENROUTER_APP_NAME")
            if site_url:
                extra_headers["HTTP-Referer"] = site_url
            if app_name:
                extra_headers["X-Title"] = app_name

            candidates.append(
                ChatOpenAI(
                    model=openrouter_model,
                    api_key=openrouter_key,
                    base_url="https://openrouter.ai/api/v1",
                    default_headers=extra_headers if extra_headers else None,
                    max_tokens=openrouter_max_tokens,
                    temperature=0,
                )
            )

        if not candidates:
            raise ValueError("GEMINI_API_KEY (or GEMINI_API_KEY_1) is required when LLM_PROVIDER=gemini")

        if len(candidates) == 1:
            return candidates[0]
        return FailoverChatModel(candidates=candidates)

    elif provider == "openrouter":
        api_key = os.getenv("OPENROUTER_API_KEY")
        if not api_key or api_key == "your_openrouter_api_key_here":
            raise ValueError("OPENROUTER_API_KEY is required when LLM_PROVIDER=openrouter")
        model_name = os.getenv("LLM_MODEL") or os.getenv("OPENROUTER_MODEL_NAME") or os.getenv("OPENROUTER_MODEL") or "openai/gpt-4o-mini"
        openrouter_max_tokens_val = os.getenv("OPENROUTER_MAX_TOKENS", "8192").strip()
        openrouter_max_tokens = int(openrouter_max_tokens_val) if openrouter_max_tokens_val.isdigit() else 8192

        extra_headers = {}
        site_url = os.getenv("OPENROUTER_SITE_URL")
        app_name = os.getenv("OPENROUTER_APP_NAME")
        if site_url:
            extra_headers["HTTP-Referer"] = site_url
        if app_name:
            extra_headers["X-Title"] = app_name

        return ChatOpenAI(
            model=model_name,
            api_key=api_key,
            base_url="https://openrouter.ai/api/v1",
            default_headers=extra_headers if extra_headers else None,
            max_tokens=openrouter_max_tokens,
            temperature=0,
        )

    elif provider == "openai":
        api_key = os.getenv("OPENAI_API_KEY")
        if not api_key or api_key == "your_openai_api_key_here":
            raise ValueError(
                "OPENAI_API_KEY is required when LLM_PROVIDER=openai. "
                "OPENAI_API_KEY environment variable is not configured."
            )
        model_name = os.getenv("LLM_MODEL") or os.getenv("OPENAI_MODEL_NAME") or "gpt-4o-mini"
        return ChatOpenAI(model=model_name, api_key=api_key, temperature=0)

    else:
        raise ValueError(f"Unsupported LLM provider: {raw_provider}")




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
    # Plan approval gate fields (Phase 15+)
    plan_approval_required = bool(state.get("plan_approval_required", False))
    plan_approval_status = state.get("plan_approval_status", "not_required")
    plan_content = state.get("plan_content")

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

            elif content.startswith("Error") or "Error" in content or "Ambiguous" in content or "Access denied" in content:
                failed_tool_messages += 1

    if plan and isinstance(plan, dict) and "tasks" in plan and plan["tasks"]:
        completed_indices = [
            idx for idx, t in enumerate(plan["tasks"]) if t.get("status") == "completed"
        ]
        if completed_indices:
            max_completed_idx = max(completed_indices)
            for idx in range(max_completed_idx):
                if plan["tasks"][idx].get("status") in ("pending", "in_progress"):
                    plan["tasks"][idx]["status"] = "completed"

        if verification_result and verification_result.get("status") == "passed":
            for t in plan["tasks"]:
                t["status"] = "completed"

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
    # Carry plan approval gate fields
    updated_state["plan_approval_required"] = plan_approval_required
    updated_state["plan_approval_status"] = plan_approval_status
    updated_state["plan_content"] = plan_content

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
        if state.get("plan_approval_required") is not None:
            res_dict["plan_approval_required"] = state["plan_approval_required"]
        if state.get("plan_approval_status") is not None:
            res_dict["plan_approval_status"] = state["plan_approval_status"]
        if state.get("plan_content") is not None:
            res_dict["plan_content"] = state["plan_content"]

        if not state.get("messages") and user_goal:
            res_dict["messages"] = [HumanMessage(content=user_goal), response]

        return res_dict

    tool_node = ToolNode(tools)

    def route_after_reason(state: AgentState) -> Literal["tools", "__end__"]:
        """Conditional edge checking if the model requested tool execution."""
        messages = state.get("messages", [])
        if not messages:
            return END

        require_plan_approval = os.getenv("REQUIRE_PLAN_APPROVAL", "false").strip().lower() in ("true", "1", "yes")
        last_message = messages[-1]
        if isinstance(last_message, AIMessage) and getattr(last_message, "tool_calls", None):
            if require_plan_approval:
                plan = state.get("plan")
                plan_st = state.get("plan_approval_status", "not_required")
                if plan and plan_st not in ("approved",):
                    for tc in last_message.tool_calls:
                        if tc.get("name") in ("write_file", "replace_in_file"):
                            return END
            return "tools"

        return END

    workflow = StateGraph(AgentState)
    workflow.add_node("reason", reason_node)
    workflow.add_node("tools", tool_node)

    workflow.add_edge(START, "reason")
    workflow.add_conditional_edges("reason", route_after_reason, ["tools", END])
    workflow.add_edge("tools", "reason")

    return workflow.compile()


def _format_plan_for_approval(plan: dict) -> str:
    """Formats an ExecutionPlan dict as a human-readable approval request block."""
    lines = [
        "=== Plan Approval Request ===",
        f"Goal: {plan.get('goal', '')}",
        f"Tasks ({len(plan.get('tasks', []))}):",
    ]
    for t in plan.get("tasks", []):
        deps = ", ".join(t.get("dependencies", [])) or "none"
        lines.append(f"  [{t.get('id', '?')}] {t.get('title', '')} (deps: {deps})")
        if t.get("description"):
            lines.append(f"      {t.get('description', '')}")
    lines.append("Status: pending")
    return "\n".join(lines)


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

    # --- Plan approval gate (Phase 15+) ---
    require_plan_approval = os.getenv("REQUIRE_PLAN_APPROVAL", "false").strip().lower() in ("true", "1", "yes")
    if require_plan_approval:
        plan = synced_state.get("plan")
        plan_app_req = bool(synced_state.get("plan_approval_required", False))
        plan_app_st = synced_state.get("plan_approval_status", "not_required")
        modified_files = synced_state.get("modified_files") or []

        # Trigger only when: plan exists, no files have been edited yet, approval not already decided
        if (
            plan
            and not modified_files
            and plan_app_st not in ("approved", "rejected")
            and not plan_app_req
        ):
            plan_text = _format_plan_for_approval(plan)
            synced_state["plan_approval_required"] = True
            synced_state["plan_approval_status"] = "pending"
            synced_state["plan_content"] = plan_text
            if not synced_state.get("approval_reason"):
                synced_state["approval_reason"] = "plan pending approval"
            plan_app_req = True
            plan_app_st = "pending"

        if plan_app_req and plan_app_st == "pending":
            synced_state["status"] = "paused"
        elif app_req and app_st == "pending":  # existing Git-delivery gate
            synced_state["status"] = "paused"
        elif ver_res and ver_res.get("status") == "passed":
            synced_state["status"] = "completed"
        elif ver_res and ver_res.get("status") == "failed":
            synced_state["status"] = "failed"
        elif val_res and val_res.get("status") in ("failed", "error") and synced_state.get("retry_count", 0) >= synced_state.get("max_retries", 3):
            synced_state["status"] = "failed"
        else:
            synced_state["status"] = "completed"
    else:
        # Default behavior: plan approval disabled
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
    """Processes a human approval decision for a persistent task and resumes if approved.

    Handles both Git-delivery approval (Phase 10) and plan approval (Phase 15+,
    REQUIRE_PLAN_APPROVAL=true).  When a plan is pending approval, approving clears the
    plan_approval gate so execution proceeds to file modification; rejecting keeps
    plan_approval_status='rejected' so no file edits occur.
    """
    from app.tools import process_human_approval

    initial_state = load_state(task_id, storage_dir=storage_dir)
    updated_state = process_human_approval(initial_state, decision=decision, notes=notes)

    # Also propagate decision to plan-approval gate if it was the pending checkpoint
    if bool(initial_state.get("plan_approval_required", False)) and initial_state.get("plan_approval_status") == "pending":
        decision_norm = str(decision or "").strip().lower()
        if decision_norm in ("approve", "approved", "yes", "pass"):
            updated_state["plan_approval_required"] = False
            updated_state["plan_approval_status"] = "approved"
        else:
            updated_state["plan_approval_required"] = False
            updated_state["plan_approval_status"] = "rejected"
            updated_state["status"] = "paused"
            save_state(task_id, updated_state, status="paused", storage_dir=storage_dir)
            return updated_state

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



def _print_cli_header(goal: str, workspace_root: str) -> None:
    """Prints clean professional header for normal CLI mode."""
    print("=" * 50)
    print("        AUTONOMOUS CODING AGENT")
    print("=" * 50)
    print("\nGoal:")
    print(goal)
    print("\nWorkspace:")
    print(os.path.abspath(workspace_root))
    print("\nAgent started...\n")


def _print_cli_progress(messages: list) -> None:
    """Prints clean high-level progress steps based on the message trace."""
    step_num = 1
    seen_steps = set()

    for msg in messages:
        role = msg.__class__.__name__
        tool_calls = getattr(msg, "tool_calls", None) or []

        for tc in tool_calls:
            name = tc.get("name")
            if name in ("list_files", "read_file", "search_code", "retrieve_relevant_context", "retrieve_hybrid_context") and "understanding" not in seen_steps:
                seen_steps.add("understanding")
                print(f"[{step_num}] Understanding task")
                step_num += 1
            elif name == "run_tests" and "initial_testing" not in seen_steps and "modifying" not in seen_steps:
                seen_steps.add("initial_testing")
                print(f"[{step_num}] Running tests")
                step_num += 1
            elif name == "create_plan" and "planning" not in seen_steps:
                seen_steps.add("planning")
                print(f"[{step_num}] Planning changes")
                step_num += 1
            elif name in ("write_file", "replace_in_file") and "modifying" not in seen_steps:
                seen_steps.add("modifying")
                print(f"[{step_num}] Modifying code")
                step_num += 1
            elif name == "run_tests" and "modifying" in seen_steps and "retesting" not in seen_steps:
                seen_steps.add("retesting")
                print(f"[{step_num}] Testing changes")
                step_num += 1
            elif name == "verify_goal" and "verifying" not in seen_steps:
                seen_steps.add("verifying")
                print(f"[{step_num}] Verifying goal")
                step_num += 1

    if not seen_steps:
        print(f"[{step_num}] Processing goal")


def _print_cli_summary(final_state: dict) -> None:
    """Prints clean final result block based on actual state and metrics."""
    exec_status = final_state.get("status", "completed")
    val_result = final_state.get("validation_result") or {}
    ver_result = final_state.get("verification_result") or {}
    report = final_state.get("evaluation_report") or {}
    outcome = final_state.get("final_outcome") or report.get("final_outcome") or "SUCCESS"
    modified_files = final_state.get("modified_files") or []
    retry_count = final_state.get("retry_count", 0)

    val_status = val_result.get("status") if isinstance(val_result, dict) else None
    ver_status_val = ver_result.get("status") if isinstance(ver_result, dict) else None

    is_success = (
        outcome == "SUCCESS"
        or (exec_status == "completed" and ver_status_val in ("passed", None) and val_status in ("passed", None))
    ) and outcome not in ("ESCALATED", "FAILED")

    if is_success:
        test_summary = val_result.get("summary") if isinstance(val_result, dict) and val_result.get("summary") else "All tests passed successfully."
        ver_display = ver_status_val.upper() if ver_status_val else "PASSED"

        print("\n" + "=" * 50)
        print("SUCCESS")
        print("=" * 50)
        print(f"\nTests: {test_summary}")
        print(f"Files modified: {len(modified_files)}")
        print(f"Recovery retries: {retry_count}")
        print(f"Goal verification: {ver_display}")
    else:
        reason = (
            final_state.get("approval_reason")
            or (ver_result.get("summary") if isinstance(ver_result, dict) else None)
            or (val_result.get("summary") if isinstance(val_result, dict) else None)
            or report.get("summary")
            or f"Task ended with status '{exec_status}'"
        )
        print("\n" + "=" * 50)
        print("FAILED")
        print("=" * 50)
        print(f"\nReason:\n{reason}")


def main():
    """CLI entrypoint for running the agent directly."""
    if len(sys.argv) < 2:
        print("Usage: python -m app.agent \"<user_goal>\" [workspace_root]")
        sys.exit(1)

    goal = sys.argv[1]
    workspace_root = sys.argv[2] if len(sys.argv) > 2 else "."
    log_level = os.getenv("AGENT_LOG_LEVEL", "normal").strip().lower()

    if log_level == "debug":
        print(f"Goal: {goal}")
        print(f"Workspace Root: {os.path.abspath(workspace_root)}")
        print("-" * 50)
    else:
        _print_cli_header(goal=goal, workspace_root=workspace_root)

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

        if log_level == "debug":
            print(f"Task ID: {task_id}")
            print(f"Execution Status: {exec_status}")
            print("\n=== DEBUG: Message Metadata Trace ===")
            for msg in messages:
                role = msg.__class__.__name__
                add_kw = getattr(msg, "additional_kwargs", {})
                resp_meta = getattr(msg, "response_metadata", {})
                print(f"[{role}] content: {repr(getattr(msg, 'content', ''))[:100]}")
                if add_kw:
                    print(f"  additional_kwargs: {sanitize_log_output(str(add_kw))}")
                if resp_meta:
                    print(f"  response_metadata: {sanitize_log_output(str(resp_meta))}")

            print("\n=== Agent Trace ===")
            for msg in messages:
                role = msg.__class__.__name__
                content = getattr(msg, "content", "")
                tool_calls = getattr(msg, "tool_calls", None)

                if role == "HumanMessage":
                    print(f"\n[User Goal]: {content}")
                elif role == "AIMessage":
                    if content:
                        print(f"\n[Agent]: {content}")
                    if tool_calls:
                        for tc in tool_calls:
                            sanitized_args = sanitize_log_output(str(tc.get('args', {})))
                            print(f"  → Tool Call: {tc.get('name')}({sanitized_args})")
                elif role == "ToolMessage":
                    sanitized_obs = sanitize_log_output(content)
                    print(f"\n[Observation]:\n{sanitized_obs}")

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

            _print_cli_summary(final_state)
        else:
            _print_cli_progress(messages)
            _print_cli_summary(final_state)

    except Exception as exc:
        clean_exc = sanitize_log_output(str(exc))
        print("\n" + "=" * 50)
        print("FAILED")
        print("=" * 50)
        print(f"\nReason:\nError executing agent: {clean_exc}")
        sys.exit(1)


if __name__ == "__main__":
    main()
