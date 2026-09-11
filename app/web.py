"""Phase 16I Web Dashboard API for Autonomous Coding Agent."""

import os
from pathlib import Path
from typing import Any, Dict, List, Optional
from fastapi import FastAPI, HTTPException, Request, status, Depends, Header
from fastapi.responses import HTMLResponse, FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

from app.agent import run_agent, _validate_workspace, sanitize_log_output
from app.memory import (
    load_state,
    save_state,
    validate_task_id,
    safe_resolve_task_memory_path,
)
from app.evaluation import sanitize_telemetry_dict, sanitize_telemetry_value


# -----------------------------------------------------------------------------
# Pydantic Schemas
# -----------------------------------------------------------------------------

class TaskSubmissionRequest(BaseModel):
    goal: str = Field(..., description="The software engineering goal or bug fix instruction.")
    workspace_root: Optional[str] = Field(".", description="Root directory of the workspace or repository.")


# -----------------------------------------------------------------------------
# FastAPI App Setup
# -----------------------------------------------------------------------------

app = FastAPI(
    title="Autonomous Coding Agent Web API",
    description="Web dashboard interface for the Autonomous Coding Agent platform.",
    version="1.0.0",
    docs_url=None,
    redoc_url=None,
)

# Store custom LLM for testing/dependency injection
app.state.llm = None


# -----------------------------------------------------------------------------
# Security & Authentication Boundary
# -----------------------------------------------------------------------------
# NOTE: Production OAuth is Phase 16Q.
# For Phase 16I, a simple WEB_API_KEY header check is supported for dev/testing.

def verify_web_auth(x_api_key: Optional[str] = Header(None)) -> None:
    """Lightweight development API key authentication check.
    
    If WEB_API_KEY is configured in the environment, validates the header.
    Otherwise, defaults to open local development mode. Production OAuth is Phase 16Q.
    """
    expected_key = os.getenv("WEB_API_KEY")
    if expected_key and expected_key != "your_web_api_key_here":
        if not x_api_key or x_api_key != expected_key:
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="Unauthorized: Invalid or missing X-API-Key header.",
            )


@app.exception_handler(Exception)
async def global_exception_handler(request: Request, exc: Exception):
    """Global exception handler returning sanitized JSON error without exposing raw tracebacks."""
    clean_err = sanitize_log_output(str(exc))
    return JSONResponse(
        status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
        content={"detail": "An internal server error occurred.", "error": clean_err},
    )


# -----------------------------------------------------------------------------
# Helper Functions
# -----------------------------------------------------------------------------

def _record_web_event(state: Dict[str, Any], event_type: str, status_str: str, metadata: Optional[Dict[str, Any]] = None) -> None:
    """Appends a web observability event to the task execution trace."""
    import time
    trace = list(state.get("execution_trace") or [])
    event = {
        "timestamp": round(time.time(), 3),
        "event_type": event_type,
        "agent_role": "web_dashboard",
        "tool_name": None,
        "status": status_str,
        "duration": None,
        "metadata": sanitize_telemetry_dict(metadata) if metadata else None,
    }
    trace.append(event)
    state["execution_trace"] = trace


def _build_final_report_response(state: Dict[str, Any]) -> Dict[str, Any]:
    """Formats raw AgentState into a web-friendly final execution report dictionary."""
    report = state.get("evaluation_report") or {}
    val_res = state.get("validation_result") or {}
    ver_res = state.get("verification_result") or {}
    messages = state.get("messages") or []
    
    outcome = state.get("final_outcome") or report.get("final_outcome")
    if not outcome:
        ver_st = ver_res.get("status") if isinstance(ver_res, dict) else None
        exec_st = state.get("status")
        if ver_st == "passed" or (exec_st == "completed" and ver_st in ("passed", None)):
            outcome = "SUCCESS"
        else:
            outcome = "FAILED"

    raw_goal = state.get("user_goal") or report.get("user_goal") or ""
    goal = sanitize_log_output(str(raw_goal))
    
    modified_files = state.get("modified_files") or []
    
    # Calculate tool calls count
    tc_count = report.get("tool_call_count")
    if tc_count is None:
        tc_count = sum(len(getattr(m, "tool_calls", None) or []) for m in messages) if isinstance(messages, list) else 0

    # Validation attempts
    val_attempts = report.get("validation_attempts")
    if val_attempts is None and isinstance(messages, list):
        v_cnt = 0
        for m in messages:
            tcs = getattr(m, "tool_calls", None) or []
            for tc in tcs:
                if isinstance(tc, dict) and tc.get("name") == "run_tests":
                    v_cnt += 1
        val_attempts = v_cnt

    # Retries
    retries = state.get("retry_count", report.get("retry_count", 0))

    # Goal verification
    goal_ver = "PASSED" if ver_res.get("status") == "passed" else ("FAILED" if ver_res.get("status") == "failed" else "N/A")

    # Human interventions
    human_int = report.get("human_interventions", 0)

    # Execution time
    exec_time = report.get("execution_time", 0.0)

    # Summary/Reason
    reason = None
    if outcome in ("FAILED", "ESCALATED"):
        reason = (
            state.get("approval_reason")
            or (ver_res.get("summary") if isinstance(ver_res, dict) else None)
            or (val_res.get("summary") if isinstance(val_res, dict) else None)
            or report.get("summary")
        )
        if reason:
            reason = sanitize_log_output(str(reason))

    return {
        "status": outcome,
        "goal": goal,
        "workspace": state.get("workspace_root", "."),
        "tests": val_res.get("summary", "N/A") if isinstance(val_res, dict) else "N/A",
        "files_modified_count": len(modified_files),
        "files_modified": modified_files,
        "tool_calls": tc_count,
        "validation_attempts": val_attempts,
        "recovery_retries": retries,
        "goal_verification": goal_ver,
        "human_interventions": human_int,
        "execution_time_seconds": exec_time,
        "reason": reason,
    }


# -----------------------------------------------------------------------------
# API Endpoints
# -----------------------------------------------------------------------------

@app.get("/health")
def health_check():
    """Safe health check endpoint. Does not expose secrets, environment variables, or credentials."""
    return {"status": "ok", "service": "autonomous-coding-agent"}


@app.post("/api/tasks", dependencies=[Depends(verify_web_auth)])
def submit_task(req: TaskSubmissionRequest):
    """Submits a new task to the EXISTING autonomous coding agent."""
    try:
        goal_str = (req.goal or "").strip()
        if not goal_str:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Task goal cannot be empty.",
            )

        workspace_root = req.workspace_root or "."
        
        # Path traversal and valid workspace validation using existing validation rules
        if ".." in workspace_root:
            resolved_ws = Path(workspace_root).resolve()
            if not resolved_ws.exists():
                raise ValueError(f"Path traversal escape or non-existent path: '{workspace_root}'")
        
        _validate_workspace(workspace_root)

        # Invoke EXISTING agent core
        llm = getattr(app.state, "llm", None)
        
        import uuid
        task_id = f"task_{uuid.uuid4().hex[:8]}"

        final_state = run_agent(
            goal=goal_str,
            workspace_root=workspace_root,
            llm=llm,
            task_id=task_id,
        )

        # Record observability web events
        outcome = final_state.get("final_outcome", "SUCCESS")
        if outcome == "SUCCESS":
            _record_web_event(final_state, "web_task_completed", "completed")
        elif outcome == "ESCALATED":
            _record_web_event(final_state, "web_task_escalated", "escalated")
        else:
            _record_web_event(final_state, "web_task_failed", "failed")

        # Save updated trace
        save_state(task_id, final_state, status=final_state.get("status", "completed"))

        report = _build_final_report_response(final_state)

        return {
            "task_id": task_id,
            "status": final_state.get("status", "completed"),
            "final_outcome": outcome,
            "goal": goal_str,
            "workspace_root": workspace_root,
            "final_report": report,
            "trace_events_count": len(final_state.get("execution_trace") or []),
        }

    except HTTPException:
        raise
    except ValueError as val_err:
        clean_msg = sanitize_log_output(str(val_err))
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Invalid workspace: {clean_msg}",
        )
    except TimeoutError as to_err:
        clean_exc = sanitize_telemetry_value(sanitize_log_output(str(to_err)))
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Agent execution timed out: {clean_exc}",
        )
    except Exception as exc:
        clean_exc = sanitize_telemetry_value(sanitize_log_output(str(exc)))
        if "timed out" in str(exc).lower() or "timeout" in str(exc).lower():
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail=f"Agent execution timed out: {clean_exc}",
            )
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Agent execution failed: {clean_exc}",
        )




@app.get("/api/tasks", dependencies=[Depends(verify_web_auth)])
def list_tasks(storage_dir: str = ".agent_memory"):
    """Returns task history list from existing persistent memory storage."""
    storage_path = Path(storage_dir)
    if not storage_path.exists() or not storage_path.is_dir():
        return []

    tasks = []
    for file_path in storage_path.glob("*.json"):
        if file_path.name.endswith(".tmp"):
            continue
        task_id = file_path.stem
        try:
            state = load_state(task_id, storage_dir=storage_dir)
            report = state.get("evaluation_report") or {}
            tasks.append({
                "task_id": task_id,
                "goal": sanitize_log_output(str(state.get("user_goal", ""))),
                "status": state.get("status", "unknown"),
                "final_outcome": state.get("final_outcome", "UNKNOWN"),
                "workspace_root": state.get("workspace_root", "."),
                "files_modified_count": len(state.get("modified_files") or []),
                "retry_count": state.get("retry_count", 0),
                "duration": report.get("execution_time"),
            })
        except Exception:
            continue

    # Sort tasks descending by task_id
    tasks.sort(key=lambda t: t["task_id"], reverse=True)
    return tasks


@app.get("/api/tasks/{task_id}", dependencies=[Depends(verify_web_auth)])
def get_task_status(task_id: str, storage_dir: str = ".agent_memory"):
    """Retrieves safe execution status and details for a specific task."""
    try:
        validate_task_id(task_id)
    except ValueError as val_err:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=sanitize_log_output(str(val_err)),
        )

    try:
        state = load_state(task_id, storage_dir=storage_dir)
    except FileNotFoundError:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Task '{task_id}' not found.",
        )
    except Exception as exc:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Error loading task state: {sanitize_log_output(str(exc))}",
        )

    report = _build_final_report_response(state)
    raw_trace = state.get("execution_trace") or []
    sanitized_trace = [sanitize_telemetry_dict(ev) for ev in raw_trace]

    return {
        "task_id": task_id,
        "status": state.get("status", "completed"),
        "final_outcome": state.get("final_outcome", "UNKNOWN"),
        "user_goal": sanitize_log_output(str(state.get("user_goal", ""))),
        "workspace_root": state.get("workspace_root", "."),
        "current_stage": state.get("status", "completed"),
        "retries": state.get("retry_count", 0),
        "modified_files": state.get("modified_files") or [],
        "approval_required": state.get("approval_required", False),
        "approval_status": state.get("approval_status", "not_required"),
        "final_report": report,
        "execution_trace": sanitized_trace,
    }


@app.get("/api/tasks/{task_id}/report", dependencies=[Depends(verify_web_auth)])
def get_task_report(task_id: str, storage_dir: str = ".agent_memory"):
    """Retrieves web-friendly final execution report for a task."""
    try:
        validate_task_id(task_id)
        state = load_state(task_id, storage_dir=storage_dir)
        return _build_final_report_response(state)
    except FileNotFoundError:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail=f"Task '{task_id}' not found.")
    except ValueError as val_err:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=sanitize_log_output(str(val_err)))


@app.get("/api/tasks/{task_id}/trace", dependencies=[Depends(verify_web_auth)])
def get_task_trace(task_id: str, storage_dir: str = ".agent_memory"):
    """Retrieves sanitized execution trace events for a task."""
    try:
        validate_task_id(task_id)
        state = load_state(task_id, storage_dir=storage_dir)
        raw_trace = state.get("execution_trace") or []
        return [sanitize_telemetry_dict(ev) for ev in raw_trace]
    except FileNotFoundError:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail=f"Task '{task_id}' not found.")
    except ValueError as val_err:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=sanitize_log_output(str(val_err)))


# -----------------------------------------------------------------------------
# Frontend Static File Serving
# -----------------------------------------------------------------------------

WEB_DIR = Path(__file__).parent.parent / "web"

@app.get("/", response_class=HTMLResponse)
def serve_index():
    """Serves the dashboard index.html frontend page."""
    index_file = WEB_DIR / "index.html"
    if index_file.exists():
        return HTMLResponse(content=index_file.read_text(encoding="utf-8"))
    return HTMLResponse(content="<h1>Autonomous Coding Agent Web Dashboard</h1>")


@app.get("/styles.css")
def serve_styles():
    """Serves styles.css static file."""
    css_file = WEB_DIR / "styles.css"
    if css_file.exists():
        return FileResponse(css_file, media_type="text/css")
    raise HTTPException(status_code=404, detail="File not found")


@app.get("/app.js")
def serve_app_js():
    """Serves app.js static file."""
    js_file = WEB_DIR / "app.js"
    if js_file.exists():
        return FileResponse(js_file, media_type="application/javascript")
    raise HTTPException(status_code=404, detail="File not found")
