"""Minimal safe read-only, code modification, planning, and validation tools for Phase 6 Autonomous Coding Agent."""

import ast
import os
import re
import subprocess
from pathlib import Path
import sys
from langchain_core.tools import tool
from app.sandbox import ExecutionSandbox, SecurityError


def safe_resolve_path(workspace_root: str | Path, target_path: str | Path) -> Path:
    """Resolves target_path relative to workspace_root, enforcing directory traversal constraints.

    Args:
        workspace_root: Base workspace directory.
        target_path: Relative or absolute path to check.

    Returns:
        Path: The absolute resolved Path object within workspace_root.

    Raises:
        ValueError: If target_path escapes workspace_root.
    """
    sandbox = ExecutionSandbox(sandbox_root=workspace_root)
    try:
        return sandbox.safe_resolve_path(target_path)
    except SecurityError as err:
        raise ValueError(str(err)) from err



def _is_binary_file(file_path: Path) -> bool:
    """Check if a file appears to be binary by sampling initial bytes."""
    try:
        with open(file_path, "rb") as f:
            chunk = f.read(1024)
            return b"\x00" in chunk
    except Exception:
        return True


def _list_files_impl(directory: str = ".", workspace_root: str = ".", max_files: int = 500) -> str:
    """Implementation of recursive file listing within a safe workspace root."""
    try:
        resolved_dir = safe_resolve_path(workspace_root, directory)
    except ValueError as err:
        return f"Error: {err}"

    if not resolved_dir.exists():
        return f"Error: Directory '{directory}' does not exist."

    if not resolved_dir.is_dir():
        return f"Error: Path '{directory}' is not a directory."

    base = Path(workspace_root).resolve()
    results = []

    try:
        for root, dirs, files in os.walk(resolved_dir, followlinks=False):
            if ".git" in dirs:
                dirs.remove(".git")
            if "__pycache__" in dirs:
                dirs.remove("__pycache__")
            if ".venv" in dirs:
                dirs.remove(".venv")

            for file in sorted(files):
                file_path = Path(root) / file
                try:
                    rel_path = file_path.relative_to(base)
                    results.append(str(rel_path))
                    if len(results) >= max_files:
                        break
                except ValueError:
                    continue

            if len(results) >= max_files:
                break

        if not results:
            return f"No files found in directory '{directory}'."

        output = "\n".join(results)
        if len(results) >= max_files:
            output += f"\n... (output truncated at {max_files} files)"
        return output

    except Exception as exc:
        return f"Error listing directory '{directory}': {str(exc)}"


def _read_file_impl(file_path: str, workspace_root: str = ".", max_bytes: int = 100_000) -> str:
    """Implementation of reading file content within a safe workspace root."""
    sandbox = ExecutionSandbox(sandbox_root=workspace_root)
    return sandbox.read_file(file_path=file_path, max_bytes=max_bytes)



def _search_code_impl(query: str, directory: str = ".", workspace_root: str = ".", max_matches: int = 100) -> str:
    """Implementation of repository code search."""
    if not query or not query.strip():
        return "Error: Search query cannot be empty."

    try:
        resolved_dir = safe_resolve_path(workspace_root, directory)
    except ValueError as err:
        return f"Error: {err}"

    if not resolved_dir.exists():
        return f"Error: Directory '{directory}' does not exist."

    base = Path(workspace_root).resolve()
    query_lower = query.lower()
    matches = []

    try:
        for root, dirs, files in os.walk(resolved_dir, followlinks=False):
            if ".git" in dirs:
                dirs.remove(".git")
            if "__pycache__" in dirs:
                dirs.remove("__pycache__")
            if ".venv" in dirs:
                dirs.remove(".venv")

            for file in sorted(files):
                file_path = Path(root) / file
                if _is_binary_file(file_path):
                    continue

                try:
                    rel_path = file_path.relative_to(base)
                    with open(file_path, "r", encoding="utf-8", errors="replace") as f:
                        for idx, line in enumerate(f, start=1):
                            if query_lower in line.lower():
                                matches.append(f"{rel_path}:{idx}: {line.strip()}")
                                if len(matches) >= max_matches:
                                    break
                except Exception:
                    continue

                if len(matches) >= max_matches:
                    break

            if len(matches) >= max_matches:
                break

        if not matches:
            return f"No matches found for query: '{query}'"

        output = "\n".join(matches)
        if len(matches) >= max_matches:
            output += f"\n... (search output truncated at {max_matches} matches)"
        return output

    except Exception as exc:
        return f"Error searching code for '{query}': {str(exc)}"


def _git_status_impl(workspace_root: str = ".") -> str:
    """Implementation of read-only git status inspection."""
    base = Path(workspace_root).resolve()
    try:
        res = subprocess.run(
            ["git", "status", "--short", "--branch"],
            cwd=base,
            capture_output=True,
            text=True,
            check=False,
        )
        if res.returncode != 0:
            return f"Git Error: {res.stderr.strip() or 'Not a git repository.'}"

        output = res.stdout.strip()
        return output if output else "Git repository is clean. No changes."
    except Exception as exc:
        return f"Error executing git status: {str(exc)}"


def _git_diff_impl(workspace_root: str = ".", max_lines: int = 200) -> str:
    """Implementation of read-only git diff inspection."""
    base = Path(workspace_root).resolve()
    try:
        res_unstaged = subprocess.run(
            ["git", "diff"],
            cwd=base,
            capture_output=True,
            text=True,
            check=False,
        )
        res_staged = subprocess.run(
            ["git", "diff", "--staged"],
            cwd=base,
            capture_output=True,
            text=True,
            check=False,
        )

        diff_lines = []
        if res_unstaged.stdout.strip():
            diff_lines.append("=== Unstaged Changes ===")
            diff_lines.extend(res_unstaged.stdout.strip().splitlines())

        if res_staged.stdout.strip():
            diff_lines.append("\n=== Staged Changes ===")
            diff_lines.extend(res_staged.stdout.strip().splitlines())

        if not diff_lines:
            return "No git diffs found."

        if len(diff_lines) > max_lines:
            truncated = diff_lines[:max_lines]
            truncated.append(f"\n... (git diff output truncated at {max_lines} lines)")
            return "\n".join(truncated)

        return "\n".join(diff_lines)
    except Exception as exc:
        return f"Error executing git diff: {str(exc)}"


def _retrieve_relevant_context_impl(
    query: str,
    directory: str = ".",
    workspace_root: str = ".",
    max_files: int = 3,
    context_window_lines: int = 10,
    max_total_chars: int = 4000,
) -> str:
    """Structured relevance ranking and surrounding code window extraction."""
    if not query or not query.strip():
        return "Error: Retrieval query cannot be empty."

    try:
        resolved_dir = safe_resolve_path(workspace_root, directory)
    except ValueError as err:
        return f"Error: {err}"

    if not resolved_dir.exists():
        return f"Error: Directory '{directory}' does not exist."

    base = Path(workspace_root).resolve()
    query_tokens = [token.lower() for token in re.findall(r"\w+", query) if len(token) > 1]
    if not query_tokens:
        query_tokens = [query.lower().strip()]

    scored_files = []

    try:
        for root, dirs, files in os.walk(resolved_dir, followlinks=False):
            if ".git" in dirs:
                dirs.remove(".git")
            if "__pycache__" in dirs:
                dirs.remove("__pycache__")
            if ".venv" in dirs:
                dirs.remove(".venv")

            for file in sorted(files):
                file_path = Path(root) / file
                if _is_binary_file(file_path):
                    continue

                try:
                    rel_path_str = str(file_path.relative_to(base))
                except ValueError:
                    continue

                path_lower = rel_path_str.lower()
                path_score = sum(10.0 for token in query_tokens if token in path_lower)

                try:
                    content = file_path.read_text(encoding="utf-8", errors="replace")
                except Exception:
                    continue

                lines = content.splitlines()
                content_score = 0.0
                match_indices = []

                for idx, line in enumerate(lines, start=1):
                    line_lower = line.lower()
                    matches_in_line = sum(1 for token in query_tokens if token in line_lower)
                    if matches_in_line > 0:
                        content_score += 3.0 * matches_in_line
                        match_indices.append(idx)

                symbol_score = 0.0
                if file_path.suffix == ".py":
                    try:
                        tree = ast.parse(content)
                        for node in ast.walk(tree):
                            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
                                name_lower = node.name.lower()
                                if any(token in name_lower for token in query_tokens):
                                    symbol_score += 5.0
                                    if hasattr(node, "lineno"):
                                        match_indices.append(node.lineno)
                    except Exception:
                        pass

                total_score = path_score + content_score + symbol_score

                if total_score > 0 and match_indices:
                    scored_files.append(
                        {
                            "rel_path": rel_path_str,
                            "score": total_score,
                            "lines": lines,
                            "matches": sorted(list(set(match_indices))),
                        }
                    )

        if not scored_files:
            return f"No relevant code context found for query: '{query}'"

        scored_files.sort(key=lambda x: x["score"], reverse=True)
        top_files = scored_files[:max_files]

        output_sections = [f"=== Retrieved Relevant Context for Query: '{query}' ==="]

        for rank, item in enumerate(top_files, start=1):
            rel_path = item["rel_path"]
            score = item["score"]
            lines = item["lines"]
            total_lines = len(lines)
            matches = item["matches"]

            windows = []
            for m in matches:
                start = max(1, m - context_window_lines)
                end = min(total_lines, m + context_window_lines)

                if not windows:
                    windows.append([start, end])
                else:
                    last = windows[-1]
                    if start <= last[1] + 1:
                        last[1] = max(last[1], end)
                    else:
                        windows.append([start, end])

            section_text = [f"Rank {rank}: {rel_path} (Relevance Score: {score:.1f})"]
            for w in windows:
                section_text.append(f"Lines {w[0]}–{w[1]}:")
                section_text.append("-" * 40)
                for line_idx in range(w[0], w[1] + 1):
                    line_str = lines[line_idx - 1] if line_idx <= total_lines else ""
                    section_text.append(f"{line_idx:4d}: {line_str}")
                section_text.append("-" * 40)

            output_sections.append("\n".join(section_text))

        full_output = "\n\n".join(output_sections)

        if len(full_output) > max_total_chars:
            full_output = (
                full_output[:max_total_chars]
                + f"\n\n... (retrieved context output truncated at {max_total_chars} characters)"
            )

        return full_output

    except Exception as exc:
        return f"Error retrieving context for '{query}': {str(exc)}"


def _retrieve_hybrid_context_impl(
    query: str,
    top_k: int = 3,
    workspace_root: str = ".",
    max_context_chars: int = 4000,
) -> str:
    """Phase 11 implementation of semantic + lexical + metadata hybrid code retrieval."""
    if not query or not query.strip():
        return "Error: Retrieval query cannot be empty."

    try:
        from app.retrieval import HybridCodeIndex

        index = HybridCodeIndex(workspace_root=workspace_root)
        results = index.search(query=query, top_k=top_k, max_context_chars=max_context_chars)

        if not results:
            return f"No relevant code context found for query: '{query}'"

        sections = [f"=== Retrieved Hybrid Context for Query: '{query}' ==="]
        for rank, res in enumerate(results, start=1):
            c = res.chunk
            symbol_str = f" ({c.symbol_type}: {c.symbol_name})" if c.symbol_name else ""
            header = (
                f"\n--- Rank {rank} | File: {c.file_path} (L{c.start_line}-L{c.end_line}){symbol_str} ---\n"
                f"Scores: Final={res.final_score:.3f} (Lexical={res.lexical_score:.3f}, "
                f"Semantic={res.semantic_score:.3f}, Metadata={res.metadata_score:.3f})"
            )
            sections.append(header)
            sections.append(c.content)

        output = "\n".join(sections)
        if len(output) > max_context_chars:
            output = output[:max_context_chars] + f"\n... (context output truncated at {max_context_chars} characters)"

        return output
    except Exception as exc:
        return f"Error performing hybrid context retrieval: {str(exc)}"



# -----------------------------------------------------------------------------
# Phase 5 Safe Code Modification Implementations
# -----------------------------------------------------------------------------
def _write_file_impl(file_path: str, content: str, workspace_root: str = ".") -> str:
    """Safe writing/creation of repository files with workspace root boundary protection."""
    sandbox = ExecutionSandbox(sandbox_root=workspace_root)
    return sandbox.write_file(file_path=file_path, content=content)


def _replace_in_file_impl(file_path: str, old_text: str, new_text: str, workspace_root: str = ".") -> str:
    """Targeted unique text replacement inside repository files with safety checks."""
    sandbox = ExecutionSandbox(sandbox_root=workspace_root)
    return sandbox.replace_in_file(file_path=file_path, old_text=old_text, new_text=new_text)


# -----------------------------------------------------------------------------
# Phase 6 Controlled Validation & Test Execution Implementation
# -----------------------------------------------------------------------------
def _run_tests_impl(
    target_directory: str = ".",
    workspace_root: str = ".",
    timeout_seconds: int = 30,
    max_output_chars: int = 4000,
) -> str:
    """Safely execute pytest validation tests inside workspace root with timeout and output bounding."""
    sandbox = ExecutionSandbox(sandbox_root=workspace_root)
    try:
        resolved_dir = sandbox.safe_resolve_path(target_directory)
    except (ValueError, SecurityError) as err:
        return f"Error: {err}"

    if not resolved_dir.exists():
        return f"Error: Target directory '{target_directory}' does not exist."

    cmd = [sys.executable, "-B", "-m", "pytest", "-o", "dont_write_bytecode=True"]
    res = sandbox.run_command(
        cmd=cmd,
        cwd=resolved_dir,
        timeout_seconds=timeout_seconds,
        max_output_chars=max_output_chars,
    )

    exit_code_str = "null" if res.get("exit_code") is None else str(res["exit_code"])
    return (
        f"=== Test Execution Result ===\n"
        f"Status: {res['status']}\n"
        f"Exit Code: {exit_code_str}\n"
        f"Summary: {res['summary']}\n"
        f"Output:\n----------------------------------------\n"
        f"{res['output']}\n"
        f"----------------------------------------"
    )



# -----------------------------------------------------------------------------
# Standalone Read-Only Repository Tools
# -----------------------------------------------------------------------------
@tool
def list_files(directory: str = ".") -> str:
    """List files recursively within the controlled workspace directory.

    Args:
        directory: Relative path of the directory to list (defaults to ".").

    Returns:
        A string listing of relative file paths, or an error message.
    """
    return _list_files_impl(directory=directory, workspace_root=".")


@tool
def read_file(file_path: str) -> str:
    """Read and return text contents of a file inside the controlled workspace directory.

    Args:
        file_path: Relative path of the file to read.

    Returns:
        Text contents of the file, or an error message if unreadable or invalid.
    """
    return _read_file_impl(file_path=file_path, workspace_root=".")


@tool
def search_code(query: str, directory: str = ".") -> str:
    """Search text/code files recursively within the controlled workspace for a string match.

    Args:
        query: String or identifier to search for in files.
        directory: Relative path of the directory to search within (defaults to ".").

    Returns:
        Formatted matching lines (file:line: snippet), or a message if no matches found.
    """
    return _search_code_impl(query=query, directory=directory, workspace_root=".")


@tool
def git_status() -> str:
    """Inspect the current Git status (branch, modified files, untracked files) of the workspace.

    Returns:
        Short git status output, or an error message.
    """
    return _git_status_impl(workspace_root=".")


@tool
def git_diff() -> str:
    """Inspect current unstaged and staged Git differences in the workspace.

    Returns:
        Git diff output, or a message if no diffs exist.
    """
    return _git_diff_impl(workspace_root=".")


@tool
def retrieve_relevant_context(query: str, directory: str = ".") -> str:
    """Search and retrieve the most relevant repository code files and surrounding code context for a query.

    Args:
        query: Goal or search query describing the feature, function, or concept.
        directory: Relative directory path inside workspace to search (defaults to ".").

    Returns:
        Ranked list of relevant files with bounded surrounding code snippets and line numbers.
    """
    return _retrieve_relevant_context_impl(query=query, directory=directory, workspace_root=".")


# -----------------------------------------------------------------------------
# Standalone Code Modification Tools
# -----------------------------------------------------------------------------
@tool
def write_file(file_path: str, content: str) -> str:
    """Safely create or overwrite a repository file with text content.

    Args:
        file_path: Relative path of the file inside workspace to write.
        content: Text content to write to the file.

    Returns:
        Confirmation observation string or error message.
    """
    return _write_file_impl(file_path=file_path, content=content, workspace_root=".")


@tool
def replace_in_file(file_path: str, old_text: str, new_text: str) -> str:
    """Perform targeted unique text replacement in an existing repository file.

    Args:
        file_path: Relative path of the file inside workspace to modify.
        old_text: Exact unique target text snippet to replace.
        new_text: Replacement text snippet.

    Returns:
        Confirmation observation string or error message (if text missing or ambiguous).
    """
    return _replace_in_file_impl(
        file_path=file_path, old_text=old_text, new_text=new_text, workspace_root="."
    )


# -----------------------------------------------------------------------------
# Standalone Validation Tool
# -----------------------------------------------------------------------------
@tool
def run_tests(target_directory: str = ".", timeout_seconds: int = 30) -> str:
    """Execute pytest validation tests inside the controlled workspace directory.

    Args:
        target_directory: Relative path of directory containing tests (defaults to ".").
        timeout_seconds: Maximum allowed execution time in seconds (defaults to 30).

    Returns:
        Structured test result observation (Status, Exit Code, Summary, Output traceback).
    """
    return _run_tests_impl(
        target_directory=target_directory, workspace_root=".", timeout_seconds=timeout_seconds
    )


# -----------------------------------------------------------------------------
# Standalone Planning Tools
# -----------------------------------------------------------------------------
@tool
def create_plan(tasks: list[dict]) -> str:
    """Initialize an execution plan with subtasks, titles, descriptions, and dependency IDs.

    Args:
        tasks: List of task dictionaries (keys: 'id', 'title', 'description', 'dependencies').

    Returns:
        Formatted plan confirmation observation.
    """
    formatted = []
    for idx, t in enumerate(tasks, start=1):
        tid = t.get("id", f"task-{idx}")
        title = t.get("title", f"Task {idx}")
        deps = t.get("dependencies", [])
        dep_str = f" [Depends on: {', '.join(deps)}]" if deps else ""
        formatted.append(f"• [{tid}] {title}{dep_str} - Status: pending")

    return f"Execution plan initialized with {len(tasks)} subtasks:\n" + "\n".join(formatted)


@tool
def update_task_status(task_id: str, status: str, notes: str = "") -> str:
    """Update the status of a specific task in the execution plan.

    Args:
        task_id: Unique task identifier (e.g. 'task-1').
        status: New status ('pending', 'in_progress', 'completed', 'failed', 'blocked').
        notes: Optional explanation or notes about the status update.

    Returns:
        Status update confirmation observation.
    """
    note_str = f" Notes: {notes}" if notes else ""
    return f"Task '{task_id}' status updated to '{status}'.{note_str}"


@tool
def revise_plan(new_tasks: list[dict], reason: str) -> str:
    """Modify or replace remaining plan tasks based on new evidence observed in the repository.

    Args:
        new_tasks: List of updated task dictionaries.
        reason: Explanation of why the plan is being revised.

    Returns:
        Plan revision confirmation observation.
    """
    formatted = []
    for idx, t in enumerate(new_tasks, start=1):
        tid = t.get("id", f"task-{idx}")
        title = t.get("title", f"Task {idx}")
        st = t.get("status", "pending")
        formatted.append(f"• [{tid}] {title} - Status: {st}")

    return (
        f"Execution plan revised ({len(new_tasks)} tasks). Reason: '{reason}'\n"
        + "\n".join(formatted)
    )


def _verify_goal_impl(status: str, summary: str, evidence: list[str] | None = None) -> str:
    """Safely format a goal verification observation."""
    valid_statuses = ["passed", "failed", "uncertain"]
    normalized_status = status.lower().strip() if status else "uncertain"
    if normalized_status not in valid_statuses:
        normalized_status = "uncertain"

    ev_list = evidence or []
    ev_str = "\n".join([f"  • {item}" for item in ev_list]) if ev_list else "  • (No specific evidence items provided)"

    return (
        f"=== Goal Verification Result ===\n"
        f"Status: {normalized_status}\n"
        f"Summary: {summary}\n"
        f"Evidence:\n{ev_str}"
    )


@tool
def verify_goal(status: str, summary: str, evidence: list[str] = []) -> str:
    """Evaluate whether the user's original goal has been verified against repository evidence.

    Args:
        status: Verification status ('passed', 'failed', or 'uncertain').
        summary: Concise explanation supporting the verification decision.
        evidence: List of concrete repository evidence lines/findings (e.g. code snippets, test results).

    Returns:
        Structured verification evaluation result.
    """
    return _verify_goal_impl(status=status, summary=summary, evidence=evidence)


# -----------------------------------------------------------------------------
# Phase 10 Git Delivery & Human Approval Implementations
# -----------------------------------------------------------------------------
def _git_current_branch_impl(workspace_root: str = ".") -> str:
    """Safely inspect current active Git branch name."""
    sandbox = ExecutionSandbox(sandbox_root=workspace_root)
    return sandbox.git_current_branch()


def _git_create_branch_impl(branch_name: str, workspace_root: str = ".") -> str:
    """Safely create a new local feature branch."""
    sandbox = ExecutionSandbox(sandbox_root=workspace_root)
    return sandbox.git_create_branch(branch_name=branch_name)


def _git_commit_impl(message: str, files: list[str] | None = None, workspace_root: str = ".") -> str:
    """Safely stage modified files and commit with validated commit message."""
    sandbox = ExecutionSandbox(sandbox_root=workspace_root)
    return sandbox.git_commit(message=message, files=files)


def _git_push_impl(remote: str = "origin", branch: str | None = None, workspace_root: str = ".") -> str:
    """Safely push local branch changes to remote repository."""
    sandbox = ExecutionSandbox(sandbox_root=workspace_root)
    return sandbox.git_push(remote=remote, branch=branch)


def _create_pull_request_impl(
    title: str, body: str, head_branch: str, base_branch: str = "main", workspace_root: str = "."
) -> str:
    """Safely create pull request representation."""
    sandbox = ExecutionSandbox(sandbox_root=workspace_root)
    return sandbox.create_pull_request(
        title=title, body=body, head_branch=head_branch, base_branch=base_branch
    )


def _request_human_approval_impl(
    action: str, reason: str, risk: str = "medium", workspace_root: str = "."
) -> str:
    """Format structured human approval request."""
    sandbox = ExecutionSandbox(sandbox_root=workspace_root)
    branch = sandbox.git_current_branch()
    diff_res = sandbox.run_command(["git", "diff", "--stat"], cwd=workspace_root)
    status_res = sandbox.run_command(["git", "status", "--short"], cwd=workspace_root)
    diff_summary = diff_res.get("output", "No diff summary") or "No unstaged diff"
    status_summary = status_res.get("output", "Clean") or "No modified files"

    return (
        f"=== Human Approval Request ===\n"
        f"Action: {action}\n"
        f"Reason: {reason}\n"
        f"Repository: {Path(workspace_root).resolve()}\n"
        f"Branch: {branch}\n"
        f"Files affected:\n{status_summary}\n"
        f"Git diff summary:\n{diff_summary}\n"
        f"Risk: {risk}\n"
        f"Status: pending"
    )


from app.state import set_active_approval_status, get_active_approval_status


def process_human_approval(state: dict, decision: str, notes: str = "") -> dict:
    """Updates agent state with human approval decision ('approved' or 'rejected')."""
    updated = dict(state)
    decision_norm = str(decision or "").strip().lower()
    ws = updated.get("workspace_root", ".")

    if decision_norm in ("approve", "approved", "yes", "pass"):
        updated["approval_status"] = "approved"
        updated["approval_required"] = False
        updated["approval_reason"] = f"Action approved by human operator: {notes}" if notes else "Action approved by human operator."
        set_active_approval_status(ws, "approved")
    else:
        updated["approval_status"] = "rejected"
        updated["approval_required"] = False
        updated["approval_reason"] = f"Action cancelled by human approval decision: {notes}" if notes else "Action cancelled by human approval decision."
        updated["status"] = "paused"
        set_active_approval_status(ws, "rejected")

    return updated


def _check_tool_approval(workspace_root: str = ".", action: str = "commit") -> tuple[str, str]:
    """Checks if human approval was granted for the delivery action in active state."""
    status = get_active_approval_status(workspace_root)
    if status == "approved":
        return "approved", "Action approved by human operator."
    elif status == "rejected":
        return "rejected", "Action cancelled by human approval decision."
    return "pending", "Human approval required before performing action."


# -----------------------------------------------------------------------------
# Standalone Phase 10 Tools
# -----------------------------------------------------------------------------
@tool
def git_current_branch() -> str:
    """Query current active Git branch name of the workspace.

    Returns:
        Name of the current active branch (e.g. 'main', 'feature-auth').
    """
    return _git_current_branch_impl(workspace_root=".")


@tool
def git_create_branch(branch_name: str) -> str:
    """Safely create and check out a new local feature branch inside the repository.

    Args:
        branch_name: Safe branch name (e.g. 'agent/add-feature').

    Returns:
        Confirmation message or error.
    """
    return _git_create_branch_impl(branch_name=branch_name, workspace_root=".")


@tool
def request_human_approval(action: str, reason: str, risk: str = "medium") -> str:
    """Request explicit human approval before performing externally impactful delivery actions.

    Args:
        action: Delivery action name ('commit', 'push', 'pull_request').
        reason: Justification for why this action should be taken.
        risk: Estimated risk level ('low', 'medium', 'high').

    Returns:
        Formatted human approval request observation block.
    """
    return _request_human_approval_impl(action=action, reason=reason, risk=risk, workspace_root=".")


@tool
def git_commit(message: str, files: list[str] = []) -> str:
    """Safely stage and commit repository changes after human approval.

    Args:
        message: Commit message describing the changes.
        files: Optional list of relative file paths to commit.

    Returns:
        Commit confirmation or error observation.
    """
    app_status, app_reason = _check_tool_approval(".", action="commit")
    if app_status == "rejected":
        return f"Error: Action cancelled by human approval decision ({app_reason})."
    elif app_status != "approved":
        return "Error: Human approval required before performing git commit (approval_status is not 'approved'). Please invoke request_human_approval tool first."
    return _git_commit_impl(message=message, files=files, workspace_root=".")


@tool
def git_push(remote: str = "origin", branch: str = "") -> str:
    """Safely push committed branch changes to remote repository after human approval.

    Args:
        remote: Remote name (defaults to 'origin').
        branch: Optional branch name to push (defaults to current branch).

    Returns:
        Push confirmation observation.
    """
    app_status, app_reason = _check_tool_approval(".", action="push")
    if app_status == "rejected":
        return f"Error: Action cancelled by human approval decision ({app_reason})."
    elif app_status != "approved":
        return "Error: Human approval required before performing git push (approval_status is not 'approved'). Please invoke request_human_approval tool first."
    return _git_push_impl(remote=remote, branch=branch if branch else None, workspace_root=".")


@tool
def create_pull_request(title: str, body: str, head_branch: str, base_branch: str = "main") -> str:
    """Create a pull request representation on GitHub after human approval.

    Args:
        title: Pull request title.
        body: Detailed description of pull request changes.
        head_branch: Source feature branch containing changes.
        base_branch: Target base branch (defaults to 'main').

    Returns:
        Pull request creation observation block.
    """
    app_status, app_reason = _check_tool_approval(".", action="pull_request")
    if app_status == "rejected":
        return f"Error: Action cancelled by human approval decision ({app_reason})."
    elif app_status != "approved":
        return "Error: Human approval required before creating pull request (approval_status is not 'approved'). Please invoke request_human_approval tool first."
    return _create_pull_request_impl(
        title=title, body=body, head_branch=head_branch, base_branch=base_branch, workspace_root="."
    )


@tool
def retrieve_hybrid_context(query: str, top_k: int = 3) -> str:
    """Perform hybrid semantic, lexical, and metadata code retrieval across repository chunks.

    Args:
        query: Goal or search query describing code functionality or symbols to retrieve.
        top_k: Maximum number of relevant code chunks to return (defaults to 3).

    Returns:
        Structured observation block containing ranked code chunks and score breakdown.
    """
    return _retrieve_hybrid_context_impl(query=query, top_k=top_k, workspace_root=".")


def create_workspace_tools(workspace_root: str = "."):
    """Create repository inspection, retrieval, code modification, validation, planning, and delivery tools bound to workspace root.

    Args:
        workspace_root: Path to the root workspace directory.

    Returns:
        List of decorated LangChain tools bound to the workspace_root.
    """

    @tool
    def list_files(directory: str = ".") -> str:
        """List files recursively within the controlled workspace directory.

        Args:
            directory: Relative path of the directory to list (defaults to ".").

        Returns:
            A string listing of relative file paths, or an error message.
        """
        return _list_files_impl(directory=directory, workspace_root=workspace_root)

    @tool
    def read_file(file_path: str) -> str:
        """Read and return text contents of a file inside the controlled workspace directory.

        Args:
            file_path: Relative path of the file to read.

        Returns:
            Text contents of the file, or an error message if unreadable or invalid.
        """
        return _read_file_impl(file_path=file_path, workspace_root=workspace_root)

    @tool
    def search_code(query: str, directory: str = ".") -> str:
        """Search text/code files recursively within the controlled workspace for a string match.

        Args:
            query: String or identifier to search for in files.
            directory: Relative path of the directory to search within (defaults to ".").

        Returns:
            Formatted matching lines (file:line: snippet), or a message if no matches found.
        """
        return _search_code_impl(query=query, directory=directory, workspace_root=workspace_root)

    @tool
    def git_status() -> str:
        """Inspect the current Git status (branch, modified files, untracked files) of the workspace.

        Returns:
            Short git status output, or an error message.
        """
        return _git_status_impl(workspace_root=workspace_root)

    @tool
    def git_diff() -> str:
        """Inspect current unstaged and staged Git differences in the workspace.

        Returns:
            Git diff output, or a message if no diffs exist.
        """
        return _git_diff_impl(workspace_root=workspace_root)

    @tool
    def retrieve_relevant_context(query: str, directory: str = ".") -> str:
        """Search and retrieve the most relevant repository code files and surrounding code context for a query.

        Args:
            query: Goal or search query describing the feature, function, or concept.
            directory: Relative directory path inside workspace to search (defaults to ".").

        Returns:
            Ranked list of relevant files with bounded surrounding code snippets and line numbers.
        """
        return _retrieve_relevant_context_impl(
            query=query, directory=directory, workspace_root=workspace_root
        )

    @tool
    def write_file(file_path: str, content: str) -> str:
        """Safely create or overwrite a repository file with text content.

        Args:
            file_path: Relative path of the file inside workspace to write.
            content: Text content to write to the file.

        Returns:
            Confirmation observation string or error message.
        """
        return _write_file_impl(file_path=file_path, content=content, workspace_root=workspace_root)

    @tool
    def replace_in_file(file_path: str, old_text: str, new_text: str) -> str:
        """Perform targeted unique text replacement in an existing repository file.

        Args:
            file_path: Relative path of the file inside workspace to modify.
            old_text: Exact unique target text snippet to replace.
            new_text: Replacement text snippet.

        Returns:
            Confirmation observation string or error message (if text missing or ambiguous).
        """
        return _replace_in_file_impl(
            file_path=file_path, old_text=old_text, new_text=new_text, workspace_root=workspace_root
        )

    @tool
    def run_tests(target_directory: str = ".", timeout_seconds: int = 30) -> str:
        """Execute pytest validation tests inside the controlled workspace directory.

        Args:
            target_directory: Relative path of directory containing tests (defaults to ".").
            timeout_seconds: Maximum allowed execution time in seconds (defaults to 30).

        Returns:
            Structured test result observation (Status, Exit Code, Summary, Output traceback).
        """
        return _run_tests_impl(
            target_directory=target_directory,
            workspace_root=workspace_root,
            timeout_seconds=timeout_seconds,
        )

    @tool
    def git_current_branch() -> str:
        """Query current active Git branch name of the workspace.

        Returns:
            Name of the current active branch (e.g. 'main', 'feature-auth').
        """
        return _git_current_branch_impl(workspace_root=workspace_root)

    @tool
    def git_create_branch(branch_name: str) -> str:
        """Safely create and check out a new local feature branch inside the repository.

        Args:
            branch_name: Safe branch name (e.g. 'agent/add-feature').

        Returns:
            Confirmation message or error.
        """
        return _git_create_branch_impl(branch_name=branch_name, workspace_root=workspace_root)

    @tool
    def request_human_approval(action: str, reason: str, risk: str = "medium") -> str:
        """Request explicit human approval before performing externally impactful delivery actions.

        Args:
            action: Delivery action name ('commit', 'push', 'pull_request').
            reason: Justification for why this action should be taken.
            risk: Estimated risk level ('low', 'medium', 'high').

        Returns:
            Formatted human approval request observation block.
        """
        return _request_human_approval_impl(
            action=action, reason=reason, risk=risk, workspace_root=workspace_root
        )

    @tool
    def git_commit(message: str, files: list[str] = []) -> str:
        """Safely stage and commit repository changes after human approval.

        Args:
            message: Commit message describing the changes.
            files: Optional list of relative file paths to commit.

        Returns:
            Commit confirmation or error observation.
        """
        app_status, app_reason = _check_tool_approval(workspace_root, action="commit")
        if app_status == "rejected":
            return f"Error: Action cancelled by human approval decision ({app_reason})."
        elif app_status != "approved":
            return "Error: Human approval required before performing git commit (approval_status is not 'approved'). Please invoke request_human_approval tool first."
        return _git_commit_impl(message=message, files=files, workspace_root=workspace_root)

    @tool
    def git_push(remote: str = "origin", branch: str = "") -> str:
        """Safely push committed branch changes to remote repository after human approval.

        Args:
            remote: Remote name (defaults to 'origin').
            branch: Optional branch name to push (defaults to current branch).

        Returns:
            Push confirmation observation.
        """
        app_status, app_reason = _check_tool_approval(workspace_root, action="push")
        if app_status == "rejected":
            return f"Error: Action cancelled by human approval decision ({app_reason})."
        elif app_status != "approved":
            return "Error: Human approval required before performing git push (approval_status is not 'approved'). Please invoke request_human_approval tool first."
        return _git_push_impl(
            remote=remote, branch=branch if branch else None, workspace_root=workspace_root
        )

    @tool
    def create_pull_request(title: str, body: str, head_branch: str, base_branch: str = "main") -> str:
        """Create a pull request representation on GitHub after human approval.

        Args:
            title: Pull request title.
            body: Detailed description of pull request changes.
            head_branch: Source feature branch containing changes.
            base_branch: Target base branch (defaults to 'main').

        Returns:
            Pull request creation observation block.
        """
        app_status, app_reason = _check_tool_approval(workspace_root, action="pull_request")
        if app_status == "rejected":
            return f"Error: Action cancelled by human approval decision ({app_reason})."
        elif app_status != "approved":
            return "Error: Human approval required before creating pull request (approval_status is not 'approved'). Please invoke request_human_approval tool first."
        return _create_pull_request_impl(
            title=title,
            body=body,
            head_branch=head_branch,
            base_branch=base_branch,
            workspace_root=workspace_root,
        )

    @tool
    def retrieve_hybrid_context(query: str, top_k: int = 3) -> str:
        """Perform hybrid semantic, lexical, and metadata code retrieval across repository chunks.

        Args:
            query: Goal or search query describing code functionality or symbols to retrieve.
            top_k: Maximum number of relevant code chunks to return (defaults to 3).

        Returns:
            Structured observation block containing ranked code chunks and score breakdown.
        """
        return _retrieve_hybrid_context_impl(
            query=query, top_k=top_k, workspace_root=workspace_root
        )

    return [
        list_files,
        read_file,
        search_code,
        git_status,
        git_diff,
        retrieve_relevant_context,
        retrieve_hybrid_context,
        write_file,
        replace_in_file,
        run_tests,
        create_plan,
        update_task_status,
        revise_plan,
        verify_goal,
        git_current_branch,
        git_create_branch,
        request_human_approval,
        git_commit,
        git_push,
        create_pull_request,
    ]

