"""Minimal safe read-only repository tools for Phase 2 Autonomous Coding Agent."""

import os
import subprocess
from pathlib import Path
from langchain_core.tools import tool


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
    base = Path(workspace_root).resolve()
    target = Path(target_path)

    if target.is_absolute():
        resolved = target.resolve()
    else:
        resolved = (base / target).resolve()

    try:
        resolved.relative_to(base)
    except ValueError:
        raise ValueError(
            f"Access denied: path '{target_path}' escapes workspace directory '{base}'"
        )

    return resolved


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
            # Skip .git directory internals
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
    try:
        resolved_file = safe_resolve_path(workspace_root, file_path)
    except ValueError as err:
        return f"Error: {err}"

    if not resolved_file.exists():
        return f"Error: File '{file_path}' does not exist."

    if not resolved_file.is_file():
        return f"Error: Path '{file_path}' is a directory, not a file."

    if _is_binary_file(resolved_file):
        return f"Error: File '{file_path}' appears to be binary and cannot be read as text."

    try:
        file_size = resolved_file.stat().st_size
        text = resolved_file.read_text(encoding="utf-8", errors="replace")

        if file_size > max_bytes:
            text = text[:max_bytes] + f"\n... (truncated file content at {max_bytes} bytes)"

        return text
    except Exception as exc:
        return f"Error reading file '{file_path}': {str(exc)}"


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
        # Unstaged diff
        res_unstaged = subprocess.run(
            ["git", "diff"],
            cwd=base,
            capture_output=True,
            text=True,
            check=False,
        )
        # Staged diff
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


@tool
def list_files(directory: str = ".") -> str:
    """List files and subdirectories recursively within the controlled workspace directory.

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


def create_workspace_tools(workspace_root: str = "."):
    """Create read-only tool instances bound to a specific workspace root directory.

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

    return [list_files, read_file, search_code, git_status, git_diff]
