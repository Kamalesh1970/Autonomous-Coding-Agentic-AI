"""Minimal safe repository tools for Phase 1 Autonomous Coding Agent."""

import os
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


def _list_files_impl(directory: str = ".", workspace_root: str = ".") -> str:
    """Implementation of listing files within a safe workspace root."""
    try:
        resolved_dir = safe_resolve_path(workspace_root, directory)
    except ValueError as err:
        return f"Error: {err}"

    if not resolved_dir.exists():
        return f"Error: Directory '{directory}' does not exist."

    if not resolved_dir.is_dir():
        return f"Error: Path '{directory}' is not a directory."

    try:
        items = sorted(os.listdir(resolved_dir))
        if not items:
            return f"Directory '{directory}' is empty."

        results = []
        base = Path(workspace_root).resolve()
        for item in items:
            item_path = resolved_dir / item
            try:
                rel_path = item_path.relative_to(base)
                suffix = "/" if item_path.is_dir() else ""
                results.append(f"{rel_path}{suffix}")
            except ValueError:
                results.append(item)

        return "\n".join(results)
    except Exception as exc:
        return f"Error listing directory '{directory}': {str(exc)}"


def _read_file_impl(file_path: str, workspace_root: str = ".") -> str:
    """Implementation of reading file content within a safe workspace root."""
    try:
        resolved_file = safe_resolve_path(workspace_root, file_path)
    except ValueError as err:
        return f"Error: {err}"

    if not resolved_file.exists():
        return f"Error: File '{file_path}' does not exist."

    if not resolved_file.is_file():
        return f"Error: Path '{file_path}' is a directory, not a file."

    try:
        return resolved_file.read_text(encoding="utf-8", errors="replace")
    except Exception as exc:
        return f"Error reading file '{file_path}': {str(exc)}"


@tool
def list_files(directory: str = ".") -> str:
    """List files and subdirectories within the controlled workspace directory.

    Args:
        directory: Relative path of the directory to list (defaults to ".").

    Returns:
        A string listing of relative file and directory paths, or an error message.
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


def create_workspace_tools(workspace_root: str = "."):
    """Create tool instances bound to a specific workspace root directory.

    Args:
        workspace_root: Path to the root workspace directory.

    Returns:
        List of decorated LangChain tools bound to the workspace_root.
    """

    @tool
    def list_files(directory: str = ".") -> str:
        """List files and subdirectories within the controlled workspace directory.

        Args:
            directory: Relative path of the directory to list (defaults to ".").

        Returns:
            A string listing of relative file and directory paths, or an error message.
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

    return [list_files, read_file]
