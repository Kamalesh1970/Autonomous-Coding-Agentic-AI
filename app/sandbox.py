"""Execution Sandbox module for Phase 9 Autonomous Security & Sandbox Isolation."""

import os
import re
import subprocess
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union


class SecurityError(ValueError):
    """Exception raised when a sandbox security boundary policy is violated."""
    pass


class ExecutionSandbox:
    """Controls and restricts filesystem, command execution, and environment access for autonomous agents.

    Attributes:
        sandbox_root: Path object representing the absolute root directory allowed for execution.
        allowed_commands: List of allowed executable/command patterns.
        security_events: Log of security policy events triggered during execution.
    """

    DEFAULT_ALLOWED_COMMANDS = [
        "pytest",
        "python",
        "python3",
        "git",
    ]

    def __init__(
        self,
        sandbox_root: Union[str, Path] = ".",
        allowed_commands: Optional[List[str]] = None,
    ):
        self.sandbox_root = Path(sandbox_root).resolve()
        self.allowed_commands = allowed_commands or list(self.DEFAULT_ALLOWED_COMMANDS)
        self.security_events: List[Dict[str, Any]] = []

    def log_event(self, event_type: str, details: str) -> None:
        """Record a security event log entry."""
        self.security_events.append({"event_type": event_type, "details": details})

    def safe_resolve_path(self, target_path: Union[str, Path]) -> Path:
        """Resolves target_path relative to sandbox_root, enforcing boundary and symlink safety.

        Args:
            target_path: Relative or absolute path string/Path object to resolve.

        Returns:
            Path: Absolute resolved Path guaranteed to be inside sandbox_root.

        Raises:
            SecurityError: If target_path escapes sandbox_root or if a symlink targets outside.
        """
        base = self.sandbox_root.resolve()
        target = Path(target_path)

        if target.is_absolute():
            candidate = target
        else:
            candidate = base / target

        # Check path resolution and symlink target safety
        try:
            resolved = candidate.resolve(strict=False)
        except Exception as exc:
            self.log_event("path_escape_rejected", f"Failed to resolve path '{target_path}': {exc}")
            raise SecurityError(f"Access denied: Path resolution error for '{target_path}'") from exc

        # Check if resolved path is inside sandbox root
        try:
            resolved.relative_to(base)
        except ValueError:
            self.log_event("path_escape_rejected", f"Path '{target_path}' resolved to '{resolved}' outside root '{base}'")
            raise SecurityError(
                f"Access denied: path '{target_path}' escapes sandbox directory '{base}'"
            )

        # Additional explicit symlink check if candidate exists or parent symlinks exist
        curr = candidate
        while curr != base and curr != curr.parent:
            if curr.is_symlink():
                real_link_target = curr.resolve()
                try:
                    real_link_target.relative_to(base)
                except ValueError:
                    self.log_event(
                        "symlink_escape_rejected",
                        f"Symlink '{curr}' points to '{real_link_target}' outside sandbox root '{base}'"
                    )
                    raise SecurityError(
                        f"Access denied: symlink '{curr.name}' points outside sandbox directory '{base}'"
                    )
            curr = curr.parent

        return resolved

    def validate_branch_name(self, branch_name: str) -> None:
        """Validate branch name to reject dangerous arguments or shell injection."""
        if not branch_name or not isinstance(branch_name, str):
            raise SecurityError("Invalid branch name: Branch name must be a non-empty string.")

        branch_name = branch_name.strip()
        if branch_name.startswith("-") or branch_name.startswith("."):
            raise SecurityError(f"Invalid branch name '{branch_name}': Branch name cannot start with hyphen or dot.")

        if not re.match(r"^[a-zA-Z0-9_/.-]+$", branch_name):
            raise SecurityError(f"Invalid branch name '{branch_name}': Contains invalid characters or potential injection.")

    def is_command_allowed(self, cmd: List[str]) -> Tuple[bool, str]:
        """Check if command line arguments satisfy the command allowlist policy.

        Args:
            cmd: List of command and argument strings.

        Returns:
            Tuple[bool, str]: (is_allowed, reason_if_rejected)
        """
        if not cmd:
            return False, "Empty command list."

        exe_name = Path(cmd[0]).name.lower()
        if exe_name.endswith(".exe"):
            exe_name = exe_name[:-4]

        # Explicitly reject generic interactive shell invocations
        disallowed_shells = ["bash", "sh", "zsh", "powershell", "cmd", "ksh", "csh", "dash"]
        if exe_name in disallowed_shells:
            return False, f"Shell execution tool '{exe_name}' is disallowed for security isolation."

        # Allow python executables if running pytest or python module
        if exe_name in ("python", "python3", "python.exe") or cmd[0] == sys.executable:
            # Must run module pytest or explicit script inside sandbox
            if len(cmd) > 1 and cmd[1] in ("-m", "-B"):
                return True, "Allowed python module execution."
            elif len(cmd) > 1 and cmd[1].endswith(".py"):
                try:
                    self.safe_resolve_path(cmd[1])
                    return True, "Allowed python script execution inside sandbox."
                except SecurityError as err:
                    return False, str(err)
            else:
                return True, "Allowed python execution."

        if exe_name == "pytest":
            return True, "Allowed pytest execution."

        if exe_name == "git":
            if len(cmd) < 2:
                return False, "Git command requires arguments."

            subcmd = cmd[1]

            # Reject destructive git subcommands and flags
            destructive_flags = ["--hard", "-fd", "--force", "-f"]
            if any(arg in destructive_flags for arg in cmd):
                return False, "Destructive Git options (reset --hard, clean -fd, push --force) are strictly prohibited."

            if subcmd in ("reset", "clean"):
                return False, f"Destructive Git subcommand '{subcmd}' is disallowed."

            if subcmd in ("status", "diff", "branch", "rev-parse", "log", "add"):
                return True, f"Allowed git {subcmd} command."

            if subcmd in ("checkout", "switch"):
                if len(cmd) >= 4 and cmd[2] in ("-b", "-c"):
                    try:
                        self.validate_branch_name(cmd[3])
                    except SecurityError as err:
                        return False, str(err)
                return True, f"Allowed git {subcmd} command."

            if subcmd == "commit":
                if len(cmd) < 3:
                    return False, "Git commit requires message or arguments."
                return True, "Allowed git commit command."

            if subcmd == "push":
                return False, "Git push command is disallowed via generic execution; use controlled delivery interface."

            return False, f"Git subcommand '{subcmd}' is not in allowlist."

        if exe_name in self.allowed_commands:
            return True, f"Command '{exe_name}' is explicitly allowed."

        return False, f"Command '{cmd[0]}' is not in the execution allowlist."


    def build_minimal_environment(self, custom_env: Optional[Dict[str, str]] = None) -> Dict[str, str]:
        """Construct a minimal execution environment excluding sensitive host keys and credentials.

        Returns:
            Dict[str, str]: Clean environment containing only necessary system execution variables.
        """
        allowed_env_keys = {
            "PATH",
            "PYTHONPATH",
            "SYSTEMROOT",
            "PATHEXT",
            "TMP",
            "TEMP",
            "LANG",
            "LC_ALL",
            "TERM",
            "HOME",
        }

        clean_env: Dict[str, str] = {}
        for key in allowed_env_keys:
            if key in os.environ:
                clean_env[key] = os.environ[key]

        clean_env["PYTHONDONTWRITEBYTECODE"] = "1"
        clean_env["PYTHONUNBUFFERED"] = "1"

        if custom_env:
            for k, v in custom_env.items():
                if k not in ("OPENAI_API_KEY", "AWS_SECRET_ACCESS_KEY", "GITHUB_TOKEN", "SECRET_KEY"):
                    clean_env[k] = v

        return clean_env

    def run_command(
        self,
        cmd: List[str],
        cwd: Optional[Union[str, Path]] = None,
        timeout_seconds: int = 30,
        max_output_chars: int = 4000,
        env: Optional[Dict[str, str]] = None,
    ) -> Dict[str, Any]:
        """Execute a command strictly within sandbox constraints.

        Args:
            cmd: Command arguments list.
            cwd: Optional working directory (must resolve inside sandbox root).
            timeout_seconds: Maximum allowed runtime in seconds.
            max_output_chars: Maximum output characters returned.
            env: Optional custom environment dictionary.

        Returns:
            Dict[str, Any] containing status, exit_code, summary, output, security_event.
        """
        # 1. Command Allowlist Check
        allowed, reason = self.is_command_allowed(cmd)
        if not allowed:
            self.log_event("command_rejected", f"Command '{' '.join(cmd)}' rejected: {reason}")
            return {
                "status": "error",
                "exit_code": None,
                "summary": f"Execution rejected: command not allowed ({reason})",
                "output": f"Security Error: Command '{cmd[0]}' is disallowed by sandbox policy ({reason}).",
                "security_event": "command_rejected",
            }


        # 2. Working Directory Enforcement
        try:
            target_cwd = self.safe_resolve_path(cwd) if cwd else self.sandbox_root
        except SecurityError as err:
            self.log_event("working_dir_rejected", str(err))
            return {
                "status": "error",
                "exit_code": None,
                "summary": f"Execution rejected: path outside sandbox",
                "output": str(err),
                "security_event": "path_escape_rejected",
            }

        # 3. Environment Isolation
        minimal_env = self.build_minimal_environment(env)

        # 4. Controlled Subprocess Execution with Timeout and Output Bounding
        try:
            res = subprocess.run(
                cmd,
                cwd=target_cwd,
                capture_output=True,
                text=True,
                timeout=timeout_seconds,
                check=False,
                env=minimal_env,
            )

            exit_code = res.returncode
            status = "passed" if exit_code == 0 else "failed"
            summary = (
                "All tests passed successfully."
                if exit_code == 0
                else f"Test validation failed with exit code {exit_code}."
            )

            combined_output = (res.stdout + "\n" + res.stderr).strip()
            security_event = None

            if len(combined_output) > max_output_chars:
                combined_output = (
                    combined_output[:max_output_chars]
                    + f"\n... (test output truncated at {max_output_chars} characters)"
                )
                security_event = "output_truncated"
                self.log_event("output_truncated", f"Output truncated at {max_output_chars} characters")

            return {
                "status": status,
                "exit_code": exit_code,
                "summary": summary,
                "output": combined_output,
                "security_event": security_event,
            }

        except subprocess.TimeoutExpired as exc:
            self.log_event("timeout", f"Command '{' '.join(cmd)}' timed out after {timeout_seconds}s")
            output_text = (exc.stdout or "") if isinstance(exc.stdout, str) else ""
            if len(output_text) > max_output_chars:
                output_text = output_text[:max_output_chars] + f"\n... (truncated)"

            return {
                "status": "timeout",
                "exit_code": None,
                "summary": f"Test execution timed out after {timeout_seconds} seconds.",
                "output": f"TimeoutExpired: Process killed after exceeding {timeout_seconds}s limit.\n{output_text}",
                "security_event": "timeout",
            }
        except Exception as exc:
            return {
                "status": "error",
                "exit_code": None,
                "summary": "Error executing validation command.",
                "output": str(exc),
                "security_event": "execution_error",
            }

    # -------------------------------------------------------------------------
    # Safe Filesystem Operations
    # -------------------------------------------------------------------------
    def read_file(self, file_path: Union[str, Path], max_bytes: int = 100_000) -> str:
        """Safely read text file inside sandbox root."""
        try:
            resolved = self.safe_resolve_path(file_path)
        except SecurityError as err:
            return f"Error: Access denied: File operation rejected: path outside sandbox ({err})"

        if not resolved.exists():
            return f"Error: File '{file_path}' does not exist."

        if not resolved.is_file():
            return f"Error: Path '{file_path}' is a directory, not a file."

        try:
            with open(resolved, "rb") as f:
                chunk = f.read(1024)
                if b"\x00" in chunk:
                    return f"Error: File '{file_path}' appears to be binary and cannot be read as text."

            text = resolved.read_text(encoding="utf-8", errors="replace")
            if len(text.encode("utf-8")) > max_bytes:
                text = text[:max_bytes] + f"\n... (truncated file content at {max_bytes} bytes)"
            return text
        except Exception as exc:
            return f"Error reading file '{file_path}': {str(exc)}"

    def write_file(self, file_path: Union[str, Path], content: str) -> str:
        """Safely create or overwrite file inside sandbox root."""
        try:
            resolved = self.safe_resolve_path(file_path)
        except SecurityError as err:
            return f"Error: Access denied: File operation rejected: path outside sandbox ({err})"

        try:
            resolved.parent.mkdir(parents=True, exist_ok=True)
            resolved.write_text(content, encoding="utf-8")
            byte_len = len(content.encode("utf-8"))
            return f"Successfully wrote file '{file_path}' ({byte_len} bytes)."
        except Exception as exc:
            return f"Error writing file '{file_path}': {str(exc)}"

    def replace_in_file(self, file_path: Union[str, Path], old_text: str, new_text: str) -> str:
        """Safely replace unique text inside a file inside sandbox root."""
        try:
            resolved = self.safe_resolve_path(file_path)
        except SecurityError as err:
            return f"Error: Access denied: File operation rejected: path outside sandbox ({err})"


        if not resolved.exists():
            return f"Error: File '{file_path}' does not exist."

        if not resolved.is_file():
            return f"Error: Path '{file_path}' is a directory, not a file."

        try:
            with open(resolved, "rb") as f:
                chunk = f.read(1024)
                if b"\x00" in chunk:
                    return f"Error: File '{file_path}' appears to be binary and cannot be edited as text."

            content = resolved.read_text(encoding="utf-8", errors="replace")
            if old_text not in content:
                return f"Error: Target text to replace was not found in '{file_path}'."

            count = content.count(old_text)
            if count > 1:
                return (
                    f"Error: Ambiguous replacement target. Found {count} occurrences of target text "
                    f"in '{file_path}'. Please provide more unique surrounding context."
                )

            new_content = content.replace(old_text, new_text, 1)
            resolved.write_text(new_content, encoding="utf-8")
            return f"Successfully replaced target text in '{file_path}'."
        except Exception as exc:
            return f"Error editing file '{file_path}': {str(exc)}"

    # -------------------------------------------------------------------------
    # Safe Git Delivery Operations
    # -------------------------------------------------------------------------
    def git_current_branch(self) -> str:
        """Query current active Git branch name inside sandbox root."""
        res = self.run_command(["git", "rev-parse", "--abbrev-ref", "HEAD"], cwd=self.sandbox_root)
        if res["status"] == "passed" and res["output"]:
            branch = res["output"].strip().splitlines()[0]
            if branch and branch != "HEAD":
                return branch
        return "main"

    def git_create_branch(self, branch_name: str) -> str:
        """Safely create a new local feature branch without overwriting existing branches."""
        try:
            self.validate_branch_name(branch_name)
        except SecurityError as err:
            return f"Error: Access denied: {err}"

        res_check = self.run_command(["git", "branch"], cwd=self.sandbox_root)
        if res_check["status"] == "passed":
            existing = [b.replace("*", "").strip() for b in res_check["output"].splitlines()]
            if branch_name in existing:
                return f"Error: Branch '{branch_name}' already exists. Overwriting existing branches is prohibited."

        res = self.run_command(["git", "checkout", "-b", branch_name], cwd=self.sandbox_root)
        if res["status"] == "passed":
            return f"Successfully created and checked out feature branch '{branch_name}'."
        return f"Error creating branch '{branch_name}': {res['output']}"

    def git_commit(self, message: str, files: Optional[List[str]] = None) -> str:
        """Safely stage modified files and commit with validated commit message."""
        if files:
            for f in files:
                try:
                    self.safe_resolve_path(f)
                except SecurityError as err:
                    return f"Error: Access denied: File '{f}' escapes sandbox boundary ({err})"

        if not message or not message.strip():
            return "Error: Commit message cannot be empty."

        msg_clean = message.strip()

        status_res = self.run_command(["git", "status", "--short"], cwd=self.sandbox_root)
        if status_res["status"] != "passed" or not status_res["output"].strip():
            return "Error: Empty commit rejected. No modified files to commit."

        if files:
            for f in files:
                self.run_command(["git", "add", f], cwd=self.sandbox_root)
        else:
            self.run_command(["git", "add", "-u"], cwd=self.sandbox_root)

        staged_res = self.run_command(["git", "diff", "--cached", "--name-only"], cwd=self.sandbox_root)
        if not staged_res["output"].strip():
            return "Error: Empty commit rejected. No staged files for commit."

        res = self.run_command(["git", "commit", "-m", msg_clean], cwd=self.sandbox_root)
        if res["status"] == "passed":
            branch = self.git_current_branch()
            return f"Successfully committed changes to branch '{branch}' with message: '{msg_clean}'."
        return f"Error creating commit: {res['output']}"

    def git_push(self, remote: str = "origin", branch: Optional[str] = None) -> str:
        """Safely push local branch changes to remote repository."""
        target_branch = branch or self.git_current_branch()

        try:
            self.validate_branch_name(target_branch)
            self.validate_branch_name(remote)
        except SecurityError as err:
            return f"Error: Access denied: {err}"

        res = self.run_command(["git", "push", remote, target_branch], cwd=self.sandbox_root)
        clean_output = res["output"].replace(os.getenv("GITHUB_TOKEN", "XYZ_UNSET_TOKEN"), "[REDACTED]")

        if res["status"] == "passed":
            return f"Successfully pushed branch '{target_branch}' to remote '{remote}'."
        return f"Push output ({res['status']}): {clean_output}"

    def create_pull_request(self, title: str, body: str, head_branch: str, base_branch: str = "main") -> str:
        """Create structured pull request delivery representation."""
        if not title or not title.strip():
            return "Error: Pull request title cannot be empty."

        try:
            self.validate_branch_name(head_branch)
            self.validate_branch_name(base_branch)
        except SecurityError as err:
            return f"Error: Access denied: {err}"

        return (
            f"=== GitHub Pull Request Created ===\n"
            f"Title: {title.strip()}\n"
            f"Head Branch: {head_branch}\n"
            f"Base Branch: {base_branch}\n"
            f"Body:\n{body.strip()}\n"
            f"Status: Pull request created successfully."
        )

