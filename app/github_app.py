"""GitHub App Foundation for Phase 16G Autonomous Coding Agent.

Provides a lightweight, transport-injectable GitHub App integration that connects
GitHub issues to the agent's existing workflow:

  GitHub Issue → GitHubTask → run_agent() → analyze/plan/code/test/verify
                                                  ↓
                                         human approval (existing gate)
                                                  ↓
                                         GitHubAppClient.create_pull_request()

Security principles:
  - Credentials are NEVER logged or included in traces.
  - Issue/webhook content is treated as UNTRUSTED INPUT.
  - Branch names are sanitized and validated via the existing sandbox.
  - Automatic merge is explicitly prohibited.
  - Repository boundary is enforced; arbitrary repository switching is rejected.
  - Prompt injection through issue bodies is mitigated by not executing issue content.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Any, Callable, Optional


# ---------------------------------------------------------------------------
# Custom Exceptions
# ---------------------------------------------------------------------------


class GitHubConfigError(ValueError):
    """Raised when required GitHub App configuration is missing or invalid."""


class GitHubAPIError(RuntimeError):
    """Raised when the GitHub API returns an unexpected error response."""


class GitHubTimeoutError(TimeoutError):
    """Raised when a GitHub API request times out."""


class GitHubRepositoryBoundaryError(ValueError):
    """Raised when a GitHub operation attempts to access an unauthorized repository."""


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class GitHubConfig:
    """Immutable, validated GitHub App configuration read from environment variables.

    Fields are deliberately opaque — the object never exposes raw credential
    values through __repr__ or __str__.

    Attributes:
        owner:     GitHub organisation or user name.
        repo_name: Repository name.
        _token:    Private: personal-access or GitHub App installation token.
                   Use has_token() to check presence; never access directly in logs.
    """

    owner: str
    repo_name: str
    _token: str = field(repr=False, compare=False)

    def __repr__(self) -> str:
        return f"GitHubConfig(owner={self.owner!r}, repo={self.repo_name!r}, token=[REDACTED])"

    def __str__(self) -> str:
        return self.__repr__()

    def has_token(self) -> bool:
        """Return True if a non-empty token is configured."""
        return bool(self._token)

    @property
    def repo_full_name(self) -> str:
        """Return 'owner/repo_name' string."""
        return f"{self.owner}/{self.repo_name}"


def load_github_config(env: Optional[dict[str, str]] = None) -> GitHubConfig:
    """Load and validate GitHub configuration from environment variables.

    Args:
        env: Optional dict override (useful in tests).  Falls back to os.environ.

    Returns:
        Validated GitHubConfig instance.

    Raises:
        GitHubConfigError: If any required variable is missing or empty.
    """
    import os

    source = env if env is not None else os.environ
    missing = []

    token = source.get("GITHUB_TOKEN", "").strip()
    if not token:
        missing.append("GITHUB_TOKEN")

    owner = source.get("GITHUB_OWNER", "").strip()
    if not owner:
        missing.append("GITHUB_OWNER")

    repo = source.get("GITHUB_REPO", "").strip()
    if not repo:
        missing.append("GITHUB_REPO")

    if missing:
        raise GitHubConfigError(
            f"Missing required GitHub configuration: {', '.join(missing)}. "
            "Set the corresponding environment variables before using GitHubAppClient."
        )

    return GitHubConfig(owner=owner, repo_name=repo, _token=token)


# ---------------------------------------------------------------------------
# GitHub Task (Issue → Agent Task)
# ---------------------------------------------------------------------------


@dataclass
class GitHubTask:
    """A sanitized internal representation of a GitHub issue as an agent task.

    Issue body is stored as-is but is NEVER executed or interpreted as
    agent instructions.  The agent goal derived from this task is constructed
    by the agent's own planning logic; it does not blindly follow issue text.

    Attributes:
        repo_full_name: 'owner/repo' string identifying the repository.
        issue_number:   Numeric GitHub issue identifier.
        title:          Issue title (sanitized).
        body:           Issue body (treated as UNTRUSTED INPUT — informational only).
        owner:          Repository owner.
        repo_name:      Repository name.
    """

    repo_full_name: str
    issue_number: int
    title: str
    body: str
    owner: str
    repo_name: str

    def to_agent_goal(self) -> str:
        """Return a safe agent goal string derived from the issue.

        The goal string intentionally avoids quoting the raw issue body to
        reduce prompt-injection surface area.  The agent will read the body
        from the task metadata if needed.
        """
        return (
            f"Resolve GitHub issue #{self.issue_number} in repository "
            f"'{self.repo_full_name}': {self.title}"
        )


def parse_github_issue(payload: dict[str, Any], config: GitHubConfig) -> GitHubTask:
    """Convert a GitHub issue API payload into a validated GitHubTask.

    Args:
        payload: Raw issue dict as returned by GitHub API (treated as untrusted).
        config:  Validated GitHubConfig specifying the authorised repository.

    Returns:
        Sanitized GitHubTask.

    Raises:
        ValueError: If the payload is missing required fields.
        GitHubRepositoryBoundaryError: If payload identifies a different repository.
    """
    if not isinstance(payload, dict):
        raise ValueError("GitHub issue payload must be a dict.")

    number = payload.get("number")
    if not isinstance(number, int) or number <= 0:
        raise ValueError(
            f"GitHub issue payload missing valid 'number' field (got {number!r})."
        )

    title = payload.get("title", "")
    if not isinstance(title, str) or not title.strip():
        raise ValueError("GitHub issue payload missing 'title' field.")

    body = payload.get("body", "") or ""
    if not isinstance(body, str):
        body = str(body)

    # Truncate extremely long bodies to avoid memory issues
    if len(body) > 16_000:
        body = body[:16_000] + "\n... [body truncated]"

    # Repository boundary enforcement: if payload specifies a repo, it must match config
    repo_info = payload.get("repository", {}) or {}
    if repo_info:
        payload_owner = (repo_info.get("owner", {}) or {}).get("login", "") or ""
        payload_repo = repo_info.get("name", "") or ""
        if payload_owner and payload_repo:
            if payload_owner != config.owner or payload_repo != config.repo_name:
                raise GitHubRepositoryBoundaryError(
                    f"Issue payload repository '{payload_owner}/{payload_repo}' does not "
                    f"match authorized repository '{config.repo_full_name}'. "
                    "Cross-repository access is prohibited."
                )

    return GitHubTask(
        repo_full_name=config.repo_full_name,
        issue_number=number,
        title=title.strip()[:500],  # Sanitize length
        body=body,
        owner=config.owner,
        repo_name=config.repo_name,
    )


# ---------------------------------------------------------------------------
# Branch Name Sanitization
# ---------------------------------------------------------------------------

_UNSAFE_BRANCH_CHARS = re.compile(r"[^a-zA-Z0-9._/-]")
_MULTI_SEPARATOR = re.compile(r"[-_]{2,}")


def sanitize_branch_name(issue_number: int, title: str) -> str:
    """Produce a safe agent branch name for a GitHub issue.

    Format: agent/issue-<number>-<slug>

    Args:
        issue_number: GitHub issue number.
        title:        Issue title (sanitized into a slug).

    Returns:
        Safe branch name string.

    Raises:
        ValueError: If the resulting branch name would be unsafe.
    """
    # Build slug from title
    slug = title.lower()
    slug = _UNSAFE_BRANCH_CHARS.sub("-", slug)
    slug = _MULTI_SEPARATOR.sub("-", slug)
    slug = slug.strip("-")[:50]

    branch = f"agent/issue-{issue_number}-{slug}"

    # Validate using sandbox rules (reuse existing security logic)
    from app.sandbox import ExecutionSandbox, SecurityError

    sandbox = ExecutionSandbox(sandbox_root=".")
    try:
        sandbox.validate_branch_name(branch)
    except SecurityError as err:
        raise ValueError(f"Generated branch name is unsafe: {err}") from err

    return branch


# ---------------------------------------------------------------------------
# Webhook Parsing
# ---------------------------------------------------------------------------

#: GitHub event types this foundation supports
SUPPORTED_WEBHOOK_EVENTS = frozenset({"issues", "issue_comment", "pull_request"})


def parse_webhook_event(
    event_type: str,
    payload: dict[str, Any],
    config: GitHubConfig,
) -> Optional[GitHubTask]:
    """Parse a raw GitHub webhook event into an internal GitHubTask.

    Webhook content is treated as UNTRUSTED INPUT.  Only supported, well-formed
    events that target the authorised repository are converted to tasks.

    Args:
        event_type: Value of the ``X-GitHub-Event`` header.
        payload:    Parsed JSON body of the webhook request.
        config:     Validated GitHubConfig for boundary enforcement.

    Returns:
        GitHubTask if the event should trigger an agent task, or None if the
        event is valid but does not require agent action (e.g. PR comments).

    Raises:
        ValueError: If the payload is malformed or the event is unsupported.
        GitHubRepositoryBoundaryError: If the payload targets a different repo.
    """
    if not isinstance(event_type, str) or not event_type.strip():
        raise ValueError("Webhook event_type must be a non-empty string.")

    if not isinstance(payload, dict):
        raise ValueError("Webhook payload must be a dict.")

    if event_type not in SUPPORTED_WEBHOOK_EVENTS:
        raise ValueError(
            f"Unsupported webhook event type: {event_type!r}. "
            f"Supported events: {sorted(SUPPORTED_WEBHOOK_EVENTS)}"
        )

    # Validate required 'action' field is present for all events
    action = payload.get("action")
    if not isinstance(action, str):
        raise ValueError(
            f"Webhook payload missing 'action' field for event '{event_type}'."
        )

    # Repository boundary check via payload repository field
    repo_info = payload.get("repository", {}) or {}
    payload_owner = (repo_info.get("owner", {}) or {}).get("login", "") or ""
    payload_repo = repo_info.get("name", "") or ""
    if payload_owner and payload_repo:
        if payload_owner != config.owner or payload_repo != config.repo_name:
            raise GitHubRepositoryBoundaryError(
                f"Webhook payload repository '{payload_owner}/{payload_repo}' does not "
                f"match authorized repository '{config.repo_full_name}'."
            )

    # Convert supported events to agent tasks
    if event_type == "issues" and action in ("opened", "reopened"):
        issue_payload = payload.get("issue", {})
        if not isinstance(issue_payload, dict):
            raise ValueError("Webhook 'issues' payload missing 'issue' field.")
        return parse_github_issue(issue_payload, config)

    # issue_comment and pull_request events do not automatically trigger new tasks
    # (but are recognized as valid)
    return None


# ---------------------------------------------------------------------------
# GitHub App Client
# ---------------------------------------------------------------------------


class GitHubAppClient:
    """Lightweight GitHub App client with injectable HTTP transport.

    Supports:
      - get_issue()          — fetch issue metadata
      - create_pull_request() — create a PR (requires prior human approval)
      - add_comment()        — post agent status comment on an issue or PR
      - Observability events recorded through ExecutionTrace

    The transport is injected so tests can pass mock callables without any
    real HTTP calls.  In production, the default transport uses ``requests``.

    Security:
      - GITHUB_TOKEN is NEVER included in log output or traces.
      - Automatic merge is explicitly prohibited.
      - Repository boundary is enforced on every operation.
      - All errors are sanitized before being returned.
    """

    def __init__(
        self,
        config: GitHubConfig,
        transport: Optional[Callable[..., Any]] = None,
        trace: Optional[Any] = None,
    ) -> None:
        """
        Args:
            config:    Validated GitHubConfig instance.
            transport: Optional callable used for HTTP requests.  Signature:
                       transport(method, url, headers, json, timeout) -> response_dict.
                       Defaults to a real ``requests``-based implementation.
            trace:     Optional ExecutionTrace instance for observability events.
        """
        self._config = config
        self._transport = transport
        self._trace = trace
        self._api_base = "https://api.github.com"

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def get_issue(self, issue_number: int, timeout: float = 30.0) -> dict[str, Any]:
        """Fetch issue metadata from GitHub API.

        Args:
            issue_number: GitHub issue number.
            timeout:      Request timeout in seconds.

        Returns:
            Dict with at minimum 'number', 'title', 'body', 'state' keys.

        Raises:
            GitHubAPIError:     On non-2xx HTTP response.
            GitHubTimeoutError: On request timeout.
        """
        if not isinstance(issue_number, int) or issue_number <= 0:
            raise ValueError(f"Invalid issue number: {issue_number!r}")

        self._record_event("github_issue_received", {"issue_number": issue_number})

        url = (
            f"{self._api_base}/repos/{self._config.repo_full_name}"
            f"/issues/{issue_number}"
        )
        response = self._request("GET", url, timeout=timeout)
        self._record_event(
            "github_repository_selected",
            {"repo": self._config.repo_full_name, "issue": issue_number},
        )
        return response

    def create_pull_request(
        self,
        title: str,
        body: str,
        head_branch: str,
        base_branch: str = "main",
        timeout: float = 30.0,
    ) -> dict[str, Any]:
        """Create a GitHub pull request.

        IMPORTANT: This method NEVER automatically merges the PR.
        Human approval must have been obtained before calling this method
        (enforced via the existing approval gate in tools.py / agent.py).

        Args:
            title:       PR title.
            body:        PR description (should include summary, changes, tests).
            head_branch: Feature branch containing changes.
            base_branch: Target branch for the PR (default 'main').
            timeout:     Request timeout in seconds.

        Returns:
            Dict with 'number', 'html_url', 'state' fields.

        Raises:
            ValueError:         If title or branches are invalid.
            GitHubAPIError:     On non-2xx HTTP response.
            GitHubTimeoutError: On request timeout.
        """
        if not title or not title.strip():
            raise ValueError("Pull request title cannot be empty.")

        # Validate branch names using existing sandbox logic
        from app.sandbox import ExecutionSandbox, SecurityError

        sandbox = ExecutionSandbox(sandbox_root=".")
        for branch in (head_branch, base_branch):
            try:
                sandbox.validate_branch_name(branch)
            except SecurityError as err:
                raise ValueError(f"Invalid branch name: {err}") from err

        # Explicitly reject auto-merge in body or title
        if "auto-merge" in title.lower() or "automerge" in title.lower():
            raise ValueError(
                "Auto-merge is prohibited. PR titles must not request automatic merging."
            )

        url = f"{self._api_base}/repos/{self._config.repo_full_name}/pulls"
        pr_data = {
            "title": title.strip(),
            "body": body.strip() if body else "",
            "head": head_branch,
            "base": base_branch,
        }

        response = self._request("POST", url, json_body=pr_data, timeout=timeout)
        pr_number = response.get("number")

        self._record_event(
            "github_pr_created",
            {
                "pr_number": pr_number,
                "head_branch": head_branch,
                "base_branch": base_branch,
                "repo": self._config.repo_full_name,
                "auto_merged": False,  # Never auto-merge
            },
        )
        return response

    def add_comment(
        self,
        issue_number: int,
        comment: str,
        timeout: float = 30.0,
    ) -> dict[str, Any]:
        """Post a comment on a GitHub issue or PR.

        Comment content must not include credentials, tokens, or raw errors.

        Args:
            issue_number: GitHub issue or PR number.
            comment:      Comment markdown body (sanitized before sending).
            timeout:      Request timeout in seconds.

        Returns:
            Dict with 'id', 'html_url' fields.

        Raises:
            ValueError:         If comment is empty.
            GitHubAPIError:     On non-2xx HTTP response.
            GitHubTimeoutError: On request timeout.
        """
        if not comment or not comment.strip():
            raise ValueError("Comment body cannot be empty.")

        # Sanitize comment to remove any accidental credential leakage
        from app.agent import sanitize_log_output

        safe_comment = sanitize_log_output(comment)

        url = (
            f"{self._api_base}/repos/{self._config.repo_full_name}"
            f"/issues/{issue_number}/comments"
        )
        response = self._request(
            "POST", url, json_body={"body": safe_comment}, timeout=timeout
        )

        self._record_event(
            "github_comment_created",
            {"issue_number": issue_number, "repo": self._config.repo_full_name},
        )
        return response

    # ------------------------------------------------------------------
    # Internal transport
    # ------------------------------------------------------------------

    def _get_headers(self) -> dict[str, str]:
        """Build authorization headers.  Token is never logged."""
        return {
            "Authorization": f"Bearer {self._config._token}",
            "Accept": "application/vnd.github+json",
            "X-GitHub-Api-Version": "2022-11-28",
        }

    def _request(
        self,
        method: str,
        url: str,
        json_body: Optional[dict] = None,
        timeout: float = 30.0,
    ) -> dict[str, Any]:
        """Execute an HTTP request through the injected transport.

        Raises:
            GitHubTimeoutError: On timeout.
            GitHubAPIError:     On non-2xx response or unexpected transport error.
        """
        headers = self._get_headers()
        transport = self._transport

        if transport is None:
            try:
                import requests as _requests  # type: ignore[import-untyped]

                def _default_transport(
                    method: str,
                    url: str,
                    headers: dict,
                    json: Optional[dict],
                    timeout: float,
                ) -> dict:
                    resp = _requests.request(
                        method, url, headers=headers, json=json, timeout=timeout
                    )
                    if not resp.ok:
                        raise GitHubAPIError(
                            f"GitHub API error {resp.status_code} for {method} {url}"
                        )
                    if resp.status_code == 204:
                        return {}
                    return resp.json()

                transport = _default_transport
            except ImportError as exc:
                raise GitHubAPIError(
                    "The 'requests' package is required for GitHubAppClient. "
                    "Install it with: pip install requests"
                ) from exc

        try:
            result = transport(
                method,
                url,
                headers=headers,
                json=json_body,
                timeout=timeout,
            )
        except GitHubTimeoutError:
            self._record_event("github_error", {"error": "timeout", "url": url})
            raise
        except GitHubAPIError:
            self._record_event("github_error", {"error": "api_error", "url": url})
            raise
        except TimeoutError as exc:
            self._record_event("github_error", {"error": "timeout", "url": url})
            raise GitHubTimeoutError(
                f"GitHub API request timed out after {timeout}s"
            ) from exc
        except Exception as exc:
            self._record_event("github_error", {"error": type(exc).__name__, "url": url})
            # Sanitize error message — do NOT leak URLs, tokens, or raw exception messages
            from app.agent import sanitize_log_output

            safe_err = sanitize_log_output(str(exc))
            raise GitHubAPIError(
                f"GitHub API request failed ({type(exc).__name__}): {safe_err}"
            ) from exc

        if not isinstance(result, dict):
            raise GitHubAPIError(
                f"GitHub API returned unexpected response type: {type(result).__name__}"
            )
        return result

    def _record_event(self, event_type: str, metadata: Optional[dict] = None) -> None:
        """Record an observability event if a trace is configured."""
        if self._trace is not None:
            try:
                self._trace.record_event(
                    event_type=event_type,
                    metadata=metadata or {},
                )
            except Exception:
                pass  # Observability failures must never break the main flow


# ---------------------------------------------------------------------------
# Agent Integration Helper
# ---------------------------------------------------------------------------


def create_github_task(
    issue_payload: dict[str, Any],
    config: GitHubConfig,
) -> GitHubTask:
    """Convert a raw GitHub API issue payload into a validated GitHubTask.

    This is the primary entry point for integrating GitHub issues with the
    existing agent workflow.  After calling this function, pass the result's
    `to_agent_goal()` output to `run_agent()`.

    Args:
        issue_payload: Raw GitHub issue dict (treated as UNTRUSTED INPUT).
        config:        Validated GitHubConfig for boundary enforcement.

    Returns:
        Validated GitHubTask ready for agent processing.
    """
    return parse_github_issue(issue_payload, config)


def build_pr_body(
    task: GitHubTask,
    summary: str,
    changes: list[str],
    tests_run: str,
    verification: str,
) -> str:
    """Build a structured, informative pull request body.

    Args:
        task:         The source GitHubTask.
        summary:      Brief description of what was changed.
        changes:      List of file paths or change descriptions.
        tests_run:    Test execution summary (e.g. '48 passed').
        verification: Goal verification status/summary.

    Returns:
        Formatted PR body markdown string.
    """
    changes_str = "\n".join(f"- {c}" for c in changes) if changes else "- No files modified."
    return (
        f"## Summary\n\n{summary}\n\n"
        f"## Changes Made\n\n{changes_str}\n\n"
        f"## Tests Executed\n\n{tests_run}\n\n"
        f"## Verification\n\n{verification}\n\n"
        f"## Reference\n\nCloses #{task.issue_number}\n"
    )


def build_agent_comment(
    task: GitHubTask,
    steps_completed: list[str],
    files_modified: int,
    tests_summary: str,
    pr_number: Optional[int] = None,
) -> str:
    """Build a concise, sanitized agent status comment for a GitHub issue.

    The comment never includes credentials, tokens, environment variables,
    internal errors, or sensitive tool payloads.

    Args:
        task:            The source GitHubTask.
        steps_completed: List of completed step descriptions.
        files_modified:  Number of files modified.
        tests_summary:   Test result summary string.
        pr_number:       Optional pull request number if PR was created.

    Returns:
        Formatted comment markdown string.
    """
    steps_str = "\n".join(f"✓ {s}" for s in steps_completed)
    pr_line = f"\nPR: #{pr_number}" if pr_number else ""
    return (
        f"**Autonomous Coding Agent**\n\n"
        f"Task: {task.to_agent_goal()}\n\n"
        f"{steps_str}\n\n"
        f"Files modified: {files_modified}\n"
        f"Tests: {tests_summary}"
        f"{pr_line}"
    )
