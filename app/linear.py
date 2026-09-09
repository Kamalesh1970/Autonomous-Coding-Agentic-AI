"""Linear Integration Foundation for Phase 16H Autonomous Coding Agent.

Provides a lightweight, transport-injectable Linear integration that connects
Linear issues to the agent's existing workflow:

  Linear Issue → LinearTask → run_agent() → analyze/plan/code/test/verify
                                                 ↓
                                        human approval (existing gate)
                                                 ↓
                                        LinearClient.update_issue()

Security principles:
  - API keys are NEVER logged or included in traces.
  - Issue title and description are treated as UNTRUSTED INPUT.
  - Prompt injection attempts in Linear descriptions are mitigated by not executing issue text as instructions.
  - Existing human approval, sandbox restrictions, and git delivery rules remain enforced.
  - Observability events are recorded via ExecutionTrace.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable, Optional


# ---------------------------------------------------------------------------
# Custom Exceptions
# ---------------------------------------------------------------------------


class LinearConfigError(ValueError):
    """Raised when required Linear configuration is missing or invalid."""


class LinearAPIError(RuntimeError):
    """Raised when the Linear API returns an unexpected error response."""


class LinearTimeoutError(TimeoutError):
    """Raised when a Linear API request times out."""


class LinearIssueError(ValueError):
    """Raised when a Linear issue payload is malformed or invalid."""


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class LinearConfig:
    """Immutable, validated Linear configuration read from environment variables.

    Fields are opaque — API key is redacted in __repr__ and __str__.

    Attributes:
        _api_key: Private: Linear API key (lin_api_...).
        team_id:  Optional Linear team ID/key.
    """

    _api_key: str = field(repr=False, compare=False)
    team_id: str = "ENG"

    def __repr__(self) -> str:
        return f"LinearConfig(team_id={self.team_id!r}, api_key=[REDACTED])"

    def __str__(self) -> str:
        return self.__repr__()

    def has_api_key(self) -> bool:
        """Return True if a non-empty API key is configured."""
        return bool(self._api_key)


def load_linear_config(env: Optional[dict[str, str]] = None) -> LinearConfig:
    """Load and validate Linear configuration from environment variables.

    Args:
        env: Optional dict override (useful in tests). Defaults to os.environ.

    Returns:
        Validated LinearConfig instance.

    Raises:
        LinearConfigError: If any required variable is missing or empty.
    """
    import os

    source = env if env is not None else os.environ
    missing = []

    api_key = source.get("LINEAR_API_KEY", "").strip()
    if not api_key:
        missing.append("LINEAR_API_KEY")

    team_id = source.get("LINEAR_TEAM_ID", "ENG").strip() or "ENG"

    if missing:
        raise LinearConfigError(
            f"Missing required Linear configuration: {', '.join(missing)}. "
            "Set the corresponding environment variables before using Linear integration."
        )

    return LinearConfig(_api_key=api_key, team_id=team_id)


# ---------------------------------------------------------------------------
# Linear Task (Linear Issue → Agent Task)
# ---------------------------------------------------------------------------


@dataclass
class LinearTask:
    """A sanitized internal representation of a Linear issue as an agent task.

    Issue description is stored as data but is NEVER executed as system instructions.
    Prompt injection attempts in title/description are contained.

    Attributes:
        issue_id:    Linear issue UUID or ID.
        identifier:  Linear issue human key (e.g. 'ENG-123').
        title:       Issue title (sanitized).
        description: Issue description (treated as UNTRUSTED INPUT).
        team_id:     Linear team ID or key.
    """

    issue_id: str
    identifier: str
    title: str
    description: str
    team_id: str

    def to_agent_goal(self) -> str:
        """Return a safe agent goal string derived from the Linear issue."""
        sanitized_title = self.title.strip()[:500]
        return f"Resolve Linear issue [{self.identifier}]: {sanitized_title}"


def parse_linear_issue(payload: dict[str, Any], config: LinearConfig) -> LinearTask:
    """Convert a raw Linear issue payload into a validated LinearTask.

    Args:
        payload: Raw issue dict (treated as untrusted).
        config:  Validated LinearConfig.

    Returns:
        Sanitized LinearTask.

    Raises:
        LinearIssueError: If payload is missing required fields.
    """
    if not isinstance(payload, dict):
        raise LinearIssueError("Linear issue payload must be a dict.")

    issue_data = payload.get("data", payload)
    if isinstance(issue_data, dict) and "issue" in issue_data:
        issue_data = issue_data["issue"]
    if not isinstance(issue_data, dict):
        raise LinearIssueError("Linear issue payload must contain issue dict.")

    issue_id = str(issue_data.get("id", "")).strip()
    if not issue_id:
        raise LinearIssueError("Linear issue payload missing 'id' field.")

    identifier = str(issue_data.get("identifier", issue_data.get("number", issue_id))).strip()
    if not identifier:
        raise LinearIssueError("Linear issue payload missing 'identifier' field.")

    title = issue_data.get("title", "")
    if not isinstance(title, str) or not title.strip():
        raise LinearIssueError("Linear issue payload missing 'title' field.")

    description = issue_data.get("description", "") or ""
    if not isinstance(description, str):
        description = str(description)

    # Truncate extremely long description to prevent memory overhead
    if len(description) > 16_000:
        description = description[:16_000] + "\n... [description truncated]"

    team_info = issue_data.get("team", {}) or {}
    team_id = (team_info.get("key", team_info.get("id", config.team_id)) if isinstance(team_info, dict) else config.team_id) or config.team_id

    return LinearTask(
        issue_id=issue_id,
        identifier=identifier,
        title=title.strip()[:500],
        description=description.strip(),
        team_id=str(team_id).strip(),
    )


# ---------------------------------------------------------------------------
# Linear Client
# ---------------------------------------------------------------------------


class LinearClient:
    """Lightweight Linear GraphQL/HTTP client with injectable transport.

    Supports:
      - get_issue()     — fetch issue metadata by ID/identifier
      - update_issue()  — post comment / update issue status
      - Observability events recorded via ExecutionTrace

    Transport is injectable for 100% offline mocking in tests.
    Credentials and API keys are NEVER exposed in logs.
    """

    def __init__(
        self,
        config: LinearConfig,
        transport: Optional[Callable[..., Any]] = None,
        trace: Optional[Any] = None,
    ) -> None:
        self._config = config
        self._transport = transport
        self._trace = trace
        self._api_base = "https://api.linear.app/graphql"

    def get_issue(self, issue_id: str, timeout: float = 30.0) -> dict[str, Any]:
        """Fetch Linear issue metadata.

        Args:
            issue_id: Linear issue ID or identifier.
            timeout:  Request timeout in seconds.

        Returns:
            Dict containing issue details ('id', 'identifier', 'title', 'description').

        Raises:
            ValueError:         If issue_id is empty.
            LinearAPIError:     On non-2xx or GraphQL error response.
            LinearTimeoutError: On request timeout.
        """
        if not issue_id or not issue_id.strip():
            raise ValueError("Linear issue_id cannot be empty.")

        self._record_event("linear_issue_received", {"issue_id": issue_id})

        query = """
        query GetIssue($id: String!) {
            issue(id: $id) {
                id
                identifier
                title
                description
                state { name }
            }
        }
        """
        response = self._request(
            "POST",
            self._api_base,
            json_body={"query": query, "variables": {"id": issue_id.strip()}},
            timeout=timeout,
        )

        data = response.get("data", {}).get("issue")
        if not data:
            self._record_event("linear_error", {"error": "issue_not_found", "issue_id": issue_id})
            raise LinearAPIError(f"Linear issue '{issue_id}' not found.")

        return data

    def update_issue(
        self,
        issue_id: str,
        comment: str,
        status: Optional[str] = None,
        timeout: float = 30.0,
    ) -> dict[str, Any]:
        """Post a comment and optionally update issue status.

        Args:
            issue_id: Linear issue ID.
            comment:  Comment text to post.
            status:   Optional state name (e.g. 'Completed').
            timeout:  Request timeout in seconds.

        Returns:
            Dict with 'success' and 'comment_id' fields.

        Raises:
            ValueError:         If issue_id or comment is empty.
            LinearAPIError:     On API error.
            LinearTimeoutError: On timeout.
        """
        if not issue_id or not issue_id.strip():
            raise ValueError("Linear issue_id cannot be empty.")
        if not comment or not comment.strip():
            raise ValueError("Linear comment cannot be empty.")

        from app.agent import sanitize_log_output
        safe_comment = sanitize_log_output(comment)

        mutation = """
        mutation CreateComment($issueId: String!, $body: String!) {
            commentCreate(input: { issueId: $issueId, body: $body }) {
                success
                comment { id }
            }
        }
        """
        response = self._request(
            "POST",
            self._api_base,
            json_body={
                "query": mutation,
                "variables": {"issueId": issue_id.strip(), "body": safe_comment.strip()},
            },
            timeout=timeout,
        )

        result = response.get("data", {}).get("commentCreate", {})
        if not result.get("success", True):
            self._record_event("linear_error", {"error": "comment_create_failed", "issue_id": issue_id})
            raise LinearAPIError("Linear failed to post comment on issue.")

        self._record_event(
            "linear_issue_updated",
            {"issue_id": issue_id, "status": status or "commented"},
        )
        return result

    def _get_headers(self) -> dict[str, str]:
        """Build authorization headers. API key is never logged."""
        return {
            "Authorization": self._config._api_key,
            "Content-Type": "application/json",
        }

    def _request(
        self,
        method: str,
        url: str,
        json_body: Optional[dict] = None,
        timeout: float = 30.0,
    ) -> dict[str, Any]:
        headers = self._get_headers()
        transport = self._transport

        if transport is None:
            try:
                import requests as _requests

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
                        raise LinearAPIError(
                            f"Linear API HTTP error {resp.status_code} for {method} {url}"
                        )
                    return resp.json()

                transport = _default_transport
            except ImportError as exc:
                raise LinearAPIError(
                    "The 'requests' package is required for LinearClient. "
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
        except LinearTimeoutError:
            self._record_event("linear_error", {"error": "timeout", "url": url})
            raise
        except LinearAPIError:
            self._record_event("linear_error", {"error": "api_error", "url": url})
            raise
        except TimeoutError as exc:
            self._record_event("linear_error", {"error": "timeout", "url": url})
            raise LinearTimeoutError(
                f"Linear API request timed out after {timeout}s"
            ) from exc
        except Exception as exc:
            self._record_event("linear_error", {"error": type(exc).__name__, "url": url})
            from app.agent import sanitize_log_output
            safe_err = sanitize_log_output(str(exc))
            raise LinearAPIError(
                f"Linear API request failed ({type(exc).__name__}): {safe_err}"
            ) from exc

        if not isinstance(result, dict):
            raise LinearAPIError(
                f"Linear API returned unexpected response type: {type(result).__name__}"
            )
        return result

    def _record_event(self, event_type: str, metadata: Optional[dict] = None) -> None:
        if self._trace is not None:
            try:
                self._trace.record_event(
                    event_type=event_type,
                    metadata=metadata or {},
                )
            except Exception:
                pass


# ---------------------------------------------------------------------------
# Helper Functions
# ---------------------------------------------------------------------------


def create_linear_task(payload: dict[str, Any], config: LinearConfig) -> LinearTask:
    """Primary entry point for converting raw Linear issue payload to a validated LinearTask."""
    return parse_linear_issue(payload, config)


def build_linear_comment(
    task: LinearTask,
    steps_completed: list[str],
    files_modified: int,
    tests_summary: str,
    retries: int = 0,
) -> str:
    """Build a formatted, concise Linear issue comment.

    Does not expose raw traces, internal errors, or credentials.
    """
    steps_str = "\n".join(f"✓ {s}" for s in steps_completed) if steps_completed else "✓ Issue processed"
    return (
        f"**Autonomous Coding Agent**\n\n"
        f"Task: {task.to_agent_goal()}\n\n"
        f"{steps_str}\n\n"
        f"Files modified: {files_modified}\n"
        f"Tests: {tests_summary}\n"
        f"Retries: {retries}"
    )
