"""Slack Integration Foundation for Phase 16H Autonomous Coding Agent.

Provides a lightweight, transport-injectable Slack integration that connects
Slack messages/events to the agent's existing workflow:

  Slack Event → SlackTask → run_agent() → analyze/plan/code/test/verify
                                                ↓
                                       human approval (existing gate)
                                                ↓
                                       SlackClient.send_message()

Security principles:
  - Credentials (tokens, secrets) are NEVER logged or included in traces.
  - Slack messages/events are treated as UNTRUSTED INPUT.
  - Prompt injection attempts in Slack messages are mitigated by not executing message text as instructions.
  - Existing human approval, sandbox restrictions, and git delivery rules remain enforced.
  - Observability events are recorded via ExecutionTrace.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable, Optional


# ---------------------------------------------------------------------------
# Custom Exceptions
# ---------------------------------------------------------------------------


class SlackConfigError(ValueError):
    """Raised when required Slack configuration is missing or invalid."""


class SlackAPIError(RuntimeError):
    """Raised when the Slack API returns an unexpected error response."""


class SlackTimeoutError(TimeoutError):
    """Raised when a Slack API request times out."""


class SlackEventError(ValueError):
    """Raised when a Slack event payload is malformed or invalid."""


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class SlackConfig:
    """Immutable, validated Slack configuration read from environment variables.

    Fields are opaque — credentials are redacted in __repr__ and __str__.

    Attributes:
        _bot_token:       Private: Bot User OAuth Token (xoxb-...).
        _signing_secret:  Private: Slack Signing Secret.
        default_channel: Optional default channel ID/name.
    """

    _bot_token: str = field(repr=False, compare=False)
    _signing_secret: str = field(repr=False, compare=False)
    default_channel: str = "general"

    def __repr__(self) -> str:
        return f"SlackConfig(default_channel={self.default_channel!r}, bot_token=[REDACTED], signing_secret=[REDACTED])"

    def __str__(self) -> str:
        return self.__repr__()

    def has_bot_token(self) -> bool:
        """Return True if a non-empty bot token is configured."""
        return bool(self._bot_token)

    def has_signing_secret(self) -> bool:
        """Return True if a non-empty signing secret is configured."""
        return bool(self._signing_secret)


def load_slack_config(env: Optional[dict[str, str]] = None) -> SlackConfig:
    """Load and validate Slack configuration from environment variables.

    Args:
        env: Optional dict override (useful in tests). Defaults to os.environ.

    Returns:
        Validated SlackConfig instance.

    Raises:
        SlackConfigError: If any required variable is missing or empty.
    """
    import os

    source = env if env is not None else os.environ
    missing = []

    bot_token = source.get("SLACK_BOT_TOKEN", "").strip()
    if not bot_token:
        missing.append("SLACK_BOT_TOKEN")

    signing_secret = source.get("SLACK_SIGNING_SECRET", "").strip()
    if not signing_secret:
        missing.append("SLACK_SIGNING_SECRET")

    default_channel = source.get("SLACK_DEFAULT_CHANNEL", "general").strip() or "general"

    if missing:
        raise SlackConfigError(
            f"Missing required Slack configuration: {', '.join(missing)}. "
            "Set the corresponding environment variables before using Slack integration."
        )

    return SlackConfig(
        _bot_token=bot_token,
        _signing_secret=signing_secret,
        default_channel=default_channel,
    )


# ---------------------------------------------------------------------------
# Slack Task (Slack Message → Agent Task)
# ---------------------------------------------------------------------------


@dataclass
class SlackTask:
    """A sanitized internal representation of a Slack message as an agent task.

    Message text is stored as data but is NEVER executed as system instructions.
    Prompt injection attempts in message content are contained.

    Attributes:
        channel:    Slack channel ID or name.
        message_id: Slack message timestamp (ts) or event ID.
        user_id:    Slack user ID who posted the message.
        text:       Slack message text (treated as UNTRUSTED INPUT).
    """

    channel: str
    message_id: str
    user_id: str
    text: str

    def to_agent_goal(self) -> str:
        """Return a safe agent goal string derived from the Slack message."""
        sanitized_text = self.text.strip()[:500]
        return f"Process Slack message in channel '{self.channel}': {sanitized_text}"


def parse_slack_message(payload: dict[str, Any], config: SlackConfig) -> SlackTask:
    """Convert a raw Slack message payload into a validated SlackTask.

    Args:
        payload: Raw message dict (treated as untrusted).
        config:  Validated SlackConfig.

    Returns:
        Sanitized SlackTask.

    Raises:
        SlackEventError: If the payload is missing required fields.
    """
    if not isinstance(payload, dict):
        raise SlackEventError("Slack message payload must be a dict.")

    channel = payload.get("channel", "") or config.default_channel
    if not isinstance(channel, str) or not channel.strip():
        raise SlackEventError("Slack message payload missing 'channel' field.")

    text = payload.get("text", "")
    if not isinstance(text, str) or not text.strip():
        raise SlackEventError("Slack message payload missing 'text' field.")

    message_id = str(payload.get("ts", payload.get("event_id", "msg_default"))).strip()
    user_id = str(payload.get("user", payload.get("user_id", "U_UNKNOWN"))).strip()

    # Truncate extremely long text to prevent memory overhead
    if len(text) > 16_000:
        text = text[:16_000] + "\n... [text truncated]"

    return SlackTask(
        channel=channel.strip(),
        message_id=message_id,
        user_id=user_id,
        text=text.strip(),
    )


# ---------------------------------------------------------------------------
# Webhook / Event Parsing
# ---------------------------------------------------------------------------

SUPPORTED_SLACK_EVENT_TYPES = frozenset({"message", "app_mention", "event_callback"})


def parse_slack_event(
    event_type: str,
    payload: dict[str, Any],
    config: SlackConfig,
) -> Optional[SlackTask]:
    """Parse a raw Slack event payload into an internal SlackTask.

    Args:
        event_type: Slack event type (header or wrapper event type).
        payload:    Parsed JSON payload of the event request.
        config:     Validated SlackConfig.

    Returns:
        SlackTask if event requires agent action, None otherwise.

    Raises:
        SlackEventError: If payload is malformed or event type is unsupported.
    """
    if not isinstance(event_type, str) or not event_type.strip():
        raise SlackEventError("Slack event_type must be a non-empty string.")

    if not isinstance(payload, dict):
        raise SlackEventError("Slack payload must be a dict.")

    # Handle outer event_callback wrapper
    outer_type = payload.get("type", event_type)
    if outer_type not in SUPPORTED_SLACK_EVENT_TYPES and event_type not in SUPPORTED_SLACK_EVENT_TYPES:
        raise SlackEventError(
            f"Unsupported Slack event type: {event_type!r}. "
            f"Supported event types: {sorted(SUPPORTED_SLACK_EVENT_TYPES)}"
        )

    # Handle Slack URL verification challenge
    if outer_type == "url_verification":
        return None

    event_data = payload.get("event", payload)
    if not isinstance(event_data, dict):
        raise SlackEventError("Slack payload missing valid 'event' dict.")

    inner_type = event_data.get("type", event_type)

    # Ignore bot messages to prevent infinite loops
    if event_data.get("bot_id") or event_data.get("subtype") == "bot_message":
        return None

    if inner_type in ("message", "app_mention"):
        return parse_slack_message(event_data, config)

    return None


# ---------------------------------------------------------------------------
# Slack Client
# ---------------------------------------------------------------------------


class SlackClient:
    """Lightweight Slack API client with injectable HTTP transport.

    Supports:
      - send_message()   — post chat message to Slack channel
      - add_reaction()   — add emoji reaction to a message
      - Observability events recorded via ExecutionTrace

    Transport is injectable for 100% offline mocking in tests.
    Credentials and authorization tokens are NEVER exposed in logs.
    """

    def __init__(
        self,
        config: SlackConfig,
        transport: Optional[Callable[..., Any]] = None,
        trace: Optional[Any] = None,
    ) -> None:
        self._config = config
        self._transport = transport
        self._trace = trace
        self._api_base = "https://slack.com/api"

    def send_message(
        self,
        channel: str,
        text: str,
        thread_ts: Optional[str] = None,
        timeout: float = 30.0,
    ) -> dict[str, Any]:
        """Post a message to a Slack channel.

        Args:
            channel:   Channel ID or name.
            text:      Message body text (markdown formatted).
            thread_ts: Optional parent message ts for threaded reply.
            timeout:   Request timeout in seconds.

        Returns:
            Dict with 'ok', 'channel', 'ts' fields.

        Raises:
            ValueError:        If channel or text is empty.
            SlackAPIError:     On non-ok response.
            SlackTimeoutError: On request timeout.
        """
        if not channel or not channel.strip():
            raise ValueError("Slack channel cannot be empty.")
        if not text or not text.strip():
            raise ValueError("Slack message text cannot be empty.")

        self._record_event("slack_event_received", {"channel": channel, "action": "send_message"})

        from app.agent import sanitize_log_output
        safe_text = sanitize_log_output(text)

        url = f"{self._api_base}/chat.postMessage"
        post_data: dict[str, Any] = {
            "channel": channel.strip(),
            "text": safe_text.strip(),
        }
        if thread_ts:
            post_data["thread_ts"] = thread_ts

        response = self._request("POST", url, json_body=post_data, timeout=timeout)

        if not response.get("ok", True):
            err_msg = response.get("error", "unknown_error")
            self._record_event("slack_error", {"error": err_msg, "channel": channel})
            raise SlackAPIError(f"Slack API error: {err_msg}")

        self._record_event("slack_response_sent", {"channel": channel, "ts": response.get("ts")})
        return response

    def add_reaction(
        self,
        channel: str,
        timestamp: str,
        name: str = "white_check_mark",
        timeout: float = 30.0,
    ) -> dict[str, Any]:
        """Add an emoji reaction to a Slack message."""
        if not channel or not timestamp:
            raise ValueError("Channel and timestamp are required for reaction.")

        url = f"{self._api_base}/reactions.add"
        post_data = {
            "channel": channel.strip(),
            "timestamp": timestamp.strip(),
            "name": name.strip(),
        }

        response = self._request("POST", url, json_body=post_data, timeout=timeout)
        return response

    def _get_headers(self) -> dict[str, str]:
        """Build authorization headers. Token is never logged."""
        return {
            "Authorization": f"Bearer {self._config._bot_token}",
            "Content-Type": "application/json; charset=utf-8",
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
                        raise SlackAPIError(
                            f"Slack API HTTP error {resp.status_code} for {method} {url}"
                        )
                    return resp.json()

                transport = _default_transport
            except ImportError as exc:
                raise SlackAPIError(
                    "The 'requests' package is required for SlackClient. "
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
        except SlackTimeoutError:
            self._record_event("slack_error", {"error": "timeout", "url": url})
            raise
        except SlackAPIError:
            self._record_event("slack_error", {"error": "api_error", "url": url})
            raise
        except TimeoutError as exc:
            self._record_event("slack_error", {"error": "timeout", "url": url})
            raise SlackTimeoutError(
                f"Slack API request timed out after {timeout}s"
            ) from exc
        except Exception as exc:
            self._record_event("slack_error", {"error": type(exc).__name__, "url": url})
            from app.agent import sanitize_log_output
            safe_err = sanitize_log_output(str(exc))
            raise SlackAPIError(
                f"Slack API request failed ({type(exc).__name__}): {safe_err}"
            ) from exc

        if not isinstance(result, dict):
            raise SlackAPIError(
                f"Slack API returned unexpected response type: {type(result).__name__}"
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


def create_slack_task(payload: dict[str, Any], config: SlackConfig) -> SlackTask:
    """Primary entry point for converting raw Slack payload to a validated SlackTask."""
    return parse_slack_message(payload, config)


def build_slack_response(
    task: SlackTask,
    steps_completed: list[str],
    files_modified: int,
    tests_summary: str,
    retries: int = 0,
) -> str:
    """Build a formatted, concise Slack markdown response.

    Does not expose raw traces, internal errors, or credentials.
    """
    steps_str = "\n".join(f"✓ {s}" for s in steps_completed) if steps_completed else "✓ Task processed"
    return (
        f"*Autonomous Coding Agent*\n\n"
        f"*Task:* {task.to_agent_goal()}\n\n"
        f"{steps_str}\n\n"
        f"*Files modified:* {files_modified}\n"
        f"*Tests:* {tests_summary}\n"
        f"*Retries:* {retries}"
    )
