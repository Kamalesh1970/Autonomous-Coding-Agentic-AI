"""MCP (Model Context Protocol) client foundation for Phase 16F.

Provides a lightweight, transport-agnostic MCPClient that the agent can use to:
  - discover_tools()  — enumerate tools exposed by an MCP server
  - invoke_tool()     — call a specific tool with arguments
  - health_check()    — verify the server is reachable

Transport is injected so tests can pass mock callables without a real HTTP server.
Security: tool definitions and payloads are validated before use; responses are
sanitized via sanitize_log_output() before being returned to the caller.
"""

from __future__ import annotations

import re
from typing import Any, Callable

# ---------------------------------------------------------------------------
# Custom exceptions
# ---------------------------------------------------------------------------


class MCPConnectionError(ConnectionError):
    """Raised when the network transport to an MCP server fails or times out."""


class MCPToolNotFoundError(KeyError):
    """Raised when the requested tool name is not present in the MCP server's registry."""


# ---------------------------------------------------------------------------
# Security constants
# ---------------------------------------------------------------------------

_SAFE_NAME_RE = re.compile(r"^[a-zA-Z0-9_-]{1,128}$")
_SAFE_KEY_RE = re.compile(r"^[a-zA-Z0-9_-]{1,64}$")
_MAX_DESCRIPTION_LEN = 2_000
_MAX_VALUE_STR_LEN = 8_000
_MAX_RESPONSE_BYTES = 1_024 * 1_024  # 1 MB
_ALLOWED_URL_SCHEMES = ("http://", "https://")


# ---------------------------------------------------------------------------
# Internal security validators
# ---------------------------------------------------------------------------


def _validate_tool_definition(tool: Any) -> None:
    """Validate a single tool definition dict received from an MCP server.

    Raises:
        ValueError: If the definition is structurally unsafe or contains
                    potentially malicious content.
    """
    if not isinstance(tool, dict):
        raise ValueError("MCP tool definition must be a dict.")

    name = tool.get("name")
    if not isinstance(name, str) or not _SAFE_NAME_RE.match(name):
        raise ValueError(
            f"MCP tool definition has an invalid or unsafe name: {name!r}. "
            "Tool names must match [a-zA-Z0-9_-] (1-128 chars)."
        )

    description = tool.get("description", "")
    if not isinstance(description, str):
        raise ValueError(f"MCP tool description for '{name}' must be a string.")
    if len(description) > _MAX_DESCRIPTION_LEN:
        raise ValueError(
            f"MCP tool description for '{name}' exceeds {_MAX_DESCRIPTION_LEN} chars."
        )

    # Reject embedded script/markup in descriptions (basic XSS/injection guard)
    if re.search(r"<\s*script|javascript:", description, re.IGNORECASE):
        raise ValueError(
            f"MCP tool description for '{name}' contains potentially malicious content."
        )


def _validate_payload(arguments: dict[str, Any]) -> None:
    """Validate argument payload before sending to an MCP server.

    Accepts only scalar values (str, int, float, bool, None) and enforces
    safe key naming and value size limits.

    Raises:
        ValueError: If any key or value is structurally unsafe.
    """
    if not isinstance(arguments, dict):
        raise ValueError("MCP tool arguments must be a dict.")

    for key, value in arguments.items():
        if not isinstance(key, str) or not _SAFE_KEY_RE.match(key):
            raise ValueError(
                f"MCP argument key {key!r} is invalid. "
                "Keys must match [a-zA-Z0-9_-] (1-64 chars)."
            )
        if not isinstance(value, (str, int, float, bool, type(None))):
            raise ValueError(
                f"MCP argument value for key '{key}' must be a scalar "
                f"(str/int/float/bool/None), got {type(value).__name__}."
            )
        if isinstance(value, str) and len(value) > _MAX_VALUE_STR_LEN:
            raise ValueError(
                f"MCP argument value for key '{key}' exceeds {_MAX_VALUE_STR_LEN} chars."
            )


# ---------------------------------------------------------------------------
# MCPClient
# ---------------------------------------------------------------------------


class MCPClient:
    """Lightweight MCP (Model Context Protocol) client.

    Args:
        server_url: Base URL of the MCP server (must start with http:// or https://).
        timeout:    Per-request timeout in seconds (default 30).
        transport:  Optional callable used as the HTTP POST transport. Injected in
                    tests; defaults to ``requests.post`` when None.

    The client is stateless between calls -- no persistent connection is held.
    All MCP responses are treated as untrusted and sanitized before being returned.
    """

    def __init__(
        self,
        server_url: str,
        timeout: float = 30.0,
        transport: Callable[..., Any] | None = None,
    ) -> None:
        if not isinstance(server_url, str) or not any(
            server_url.startswith(scheme) for scheme in _ALLOWED_URL_SCHEMES
        ):
            raise ValueError(
                f"MCP server_url must start with http:// or https://, got: {server_url!r}"
            )
        self._server_url = server_url.rstrip("/")
        self._timeout = timeout
        self._transport = transport  # injected; real default resolved lazily

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def health_check(self) -> bool:
        """Return True if the MCP server is reachable, False otherwise."""
        try:
            resp = self._post("/health", {})
            return resp.get("status") in ("ok", "healthy", True, "true", 200)
        except (MCPConnectionError, ValueError):
            return False

    def discover_tools(self) -> list[dict]:
        """Query the MCP server for its tool registry.

        Returns:
            A list of normalized tool metadata dicts, each with at least
            ``name`` and ``description`` keys.

        Raises:
            MCPConnectionError: If the transport call fails.
            ValueError:         If any returned tool definition is invalid/malicious.
        """
        response = self._post("/tools/list", {})
        raw_tools = response.get("tools", [])
        if not isinstance(raw_tools, list):
            raise ValueError("MCP /tools/list response 'tools' field must be a list.")

        normalized: list[dict] = []
        for tool in raw_tools:
            _validate_tool_definition(tool)
            normalized.append(
                {
                    "name": tool["name"],
                    "description": tool.get("description", ""),
                    "input_schema": tool.get("input_schema", {}),
                }
            )
        return normalized

    def invoke_tool(self, tool_name: str, arguments: dict[str, Any] | None = None) -> dict:
        """Invoke a named MCP tool with optional arguments.

        Args:
            tool_name:  Name of the tool to invoke (must pass safe-name validation).
            arguments:  Dict of scalar argument key/value pairs (default empty).

        Returns:
            A dict with at least ``tool_name``, ``result``, and ``content`` keys.
            ``content`` is sanitized before being returned.

        Raises:
            ValueError:           If tool_name or arguments are invalid.
            MCPToolNotFoundError: If the server reports the tool does not exist.
            MCPConnectionError:   If the transport call fails.
        """
        if not isinstance(tool_name, str) or not _SAFE_NAME_RE.match(tool_name):
            raise ValueError(
                f"Invalid MCP tool name: {tool_name!r}. "
                "Names must match [a-zA-Z0-9_-] (1-128 chars)."
            )
        args = arguments or {}
        _validate_payload(args)

        response = self._post("/tools/call", {"name": tool_name, "arguments": args})

        # Detect tool-not-found signals from the server
        error = response.get("error") or {}
        if isinstance(error, dict) and error.get("code") in ("tool_not_found", "NOT_FOUND"):
            raise MCPToolNotFoundError(
                f"MCP server: tool '{tool_name}' was not found."
            )
        if isinstance(error, str) and "not found" in error.lower():
            raise MCPToolNotFoundError(
                f"MCP server: tool '{tool_name}' was not found."
            )

        # Sanitize content before returning (no credentials should leak)
        from app.agent import sanitize_log_output

        raw_content = str(response.get("content", ""))
        if len(raw_content.encode()) > _MAX_RESPONSE_BYTES:
            raw_content = raw_content[: _MAX_RESPONSE_BYTES // 4] + "... [truncated]"
        safe_content = sanitize_log_output(raw_content)

        return {
            "tool_name": tool_name,
            "result": response.get("result", "success"),
            "content": safe_content,
        }

    # ------------------------------------------------------------------
    # Internal transport
    # ------------------------------------------------------------------

    def _post(self, path: str, body: dict) -> dict:
        """POST ``body`` to ``self._server_url + path`` and return parsed JSON.

        Raises:
            MCPConnectionError: On network errors or timeouts.
            ValueError:         On unexpected response format.
        """
        url = self._server_url + path
        transport = self._transport

        if transport is None:
            try:
                import requests as _requests  # type: ignore[import-untyped]
                transport = _requests.post
            except ImportError as exc:
                raise MCPConnectionError(
                    "The 'requests' package is required for MCPClient HTTP transport. "
                    "Install it with: pip install requests"
                ) from exc

        try:
            raw_response = transport(
                url,
                json=body,
                timeout=self._timeout,
            )
        except TimeoutError as exc:
            raise MCPConnectionError(
                f"MCP server timed out after {self._timeout}s: {url}"
            ) from exc
        except Exception as exc:
            # Treat any transport-level failure as a connection error;
            # do NOT expose the raw exception message (may contain URL/secrets).
            err_type = type(exc).__name__
            raise MCPConnectionError(
                f"MCP transport error ({err_type}) connecting to {url}"
            ) from exc

        # Accept both real response objects (with .json()) and plain dicts (mocks)
        if isinstance(raw_response, dict):
            return raw_response

        if hasattr(raw_response, "json"):
            try:
                return raw_response.json()
            except Exception as exc:
                raise ValueError(
                    f"MCP server returned non-JSON response from {url}"
                ) from exc

        raise ValueError(
            f"Unexpected MCP transport return type: {type(raw_response).__name__}"
        )
