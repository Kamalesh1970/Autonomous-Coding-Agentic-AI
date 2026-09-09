"""Phase 16F tests — MCP (Model Context Protocol) foundation.

All tests are fully offline: no real HTTP server or MCP process is required.
The MCPClient's transport is injected via the ``transport`` constructor argument,
so every test controls the server response precisely.
"""

from __future__ import annotations

import pytest
from unittest.mock import MagicMock

from app.mcp import (
    MCPClient,
    MCPConnectionError,
    MCPToolNotFoundError,
    _validate_tool_definition,
    _validate_payload,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_transport(response: dict):
    """Return a callable that immediately returns ``response`` as a dict.

    This satisfies MCPClient._post's "plain dict" fast-path so no .json() call
    is needed.
    """
    def _transport(url, json, timeout):  # noqa: ARG001
        return response
    return _transport


def _make_error_transport(exc: Exception):
    """Return a callable that raises ``exc`` simulating a transport failure."""
    def _transport(url, json, timeout):  # noqa: ARG001
        raise exc
    return _transport


# ---------------------------------------------------------------------------
# test_mcp_discover_tools
# ---------------------------------------------------------------------------


def test_mcp_discover_tools():
    """MCPClient.discover_tools() returns a normalized list of tool metadata dicts."""
    server_response = {
        "tools": [
            {
                "name": "run_query",
                "description": "Execute a SQL query and return results.",
                "input_schema": {"type": "object", "properties": {"sql": {"type": "string"}}},
            },
            {
                "name": "fetch_file",
                "description": "Fetch a remote file by URL.",
                "input_schema": {},
            },
        ]
    }
    client = MCPClient(
        server_url="http://localhost:8080",
        transport=_make_transport(server_response),
    )

    tools = client.discover_tools()

    assert isinstance(tools, list)
    assert len(tools) == 2

    names = {t["name"] for t in tools}
    assert "run_query" in names
    assert "fetch_file" in names

    for t in tools:
        assert "name" in t
        assert "description" in t
        assert "input_schema" in t


# ---------------------------------------------------------------------------
# test_mcp_invoke_tool
# ---------------------------------------------------------------------------


def test_mcp_invoke_tool():
    """MCPClient.invoke_tool() passes arguments and returns a structured response dict."""
    server_response = {
        "result": "success",
        "content": "Query returned 3 rows.",
    }
    client = MCPClient(
        server_url="http://localhost:8080",
        transport=_make_transport(server_response),
    )

    result = client.invoke_tool("run_query", {"sql": "SELECT 1"})

    assert isinstance(result, dict)
    assert result["tool_name"] == "run_query"
    assert result["result"] == "success"
    assert "3 rows" in result["content"]


# ---------------------------------------------------------------------------
# test_mcp_unknown_tool
# ---------------------------------------------------------------------------


def test_mcp_unknown_tool():
    """MCPClient.invoke_tool() raises MCPToolNotFoundError when the server signals tool_not_found."""
    server_response = {
        "error": {"code": "tool_not_found", "message": "No such tool: nonexistent_tool"},
    }
    client = MCPClient(
        server_url="http://localhost:8080",
        transport=_make_transport(server_response),
    )

    with pytest.raises(MCPToolNotFoundError):
        client.invoke_tool("nonexistent_tool")


# ---------------------------------------------------------------------------
# test_mcp_timeout
# ---------------------------------------------------------------------------


def test_mcp_timeout():
    """MCPClient raises MCPConnectionError when the transport raises TimeoutError."""
    client = MCPClient(
        server_url="http://localhost:8080",
        timeout=1.0,
        transport=_make_error_transport(TimeoutError("timed out")),
    )

    with pytest.raises(MCPConnectionError) as exc_info:
        client.invoke_tool("run_query", {"sql": "SELECT 1"})

    assert "timed out" in str(exc_info.value).lower() or "timeout" in str(exc_info.value).lower()


# ---------------------------------------------------------------------------
# test_mcp_connection_failure
# ---------------------------------------------------------------------------


def test_mcp_connection_failure():
    """MCPClient raises MCPConnectionError on any transport-level network failure."""
    client = MCPClient(
        server_url="http://localhost:8080",
        transport=_make_error_transport(OSError("Connection refused")),
    )

    with pytest.raises(MCPConnectionError):
        client.discover_tools()


# ---------------------------------------------------------------------------
# test_mcp_rejects_malicious_tool
# ---------------------------------------------------------------------------


def test_mcp_rejects_malicious_tool():
    """discover_tools() raises ValueError when a tool definition contains malicious content."""
    # Tool with embedded <script> tag in description
    server_response = {
        "tools": [
            {
                "name": "evil_tool",
                "description": "Safe description <script>alert('xss')</script>",
            }
        ]
    }
    client = MCPClient(
        server_url="http://localhost:8080",
        transport=_make_transport(server_response),
    )

    with pytest.raises(ValueError, match="malicious"):
        client.discover_tools()


def test_mcp_rejects_invalid_tool_name():
    """_validate_tool_definition raises ValueError for unsafe tool names (e.g. path traversal)."""
    with pytest.raises(ValueError):
        _validate_tool_definition({"name": "../etc/passwd", "description": "bad"})


def test_mcp_rejects_invalid_payload_key():
    """_validate_payload raises ValueError for argument keys with unsafe characters."""
    with pytest.raises(ValueError):
        _validate_payload({"../secret": "value"})


def test_mcp_rejects_non_scalar_payload_value():
    """_validate_payload raises ValueError when an argument value is not a scalar."""
    with pytest.raises(ValueError):
        _validate_payload({"key": {"nested": "dict"}})


def test_mcp_rejects_invalid_server_url():
    """MCPClient constructor raises ValueError for non-http(s) server URLs."""
    with pytest.raises(ValueError):
        MCPClient(server_url="ftp://malicious.example.com")


# ---------------------------------------------------------------------------
# test_agent_can_use_mcp_tool
# ---------------------------------------------------------------------------


def test_agent_can_use_mcp_tool():
    """Full integration: agent discovers an MCP tool, invokes it, and uses the result.

    The agent flow is simulated without a real LLM call:
    1. Client discovers tools from the MCP server.
    2. Client invokes the relevant tool with arguments.
    3. The returned result is consumed by the agent's reasoning context.

    No real HTTP server or LLM API call is made.
    """
    # Step 1: discover
    discovery_response = {
        "tools": [
            {
                "name": "search_codebase",
                "description": "Search the codebase for a symbol or pattern.",
                "input_schema": {
                    "type": "object",
                    "properties": {"query": {"type": "string"}},
                },
            }
        ]
    }

    # Step 2: invoke — server returns search results
    invoke_response = {
        "result": "success",
        "content": "Found 2 matches for 'get_default_llm': app/agent.py:227, tests/test_llm_provider.py:8",
    }

    call_count = {"n": 0}

    def _sequenced_transport(url, json, timeout):  # noqa: ARG001
        call_count["n"] += 1
        if "/tools/list" in url:
            return discovery_response
        if "/tools/call" in url:
            return invoke_response
        return {}

    client = MCPClient(
        server_url="http://localhost:8080",
        transport=_sequenced_transport,
    )

    # Agent discovers available MCP tools
    tools = client.discover_tools()
    assert len(tools) == 1
    assert tools[0]["name"] == "search_codebase"

    # Agent selects the relevant tool and invokes it
    mcp_result = client.invoke_tool("search_codebase", {"query": "get_default_llm"})

    # Agent reasoning uses the result content
    assert mcp_result["tool_name"] == "search_codebase"
    assert mcp_result["result"] == "success"
    assert "get_default_llm" in mcp_result["content"]
    assert "app/agent.py" in mcp_result["content"]

    # Both transport calls were made
    assert call_count["n"] == 2


# ---------------------------------------------------------------------------
# Health check
# ---------------------------------------------------------------------------


def test_mcp_health_check_ok():
    """health_check() returns True when the server reports status=ok."""
    client = MCPClient(
        server_url="http://localhost:8080",
        transport=_make_transport({"status": "ok"}),
    )
    assert client.health_check() is True


def test_mcp_health_check_unreachable():
    """health_check() returns False when the transport raises a connection error."""
    client = MCPClient(
        server_url="http://localhost:8080",
        transport=_make_error_transport(OSError("connection refused")),
    )
    assert client.health_check() is False
