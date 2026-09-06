"""Tests for configurable LLM provider selection and Gemini key failover behavior."""

import os
import pytest
from unittest.mock import MagicMock
from langchain_core.messages import AIMessage
from langchain_openai import ChatOpenAI
from app.agent import get_default_llm, FailoverChatModel, is_retryable_error
from app.tools import create_workspace_tools


def test_gemini_36_flash_default_model(monkeypatch):
    """Verify gemini-3.6-flash is the default model for Gemini provider."""
    monkeypatch.setenv("LLM_PROVIDER", "gemini")
    monkeypatch.setenv("GEMINI_API_KEY_1", "test-gemini-key-1")
    monkeypatch.delenv("LLM_MODEL", raising=False)
    monkeypatch.delenv("GEMINI_MODEL_NAME", raising=False)

    llm = get_default_llm()
    if isinstance(llm, FailoverChatModel):
        target = llm.candidates[0]
    else:
        target = llm
    model_val = getattr(target, "model", getattr(target, "model_name", None))
    assert model_val == "gemini-3.6-flash"


def test_one_gemini_key_succeeds(monkeypatch):
    """Verify single Gemini key invocation succeeds."""
    monkeypatch.setenv("LLM_PROVIDER", "gemini")
    monkeypatch.setenv("GEMINI_API_KEY_1", "test-gemini-key-1")
    monkeypatch.setenv("GEMINI_API_KEY_2", "")
    monkeypatch.setenv("GEMINI_API_KEY_3", "")
    monkeypatch.setenv("GEMINI_API_KEY", "")
    monkeypatch.setenv("OPENROUTER_API_KEY", "")

    llm = get_default_llm()
    key_val = getattr(llm, "google_api_key", getattr(llm, "openai_api_key", None))
    if hasattr(key_val, "get_secret_value"):
        key_val = key_val.get_secret_value()
    assert key_val == "test-gemini-key-1"


def test_key_1_429_fails_over_to_key_2(monkeypatch):
    """Verify Key 1 receiving a 429 quota error fails over to Key 2."""
    candidate1 = MagicMock()
    candidate1._generate.side_effect = Exception("Error code: 429 - quota exceeded")
    
    candidate2 = MagicMock()
    candidate2._generate.return_value = "Success Response"

    failover_model = FailoverChatModel(candidates=[candidate1, candidate2])
    res = failover_model._generate(messages=[])
    assert res == "Success Response"
    assert candidate1._generate.call_count == 1
    assert candidate2._generate.call_count == 1


def test_key_1_and_2_fail_key_3_succeeds():
    """Verify Key 1 and Key 2 failing with 429 fail over to Key 3."""
    c1 = MagicMock()
    c1._generate.side_effect = Exception("429 rate limit")
    c2 = MagicMock()
    c2._generate.side_effect = Exception("429 resource_exhausted")
    c3 = MagicMock()
    c3._generate.return_value = "Key 3 Response"

    model = FailoverChatModel(candidates=[c1, c2, c3])
    res = model._generate(messages=[])
    assert res == "Key 3 Response"


def test_all_gemini_keys_fail_openrouter_fallback_succeeds():
    """Verify all Gemini keys failing with 429 fail over to OpenRouter fallback."""
    gemini1 = MagicMock()
    gemini1._generate.side_effect = Exception("429 quota")
    gemini2 = MagicMock()
    gemini2._generate.side_effect = Exception("429 quota")
    openrouter = MagicMock()
    openrouter._generate.return_value = "OpenRouter Fallback Response"

    model = FailoverChatModel(candidates=[gemini1, gemini2, openrouter])
    res = model._generate(messages=[])
    assert res == "OpenRouter Fallback Response"


def test_all_keys_fail_openrouter_unavailable_raises_error():
    """Verify all keys failing with no further candidates raises the last provider failure."""
    c1 = MagicMock()
    c1._generate.side_effect = Exception("429 quota 1")
    c2 = MagicMock()
    c2._generate.side_effect = Exception("429 quota 2")

    model = FailoverChatModel(candidates=[c1, c2])
    with pytest.raises(Exception, match="429 quota 2"):
        model._generate(messages=[])


def test_missing_key_2_is_skipped(monkeypatch):
    """Verify missing GEMINI_API_KEY_2 results in Key 1 -> Key 3 sequence."""
    monkeypatch.setenv("LLM_PROVIDER", "gemini")
    monkeypatch.setenv("GEMINI_API_KEY_1", "key-1")
    monkeypatch.setenv("GEMINI_API_KEY_2", "")
    monkeypatch.setenv("GEMINI_API_KEY_3", "key-3")
    monkeypatch.setenv("GEMINI_API_KEY", "")
    monkeypatch.setenv("OPENROUTER_API_KEY", "")

    llm = get_default_llm()
    assert isinstance(llm, FailoverChatModel)
    assert len(llm.candidates) == 2
    k0 = getattr(llm.candidates[0], "google_api_key", getattr(llm.candidates[0], "openai_api_key", None))
    if hasattr(k0, "get_secret_value"):
        k0 = k0.get_secret_value()
    k1 = getattr(llm.candidates[1], "google_api_key", getattr(llm.candidates[1], "openai_api_key", None))
    if hasattr(k1, "get_secret_value"):
        k1 = k1.get_secret_value()
    assert k0 == "key-1"
    assert k1 == "key-3"


def test_non_retryable_404_does_not_rotate():
    """Verify non-retryable 404 model error fails immediately without key rotation."""
    c1 = MagicMock()
    c1._generate.side_effect = Exception("404 model models/gemini-2.5-flash is no longer available")
    c2 = MagicMock()

    model = FailoverChatModel(candidates=[c1, c2])
    with pytest.raises(Exception, match="404"):
        model._generate(messages=[])

    assert c1._generate.call_count == 1
    assert c2._generate.call_count == 0


def test_non_retryable_400_malformed_request_does_not_rotate():
    """Verify non-retryable 400 malformed request fails immediately without key rotation."""
    c1 = MagicMock()
    c1._generate.side_effect = Exception("400 invalid_argument malformed request")
    c2 = MagicMock()

    model = FailoverChatModel(candidates=[c1, c2])
    with pytest.raises(Exception, match="400"):
        model._generate(messages=[])

    assert c1._generate.call_count == 1
    assert c2._generate.call_count == 0


def test_explicit_openrouter_provider(monkeypatch):
    """Verify explicit LLM_PROVIDER=openrouter uses OpenRouter only."""
    monkeypatch.setenv("LLM_PROVIDER", "openrouter")
    monkeypatch.setenv("OPENROUTER_API_KEY", "test-openrouter-key")
    monkeypatch.setenv("LLM_MODEL", "openai/gpt-4o-mini")

    llm = get_default_llm()
    assert isinstance(llm, ChatOpenAI)
    assert llm.openai_api_key.get_secret_value() == "test-openrouter-key"
    assert "openrouter.ai/api/v1" in str(llm.openai_api_base)


def test_explicit_openai_provider(monkeypatch):
    """Verify explicit LLM_PROVIDER=openai uses OpenAI only."""
    monkeypatch.setenv("LLM_PROVIDER", "openai")
    monkeypatch.setenv("OPENAI_API_KEY", "test-openai-key")
    monkeypatch.setenv("LLM_MODEL", "gpt-4o-mini")

    llm = get_default_llm()
    assert isinstance(llm, ChatOpenAI)
    assert llm.openai_api_key.get_secret_value() == "test-openai-key"


def test_api_keys_never_appear_in_logs_or_errors(monkeypatch):
    """Verify confidential API keys are never included in failover log outputs or exception messages."""
    secret_key_1 = "sk-secret-gemini-key-1111"
    secret_key_2 = "sk-secret-gemini-key-2222"

    monkeypatch.setenv("LLM_PROVIDER", "gemini")
    monkeypatch.setenv("GEMINI_API_KEY_1", secret_key_1)
    monkeypatch.setenv("GEMINI_API_KEY_2", secret_key_2)

    llm = get_default_llm()
    assert secret_key_1 not in str(llm)
    assert secret_key_2 not in str(llm)


def test_tool_binding_compatibility(monkeypatch):
    """Verify FailoverChatModel retains tool binding functionality across candidates."""
    monkeypatch.setenv("LLM_PROVIDER", "gemini")
    monkeypatch.setenv("GEMINI_API_KEY_1", "key-1")
    monkeypatch.setenv("GEMINI_API_KEY_2", "key-2")
    monkeypatch.setenv("GEMINI_API_KEY_3", "")
    monkeypatch.setenv("GEMINI_API_KEY", "")
    monkeypatch.setenv("OPENROUTER_API_KEY", "")

    llm = get_default_llm()
    tools = create_workspace_tools()
    bound_llm = llm.bind_tools(tools)
    assert isinstance(bound_llm, FailoverChatModel)
    assert len(bound_llm.candidates) == 2


def test_gemini_tool_call_preserves_thought_signature_in_payload(monkeypatch):
    """Verify GeminiChatOpenAI preserves thought_signature in _get_request_payload on subsequent turns."""
    from app.agent import GeminiChatOpenAI
    from langchain_core.messages import HumanMessage, AIMessage, ToolMessage

    model = GeminiChatOpenAI(model="gemini-3.6-flash", api_key="test-key")

    ai_msg = AIMessage(
        content="",
        tool_calls=[{"name": "run_tests", "args": {}, "id": "call_123"}],
        additional_kwargs={"thought_signature": "gemini_thought_sig_xyz"}
    )
    tool_msg = ToolMessage(content="Test result", tool_call_id="call_123", name="run_tests")

    messages = [
        HumanMessage(content="Run the tests"),
        ai_msg,
        tool_msg,
    ]

    payload = model._get_request_payload(messages)
    payload_messages = payload.get("messages", [])
    
    # Assistant message at index 1 must preserve thought_signature in payload dictionary
    assistant_payload = payload_messages[1]
    assert assistant_payload.get("thought_signature") == "gemini_thought_sig_xyz"


def test_memory_serialization_preserves_additional_kwargs_and_metadata():
    """Verify memory serialize_message and deserialize_message retain additional_kwargs and response_metadata."""
    from app.memory import serialize_message, deserialize_message
    from langchain_core.messages import AIMessage

    orig_msg = AIMessage(
        content="calling tool",
        tool_calls=[{"name": "read_file", "args": {"file_path": "main.py"}, "id": "call_1"}],
        additional_kwargs={"thought_signature": "sig_abc_123"},
        response_metadata={"finish_reason": "tool_calls"}
    )

    serialized = serialize_message(orig_msg)
    assert serialized.get("additional_kwargs") == {"thought_signature": "sig_abc_123"}
    assert serialized.get("response_metadata") == {"finish_reason": "tool_calls"}

    deserialized = deserialize_message(serialized)
    assert isinstance(deserialized, AIMessage)
    assert deserialized.additional_kwargs == {"thought_signature": "sig_abc_123"}
    assert deserialized.response_metadata == {"finish_reason": "tool_calls"}


def test_gemini_payload_does_not_contain_null_content(monkeypatch):
    """Verify GeminiChatOpenAI payload converts content=None to content='' and strips None fields."""
    from app.agent import GeminiChatOpenAI
    from langchain_core.messages import AIMessage

    model = GeminiChatOpenAI(model="gemini-3.6-flash", api_key="test-key")

    ai_msg = AIMessage(
        content="",
        tool_calls=[{"name": "run_tests", "args": {}, "id": "call_999"}],
        name=None
    )

    payload = model._get_request_payload([ai_msg])
    msg_dict = payload["messages"][0]

    assert msg_dict["content"] == ""
    assert "name" not in msg_dict


def test_multi_turn_tool_calling_sequence_preserves_thought_signature_in_persisted_history():
    """Regression test: verify 2+ sequential tool calls preserve thought_signature through production persistence round-trip (position 2 payload)."""
    from app.agent import GeminiChatOpenAI
    from app.memory import serialize_state, deserialize_state
    from app.state import AgentState
    from langchain_core.messages import HumanMessage, AIMessage, ToolMessage

    model = GeminiChatOpenAI(model="gemini-3.6-flash", api_key="test-key")

    # Turn 1: Assistant calls tool 1 with thought_signature
    ai_msg_1 = AIMessage(
        content="",
        tool_calls=[{"name": "read_file", "args": {"file_path": "calculator.py"}, "id": "call_1"}],
        additional_kwargs={"thought_signature": "sig_turn_1_abc"}
    )
    tool_msg_1 = ToolMessage(content="def add(a, b): return a + b", tool_call_id="call_1", name="read_file")

    # Turn 2: Assistant calls tool 2 (run_tests) with thought_signature
    ai_msg_2 = AIMessage(
        content="",
        tool_calls=[{"name": "run_tests", "args": {}, "id": "call_2"}],
        additional_kwargs={"thought_signature": "sig_turn_2_xyz"}
    )
    tool_msg_2 = ToolMessage(content="Status: passed", tool_call_id="call_2", name="run_tests")

    messages = [
        HumanMessage(content="Fix calculator tests"),
        ai_msg_1,
        tool_msg_1,
        ai_msg_2,
        tool_msg_2,
    ]

    state: AgentState = {
        "task_id": "test_multi_turn_persisted",
        "user_goal": "Fix calculator tests",
        "messages": messages,
        "status": "running"
    }

    # Simulate production persistence: state -> serialized dict -> JSON -> deserialized state
    serialized_state = serialize_state(state)
    reconstructed_state = deserialize_state(serialized_state)

    reconstructed_messages = reconstructed_state["messages"]

    # Verify deserialized AIMessages maintain thought_signature in additional_kwargs
    reconstructed_ai_1 = reconstructed_messages[1]
    reconstructed_ai_2 = reconstructed_messages[3]

    assert isinstance(reconstructed_ai_1, AIMessage)
    assert isinstance(reconstructed_ai_2, AIMessage)
    assert reconstructed_ai_1.additional_kwargs.get("thought_signature") == "sig_turn_1_abc"
    assert reconstructed_ai_2.additional_kwargs.get("thought_signature") == "sig_turn_2_xyz"

    # Verify request payload generation for the LLM call preserves thought_signature on BOTH tool calls (position 1 and position 2)
    payload = model._get_request_payload(reconstructed_messages)
    payload_messages = payload.get("messages", [])

    # payload_messages[1] is position 1 assistant call
    assert payload_messages[1].get("thought_signature") == "sig_turn_1_abc"
    assert payload_messages[1]["tool_calls"][0].get("thought_signature") == "sig_turn_1_abc"

    # payload_messages[3] is position 2 assistant call (the exact failure point in multi-tool-call sequences)
    assert payload_messages[3].get("thought_signature") == "sig_turn_2_xyz"
    assert payload_messages[3]["tool_calls"][0].get("thought_signature") == "sig_turn_2_xyz"



