"""
Standalone diagnostic: tests whether a bare ChatGoogleGenerativeAI call
(with NO custom history/state handling) can survive 2+ sequential tool
calls without the "missing thought_signature" error.

If this script WORKS -> the bug is in your app's history/state handling
                         (dict round-trips, checkpointing, FailoverChatModel, etc.)
If this script FAILS -> the bug is in langchain-google-genai itself for
                         your installed version / gemini-3.6-flash combo.

Run:
    python test_gemini_thought_signature.py

Requires GOOGLE_API_KEY (or GEMINI_API_KEY, depending on your setup) in env.
"""

import os
import sys
from langchain_core.messages import HumanMessage
from langchain_core.tools import tool
from langchain_google_genai import ChatGoogleGenerativeAI

print("langchain-google-genai version check:")
import langchain_google_genai
print("  ", langchain_google_genai.__version__)
print()


@tool
def list_files(directory: str) -> str:
    """List files in a directory. Just a fake tool for testing."""
    return "calculator.py, test_calculator.py, string_utils.py"


@tool
def run_tests() -> str:
    """Run the test suite. Just a fake tool for testing."""
    return "2 failed, 2 passed"


def dump_message(i, m):
    print(f"--- message {i}: {type(m).__name__} ---")
    print("  content:", repr(getattr(m, "content", None))[:150])
    tool_calls = getattr(m, "tool_calls", None)
    print("  tool_calls:", tool_calls)
    ak = getattr(m, "additional_kwargs", None)
    print("  additional_kwargs keys:", list(ak.keys()) if ak else ak)
    if ak and "thought_signature" in ak:
        print("  -> thought_signature PRESENT, length:", len(str(ak["thought_signature"])))
    elif ak:
        print("  -> NO thought_signature key in additional_kwargs")
    print()


def main():
    model = ChatGoogleGenerativeAI(
        model="gemini-3.6-flash",
        temperature=0,
    ).bind_tools([list_files, run_tests])

    messages = [
        HumanMessage(content="List the files in the repo, then run the tests. Use the tools.")
    ]

    print("=== TURN 1: first LLM call ===")
    ai_msg_1 = model.invoke(messages)
    dump_message("AI-1", ai_msg_1)

    # Append the EXACT object returned, no dict round-trip
    messages.append(ai_msg_1)

    if not ai_msg_1.tool_calls:
        print("Model didn't call a tool on turn 1 — can't continue the test as designed.")
        print("Raw response:", ai_msg_1)
        sys.exit(1)

    # Simulate executing the first tool call and appending its ToolMessage
    from langchain_core.messages import ToolMessage
    for tc in ai_msg_1.tool_calls:
        if tc["name"] == "list_files":
            result = list_files.invoke(tc["args"])
        elif tc["name"] == "run_tests":
            result = run_tests.invoke(tc["args"])
        else:
            result = "unknown tool"
        messages.append(ToolMessage(content=str(result), tool_call_id=tc["id"]))

    print("=== TURN 2: second LLM call (this is where the bug historically fires) ===")
    try:
        ai_msg_2 = model.invoke(messages)
        dump_message("AI-2", ai_msg_2)
        print("SUCCESS: second call completed without thought_signature error.")
        print()
        print("CONCLUSION: langchain-google-genai + gemini-3.6-flash handles this")
        print("correctly in a clean, no-custom-state scenario. Your bug is almost")
        print("certainly in how your app stores/reconstructs messages between turns")
        print("(dict conversion, checkpointing, FailoverChatModel wrapping, etc).")
    except Exception as e:
        print("FAILED on second call with:")
        print(" ", e)
        print()
        print("CONCLUSION: this reproduces even with a completely clean, minimal")
        print("history (no custom state, no FailoverChatModel, no dict round-trips).")
        print("This means it's a library-level bug in your installed")
        print("langchain-google-genai version for gemini-3.6-flash multi-turn tool use.")
        print("Next step: paste this full output back for a targeted fix (native SDK")
        print("bypass, or pin/patch the library).")


if __name__ == "__main__":
    main()