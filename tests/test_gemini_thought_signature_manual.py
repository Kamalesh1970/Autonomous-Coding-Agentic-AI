"""Manual Diagnostic Script: Gemini Thought Signature Tool-Calling Test.

Tests whether a bare ChatGoogleGenerativeAI call survives sequential tool calls
without missing thought_signature errors.

Note: This is a manual diagnostic script and is excluded from automatic pytest execution
if no GOOGLE_API_KEY / GEMINI_API_KEY is configured in the environment.
"""

import os
import sys
from langchain_core.messages import HumanMessage
from langchain_core.tools import tool
from langchain_google_genai import ChatGoogleGenerativeAI


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
    api_key = os.getenv("GEMINI_API_KEY") or os.getenv("GOOGLE_API_KEY")
    if not api_key:
        print("GEMINI_API_KEY or GOOGLE_API_KEY is not configured. Skipping manual test.")
        return

    model = ChatGoogleGenerativeAI(
        model="gemini-3.6-flash",
        google_api_key=api_key,
        temperature=0,
    ).bind_tools([list_files, run_tests])

    messages = [HumanMessage(content="Hello, please list the files in the current repo.")]
    print("Turn 1 request sent.")
    res1 = model.invoke(messages)
    messages.append(res1)
    dump_message(1, res1)


if __name__ == "__main__":
    main()
