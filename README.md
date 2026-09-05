# Autonomous Coding Agentic AI (Phase 9: Secure Sandbox & Execution Isolation)

Minimal autonomous software engineering agent built with **Python** and **LangGraph**.

Reference Architecture: Inspired by [Open SWE](https://github.com/langchain-ai/open-swe) as a conceptual benchmark, built from the ground up as a minimal agentic foundation.

---

## 🎯 What is Phase 9?

Phase 9 introduces an explicit, security-oriented **Execution Boundary / Sandbox Layer** (`app/sandbox.py`). The autonomous agent is constrained to a controlled execution environment so it cannot receive unrestricted access to the host machine.

The agentic lifecycle for Phase 9 is:

$$\text{GOAL} \longrightarrow \text{PLAN} \longrightarrow \text{RETRIEVE} \longrightarrow \text{DECIDE ACTION} \longrightarrow \text{SAFETY CHECK} \longrightarrow \text{SANDBOX EXECUTION} \longrightarrow \text{OBSERVE} \longrightarrow \text{EVALUATE} \longrightarrow \text{RECOVER} \longrightarrow \text{VERIFY} \longrightarrow \text{PERSIST}$$

---

## 🛡️ Security and Sandbox Architecture (`app/sandbox.py`)

The sandbox enforces security policies independently in code. The LLM is **not** treated as a security boundary:

1. **Repository Boundary (`sandbox_root`)**: All filesystem operations and command executions are locked within the designated repository root directory. Attempts to traverse outside (`../../`) or access absolute paths outside root are rejected.
2. **Safe Path Resolution & Symlink Checks**: Every path is resolved (`Path.resolve()`). Symlinks pointing outside the repository root are detected and blocked.
3. **Controlled Command Allowlist**: Low-level shell execution (`execute_shell`, `bash`, `sh`, `zsh`, `powershell`, `cmd`, `curl`, `wget`, `rm`) is **disallowed**. Only explicit allowlisted commands (such as `python -m pytest` and read-only `git status/diff`) are permitted.
4. **Environment Isolation**: Subprocesses run with a minimal environment. Host secrets, API keys (`OPENAI_API_KEY`), credentials, and `.env` contents are stripped from executed subprocess environments.
5. **Working Directory Enforcement**: Subprocesses are forced to execute strictly inside the sandbox root directory.
6. **Execution Timeout**: Bounded execution timeout (default 30 seconds) prevents infinite process hangs.
7. **Output Limits**: Stdout/stderr output is truncated at bounded limits with explicit truncation indicators.
8. **Structured Security Events**: Tracks security rejection events (`path_escape_rejected`, `command_rejected`, `timeout`, `output_truncated`).

---

## 🔒 Security Threat Model

| Threat Scenario | Sandbox Defense / Protection |
| :--- | :--- |
| **LLM generates malicious path (`../../etc/passwd`)** | `safe_resolve_path` checks boundary against `sandbox_root` and blocks traversal. |
| **LLM requests arbitrary shell execution (`rm -rf /`)** | Execution interface rejects non-allowlisted binaries (`bash`, `sh`, `rm`). |
| **Symlink points to host secret outside repository** | Symlink target resolution detects external pointer and blocks file access/edits. |
| **Code execution hangs indefinitely** | `run_command` timeout kills subprocess gracefully after threshold. |
| **Excessive output overwhelms LLM context** | Bounded stdout/stderr truncates output with explicit notification. |
| **Executed code leaks host API keys (`OPENAI_API_KEY`)** | Minimal environment construction filters out sensitive host keys. |

---

## ⚠️ Known Limitations

- **Local Subprocess Isolation**: The Phase 9 sandbox uses local subprocess isolation and environment sanitization. It is **not** equivalent to kernel-level containerization (Docker) or virtual machine isolation.
- **Academic Prototype**: Designed as an academic software engineering prototype illustrating autonomous safety boundaries.

---

## 🤖 Why Agentic (Not a Single-Pass Code Generator)?

A simple LLM tool merely generates code once.

This agentic AI operates in an **environmental feedback loop**:
```
User Goal (or Resumed Task ID)
   │
   ▼
[ Plan & Retrieve Context ]
   │
   ▼
[ Safety Check ] ───► (Rejects unsafe paths or commands)
   │
   ▼
[ Modify Repository Code ] (write_file / replace_in_file inside sandbox)
   │
   ▼
[ Checkpoint State to Disk ] (atomic save to .agent_memory/<task_id>.json)
   │
   ▼
[ Run Automated Tests ] (run_tests: pytest in sandbox) ───► (Validation)
   │
   ├────────► [ PASS ] ───► [ Inspect Repository Evidence ]
   │                              │
   │                              ▼
   │                      [ Verify Goal ] (verify_goal: passed / failed / uncertain)
   │                              │
   │                              ├────────► [ PASSED ] ───► [ Complete Goal ]
   │                              │
   ▼ (FAIL Traceback)             ▼ (FAILED / UNCERTAIN Evidence)
[ Diagnose Error & Retrieve ] ────┴────► [ Apply Fix & Recovery ] ───► (RETRY loop)
```

---

## 🛠️ Tool & Sandbox Layer

All tools enforce strict repository boundary and sandbox isolation:

1. `list_files(directory=".")`: Recursively lists relative file paths in workspace.
2. `read_file(file_path)`: Reads text file content inside sandbox with byte limits and binary file detection.
3. `search_code(query, directory=".")`: Plain-text search returning `file:line: snippet` matches.
4. `git_status()`: Inspects repository Git branch, modified files, and untracked files.
5. `git_diff()`: Inspects current unstaged and staged Git differences for edit feedback.
6. `retrieve_relevant_context(query, directory=".")`: Ranks relevant files and returns surrounding code context snippets.
7. `write_file(file_path, content)`: Safely creates or overwrites repository files inside sandbox.
8. `replace_in_file(file_path, old_text, new_text)`: Targeted unique text replacement inside sandbox.
9. `run_tests(target_directory=".", timeout_seconds=30)`: Controlled pytest execution in sandbox minimal environment.
10. `verify_goal(status, summary, evidence)`: Evaluates whether original user goal is satisfied based on repository evidence.
11. `create_plan(tasks)`: Decomposes goal into structured subtasks with dependencies.
12. `update_task_status(task_id, status, notes)`: Updates task status.
13. `revise_plan(new_tasks, reason)`: Dynamically modifies remaining tasks based on new findings.

---

## 📦 Installation

```bash
# Navigate to project directory
cd autonomous-coding-agent

# Create and activate virtual environment (Python 3.10+)
python3 -m venv .venv
source .venv/bin/activate

# Install dependencies
pip install -r requirements.txt
```

---

## 🧪 Running Tests

The test suite is 100% deterministic and runs offline using mocked model responses and temporary sandbox/Git fixtures (`pytest`):

```bash
pytest -v
```

---

## 🚀 Running the Agent

Set up your `.env` file with your OpenAI key:
```env
OPENAI_API_KEY=your_actual_api_key
OPENAI_MODEL_NAME=gpt-4o-mini
```

Run a new task:
```bash
python3 -m app.agent "Fix the multiply function so that the project's tests pass." .
```

---

## 🚧 Intentionally NOT Implemented (Reserved for Future Phases)

To keep Phase 9 strictly focused on secure sandbox & execution isolation, the following components are intentionally omitted:

- ❌ Mandatory Docker / VM dependencies (optional container sandbox backend left uncoupled)
- ❌ Human-in-the-loop approval mechanism (reserved for Phase 10)
- ❌ GitHub API / branch creation / commits / pushes / PRs
- ❌ Vector databases / RAG / Qdrant / Pinecone / Chroma
- ❌ Multi-agent systems / MCP / Observability platforms

