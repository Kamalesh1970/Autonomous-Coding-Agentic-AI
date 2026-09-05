# Autonomous Coding Agentic AI (Phase 8: Persistent Memory & Long-Running Agent State)

Minimal autonomous software engineering agent built with **Python** and **LangGraph**.

Reference Architecture: Inspired by [Open SWE](https://github.com/langchain-ai/open-swe) as a conceptual benchmark, built from the ground up as a minimal agentic foundation.

---

## 🎯 What is Phase 8?

Phase 8 equips the agent with **Persistent Memory and Long-Running Agent State** capabilities (`save_state`, `load_state`, `resume_agent`). The agent can:

1. Start a long-running software engineering task.
2. Execute agentic reasoning, planning, code modification, and validation steps.
3. Automatically persist structured execution state to disk (`.agent_memory/<task_id>.json`).
4. Survive process stops, interruptions, or crashes.
5. Resume execution via `resume_agent(task_id)` without losing the original user goal, task plan progress, modified files, validation results, verification results, or recovery `retry_count`.

The agentic lifecycle for Phase 8 is:

$$\text{GOAL} \longrightarrow \text{PLAN} \longrightarrow \text{ACT} \longrightarrow \text{PERSIST STATE} \longrightarrow \text{[ INTERRUPT / STOP ]} \longrightarrow \text{RESUME} \longrightarrow \text{VALIDATE} \longrightarrow \text{VERIFY} \longrightarrow \text{COMPLETE}$$

---

## 💾 Persistent Execution Memory ($\text{Memory} \neq \text{RAG / Vector DB}$)

- **Persistent Execution Memory (`app/memory.py`)**: Stores JSON-safe task execution state (`task_id`, `user_goal`, `workspace_root`, `plan`, `retrieved_context`, `modified_files`, `validation_result`, `verification_result`, `retry_count`, `max_retries`, `status`, `messages`).
- **Atomic Persistence**: Writes to a temporary `.tmp` file and replaces the destination JSON file atomically using `os.replace` to prevent state corruption.
- **Path Traversal Security**: Strict task ID validation (`^[a-zA-Z0-9_-]+$`) prevents path traversal attacks outside `.agent_memory`.
- **Runtime Cleanliness**: Deliberately omits non-serializable objects (LLM instances, active subprocesses, tool callbacks) and secret environment keys.

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
[ Modify Repository Code ] (write_file / replace_in_file)
   │
   ▼
[ Checkpoint State to Disk ] (atomic save to .agent_memory/<task_id>.json)
   │
   ▼
[ Run Automated Tests ] (run_tests: pytest) ───► (Validation)
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

## 🛠️ Tool & Memory Layer

All tools enforce strict repository and memory storage boundary protection:

1. `list_files(directory=".")`: Recursively lists relative file paths in workspace.
2. `read_file(file_path)`: Reads text file content with line/byte limits and binary file detection.
3. `search_code(query, directory=".")`: Plain-text search returning `file:line: snippet` matches.
4. `git_status()`: Inspects repository Git branch, modified files, and untracked files.
5. `git_diff()`: Inspects current unstaged and staged Git differences for edit feedback.
6. `retrieve_relevant_context(query, directory=".")`: Ranks relevant files and returns surrounding code context snippets.
7. `write_file(file_path, content)`: Safely creates or overwrites repository files.
8. `replace_in_file(file_path, old_text, new_text)`: Targeted unique text replacement (fails safely if missing or ambiguous).
9. `run_tests(target_directory=".", timeout_seconds=30)`: Safe pytest execution inside workspace with process timeout protection.
10. `verify_goal(status, summary, evidence)`: Evaluates whether the original user goal is satisfied based on repository evidence (`passed`, `failed`, `uncertain`).
11. `create_plan(tasks)`: Decomposes goal into structured subtasks with dependencies.
12. `update_task_status(task_id, status, notes)`: Updates task status.
13. `revise_plan(new_tasks, reason)`: Dynamically modifies remaining tasks based on new findings.
14. **Memory API (`app/memory.py`)**: `save_state`, `load_state`, `delete_state`, `safe_resolve_task_memory_path`.

### 🛡️ Execution & Memory Protection
- ❌ State serialization excludes secrets, API keys, runtime handles, and open subprocesses.
- ❌ Task IDs are restricted to alphanumeric characters, hyphens, and underscores to block path traversal (`../../`).
- ❌ `retry_count` is preserved on task resume to enforce retry limits across process restarts.

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

The test suite is 100% deterministic and runs offline using mocked model responses and temporary Git fixtures (`pytest` + `git init`):

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

Resume an existing task from memory:
```python
from app.agent import resume_agent

final_state = resume_agent(task_id="task_abc123")
```

---

## 🚧 Intentionally NOT Implemented (Reserved for Future Phases)

To keep Phase 8 strictly focused on persistent execution state, the following components are intentionally omitted:

- ❌ Unrestricted shell execution
- ❌ Docker / cloud sandbox infrastructure
- ❌ GitHub API / branch creation / commits / pushes / PRs
- ❌ Vector databases / RAG / Qdrant / Pinecone / Chroma
- ❌ Multi-agent systems / MCP / Observability platforms
