# Autonomous Coding Agentic AI (Phase 3: Planning & Decomposition)

Minimal autonomous software engineering agent built with **Python** and **LangGraph**.

Reference Architecture: Inspired by [Open SWE](https://github.com/langchain-ai/open-swe) as a conceptual benchmark, built from the ground up as a minimal agentic foundation.

---

## 🎯 What is Phase 3?

Phase 3 introduces an intelligent **Planning and Task Decomposition** capability. Given a high-level software engineering goal, the agent dynamically:
1. Decomposes the goal into structured, actionable subtasks with explicit execution dependencies.
2. Maintains structured plan state (`pending`, `in_progress`, `completed`, `failed`, `blocked`).
3. Determines the next available task based on completed dependencies.
4. **Revises the plan dynamically** when new repository observations contradict initial assumptions.

---

## 🤖 Why Agentic (Not a Chatbot)?

A standard chatbot operates in a simple single-turn request-response loop (`User Prompt → LLM Response`).

This agentic AI operates in an **iterative control loop**:
```
User Goal
   │
   ▼
[ Understand Repository ]
   │
   ▼
[ Create Plan ] ◄──────────────────┐
   │                               │
[ Select Next Task ]               │ (Observation / New Evidence)
   │                               │
   ├──────────► [ Act ] ───────────┼────────► [ Revise Plan ]
   │         (Invoke Tool)         │
   ▼                               │
[ Finish ] ◄───────────────────────┘
(Final Answer)
```

The agent maintains explicit state, observes tool execution outputs (file listings, code searches, file contents, git diffs), updates its internal context with those observations, and dynamically determines the next step or revises its plan.

---

## 📋 Task & Plan Schema

- **`Task`**: `id`, `title`, `description`, `status` (`pending`, `in_progress`, `completed`, `failed`, `blocked`), `dependencies`.
- **`ExecutionPlan`**: `goal`, `tasks`, `current_task_id`, `revision_count`, `revision_reason`.

---

## 🛠️ Tool Layer (Read-Only + Planning)

All tools operate strictly against a configured repository root with path traversal security:

1. `list_files(directory=".")`: Recursively lists relative file paths in workspace.
2. `read_file(file_path)`: Reads text file content with path traversal validation, binary file detection, and line/byte limits.
3. `search_code(query, directory=".")`: Plain-text search returning `file:line: snippet` matches.
4. `git_status()`: Inspects repository Git branch, modified files, and untracked files.
5. `git_diff()`: Inspects current unstaged and staged Git differences.
6. `create_plan(tasks)`: Decomposes goal into structured subtasks with dependencies.
7. `update_task_status(task_id, status, notes)`: Updates task status.
8. `revise_plan(new_tasks, reason)`: Dynamically modifies remaining tasks based on new repository findings.

### 🛡️ Read-Only Security Boundary
- ❌ No file writing, editing, or deletion
- ❌ No `git commit`, `git push`, `git checkout`, or branch modifications
- ❌ No arbitrary shell/command execution
- ❌ Path traversal attacks (`../`, absolute paths outside workspace) are blocked at the tool boundary.

---

## 📦 Installation

```bash
# Navigate to project directory
cd autonomous-coding-agent

# Create and activate virtual environment (Python 3.10+)
python3 -m venv .venv
source .venv/bin/activate

# Install dependencies
pip install -e .
pip install pytest
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

Run the agent on a software engineering task:
```bash
python3 -m app.agent "Understand and improve authentication in this repository." .
```

---

## 🚧 Intentionally NOT Implemented (Reserved for Future Phases)

To keep Phase 3 strictly focused on read-only repository understanding and planning, the following components are intentionally omitted:

- ❌ Code modification / editing / writing tools
- ❌ Sandboxed code execution / running tests
- ❌ Git write operations (`commit`, `push`, `PRs`)
- ❌ Arbitrary shell execution / terminal execution
- ❌ FastAPI / Web UI
- ❌ Vector databases / RAG / Qdrant / Embeddings
- ❌ Multi-agent systems / MCP / Observability platforms
