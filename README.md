# Autonomous Coding Agentic AI (Phase 5: Safe Code Modification)

Minimal autonomous software engineering agent built with **Python** and **LangGraph**.

Reference Architecture: Inspired by [Open SWE](https://github.com/langchain-ai/open-swe) as a conceptual benchmark, built from the ground up as a minimal agentic foundation.

---

## 🎯 What is Phase 5?

Phase 5 equips the agent with controlled **Repository Code Modification** capabilities (`write_file`, `replace_in_file`). The agent progresses from repository understanding to performing precise code edits and verifying the actual filesystem impact using **Git diff feedback**.

---

## 🤖 Why Agentic (Not a Chatbot / Code Generator)?

A simple code generator merely outputs code text in chat.

This agentic AI operates in an **environment-backed reasoning loop**:
```
User Goal
   │
   ▼
[ Retrieve Relevant Context ]
   │
   ▼
[ Create / Update Plan ]
   │
   ▼
[ Modify Repository Code ] ───► (Real Filesystem Change: write_file / replace_in_file)
   │
   ▼
[ Observe Git Diff Feedback ] ◄── (Git Repository feedback)
   │
   ▼
[ Evaluate Edit & Conclude ]
(Final Answer)
```

The agent applies edits directly to the repository filesystem and uses `git_diff()` observations to verify that the change matches the user requirement.

---

## 🛠️ Tool Layer (Inspection + Retrieval + Editing + Planning)

All tools enforce strict repository boundary protection:

1. `list_files(directory=".")`: Recursively lists relative file paths in workspace.
2. `read_file(file_path)`: Reads text file content with line/byte limits and binary file detection.
3. `search_code(query, directory=".")`: Plain-text search returning `file:line: snippet` matches.
4. `git_status()`: Inspects repository Git branch, modified files, and untracked files.
5. `git_diff()`: Inspects current unstaged and staged Git differences for edit feedback.
6. `retrieve_relevant_context(query, directory=".")`: Ranks relevant files and returns surrounding code context snippets.
7. `write_file(file_path, content)`: **[NEW]** Safely creates or overwrites repository files.
8. `replace_in_file(file_path, old_text, new_text)`: **[NEW]** Targeted unique text replacement (fails safely if missing or ambiguous).
9. `create_plan(tasks)`: Decomposes goal into structured subtasks with dependencies.
10. `update_task_status(task_id, status, notes)`: Updates task status.
11. `revise_plan(new_tasks, reason)`: Dynamically modifies remaining tasks based on new findings.

### 🛡️ Repository Boundary Protection
- ❌ All write operations enforce `safe_resolve_path(workspace_root, path)`. Path traversal (`../`), absolute paths outside workspace, and symlink escapes are strictly blocked.
- ❌ Controlled failure handling for missing or ambiguous replacement targets.
- ❌ No arbitrary shell execution or repository code execution in Phase 5.

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

Run the agent on a software engineering modification goal:
```bash
python3 -m app.agent "Change the greeting returned in sample app to 'Hello from autonomous agent'." .
```

---

## 🚧 Intentionally NOT Implemented (Reserved for Future Phases)

To keep Phase 5 strictly focused on safe repository modification and Git diff feedback, the following components are intentionally omitted:

- ❌ Automated test execution / build execution / linting
- ❌ Autonomous debugging & test-driven retry loops
- ❌ Docker / cloud sandbox infrastructure
- ❌ Git write operations (`commit`, `push`, `PRs`)
- ❌ Arbitrary shell execution / terminal execution
- ❌ Vector databases / RAG / Qdrant
- ❌ Multi-agent systems / MCP / Observability platforms
