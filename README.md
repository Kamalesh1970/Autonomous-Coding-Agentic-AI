# Autonomous Coding Agentic AI (Phase 6: Testing & Self-Correction)

Minimal autonomous software engineering agent built with **Python** and **LangGraph**.

Reference Architecture: Inspired by [Open SWE](https://github.com/langchain-ai/open-swe) as a conceptual benchmark, built from the ground up as a minimal agentic foundation.

---

## 🎯 What is Phase 6?

Phase 6 equips the agent with autonomous **Testing, Debugging, Failure Diagnosis, and Recovery** capabilities (`run_tests`). The agent implements the core self-correction loop:

$$\text{ACT} \longrightarrow \text{OBSERVE} \longrightarrow \text{EVALUATE} \longrightarrow \text{RECOVER} \longrightarrow \text{RETRY}$$

When validation tests fail, the agent observes error tracebacks/assertions, retrieves relevant code context, applies targeted fixes (`replace_in_file`), verifies git changes (`git_diff`), and re-runs validation until tests pass or the max retry limit is reached.

---

## 🤖 Why Agentic (Not a Single-Pass Code Generator)?

A simple LLM tool merely generates code once.

This agentic AI operates in an **environmental feedback loop**:
```
User Goal
   │
   ▼
[ Plan & Retrieve Context ]
   │
   ▼
[ Modify Repository Code ] (write_file / replace_in_file)
   │
   ▼
[ Inspect Git Diff ] (git_diff)
   │
   ▼
[ Run Automated Tests ] (run_tests: pytest)
   │
   ├────────► [ PASS ] ──────────────────────► [ Complete Goal ]
   │
   ▼ (FAIL / Traceback Observation)
[ Diagnose Error & Retrieve ]
   │
   ▼
[ Apply Corrective Fix ] ───► (Re-run Tests: RETRY loop up to max_retries)
```

---

## 🛠️ Tool Layer (Inspection + Retrieval + Editing + Validation + Planning)

All tools enforce strict repository boundary protection:

1. `list_files(directory=".")`: Recursively lists relative file paths in workspace.
2. `read_file(file_path)`: Reads text file content with line/byte limits and binary file detection.
3. `search_code(query, directory=".")`: Plain-text search returning `file:line: snippet` matches.
4. `git_status()`: Inspects repository Git branch, modified files, and untracked files.
5. `git_diff()`: Inspects current unstaged and staged Git differences for edit feedback.
6. `retrieve_relevant_context(query, directory=".")`: Ranks relevant files and returns surrounding code context snippets.
7. `write_file(file_path, content)`: Safely creates or overwrites repository files.
8. `replace_in_file(file_path, old_text, new_text)`: Targeted unique text replacement (fails safely if missing or ambiguous).
9. `run_tests(target_directory=".", timeout_seconds=30)`: **[NEW]** Executes pytest validation inside workspace with process timeout protection and bounded output.
10. `create_plan(tasks)`: Decomposes goal into structured subtasks with dependencies.
11. `update_task_status(task_id, status, notes)`: Updates task status.
12. `revise_plan(new_tasks, reason)`: Dynamically modifies remaining tasks based on new findings.

### 🛡️ Execution & Boundary Protection
- ❌ `run_tests` executes strictly `python3 -m pytest` inside `workspace_root` with process timeout protection (default 30s) and bounded output (max 4,000 chars).
- ❌ No unrestricted arbitrary shell execution tool (`execute_shell`) is exposed.
- ❌ `retry_count` prevents infinite recovery loops (max 3 retries by default).

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

Run the agent on a bug fix / validation goal:
```bash
python3 -m app.agent "Fix the multiply function so that the project's tests pass." .
```

---

## 🚧 Intentionally NOT Implemented (Reserved for Future Phases)

To keep Phase 6 strictly focused on autonomous testing, debugging, and recovery loops, the following components are intentionally omitted:

- ❌ Unrestricted shell execution
- ❌ Docker / cloud sandbox infrastructure
- ❌ GitHub API / branch creation / commits / pushes / PRs
- ❌ Vector databases / RAG / Qdrant
- ❌ Multi-agent systems / MCP / Observability platforms
