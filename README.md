# Autonomous Coding Agentic AI (Phase 10: Safe Git/GitHub Delivery & Human Approval)

Minimal autonomous software engineering agent built with **Python** and **LangGraph**.

Reference Architecture: Inspired by [Open SWE](https://github.com/langchain-ai/open-swe) as a conceptual benchmark, built from the ground up as a minimal agentic foundation.

---

## 🎯 What is Phase 10?

Phase 10 equips the agent with **Safe Git/GitHub Delivery and Human-in-the-Loop Approval** capabilities (`git_status`, `git_diff`, `git_current_branch`, `git_create_branch`, `request_human_approval`, `git_commit`, `git_push`, `create_pull_request`).

The agentic lifecycle for Phase 10 is:

$$\text{VERIFIED CODE CHANGES} \longrightarrow \text{INSPECT GIT STATE} \longrightarrow \text{PREPARE DELIVERY} \longrightarrow \text{REQUEST HUMAN APPROVAL} \longrightarrow \text{[ HUMAN APPROVAL DECISION ]} \longrightarrow \text{EXECUTE APPROVED GIT ACTION}$$

---

## 🚦 Human-in-the-Loop Approval Model

The agent is prohibited from blindly executing externally impactful delivery actions:

1. **Actions Requiring Explicit Approval**: `git commit`, `git push`, `create_pull_request`.
2. **Read-Only / Safe Local Actions**: `git_status`, `git_diff`, `git_current_branch`, `git_create_branch` (read-only and safe local feature branch creation run autonomously).
3. **Approval States**:
   - `not_required`: Read-only / inspection operations.
   - `pending`: Delivery action requested; waiting for human decision.
   - `approved`: Human operator approved action (`approve_task`); agent proceeds with commit/push/PR.
   - `rejected`: Human operator rejected action (`approve_task`); delivery action is cancelled without modifying Git history.
4. **State Persistence**: Approval status (`pending`, `approved`, `rejected`) is persisted to disk and survives task pause/resume.

---

## 🛡️ Security & Git Boundary Controls (`app/sandbox.py`)

- ❌ Destructive Git commands (`git reset --hard`, `git clean -fd`, `git push --force`) are strictly **prohibited**.
- ❌ Branch name injection (names starting with `-` or containing illegal shell characters) is blocked.
- ❌ Overwriting existing Git branches is prevented.
- ❌ Empty commits without staged changes are rejected.
- ❌ Credentials and secret tokens (`GITHUB_TOKEN`, `OPENAI_API_KEY`) are stripped/redacted from command outputs.

---

## 🔒 Security Threat Model

| Threat Scenario | Sandbox & Delivery Protection |
| :--- | :--- |
| **LLM attempts unapproved commit or push** | Tool checks `approval_status`; blocks operation unless explicitly `approved`. |
| **Human rejects delivery request** | `approval_status = "rejected"` cancels operation and halts execution cleanly. |
| **Branch name injection (`-b_malicious` or `; rm`)** | `validate_branch_name` rejects invalid characters and flags. |
| **Destructive Git commands (`git reset --hard`)** | `is_command_allowed` rejects destructive commands before execution. |
| **Secrets leak into Git commits or push logs** | Subprocess environment isolation & token redaction mask credentials. |

---

## ⚠️ Known Limitations

- **Local Subprocess Isolation**: Uses local subprocess execution and environment sanitization (not equivalent to VM/Docker).
- **GitHub Mocking**: Remote push and PR creation operate safely offline with token redaction for test environments.
- **Academic Prototype**: Designed as an academic software engineering prototype.


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

