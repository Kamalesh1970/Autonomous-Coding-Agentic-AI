# Autonomous Coding Agentic AI (Phase 2: Repository Understanding)

Minimal autonomous software engineering agent built with **Python** and **LangGraph**.

Reference Architecture: Inspired by [Open SWE](https://github.com/langchain-ai/open-swe) as a conceptual benchmark, built from the ground up as a minimal agentic foundation.

---

## 🎯 What is Phase 2?

Phase 2 upgrades the single-agent loop to safely inspect and understand a **real local Git repository** in a strictly **READ-ONLY** capacity. The agent autonomously navigates repository file trees, searches source code, reads file contents, and checks Git status and differences to answer complex engineering questions.

---

## 🤖 Why Agentic (Not a Chatbot)?

A standard chatbot operates in a simple single-turn request-response loop (`User Prompt → LLM Response`).

This agentic AI operates in an **iterative control loop**:
```
User Goal
   │
   ▼
[ Reason ] ◄────────────────────┐
   │                            │
[ Decide ]                      │ (Observation)
   │                            │
   ├──────────► [ Act ] ────────┴
   │         (Invoke Tool)
   ▼
[ Finish ]
(Final Answer)
```

The agent maintains explicit state, observes tool execution outputs (file listings, code searches, file contents, git diffs), updates its internal context with those observations, and dynamically determines the next step.

---

## 🛠️ Read-Only Tool Layer

All tools operate strictly against a configured repository root with path traversal security:

1. `list_files(directory=".")`: Recursively lists relative file paths in the workspace (skipping `.git` internals and bounding output).
2. `read_file(file_path)`: Reads text file content with path traversal validation, binary file detection, and line/byte limits.
3. `search_code(query, directory=".")`: Plain-text search across repository files returning `file:line: snippet` matches.
4. `git_status()`: Inspects repository Git branch, modified files, and untracked files.
5. `git_diff()`: Inspects current unstaged and staged Git differences.

### 🛡️ Read-Only Security Boundary
- ❌ No file writing, editing, or deletion
- ❌ No `git commit`, `git push`, `git checkout`, or branch modifications
- ❌ No arbitrary shell/command execution
- ❌ Path traversal attacks (`../`, absolute paths outside workspace) are blocked at the tool boundary.

---

## 🏗️ Architecture

- **State (`app/state.py`)**: `AgentState` contains `messages` (using `add_messages` reducer), `user_goal`, and `workspace_root`.
- **Tools (`app/tools.py`)**: 5 read-only repository inspection tools with path traversal protection.
- **Agent (`app/agent.py`)**: Compiled LangGraph `StateGraph` linking reasoning (`reason`) and execution (`tools`) nodes via conditional routing (`route_after_reason`).

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

Run the agent on a repository understanding goal:
```bash
python3 -m app.agent "Understand how authentication is implemented in this repository." .
```

---

## 🚧 Intentionally NOT Implemented (Reserved for Future Phases)

To keep Phase 2 strictly focused on read-only repository understanding, the following components are intentionally omitted:

- ❌ Code modification / editing / writing tools
- ❌ Sandboxed code execution / running tests
- ❌ Git write operations (`commit`, `push`, `PRs`)
- ❌ Arbitrary shell execution / terminal execution
- ❌ FastAPI / Web UI
- ❌ Vector databases / RAG / Qdrant / Embeddings
- ❌ Multi-agent systems / MCP / Observability platforms
