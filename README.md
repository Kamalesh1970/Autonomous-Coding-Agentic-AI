# Autonomous Coding Agentic AI (Phase 12: Multi-Agent Software Engineering Architecture)

Minimal autonomous software engineering agent built with **Python** and **LangGraph**.

Reference Architecture: Inspired by [Open SWE](https://github.com/langchain-ai/open-swe) as a conceptual benchmark, built from the ground up as a minimal agentic foundation.

---

## 🎯 What is Phase 12?

Phase 12 introduces a **specialized Multi-Agent Software Engineering Architecture** (`app/multi_agent.py`) using **LangGraph**, where specialized agents collaborate through shared state (`AgentState`) under strict least-privilege tool subsets to solve repository-level tasks:

```text
               +-------------------+
               |  User Goal / Task |
               +---------+---------+
                         |
                         v
               +-------------------+
               |   Analyzer Agent  |  (Read-Only Retrieval & Architectural Planning)
               +---------+---------+
                         |
                         v
               +-------------------+
               |    Coder Agent    |  (Targeted File Editing & Local Test Running)
               +---------+---------+
                         |
                         v
               +-------------------+
               |   Reviewer Agent  |  (Read-Only Diff & Quality Verification)
               +---------+---------+
                         |
                 [Review Status]
              /         |         \
    "approved"  "changes_requested" "blocked"
           /            |            \
          v             v             v
       [END]     [Retry Iteration]   [END]
```

---

## 👥 Specialized Agent Roles & Capability Matrix

| Role | Tool Subset & Access Boundaries | Responsibilities | Output State Field |
| :--- | :--- | :--- | :--- |
| **Analyzer** | `list_files`, `read_file`, `search_code`, `git_status`, `git_diff`, `retrieve_hybrid_context`, planning | Repository structure analysis, risk assessment, architectural change planning. **Read-only** (No file edits). | `analysis_result` |
| **Coder** | `write_file`, `replace_in_file`, `run_tests`, read & retrieval tools, planning | Applying code modifications, executing tests, addressing reviewer feedback. Cannot execute direct Git delivery. | `coding_result`, `modified_files` |
| **Reviewer** | `git_diff`, `git_status`, `read_file`, `search_code`, `run_tests`, `verify_goal` | Inspecting git diffs, evaluating test output, approving or requesting code changes. **Read-only** (No file edits). | `review_result`, `review_status`, `review_feedback` |
| **Orchestrator** | Shared state evaluation & flow control | Iteration tracking (`multi_agent_iteration`), review feedback routing, enforcing iteration limits (`max_multi_agent_iterations`). | `multi_agent_iteration` |

---

## 🔄 Self-Correcting Review Loop

If the **Reviewer Agent** issues `STATUS: CHANGES_REQUESTED`, the **Orchestrator** automatically routes the state back to the **Coder Agent** with the detailed `review_feedback`. The Coder fixes the code, runs tests, and resubmits to the Reviewer. The loop repeats until:
1. `review_status == "approved"` $\rightarrow$ Task completed.
2. `review_status == "blocked"` $\rightarrow$ Escalate & stop.
3. `multi_agent_iteration >= max_multi_agent_iterations` $\rightarrow$ Max iterations reached, stop safely.

---

## 📊 Single-Agent vs Multi-Agent Comparison (`MultiAgentEvaluator`)

`MultiAgentEvaluator` provides standard benchmarking metrics comparing single-agent and multi-agent modes across identical tasks:
- **Execution Time (s)**
- **Modified Files Count**
- **Review Status & Retries**
- **Test Pass Rate**

---

## 🔍 Advanced Code Retrieval / RAG (Phase 11)

- **Repository-Aware Chunking**: Functions, classes, and logical line sections.
- **Hybrid Ranking Formula**:
  $$\text{Final Score} = (\text{Lexical Score} \times 0.4) + (\text{Semantic Score} \times 0.5) + (\text{Metadata Score} \times 0.1)$$

---

## 🚦 Human-in-the-Loop Approval & Delivery Model (Phase 10)

- **Actions Requiring Explicit Approval**: `git commit`, `git push`, `create_pull_request`.
- **Read-Only / Safe Local Actions**: `git_status`, `git_diff`, `retrieve_hybrid_context`.

---

## 🛡️ Security & Sandbox Controls (`app/sandbox.py`)

- ❌ Workspace boundary enforced (file traversal outside sandbox root blocked).
- ❌ Secrets (`.env`, `*.pem`, `*.key`, `id_rsa`, `credentials*`, `secrets*`) excluded.
- ❌ Destructive Git commands (`git reset --hard`, `git clean -fd`, `git push --force`) prohibited.
- ❌ Role tool separation strictly enforced.

---

## 🛠️ Execution Modes

```python
from app.agent import run_agent

# Default Single-Agent Mode (Backward compatible with Phases 1-11)
result_single = run_agent(
    goal="Fix bug in calculation module",
    mode="single_agent",
)

# Multi-Agent Orchestration Mode (Phase 12)
result_multi = run_agent(
    goal="Fix bug in calculation module",
    mode="multi_agent",
)
```

---

## 🧪 Running Tests

```bash
# Run multi-agent test suite
pytest -v tests/test_multi_agent.py

# Run full project test suite
pytest -v
```
