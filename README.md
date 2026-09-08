# Autonomous Coding Agentic AI (Phase 15: End-to-End Autonomous Agent System)

Minimal autonomous software engineering agent built with **Python** and **LangGraph**.

Technical Title: *"An Autonomous Agentic AI System for Repository-Level Software Engineering"*

---

## 🎯 What is Phase 15?

Phase 15 integrates all capabilities built across Phases 1–14 into a unified, observation-driven **End-to-End Autonomous Agent System**. The framework accepts high-level software engineering goals, understands repository structure, retrieves relevant code context, designs architectural plans, executes code modifications, runs tests, performs self-correction recovery on test failures, verifies goals against concrete repository evidence, enforces human-approval security boundaries, and outputs explicit task outcomes (`SUCCESS`, `FAILED`, `ESCALATED`).

```text
User Goal
   ↓
Understand Repository Structure
   ↓
Advanced Context Retrieval (Lexical / Semantic / Hybrid)
   ↓
Dynamic Architectural Planning & Task Decomposition
   ↓
[Optional] Plan Approval Gate  ←─ REQUIRE_PLAN_APPROVAL=true pauses here
   ↓
Tool Selection & Safe File Modification
   ↓
Automated Testing & Observation
   ↓
Failure Diagnosis & Autonomous Self-Correction Retry Loop
   ↓
Multi-Agent Code Review (Analyzer → Coder → Reviewer)
   ↓
Autonomous Goal Verification
   ↓
Human Approval & Git Delivery Boundary
   ↓
Final Outcome (SUCCESS / FAILED / ESCALATED)
```

---

## 🏁 Explicit Final Outcome Model

Every task execution produces a unambiguous, explicit final outcome status:

| Outcome | Description | Criteria |
| :--- | :--- | :--- |
| `SUCCESS` | Task completed & goal satisfied. | Original goal verified (`verify_goal` status is `passed`) or tests passed and goal requirements satisfied. |
| `FAILED` | Task failed to satisfy goal. | Goal verification failed, tests failed after exhausting retry attempts (`retry_count >= max_retries`), or execution error occurred. |
| `ESCALATED` | Task requires human approval/intervention. | External delivery action pending human approval (`approval_required=True` with `approval_status="pending"`), plan pending pre-execution approval (`plan_approval_required=True` with `plan_approval_status="pending"`), or review status is `blocked`. |

---

## 👥 Execution Modes & Capability Matrix

- **Single-Agent Mode**: Compact, fast reasoning loop for targeted file edits and self-correcting bug fixes.
- **Multi-Agent Mode**: Specialized role orchestration (`Analyzer` $\rightarrow$ `Coder` $\rightarrow$ `Reviewer` $\rightarrow$ `Orchestrator`) enforcing least-privilege tool subsets.

---

## 🏆 Repeatable Benchmark & Evaluation System (Phases 13 & 14)

- **Deterministic Benchmark Suite**: Built-in benchmark tasks (`BUILTIN_BENCHMARK_TASKS`) across bug fixes, feature additions, test fixes, repository navigation, multi-file edits, and self-correction recovery.
- **Isolated Repositories**: Executes tasks in temporary isolated sandbox directories with automatic cleanup.
- **Telemetry & Sanitized Exports**: Exports performance metrics and telemetry traces to JSON (`export_benchmark_json`) and CSV (`export_benchmark_csv`) with 100% credential redaction.

---

## ⚙️ Configurable LLM Providers

The agent supports multiple LLM providers without architectural changes:

* **Google Gemini** (`LLM_PROVIDER=gemini`)
* **OpenRouter** (`LLM_PROVIDER=openrouter`)
* **OpenAI** (`LLM_PROVIDER=openai`)

### Configuration Examples (`.env`)

**Using Gemini (with Automatic Key Failover & OpenRouter Fallback):**
```bash
LLM_PROVIDER=gemini
GEMINI_API_KEY_1=your_first_gemini_key
GEMINI_API_KEY_2=your_second_gemini_key
GEMINI_API_KEY_3=your_third_gemini_key
LLM_MODEL=gemini-3.6-flash

# Optional OpenRouter fallback if all Gemini keys fail
OPENROUTER_API_KEY=your_openrouter_key
```

**Using OpenRouter:**
```bash
LLM_PROVIDER=openrouter
OPENROUTER_API_KEY=your_openrouter_api_key
LLM_MODEL=anthropic/claude-3.5-sonnet
```

**Using OpenAI:**
```bash
LLM_PROVIDER=openai
OPENAI_API_KEY=your_openai_api_key
LLM_MODEL=gpt-4o-mini
```

---

## 🛠️ CLI Usage

```bash
# Execute agent with configured provider
python -m app.agent "Fix the failing calculator test." ~/agent-test-repo

# Execute single-agent benchmark suite
python -m app.benchmark --mode single_agent --retrieval hybrid --runs 3 --export-json results.json --export-csv results.csv

# Execute multi-agent benchmark suite
python -m app.benchmark --mode multi_agent --retrieval lexical --runs 1 --export-json multi_results.json
```

---

## 🧪 Running Pytest Suite

```bash
# Run LLM provider selection test suite
pytest -v tests/test_llm_provider.py

# Run Phase 15 end-to-end integration test suite
pytest -v tests/test_end_to_end.py

# Run complete project test suite
pytest -v
```

---

## 🔒 Pre-Execution Plan Approval Gate (`REQUIRE_PLAN_APPROVAL`)

Inspired by the [Open SWE](https://github.com/langchain-ai/open-swe) three-agent architecture, the agent supports an optional **human checkpoint on the plan itself** — before any file modification begins. This mirrors how the existing Phase 10 Git-delivery approval gate works.

### Environment Variable

| Variable | Default | Description |
| :--- | :--- | :--- |
| `REQUIRE_PLAN_APPROVAL` | `false` | Set to `true`, `1`, or `yes` to enable the pre-execution plan approval gate. |

### Behavior

| Mode | Behavior |
| :--- | :--- |
| `REQUIRE_PLAN_APPROVAL=false` (default) | Unchanged from Phase 15 flow — agent plans and executes autonomously. |
| `REQUIRE_PLAN_APPROVAL=true` | After the plan is generated and **before** any `write_file` / `replace_in_file` call executes, the agent **pauses**, sets `plan_approval_required=True` / `plan_approval_status="pending"`, and exposes the formatted plan in `state["plan_content"]`. No file edits occur until the plan is approved. Final outcome is `ESCALATED`. |

When plan approval is pending:
- `state["plan_approval_required"]` → `True`
- `state["plan_approval_status"]` → `"pending"`
- `state["plan_content"]` → Human-readable plan block (goal + task list)
- `state["modified_files"]` → `[]` (no edits made yet)
- `evaluation_report["final_outcome"]` → `"ESCALATED"`

### Approving / Rejecting a Plan

Use the same `approve_task()` API already used for Git-delivery approval:

```python
from app.agent import approve_task

# Approve — execution resumes from the approved plan
result = approve_task(
    task_id="task_abc123",
    decision="approved",          # or "approve", "yes", "pass"
    notes="Plan looks correct.",
    workspace_root="/path/to/repo",
    storage_dir=".agent_memory",
    llm=my_llm,
)

# Reject — gate remains closed, no file edits occur
result = approve_task(
    task_id="task_abc123",
    decision="rejected",
    notes="Plan scope is too broad — needs revision.",
    workspace_root="/path/to/repo",
    storage_dir=".agent_memory",
)
```

### `.env` Example

```bash
# Enable pre-execution plan approval gate
REQUIRE_PLAN_APPROVAL=true
```

> **Note:** Disabling `REQUIRE_PLAN_APPROVAL` (or omitting it entirely) restores the default Phase 15 autonomous behavior — no tests are affected.

---
