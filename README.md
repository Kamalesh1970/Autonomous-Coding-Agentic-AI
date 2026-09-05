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
| `ESCALATED` | Task requires human approval/intervention. | External delivery action pending human approval (`approval_required=True` with `approval_status="pending"`) or review status is `blocked`. |

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

## 🛠️ CLI Usage

```bash
# Execute single-agent benchmark suite
python -m app.benchmark --mode single_agent --retrieval hybrid --runs 3 --export-json results.json --export-csv results.csv

# Execute multi-agent benchmark suite
python -m app.benchmark --mode multi_agent --retrieval lexical --runs 1 --export-json multi_results.json
```

---

## 🧪 Running Pytest Suite

```bash
# Run Phase 15 end-to-end integration test suite
pytest -v tests/test_end_to_end.py

# Run complete project test suite (180+ tests)
pytest -v
```
