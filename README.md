# Autonomous Coding Agentic AI (Phase 13: Observability & Evaluation System)

Minimal autonomous software engineering agent built with **Python** and **LangGraph**.

Reference Architecture: Inspired by [Open SWE](https://github.com/langchain-ai/open-swe) as a conceptual benchmark, built from the ground up as a minimal agentic foundation.

---

## 🎯 What is Phase 13?

Phase 13 equips the agent framework with a lightweight, non-intrusive **Observability, Telemetry, and Evaluation System** (`app/evaluation.py`). The system collects structured execution events, tracks task metrics, classifies errors, evaluates goal verification and recovery success, exports sanitized JSON reports, and enables repeatable single-agent vs multi-agent benchmark comparisons with statistical aggregations.

```text
  [AGENT EXECUTION]  --->  [EXECUTION TRACE (In-Memory)]  --->  [GOAL VERIFICATION & METRICS]  --->  [SANITIZED JSON REPORT]
```

---

## 📊 Tracked Metrics & Observability Architecture

| Metric Area | Tracked Fields | Description / Purpose |
| :--- | :--- | :--- |
| **Task Success** | `task_success`, `final_status` | Evaluates true task completion via Phase 7 goal verification (`success`, `failed`, `blocked`, `uncertain`). |
| **Execution Duration** | `execution_time`, `start_time`, `end_time` | Monotonic timing for agent execution overhead without wall-clock drift. |
| **Tool Call Metrics** | `tool_call_count`, `successful_tool_calls`, `failed_tool_calls`, `tool_calls_by_tool` | Granular per-tool usage and failure counts without logging sensitive arguments. |
| **Iterative Execution** | `iteration_count`, `retry_count` | Number of meaningful reasoning iterations and code modification retries. |
| **Recovery Metrics** | `recovery_attempts`, `successful_recoveries`, `failed_recoveries` | Tracks self-correction loops where failed test runs are followed by code fixes and passing tests. |
| **Validation Metrics** | `validation_attempts`, `validation_passes`, `validation_failures`, `timeouts` | Tracks pytest execution passes, failures, and timeout events. |
| **Retrieval Metrics** | `query_count`, `retrieved_chunks`, `queries` | Integrates Phase 11 hybrid semantic/lexical/metadata context retrieval telemetry. |
| **Multi-Agent Metrics** | `orchestration_iterations`, `review_iterations`, `review_status`, `review_approvals`, `review_rejections` | Phase 12 multi-agent role execution distribution and reviewer feedback decisions. |
| **Human Interventions**| `human_interventions` | Tracks explicit human approval requests (`commit`, `push`, `pull_request`). |
| **Error Classification**| `error_count`, `error_categories` | Categorizes errors (`tool_error`, `validation_error`, `verification_error`, `security_error`, `retrieval_error`, `agent_error`, `approval_error`, `timeout`). |

---

## 🛡️ Security & Secret Scrubbing

Observability is strictly isolated from secret leaks:
- ❌ API keys (`sk-*`, `ghp_*`), passwords, bearer tokens, private keys, `.env` file dumps are automatically redacted to `[REDACTED_SECRET]`.
- ❌ Raw source code dumps, full prompts, or authorization headers are never logged in telemetry or JSON exports.

---

## ⚙️ JSON Report Export

Evaluation reports are serializable to clean JSON:

```json
{
  "task_id": "task_12345",
  "task_success": true,
  "final_status": "success",
  "execution_time": 4.12,
  "tool_call_count": 5,
  "successful_tool_calls": 5,
  "failed_tool_calls": 0,
  "tool_calls_by_tool": {
    "read_file": 2,
    "replace_in_file": 1,
    "run_tests": 1,
    "verify_goal": 1
  },
  "iteration_count": 1,
  "retry_count": 0,
  "recovery_attempts": 0,
  "successful_recoveries": 0,
  "human_interventions": 0,
  "error_count": 0
}
```

---

## 📈 Benchmark Aggregation & Mode Comparison

`calculate_benchmark_statistics` and `compare_agent_evaluations` compute aggregate statistics across $N$ task runs:
- **Statistical Aggregation**: `mean`, `median`, `min`, `max`, and `success_rate`.
- **Single-Agent vs Multi-Agent Comparison**: Side-by-side performance delta comparisons (execution time, tool call counts, success rate, iterations).

---

## 👥 Multi-Agent Architecture (Phase 12)

- **Specialized Roles**: Analyzer (Read-only), Coder (Edits & tests), Reviewer (Read-only quality evaluation), Orchestrator (Iterative routing).
- **Self-Correcting Review Loop**: Automatic retry on `STATUS: CHANGES_REQUESTED`.

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

## 🛠️ Usage Example

```python
from app.agent import run_agent
from app.evaluation import export_report_json

# Execute task with automated telemetry and evaluation report generation
state = run_agent(goal="Fix return value in math_utils.py", mode="single_agent")

# Access evaluation report
report = state["evaluation_report"]
print(f"Task Success: {report['task_success']}")
print(f"Execution Time: {report['execution_time']}s")

# Export sanitized JSON report to file
export_report_json(report, file_path="evaluation_report.json")
```

---

## 🧪 Running Tests

```bash
# Run Phase 13 observability & evaluation test suite
pytest -v tests/test_evaluation.py

# Run complete project test suite
pytest -v
```
