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

### Intelligent Failover & Retry Classification

- **Automatic Failover Chain**: When `LLM_PROVIDER=gemini`, the system attempts requests in sequence across `GEMINI_API_KEY_1` $\rightarrow$ `GEMINI_API_KEY_2` $\rightarrow$ `GEMINI_API_KEY_3`. If all configured Gemini keys fail with retryable errors, execution automatically falls back to `OPENROUTER_API_KEY`.
- **Smart Exception Classification**:
  - **Retryable Errors** (triggers key failover / retry): HTTP `429` (rate limits, quota exhaustion, `resource_exhausted`), HTTP `500`/`502`/`503`/`504` server errors, and `401`/`403` auth key failures.
  - **Non-Retryable Errors** (fails fast without consuming remaining keys): HTTP `400` malformed request, HTTP `402` payment required/insufficient credits, `404` model not found, or schema errors.

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

## 🎬 Demo Scenarios

The Autonomous Coding Agent includes a deterministic demonstration suite covering 9 core capabilities:

| Demo Scenario | User Goal | Workflow Steps | Expected Outcome | CLI / Test Command |
| :--- | :--- | :--- | :--- | :--- |
| **Demo 1 — Simple Success** | `"Add a divide function to calculator.py and make sure the tests pass."` | Plans tasks $\rightarrow$ writes code and test $\rightarrow$ runs tests $\rightarrow$ verifies goal. | `SUCCESS` | `pytest -v tests/test_demo_scenarios.py -k test_demo_1_simple_success` |
| **Demo 2 — Real Bug Fix** | `"Fix the multiply bug in calculator.py so multiply(4,5) returns 20."` | Initial test fails $\rightarrow$ reads `calculator.py` $\rightarrow$ replaces `+` with `*` $\rightarrow$ tests pass $\rightarrow$ verifies goal. | `SUCCESS` | `pytest -v tests/test_demo_scenarios.py -k test_demo_2_real_bug_fix` |
| **Demo 3 — Self-Correction** | `"Fix square in math_utils.py"` | First edit fails test $\rightarrow$ observes traceback $\rightarrow$ applies exponent fix `x**2` $\rightarrow$ tests pass (`recovery_retries > 0`). | `SUCCESS` | `pytest -v tests/test_demo_scenarios.py -k test_demo_3_self_correction` |
| **Demo 4 — Repository Retrieval** | `"Fix the string reversal functionality."` | Hybrid search locates `string_utils.py` $\rightarrow$ edits target file $\rightarrow$ tests pass. | `SUCCESS` | `pytest -v tests/test_demo_scenarios.py -k test_demo_4_repository_understanding_retrieval` |
| **Demo 5 — Multi-Agent Workflow** | `"Implement main in app.py"` | `Analyzer` scans repo $\rightarrow$ `Coder` writes code $\rightarrow$ `Reviewer` approves change. | `SUCCESS` | `pytest -v tests/test_demo_scenarios.py -k test_demo_5_multi_agent_workflow` |
| **Demo 6 — Security Rejection** | `"Read system passwd file at ../../etc/passwd"` | Sandbox security policy detects path traversal escape $\rightarrow$ operation rejected. | `FAILED` / Rejection | `pytest -v tests/test_demo_scenarios.py -k test_demo_6_security_sandbox_rejection` |
| **Demo 7 — Human Approval** | `"Push changes to remote repository"` | Protected delivery action triggers human approval gate $\rightarrow$ execution pauses. | `ESCALATED` | `pytest -v tests/test_demo_scenarios.py -k test_demo_7_human_approval_boundary` |
| **Demo 8 — Failure / Escalation** | `"Fix add in calc.py"` | Repeated failing edits exhaust max retries (`retry_count >= 3`) $\rightarrow$ execution fails gracefully. | `FAILED` | `pytest -v tests/test_demo_scenarios.py -k test_demo_8_failure_escalation` |
| **Demo 9 — Final Report** | `"Verify calc.py functionality"` | CLI outputs full Phase 16C report block (Status, Goal, Workspace, Tests, Tool calls, Execution time). | `SUCCESS` | `pytest -v tests/test_demo_scenarios.py -k test_demo_9_final_execution_report` |

### Running the Demo Suite
```bash
# Run all demo scenarios
pytest -v tests/test_demo_scenarios.py

# Run an individual demo scenario CLI execution
python -m app.agent "Add divide function to calculator.py" /path/to/target/repo
```

---

## 🔧 Configuration Troubleshooting (Phase 16E)

### Common Provider Configuration Errors

| Error Message | Cause | Fix |
| :--- | :--- | :--- |
| `GEMINI_API_KEY (or GEMINI_API_KEY_1) is required when LLM_PROVIDER=gemini` | `LLM_PROVIDER=gemini` set but no valid Gemini API key found. | Set `GEMINI_API_KEY_1=<your_key>` in `.env`. |
| `OPENAI_API_KEY is required when LLM_PROVIDER=openai` | `LLM_PROVIDER=openai` set but key is missing or still the placeholder. | Set `OPENAI_API_KEY=<your_key>` in `.env`. |
| `OPENROUTER_API_KEY is required when LLM_PROVIDER=openrouter` | `LLM_PROVIDER=openrouter` set but key not configured. | Set `OPENROUTER_API_KEY=<your_key>` in `.env`. |
| `Unsupported LLM provider: <value>` | `LLM_PROVIDER` has an unrecognised value. | Use one of: `gemini`, `openai`, or `openrouter`. |

### Common Workspace Errors

| Error Message | Cause | Fix |
| :--- | :--- | :--- |
| `Workspace directory does not exist` | Provided workspace path does not exist on disk. | Create the directory first or provide an existing path. |
| `Workspace path is not a directory` | A file path was passed as workspace instead of a directory. | Provide a directory path as the second CLI argument. |
| `Workspace directory is not readable` | Agent process lacks read permission on the workspace. | Run `chmod +r <workspace>` or fix directory permissions. |

### Log Level (`AGENT_LOG_LEVEL`)

| Value | Behavior |
| :--- | :--- |
| `normal` (default) | Human-readable progress steps + final execution report only. |
| `debug` | Full message metadata trace, tool call trace, modified files, plan state, and detailed validation/verification output. |

```bash
# Enable debug output
AGENT_LOG_LEVEL=debug python -m app.agent "Fix failing tests." /path/to/repo
```

### Credential & Security Safety

- **API keys are never logged.** All output is sanitized — `AIzaSy*`, `sk-*`, and `ghp_*` patterns are always redacted to `[REDACTED_API_KEY]`.
- **Thought signatures are never logged.** Internal Gemini metadata (`__gemini_function_call_thought_signatures__`) is excluded from all normal-mode output.
- **Raw `AIMessage` objects are never printed.** Only human-readable summaries derived from message content appear in CLI output.
- **Subprocess environment isolation.** Test/command subprocesses run in a minimal clean environment that explicitly excludes `OPENAI_API_KEY`, `GEMINI_API_KEY*`, `OPENROUTER_API_KEY`, and other sensitive host variables.

---

## 🐳 Docker Deployment (Phase 16J)

The Autonomous Coding Agent and Web Dashboard are containerized using a production-oriented, non-root Docker architecture.

### Build Image
```bash
docker build -t autonomous-coding-agent:latest .
```

### Run Container
```bash
docker run -d \
  --name coding-agent \
  -p 8000:8000 \
  -e LLM_PROVIDER=gemini \
  -e GEMINI_API_KEY_1=your_gemini_api_key_here \
  autonomous-coding-agent:latest
```

### Check Health & Access Dashboard
- **Healthcheck**: `curl http://localhost:8000/health` (Returns `{"status": "ok", "service": "autonomous-coding-agent"}`)
- **Web Dashboard**: `http://localhost:8000`

---

## 🛠️ Recent Fixes & Engineering Rigor

The codebase has undergone key hardening and reliability enhancements:

1. **Gemini `thought_signature` Multi-Turn Tool-Calling Fix**: Implemented custom `GeminiChatOpenAI` wrapper and state serializer preserving Google Gemini internal thought signatures across multi-turn tool calling, preventing `thought_signature` schema errors during iterative reasoning.
2. **Intelligent Failover & Retry Classification**: Refined `is_retryable_error()` classification to distinguish retryable quota/rate-limit errors (`429`, `500`s, auth failover) from non-retryable configuration/billing errors (`400`, `402`, `404`), enabling smooth failover across `GEMINI_API_KEY_1` $\rightarrow$ `2` $\rightarrow$ `3` $\rightarrow$ OpenRouter.
3. **Real-Time Task State & Plan Synchronization**: Fixed task auto-completion and plan state sync (`sync_plan_from_messages`) so `modified_files`, `validation_result`, and `verification_result` update cleanly in `.agent_memory/`.
4. **Web API Error Message Specificity & Security**: Enhanced FastAPI exception handling in `app/web.py` to differentiate execution timeouts (`TimeoutError` returning `"Agent execution timed out"`) from general task failures (`"Agent execution failed"`), while sanitizing credentials and enforcing system directory workspace boundary protections (`/etc`, `/proc`, `/sys`).
