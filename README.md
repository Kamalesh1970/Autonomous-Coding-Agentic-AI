# Autonomous Coding Agentic AI (Phase 14: Repeatable Benchmark & Experimental Framework)

Minimal autonomous software engineering agent built with **Python** and **LangGraph**.

Reference Architecture: Inspired by [Open SWE](https://github.com/langchain-ai/open-swe) as a conceptual benchmark, built from the ground up as a minimal agentic foundation.

---

## 🎯 What is Phase 14?

Phase 14 builds a **Repeatable Benchmark and Experimental Evaluation Framework** (`app/benchmark.py`) for the Autonomous Coding Agentic AI framework. It enables automated, reproducible software-engineering benchmark task execution against isolated temporary repositories, evaluating task completion, self-correction recovery, retrieval mode impact, single-agent vs multi-agent performance, software quality metrics, and exporting secret-sanitized JSON and CSV reports.

```text
  [BENCHMARK TASK]  --->  [ISOLATED TEMP REPO]  --->  [AGENT EXECUTION]  --->  [STATS AGGREGATION]  --->  [JSON & CSV EXPORTS]
```

---

## 🏆 Benchmark Tasks & Categories

The built-in deterministic benchmark suite (`BUILTIN_BENCHMARK_TASKS`) covers realistic software-engineering tasks across categories and difficulty levels:

| Category | Task ID | Description | Difficulty |
| :--- | :--- | :--- | :--- |
| **Bug Fix** | `T001` | Fix arithmetic operation in `calculator.py`. | `easy` |
| **Feature Addition** | `T002` | Add new function `multiply()` in `math_ops.py`. | `easy` |
| **Test Fix** | `T003` | Update `format_name()` implementation to satisfy test assertion. | `easy` |
| **Repo Understanding** | `T004` | Locate `APP_TIMEOUT` in `config.py` and update usages in `service.py`. | `medium` |
| **Multi-File Change** | `T005` | Refactor model schema in `models.py` and creation service in `service.py`. | `medium` |
| **Self-Correction** | `T006` | Fix buggy function in `utils.py` that fails initial test run. | `medium` |

---

## 🧪 Experiment Configurations & Repeatable Runs

`ExperimentConfig` supports programmatic matrix experiments:
- **Agent Modes**: `single_agent`, `multi_agent`.
- **Retrieval Modes**: `lexical`, `semantic`, `hybrid`.
- **Repeatable Runs**: `run_count` ($N$ runs) storing distinct `BenchmarkResult` instances.
- **Reproducibility**: Explicit seed logging (`seed`).

---

## 📊 Aggregated Metrics & Exports

| Metric | Calculation | Description |
| :--- | :--- | :--- |
| **Pass@1** | $\text{Success Rate at } \text{run\_index}=1$ | Standard benchmark metric measuring first-pass success. |
| **Success Rate** | $\frac{\text{Successful Runs}}{\text{Total Runs}}$ | Overall percentage of benchmark runs completing task verification. |
| **Mean / Median Time** | $\text{mean}(t), \text{median}(t)$ | Execution overhead timing. |
| **Recovery Rate** | $\frac{\text{Successful Recoveries}}{\text{Recovery Attempts}}$ | Self-correction capability score following test failures. |
| **Validation Rate** | $\frac{\text{Validation Passes}}{\text{Validation Attempts}}$ | Percentage of pytest executions passing cleanly. |

Evaluation reports and benchmark summaries export to secret-sanitized JSON (`export_benchmark_json`) and CSV (`export_benchmark_csv`).

> [!NOTE]
> Benchmark results must be generated from actual executions and are not guaranteed to favor any architecture.

---

## 🛠️ Running Benchmarks via CLI

```bash
# Run single-agent benchmark with hybrid retrieval (3 runs)
python -m app.benchmark --mode single_agent --retrieval hybrid --runs 3 --export-json results.json --export-csv results.csv

# Run multi-agent benchmark with lexical retrieval
python -m app.benchmark --mode multi_agent --retrieval lexical --runs 1 --export-json multi_results.json
```

---

## 📊 Observability & Evaluation System (Phase 13)

- **Execution Trace**: Monotonic event logging (`agent_start`, `tool_call`, `tool_result`, `agent_end`).
- **Secret Scrubbing**: Automatic redaction (`[REDACTED_SECRET]`) for credentials, API keys (`sk-*`, `ghp_*`), passwords, and private keys.

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

## 🧪 Running Pytest Suite

```bash
# Run Phase 14 benchmark test suite
pytest -v tests/test_benchmark.py

# Run complete project test suite
pytest -v
```
