"""Repeatable benchmark and experimental evaluation framework for Phase 14 Autonomous Coding Agent."""

import argparse
import csv
import io
import json
import math
import os
import shutil
import statistics
import subprocess
import tempfile
import time
from pathlib import Path
from typing import Any, Literal, Sequence, TypedDict

from app.state import AgentState
from app.evaluation import (
    EvaluationReport,
    ExecutionTrace,
    export_report_json,
    generate_evaluation_report,
    sanitize_telemetry_dict,
)


# -----------------------------------------------------------------------------
# Data Models
# -----------------------------------------------------------------------------
class BenchmarkTask(TypedDict, total=False):
    """Structured representation of a repeatable software-engineering benchmark task."""
    task_id: str
    name: str
    description: str
    category: Literal["bug_fix", "feature_addition", "test_fix", "repo_understanding", "multi_file_change", "self_correction"]
    difficulty: Literal["easy", "medium", "hard"]
    repository_setup: dict[str, str]
    expected_behavior: str
    verification: dict[str, Any]


class ExperimentConfig(TypedDict, total=False):
    """Structured configuration for a benchmark experiment run."""
    agent_mode: Literal["single_agent", "multi_agent"]
    retrieval_mode: Literal["lexical", "semantic", "hybrid"]
    max_iterations: int
    max_retries: int
    seed: int | None


class BenchmarkResult(TypedDict, total=False):
    """Structured result recorded for a single benchmark task execution run."""
    experiment_id: str
    task_id: str
    task_name: str
    category: str
    difficulty: str
    configuration: ExperimentConfig
    run_index: int
    success: bool
    execution_time: float
    tool_call_count: int
    successful_tool_calls: int
    failed_tool_calls: int
    tool_calls_by_tool: dict[str, int]
    iteration_count: int
    retry_count: int
    recovery_attempted: bool
    recovery_successful: bool
    validation_attempts: int
    validation_passes: int
    review_iterations: int
    human_interventions: int
    error_category: str | None
    error_message: str | None
    seed: int | None
    timestamp: float


class ExperimentSummary(TypedDict, total=False):
    """Aggregated statistical summary across multiple benchmark task runs."""
    experiment_id: str
    configuration: ExperimentConfig
    total_runs: int
    successful_runs: int
    success_rate: float
    pass_at_1: float
    mean_execution_time: float
    median_execution_time: float
    min_execution_time: float
    max_execution_time: float
    std_dev_execution_time: float
    mean_tool_calls: float
    mean_iterations: float
    mean_retries: float
    recovery_success_rate: float
    validation_success_rate: float
    human_intervention_rate: float
    error_rate: float


# -----------------------------------------------------------------------------
# Built-in Deterministic Benchmark Suite
# -----------------------------------------------------------------------------
BUILTIN_BENCHMARK_TASKS: list[BenchmarkTask] = [
    {
        "task_id": "T001",
        "name": "Fix Arithmetic Bug",
        "description": "Fix incorrect addition logic in src/calculator.py where subtract is called instead of add.",
        "category": "bug_fix",
        "difficulty": "easy",
        "repository_setup": {
            "src/calculator.py": "def add(a, b):\n    return a - b  # Bug: should be a + b\n",
            "tests/test_calculator.py": "from src.calculator import add\n\ndef test_add():\n    assert add(2, 3) == 5\n",
        },
        "expected_behavior": "add(a, b) returns a + b",
        "verification": {"test_file": "tests/test_calculator.py"},
    },
    {
        "task_id": "T002",
        "name": "Add Multiply Feature",
        "description": "Add a new function multiply(a, b) in src/math_ops.py that returns a * b.",
        "category": "feature_addition",
        "difficulty": "easy",
        "repository_setup": {
            "src/math_ops.py": "def divide(a, b):\n    return a / b\n",
            "tests/test_math_ops.py": "from src.math_ops import multiply\n\ndef test_multiply():\n    assert multiply(3, 4) == 12\n",
        },
        "expected_behavior": "multiply(a, b) returns a * b",
        "verification": {"test_file": "tests/test_math_ops.py"},
    },
    {
        "task_id": "T003",
        "name": "Fix Failing Test Assertion",
        "description": "Update implementation in src/formatter.py so test_format_name in tests/test_formatter.py passes.",
        "category": "test_fix",
        "difficulty": "easy",
        "repository_setup": {
            "src/formatter.py": "def format_name(first, last):\n    return f'{first} {last}'\n",
            "tests/test_formatter.py": "from src.formatter import format_name\n\ndef test_format_name():\n    assert format_name('Jane', 'Doe') == 'Doe, Jane'\n",
        },
        "expected_behavior": "format_name(first, last) returns f'{last}, {first}'",
        "verification": {"test_file": "tests/test_formatter.py"},
    },
    {
        "task_id": "T004",
        "name": "Locate Constant & Update Service",
        "description": "Locate APP_TIMEOUT constant in src/config.py and update src/service.py to use APP_TIMEOUT.",
        "category": "repo_understanding",
        "difficulty": "medium",
        "repository_setup": {
            "src/config.py": "APP_TIMEOUT = 60\n",
            "src/service.py": "TIMEOUT = 10  # Should use APP_TIMEOUT from src.config\n\ndef get_timeout():\n    return TIMEOUT\n",
            "tests/test_service.py": "from src.service import get_timeout\n\ndef test_timeout():\n    assert get_timeout() == 60\n",
        },
        "expected_behavior": "get_timeout() returns 60",
        "verification": {"test_file": "tests/test_service.py"},
    },
    {
        "task_id": "T005",
        "name": "Multi-File Schema Refactor",
        "description": "Update User model in src/models.py to include status field and update src/service.py user creation.",
        "category": "multi_file_change",
        "difficulty": "medium",
        "repository_setup": {
            "src/models.py": "class User:\n    def __init__(self, name):\n        self.name = name\n",
            "src/service.py": "from src.models import User\n\ndef create_user(name):\n    u = User(name)\n    return u\n",
            "tests/test_user.py": "from src.service import create_user\n\ndef test_create_user():\n    u = create_user('Alice')\n    assert hasattr(u, 'status') and u.status == 'active'\n",
        },
        "expected_behavior": "User has status attribute defaulting to 'active'",
        "verification": {"test_file": "tests/test_user.py"},
    },
    {
        "task_id": "T006",
        "name": "Self-Correction Recovery",
        "description": "Fix buggy function src/utils.py which initially fails tests and requires recovery fix.",
        "category": "self_correction",
        "difficulty": "medium",
        "repository_setup": {
            "src/utils.py": "def is_even(n):\n    return n % 2 != 0  # Bug: returns True for odd\n",
            "tests/test_utils.py": "from src.utils import is_even\n\ndef test_is_even():\n    assert is_even(4) is True\n    assert is_even(5) is False\n",
        },
        "expected_behavior": "is_even(n) returns True for even numbers",
        "verification": {"test_file": "tests/test_utils.py"},
    },
]


# -----------------------------------------------------------------------------
# Temporary Repository Lifecycle
# -----------------------------------------------------------------------------
def create_temp_benchmark_repo(task: BenchmarkTask) -> str:
    """Creates an isolated temporary directory, populates task source files, and initializes a Git repository."""
    temp_dir = tempfile.mkdtemp(prefix=f"bench_{task.get('task_id', 'task')}_")
    base_path = Path(temp_dir).resolve()

    setup_files = task.get("repository_setup", {})
    for rel_path, content in setup_files.items():
        file_path = (base_path / rel_path).resolve()
        file_path.parent.mkdir(parents=True, exist_ok=True)
        file_path.write_text(content, encoding="utf-8")

    # Initialize Git repo for diff and branch operations
    try:
        subprocess.run(["git", "init"], cwd=str(base_path), capture_output=True, check=True)
        subprocess.run(["git", "config", "user.name", "BenchmarkRunner"], cwd=str(base_path), capture_output=True, check=True)
        subprocess.run(["git", "config", "user.email", "benchmark@agent.local"], cwd=str(base_path), capture_output=True, check=True)
        subprocess.run(["git", "add", "."], cwd=str(base_path), capture_output=True, check=True)
        subprocess.run(["git", "commit", "-m", "Initial benchmark state"], cwd=str(base_path), capture_output=True, check=True)
    except Exception:
        pass

    return str(base_path)


def cleanup_temp_benchmark_repo(repo_path: str) -> None:
    """Safely cleans up an isolated temporary benchmark repository directory."""
    if repo_path and os.path.exists(repo_path):
        try:
            shutil.rmtree(repo_path, ignore_errors=True)
        except Exception:
            pass


# -----------------------------------------------------------------------------
# Benchmark Runner & Execution Engine
# -----------------------------------------------------------------------------
def run_benchmark_task(
    task: BenchmarkTask,
    config: ExperimentConfig,
    run_index: int = 1,
    experiment_id: str = "exp_default",
    llm: Any = None,
) -> BenchmarkResult:
    """Executes a single benchmark task against an isolated temporary repository."""
    repo_path = create_temp_benchmark_repo(task)
    t_start = time.time()

    agent_mode = config.get("agent_mode", "single_agent")
    retrieval_mode = config.get("retrieval_mode", "hybrid")
    seed = config.get("seed")

    try:
        from app.agent import run_agent

        final_state = run_agent(
            goal=task.get("description", ""),
            workspace_root=repo_path,
            llm=llm,
            mode=agent_mode,
        )
        t_end = time.time()
        exec_time = round(max(0.001, t_end - t_start), 3)

        report = final_state.get("evaluation_report") or generate_evaluation_report(final_state, start_time=t_start, end_time=t_end)

        # Verification check
        ver_res = final_state.get("verification_result")
        val_res = final_state.get("validation_result")
        task_success = bool(report.get("task_success", False))

        err_cat = None
        err_msg = None
        if not task_success:
            err_cat = "failed"
            if val_res and isinstance(val_res, dict) and val_res.get("output"):
                err_msg = str(val_res.get("output"))[:200]

        result: BenchmarkResult = {
            "experiment_id": experiment_id,
            "task_id": str(task.get("task_id", "T000")),
            "task_name": str(task.get("name", "Task")),
            "category": str(task.get("category", "bug_fix")),
            "difficulty": str(task.get("difficulty", "easy")),
            "configuration": config,
            "run_index": run_index,
            "success": task_success,
            "execution_time": exec_time,
            "tool_call_count": int(report.get("tool_call_count", 0)),
            "successful_tool_calls": int(report.get("successful_tool_calls", 0)),
            "failed_tool_calls": int(report.get("failed_tool_calls", 0)),
            "tool_calls_by_tool": dict(report.get("tool_calls_by_tool", {})),
            "iteration_count": int(report.get("iteration_count", 1)),
            "retry_count": int(report.get("retry_count", 0)),
            "recovery_attempted": int(report.get("recovery_attempts", 0)) > 0,
            "recovery_successful": int(report.get("successful_recoveries", 0)) > 0,
            "validation_attempts": int(report.get("validation_attempts", 0)),
            "validation_passes": int(report.get("validation_passes", 0)),
            "review_iterations": int(final_state.get("multi_agent_iteration", 0) if agent_mode == "multi_agent" else 0),
            "human_interventions": int(report.get("human_interventions", 0)),
            "error_category": err_cat,
            "error_message": err_msg,
            "seed": seed,
            "timestamp": round(time.time(), 3),
        }
        return result

    finally:
        cleanup_temp_benchmark_repo(repo_path)


def run_benchmark_suite(
    tasks: list[BenchmarkTask],
    config: ExperimentConfig,
    run_count: int = 1,
    experiment_id: str = "exp_default",
    llm: Any = None,
) -> list[BenchmarkResult]:
    """Executes a suite of benchmark tasks N times, producing distinct BenchmarkResult records."""
    results: list[BenchmarkResult] = []
    for task in tasks:
        for idx in range(1, run_count + 1):
            res = run_benchmark_task(
                task=task,
                config=config,
                run_index=idx,
                experiment_id=experiment_id,
                llm=llm,
            )
            results.append(res)
    return results


# -----------------------------------------------------------------------------
# Statistical Aggregation & Comparisons
# -----------------------------------------------------------------------------
def aggregate_benchmark_results(
    results: list[BenchmarkResult],
    experiment_id: str = "exp_default",
) -> ExperimentSummary:
    """Calculates statistical aggregations (mean, median, pass@1, recovery rate) across benchmark results."""
    if not results:
        return ExperimentSummary(
            experiment_id=experiment_id,
            configuration={},
            total_runs=0,
            successful_runs=0,
            success_rate=0.0,
            pass_at_1=0.0,
            mean_execution_time=0.0,
            median_execution_time=0.0,
            min_execution_time=0.0,
            max_execution_time=0.0,
            std_dev_execution_time=0.0,
            mean_tool_calls=0.0,
            mean_iterations=0.0,
            mean_retries=0.0,
            recovery_success_rate=0.0,
            validation_success_rate=0.0,
            human_intervention_rate=0.0,
            error_rate=0.0,
        )

    config = results[0].get("configuration", {})
    total_runs = len(results)
    successful_runs = sum(1 for r in results if r.get("success"))
    success_rate = round(successful_runs / total_runs, 3)

    # Calculate Pass@1 (success rate of run_index == 1)
    run_1_results = [r for r in results if r.get("run_index") == 1]
    pass_at_1 = round(sum(1 for r in run_1_results if r.get("success")) / len(run_1_results), 3) if run_1_results else success_rate

    exec_times = [float(r.get("execution_time", 0.0)) for r in results]
    tool_counts = [int(r.get("tool_call_count", 0)) for r in results]
    iter_counts = [int(r.get("iteration_count", 0)) for r in results]
    retry_counts = [int(r.get("retry_count", 0)) for r in results]

    mean_exec = round(statistics.mean(exec_times), 3)
    med_exec = round(statistics.median(exec_times), 3)
    min_exec = round(min(exec_times), 3)
    max_exec = round(max(exec_times), 3)
    std_dev_exec = round(statistics.stdev(exec_times), 3) if len(exec_times) > 1 else 0.0

    mean_tools = round(statistics.mean(tool_counts), 3)
    mean_iters = round(statistics.mean(iter_counts), 3)
    mean_retries = round(statistics.mean(retry_counts), 3)

    rec_attempts = sum(1 for r in results if r.get("recovery_attempted"))
    rec_successes = sum(1 for r in results if r.get("recovery_successful"))
    recovery_rate = round(rec_successes / rec_attempts, 3) if rec_attempts > 0 else 0.0

    val_attempts = sum(int(r.get("validation_attempts", 0)) for r in results)
    val_passes = sum(int(r.get("validation_passes", 0)) for r in results)
    val_rate = round(val_passes / val_attempts, 3) if val_attempts > 0 else 0.0

    human_interventions = sum(int(r.get("human_interventions", 0)) for r in results)
    human_rate = round(human_interventions / total_runs, 3)
    error_rate = round(sum(1 for r in results if not r.get("success")) / total_runs, 3)

    return ExperimentSummary(
        experiment_id=experiment_id,
        configuration=config,
        total_runs=total_runs,
        successful_runs=successful_runs,
        success_rate=success_rate,
        pass_at_1=pass_at_1,
        mean_execution_time=mean_exec,
        median_execution_time=med_exec,
        min_execution_time=min_exec,
        max_execution_time=max_exec,
        std_dev_execution_time=std_dev_exec,
        mean_tool_calls=mean_tools,
        mean_iterations=mean_iters,
        mean_retries=mean_retries,
        recovery_success_rate=recovery_rate,
        validation_success_rate=val_rate,
        human_intervention_rate=human_rate,
        error_rate=error_rate,
    )


def compare_benchmark_experiments(
    summary_a: ExperimentSummary | dict[str, Any],
    summary_b: ExperimentSummary | dict[str, Any],
) -> dict[str, Any]:
    """Generates a side-by-side comparative metric table dictionary between two benchmark experiment summaries."""
    dict_a = dict(summary_a)
    dict_b = dict(summary_b)

    return {
        "experiment_a": dict_a.get("experiment_id", "Exp_A"),
        "experiment_b": dict_b.get("experiment_id", "Exp_B"),
        "metrics_comparison": {
            "success_rate": {"Exp_A": dict_a.get("success_rate", 0.0), "Exp_B": dict_b.get("success_rate", 0.0), "diff": round(dict_b.get("success_rate", 0.0) - dict_a.get("success_rate", 0.0), 3)},
            "pass_at_1": {"Exp_A": dict_a.get("pass_at_1", 0.0), "Exp_B": dict_b.get("pass_at_1", 0.0), "diff": round(dict_b.get("pass_at_1", 0.0) - dict_a.get("pass_at_1", 0.0), 3)},
            "mean_execution_time": {"Exp_A": dict_a.get("mean_execution_time", 0.0), "Exp_B": dict_b.get("mean_execution_time", 0.0), "diff": round(dict_b.get("mean_execution_time", 0.0) - dict_a.get("mean_execution_time", 0.0), 3)},
            "mean_tool_calls": {"Exp_A": dict_a.get("mean_tool_calls", 0.0), "Exp_B": dict_b.get("mean_tool_calls", 0.0), "diff": round(dict_b.get("mean_tool_calls", 0.0) - dict_a.get("mean_tool_calls", 0.0), 3)},
            "recovery_success_rate": {"Exp_A": dict_a.get("recovery_success_rate", 0.0), "Exp_B": dict_b.get("recovery_success_rate", 0.0), "diff": round(dict_b.get("recovery_success_rate", 0.0) - dict_a.get("recovery_success_rate", 0.0), 3)},
        },
    }


# -----------------------------------------------------------------------------
# Export Engine (JSON & CSV)
# -----------------------------------------------------------------------------
def export_benchmark_json(
    results: list[BenchmarkResult] | dict[str, Any],
    summary: ExperimentSummary | dict[str, Any] | None = None,
    file_path: str | Path | None = None,
) -> str:
    """Exports benchmark results and optional summary into a secret-sanitized JSON string or file."""
    if isinstance(results, list):
        clean_results = [sanitize_telemetry_dict(dict(r)) if isinstance(r, dict) else sanitize_telemetry_value(r) for r in results]
    elif isinstance(results, dict):
        clean_results = sanitize_telemetry_dict(dict(results))
    else:
        clean_results = sanitize_telemetry_value(results)

    clean_summary = sanitize_telemetry_dict(dict(summary)) if summary and isinstance(summary, dict) else None

    payload = {
        "summary": clean_summary,
        "results": clean_results,
    }

    json_text = json.dumps(payload, indent=2, ensure_ascii=False)

    if file_path:
        p = Path(file_path)
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(json_text, encoding="utf-8")

    return json_text


def export_benchmark_csv(
    results: list[BenchmarkResult],
    file_path: str | Path | None = None,
) -> str:
    """Exports benchmark results into a secret-sanitized CSV string or file."""
    output = io.StringIO()
    fieldnames = [
        "experiment_id",
        "task_id",
        "task_name",
        "category",
        "difficulty",
        "run_index",
        "success",
        "execution_time",
        "tool_call_count",
        "successful_tool_calls",
        "failed_tool_calls",
        "iteration_count",
        "retry_count",
        "recovery_attempted",
        "recovery_successful",
        "validation_attempts",
        "validation_passes",
        "review_iterations",
        "human_interventions",
        "error_category",
        "seed",
        "timestamp",
    ]

    writer = csv.DictWriter(output, fieldnames=fieldnames, extrasaction="ignore")
    writer.writeheader()

    for r in results:
        clean_row = sanitize_telemetry_dict(dict(r))
        writer.writerow(clean_row)

    csv_text = output.getvalue()

    if file_path:
        p = Path(file_path)
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(csv_text, encoding="utf-8")

    return csv_text


# -----------------------------------------------------------------------------
# CLI Entry Point
# -----------------------------------------------------------------------------
def main():
    """CLI entry point for running agent benchmark evaluations."""
    parser = argparse.ArgumentParser(description="Autonomous Coding Agent Benchmark Suite")
    parser.add_argument("--mode", choices=["single_agent", "multi_agent"], default="single_agent", help="Agent execution mode")
    parser.add_argument("--retrieval", choices=["lexical", "semantic", "hybrid"], default="hybrid", help="Retrieval mode")
    parser.add_argument("--runs", type=int, default=1, help="Number of benchmark runs per task")
    parser.add_argument("--seed", type=int, default=None, help="Random seed")
    parser.add_argument("--export-json", type=str, default=None, help="Export JSON report filepath")
    parser.add_argument("--export-csv", type=str, default=None, help="Export CSV report filepath")

    args = parser.parse_args()

    config: ExperimentConfig = {
        "agent_mode": args.mode,
        "retrieval_mode": args.retrieval,
        "max_iterations": 3,
        "max_retries": 3,
        "seed": args.seed,
    }

    print(f"=== Starting Benchmark Suite ({args.mode}, {args.retrieval}, {args.runs} runs) ===")
    results = run_benchmark_suite(BUILTIN_BENCHMARK_TASKS, config=config, run_count=args.runs)
    summary = aggregate_benchmark_results(results, experiment_id=f"exp_{args.mode}_{args.retrieval}")

    print("\n=== Benchmark Summary ===")
    print(f"Total Runs: {summary['total_runs']}")
    print(f"Success Rate: {summary['success_rate'] * 100:.1f}%")
    print(f"Pass@1: {summary['pass_at_1'] * 100:.1f}%")
    print(f"Mean Time: {summary['mean_execution_time']}s")
    print(f"Mean Tools: {summary['mean_tool_calls']}")

    if args.export_json:
        export_benchmark_json(results, summary=summary, file_path=args.export_json)
        print(f"Exported JSON report to {args.export_json}")

    if args.export_csv:
        export_benchmark_csv(results, file_path=args.export_csv)
        print(f"Exported CSV report to {args.export_csv}")


if __name__ == "__main__":
    main()
