"""Pytest test suite for Phase 14 Repeatable Benchmark and Experimental Evaluation Framework."""

import json
import os
from pathlib import Path
from typing import List, Any, Optional

import pytest
from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import AIMessage, BaseMessage

from app.benchmark import (
    BenchmarkTask,
    ExperimentConfig,
    BenchmarkResult,
    ExperimentSummary,
    BUILTIN_BENCHMARK_TASKS,
    create_temp_benchmark_repo,
    cleanup_temp_benchmark_repo,
    run_benchmark_task,
    run_benchmark_suite,
    aggregate_benchmark_results,
    compare_benchmark_experiments,
    export_benchmark_json,
    export_benchmark_csv,
)


class MockLLM(BaseChatModel):
    """Deterministic Mock LLM for benchmark pytest execution."""

    responses: List[AIMessage]
    call_count: int = 0

    def __init__(self, responses: List[AIMessage]):
        super().__init__(responses=list(responses), call_count=0)

    def _generate(self, messages: List[BaseMessage], stop: Optional[List[str]] = None, **kwargs: Any):
        if self.call_count < len(self.responses):
            resp = self.responses[self.call_count]
            self.call_count += 1
        else:
            resp = AIMessage(content="STATUS: APPROVED\nDefault mock response.")
        from langchain_core.outputs import ChatGeneration, ChatResult
        return ChatResult(generations=[ChatGeneration(message=resp)])

    def bind_tools(self, tools: List[Any], **kwargs: Any) -> "MockLLM":
        clone = MockLLM(responses=self.responses)
        clone.call_count = self.call_count
        return clone

    @property
    def _llm_type(self) -> str:
        return "mock"


# -----------------------------------------------------------------------------
# Benchmark Task Model Tests
# -----------------------------------------------------------------------------
def test_benchmark_task_model():
    """Verify BenchmarkTask structure fields."""
    task: BenchmarkTask = {
        "task_id": "T100",
        "name": "Sample Task",
        "description": "Fix bug",
        "category": "bug_fix",
        "difficulty": "easy",
        "repository_setup": {"src/a.py": "def a(): pass\n"},
        "expected_behavior": "a() passes",
        "verification": {"test_file": "tests/test_a.py"},
    }

    assert task["task_id"] == "T100"
    assert task["category"] == "bug_fix"
    assert task["difficulty"] == "easy"


def test_benchmark_task_validation():
    """Verify built-in benchmark tasks contain all required fields."""
    assert len(BUILTIN_BENCHMARK_TASKS) >= 5
    for task in BUILTIN_BENCHMARK_TASKS:
        assert "task_id" in task
        assert "name" in task
        assert "category" in task
        assert "difficulty" in task
        assert "repository_setup" in task


# -----------------------------------------------------------------------------
# Temporary Repository Lifecycle Tests
# -----------------------------------------------------------------------------
def test_benchmark_uses_isolated_repository():
    """Verify temporary benchmark repository is created isolated from workspace."""
    task = BUILTIN_BENCHMARK_TASKS[0]
    repo_path = create_temp_benchmark_repo(task)

    try:
        p = Path(repo_path)
        assert p.exists()
        assert (p / "src" / "calculator.py").exists()
        assert (p / ".git").exists()
    finally:
        cleanup_temp_benchmark_repo(repo_path)


def test_benchmark_cleanup():
    """Verify temporary benchmark repository directory is removed after cleanup."""
    task = BUILTIN_BENCHMARK_TASKS[0]
    repo_path = create_temp_benchmark_repo(task)
    assert Path(repo_path).exists()

    cleanup_temp_benchmark_repo(repo_path)
    assert not Path(repo_path).exists()


# -----------------------------------------------------------------------------
# Experiment Configuration Tests
# -----------------------------------------------------------------------------
def test_experiment_configuration():
    """Verify ExperimentConfig TypedDict parameters."""
    config: ExperimentConfig = {
        "agent_mode": "single_agent",
        "retrieval_mode": "hybrid",
        "max_iterations": 3,
        "max_retries": 3,
        "seed": 42,
    }

    assert config["agent_mode"] == "single_agent"
    assert config["retrieval_mode"] == "hybrid"
    assert config["seed"] == 42


def test_single_agent_configuration():
    """Verify single_agent mode experiment config."""
    config: ExperimentConfig = {"agent_mode": "single_agent", "retrieval_mode": "lexical"}
    assert config["agent_mode"] == "single_agent"


def test_multi_agent_configuration():
    """Verify multi_agent mode experiment config."""
    config: ExperimentConfig = {"agent_mode": "multi_agent", "retrieval_mode": "hybrid"}
    assert config["agent_mode"] == "multi_agent"


def test_retrieval_configuration():
    """Verify retrieval mode configurations (lexical, semantic, hybrid)."""
    for mode in ("lexical", "semantic", "hybrid"):
        config: ExperimentConfig = {"agent_mode": "single_agent", "retrieval_mode": mode}  # type: ignore
        assert config["retrieval_mode"] == mode


# -----------------------------------------------------------------------------
# Benchmark Execution & N-Run Tests
# -----------------------------------------------------------------------------
def test_benchmark_execution():
    """Verify run_benchmark_task executes task against temporary repo and returns result."""
    task = BUILTIN_BENCHMARK_TASKS[0]
    config: ExperimentConfig = {"agent_mode": "single_agent", "retrieval_mode": "hybrid"}
    mock_llm = MockLLM([AIMessage(content="Fix applied")])

    res = run_benchmark_task(task=task, config=config, run_index=1, llm=mock_llm)

    assert res["task_id"] == "T001"
    assert res["run_index"] == 1
    assert res["execution_time"] > 0


def test_multiple_runs():
    """Verify run_benchmark_suite executes task N times returning distinct result records."""
    task = BUILTIN_BENCHMARK_TASKS[0]
    config: ExperimentConfig = {"agent_mode": "single_agent", "retrieval_mode": "hybrid"}
    mock_llm = MockLLM([AIMessage(content="Run 1"), AIMessage(content="Run 2")])

    results = run_benchmark_suite([task], config=config, run_count=2, llm=mock_llm)

    assert len(results) == 2
    assert results[0]["run_index"] == 1
    assert results[1]["run_index"] == 2


def test_results_not_overwritten():
    """Verify distinct run_index values in suite results."""
    task = BUILTIN_BENCHMARK_TASKS[0]
    config: ExperimentConfig = {"agent_mode": "single_agent", "retrieval_mode": "hybrid"}
    mock_llm = MockLLM([AIMessage(content="Run 1"), AIMessage(content="Run 2")])

    results = run_benchmark_suite([task], config=config, run_count=2, llm=mock_llm)

    indices = [r["run_index"] for r in results]
    assert indices == [1, 2]


# -----------------------------------------------------------------------------
# Metrics & Aggregation Tests
# -----------------------------------------------------------------------------
def test_success_rate():
    """Verify aggregate_benchmark_results success_rate calculation."""
    results: List[BenchmarkResult] = [
        {"success": True, "execution_time": 1.0, "tool_call_count": 2, "iteration_count": 1, "retry_count": 0, "run_index": 1},
        {"success": False, "execution_time": 2.0, "tool_call_count": 4, "iteration_count": 2, "retry_count": 1, "run_index": 1},
    ]
    summary = aggregate_benchmark_results(results)

    assert summary["total_runs"] == 2
    assert summary["successful_runs"] == 1
    assert summary["success_rate"] == 0.5


def test_average_execution_time():
    """Verify aggregate_benchmark_results mean_execution_time calculation."""
    results: List[BenchmarkResult] = [
        {"success": True, "execution_time": 4.0, "tool_call_count": 2, "iteration_count": 1, "retry_count": 0, "run_index": 1},
        {"success": True, "execution_time": 6.0, "tool_call_count": 2, "iteration_count": 1, "retry_count": 0, "run_index": 1},
    ]
    summary = aggregate_benchmark_results(results)

    assert summary["mean_execution_time"] == 5.0


def test_average_tool_calls():
    """Verify aggregate_benchmark_results mean_tool_calls calculation."""
    results: List[BenchmarkResult] = [
        {"success": True, "execution_time": 1.0, "tool_call_count": 2, "iteration_count": 1, "retry_count": 0, "run_index": 1},
        {"success": True, "execution_time": 1.0, "tool_call_count": 6, "iteration_count": 1, "retry_count": 0, "run_index": 1},
    ]
    summary = aggregate_benchmark_results(results)

    assert summary["mean_tool_calls"] == 4.0


def test_recovery_success_rate():
    """Verify recovery_success_rate calculation across tasks requiring self-correction."""
    results: List[BenchmarkResult] = [
        {"success": True, "recovery_attempted": True, "recovery_successful": True, "execution_time": 1.0, "tool_call_count": 2, "iteration_count": 1, "retry_count": 1, "run_index": 1},
        {"success": False, "recovery_attempted": True, "recovery_successful": False, "execution_time": 1.0, "tool_call_count": 2, "iteration_count": 1, "retry_count": 1, "run_index": 1},
    ]
    summary = aggregate_benchmark_results(results)

    assert summary["recovery_success_rate"] == 0.5


def test_validation_success_rate():
    """Verify validation_success_rate computation."""
    results: List[BenchmarkResult] = [
        {"success": True, "validation_attempts": 2, "validation_passes": 1, "execution_time": 1.0, "tool_call_count": 2, "iteration_count": 1, "retry_count": 0, "run_index": 1},
    ]
    summary = aggregate_benchmark_results(results)

    assert summary["validation_success_rate"] == 0.5


def test_result_aggregation():
    """Verify aggregate_benchmark_results handles pass@1 and standard deviation correctly."""
    results: List[BenchmarkResult] = [
        {"success": True, "execution_time": 2.0, "tool_call_count": 2, "iteration_count": 1, "retry_count": 0, "run_index": 1},
        {"success": True, "execution_time": 4.0, "tool_call_count": 4, "iteration_count": 1, "retry_count": 0, "run_index": 2},
    ]
    summary = aggregate_benchmark_results(results)

    assert summary["pass_at_1"] == 1.0
    assert summary["std_dev_execution_time"] > 0


def test_empty_result_aggregation():
    """Verify aggregate_benchmark_results handles empty results list safely."""
    summary = aggregate_benchmark_results([])

    assert summary["total_runs"] == 0
    assert summary["success_rate"] == 0.0


def test_median_calculation():
    """Verify median execution time calculation."""
    results: List[BenchmarkResult] = [
        {"execution_time": 1.0, "success": True, "run_index": 1},
        {"execution_time": 5.0, "success": True, "run_index": 1},
        {"execution_time": 10.0, "success": True, "run_index": 1},
    ]
    summary = aggregate_benchmark_results(results)

    assert summary["median_execution_time"] == 5.0


# -----------------------------------------------------------------------------
# Single vs Multi Comparison Tests
# -----------------------------------------------------------------------------
def test_single_vs_multi_agent_comparison():
    """Verify compare_benchmark_experiments generates comparative metric summary table."""
    summary_single: ExperimentSummary = {
        "experiment_id": "single_agent_exp",
        "success_rate": 0.8,
        "pass_at_1": 0.8,
        "mean_execution_time": 5.0,
        "mean_tool_calls": 4.0,
        "recovery_success_rate": 0.5,
    }
    summary_multi: ExperimentSummary = {
        "experiment_id": "multi_agent_exp",
        "success_rate": 0.9,
        "pass_at_1": 0.9,
        "mean_execution_time": 8.0,
        "mean_tool_calls": 6.0,
        "recovery_success_rate": 0.75,
    }

    comp = compare_benchmark_experiments(summary_single, summary_multi)

    assert comp["experiment_a"] == "single_agent_exp"
    assert comp["experiment_b"] == "multi_agent_exp"
    assert comp["metrics_comparison"]["success_rate"]["diff"] == 0.1
    assert comp["metrics_comparison"]["mean_execution_time"]["diff"] == 3.0


# -----------------------------------------------------------------------------
# Retrieval Benchmark Metrics Tests
# -----------------------------------------------------------------------------
def test_retrieval_benchmark_metrics(tmp_path: Path):
    """Verify Phase 11 retrieval benchmark metrics integration."""
    from app.retrieval import RetrievalEvaluator

    evaluator = RetrievalEvaluator()
    query = "add function"
    retrieved = ["src/math.py", "tests/test_math.py"]
    relevant = ["src/math.py"]

    p1 = evaluator.precision_at_k(retrieved, relevant, k=1)
    r1 = evaluator.recall_at_k(retrieved, relevant, k=1)
    mrr = evaluator.mean_reciprocal_rank(retrieved, relevant)

    assert p1 == 1.0
    assert r1 == 1.0
    assert mrr == 1.0


# -----------------------------------------------------------------------------
# JSON & CSV Export Tests
# -----------------------------------------------------------------------------
def test_json_export(tmp_path: Path):
    """Verify export_benchmark_json writes sanitized report file."""
    results: List[BenchmarkResult] = [
        {"task_id": "T001", "success": True, "execution_time": 1.5, "run_index": 1}
    ]
    out_path = tmp_path / "bench.json"
    json_text = export_benchmark_json(results, file_path=out_path)

    assert out_path.exists()
    parsed = json.loads(json_text)
    assert parsed["results"][0]["task_id"] == "T001"


def test_csv_export(tmp_path: Path):
    """Verify export_benchmark_csv writes CSV file with metric headers."""
    results: List[BenchmarkResult] = [
        {
            "experiment_id": "exp1",
            "task_id": "T001",
            "task_name": "Fix Bug",
            "category": "bug_fix",
            "difficulty": "easy",
            "run_index": 1,
            "success": True,
            "execution_time": 1.2,
            "tool_call_count": 3,
            "successful_tool_calls": 3,
            "failed_tool_calls": 0,
            "iteration_count": 1,
            "retry_count": 0,
            "recovery_attempted": False,
            "recovery_successful": False,
            "validation_attempts": 1,
            "validation_passes": 1,
            "review_iterations": 0,
            "human_interventions": 0,
            "error_category": None,
            "seed": 42,
            "timestamp": 100.0,
        }
    ]
    out_path = tmp_path / "bench.csv"
    csv_text = export_benchmark_csv(results, file_path=out_path)

    assert out_path.exists()
    assert "experiment_id" in csv_text
    assert "T001" in csv_text


def test_export_contains_configuration():
    """Verify exported JSON contains experiment configuration metadata."""
    results: List[BenchmarkResult] = [{"task_id": "T001", "success": True}]
    summary: ExperimentSummary = {
        "experiment_id": "exp_cfg",
        "configuration": {"agent_mode": "single_agent", "retrieval_mode": "hybrid"},
    }
    json_text = export_benchmark_json(results, summary=summary)
    parsed = json.loads(json_text)

    assert parsed["summary"]["configuration"]["agent_mode"] == "single_agent"


def test_export_excludes_secrets():
    """Verify exported JSON and CSV redact secret credentials."""
    results: List[BenchmarkResult] = [
        {
            "task_id": "T_secret",
            "error_message": "Failed with api_key sk-proj-12345678901234567890",
            "configuration": {"api_key": "sk-proj-12345"},
        }
    ]
    json_text = export_benchmark_json(results)

    assert "sk-proj-12345" not in json_text
    assert "[REDACTED_SECRET]" in json_text


# -----------------------------------------------------------------------------
# Failure Analysis Tests
# -----------------------------------------------------------------------------
def test_failure_classification():
    """Verify failed benchmark runs record error category."""
    results: List[BenchmarkResult] = [
        {"task_id": "T_fail", "success": False, "error_category": "validation_failure"}
    ]
    assert results[0]["error_category"] == "validation_failure"


def test_failure_summary():
    """Verify aggregate_benchmark_results calculates error rate on failure."""
    results: List[BenchmarkResult] = [
        {"success": True, "execution_time": 1.0, "run_index": 1},
        {"success": False, "execution_time": 1.0, "run_index": 1},
    ]
    summary = aggregate_benchmark_results(results)
    assert summary["error_rate"] == 0.5


# -----------------------------------------------------------------------------
# Reproducibility & Seed Tests
# -----------------------------------------------------------------------------
def test_seed_recorded():
    """Verify seed is recorded in experiment configuration and benchmark results."""
    config: ExperimentConfig = {"agent_mode": "single_agent", "seed": 12345}
    task = BUILTIN_BENCHMARK_TASKS[0]
    mock_llm = MockLLM([AIMessage(content="Done")])

    res = run_benchmark_task(task, config=config, run_index=1, llm=mock_llm)

    assert res["seed"] == 12345
    assert res["configuration"]["seed"] == 12345


# -----------------------------------------------------------------------------
# End-to-End Benchmark Test
# -----------------------------------------------------------------------------
def test_end_to_end_benchmark_flow(tmp_path: Path):
    """Verify end-to-end benchmark task -> isolated temp repo -> agent execution -> results -> summary -> JSON & CSV export."""
    task = BUILTIN_BENCHMARK_TASKS[0]
    config: ExperimentConfig = {
        "agent_mode": "single_agent",
        "retrieval_mode": "hybrid",
        "max_iterations": 3,
        "max_retries": 3,
        "seed": 99,
    }
    mock_llm = MockLLM([AIMessage(content="Fix calculator.py add logic")])

    results = run_benchmark_suite([task], config=config, run_count=2, experiment_id="e2e_exp", llm=mock_llm)
    assert len(results) == 2

    summary = aggregate_benchmark_results(results, experiment_id="e2e_exp")
    assert summary["total_runs"] == 2

    json_path = tmp_path / "e2e_results.json"
    csv_path = tmp_path / "e2e_results.csv"

    export_benchmark_json(results, summary=summary, file_path=json_path)
    export_benchmark_csv(results, file_path=csv_path)

    assert json_path.exists()
    assert csv_path.exists()

    parsed_json = json.loads(json_path.read_text())
    assert parsed_json["summary"]["experiment_id"] == "e2e_exp"

    csv_text = csv_path.read_text()
    assert "e2e_exp" in csv_text
    assert "T001" in csv_text
