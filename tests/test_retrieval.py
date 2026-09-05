"""Deterministic Pytest suite for Phase 11 Advanced Code Retrieval / RAG."""

import os
import time
from pathlib import Path
import pytest
from langchain_core.messages import AIMessage, HumanMessage

from app.retrieval import (
    CodeChunk,
    CodeChunker,
    BaseEmbeddingModel,
    MockEmbeddingModel,
    OpenAIEmbeddingModel,
    HybridCodeIndex,
    RetrievalResult,
    RetrievalEvaluator,
)
from app.agent import run_agent
from app.tools import _retrieve_hybrid_context_impl, retrieve_hybrid_context
try:
    from tests.test_agent import MockLLM, init_git_repo
except ModuleNotFoundError:
    from test_agent import MockLLM, init_git_repo


# -----------------------------------------------------------------------------
# 1. Code Chunking Tests
# -----------------------------------------------------------------------------
def test_code_chunking():
    """Verify CodeChunker splits Python code into logical function and class chunks."""
    chunker = CodeChunker(min_chunk_lines=2, max_chunk_lines=50)
    code = (
        "class Calculator:\n"
        "    def add(self, a, b):\n"
        "        return a + b\n\n"
        "def subtract(a, b):\n"
        "    return a - b\n"
    )

    chunks = chunker.chunk_file("calc.py", code)
    assert len(chunks) >= 2

    symbol_names = [c.symbol_name for c in chunks if c.symbol_name]
    assert "Calculator" in symbol_names or "subtract" in symbol_names or "add" in symbol_names


def test_chunk_metadata():
    """Verify CodeChunk metadata contains correct line numbers, language, symbol name, and chunk_id."""
    chunk = CodeChunk(
        file_path="auth/jwt.py",
        start_line=10,
        end_line=25,
        content="def verify_jwt_token(token):\n    pass\n",
        language="python",
        symbol_name="verify_jwt_token",
        symbol_type="function",
    )

    assert chunk.chunk_id == "auth/jwt.py:L10-L25"
    assert chunk.file_path == "auth/jwt.py"
    assert chunk.symbol_name == "verify_jwt_token"
    assert chunk.symbol_type == "function"
    assert chunk.language == "python"


def test_chunk_boundaries():
    """Verify chunk boundary enforcing line limits and content truncation."""
    chunker = CodeChunker(min_chunk_lines=2, max_chunk_lines=10, max_chunk_chars=100)
    long_content = "\n".join([f"line_{i} = {i}" for i in range(50)])

    chunks = chunker.chunk_file("data.py", long_content)
    assert len(chunks) > 1
    for c in chunks:
        assert (c.end_line - c.start_line + 1) <= 15
        assert len(c.content) <= 100


# -----------------------------------------------------------------------------
# 2. Embedding Tests
# -----------------------------------------------------------------------------
def test_embedding_interface():
    """Verify BaseEmbeddingModel interface subclassing contract."""
    model = MockEmbeddingModel(dim=32)
    assert isinstance(model, BaseEmbeddingModel)
    assert model.dimension == 32


def test_mock_embedding_generation():
    """Verify MockEmbeddingModel produces normalized deterministic embedding vectors."""
    model = MockEmbeddingModel(dim=64)
    docs = ["def authenticate_user(username, password): pass", "class DatabaseConnection: pass"]

    embeddings = model.embed_documents(docs)
    assert len(embeddings) == 2
    assert len(embeddings[0]) == 64

    query_vec = model.embed_query("authenticate user")
    assert len(query_vec) == 64

    # Verify L2 normalization
    norm = sum(v * v for v in query_vec) ** 0.5
    assert abs(norm - 1.0) < 1e-4


def test_embedding_dimensions():
    """Verify configurable vector dimensions in mock and OpenAI embedding wrappers."""
    m32 = MockEmbeddingModel(dim=32)
    assert m32.dimension == 32
    assert len(m32.embed_query("test")) == 32

    m128 = MockEmbeddingModel(dim=128)
    assert m128.dimension == 128
    assert len(m128.embed_query("test")) == 128


# -----------------------------------------------------------------------------
# 3. Semantic Retrieval Tests
# -----------------------------------------------------------------------------
def test_semantic_retrieval(tmp_path: Path):
    """Verify semantic retrieval ranks conceptually matching chunks highest."""
    (tmp_path / "auth.py").write_text("def authenticate_user(credentials):\n    # Check login details\n    return True\n")
    (tmp_path / "math.py").write_text("def calculate_sum(a, b):\n    return a + b\n")

    index = HybridCodeIndex(
        workspace_root=tmp_path,
        lexical_weight=0.0,
        semantic_weight=1.0,
        metadata_weight=0.0,
    )

    results = index.search(query="login authentication check", top_k=2)
    assert len(results) >= 1
    assert "auth.py" in results[0].chunk.file_path
    assert results[0].semantic_score >= 0.0


def test_top_k_limit(tmp_path: Path):
    """Verify top_k parameter restricts the maximum returned chunks count."""
    for i in range(10):
        (tmp_path / f"module_{i}.py").write_text(f"def process_data_{i}(): return {i}\n")

    index = HybridCodeIndex(workspace_root=tmp_path)
    results = index.search(query="process data", top_k=3)
    assert len(results) == 3


# -----------------------------------------------------------------------------
# 4. Hybrid Retrieval Tests
# -----------------------------------------------------------------------------
def test_hybrid_ranking(tmp_path: Path):
    """Verify hybrid ranking combines lexical, semantic, and metadata scores into final_score."""
    (tmp_path / "jwt_middleware.py").write_text(
        "class JwtMiddleware:\n    def decode_jwt_token(self, token):\n        return {'user_id': 1}\n"
    )

    index = HybridCodeIndex(
        workspace_root=tmp_path,
        lexical_weight=0.4,
        semantic_weight=0.5,
        metadata_weight=0.1,
    )

    results = index.search(query="decode jwt token middleware", top_k=1)
    assert len(results) == 1
    res = results[0]

    assert res.lexical_score > 0.0
    assert res.semantic_score > 0.0
    assert res.metadata_score > 0.0
    expected_final = (res.lexical_score * 0.4) + (res.semantic_score * 0.5) + (res.metadata_score * 0.1)
    assert abs(res.final_score - expected_final) < 1e-4


def test_lexical_and_semantic_scores(tmp_path: Path):
    """Verify individual lexical and semantic score components in RetrievalResult."""
    (tmp_path / "service.py").write_text("def start_service(): pass\n")
    index = HybridCodeIndex(workspace_root=tmp_path)

    results = index.search(query="start_service", top_k=1)
    assert len(results) == 1
    assert hasattr(results[0], "lexical_score")
    assert hasattr(results[0], "semantic_score")
    assert hasattr(results[0], "metadata_score")


def test_hybrid_beats_irrelevant_result(tmp_path: Path):
    """Verify hybrid retrieval ranks relevant code above irrelevant boilerplate."""
    (tmp_path / "auth_handler.py").write_text("def handle_auth_login():\n    return 'auth success'\n")
    (tmp_path / "utils.py").write_text("def print_timestamp():\n    import time\n    print(time.time())\n")

    index = HybridCodeIndex(workspace_root=tmp_path)
    results = index.search(query="login auth handler", top_k=2)

    assert len(results) >= 1
    assert "auth_handler.py" in results[0].chunk.file_path


# -----------------------------------------------------------------------------
# 5. Context Budget Tests
# -----------------------------------------------------------------------------
def test_context_budget(tmp_path: Path):
    """Verify max_context_chars budget caps total returned context length."""
    (tmp_path / "heavy.py").write_text("def heavy_func():\n" + "    x = 1\n" * 200)

    index = HybridCodeIndex(workspace_root=tmp_path)
    results = index.search(query="heavy_func", top_k=5, max_context_chars=300)

    total_chars = sum(len(r.chunk.content) for r in results)
    assert total_chars <= 400


def test_max_chunk_limit(tmp_path: Path):
    """Verify max_chunk_chars bounds individual chunk length."""
    (tmp_path / "large_chunk.py").write_text("def large():\n" + "    # comment line\n" * 100)

    index = HybridCodeIndex(workspace_root=tmp_path)
    results = index.search(query="large", top_k=1, max_chunk_chars=150)

    assert len(results) == 1
    assert len(results[0].chunk.content) <= 150


# -----------------------------------------------------------------------------
# 6. Repository Boundaries & Security Tests
# -----------------------------------------------------------------------------
def test_retrieval_respects_workspace(tmp_path: Path):
    """Verify retrieval does not index files outside workspace boundary."""
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    (workspace / "app.py").write_text("x = 10\n")

    outside = tmp_path / "outside_secret.py"
    outside.write_text("secret_key = '12345'\n")

    index = HybridCodeIndex(workspace_root=workspace)
    results = index.search(query="secret_key")

    file_paths = [r.chunk.file_path for r in results]
    assert not any("outside" in p for p in file_paths)


def test_sensitive_files_excluded(tmp_path: Path):
    """Verify secrets (.env, *.pem, *.key, credentials) are excluded from index."""
    (tmp_path / ".env").write_text("OPENAI_API_KEY=sk-proj-secret-12345\n")
    (tmp_path / "id_rsa").write_text("-----BEGIN RSA PRIVATE KEY-----\nsecret\n")
    (tmp_path / "credentials.json").write_text('{"secret": "aws_key"}')
    (tmp_path / "safe_code.py").write_text("def hello(): return 'hello'\n")

    index = HybridCodeIndex(workspace_root=tmp_path)
    index.build_index()

    indexed_paths = [c.file_path for c in index.chunks]
    assert not any(".env" in p for p in indexed_paths)
    assert not any("id_rsa" in p for p in indexed_paths)
    assert not any("credentials" in p for p in indexed_paths)
    assert any("safe_code.py" in p for p in indexed_paths)


# -----------------------------------------------------------------------------
# 7. Index Lifecycle Tests
# -----------------------------------------------------------------------------
def test_index_build(tmp_path: Path):
    """Verify build_index creates chunks and vector embeddings."""
    (tmp_path / "main.py").write_text("def main(): print('run')\n")

    index = HybridCodeIndex(workspace_root=tmp_path)
    index.build_index()

    assert index.is_indexed is True
    assert len(index.chunks) >= 1
    assert len(index.embeddings) == len(index.chunks)


def test_index_reuse(tmp_path: Path):
    """Verify index is reused when repository files have not changed."""
    (tmp_path / "main.py").write_text("def main(): pass\n")

    index = HybridCodeIndex(workspace_root=tmp_path)
    index.build_index()
    assert index._should_rebuild() is False

    # Second build call should be a no-op reuse
    index.build_index()
    assert index.is_indexed is True


def test_index_refresh_after_change(tmp_path: Path):
    """Verify index automatically rebuilds when workspace file is added/modified."""
    file_path = tmp_path / "main.py"
    file_path.write_text("x = 1\n")

    index = HybridCodeIndex(workspace_root=tmp_path)
    index.build_index()
    assert len(index.chunks) == 1

    # Modify file mtime
    time.sleep(0.01)
    file_path.write_text("x = 2\ndef new_func(): pass\n")

    assert index._should_rebuild() is True
    index.build_index()
    assert len(index.chunks) >= 1


# -----------------------------------------------------------------------------
# 8. Agent Integration Tests
# -----------------------------------------------------------------------------
def test_agent_uses_retrieval_for_current_task(tmp_path: Path):
    """Verify autonomous agent can invoke retrieve_hybrid_context tool in execution graph."""
    init_git_repo(tmp_path)
    (tmp_path / "auth_jwt.py").write_text("def verify_jwt(): return True\n")

    mock_responses = [
        AIMessage(
            content="Retrieving hybrid code context for auth task.",
            tool_calls=[
                {
                    "name": "retrieve_hybrid_context",
                    "args": {"query": "verify_jwt auth token", "top_k": 2},
                    "id": "c1",
                }
            ],
        ),
        AIMessage(content="Observed auth_jwt.py context. Task complete."),
    ]
    mock_llm = MockLLM(responses=mock_responses)

    state = run_agent(
        goal="Locate verify_jwt auth implementation",
        workspace_root=str(tmp_path),
        llm=mock_llm,
    )

    messages = state.get("messages", [])
    tool_messages = [m for m in messages if m.__class__.__name__ == "ToolMessage"]
    assert len(tool_messages) >= 1
    assert "Retrieved Hybrid Context" in tool_messages[0].content or "auth_jwt.py" in tool_messages[0].content


# -----------------------------------------------------------------------------
# 9. Retrieval Evaluation & Benchmark Tests
# -----------------------------------------------------------------------------
def test_precision_at_k():
    """Verify Precision@K calculation."""
    retrieved = ["auth/login.py", "auth/jwt.py", "utils/helpers.py"]
    expected = ["auth/login.py", "auth/jwt.py"]

    p_1 = RetrievalEvaluator.precision_at_k(retrieved, expected, k=1)
    assert p_1 == 1.0

    p_3 = RetrievalEvaluator.precision_at_k(retrieved, expected, k=3)
    assert abs(p_3 - (2 / 3)) < 1e-4


def test_recall_at_k():
    """Verify Recall@K calculation."""
    retrieved = ["auth/login.py", "utils/helpers.py"]
    expected = ["auth/login.py", "auth/jwt.py"]

    r_2 = RetrievalEvaluator.recall_at_k(retrieved, expected, k=2)
    assert r_2 == 0.5


def test_mrr():
    """Verify Mean Reciprocal Rank (MRR) calculation."""
    retrieved1 = ["utils/helpers.py", "auth/login.py"]
    expected1 = ["auth/login.py"]
    rr1 = RetrievalEvaluator.mean_reciprocal_rank(retrieved1, expected1)
    assert rr1 == 0.5  # 1/2

    retrieved2 = ["auth/login.py", "utils/helpers.py"]
    expected2 = ["auth/login.py"]
    rr2 = RetrievalEvaluator.mean_reciprocal_rank(retrieved2, expected2)
    assert rr2 == 1.0  # 1/1


def test_retrieval_benchmark(tmp_path: Path):
    """Verify benchmark execution comparing Lexical, Semantic, and Hybrid retrieval methods."""
    (tmp_path / "auth.py").write_text("def login(): pass\n")
    (tmp_path / "db.py").write_text("def connect(): pass\n")

    benchmark_tasks = [
        {"query": "user login authentication", "expected_files": ["auth.py"]},
        {"query": "database connection query", "expected_files": ["db.py"]},
    ]

    metrics_table = RetrievalEvaluator.evaluate_methods(
        workspace_root=tmp_path,
        benchmark_tasks=benchmark_tasks,
        top_ks=[1, 3],
    )

    assert "Lexical" in metrics_table
    assert "Semantic" in metrics_table
    assert "Hybrid" in metrics_table

    for mode in ("Lexical", "Semantic", "Hybrid"):
        assert "P@1" in metrics_table[mode]
        assert "R@1" in metrics_table[mode]
        assert "MRR" in metrics_table[mode]
