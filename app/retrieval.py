"""Phase 11 Advanced Code Retrieval / RAG: Chunking, Embeddings, Hybrid Ranking, Caching & Evaluation."""

import ast
import hashlib
import math
import os
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

from app.sandbox import ExecutionSandbox, SecurityError


# -----------------------------------------------------------------------------
# 1. Data Structures
# -----------------------------------------------------------------------------
@dataclass
class CodeChunk:
    """Metadata-rich representation of a code chunk."""
    file_path: str
    start_line: int
    end_line: int
    content: str
    language: str = "python"
    symbol_name: Optional[str] = None
    symbol_type: Optional[str] = None  # "function", "class", "method", "module", "block"
    chunk_id: str = field(default="")

    def __post_init__(self):
        if not self.chunk_id:
            self.chunk_id = f"{self.file_path}:L{self.start_line}-L{self.end_line}"


@dataclass
class RetrievalResult:
    """Structured result item from hybrid code retrieval."""
    chunk: CodeChunk
    lexical_score: float
    semantic_score: float
    metadata_score: float
    final_score: float

    def to_dict(self) -> Dict[str, Any]:
        return {
            "chunk_id": self.chunk.chunk_id,
            "file_path": self.chunk.file_path,
            "start_line": self.chunk.start_line,
            "end_line": self.chunk.end_line,
            "language": self.chunk.language,
            "symbol_name": self.chunk.symbol_name,
            "symbol_type": self.chunk.symbol_type,
            "content": self.chunk.content,
            "lexical_score": round(self.lexical_score, 4),
            "semantic_score": round(self.semantic_score, 4),
            "metadata_score": round(self.metadata_score, 4),
            "final_score": round(self.final_score, 4),
        }


# -----------------------------------------------------------------------------
# 2. Repository-Aware Code Chunker
# -----------------------------------------------------------------------------
class CodeChunker:
    """Divides repository files into meaningful, metadata-rich code chunks."""

    SENSITIVE_PATTERNS = [
        re.compile(r"^\.env(\..*)?$", re.IGNORECASE),
        re.compile(r".*\.pem$", re.IGNORECASE),
        re.compile(r".*\.key$", re.IGNORECASE),
        re.compile(r".*id_rsa.*", re.IGNORECASE),
        re.compile(r".*credentials.*", re.IGNORECASE),
        re.compile(r".*secrets.*", re.IGNORECASE),
    ]

    IGNORED_DIRS = {".git", "__pycache__", ".venv", "venv", "node_modules", ".pytest_cache", ".agent_memory"}

    def __init__(
        self,
        min_chunk_lines: int = 4,
        max_chunk_lines: int = 50,
        max_chunk_chars: int = 1500,
    ):
        self.min_chunk_lines = min_chunk_lines
        self.max_chunk_lines = max_chunk_lines
        self.max_chunk_chars = max_chunk_chars

    @classmethod
    def is_sensitive_file(cls, file_name: str) -> bool:
        """Check if filename matches sensitive credential or secret patterns."""
        base_name = Path(file_name).name
        return any(pattern.match(base_name) for pattern in cls.SENSITIVE_PATTERNS)

    def detect_language(self, file_path: str) -> str:
        ext = Path(file_path).suffix.lower()
        mapping = {
            ".py": "python",
            ".js": "javascript",
            ".ts": "typescript",
            ".jsx": "javascript",
            ".tsx": "typescript",
            ".html": "html",
            ".css": "css",
            ".json": "json",
            ".md": "markdown",
            ".sh": "bash",
            ".yml": "yaml",
            ".yaml": "yaml",
            ".toml": "toml",
        }
        return mapping.get(ext, "text")

    def _split_lines_into_chunks(
        self,
        file_path: str,
        lines: List[str],
        start_line: int,
        language: str,
        symbol_name: Optional[str] = None,
        symbol_type: str = "block",
    ) -> List[CodeChunk]:
        """Splits lines into chunks of at most max_chunk_lines, preserving line numbers and metadata."""
        chunks: List[CodeChunk] = []
        if not lines:
            return chunks

        total_lines = len(lines)
        chunk_line_size = max(1, self.max_chunk_lines)

        for i in range(0, total_lines, chunk_line_size):
            sub_lines = lines[i : i + chunk_line_size]
            sub_content = "\n".join(sub_lines)
            if not sub_content.strip():
                continue
            sub_start = start_line + i
            sub_end = start_line + i + len(sub_lines) - 1
            chunks.append(
                CodeChunk(
                    file_path=file_path,
                    start_line=sub_start,
                    end_line=sub_end,
                    content=sub_content[: self.max_chunk_chars],
                    language=language,
                    symbol_name=symbol_name,
                    symbol_type=symbol_type,
                )
            )

        return chunks

    def chunk_python_file(self, file_path: str, content: str) -> List[CodeChunk]:
        """AST-based chunking for Python files with fallback to block chunking."""
        chunks: List[CodeChunk] = []
        lines = content.splitlines()
        if not lines:
            return []

        try:
            tree = ast.parse(content)
            visited_lines = set()

            for node in ast.iter_child_nodes(tree):
                if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
                    start = getattr(node, "lineno", 1)
                    end = getattr(node, "end_lineno", len(lines))
                    node_lines = lines[start - 1 : end]
                    symbol_type = "class" if isinstance(node, ast.ClassDef) else "function"

                    sub_chunks = self._split_lines_into_chunks(
                        file_path=file_path,
                        lines=node_lines,
                        start_line=start,
                        language="python",
                        symbol_name=node.name,
                        symbol_type=symbol_type,
                    )
                    chunks.extend(sub_chunks)
                    visited_lines.update(range(start, end + 1))

            # Collect unparsed top-level blocks or module preamble
            unvisited_start = None
            for idx in range(1, len(lines) + 1):
                if idx not in visited_lines:
                    if unvisited_start is None:
                        unvisited_start = idx
                else:
                    if unvisited_start is not None:
                        block_lines = lines[unvisited_start - 1 : idx - 1]
                        sub_chunks = self._split_lines_into_chunks(
                            file_path=file_path,
                            lines=block_lines,
                            start_line=unvisited_start,
                            language="python",
                            symbol_name=None,
                            symbol_type="module",
                        )
                        chunks.extend(sub_chunks)
                        unvisited_start = None

            if unvisited_start is not None:
                block_lines = lines[unvisited_start - 1 :]
                sub_chunks = self._split_lines_into_chunks(
                    file_path=file_path,
                    lines=block_lines,
                    start_line=unvisited_start,
                    language="python",
                    symbol_name=None,
                    symbol_type="module",
                )
                chunks.extend(sub_chunks)

            if chunks:
                return chunks
        except Exception:
            pass

        # Fallback to line-block chunking if AST parsing fails or yields nothing
        return self.chunk_line_blocks(file_path, content, language="python")

    def chunk_line_blocks(self, file_path: str, content: str, language: str) -> List[CodeChunk]:
        """Generic line-block chunker for source code files."""
        lines = content.splitlines() if isinstance(content, str) else content
        return self._split_lines_into_chunks(
            file_path=file_path,
            lines=lines,
            start_line=1,
            language=language,
            symbol_name=None,
            symbol_type="block",
        )

    def chunk_file(self, file_path: str, content: str) -> List[CodeChunk]:
        """Parse file content into structured chunks based on file type."""
        if self.is_sensitive_file(file_path):
            return []

        language = self.detect_language(file_path)
        if language == "python":
            return self.chunk_python_file(file_path, content)
        else:
            return self.chunk_line_blocks(file_path, content, language)


# -----------------------------------------------------------------------------
# 3. Embedding Model Abstraction
# -----------------------------------------------------------------------------
class BaseEmbeddingModel:
    """Abstract interface for document and query text embeddings."""

    @property
    def dimension(self) -> int:
        raise NotImplementedError

    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        raise NotImplementedError

    def embed_query(self, text: str) -> List[float]:
        raise NotImplementedError


def _stable_hash(s: str) -> int:
    """Deterministic hash function consistent across processes and platforms."""
    return int(hashlib.md5(s.encode("utf-8")).hexdigest()[:8], 16)


def tokenize_code_text(text: str) -> List[str]:
    """Tokenize text into normalized terms, splitting camelCase and snake_case identifiers."""
    raw_words = re.findall(r"\w+", text)
    tokens: List[str] = []

    stop_words = {
        "def", "return", "pass", "self", "import", "from", "class", "if", "else", "elif",
        "for", "while", "in", "and", "or", "not", "is", "true", "false", "none", "val",
        "var", "let", "const", "function", "async", "await", "try", "except", "finally",
        "with", "as", "assert", "break", "continue", "lambda", "global", "nonlocal",
    }

    concept_map = {
        "auth": ["auth", "authenticate", "authentication", "login", "signin", "credentials", "password", "token", "jwt", "session"],
        "math": ["math", "calculate", "calculation", "sum", "add", "subtract", "multiply", "divide", "number", "numeric"],
        "db": ["db", "database", "sql", "query", "connect", "connection", "table", "row", "column", "select", "insert"],
        "web": ["web", "http", "api", "request", "response", "route", "router", "endpoint", "middleware", "controller"],
    }

    term_to_concept = {}
    for concept, terms in concept_map.items():
        for term in terms:
            term_to_concept[term] = f"concept_{concept}"

    for raw in raw_words:
        sub_words = re.findall(r"[A-Z]?[a-z]+|[A-Z]+(?=[A-Z][a-z]|\d|\b)|[0-9]+", raw)
        if not sub_words:
            sub_words = [raw]

        all_terms = [raw.lower()] + [w.lower() for w in sub_words]
        for w in all_terms:
            if not w or w in stop_words or len(w) < 2:
                continue
            tokens.append(w)

            if len(w) >= 4:
                tokens.append(w[:4])
            if len(w) >= 5:
                tokens.append(w[:5])

            if w in term_to_concept:
                tokens.append(term_to_concept[w])

    return tokens


class MockEmbeddingModel(BaseEmbeddingModel):
    """Deterministic, lightweight local embedding model for testing without network/API dependencies.

    Uses identifier tokenization, sub-word stemming, concept mapping, and stable L2-normalized vector hashing.
    """

    def __init__(self, dim: int = 64):
        self._dim = dim

    @property
    def dimension(self) -> int:
        return self._dim

    def _vectorize(self, text: str) -> List[float]:
        vec = [0.0] * self._dim
        clean_text = text.strip()
        if not clean_text:
            return vec

        tokens = tokenize_code_text(clean_text)
        if not tokens:
            return vec

        for token in tokens:
            idx = _stable_hash(token) % self._dim
            weight = 2.0 if token.startswith("concept_") else 1.0
            vec[idx] += weight

        # L2 Normalize
        norm = math.sqrt(sum(v * v for v in vec))
        if norm > 1e-9:
            vec = [v / norm for v in vec]
        return vec

    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        return [self._vectorize(t) for t in texts]

    def embed_query(self, text: str) -> List[float]:
        return self._vectorize(text)


class OpenAIEmbeddingModel(BaseEmbeddingModel):
    """Optional production embedding model wrapper using OpenAI text-embedding-3-small."""

    def __init__(self, model_name: str = "text-embedding-3-small", dim: int = 1536):
        self.model_name = model_name
        self._dim = dim

    @property
    def dimension(self) -> int:
        return self._dim

    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        try:
            from langchain_openai import OpenAIEmbeddings

            embedder = OpenAIEmbeddings(model=self.model_name)
            return embedder.embed_documents(texts)
        except Exception:
            # Fallback to mock vectorization if API call fails
            mock = MockEmbeddingModel(dim=self._dim)
            return mock.embed_documents(texts)

    def embed_query(self, text: str) -> List[float]:
        try:
            from langchain_openai import OpenAIEmbeddings

            embedder = OpenAIEmbeddings(model=self.model_name)
            return embedder.embed_query(text)
        except Exception:
            mock = MockEmbeddingModel(dim=self._dim)
            return mock.embed_query(text)


# -----------------------------------------------------------------------------
# 4. Hybrid Code Retrieval Index
# -----------------------------------------------------------------------------
class HybridCodeIndex:
    """In-memory hybrid vector and lexical code retrieval index for a workspace."""

    def __init__(
        self,
        workspace_root: Union[str, Path] = ".",
        embedding_model: Optional[BaseEmbeddingModel] = None,
        lexical_weight: float = 0.4,
        semantic_weight: float = 0.5,
        metadata_weight: float = 0.1,
    ):
        self.sandbox = ExecutionSandbox(sandbox_root=workspace_root)
        self.workspace_root = self.sandbox.sandbox_root
        self.embedding_model = embedding_model or MockEmbeddingModel(dim=64)

        self.lexical_weight = lexical_weight
        self.semantic_weight = semantic_weight
        self.metadata_weight = metadata_weight

        self.chunker = CodeChunker()
        self.chunks: List[CodeChunk] = []
        self.embeddings: List[List[float]] = []
        self.file_mtimes: Dict[str, float] = {}
        self.is_indexed: bool = False

    @staticmethod
    def cosine_similarity(v1: List[float], v2: List[float]) -> float:
        """Compute cosine similarity between two vector lists."""
        if not v1 or not v2 or len(v1) != len(v2):
            return 0.0
        dot = sum(a * b for a, b in zip(v1, v2))
        norm1 = math.sqrt(sum(a * a for a in v1))
        norm2 = math.sqrt(sum(b * b for b in v2))
        if norm1 <= 1e-9 or norm2 <= 1e-9:
            return 0.0
        return max(0.0, min(1.0, dot / (norm1 * norm2)))

    def _should_rebuild(self) -> bool:
        """Check if any workspace files have been modified or added since last indexing."""
        if not self.is_indexed:
            return True

        current_files: Dict[str, float] = {}
        for root, dirs, files in os.walk(self.workspace_root, followlinks=False):
            # Remove ignored directories
            dirs[:] = [d for d in dirs if d not in CodeChunker.IGNORED_DIRS]

            for file in files:
                if CodeChunker.is_sensitive_file(file):
                    continue
                file_path = Path(root) / file
                try:
                    rel_path = str(file_path.relative_to(self.workspace_root))
                    mtime = file_path.stat().st_mtime
                    current_files[rel_path] = mtime
                except Exception:
                    continue

        if current_files != self.file_mtimes:
            return True
        return False

    def build_index(self, force: bool = False) -> None:
        """Scans workspace, chunks source files, and builds vector/lexical embeddings."""
        if not force and not self._should_rebuild():
            return

        self.chunks.clear()
        self.embeddings.clear()
        self.file_mtimes.clear()

        new_chunks: List[CodeChunk] = []
        new_mtimes: Dict[str, float] = {}

        for root, dirs, files in os.walk(self.workspace_root, followlinks=False):
            dirs[:] = [d for d in dirs if d not in CodeChunker.IGNORED_DIRS]

            for file in sorted(files):
                if CodeChunker.is_sensitive_file(file):
                    continue

                file_path = Path(root) / file
                try:
                    rel_path = str(file_path.relative_to(self.workspace_root))
                    mtime = file_path.stat().st_mtime
                except Exception:
                    continue

                # Read text safely using ExecutionSandbox
                content = self.sandbox.read_file(rel_path)
                if content.startswith("Error:"):
                    continue

                file_chunks = self.chunker.chunk_file(rel_path, content)
                new_chunks.extend(file_chunks)
                new_mtimes[rel_path] = mtime

        self.chunks = new_chunks
        self.file_mtimes = new_mtimes

        if self.chunks:
            texts = [f"{c.file_path} {c.symbol_name or ''}\n{c.content}" for c in self.chunks]
            self.embeddings = self.embedding_model.embed_documents(texts)
        else:
            self.embeddings = []

        self.is_indexed = True

    def calculate_lexical_score(self, query_tokens: List[str], chunk: CodeChunk) -> float:
        """Calculate lexical term match score between query tokens and chunk content."""
        if not query_tokens:
            return 0.0

        content_lower = chunk.content.lower()
        file_lower = chunk.file_path.lower()
        symbol_lower = (chunk.symbol_name or "").lower()

        token_hits = 0
        total_frequency = 0

        for token in query_tokens:
            if len(token) < 2:
                continue
            count = content_lower.count(token)
            if count > 0:
                token_hits += 1
                total_frequency += count
            if token in file_lower:
                token_hits += 2
            if token in symbol_lower:
                token_hits += 3

        if not query_tokens:
            return 0.0

        coverage_ratio = token_hits / (len(query_tokens) * 3)
        frequency_boost = min(1.0, math.log1p(total_frequency) / 5.0)
        score = (coverage_ratio * 0.7) + (frequency_boost * 0.3)
        return max(0.0, min(1.0, score))

    def calculate_metadata_score(self, query_tokens: List[str], chunk: CodeChunk) -> float:
        """Calculate metadata relevance score based on symbol names and file paths."""
        if not query_tokens:
            return 0.0

        score = 0.0
        file_path_lower = chunk.file_path.lower()
        symbol_lower = (chunk.symbol_name or "").lower()

        for token in query_tokens:
            if len(token) < 2:
                continue
            if token in file_path_lower:
                score += 0.4
            if symbol_lower and token in symbol_lower:
                score += 0.6

        return min(1.0, score)

    def search(
        self,
        query: str,
        top_k: int = 3,
        max_context_chars: int = 4000,
        max_chunk_chars: int = 1500,
    ) -> List[RetrievalResult]:
        """Perform hybrid retrieval search over indexed workspace code chunks."""
        self.build_index()

        if not self.chunks or not query or not query.strip():
            return []

        effective_max_chunk_chars = min(max_chunk_chars, max_context_chars)

        query_clean = query.strip()
        query_tokens = [t.lower() for t in re.findall(r"\w+", query_clean) if len(t) > 1]
        query_vec = self.embedding_model.embed_query(query_clean)

        results: List[RetrievalResult] = []

        for idx, chunk in enumerate(self.chunks):
            # 1. Semantic Similarity
            chunk_vec = self.embeddings[idx] if idx < len(self.embeddings) else []
            sem_score = self.cosine_similarity(query_vec, chunk_vec)

            # 2. Lexical Score
            lex_score = self.calculate_lexical_score(query_tokens, chunk)

            # 3. Metadata Score
            meta_score = self.calculate_metadata_score(query_tokens, chunk)

            # 4. Final Hybrid Score
            final_score = (
                (lex_score * self.lexical_weight)
                + (sem_score * self.semantic_weight)
                + (meta_score * self.metadata_weight)
            )

            if final_score > 0.001:
                bounded_content = chunk.content[:effective_max_chunk_chars]
                bounded_chunk = CodeChunk(
                    file_path=chunk.file_path,
                    start_line=chunk.start_line,
                    end_line=chunk.end_line,
                    content=bounded_content,
                    language=chunk.language,
                    symbol_name=chunk.symbol_name,
                    symbol_type=chunk.symbol_type,
                    chunk_id=chunk.chunk_id,
                )
                results.append(
                    RetrievalResult(
                        chunk=bounded_chunk,
                        lexical_score=lex_score,
                        semantic_score=sem_score,
                        metadata_score=meta_score,
                        final_score=final_score,
                    )
                )

        # Sort by final_score descending
        results.sort(key=lambda r: r.final_score, reverse=True)

        # Strict Context Budget Bounding
        selected: List[RetrievalResult] = []
        current_chars = 0

        for item in results[:top_k]:
            remaining_budget = max_context_chars - current_chars
            if remaining_budget <= 0:
                break

            item_len = len(item.chunk.content)
            if item_len > remaining_budget:
                item.chunk.content = item.chunk.content[:remaining_budget]
                selected.append(item)
                current_chars += len(item.chunk.content)
                break
            else:
                selected.append(item)
                current_chars += item_len

        return selected


# -----------------------------------------------------------------------------
# 5. Retrieval Benchmarking & Evaluation
# -----------------------------------------------------------------------------
class RetrievalEvaluator:
    """Evaluator for measuring Precision@K, Recall@K, and MRR across retrieval methods."""

    @staticmethod
    def precision_at_k(retrieved_files: List[str], expected_files: List[str], k: int) -> float:
        """Compute Precision@K."""
        if k <= 0:
            return 0.0
        top_k = retrieved_files[:k]
        if not top_k:
            return 0.0
        relevant_count = sum(1 for f in top_k if any(exp in f for exp in expected_files))
        return relevant_count / len(top_k)

    @staticmethod
    def recall_at_k(retrieved_files: List[str], expected_files: List[str], k: int) -> float:
        """Compute Recall@K."""
        if not expected_files:
            return 1.0
        top_k = retrieved_files[:k]
        relevant_count = sum(1 for f in top_k if any(exp in f for exp in expected_files))
        return relevant_count / len(expected_files)

    @staticmethod
    def mean_reciprocal_rank(retrieved_files: List[str], expected_files: List[str]) -> float:
        """Compute Reciprocal Rank (RR) for a single query."""
        for idx, f in enumerate(retrieved_files, start=1):
            if any(exp in f for exp in expected_files):
                return 1.0 / idx
        return 0.0

    @classmethod
    def evaluate_methods(
        cls,
        workspace_root: Union[str, Path],
        benchmark_tasks: List[Dict[str, Any]],
        top_ks: List[int] = [1, 3, 5],
    ) -> Dict[str, Dict[str, float]]:
        """Evaluates Lexical, Semantic, and Hybrid retrieval modes on benchmark tasks."""
        modes = {
            "Lexical": (1.0, 0.0, 0.0),
            "Semantic": (0.0, 1.0, 0.0),
            "Hybrid": (0.4, 0.5, 0.1),
        }

        eval_summary: Dict[str, Dict[str, float]] = {}

        for mode_name, (lw, sw, mw) in modes.items():
            index = HybridCodeIndex(
                workspace_root=workspace_root,
                lexical_weight=lw,
                semantic_weight=sw,
                metadata_weight=mw,
            )

            p_at_k: Dict[int, List[float]] = {k: [] for k in top_ks}
            r_at_k: Dict[int, List[float]] = {k: [] for k in top_ks}
            mrrs: List[float] = []

            for task in benchmark_tasks:
                query = task["query"]
                expected = task["expected_files"]
                max_k = max(top_ks)

                results = index.search(query=query, top_k=max_k)
                retrieved = [r.chunk.file_path for r in results]

                for k in top_ks:
                    p_at_k[k].append(cls.precision_at_k(retrieved, expected, k))
                    r_at_k[k].append(cls.recall_at_k(retrieved, expected, k))

                mrrs.append(cls.mean_reciprocal_rank(retrieved, expected))

            mode_metrics: Dict[str, float] = {}
            for k in top_ks:
                mode_metrics[f"P@{k}"] = round(sum(p_at_k[k]) / len(p_at_k[k]), 4) if p_at_k[k] else 0.0
                mode_metrics[f"R@{k}"] = round(sum(r_at_k[k]) / len(r_at_k[k]), 4) if r_at_k[k] else 0.0
            mode_metrics["MRR"] = round(sum(mrrs) / len(mrrs), 4) if mrrs else 0.0

            eval_summary[mode_name] = mode_metrics

        return eval_summary
