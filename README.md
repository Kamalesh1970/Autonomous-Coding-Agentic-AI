# Autonomous Coding Agentic AI (Phase 11: Advanced Code Retrieval / RAG)

Minimal autonomous software engineering agent built with **Python** and **LangGraph**.

Reference Architecture: Inspired by [Open SWE](https://github.com/langchain-ai/open-swe) as a conceptual benchmark, built from the ground up as a minimal agentic foundation.

---

## 🎯 What is Phase 11?

Phase 11 equips the agent with **Advanced Code Retrieval / RAG** capabilities (`retrieve_hybrid_context`), integrating repository-aware code chunking, vector embeddings, hybrid ranking, local index lifecycle caching, and retrieval benchmarking.

The agentic retrieval workflow is:

$$\text{USER GOAL / SUBTASK} \longrightarrow \text{CODE CHUNKING} \longrightarrow \text{EMBEDDINGS} \longrightarrow \text{HYBRID RANKING (Lexical + Semantic + Metadata)} \longrightarrow \text{CONTEXT BUDGET BOUNDING} \longrightarrow \text{AGENT REASONING}$$

---

## 🔍 Hybrid Retrieval Architecture (`app/retrieval.py`)

1. **Repository-Aware Code Chunking**:
   - Parses Python files into AST function/class definitions and structured line/block sections with metadata (`file_path`, `start_line`, `end_line`, `content`, `language`, `symbol_name`, `symbol_type`).
   - Automatically excludes secrets (`.env`, `*.pem`, `*.key`, `id_rsa`, `credentials*`, `secrets*`) and binary files.

2. **Embeddings & In-Memory Vector Index**:
   - `BaseEmbeddingModel` interface supporting deterministic local `MockEmbeddingModel` (for 100% offline, zero-API-key testing) and optional `OpenAIEmbeddingModel` (`text-embedding-3-small`).
   - `HybridCodeIndex`: In-memory index with automatic `mtime` file modification tracking to reuse cached index when files are unchanged and refresh automatically when code changes.

3. **Hybrid Ranking Formula**:
   $$\text{Final Score} = (\text{Lexical Score} \times 0.4) + (\text{Semantic Score} \times 0.5) + (\text{Metadata Score} \times 0.1)$$
   - **Lexical Score**: BM25 / term frequency match across query terms and code content.
   - **Semantic Score**: Cosine similarity between query embedding vector and chunk vector.
   - **Metadata Score**: Boosts matches on symbol names (function/class) and file paths.

4. **Context Budget Bounding**:
   - Bounded by `top_k`, `max_context_chars`, and `max_chunk_chars` to return concise, high-value repository context without blowing up the LLM context window.

5. **Retrieval Evaluation & Benchmarks**:
   - `RetrievalEvaluator` measures Precision@K, Recall@K, and MRR (Mean Reciprocal Rank) to compare **Lexical**, **Semantic**, and **Hybrid** retrieval modes across benchmark tasks.

---

## 🚦 Human-in-the-Loop Approval & Delivery Model (Phase 10)

- **Actions Requiring Explicit Approval**: `git commit`, `git push`, `create_pull_request`.
- **Read-Only / Safe Local Actions**: `git_status`, `git_diff`, `git_current_branch`, `git_create_branch`, `retrieve_hybrid_context`.

---

## 🛡️ Security & Sandbox Controls (`app/sandbox.py`)

- ❌ Workspace boundary enforced (file traversal outside sandbox root blocked).
- ❌ Secrets (`.env`, `*.pem`, `*.key`, `id_rsa`, `credentials*`, `secrets*`) excluded from chunking and retrieval.
- ❌ Destructive Git commands (`git reset --hard`, `git clean -fd`, `git push --force`) prohibited.
- ❌ Credentials redacted from output logs.

---

## 🛠️ Agent Toolset

1. `retrieve_hybrid_context(query, top_k=3)`: Retrieves ranked hybrid semantic + lexical + metadata code context chunks.
2. `list_files(directory=".")`: Recursively lists relative file paths in workspace.
3. `read_file(file_path)`: Reads text file content inside sandbox with byte limits and binary file detection.
4. `search_code(query, directory=".")`: Plain-text search returning `file:line: snippet` matches.
5. `git_status()`: Inspects repository Git branch, modified files, and untracked files.
6. `git_diff()`: Inspects current unstaged and staged Git differences for edit feedback.
7. `retrieve_relevant_context(query, directory=".")`: Phase 4 context retrieval snippet builder.
8. `write_file(file_path, content)`: Safely creates or overwrites repository files inside sandbox.
9. `replace_in_file(file_path, old_text, new_text)`: Targeted unique text replacement inside sandbox.
10. `run_tests(target_directory=".", timeout_seconds=30)`: Controlled pytest execution in sandbox minimal environment.
11. `verify_goal(status, summary, evidence)`: Evaluates whether original user goal is satisfied based on repository evidence.
12. `request_human_approval(action, reason, risk)`: Requests human approval for commit/push/PR delivery actions.
13. `git_commit(message, files)`: Performs approved Git commit.
14. `git_push(remote, branch)`: Performs approved Git push.
15. `create_pull_request(title, body, head_branch, base_branch)`: Creates pull request representation.

---

## 📦 Installation

```bash
# Navigate to project directory
cd autonomous-coding-agent

# Create and activate virtual environment (Python 3.10+)
python3 -m venv .venv
source .venv/bin/activate

# Install dependencies
pip install -r requirements.txt
```

---

## 🧪 Running Tests

The test suite is 100% deterministic and runs offline using mock model responses and temporary sandbox/Git fixtures (`pytest`):

```bash
pytest -v
```

---

## 🚀 Running the Agent

Set up your `.env` file with your OpenAI key:
```env
OPENAI_API_KEY=your_actual_api_key
OPENAI_MODEL_NAME=gpt-4o-mini
```

Run a new task:
```bash
python3 -m app.agent "Fix the multiply function so that the project's tests pass." .
```
