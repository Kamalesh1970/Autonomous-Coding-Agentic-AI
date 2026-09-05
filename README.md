# Autonomous Coding Agentic AI (Phase 1)

Minimal autonomous software engineering agent built with **Python** and **LangGraph**.

Reference Architecture: Inspired by [Open SWE](https://github.com/langchain-ai/open-swe) as a conceptual benchmark, built from the ground up as a minimal agentic foundation.

---

## 🎯 What is Phase 1?

Phase 1 establishes the minimal single-agent agentic loop foundation. Rather than executing a hardcoded script or static sequence of commands, the agent dynamically decides at runtime whether to call a tool, which tool to call, and when to conclude the task based on observations.

---

## 🤖 Why Agentic (Not a Chatbot)?

A standard chatbot operates in a simple single-turn request-response loop (`User Prompt → LLM Response`).

This agentic AI operates in an **iterative control loop**:
```
User Goal
   │
   ▼
[ Reason ] ◄────────────────────┐
   │                            │
[ Decide ]                      │ (Observation)
   │                            │
   ├──────────► [ Act ] ────────┴
   │         (Invoke Tool)
   ▼
[ Finish ]
(Final Answer)
```

The agent maintains explicit state, observes tool execution outputs, updates its internal context with those observations, and dynamically determines the next step.

---

## 🏗️ Architecture

- **State (`app/state.py`)**: `AgentState` contains `messages` (using `add_messages` reducer), `user_goal`, and `workspace_root`.
- **Tools (`app/tools.py`)**: Safe workspace tools (`list_files`, `read_file`) with path traversal protection. No arbitrary shell execution is permitted.
- **Agent (`app/agent.py`)**: A compiled LangGraph `StateGraph` linking a reasoning node (`reason`) to a tool execution node (`tools`) via a conditional edge (`route_after_reason`).

---

## 📦 Installation

```bash
# Clone or navigate to directory
cd autonomous-coding-agent

# Create and activate virtual environment (Python 3.10+)
python3 -m venv .venv
source .venv/bin/activate

# Install dependencies
pip install -e .
pip install pytest
```

---

## 🧪 Running Tests

The test suite is 100% deterministic and runs offline using mocked model responses (no paid API key required):

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

Run the agent on a user goal:
```bash
python3 -m app.agent "Understand the structure of this repository." .
```

---

## 🚧 Intentionally NOT Implemented (Reserved for Future Phases)

To keep Phase 1 lean and focused on core agentic loop correctness, the following components are intentionally omitted:

- ❌ Arbitrary shell execution / terminal execution
- ❌ FastAPI / Web UI
- ❌ PostgreSQL / Redis persistence
- ❌ Multi-agent collaboration
- ❌ Vector databases / RAG / Qdrant
- ❌ GitHub API integration
- ❌ Model routing / complex observability platforms
# Autonomous-Coding-Agentic-AI
