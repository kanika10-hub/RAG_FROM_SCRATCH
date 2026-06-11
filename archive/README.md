# AI Engineering Archive

## Development Environment

This project was built and experimented on using:

* Python
* UV (Python package manager)
* Rust-based tooling through UV
* Virtual Environments
* Git & GitHub

Why UV?

* Faster package installation
* Faster dependency resolution
* Better environment management
* Built on Rust, making it significantly faster than traditional pip workflows

Typical workflow:

```bash
uv venv
uv pip install -r requirements.txt
```

Throughout this project, UV was used to manage dependencies, create isolated environments, and maintain reproducible setups while experimenting with LangChain, LangGraph, RAG, Agents, and Memory systems.

---

# Learning Flow

# AI Engineering Archive

This archive contains the experiments, prototypes, and learning projects I built while exploring modern AI Engineering concepts.

The goal was to understand how AI applications are built from the ground up before moving to more advanced frameworks and workflows.

---

# Learning Flow

## Phase 1: Retrieval-Augmented Generation (RAG)

Started by learning the fundamentals of RAG.

Topics explored:

* Open Book RAG
* Semantic Search
* Embeddings
* PDF RAG
* FAISS Vector Database
* Retrieval Pipelines

### Basic RAG Flow

User Question
↓
Retriever
↓
Relevant Chunks
↓
Prompt Builder
↓
LLM
↓
Answer

Key Learnings:

* Documents are converted into chunks.
* Chunks are transformed into embeddings.
* Similar chunks are retrieved using vector similarity.
* Retrieved context is injected into prompts before sending to the LLM.

---

## Phase 2: LangChain Fundamentals

After understanding RAG fundamentals, I moved to LangChain.

Topics explored:

* Document Loaders
* Text Splitters
* Embeddings
* Vector Stores
* Retrievers
* Chains
* Agents

Key Learning:

LangChain provides reusable building blocks that simplify AI application development.

---

## Phase 3: LCEL (LangChain Expression Language)

Explored LCEL, LangChain's newer and cleaner way of connecting AI components.

Topics explored:

* Runnable Pipelines
* Component Chaining
* Modular Workflows
* Cleaner Composition Patterns

Key Learning:

LCEL makes AI pipelines easier to build, read, and maintain compared to traditional chains.

---

## Phase 4: Conversational Tools & Tool Calling

Built custom tools and explored tool-calling systems.

Custom Tools:

* Calculator
* Uppercase Converter
* Word Counter

### Tool Calling Flow

User Input
↓
Tool Decision
↓
Tool Selection
↓
Tool Execution
↓
Result

Key Learning:

LLMs can decide when to use external tools and incorporate their outputs into responses.

---

## Phase 5: Manual Tool Agent

Implemented a simple tool-calling agent manually without relying on frameworks.

Topics explored:

* Tool Selection via Prompting
* Output Parsing
* Tool Execution
* Response Generation

This helped me understand what happens internally before using agent frameworks.

---

## Phase 6: ReAct Agents

Built ReAct-style agents using LangChain.

### ReAct Loop

Question
↓
Reason
↓
Action
↓
Observation
↓
Reason
↓
Final Answer

Key Learning:

Agents can iteratively reason, use tools, observe results, and continue thinking before producing a final response.

---

## Phase 7: LangGraph

After understanding agents, I moved to LangGraph.

Topics explored:

* State
* Nodes
* Edges
* Graph Execution
* Workflow Design

### LangGraph Flow

Input
↓
Node
↓
Node
↓
Decision
↓
Next Node
↓
Output

Key Learning:

LangGraph allows building structured workflows instead of relying solely on agent loops.

---

## Phase 8: LangSmith

Integrated LangSmith for observability.

### LangChain Ecosystem Flow

LangChain
↓
Build Components

LangGraph
↓
Build Workflows

LangSmith
↓
Observe
Debug
Evaluate

Key Learning:

LangSmith is not a framework.

It is an observability and evaluation platform used to:

* Trace executions
* Debug workflows
* Monitor behavior
* Evaluate outputs

---

## Phase 9: Memory

### Manual Memory

Initially implemented memory manually by storing conversation history and passing it back to the model.

Advantages:

* Easy to understand
* Easy to debug

---

### LangGraph Memory

Later upgraded to LangGraph Memory using MemorySaver.

Topics explored:

* Checkpointing
* Thread-based Conversations
* Persistent State

Key Learning:

LangGraph handles memory management more elegantly than manually maintaining chat history.

---

## Phase 10: Workflows

Built workflow-based applications using LangGraph.

### Basic Workflow

Start
↓
Node A
↓
Node B
↓
End

---

### Conditional Workflow

Start
↓
Decision Node
↙        ↘
Path A   Path B
↓          ↓
Output

Key Learning:

Workflows allow deterministic execution paths, while agents are more dynamic and reasoning-driven.

---

# Challenges Faced

Throughout the learning process I encountered several real-world issues:

* Hugging Face model compatibility problems
* Unsupported provider errors
* LangSmith 403 Forbidden errors
* Gemini API authentication issues
* Gemini API quota limits
* Chat history formatting bugs
* Agent execution issues
* Memory integration problems

Debugging these issues provided a better understanding of how production AI systems behave.

---

# Current Status

Completed:

✅ RAG Fundamentals
✅ FAISS Vector Search
✅ LangChain Fundamentals
✅ LCEL
✅ Tool Calling
✅ Manual Agents
✅ ReAct Agents
✅ LangGraph Basics
✅ LangSmith Basics
✅ Manual Memory
✅ LangGraph Memory
✅ Basic Workflows
✅ Conditional Workflows

Currently Exploring:

* Agentic RAG
* LangGraph RAG
* RAG Evaluation
* Production RAG Systems
* Multi-Agent Systems

This archive documents the journey from basic RAG systems to workflow-driven AI applications and agent architectures.


## Technologies Explored

- Python
- UV
- LangChain
- LangGraph
- LangSmith
- Google Gemini
- Hugging Face
- FAISS
- Vector Embeddings
- Retrieval-Augmented Generation (RAG)
- ReAct Agents
- Tool Calling
- LCEL
