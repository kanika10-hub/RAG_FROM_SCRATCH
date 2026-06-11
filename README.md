# Agentic RAG Portfolio

An evolving Retrieval-Augmented Generation (RAG) system built with **LangGraph** and **Gemini**, progressing from a basic RAG pipeline to a production-style multi-agent architecture.

> **Current Status:** Phase 2 — Agentic RAG with Corrective Retrieval

---

## Overview

This project explores how modern RAG systems can move beyond simple similarity search.

Instead of blindly trusting retrieved chunks, the system evaluates whether the retrieved context actually answers the user's question. If retrieval quality is poor, it automatically rewrites the query and retries before generating a response.

This helps reduce hallucinations and improves answer reliability.

---

## Key Agentic Behavior

### Corrective RAG Workflow

1. Retrieve relevant document chunks from the vector store.
2. Use an LLM to grade retrieved chunks for relevance.
3. If the context is insufficient:

   * Rewrite the query.
   * Retry retrieval.
4. Generate an answer only when sufficient context is found.
5. If retries are exhausted, return an honest failure instead of hallucinating.

---

## Features

### Document Processing

* PDF ingestion pipeline
* Recursive chunking with overlap
* Metadata preservation for citations
* Persistent Chroma vector database

### Intelligent Routing

* Route mathematical queries to a calculator tool
* Route document-based questions to retrieval
* Route general knowledge questions directly to the LLM

### Retrieval Quality Control

* LLM-based document relevance grading
* Corrective retrieval loop with query rewriting
* Maximum retry limit to prevent infinite loops

### Grounded Generation

* Answers generated from retrieved context
* Source-aware responses
* Refuses to invent unsupported information

---

## Tech Stack

* LangGraph (orchestration)
* Gemini 2.5 Flash (LLM)
* Gemini Embeddings
* Chroma (persistent vector store)
* uv (package management)

---

## Setup

```bash
git clone <repo-url>
cd agentic-rag-portfolio

uv sync
# or
pip install -r requirements.txt
```

Create a `.env` file:

```env
GOOGLE_API_KEY=your_api_key
```

---

## Usage

### 1. Ingest Documents

Place PDFs inside:

```text
data/raw/
```

Run ingestion:

```bash
python src/ingest.py data/raw
```

This step:

* Loads PDFs
* Splits them into chunks
* Creates embeddings
* Stores vectors in Chroma

### 2. Start the Agent

```bash
python src/rag_graph.py
```

---

## Example Session

```text
Ask: 25 * 4 + 10

[route: calculator]

110
```

```text
Ask: According to the document, what is Retrieval-Augmented Generation?

[route: retrieve]
[retries: 0]

<grounded answer generated from retrieved context>
```

```text
Ask: Explain LCEL in simple terms

[route: direct_llm]

<general knowledge answer>
```

---

## Project Structure

```text
agentic-rag-portfolio/
│
├── data/
│   └── raw/
│
├── src/
│   ├── ingest.py
│   └── rag_graph.py
│
├── evals/
├── tests/
├── archive/
│
└── README.md
```

---

## Roadmap

### Phase 1 — RAG Foundation

* PDF ingestion pipeline
* Chunking strategy
* Persistent vector store

### Phase 2 — Agentic RAG

* Conditional routing
* Document grading
* Corrective retrieval loop

### Phase 3 — Chatbot + Memory

* Streamlit chat interface
* LangGraph checkpointer
* SQLite-backed conversation memory

### Phase 4 — Evaluation

* Golden Q&A dataset
* LangSmith tracing
* RAGAS metrics
* Faithfulness evaluation
* Context precision and recall

### Phase 5 — Production Hardening

* Semantic caching
* Input and output guardrails
* Retry and fallback mechanisms
* Structured logging

### Phase 6 — Multi-Agent Architecture

* Supervisor agent
* Researcher agent
* Analyst agent
* Critic agent

### Phase 7 — GraphRAG

* Neo4j knowledge graph
* Relationship-aware retrieval
* Hybrid graph and vector search

---

## Phase Log

### Phase 1 — `phase-1-foundation`

**Added**

* PDF ingestion pipeline
* Recursive chunking
* Persistent Chroma vector store

**Learned**

* Why ingestion and runtime should be separated
* How chunk size and overlap impact retrieval quality

### Phase 2 — `phase-2-agentic-rag`

**Added**

* Conditional routing
* Document relevance grading
* Corrective retrieval loop

**Learned**

* How LangGraph handles loops and control flow
* Why top-k similarity search alone is often insufficient
* How query rewriting improves retrieval quality

---

## Future Improvements

* Streamlit-based user interface
* Conversation memory with LangGraph checkpointers
* Evaluation pipeline using RAGAS
* Semantic caching
* Multi-agent workflows
* GraphRAG integration with Neo4j
* Production monitoring and observability

This repository is designed as both a working RAG system and a public learning journey. Each phase is tagged in Git and documents not only what was built, but also the reasoning and lessons learned along the way.
