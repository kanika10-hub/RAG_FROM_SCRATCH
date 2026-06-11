"""
Phase 2 — Agentic RAG graph.

This is your conditional_workflow.py merged with langchain_rag.py, upgraded
with the two patterns that make RAG "agentic":

  1. Document grading  — after retrieval, an LLM judges if the chunks are
                          actually relevant (don't blindly trust top-k).
  2. Corrective loop   — if chunks are irrelevant, REWRITE the query and
                          retry retrieval (max 2 attempts), instead of
                          generating a hallucinated answer.

Graph shape:

    START → router ─┬→ retrieve → grade ─┬→ generate → END
                    │       ↑            └→ rewrite ──┘ (loops back to retrieve)
                    ├→ calculator ───────→ generate
                    └→ direct_llm ───────→ generate

Prerequisite: run `python ingest.py data/raw` once first.
"""

import re
from typing import TypedDict, List

from dotenv import load_dotenv
from langchain_google_genai import ChatGoogleGenerativeAI, GoogleGenerativeAIEmbeddings
from langchain_chroma import Chroma
from langgraph.graph import StateGraph, START, END

load_dotenv()

# -----------------------
# MODELS & RETRIEVER
# -----------------------
llm = ChatGoogleGenerativeAI(model="gemini-2.5-flash", temperature=0)

embeddings = GoogleGenerativeAIEmbeddings(model="models/gemini-embedding-001")
vectorstore = Chroma(
    collection_name="portfolio_docs",
    embedding_function=embeddings,
    persist_directory="chroma_db",
)
retriever = vectorstore.as_retriever(search_kwargs={"k": 4})


# -----------------------
# STATE
# -----------------------
class State(TypedDict):
    question: str          # original user question (never overwritten)
    search_query: str      # what we actually send to the retriever (gets rewritten)
    route: str
    documents: List[str]   # retrieved chunk texts
    docs_relevant: bool
    retries: int           # corrective-loop counter
    tool_result: str
    final_answer: str


# -----------------------
# NODES
# -----------------------
def router_node(state: State):
    """Same idea as your conditional_workflow router, with 'retrieve' as a route."""
    q = state["question"].lower()

    if re.search(r"\d+\s*[+\-*/]\s*\d+", q):
        return {"route": "calculator"}

    # TODO: replace keyword routing with an LLM router (structured output:
    # 'vectorstore' | 'calculator' | 'direct') — better generalization,
    # and a great thing to A/B test in Phase 4 evals.
    doc_keywords = ["document", "pdf", "according to", "report", "paper"]
    if any(k in q for k in doc_keywords):
        return {"route": "retrieve", "search_query": state["question"]}

    return {"route": "direct_llm"}


def retrieve_node(state: State):
    docs = retriever.invoke(state["search_query"])
    return {"documents": [d.page_content for d in docs]}


def grade_documents_node(state: State):
    """Agentic step #1: judge relevance instead of trusting similarity scores."""
    docs_text = "\n\n---\n\n".join(state["documents"])[:6000]
    prompt = (
        "You are a grader. Question:\n{q}\n\nRetrieved context:\n{ctx}\n\n"
        "Does the context contain information that helps answer the question? "
        "Reply with exactly one word: yes or no."
    ).format(q=state["question"], ctx=docs_text)

    verdict = llm.invoke(prompt).content.strip().lower()
    return {"docs_relevant": verdict.startswith("yes")}


def rewrite_query_node(state: State):
    """Agentic step #2: corrective RAG — rephrase and try again."""
    prompt = (
        "The search query below failed to retrieve relevant documents. "
        "Rewrite it to be more specific and keyword-rich for semantic search. "
        "Return ONLY the rewritten query.\n\nOriginal: {q}"
    ).format(q=state["search_query"])

    new_query = llm.invoke(prompt).content.strip()
    print(f"  ↻ rewriting query → {new_query}")
    return {"search_query": new_query, "retries": state["retries"] + 1}


def calculator_node(state: State):
    try:
        match = re.search(r"\d+(?:\s*[+\-*/]\s*\d+)+", state["question"])
        if not match:
            return {"tool_result": "Invalid expression"}
        result = eval(match.group())  # demo only — swap for ast-based eval later
        return {"tool_result": f"Calculator result: {result}"}
    except Exception as e:
        return {"tool_result": f"Invalid expression ({e})"}


def direct_llm_node(state: State):
    return {"tool_result": llm.invoke(state["question"]).content}


def generate_node(state: State):
    """Single aggregator. RAG answers MUST be grounded in retrieved context."""
    if state["route"] == "retrieve":
        if state["documents"] and state["docs_relevant"]:
            context = "\n\n---\n\n".join(state["documents"])
            prompt = (
                "Answer using ONLY the context below. If the context doesn't "
                "contain the answer, say you don't know — do not invent facts. "
                "Mention which part of the context supports your answer.\n\n"
                "Context:\n{ctx}\n\nQuestion: {q}"
            ).format(ctx=context, q=state["question"])
        else:
            # retries exhausted, still nothing relevant → honest fallback
            prompt = (
                "We could not find relevant information in the documents for: "
                "'{q}'. Briefly tell the user this, and answer from general "
                "knowledge if you can, clearly labeling it as such."
            ).format(q=state["question"])
        return {"final_answer": llm.invoke(prompt).content}

    # calculator / direct_llm path — same as your old final_node
    prompt = (
        "Question: {q}\nTool output: {t}\n\n"
        "Write a clean, direct final answer using the tool output."
    ).format(q=state["question"], t=state["tool_result"])
    return {"final_answer": llm.invoke(prompt).content}


# -----------------------
# CONDITIONAL EDGES
# -----------------------
def route_selector(state: State) -> str:
    return state["route"]


def grade_selector(state: State) -> str:
    """Relevant docs → generate. Irrelevant → rewrite & retry, max 2 attempts."""
    if state["docs_relevant"] or state["retries"] >= 2:
        return "generate"
    return "rewrite"


# -----------------------
# BUILD GRAPH
# -----------------------
graph = StateGraph(State)

graph.add_node("router", router_node)
graph.add_node("retrieve", retrieve_node)
graph.add_node("grade", grade_documents_node)
graph.add_node("rewrite", rewrite_query_node)
graph.add_node("calculator", calculator_node)
graph.add_node("direct_llm", direct_llm_node)
graph.add_node("generate", generate_node)

graph.add_edge(START, "router")
graph.add_conditional_edges("router", route_selector, {
    "retrieve": "retrieve",
    "calculator": "calculator",
    "direct_llm": "direct_llm",
})

graph.add_edge("retrieve", "grade")
graph.add_conditional_edges("grade", grade_selector, {
    "generate": "generate",
    "rewrite": "rewrite",
})
graph.add_edge("rewrite", "retrieve")          # ← the corrective loop

graph.add_edge("calculator", "generate")
graph.add_edge("direct_llm", "generate")
graph.add_edge("generate", END)

app = graph.compile()


# -----------------------
# CHAT LOOP
# -----------------------
if __name__ == "__main__":
    print("\n🚀 Agentic RAG (Phase 2) — type 'exit' to quit\n")
    while True:
        q = input("Ask: ").strip()
        if q.lower() == "exit":
            break
        result = app.invoke({
            "question": q,
            "search_query": q,
            "route": "",
            "documents": [],
            "docs_relevant": False,
            "retries": 0,
            "tool_result": "",
            "final_answer": "",
        })
        print(f"\n[route: {result['route']}"
              + (f", retries: {result['retries']}" if result["route"] == "retrieve" else "")
              + "]")
        print("👉", result["final_answer"])
        print("-" * 50)
