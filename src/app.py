"""
Phase 3 — Streamlit chat UI.

Run from the project root:

    streamlit run src/app.py

Notes:
- Imports the compiled graph from rag_graph.py (the CLI loop there is under
  __main__, so importing it doesn't start it).
- Chat history is loaded FROM THE CHECKPOINTER (graph_app.get_state), not from
  Streamlit session state — so conversations survive app restarts, and the
  sidebar thread switcher is really switching LangGraph thread_ids.
"""

import streamlit as st

from rag_graph import app as graph_app

st.set_page_config(page_title="Agentic RAG")
st.title(" Agentic RAG Chatbot")

# -----------------------
# SIDEBAR — thread management
# -----------------------
if "threads" not in st.session_state:
    st.session_state.threads = ["chat-1"]

with st.sidebar:
    st.header("Conversations")

    new_thread = st.text_input("New conversation id", placeholder="e.g. chat-2")
    if st.button("➕ Create") and new_thread.strip():
        tid = new_thread.strip()
        if tid not in st.session_state.threads:
            st.session_state.threads.append(tid)

    current_thread = st.selectbox("Active conversation", st.session_state.threads)
    st.caption("Each conversation is a LangGraph thread_id — "
               "memory persists in checkpoints.db.")

config = {"configurable": {"thread_id": current_thread}}

# -----------------------
# RENDER HISTORY (from the checkpointer, not session state)
# -----------------------
snapshot = graph_app.get_state(config)
messages = snapshot.values.get("messages", []) if snapshot.values else []

for m in messages:
    role = "user" if m.type == "human" else "assistant"
    with st.chat_message(role):
        st.markdown(m.content)

# -----------------------
# CHAT INPUT
# -----------------------
if question := st.chat_input("Ask about your documents, do math, or just chat…"):
    with st.chat_message("user"):
        st.markdown(question)

    with st.chat_message("assistant"):
        with st.spinner("Thinking…"):
            result = graph_app.invoke({
                "question": question,
                "search_query": question,
                "route": "",
                "documents": [],
                "docs_relevant": False,
                "retries": 0,
                "tool_result": "",
                "final_answer": "",
            }, config=config)

        st.markdown(result["final_answer"])

        # Route badge — small, but great for demos and debugging
        badge = {"retrieve": "📄 documents", "calculator": "🧮 calculator",
                 "direct_llm": "💬 direct"}.get(result["route"], result["route"])
        extra = (f" · retries: {result['retries']}"
                 if result["route"] == "retrieve" and result["retries"] else "")
        st.caption(f"route: {badge}{extra}")