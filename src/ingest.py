"""
Phase 1 — Ingestion pipeline.

Run ONCE (or whenever documents change), separately from the chat app:

    python ingest.py data/raw

This replaces the "load + embed every time the script runs" pattern from
langchain_rag.py. The vector store is persisted to disk so rag_graph.py
can just open it instantly.
"""

import sys
from pathlib import Path

from dotenv import load_dotenv
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_chroma import Chroma

load_dotenv()

PERSIST_DIR = "chroma_db"
COLLECTION = "portfolio_docs"

# If this 404s like gemini-1.5-flash did, list available embedding models:
#   from google import genai; [print(m.name) for m in genai.Client().models.list()]
EMBEDDINGS = GoogleGenerativeAIEmbeddings(model="models/gemini-embedding-001")


def load_documents(folder: str):
    """Load every PDF in the folder, keeping source metadata for citations later."""
    docs = []
    for pdf_path in Path(folder).glob("**/*.pdf"):
        loader = PyPDFLoader(str(pdf_path))
        loaded = loader.load()
        for d in loaded:
            d.metadata["source_file"] = pdf_path.name  # used for citations in Phase 2+
        docs.extend(loaded)
        print(f"  loaded {pdf_path.name}: {len(loaded)} pages")
    return docs


def chunk_documents(docs):
    """
    Recursive chunking with overlap. Tune these two numbers and re-run evals
    in Phase 4 — chunk size is one of the highest-leverage RAG knobs.
    """
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=150,
        separators=["\n\n", "\n", ". ", " ", ""],
    )
    chunks = splitter.split_documents(docs)
    print(f"  produced {len(chunks)} chunks")
    return chunks


def build_vectorstore(chunks):
    """Embed and persist. Chroma writes to disk automatically with persist_directory."""
    vs = Chroma.from_documents(
        documents=chunks,
        embedding=EMBEDDINGS,
        collection_name=COLLECTION,
        persist_directory=PERSIST_DIR,
    )
    print(f"  persisted to ./{PERSIST_DIR}")
    return vs


if __name__ == "__main__":
    folder = sys.argv[1] if len(sys.argv) > 1 else "data/raw"
    print(f"\n📥 Ingesting PDFs from {folder}\n")
    docs = load_documents(folder)
    if not docs:
        print("No PDFs found. Put pdfs (and others) in data/raw/ first.")
        sys.exit(1)
    chunks = chunk_documents(docs)
    build_vectorstore(chunks)
    print("\n✅ Done. Now run: python rag_graph.py\n")

    # TODO (Phase 1.5): build a BM25 index here too (rank_bm25) and save it,
    # so retrieval can be hybrid (keyword + vector) via EnsembleRetriever.
