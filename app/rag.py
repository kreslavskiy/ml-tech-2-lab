"""Core RAG logic: chunking, embedding, and the Chroma vector store.

This module is the heart of Task 1 — it turns documents into vectors and
stores/retrieves them. Embeddings come from a small local model
(all-MiniLM-L6-v2, 384-dim), so there is no API key or cost involved.
"""

from __future__ import annotations

import uuid

import chromadb
from sentence_transformers import SentenceTransformer

_model = SentenceTransformer("all-MiniLM-L6-v2")

_client = chromadb.PersistentClient(path="chroma_db")
_collection = _client.get_or_create_collection("documents")


def chunk(text: str, size: int = 800, overlap: int = 100) -> list[str]:
    text = text.strip()
    if len(text) <= size:
        return [text] if text else []

    chunks: list[str] = []
    start = 0
    while start < len(text):
        end = start + size
        window = text[start:end]
        if end < len(text):
            split = window.rfind("\n\n")
            if split == -1:
                split = window.rfind(". ")
            if split == -1:
                split = window.rfind(" ")
            if split > size // 2:
                end = start + split + 1
                window = text[start:end]
        cleaned = window.strip()
        if cleaned:
            chunks.append(cleaned)
        start = end - overlap
    return chunks


def embed(texts: list[str]) -> list[list[float]]:
    return _model.encode(texts).tolist()


def add_document(text: str, source: str) -> int:
    chunks = chunk(text)
    if not chunks:
        return 0
    embeddings = embed(chunks)
    ids = [str(uuid.uuid4()) for _ in chunks]
    metadatas = [{"source": source} for _ in chunks]
    _collection.add(
        ids=ids, embeddings=embeddings, documents=chunks, metadatas=metadatas
    )
    return len(chunks)


def query(vector: list[float], k: int = 5) -> list[dict]:
    result = _collection.query(query_embeddings=[vector], n_results=k)
    documents = result["documents"][0]
    metadatas = result["metadatas"][0]
    distances = result["distances"][0]
    return [
        {"text": doc, "source": meta.get("source"), "distance": dist}
        for doc, meta, dist in zip(documents, metadatas, distances)
    ]
