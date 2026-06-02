"""Q&A assistant API (Task 2): answer a question from retrieved chunks."""

from __future__ import annotations

from fastapi import APIRouter
from pydantic import BaseModel

from app import llm, rag

router = APIRouter(prefix="/qa", tags=["qa"])


class QueryRequest(BaseModel):
    question: str
    k: int = 5


@router.post("/query")
def query(req: QueryRequest) -> dict:
    vector = rag.embed([req.question])[0]
    chunks = rag.query(vector, req.k)
    response = llm.answer(req.question, chunks)
    sources = sorted({c["source"] for c in chunks})
    return {"answer": response, "sources": sources}
