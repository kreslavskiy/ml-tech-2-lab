"""Index management API (Task 1): add documents and query by vector."""

from __future__ import annotations

from fastapi import APIRouter
from pydantic import BaseModel

from app import rag

router = APIRouter(prefix="/index", tags=["index"])


class AddDocumentRequest(BaseModel):
    text: str
    source: str = "untitled"


class QueryRequest(BaseModel):
    vector: list[float]
    k: int = 5


@router.post("/add_document")
def add_document(req: AddDocumentRequest) -> dict:
    chunks_added = rag.add_document(req.text, req.source)
    return {"source": req.source, "chunks_added": chunks_added}


@router.post("/query")
def query(req: QueryRequest) -> dict:
    return {"results": rag.query(req.vector, req.k)}
