"""FastAPI app hosting both the index (Task 1) and Q&A (Task 2) APIs."""

from dotenv import load_dotenv

load_dotenv()  # load ANTHROPIC_API_KEY / LLM_MODEL from a local .env file

from fastapi import FastAPI  # noqa: E402

from app import index_api, qa_api  # noqa: E402

app = FastAPI(
    title="Lab 2 — RAG Q&A",
    version="1.0",
    description="Vector index + retrieval-augmented Q&A over a document library.",
)

app.include_router(index_api.router)
app.include_router(qa_api.router)


@app.get("/health")
def health() -> dict:
    return {"status": "ok"}
