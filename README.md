# Lab 2 — LLM + Retrieval-Augmented Generation (RAG)

A two-part RAG pipeline served behind a single FastAPI app:

- **Task 1 — Index:** a vector store ([Chroma](https://www.trychroma.com/), embedded) plus an index-management API that chunks, vectorizes, and stores documents, and retrieves the closest chunks for a query vector.
- **Task 2 — Q&A assistant:** a `/qa/query` endpoint that embeds a question, retrieves relevant chunks, and asks **Claude** to answer using only that context.

Embeddings use a small local model (`all-MiniLM-L6-v2`, 384-dim) — no API key or cost. Only the Q&A endpoint calls Claude.

## Layout

```
.
├── app/
│   ├── main.py          FastAPI app: /health + index & qa routers
│   ├── rag.py           chunking, embeddings, Chroma store (Task 1 core)
│   ├── llm.py           Claude client + prompt template (Task 2 core)
│   ├── index_api.py     router: /index/add_document, /index/query
│   └── qa_api.py        router: /qa/query
├── ingest.py            populate the store from the Lab 1 documents
├── requirements.txt
├── .env.example
└── report/report.md     write-up / deliverable
```

## Endpoints

| Method | Path | Task | Body | Returns |
|---|---|---|---|---|
| GET  | `/health`             | —      | —                              | `{"status":"ok"}` |
| POST | `/index/add_document` | 1      | `{"text": "...", "source": "name"}` | `{"source","chunks_added"}` |
| POST | `/index/query`        | 1      | `{"vector": [floats], "k": 5}` | `{"results":[{text,source,distance}]}` |
| POST | `/qa/query`           | 2      | `{"question": "...", "k": 5}`  | `{"answer","sources":[...]}` |

## 1. Install

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

> First install is large (`sentence-transformers` pulls in PyTorch) and the embedding model (~80 MB) downloads on first use.

## 2. Configure the Claude key

```bash
cp .env.example .env
# edit .env and set ANTHROPIC_API_KEY=...
```

## 3. Run the API

```bash
uvicorn app.main:app --reload
```

Open the interactive docs at <http://localhost:8000/docs> (all four endpoints, grouped by router).

## 4. Populate the index (Task 1)

With the server running, in another terminal:

```bash
python ingest.py
```

This loads the Lab 1 report, README, and training notebook into the vector store and prints the chunk count per document. (Re-running adds the documents again — delete the `chroma_db/` folder first for a clean reload.)

## 5. Ask a question (Task 2)

```bash
curl -X POST http://localhost:8000/qa/query \
  -H 'Content-Type: application/json' \
  -d '{"question":"What model architecture was trained in Lab 1 and on what dataset?","k":5}'
```

Example response:

```json
{
  "answer": "A ResNet-18 implemented from scratch in PyTorch, trained on the CIFAR-100 dataset.",
  "sources": ["README.md", "report.md", "train_resnet_cifar100.ipynb"]
}
```

Ask something outside the documents (e.g. *"What is the capital of France?"*) and the assistant answers that it doesn't know — confirming it only uses the retrieved context.
