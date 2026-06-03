"""Claude-backed answer generation for the Q&A assistant (Task 2).

Builds a prompt from a fixed task definition (the system prompt) + the
retrieved context + the user's question, then calls the Anthropic Messages API.
"""

from __future__ import annotations

import os

import anthropic

# Default model is claude-sonnet-4-6; override with the LLM_MODEL env var.
MODEL = os.environ.get("LLM_MODEL", "claude-sonnet-4-6")

SYSTEM_PROMPT = (
    "You are a question-answering assistant. Answer the user's question using "
    "ONLY the provided context. If the answer is not contained in the context, "
    "say you don't know — do not make anything up. Keep answers concise."
)

_client: anthropic.Anthropic | None = None


def _get_client() -> anthropic.Anthropic:
    global _client
    if _client is None:
        _client = anthropic.Anthropic()
    return _client


def answer(question: str, chunks: list[dict]) -> str:
    context = "\n\n---\n\n".join(c["text"] for c in chunks) or "(no context found)"

    response = _get_client().messages.create(
        model=MODEL,
        max_tokens=1024,
        system=SYSTEM_PROMPT,
        messages=[
            {
                "role": "user",
                "content": [
                    {
                        "type": "text",
                        "text": f"Context:\n{context}",
                        "cache_control": {"type": "ephemeral"},
                    },
                    {"type": "text", "text": f"Question: {question}"},
                ],
            }
        ],
    )
    return "".join(block.text for block in response.content if block.type == "text")
