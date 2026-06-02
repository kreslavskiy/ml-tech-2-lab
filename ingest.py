"""Populate the vector index from the Lab 1 documents (Task 1 data load).

Run this AFTER the server is up:  python ingest.py

It reads the Lab 1 report, README, and training notebook, then POSTs each one
to the running index API's /add_document endpoint — exercising the real Task 1
endpoint end to end.
"""

from __future__ import annotations

import json
import os

import requests

API = os.environ.get("API_URL", "http://localhost:8000")
LAB1 = os.path.join("..", "ml-tech-1-lab")


def read_text(path: str) -> str:
    with open(path, encoding="utf-8") as f:
        return f.read()


def read_notebook(path: str) -> str:
    nb = json.loads(read_text(path))
    parts = []
    for cell in nb["cells"]:
        source = cell["source"]
        if isinstance(source, list):
            source = "".join(source)
        if source.strip():
            parts.append(source)
    return "\n\n".join(parts)


def documents() -> list[tuple[str, str]]:
    return [
        ("report.md", read_text(os.path.join(LAB1, "report", "report.md"))),
        ("README.md", read_text(os.path.join(LAB1, "README.md"))),
        (
            "train_resnet_cifar100.ipynb",
            read_notebook(
                os.path.join(LAB1, "notebooks", "train_resnet_cifar100.ipynb")
            ),
        ),
    ]


def main() -> None:
    for source, text in documents():
        resp = requests.post(
            f"{API}/index/add_document", json={"text": text, "source": source}
        )
        resp.raise_for_status()
        print(f"{source}: {resp.json()['chunks_added']} chunks")


if __name__ == "__main__":
    main()
