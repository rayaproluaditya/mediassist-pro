#!/usr/bin/env python3
"""
scripts/seed_vectordb.py
========================
One-shot script to load all JSON guideline files into ChromaDB.
Run this AFTER `docker-compose up` to populate the knowledge base, or let
the backend auto-seed on first start (if the collection is empty).

Usage:
    python scripts/seed_vectordb.py [--host chromadb] [--port 8000]
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-8s | %(message)s",
)
log = logging.getLogger(__name__)


def seed(host: str, port: int, guidelines_dir: str) -> None:
    # ── Embeddings ────────────────────────────────────────────────────────────
    from langchain_community.embeddings import HuggingFaceEmbeddings
    from langchain_chroma import Chroma
    from langchain_core.documents import Document
    import chromadb

    log.info("Loading embedding model …")
    embeddings = HuggingFaceEmbeddings(
        model_name="all-MiniLM-L6-v2",
        model_kwargs={"device": "cpu"},
        encode_kwargs={"normalize_embeddings": True},
    )

    # ── ChromaDB connection ───────────────────────────────────────────────────
    log.info("Connecting to ChromaDB at %s:%d …", host, port)
    client = chromadb.HttpClient(host=host, port=port)
    client.heartbeat()   # raises if unreachable

    vectorstore = Chroma(
        client=client,
        collection_name="medical_guidelines",
        embedding_function=embeddings,
    )

    # ── Load guideline files ──────────────────────────────────────────────────
    base = Path(guidelines_dir)
    if not base.exists():
        log.error("Guidelines directory not found: %s", base)
        sys.exit(1)

    docs: list[Document] = []
    for jf in sorted(base.glob("**/*.json")):
        try:
            data = json.loads(jf.read_text())
            entries = data if isinstance(data, list) else [data]
            for e in entries:
                if isinstance(e, dict) and "content" in e:
                    docs.append(Document(
                        page_content=e["content"],
                        metadata={
                            "source":   e.get("source", jf.stem),
                            "category": e.get("category", "general"),
                            "title":    e.get("title", ""),
                        },
                    ))
        except Exception as exc:
            log.warning("Skipping %s: %s", jf, exc)

    if not docs:
        log.warning("No documents found — nothing to index.")
        return

    log.info("Indexing %d guideline chunks …", len(docs))
    vectorstore.add_documents(docs)
    log.info("✅ Done — %d documents indexed into 'medical_guidelines'.", len(docs))


def main() -> None:
    parser = argparse.ArgumentParser(description="Seed ChromaDB with medical guidelines.")
    parser.add_argument("--host",           default="chromadb",       help="ChromaDB host")
    parser.add_argument("--port",    type=int, default=8000,           help="ChromaDB port")
    parser.add_argument("--guidelines",     default="data/guidelines", help="Path to JSON guidelines dir")
    args = parser.parse_args()

    seed(args.host, args.port, args.guidelines)


if __name__ == "__main__":
    main()
