"""
MediAssist Pro — Medical Guidelines RAG Retriever
==================================================
Two modes:
  1. HTTP client  → connects to the ChromaDB container (production / Docker).
  2. Local client → persistent Chroma on disk (local dev / fallback).

Embeddings are computed with sentence-transformers (all-MiniLM-L6-v2).
"""

from __future__ import annotations

import json
import logging
import os
from pathlib import Path
from typing import List, Dict, Any

logger = logging.getLogger(__name__)


class MedicalGuidelinesRetriever:
    """
    RAG retriever that:
      1. Loads medical guidelines from JSON files on first run.
      2. Stores / retrieves chunks via ChromaDB.
      3. Formats retrieved context for injection into the LLM prompt.
    """

    def __init__(
        self,
        chroma_host: str | None = None,
        chroma_port: int | None = None,
        persist_dir: str | None = None,
        collection_name: str | None = None,
        embedding_model: str | None = None,
        guidelines_path: str | None = None,
        top_k: int | None = None,
    ) -> None:
        from app.config import settings

        self.chroma_host = chroma_host or settings.CHROMA_HOST
        self.chroma_port = chroma_port or settings.CHROMA_PORT
        self.persist_dir = persist_dir or settings.CHROMA_PERSIST_DIR
        self.collection_name = collection_name or settings.CHROMA_COLLECTION
        self.embedding_model_name = embedding_model or settings.EMBEDDING_MODEL
        self.guidelines_path = guidelines_path or settings.GUIDELINES_PATH
        self.top_k = top_k or settings.RAG_TOP_K

        self._embeddings = None
        self._vectorstore = None

        self._init_embeddings()
        self._init_vectorstore()
        self._seed_if_empty()

    # ── Initialisation ───────────────────────────────────────────────────────

    def _init_embeddings(self) -> None:
        """Load sentence-transformer embedding model."""
        try:
            from langchain_community.embeddings import HuggingFaceEmbeddings
            self._embeddings = HuggingFaceEmbeddings(
                model_name=self.embedding_model_name,
                model_kwargs={"device": "cpu"},
                encode_kwargs={"normalize_embeddings": True},
            )
            logger.info("Embedding model loaded: %s", self.embedding_model_name)
        except Exception as exc:
            logger.error("Failed to load embedding model: %s", exc)
            raise

    def _init_vectorstore(self) -> None:
        """
        Try HTTP client first (Docker / remote ChromaDB).
        Fall back to local persistent client.
        """
        # ── HTTP client (Docker) ─────────────────────────────────────────────
        try:
            import chromadb
            from langchain_chroma import Chroma

            http_client = chromadb.HttpClient(
                host=self.chroma_host,
                port=self.chroma_port,
            )
            # Ping to verify connectivity
            http_client.heartbeat()

            self._vectorstore = Chroma(
                client=http_client,
                collection_name=self.collection_name,
                embedding_function=self._embeddings,
            )
            logger.info(
                "Connected to ChromaDB HTTP at %s:%s",
                self.chroma_host,
                self.chroma_port,
            )
            return
        except Exception as exc:
            logger.warning("ChromaDB HTTP unavailable (%s) — using local storage.", exc)

        # ── Local persistent client (fallback) ───────────────────────────────
        try:
            from langchain_chroma import Chroma

            os.makedirs(self.persist_dir, exist_ok=True)
            self._vectorstore = Chroma(
                persist_directory=self.persist_dir,
                collection_name=self.collection_name,
                embedding_function=self._embeddings,
            )
            logger.info("Using local ChromaDB at %s", self.persist_dir)
        except Exception as exc:
            logger.error("Failed to initialise any ChromaDB backend: %s", exc)
            self._vectorstore = None

    def _seed_if_empty(self) -> None:
        """Load guideline JSON files if the collection is empty."""
        if self._vectorstore is None:
            return
        try:
            count = self._vectorstore._collection.count()
            if count == 0:
                logger.info("Vector store is empty — seeding from %s", self.guidelines_path)
                docs = self._load_guideline_files()
                if docs:
                    self.add_documents(docs)
            else:
                logger.info("Vector store already contains %d documents.", count)
        except Exception as exc:
            logger.warning("Could not check / seed vector store: %s", exc)

    # ── Data loading ─────────────────────────────────────────────────────────

    def _load_guideline_files(self) -> List[Dict[str, Any]]:
        """Parse all *.json files under self.guidelines_path."""
        path = Path(self.guidelines_path)
        if not path.exists():
            logger.warning("Guidelines path does not exist: %s", path)
            return []

        docs: List[Dict[str, Any]] = []
        for jf in path.glob("**/*.json"):
            try:
                with jf.open() as fh:
                    data = json.load(fh)
                entries = data if isinstance(data, list) else [data]
                for entry in entries:
                    if isinstance(entry, dict) and "content" in entry:
                        docs.append(entry)
            except Exception as exc:
                logger.warning("Could not load %s: %s", jf, exc)

        logger.info("Loaded %d guideline documents from disk.", len(docs))
        return docs

    # ── Indexing ─────────────────────────────────────────────────────────────

    def add_documents(self, guidelines: List[Dict[str, Any]]) -> None:
        """
        Add a list of guideline dicts (must have a 'content' key) to the
        vector store.
        """
        from langchain_core.documents import Document

        if self._vectorstore is None:
            logger.warning("Vector store not initialised — skipping add_documents.")
            return

        lc_docs = [
            Document(
                page_content=g["content"],
                metadata={
                    "source": g.get("source", "unknown"),
                    "category": g.get("category", "general"),
                    "title": g.get("title", ""),
                    "symptoms": ", ".join(g.get("symptoms", [])),
                },
            )
            for g in guidelines
            if g.get("content")
        ]

        if not lc_docs:
            return

        self._vectorstore.add_documents(lc_docs)
        logger.info("Indexed %d guideline chunks.", len(lc_docs))

    # ── Retrieval ────────────────────────────────────────────────────────────

    def retrieve(self, query: str, k: int | None = None) -> list:
        """Return top-k LangChain Document objects for *query*."""
        if self._vectorstore is None:
            return []
        k = k or self.top_k
        try:
            docs = self._vectorstore.similarity_search(query, k=k)
            logger.debug("Retrieved %d docs for query: %.60s", len(docs), query)
            return docs
        except Exception as exc:
            logger.error("Retrieval error: %s", exc)
            return []

    def get_context(self, query: str) -> str:
        """
        Retrieve relevant chunks and format them as a plain-text context
        block ready for injection into the LLM prompt.
        """
        docs = self.retrieve(query)
        if not docs:
            return ""

        parts = []
        for doc in docs:
            src = doc.metadata.get("source", "Medical Guidelines")
            parts.append(f"[{src}]\n{doc.page_content[:600]}")

        return "\n\n".join(parts)

    def get_sources(self, query: str) -> List[str]:
        """Return a list of source labels for the retrieved documents."""
        docs = self.retrieve(query)
        return [doc.metadata.get("source", "unknown") for doc in docs]
