"""Hybrid BM25 + semantic search retriever over LanceDB."""

from __future__ import annotations

import logging
from pathlib import Path

import lancedb
import ollama

logger = logging.getLogger(__name__)

EMBED_MODEL = "nomic-embed-text"
TABLE_NAME = "coral_docs"


class CoralRetriever:
    """Hybrid BM25 + semantic vector search over indexed documents."""

    def __init__(self, db_path: str = "~/.coral/vectordb"):
        self.db_path = str(Path(db_path).expanduser())
        self.db = lancedb.connect(self.db_path)

    def search(self, query: str, top_k: int = 5) -> list[dict]:
        """Hybrid search: combine semantic vector search with full-text BM25."""
        if TABLE_NAME not in self.db.table_names():
            return []

        table = self.db.open_table(TABLE_NAME)

        # Generate query embedding
        resp = ollama.embed(model=EMBED_MODEL, input=query)
        query_vector = resp["embeddings"][0]

        # Vector search
        vector_results = table.search(query_vector).limit(top_k).to_list()

        # Full-text search (LanceDB FTS)
        fts_results = []
        try:
            fts_results = table.search(query, query_type="fts").limit(top_k).to_list()
        except Exception:
            # FTS index may not exist yet — that's fine, vector search alone works
            pass

        return self._rrf_merge(vector_results, fts_results, top_k)

    def _rrf_merge(self, vector_results: list, fts_results: list, top_k: int, k: int = 60) -> list[dict]:
        """Reciprocal Rank Fusion to combine vector and BM25 results."""
        scores: dict[str, float] = {}
        data: dict[str, dict] = {}

        for rank, r in enumerate(vector_results):
            key = r.get("file_path", "") + ":" + str(r.get("start_line", ""))
            scores[key] = scores.get(key, 0) + 1.0 / (k + rank + 1)
            if key not in data:
                data[key] = r

        for rank, r in enumerate(fts_results):
            key = r.get("file_path", "") + ":" + str(r.get("start_line", ""))
            scores[key] = scores.get(key, 0) + 1.0 / (k + rank + 1)
            if key not in data:
                data[key] = r

        sorted_keys = sorted(scores, key=lambda k: scores[k], reverse=True)[:top_k]
        return [data[key] for key in sorted_keys if key in data]
