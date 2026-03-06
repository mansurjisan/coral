"""Tests for the RAG retriever.

Mocks LanceDB and Ollama to test search logic without external services.
"""

from unittest.mock import MagicMock, patch


from coral.rag.retriever import CoralRetriever


FAKE_EMBEDDING = [0.1] * 768


class TestRRFMerge:
    """Test Reciprocal Rank Fusion merge logic (no mocks needed)."""

    def setup_method(self):
        with patch("coral.rag.retriever.lancedb"):
            self.retriever = CoralRetriever(db_path="/tmp/test_db")

    def test_vector_only(self):
        vector = [
            {"file_path": "a.f90", "start_line": 1, "text": "sub a"},
            {"file_path": "b.f90", "start_line": 10, "text": "sub b"},
        ]
        results = self.retriever._rrf_merge(vector, [], top_k=5)
        assert len(results) == 2
        assert results[0]["file_path"] == "a.f90"

    def test_fts_only(self):
        fts = [
            {"file_path": "c.f90", "start_line": 5, "text": "sub c"},
        ]
        results = self.retriever._rrf_merge([], fts, top_k=5)
        assert len(results) == 1

    def test_both_boost_overlapping(self):
        """Items appearing in both lists should rank higher."""
        shared = {"file_path": "shared.f90", "start_line": 1, "text": "shared"}
        vector_only = {"file_path": "vec.f90", "start_line": 1, "text": "vec only"}

        vector = [shared, vector_only]
        fts = [shared]

        results = self.retriever._rrf_merge(vector, fts, top_k=5)
        assert results[0]["file_path"] == "shared.f90"

    def test_top_k_limits_results(self):
        items = [{"file_path": f"f{i}.f90", "start_line": i, "text": f"item {i}"} for i in range(10)]
        results = self.retriever._rrf_merge(items, [], top_k=3)
        assert len(results) == 3

    def test_empty_inputs(self):
        results = self.retriever._rrf_merge([], [], top_k=5)
        assert results == []


class TestSearch:
    @patch("coral.rag.retriever.ollama")
    @patch("coral.rag.retriever.lancedb")
    def test_returns_empty_when_no_table(self, mock_lancedb, mock_ollama):
        mock_db = mock_lancedb.connect.return_value
        mock_db.table_names.return_value = []

        retriever = CoralRetriever(db_path="/tmp/test_db")
        results = retriever.search("test query")
        assert results == []

    @patch("coral.rag.retriever.ollama")
    @patch("coral.rag.retriever.lancedb")
    def test_calls_vector_search(self, mock_lancedb, mock_ollama):
        mock_ollama.embed.return_value = {"embeddings": [FAKE_EMBEDDING]}

        mock_db = mock_lancedb.connect.return_value
        mock_db.table_names.return_value = ["coral_docs"]
        mock_table = mock_db.open_table.return_value
        mock_table.search.return_value.limit.return_value.to_list.return_value = [
            {"file_path": "test.f90", "start_line": 1, "text": "subroutine test"}
        ]

        retriever = CoralRetriever(db_path="/tmp/test_db")
        results = retriever.search("test subroutine", top_k=5)

        assert len(results) >= 1
        assert results[0]["file_path"] == "test.f90"

    @patch("coral.rag.retriever.ollama")
    @patch("coral.rag.retriever.lancedb")
    def test_fts_failure_graceful(self, mock_lancedb, mock_ollama):
        """FTS search failure should not crash — vector results still returned."""
        mock_ollama.embed.return_value = {"embeddings": [FAKE_EMBEDDING]}

        mock_db = mock_lancedb.connect.return_value
        mock_db.table_names.return_value = ["coral_docs"]
        mock_table = mock_db.open_table.return_value

        # Vector search works
        vector_chain = MagicMock()
        vector_chain.limit.return_value.to_list.return_value = [
            {"file_path": "a.f90", "start_line": 1, "text": "vec result"}
        ]

        # FTS search raises
        def search_side_effect(query, **kwargs):
            if kwargs.get("query_type") == "fts":
                raise Exception("FTS index not built")
            return vector_chain

        mock_table.search.side_effect = search_side_effect

        retriever = CoralRetriever(db_path="/tmp/test_db")
        results = retriever.search("test")

        assert len(results) == 1
        assert results[0]["file_path"] == "a.f90"
