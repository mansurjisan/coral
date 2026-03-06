"""Tests for the RAG indexer.

Since LanceDB and Ollama embeddings are needed for real indexing,
these tests mock the embedding calls and verify chunking + storage logic.
"""

import tempfile
from pathlib import Path
from unittest.mock import patch


from coral.rag.indexer import (
    ALL_EXTENSIONS,
    CoralIndexer,
    _C_EXTS,
    _ECFLOW_EXTS,
    _FORTRAN_EXTS,
    _NAMELIST_EXTS,
    _TEXT_EXTS,
)

FIXTURES = Path(__file__).parent / "fixtures"

# Fake 768-dim embedding
FAKE_EMBEDDING = [0.1] * 768


def _fake_embed(model, input):
    """Mock ollama.embed to return fake embeddings."""
    if isinstance(input, list):
        return {"embeddings": [FAKE_EMBEDDING] * len(input)}
    return {"embeddings": [FAKE_EMBEDDING]}


class TestChunkerSelection:
    def setup_method(self):
        with patch("coral.rag.indexer.lancedb"):
            self.indexer = CoralIndexer(db_path="/tmp/test_coral_db")

    def test_fortran_extensions(self):
        for ext in [".f90", ".f", ".f77", ".ftn"]:
            chunker = self.indexer._get_chunker(f"test{ext}")
            assert chunker is self.indexer._fortran

    def test_c_extensions(self):
        for ext in [".c", ".h", ".cpp"]:
            chunker = self.indexer._get_chunker(f"test{ext}")
            assert chunker is self.indexer._c

    def test_namelist_extensions(self):
        for ext in [".nml", ".namelist"]:
            chunker = self.indexer._get_chunker(f"test{ext}")
            assert chunker is self.indexer._namelist

    def test_ecflow_extensions(self):
        for ext in [".def", ".ecf"]:
            chunker = self.indexer._get_chunker(f"test{ext}")
            assert chunker is self.indexer._ecflow

    def test_pdf_returns_none(self):
        assert self.indexer._get_chunker("doc.pdf") is None

    def test_text_falls_back_to_markdown(self):
        for ext in [".md", ".txt", ".py", ".yaml"]:
            chunker = self.indexer._get_chunker(f"test{ext}")
            assert chunker is self.indexer._markdown


class TestIndexFile:
    @patch("coral.rag.indexer.ollama")
    @patch("coral.rag.indexer.lancedb")
    def test_index_fortran_file(self, mock_lancedb, mock_ollama):
        mock_ollama.embed.side_effect = _fake_embed
        mock_db = mock_lancedb.connect.return_value
        mock_db.table_names.return_value = []
        indexer = CoralIndexer(db_path="/tmp/test_db")
        count = indexer.index_file(str(FIXTURES / "sample_fortran.f90"))

        assert count > 0
        mock_db.create_table.assert_called_once()

    @patch("coral.rag.indexer.ollama")
    @patch("coral.rag.indexer.lancedb")
    def test_index_namelist_file(self, mock_lancedb, mock_ollama):
        mock_ollama.embed.side_effect = _fake_embed
        mock_db = mock_lancedb.connect.return_value
        mock_db.table_names.return_value = []

        indexer = CoralIndexer(db_path="/tmp/test_db")
        count = indexer.index_file(str(FIXTURES / "sample_namelist.nml"))

        assert count > 0

    @patch("coral.rag.indexer.ollama")
    @patch("coral.rag.indexer.lancedb")
    def test_index_empty_file_returns_zero(self, mock_lancedb, mock_ollama):
        with tempfile.NamedTemporaryFile(mode="w", suffix=".txt", delete=False) as f:
            f.write("")
            f.flush()

            indexer = CoralIndexer(db_path="/tmp/test_db")
            count = indexer.index_file(f.name)

        assert count == 0

    @patch("coral.rag.indexer.ollama")
    @patch("coral.rag.indexer.lancedb")
    def test_appends_to_existing_table(self, mock_lancedb, mock_ollama):
        mock_ollama.embed.side_effect = _fake_embed
        mock_db = mock_lancedb.connect.return_value
        mock_db.table_names.return_value = ["coral_docs"]
        mock_table = mock_db.open_table.return_value

        indexer = CoralIndexer(db_path="/tmp/test_db")
        indexer.index_file(str(FIXTURES / "sample_fortran.f90"))

        mock_table.add.assert_called_once()


class TestIndexDirectory:
    @patch("coral.rag.indexer.ollama")
    @patch("coral.rag.indexer.lancedb")
    def test_indexes_fixture_directory(self, mock_lancedb, mock_ollama):
        mock_ollama.embed.side_effect = _fake_embed
        mock_db = mock_lancedb.connect.return_value
        mock_db.table_names.return_value = []

        indexer = CoralIndexer(db_path="/tmp/test_db")
        total = indexer.index_directory(str(FIXTURES), extensions=[".f90", ".nml"])

        assert total > 0

    @patch("coral.rag.indexer.ollama")
    @patch("coral.rag.indexer.lancedb")
    def test_skips_non_matching_extensions(self, mock_lancedb, mock_ollama):
        mock_ollama.embed.side_effect = _fake_embed
        mock_db = mock_lancedb.connect.return_value
        mock_db.table_names.return_value = []

        indexer = CoralIndexer(db_path="/tmp/test_db")
        total = indexer.index_directory(str(FIXTURES), extensions=[".xyz"])

        assert total == 0


class TestAllExtensions:
    def test_all_extensions_complete(self):
        """ALL_EXTENSIONS should include all defined extension sets."""
        all_set = set(ALL_EXTENSIONS)
        for ext_set in [_FORTRAN_EXTS, _C_EXTS, _NAMELIST_EXTS, _ECFLOW_EXTS, _TEXT_EXTS]:
            for ext in ext_set:
                assert ext in all_set, f"{ext} missing from ALL_EXTENSIONS"
        assert ".pdf" in all_set
