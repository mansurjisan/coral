"""Tests for namelist chunker."""

from pathlib import Path

from coral.rag.chunkers.namelist_chunker import NamelistChunker

FIXTURES = Path(__file__).parent / "fixtures"


class TestNamelistChunker:
    def setup_method(self):
        self.chunker = NamelistChunker()

    def test_parses_groups(self):
        content = (FIXTURES / "sample_namelist.nml").read_text()
        chunks = self.chunker.chunk(content, "param.nml")

        assert len(chunks) == 2
        groups = [c["metadata"]["namelist_group"] for c in chunks]
        assert "core" in groups
        assert "opt" in groups

    def test_group_content(self):
        content = (FIXTURES / "sample_namelist.nml").read_text()
        chunks = self.chunker.chunk(content, "param.nml")

        core = [c for c in chunks if c["metadata"]["namelist_group"] == "core"][0]
        assert "dt" in core["text"]
        assert "120.0" in core["text"]

    def test_metadata_type(self):
        content = (FIXTURES / "sample_namelist.nml").read_text()
        chunks = self.chunker.chunk(content, "param.nml")

        for c in chunks:
            assert c["metadata"]["type"] == "namelist"

    def test_inline_namelist(self):
        content = """\
&grid
  nx = 100
  ny = 200
/
"""
        chunks = self.chunker.chunk(content, "grid.nml")
        assert len(chunks) == 1
        assert chunks[0]["metadata"]["namelist_group"] == "grid"
        assert "nx" in chunks[0]["text"]

    def test_fallback_on_bad_input(self):
        content = "This is not a valid namelist"
        chunks = self.chunker.chunk(content, "bad.nml")
        assert len(chunks) == 1
        assert chunks[0]["metadata"]["type"] == "namelist"
