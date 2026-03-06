"""Tests for markdown chunker."""

from coral.rag.chunkers.markdown_chunker import MarkdownChunker


class TestMarkdownChunker:
    def setup_method(self):
        self.chunker = MarkdownChunker()

    def test_splits_on_headers(self):
        content = """\
# Introduction
This is the intro.

# Methods
This is the methods section.

# Results
Results go here.
"""
        chunks = self.chunker.chunk(content, "doc.md")
        assert len(chunks) == 3
        sections = [c["metadata"]["section"] for c in chunks]
        assert "Introduction" in sections
        assert "Methods" in sections
        assert "Results" in sections

    def test_large_section_split(self):
        content = "# Big Section\n" + ("x" * 100 + "\n") * 50
        chunks = self.chunker.chunk(content, "big.md", max_chunk=500)
        assert len(chunks) > 1

    def test_no_headers(self):
        content = "Just plain text\nwith multiple lines\n"
        chunks = self.chunker.chunk(content, "plain.txt")
        assert len(chunks) >= 1
        assert "Just plain text" in chunks[0]["text"]
