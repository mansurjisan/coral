"""Tests for Fortran chunker."""

from pathlib import Path

from coral.rag.chunkers.fortran_chunker import FortranChunker

FIXTURES = Path(__file__).parent / "fixtures"


class TestFortranChunker:
    def setup_method(self):
        self.chunker = FortranChunker()

    def test_extracts_subroutines(self):
        code = (FIXTURES / "sample_fortran.f90").read_text()
        chunks = self.chunker.chunk(code, "sample_fortran.f90")

        names = [c["metadata"].get("name", "") for c in chunks]
        assert "schism_init" in names
        assert "timestep" in names

    def test_extracts_function(self):
        code = (FIXTURES / "sample_fortran.f90").read_text()
        chunks = self.chunker.chunk(code, "sample_fortran.f90")

        names = [c["metadata"].get("name", "") for c in chunks]
        assert "compute_cfl" in names

    def test_extracts_module(self):
        code = (FIXTURES / "sample_fortran.f90").read_text()
        chunks = self.chunker.chunk(code, "sample_fortran.f90")

        types = [c["metadata"].get("node_type", "") for c in chunks]
        assert "module" in types

    def test_metadata_has_language(self):
        code = (FIXTURES / "sample_fortran.f90").read_text()
        chunks = self.chunker.chunk(code, "sample_fortran.f90")

        for c in chunks:
            assert c["metadata"]["language"] == "fortran"

    def test_inline_subroutine(self):
        code = """\
subroutine test_sub(x, y)
  implicit none
  real, intent(in) :: x
  real, intent(out) :: y
  y = x * 2.0
end subroutine test_sub
"""
        chunks = self.chunker.chunk(code, "test.f90")
        assert len(chunks) >= 1
        assert "test_sub" in chunks[0]["text"]
        assert chunks[0]["metadata"]["language"] == "fortran"

    def test_fallback_on_empty(self):
        code = "! Just a comment\n! No program units\n"
        chunks = self.chunker.chunk(code, "comments.f90")
        assert len(chunks) >= 1  # Should produce fallback chunks

    def test_start_end_lines(self):
        code = (FIXTURES / "sample_fortran.f90").read_text()
        chunks = self.chunker.chunk(code, "sample_fortran.f90")

        for c in chunks:
            meta = c["metadata"]
            if "start_line" in meta and "end_line" in meta:
                assert meta["end_line"] >= meta["start_line"]
