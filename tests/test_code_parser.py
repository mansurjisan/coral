"""Tests for the generic code parser dispatcher."""

from coral.rag.parsers.code_parser import parse_code_file


class TestParseCodeFile:
    def test_fortran_file_uses_fortran_chunker(self):
        code = "subroutine foo(x)\n  real :: x\nend subroutine foo\n"
        chunks = parse_code_file(code, "test.f90")
        assert len(chunks) >= 1
        assert "foo" in chunks[0]["text"]

    def test_c_file_uses_c_chunker(self):
        code = "void bar(int x) {\n  return;\n}\n"
        chunks = parse_code_file(code, "test.c")
        assert len(chunks) >= 1
        assert "bar" in chunks[0]["text"]

    def test_header_file_uses_c_chunker(self):
        code = "int baz(void);\n"
        chunks = parse_code_file(code, "utils.h")
        assert len(chunks) >= 1

    def test_markdown_fallback(self):
        code = "# Title\n\nSome text about stuff.\n"
        chunks = parse_code_file(code, "readme.md")
        assert len(chunks) >= 1

    def test_unknown_extension_uses_markdown(self):
        code = "random content\nmore lines\n"
        chunks = parse_code_file(code, "data.txt")
        assert len(chunks) >= 1

    def test_fortran_extensions(self):
        code = "program main\n  print *, 'hi'\nend program main\n"
        for ext in [".f90", ".f", ".f77", ".ftn"]:
            chunks = parse_code_file(code, f"test{ext}")
            assert len(chunks) >= 1

    def test_cpp_extension(self):
        code = "class Foo {};\n"
        chunks = parse_code_file(code, "test.cpp")
        assert len(chunks) >= 1
