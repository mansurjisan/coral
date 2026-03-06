"""Tests for C chunker."""

from coral.rag.chunkers.c_chunker import CChunker


class TestCChunker:
    def setup_method(self):
        self.chunker = CChunker()

    def test_extracts_function(self):
        code = """\
int main(int argc, char *argv[]) {
    printf("hello\\n");
    return 0;
}
"""
        chunks = self.chunker.chunk(code, "main.c")
        assert len(chunks) >= 1
        assert "main" in chunks[0]["metadata"].get("name", "")
        assert chunks[0]["metadata"]["language"] == "c"

    def test_multiple_functions(self):
        code = """\
void init_model() {
    setup();
}

double compute_flux(double u, double v) {
    return u * v;
}
"""
        chunks = self.chunker.chunk(code, "model.c")
        names = [c["metadata"].get("name", "") for c in chunks]
        assert "init_model" in names
        assert "compute_flux" in names

    def test_fallback_for_header(self):
        code = """\
#ifndef MODEL_H
#define MODEL_H

typedef struct {
    double x, y, z;
} Point;

#endif
"""
        chunks = self.chunker.chunk(code, "model.h")
        assert len(chunks) >= 1  # Should produce fallback chunks
