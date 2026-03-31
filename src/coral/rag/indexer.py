"""Document indexer: chunks files and stores embeddings in LanceDB."""

from __future__ import annotations

import logging
from pathlib import Path

import lancedb
import ollama
import pyarrow as pa

from coral.rag.chunkers.fortran_chunker import FortranChunker
from coral.rag.chunkers.c_chunker import CChunker
from coral.rag.chunkers.namelist_chunker import NamelistChunker
from coral.rag.chunkers.markdown_chunker import MarkdownChunker
from coral.rag.chunkers.ecflow_chunker import EcflowChunker
from coral.rag.parsers.pdf_parser import PDFParser

logger = logging.getLogger(__name__)

EMBED_MODEL = "nomic-embed-text"
TABLE_NAME = "coral_docs"

# Extension → chunker mapping
_FORTRAN_EXTS = {".f90", ".f", ".f77", ".ftn"}
_C_EXTS = {".c", ".h", ".cpp", ".cc", ".cxx", ".hpp"}
_NAMELIST_EXTS = {".nml", ".namelist"}
_ECFLOW_EXTS = {".def", ".ecf"}
_TEXT_EXTS = {".md", ".rst", ".txt", ".yaml", ".yml", ".py", ".sh", ".json", ".cfg", ".conf"}

ALL_EXTENSIONS = (
    list(_FORTRAN_EXTS) + list(_C_EXTS) + list(_NAMELIST_EXTS) + list(_ECFLOW_EXTS) + list(_TEXT_EXTS) + [".pdf"]
)

# LanceDB schema
_SCHEMA = pa.schema(
    [
        pa.field("text", pa.utf8()),
        pa.field("vector", pa.list_(pa.float32(), 768)),
        pa.field("file_path", pa.utf8()),
        pa.field("start_line", pa.int32()),
        pa.field("node_type", pa.utf8()),
        pa.field("name", pa.utf8()),
        pa.field("language", pa.utf8()),
        pa.field("section", pa.utf8()),
        pa.field("namelist_group", pa.utf8()),
        pa.field("type", pa.utf8()),
    ]
)


class CoralIndexer:
    """Index documents into LanceDB with appropriate chunking per file type."""

    def __init__(self, db_path: str = "~/.coral/vectordb"):
        self.db_path = str(Path(db_path).expanduser())
        self.db = lancedb.connect(self.db_path)
        self._fortran = FortranChunker()
        self._c = CChunker()
        self._namelist = NamelistChunker()
        self._markdown = MarkdownChunker()
        self._ecflow = EcflowChunker()
        self._pdf = PDFParser()

    def _embed(self, texts: list[str]) -> list[list[float]]:
        """Generate embeddings via Ollama in batches."""
        results = []
        batch_size = 32
        for i in range(0, len(texts), batch_size):
            batch = texts[i : i + batch_size]
            resp = ollama.embed(model=EMBED_MODEL, input=batch)
            results.extend(resp["embeddings"])
        return results

    def _get_chunker(self, file_path: str):
        ext = Path(file_path).suffix.lower()
        if ext in _FORTRAN_EXTS:
            return self._fortran
        elif ext in _C_EXTS:
            return self._c
        elif ext in _NAMELIST_EXTS:
            return self._namelist
        elif ext in _ECFLOW_EXTS:
            return self._ecflow
        elif ext == ".pdf":
            return None
        else:
            return self._markdown

    def index_file(self, file_path: str) -> int:
        """Index a single file. Returns number of chunks indexed."""
        file_path = str(Path(file_path).resolve())

        if file_path.endswith(".pdf"):
            chunks = self._pdf.parse(file_path)
        else:
            with open(file_path, errors="replace") as f:
                content = f.read()
            if not content.strip():
                return 0
            chunker = self._get_chunker(file_path)
            chunks = chunker.chunk(content, file_path)

        if not chunks:
            return 0

        texts = [c["text"] for c in chunks]
        embeddings = self._embed(texts)

        records = []
        for chunk, embedding in zip(chunks, embeddings):
            meta = chunk.get("metadata", {})
            records.append(
                {
                    "text": chunk["text"],
                    "vector": embedding,
                    "file_path": meta.get("file_path", file_path),
                    "start_line": meta.get("start_line", 0),
                    "node_type": meta.get("node_type", ""),
                    "name": meta.get("name", ""),
                    "language": meta.get("language", ""),
                    "section": meta.get("section", ""),
                    "namelist_group": meta.get("namelist_group", ""),
                    "type": meta.get("type", ""),
                }
            )

        if TABLE_NAME in self.db.table_names():
            table = self.db.open_table(TABLE_NAME)
            table.add(records)
        else:
            self.db.create_table(TABLE_NAME, records, schema=_SCHEMA)

        return len(records)

    def index_directory(self, dir_path: str, extensions: list[str] | None = None) -> int:
        """Recursively index all matching files in a directory."""
        if extensions is None:
            extensions = ALL_EXTENSIONS

        dir_path = Path(dir_path)
        total = 0
        seen = set()

        for ext in extensions:
            for fp in sorted(dir_path.rglob(f"*{ext}")):
                if fp in seen or not fp.is_file():
                    continue
                seen.add(fp)
                try:
                    count = self.index_file(str(fp))
                    total += count
                    if count > 0:
                        logger.info("Indexed %s: %d chunks", fp.name, count)
                        print(f"  Indexed {fp.name}: {count} chunks")
                except Exception as e:
                    logger.warning("Error indexing %s: %s", fp.name, e)
                    print(f"  Error indexing {fp.name}: {e}")

        return total
