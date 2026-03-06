# Adding Documents to the RAG Knowledge Base

CORAL's RAG pipeline indexes documents into a LanceDB vector database, enabling the LLM to search and retrieve relevant context.

## Supported file types

| Type | Extensions | Chunking strategy |
|------|-----------|-------------------|
| Fortran | `.f90`, `.f`, `.f77`, `.ftn` | Subroutine/function/module extraction |
| C/C++ | `.c`, `.h`, `.cpp` | Function-level extraction |
| Namelists | `.nml`, `.namelist` | Per-namelist-group |
| ecFlow | `.def`, `.ecf` | Family/task block extraction |
| Markdown/text | `.md`, `.rst`, `.txt` | Header-based splitting |
| PDF | `.pdf` | Docling parsing + header splitting |
| Config/code | `.yaml`, `.py`, `.sh`, `.json` | Fixed-size with overlap |

## Indexing commands

```bash
# Single file
coral index /path/to/param.nml

# Entire directory (recursive)
coral index /path/to/schism/src

# Custom vector DB location
coral index /path/to/docs --db-path /scratch5/purged/$USER/coral_vectordb
```

## What to index

For best results, index:

- **Model source code**: SCHISM `src/`, ADCIRC `src/`, UFS-Coastal
- **Configuration files**: `param.nml`, `fort.15`, ecFlow suite definitions
- **Documentation**: README files, user guides, NOAA technical memorandums (PDFs)
- **Run scripts**: Slurm job scripts, shell workflows

## How it works

1. Files are split into chunks using type-specific chunkers
2. Each chunk is embedded using `nomic-embed-text` via Ollama
3. Chunks + embeddings are stored in LanceDB
4. At query time, hybrid search (vector + BM25) retrieves the most relevant chunks
5. The LLM uses these chunks as context to answer questions

## Updating the index

Re-running `coral index` on the same files will add duplicate chunks. To re-index cleanly, delete the vector DB first:

```bash
rm -rf ~/.coral/vectordb
coral index /path/to/docs
```
