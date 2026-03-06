"""PDF parser for NOAA technical memorandums.

Uses Docling if available, otherwise falls back to a simple text extractor.
"""

from __future__ import annotations

import logging

logger = logging.getLogger(__name__)


class PDFParser:
    """Parse scientific PDFs into section-level chunks."""

    def __init__(self):
        self._converter = None

    def _get_converter(self):
        if self._converter is None:
            try:
                from docling.document_converter import DocumentConverter
                self._converter = DocumentConverter()
            except ImportError:
                logger.warning("docling not installed; PDF parsing unavailable. Install with: pip install docling")
                return None
        return self._converter

    def parse(self, pdf_path: str) -> list[dict]:
        converter = self._get_converter()
        if converter is None:
            return [{"text": f"[PDF file: {pdf_path} — install docling to parse]",
                     "metadata": {"file_path": pdf_path, "type": "pdf"}}]

        result = converter.convert(pdf_path)
        markdown = result.document.export_to_markdown()

        # Split on headers
        chunks = []
        current_section = ""
        current_text = ""

        for line in markdown.split("\n"):
            if line.startswith("#"):
                if current_text.strip():
                    chunks.append({
                        "text": current_text.strip(),
                        "metadata": {
                            "file_path": pdf_path,
                            "section": current_section,
                            "type": "pdf",
                        },
                    })
                current_section = line.lstrip("# ").strip()
                current_text = line + "\n"
            else:
                current_text += line + "\n"

        if current_text.strip():
            chunks.append({
                "text": current_text.strip(),
                "metadata": {
                    "file_path": pdf_path,
                    "section": current_section,
                    "type": "pdf",
                },
            })

        return chunks
