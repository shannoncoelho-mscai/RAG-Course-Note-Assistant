"""PDF parser — extracts text + metadata from textbook PDFs.

Reads a PDF with PyMuPDF (fitz), returns a list of page dicts:
  [{page: int, text: str, has_tables: bool}]

`parse_pdf` accepts either a filesystem path (the offline CLI path) or
raw PDF bytes (the live POST /upload path — Cloud Run has no filesystem
to write temp uploads to, so bytes are parsed in memory).
"""
from __future__ import annotations

import fitz  # PyMuPDF


def parse_pdf(path_or_bytes: str | bytes) -> list[dict]:
    """Parse a PDF (path or in-memory bytes) into per-page text records."""
    if isinstance(path_or_bytes, (bytes, bytearray)):
        doc = fitz.open(stream=path_or_bytes, filetype="pdf")
    else:
        doc = fitz.open(path_or_bytes)

    pages: list[dict] = []
    for page in doc:
        text = page.get_text("text").strip()
        if not text:
            continue  # likely a scanned page with no text layer; log + skip
        has_tables = False
        try:
            has_tables = len(page.find_tables().tables) > 0
        except Exception:
            has_tables = False
        pages.append({
            "page": page.number + 1,  # 1-based for citations
            "text": text,
            "has_tables": has_tables,
        })
    return pages


def parse_pdf_to_chunks(path: str) -> list[dict]:
    """Convenience: parse + chunk a single PDF from a path."""
    from ingest.chunk import chunk_pages
    return chunk_pages(parse_pdf(path), source_name=os.path.basename(path))


    