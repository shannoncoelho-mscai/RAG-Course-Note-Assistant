"""Ingestion pipeline — PDF bytes → chunks → embeddings.

One reusable function used by BOTH:
  - the offline CLI (ingest/embed.py builds the full index)
  - the live POST /upload endpoint (appends a new document to the
    existing index)

Runs the same heading-aware chunking and BGE-small embedding (with the
`passage:` prefix) as the original index build, so newly uploaded
documents blend seamlessly into the existing vector space.
"""
from __future__ import annotations

from ingest.chunk import chunk_pages
from ingest.embed import embed_chunks
from ingest.parse_pdf import parse_pdf


def ingest_pdf_bytes(pdf_bytes: bytes, source_name: str, model=None) -> dict:
    """Parse + chunk + embed one in-memory PDF.

    `model` optionally receives the API's already-loaded SentenceTransformer
    (retrieve._embedder()) so /upload doesn't load a second copy.

    Returns {chunks, embeddings, pages} ready for retrieve.add_documents().
    """
    pages = parse_pdf(pdf_bytes)
    if not pages:
        raise ValueError(
            "No extractable text found — the PDF may be scanned images "
            "(OCR is not supported). Try a text-based PDF."
        )
    chunks = chunk_pages(pages, source_name=source_name)
    embeddings = embed_chunks(chunks, model=model)
    return {
        "chunks": chunks,
        "embeddings": embeddings,
        "pages": len(pages),
    }