"""Embed + index builder — BGE-small (CPU) → FAISS on GCS.

Reads chunks from parse_pdf + chunk_pages, embeds them with
BAAI/bge-small-en-v1.5 (384-dim, normalized for cosine search),
builds a FAISS IndexFlatIP index, and writes the artifacts to
index/faiss.idx (+ chunks.json) locally. Phase 2 uploads those to GCS
(gsutil cp index/faiss.idx gs://<bucket>/index/).

Also writes an eval-ready chunk id map so eval_hitrate.py can measure
exact hit@3 once expected_chunks are filled in.

Usage:
  uv run python ingest/embed.py /path/to/textbook.pdf [--out ./index]
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import faiss
import numpy as np
from sentence_transformers import SentenceTransformer

from ingest.parse_pdf import parse_pdf
from ingest.chunk import chunk_pages

MODEL_ID = "BAAI/bge-small-en-v1.5"
VECTOR_DIM = 384


def embed_chunks(chunks: list[dict], model: SentenceTransformer | None = None) -> np.ndarray:
    """Embed chunk texts -> float32 array (n, 384), L2-normalized.

    Documents are prefixed with "passage: " to match the BGE training
    format. Queries in retrieve/retrieval.py must use the matching
    "query: " prefix for the similarity scores to be comparable.
    """
    model = model or SentenceTransformer(MODEL_ID)
    texts = ["passage: " + c["text"] for c in chunks]
    embeddings = model.encode(
        texts,
        convert_to_numpy=True,
        normalize_embeddings=True,
        show_progress_bar=True,
        batch_size=32,
    )
    return np.asarray(embeddings, dtype="float32")


def build_index(embeddings: np.ndarray) -> faiss.IndexFlatIP:
    """FAISS inner-product index (cosine search on normalized vectors)."""
    index = faiss.IndexFlatIP(VECTOR_DIM)
    index.add(embeddings)
    return index


def main() -> None:
    parser = argparse.ArgumentParser(description="Embed textbook PDF -> FAISS index")
    parser.add_argument("pdf_path", help="Path to the AI textbook PDF")
    parser.add_argument("--out", default="index", help="Output directory (default: index/)")
    args = parser.parse_args()

    if not os.path.exists(args.pdf_path):
        print(f"PDF not found: {args.pdf_path}")
        sys.exit(1)

    print(f"Parsing {args.pdf_path} ...")
    pages = parse_pdf(args.pdf_path)
    print(f"  pages with text: {len(pages)}")

    chunks = chunk_pages(pages, source_name=Path(args.pdf_path).stem)
    print(f"  chunks created: {len(chunks)}")

    print(f"Embedding with {MODEL_ID} (CPU) ...")
    embeddings = embed_chunks(chunks)
    print(f"  embedding shape: {embeddings.shape}")

    index = build_index(embeddings)
    print(f"  FAISS index built: {index.ntotal} vectors, dim {index.d}")

    os.makedirs(args.out, exist_ok=True)
    faiss.write_index(index, str(Path(args.out) / "faiss.idx"))
    with open(Path(args.out) / "chunks.json", "w", encoding="utf-8") as f:
        json.dump(chunks, f, indent=2, ensure_ascii=False)
    print(f"Wrote {args.out}/faiss.idx + {args.out}/chunks.json")


if __name__ == "__main__":
    main()
