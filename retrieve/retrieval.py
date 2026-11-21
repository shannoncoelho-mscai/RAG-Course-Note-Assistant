"""Hybrid retrieval — FAISS dense + BM25 keyword, merged with RRF.

Pipeline:
  query → BGE-small embedding with "query: " prefix (CPU, 384-dim)
        → FAISS IndexFlatIP top-20  +  BM25 top-20
        → Reciprocal Rank Fusion (RRF k=60) → top-N
        → relevance threshold filter (RELEVANCE_THRESHOLD)

The FAISS index and chunk metadata are loaded from GCS at startup by
`load_index()` and cached as module globals. When the index is absent,
`retrieve()` returns [] which drives api/main.py's "Not in notes" path
without ever calling the LLM.
"""

import os

# --- module-global cache (loaded once per process) ------------------------
_INDEX = None          # FAISS index object
_CHUNKS: list[dict] = []   # chunk metadata, parallel to index rows
_BM25 = None           # BM25Index (lazy-built from chunks)
_EMBEDDER = None       # SentenceTransformer singleton
_MODEL_ID = "BAAI/bge-small-en-v1.5"


def _embedder():
    """Lazily load the BGE-small model (CPU-friendly, 33M params)."""
    global _EMBEDDER
    if _EMBEDDER is None:
        from sentence_transformers import SentenceTransformer
        _EMBEDDER = SentenceTransformer(_MODEL_ID)
    return _EMBEDDER


def _bm25_tokens(text: str) -> list[str]:
    """Tokenise for BM25 (lowercase, strip punctuation, ignore short stopwords)."""
    import re
    tokens = re.findall(r"[a-z0-9]+", text.lower())
    return [t for t in tokens if len(t) > 2]


def load_index(index_path: str | None = None, meta_path: str | None = None) -> int:
    """Load FAISS index + chunk metadata from local files or GCS.

    Returns the number of chunks loaded (0 if no index is available).
    In the MVP this is a no-op returning 0; Phase 2 plugs in the GCS
    download (gsutil cp gs://<bucket>/index/faiss.idx ./index/).
    """
    global _INDEX, _CHUNKS, _BM25

    index_path = index_path or os.getenv("FAISS_INDEX_PATH", "index/faiss.idx")
    meta_path = meta_path or os.getenv("CHUNKS_META_PATH", "index/chunks.json")

    # On Cloud Run the filesystem is ephemeral: fetch the index from GCS when
    # the local files are missing. Bucket configured via GCS_INDEX_BUCKET
    # (default rag-index-514120); objects expected at the bucket root.
    if not (os.path.exists(index_path) and os.path.exists(meta_path)):
        bucket_name = os.getenv("GCS_INDEX_BUCKET", "rag-index-514120")
        try:
            from google.cloud import storage

            client = storage.Client()
            bucket = client.bucket(bucket_name)
            os.makedirs(os.path.dirname(index_path) or ".", exist_ok=True)
            os.makedirs(os.path.dirname(meta_path) or ".", exist_ok=True)
            bucket.blob(os.path.basename(index_path)).download_to_filename(index_path)
            bucket.blob(os.path.basename(meta_path)).download_to_filename(meta_path)
        except Exception:
            pass  # fall through to the normal loader, which reports 0 chunks

    import json

    import faiss  # local import: only needed when an index exists

    try:
        _INDEX = faiss.read_index(index_path)
        with open(meta_path, encoding="utf-8") as f:
            _CHUNKS = json.load(f)
    except Exception:
        # No index file, corrupt index, or missing metadata -> MVP "no index"
        _BM25 = None
        _INDEX = None
        _CHUNKS = []
        return 0

    # Rebuild BM25 over the chunk texts
    from rank_bm25 import BM25Okapi
    _BM25 = BM25Okapi([_bm25_tokens(c.get("text", "")) for c in _CHUNKS])
    return len(_CHUNKS)


def retrieve(
    query: str,
    top_k: int = 3,
    # Calibrated via eval sweep (REPORT §19): 0.70 = refusal accuracy 1.0
    # + hit@3 1.0 on the gold set. Mirrors .env / infra/main.tf — keep in sync.
    threshold: float = 0.70,
    dense_k: int = 20,
    bm25_k: int = 20,
    rrf_k: int = 60,
) -> list[dict]:
    """Return up to `top_k` chunks above `threshold`, best-first.

    Each returned dict: {chunk_id, page, section, score, text}.
    Returns [] when the index is empty or nothing clears the threshold —
    api/main.py converts that to "Not in notes" without an LLM call.
    """
    if _INDEX is None or not _CHUNKS:
        return []

    embedder = _embedder()

    # --- dense: embed query, FAISS search ---
    # BGE-small training format: queries prefixed "query: ", passages "passage: ".
    q_vec = embedder.encode(["query: " + query], convert_to_numpy=True, normalize_embeddings=True)
    q_vec = q_vec.astype("float32")
    dense_scores, dense_ids = _INDEX.search(q_vec, min(dense_k, _INDEX.ntotal))
    dense_ids = dense_ids[0].tolist()
    dense_scores = dense_scores[0].tolist()

    # --- keyword: BM25 search ---
    bm25_ids = []
    if _BM25 is not None:
        tokens = _bm25_tokens(query)
        bm25_scores = _BM25.get_scores(tokens)
        bm25_ids = sorted(range(len(bm25_scores)), key=lambda i: bm25_scores[i], reverse=True)[:bm25_k]
        # Drop zero-score hits so BM25-only matches have dense scores
        bm25_ids = [idx for idx in bm25_ids if bm25_scores[idx] > 0]

    # --- RRF merge: sum 1/(k + rank) across both lists ---
    rrf: dict[int, float] = {}
    for rank, idx in enumerate(dense_ids):
        if idx >= 0:
            rrf[idx] = rrf.get(idx, 0.0) + 1.0 / (rrf_k + rank)
    for rank, idx in enumerate(bm25_ids):
        rrf[idx] = rrf.get(idx, 0.0) + 1.0 / (rrf_k + rank)

    # Map idx -> dense similarity score for the threshold filter
    dense_score_by_idx = {idx: float(sc) for idx, sc in zip(dense_ids, dense_scores) if idx >= 0}

    results: list[dict] = []
    for idx, rrf_score in sorted(rrf.items(), key=lambda kv: kv[1], reverse=True):
        chunk = _CHUNKS[idx]
        sim = dense_score_by_idx.get(idx, 0.0)
        # Strict threshold: chunks with sim < threshold (including 0.0)
        # are excluded so the LLM never sees irrelevant noise.
        if sim >= threshold:
            results.append({
                "chunk_id": chunk.get("chunk_id", f"chunk_{idx:03d}"),
                "page": chunk.get("page"),
                "section": chunk.get("section"),
                "score": round(sim, 4),
                "text": chunk.get("text", ""),
            })
        if len(results) >= top_k:
            break
    return results


# --- startup hook -----------------------------------------------------------
# On import, attempt to load the index. If absent (MVP state), retrieve()
# returns [] and the API answers "Not in notes" — no crash, no LLM billing.
load_index()


# --- live upload support ------------------------------------------------------
# Appends newly ingested documents to the in-memory index and persists the
# enlarged index (local + GCS bucket). Used by api/main.py POST /upload so
# new documents are searchable immediately and survive cold starts.

def indexed_sources() -> set[str]:
    """Set of source names already in the index (duplicate-upload check)."""
    return {c.get("source", "") for c in _CHUNKS if c.get("source")}


def add_documents(new_chunks: list[dict], new_embeddings) -> int:
    """Append an ingested document to the in-memory index + metadata.

    new_embeddings: numpy float32 array (n, 384) from ingest.embed.embed_chunks().
    Returns the new total chunk count. Call persist_index() afterwards to
    write the enlarged index to disk / GCS bucket.

    NOTE: appends vectors via FAISS .add() — never re-embeds existing chunks,
    so an upload costs only the NEW document's embedding time.
    """
    global _INDEX, _BM25  # _CHUNKS is mutated in place (extend) — no rebinding
    import numpy as np

    if not new_chunks:
        return len(_CHUNKS)

    vecs = np.asarray(new_embeddings, dtype="float32")

    if _INDEX is None:
        # First document in an empty index (fresh deployment, no GCS files yet)
        import faiss
        _INDEX = faiss.IndexFlatIP(vecs.shape[1])

    _INDEX.add(vecs)
    _CHUNKS.extend(new_chunks)

    from rank_bm25 import BM25Okapi
    _BM25 = BM25Okapi([_bm25_tokens(c.get("text", "")) for c in _CHUNKS])
    return len(_CHUNKS)


def persist_index(index_path: str | None = None, meta_path: str | None = None) -> bool:
    """Write faiss.idx + chunks.json locally and upload to the GCS bucket.

    Bucket comes from GCS_INDEX_BUCKET (default rag-index-514120) — objects at
    the bucket root, exactly the names load_index() downloads at cold start.
    Returns True if the bucket upload succeeded, False if only local files
    were written (local dev mode without GCP credentials).
    """
    import json

    import faiss

    index_path = index_path or os.getenv("FAISS_INDEX_PATH", "index/faiss.idx")
    meta_path = meta_path or os.getenv("CHUNKS_META_PATH", "index/chunks.json")

    os.makedirs(os.path.dirname(index_path) or ".", exist_ok=True)
    faiss.write_index(_INDEX, index_path)
    with open(meta_path, "w", encoding="utf-8") as f:
        json.dump(_CHUNKS, f, indent=2, ensure_ascii=False)

    bucket_name = os.getenv("GCS_INDEX_BUCKET", "rag-index-514120")
    try:
        from google.cloud import storage

        client = storage.Client()
        bucket = client.bucket(bucket_name)
        bucket.blob(os.path.basename(index_path)).upload_from_filename(index_path)
        bucket.blob(os.path.basename(meta_path)).upload_from_filename(meta_path)
        return True
    except Exception:
        return False  # local-only mode (no GCP credentials) — files persist on disk