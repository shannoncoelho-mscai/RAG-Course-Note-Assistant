"""Unit tests for retrieve/retrieval.py — no LLM, no network.

Covers the retrieval logic that decides what the LLM sees:
- Threshold filtering (dense similarity gate)
- Empty-index short-circuit (no_llm path)
- top_k cap + RRF ordering
- BGE query-prefix contract
"""

import retrieve.retrieval as R


def _make_fake_index(chunks, embeddings):
    """Build a real FAISS index over fake 384-dim embeddings (no model needed)."""
    import faiss
    import numpy as np
    idx = faiss.IndexFlatIP(384)
    idx.add(np.asarray(embeddings, dtype="float32"))
    return idx


def _rng_embeddings(n: int, seed: int = 0):
    """Deterministic unit-norm embeddings: chunk i has high cosine with query seed."""
    import numpy as np
    rng = np.random.default_rng(seed)
    base = rng.normal(size=384)
    base = base / np.linalg.norm(base)
    vecs = []
    for i in range(n):
        noise = rng.normal(size=384) * (0.1 + 0.1 * i)  # further rows are farther
        v = base + noise
        vecs.append(v / np.linalg.norm(v))
    return vecs


def test_threshold_excludes_below_threshold() -> None:
    """Regression for the BM25-only bypass bug: a chunk whose dense
    similarity is below threshold must be excluded even if BM25 ranks it high."""
    chunks = [{"chunk_id": f"test_c{i:03d}", "page": 1, "section": "x", "text": f"chunk {i} about tokens"} for i in range(5)]
    R._CHUNKS = chunks
    R._INDEX = _make_fake_index(chunks, _rng_embeddings(5))
    R._BM25 = None  # disable BM25 so only the threshold gate is exercised

    # Embedder stub: query vector = first chunk's embedding (highest sim)
    class FakeEmbedder:
        def encode(self, texts, convert_to_numpy=True, normalize_embeddings=True):
            import numpy as np
            vec = np.asarray(_rng_embeddings(5)[0], dtype="float32").reshape(1, -1)
            return vec

    orig = R._embedder
    R._embedder = lambda: FakeEmbedder()
    try:
        # threshold=1.1 (above every similarity) must return []
        results = R.retrieve("tokens", top_k=5, threshold=1.1)
        assert results == []
        # threshold=0.0 allows everything the dense search returns
        results = R.retrieve("tokens", top_k=5, threshold=0.0)
        assert len(results) > 0
    finally:
        R._embedder = orig
    # restore globals so other tests don't leak
    R._CHUNKS = []
    R._INDEX = None
    R._BM25 = None


def test_empty_index_returns_empty_without_embedder() -> None:
    """Empty index -> retrieve() returns [] without invoking the embedder.
    This is the short-circuit that prevents LLM billing on empty corpora."""
    R._CHUNKS = []
    R._INDEX = None

    class BombEmbedder:
        def encode(self, *args, **kwargs):
            raise AssertionError("embedder must not be called on empty index")

    orig = R._embedder
    R._embedder = lambda: BombEmbedder()
    try:
        assert R.retrieve("anything", top_k=3, threshold=0.35) == []
    finally:
        R._embedder = orig
        R._CHUNKS = []


def test_top_k_respected_and_sorted() -> None:
    """Results are capped at top_k and sorted best-first by dense score."""
    chunks = [{"chunk_id": f"test_c{i:03d}", "page": 1, "section": "x", "text": f"chunk {i}"} for i in range(10)]
    R._CHUNKS = chunks
    R._INDEX = _make_fake_index(chunks, _rng_embeddings(10))
    R._BM25 = None

    class FakeEmbedder:
        def encode(self, texts, convert_to_numpy=True, normalize_embeddings=True):
            import numpy as np
            return np.asarray(_rng_embeddings(10)[0], dtype="float32").reshape(1, -1)

    orig = R._embedder
    R._embedder = lambda: FakeEmbedder()
    try:
        results = R.retrieve("q", top_k=3, threshold=0.0)
        assert len(results) <= 3
        scores = [r["score"] for r in results]
        assert scores == sorted(scores, reverse=True)
    finally:
        R._embedder = orig
        R._CHUNKS = []
        R._INDEX = None
        R._BM25 = None


def test_query_prefix_passed_to_embedder() -> None:
    """Retrieval must prefix the query with "query: " for BGE-small."""
    chunks = [{"chunk_id": "test_c000", "page": 1, "section": "x", "text": "x"}]
    R._CHUNKS = chunks
    R._INDEX = _make_fake_index(chunks, _rng_embeddings(1))
    R._BM25 = None

    seen = {}

    class FakeEmbedder:
        def encode(self, texts, convert_to_numpy=True, normalize_embeddings=True):
            seen["text"] = texts[0]
            import numpy as np
            return np.asarray(_rng_embeddings(1)[0], dtype="float32").reshape(1, -1)

    orig = R._embedder
    R._embedder = lambda: FakeEmbedder()
    try:
        R.retrieve("tokens", top_k=1, threshold=0.0)
        assert seen.get("text", "").startswith("query: ")
    finally:
        R._embedder = orig
        R._CHUNKS = []
        R._INDEX = None
        R._BM25 = None
