"""Test conftest — stub the embedder so pytest never downloads the model.

`retrieve/retrieval.py` loads BGE-small lazily via `_embedder()`. Without
this stub, `tests/test_api.py` (which imports `api.main` -> `retrieve`
-> `load_index()` at module import) triggers a SentenceTransformer load
on every test collection, making the whole suite hang for 60-120s.

The stub returns a deterministic 384-dim unit vector, identical for every
query. Retrieval tests (`tests/test_retrieval.py`) override `_embedder`
with their own fakes; this conftest just guarantees the *collection* step
is fast and hermetic.
"""

import sys
from pathlib import Path

# Ensure the repo root is importable (same as root conftest.py)
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))


def _stub_embedder():
    import numpy as np

    class _Stub:
        def encode(self, texts, *args, **kwargs):
            # Deterministic 384-dim vector; every query gets the same one.
            # *args/**kwargs absorb whatever callers pass (show_progress_bar,
            # batch_size, ...) — real SentenceTransformer accepts them too.
            vec = np.zeros(384, dtype="float32")
            vec[0] = 1.0
            return np.tile(vec, (len(texts), 1))

    return _Stub()


def _install_stub():
    try:
        import retrieve.retrieval as R
        R._embedder = _stub_embedder
    except ImportError:
        pass  # retrieval not importable yet — tests will fail for other reasons


_install_stub()
