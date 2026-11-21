"""Tests for the live /upload endpoint (ingest → append → persist).

Fully isolated: a fixture swaps the module globals for a tiny fake
index, redirects index writes to a tmp dir, and disables the GCS bucket
upload — so tests never touch the real index/ files or the GCS bucket.
"""

import io

import fitz
import pytest
from fastapi.testclient import TestClient

import retrieve.retrieval as R
from api.main import app

client = TestClient(app)


def _make_pdf(text: str) -> bytes:
    """Build a real 1-page PDF in memory with the given text."""
    doc = fitz.open()
    page = doc.new_page()
    page.insert_text((72, 72), text)
    buf = io.BytesIO()
    doc.save(buf)
    doc.close()
    return buf.getvalue()


def _tiny_index():
    """Real FAISS index with one stub vector (matches the conftest embedder)."""
    import faiss
    import numpy as np

    idx = faiss.IndexFlatIP(384)
    vec = np.zeros(384, dtype="float32")
    vec[0] = 1.0
    idx.add(vec.reshape(1, -1))
    return idx


@pytest.fixture(autouse=True)
def _isolated_index(monkeypatch, tmp_path):
    """Swap globals for a tiny fake index; redirect writes; disable GCS."""
    monkeypatch.setattr(R, "_INDEX", _tiny_index())
    monkeypatch.setattr(R, "_CHUNKS", [
        {"chunk_id": "base_c000", "page": 1, "section": "x", "text": "base chunk", "source": "base"}
    ])
    monkeypatch.setattr(R, "_BM25", None)
    monkeypatch.setenv("FAISS_INDEX_PATH", str(tmp_path / "faiss.idx"))
    monkeypatch.setenv("CHUNKS_META_PATH", str(tmp_path / "chunks.json"))
    monkeypatch.setenv("GCS_INDEX_BUCKET", "")  # empty → persist_index goes local-only
    yield


def _upload(name: str, text: str):
    pdf = _make_pdf(text)
    return client.post("/upload", files={"file": (f"{name}.pdf", pdf, "application/pdf")})


def test_upload_ingests_and_indexes_new_document() -> None:
    resp = _upload("sample_doc", "uvicorn serves ASGI applications over HTTP.")
    assert resp.status_code == 200
    data = resp.json()
    assert data["chunks_added"] >= 1
    assert data["total_chunks"] > 1
    assert data["source"] == "sample_doc"
    assert data["persisted_to_bucket"] is False  # GCS disabled in tests


def test_upload_duplicate_source_returns_409() -> None:
    _upload("dup_doc", "First upload of the duplicate doc.")
    resp = _upload("dup_doc", "Second upload of the same source name.")
    assert resp.status_code == 409
    assert "already indexed" in resp.json()["detail"]


def test_upload_rejects_non_pdf() -> None:
    resp = client.post("/upload", files={"file": ("notes.txt", b"plain text, not a pdf", "text/plain")})
    assert resp.status_code == 400


def test_upload_rejects_oversize() -> None:
    big = b"%PDF-1.4 " + b"x" * (31 * 1024 * 1024)
    resp = client.post("/upload", files={"file": ("big.pdf", big, "application/pdf")})
    assert resp.status_code == 413


def test_documents_lists_sources() -> None:
    resp = client.get("/documents")
    assert resp.status_code == 200
    data = resp.json()
    assert data["count"] >= 1
    assert "sample_doc" in data["sources"] or "base" in data["sources"]
