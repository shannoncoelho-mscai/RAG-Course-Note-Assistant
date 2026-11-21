"""Dashboard page — eval metrics + live system params.

Streamlit auto-discovers files in ui/pages/. This page reads
eval/metrics.json (offline eval of the gold set) and queries the
live API /params (current runtime configuration).

URL: http://localhost:8501/Dashboard
"""

import json
import os
from pathlib import Path

import httpx
import streamlit as st

API_URL = os.getenv("API_URL", "http://localhost:8000")

st.title("📊 Dashboard — Retrieval & System Health")
st.caption("Measured retrieval quality (offline eval) + live runtime config.")

# --- eval metrics (offline, from eval/metrics.json) -----------------------
st.subheader("Retrieval Quality (Gold-Set Eval, 12 Questions)")

metrics_path = Path("eval/metrics.json")
if metrics_path.exists():
    metrics = json.loads(metrics_path.read_text(encoding="utf-8"))
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Hit@3", metrics.get("hit_at_k", "—"),
                  help="Gold chunk in top-3 retrieved (exact ID match)")
    with col2:
        st.metric("MRR", metrics.get("mrr", "—"),
                  help="Mean reciprocal rank of first gold chunk (1/rank)")
    with col3:
        st.metric("Refusal Accuracy", metrics.get("refusal_accuracy", "—"),
                  help="% off-topic questions correctly short-circuited (no LLM call)")

    st.caption(
        f"Answerable: {metrics.get('answerable_passed', '?')} · "
        f"Refusals: {metrics.get('refusals_passed', '?')} · "
        f"Computed by `uv run python eval/eval_hitrate.py`"
    )

    # Per-question breakdown (expandable)
    per_q = metrics.get("per_question", [])
    if per_q:
        with st.expander("Per-question top-3 (exact chunk IDs)"):
            for pq in per_q:
                if pq["type"] == "exact":
                    st.write(
                        f"- **{pq['id']}** rank={pq['rank']} (rr={pq['rr']}): "
                        f"{', '.join(pq['retrieved'][:3])}"
                    )
                elif pq["type"] == "refusal":
                    st.write(f"- **{pq['id']}** refusal: pass={pq['pass']}")
else:
    st.warning("`eval/metrics.json` not found — run `uv run python eval/eval_hitrate.py` first.")

# --- live params (API /params) -------------------------------------------
st.subheader("Live Runtime Config (from API /params)")

try:
    params = httpx.get(f"{API_URL}/params", timeout=5).json()
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("top_k", params.get("top_k"))
    with col2:
        st.metric("Threshold", params.get("relevance_threshold"))
    with col3:
        st.metric("Temperature", params.get("temperature"))
    with col4:
        st.metric("Max tokens", params.get("max_tokens"))
    st.caption(
        f"Provider: {params.get('llm_provider')} · "
        f"Model: {params.get('nvidia_model')}"
    )
    st.caption(
        "Tune live with: `Invoke-RestMethod -Method Post -Uri "
        f"{API_URL}/params -ContentType 'application/json' "
        '-Body \'{"top_k":5}\'`'
    )
except Exception as e:
    st.error(f"API not reachable at {API_URL}: {e}")
    st.caption("Start the API first: `uv run uvicorn api.main:app --app-dir . --port 8000`")

# --- corpus stats (live, from API /documents — works on Cloud Run too) ----
st.subheader("Corpus")
try:
    docs = httpx.get(f"{API_URL}/documents", timeout=10).json()
    sources = docs.get("sources", [])
    total = docs.get("total_chunks", 0)
    st.write(f"**{total} chunks** across **{docs.get('count', len(sources))} document(s)**: "
             + ", ".join(f"`{s}`" for s in sources))
    st.caption("Index lives in Cloud Storage (rag-index-514120), downloaded by the API at "
               "cold start; live uploads via POST /upload append to it and persist back.")
except Exception as e:
    st.warning(f"Corpus stats unavailable — API unreachable at {API_URL}: {e}")
