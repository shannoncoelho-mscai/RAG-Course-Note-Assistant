"""Streamlit chat UI for the RAG assistant.

Talks to the FastAPI backend via HTTP (httpx). The API URL is set via
`API_URL` env var (default http://localhost:8000). Sidebar toggles let
you switch LLM provider/model without restarting — this is the per-request
override wired through QueryRequest.provider/model in api/main.py.
"""

import os
import re

import httpx
import streamlit as st

API_URL = os.getenv("API_URL", "http://localhost:8000")
NVIDIA_MODELS = [
    m.strip()
    for m in os.getenv(
        "NVIDIA_MODELS",
        "deepseek-ai/deepseek-v4.1-flash,z-ai/glm-5.3-flash,nvidia/nemotron-3.5-lightning-30b-a3b",
    ).split(",")
    if m.strip()
]

st.set_page_config(page_title="RAG Assistant — AI Textbook", page_icon="📚", layout="centered")

# --- CSS: hero header, chat bubbles, metric cards, page background --------
st.markdown(
    """
    <style>
    .stApp {
        background: linear-gradient(180deg, #fafafa 0%, #f0f4ff 100%);
    }
    /* Hero header card */
    .hero {
        background: linear-gradient(135deg, #1a4d8f 0%, #4d1a8f 100%);
        border-radius: 16px;
        padding: 28px 32px;
        color: white;
        margin-bottom: 18px;
        box-shadow: 0 6px 20px rgba(26, 77, 143, 0.25);
    }
    .hero h1 {
        margin: 0 0 6px 0;
        font-size: 2.1em;
        font-weight: 700;
        color: white;
    }
    .hero p {
        margin: 0;
        font-size: 1.05em;
        opacity: 0.92;
    }
    .hero .badge {
        display: inline-block;
        background: rgba(255, 255, 255, 0.18);
        border-radius: 12px;
        padding: 3px 12px;
        font-size: 0.85em;
        margin-right: 8px;
        margin-top: 10px;
    }
    /* Chat bubbles */
    [data-testid="stChatMessage"] {
        border-radius: 14px;
        padding: 6px 4px;
        margin-bottom: 10px;
    }
    /* Assistant bubble: white card with border */
    [data-testid="stChatMessage"].stChatMessage-assistant {
        background: #ffffff;
        border: 1px solid #e2e8f0;
        box-shadow: 0 2px 8px rgba(0, 0, 0, 0.04);
    }
    /* Metric cards */
    [data-testid="stMetric"] {
        background: #ffffff;
        border: 1px solid #e2e8f0;
        border-radius: 10px;
        padding: 12px 16px;
        box-shadow: 0 2px 6px rgba(0, 0, 0, 0.05);
    }
    /* Sidebar styling */
    [data-testid="stSidebar"] {
        background: #f7f9fc;
        border-right: 1px solid #e2e8f0;
    }
    /* Citation chips */
    .cite {
        background: #dbeafe;
        border: 1px solid #93c5fd;
        border-radius: 6px;
        padding: 1px 6px;
        font-family: monospace;
        font-size: 0.88em;
        color: #1a4d8f;
        white-space: nowrap;
    }
    /* Sources expander accent */
    [data-testid="stExpander"] {
        border-left: 3px solid #1a4d8f;
        border-radius: 8px;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

# --- Hero header (badges are LIVE — fetched from the API /documents) --------
def _hero_badges() -> str:
    """Fetch live corpus stats so the hero badges are always accurate."""
    try:
        docs = httpx.get(f"{API_URL}/documents", timeout=5).json()
        return (
            f'<span class="badge">⚡ GLM 5.3 Flash</span>'
            f'<span class="badge">📖 {docs.get("total_chunks", 0):,} chunks · {docs.get("count", 0)} docs</span>'
            f'<span class="badge">📄 live PDF upload</span>'
            f'<span class="badge">✅ Hit@3 = 1.0</span>'
        )
    except Exception:
        return '<span class="badge">● API offline</span>'


st.markdown(
    f"""
    <div class="hero">
        <h1>📚 RAG Assistant</h1>
        <p>Multi-document assistant · answers with citations · says "Not in notes" instead of hallucinating</p>
        {_hero_badges()}
    </div>
    """,
    unsafe_allow_html=True,
)

_CITATION_RE = re.compile(r"\[\s*p\.\s*([^\],]+)\s*,\s*([^\]]+)\s*\]")


def _highlight_citations(text: str) -> str:
    """Wrap [p.<page>, <chunk_id>] citations in <span class="cite"> for styling."""
    return _CITATION_RE.sub(r'<span class="cite">[p.\1, \2]</span>', text)


def _is_refusal(text: str) -> bool:
    """True if the answer is the 'Not in notes' refusal path."""
    return text.strip().lower().startswith("not in notes")


def _render_assistant(data: dict) -> None:
    """Render one assistant turn: answer, meta caption, sources expander."""
    answer = data.get("answer", "Not in notes.")
    sources = data.get("sources", [])

    if _is_refusal(answer):
        st.info(answer)
    else:
        st.markdown(_highlight_citations(answer), unsafe_allow_html=True)

    meta = (
        f"{data.get('llm_provider')}/{data.get('llm_model')} · "
        f"{data.get('latency_ms')}ms · {data.get('tokens_used')} tokens"
    )
    st.caption(meta)

    if sources:
        with st.expander("🔎 Sources"):
            for s in sources:
                st.write(
                    f"- **{s.get('chunk_id')}** — p.{s.get('page')} — "
                    f"{s.get('section') or 'untitled'} "
                    f"(score {s.get('score')})"
                )


# --- sidebar: health + provider + sample questions ------------------------
with st.sidebar:
    st.header("⚙️ LLM Settings")
    provider_choice = st.radio(
        "Provider",
        ["GLM 5.3 Flash", "Gemini (default)", "DeepSeek V4.1 Flash", "Nemotron 3.5 Lightning", "Auto"],
        index=0,
        horizontal=True,
    )
    if provider_choice == "GLM 5.3 Flash":
        provider, model = "nvidia", "z-ai/glm-5.3-flash"
    elif provider_choice == "DeepSeek V4.1 Flash":
        provider, model = "nvidia", "deepseek-ai/deepseek-v4.1-flash"
    elif provider_choice == "Nemotron 3.5 Lightning":
        provider, model = "nvidia", "nvidia/nemotron-3.5-lightning-30b-a3b"
    elif provider_choice == "Auto":
        provider, model = "auto", None
    else:
        provider, model = "gemini", None
    st.caption(
        f"Backend: `{API_URL}`\n"
        f"Provider: {provider}{'/'+model if model else ''}"
    )

    # --- health check ---------------------------------------------------
    try:
        health = httpx.get(f"{API_URL}/health", timeout=5).json()
        st.success(f"● API online ({health.get('status')}, provider={health.get('provider')})")
    except Exception:
        st.error("● API offline — is `uv run uvicorn api.main:app` running?")

    # --- last-query metrics --------------------------------------------
    if "last_meta" in st.session_state:
        m = st.session_state["last_meta"]
        st.subheader("📊 Last Query")
        col1, col2 = st.columns(2)
        with col1:
            st.metric("Latency", f"{m.get('latency_ms')}ms")
        with col2:
            st.metric("Tokens", f"{m.get('tokens_used')}")

    # --- sample questions (1-click demo) -------------------------------
    st.subheader("💡 Try a Question")
    samples = [
        "How do LLMs represent text as tokens?",
        "What is the difference between traditional ML and foundation models?",
        "What is today's stock price of Apple?",
    ]
    for s in samples:
        if st.button(s, key=f"sample_{s}", use_container_width=True):
            st.session_state["sample"] = s

    # --- previous-question selector ------------------------------------
    prior = [m["content"] for m in st.session_state.get("messages", []) if m["role"] == "user"]
    if prior:
        st.subheader("🔁 Re-ask")
        choice = st.selectbox("Prior question", prior, key="reask_select")
        if st.button("Re-ask this", key="reask_btn", use_container_width=True):
            st.session_state["sample"] = choice

    # --- live PDF upload --------------------------------------------------
    st.subheader("📄 Upload PDF")
    up = st.file_uploader("Add a document to the corpus", type=["pdf"])
    if up is not None and st.button("Ingest this PDF", key="ingest_btn", use_container_width=True):
        with st.spinner("Ingesting… (parse → chunk → embed)"):
            try:
                files = {"file": (up.name, up.getvalue(), "application/pdf")}
                resp = httpx.post(f"{API_URL}/upload", files=files, timeout=600)
                data = resp.json()
            except Exception as e:
                st.error(f"Upload failed: {e}")
            else:
                if resp.status_code == 200:
                    st.success(
                        f"✅ **{data['source']}** added — {data['pages']} pages → "
                        f"{data['chunks_added']} chunks (corpus now {data['total_chunks']})"
                    )
                    st.toast("📄 Document is now searchable!")
                elif resp.status_code == 409:
                    st.info(data.get("detail", "Already indexed"))
                else:
                    st.error(data.get("detail", f"Upload failed ({resp.status_code})"))


st.caption(
    "Answers only from your textbook with citations. "
    "Says \"Not in notes\" instead of hallucinating."
)

# --- chat history -----------------------------------------------------------
if "messages" not in st.session_state:
    st.session_state.messages = []

for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        if msg["role"] == "assistant":
            _render_assistant(msg["data"])
        else:
            st.markdown(msg["content"], unsafe_allow_html=True)

# --- chat input (prefilled by sample button) --------------------------------
prompt = st.chat_input("Ask about the AI textbook...", key="chat_input")
if "sample" in st.session_state:
    prompt = st.session_state.pop("sample")

if prompt:
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    try:
        resp = httpx.post(
            f"{API_URL}/query",
            json={"question": prompt, "provider": provider, "model": model},
            timeout=900,  # match the API's own request timeout (NVIDIA can hang minutes)
        )
        data = resp.json()
    except Exception as e:  # network / timeout
        st.error(f"Backend error: {e}")
        st.stop()

    st.session_state["last_meta"] = data
    st.session_state.messages.append({"role": "assistant", "data": data})

    with st.chat_message("assistant"):
        _render_assistant(data)
    st.rerun()
