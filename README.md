# RAG Course-Note Assistant — AI Textbook

Private RAG assistant that answers only from uploaded AI textbook PDFs with citations.
Says "Not in notes" instead of hallucinating. Deployed on GCP (Cloud Run, free-tier safe).

## Stack
- API: FastAPI + Uvicorn
- UI: Streamlit
- Embeddings: BAAI/bge-small-en-v1.5 (384-dim, CPU)
- Vector store: FAISS + Google Cloud Storage
- Logs: BigQuery (query_logs table)
- LLM: Gemini 2.5 Flash (default) + NVIDIA NIM hosted (alternative, OpenAI-compatible), switch via LLM_PROVIDER

## Local quickstart
  uv sync
  uv run uvicorn api.main:app --host 0.0.0.0 --port 8080
  curl http://localhost:8080/health

## Day-1 cost guardrails (set before FIRST deploy)
1. GCP Billing -> Budgets & alerts: $40 budget, email at 50/90/100%
2. Region: us-central1 for everything
3. Cloud Run: --min-instances=0 --max-instances=5 (never 1 = idle charge)
4. BigQuery: every query uses maximum_bytes_billed=10737418240 (10 GB cap)
5. LLM token caps: MAX_TOKENS=768, TEMPERATURE=0.2, daily query cap in code
6. Retrieval threshold: RELEVANCE_THRESHOLD=0.70 (calibrated via eval sweep — see REPORT §19; never ship the uncalibrated 0.35)

## Never deploy
- Vertex AI endpoints ($1.375/hr ~= $990/mo if left on)
- GKE clusters (~$60/mo compute)
- Compute GPU instances (blocked on free trial)
- Self-hosted NIMs (need NVIDIA GPU)

## Teardown
  gcloud run services delete rag-assistant --region=us-central1
  gsutil -m rm -r gs://<bucket>/raw/tmp/
  # Keep /index/faiss.idx + chunks.json (~77 MB) for 2-min revival