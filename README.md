# RAG Course-Note Assistant

An AI-powered study assistant built using **Retrieval-Augmented Generation (RAG)** that answers questions from course materials with contextual accuracy, source citations, and hallucination prevention.

The application combines hybrid information retrieval, Large Language Models (LLMs), and Google Cloud services to provide students with reliable, document-grounded answers.

## Overview

The RAG Course-Note Assistant allows students to interact with course materials through a conversational interface. Instead of relying entirely on the pretrained knowledge of an LLM, the system retrieves relevant passages from uploaded study documents and uses them as context for generating answers.

The project was developed around an AI Engineering textbook containing 529 pages and initially 721 document chunks.

**Key highlights:**
- Retrieval-Augmented Generation for document-grounded question answering.
- Hybrid retrieval using FAISS and BM25.
- Reciprocal Rank Fusion (RRF) for combining retrieval results.
- Dual-provider LLM integration using NVIDIA and Google Gemini.
- Source citations with page numbers and chunk identifiers.
- Two independent hallucination-prevention mechanisms.
- PDF upload and live index updates.
- Interactive Streamlit dashboard.
- Cloud deployment using Google Cloud Run.
- Automated CI/CD using GitHub Actions and Google Cloud Build.
- Infrastructure management using Terraform.

## Features

### 1. Document Ingestion
- Extracts text from PDF documents using PyMuPDF.
- Uses heading-aware recursive text chunking.
- Maintains page-level metadata and chunk identifiers.
- Generates embeddings for document chunks.
- Stores the vector index and metadata in Google Cloud Storage.

### 2. Hybrid Retrieval

The system combines two retrieval approaches:

| Retrieval Method | Purpose |
|---|---|
| FAISS | Dense vector similarity search |
| BM25 | Keyword-based lexical retrieval |
| RRF | Combines rankings from both retrieval methods |

The system uses the BAAI/bge-small-en-v1.5 embedding model and Reciprocal Rank Fusion with k = 60.

A calibrated dense similarity threshold of 0.70 filters irrelevant results before generation.

### 3. LLM-Based Answer Generation

The generation layer supports two LLM providers:

- NVIDIA-hosted `z-ai/glm-5.3-flash` (default).
- Google Gemini (fallback).

The LLM generates answers using retrieved document passages as context.

### 4. Hallucination Prevention

The application implements two independent refusal mechanisms:

- **Retrieval-level refusal:** If no relevant passages are retrieved, the system returns "Not in notes" without calling the LLM.
- **Citation validation:** Generated citations are checked against the retrieved chunk identifiers. Invalid citations result in refusal.

Every factual answer is expected to include page-level citations and source chunk identifiers.

### 5. Interactive Dashboard

The Streamlit interface provides:

- Conversational question-answering.
- Chat history.
- Source expanders and citation indicators.
- LLM provider selection.
- Response latency and token usage.
- Retrieval evaluation metrics.
- Dashboard for monitoring system performance.

### 6. Cloud Deployment

The application is deployed using two independent Google Cloud Run services:

- **rag-api:** FastAPI backend responsible for document ingestion, retrieval and answer generation.
- **rag-ui:** Streamlit frontend responsible for user interaction.

Additional Google Cloud services include:
- Cloud Storage for persistent retrieval indexes.
- Secret Manager for API key management.
- Artifact Registry for Docker images.
- Cloud Build for automated deployment.
- Terraform for Infrastructure as Code.

## System Architecture

The application follows this workflow:

1. Course PDFs are uploaded to the application.
2. Text is extracted and divided into smaller chunks.
3. BGE embeddings are generated for the chunks.
4. FAISS and BM25 indexes are used for retrieval.
5. Reciprocal Rank Fusion combines the retrieved results.
6. A similarity threshold filters irrelevant passages.
7. Retrieved passages are supplied to the selected LLM.
8. The generated response undergoes citation validation.
9. The final answer, with source references, is displayed to the user.

The FAISS index and chunk metadata are persisted in Google Cloud Storage and downloaded when the backend starts.

## Technology Stack

| Category | Technologies |
|---|---|
| Programming Language | Python |
| Backend | FastAPI |
| Frontend | Streamlit |
| RAG Framework | Custom RAG pipeline |
| Embedding Model | BAAI/bge-small-en-v1.5 |
| Vector Search | FAISS |
| Keyword Retrieval | BM25 |
| Ranking | Reciprocal Rank Fusion |
| LLM Providers | NVIDIA, Google Gemini |
| PDF Processing | PyMuPDF |
| Cloud Platform | Google Cloud Platform |
| Deployment | Google Cloud Run |
| Storage | Google Cloud Storage |
| Secrets Management | Google Secret Manager |
| Containerization | Docker |
| CI/CD | GitHub Actions, Google Cloud Build |
| Infrastructure as Code | Terraform |
| Testing | Pytest |
| Code Quality | Ruff |
| Security Scanning | Trivy |
| Dependency Management | uv |

## Project Structure

```text
RAG-Course-Note-Assistant/
│
├── .github/
│   └── workflows/
│       └── ci.yml
│
├── .streamlit/
│   └── config.toml
│
├── api/
│   └── main.py
│
├── eval/
│   ├── eval_hitrate.py
│   ├── failed_cases.json
│   ├── metrics.json
│   └── questions.yaml
│
├── generate/
│   ├── llm_client.py
│   └── prompt.py
│
├── infra/
│   ├── main.tf
│   ├── cloudbuild-deploy-api.yaml
│   ├── cloudbuild-deploy-ui.yaml
│   └── cloudbuild-deploy.yaml
│
├── ingest/
│   ├── chunk.py
│   ├── embed.py
│   ├── parse_pdf.py
│   └── pipeline.py
│
├── retrieve/
│   └── retrieval.py
│
├── tests/
│   ├── conftest.py
│   ├── test_api.py
│   ├── test_retrieval.py
│   └── test_upload.py
│
├── ui/
│   ├── app.py
│   └── pages/
│       └── 01_Dashboard.py
│
├── Dockerfile
├── Dockerfile.ui
├── pyproject.toml
├── uv.lock
└── README.md
```

## Evaluation Results

The retrieval system was evaluated using a 12-question gold dataset consisting of 10 answerable questions and 2 deliberate refusal tests.

| Metric | Result |
|---|---:|
| Initial document size | 529 pages |
| Initial document chunks | 721 |
| Chunks after live uploads | 787 |
| Hit@3 | 10/10 |
| Refusal Accuracy | 2/2 |
| Selected Similarity Threshold | 0.70 |
| Automated Offline Tests | 14 |

The threshold was calibrated experimentally to achieve perfect refusal accuracy without sacrificing retrieval performance on the answerable questions.

## API Endpoints

The FastAPI backend exposes the following endpoints:

| Method | Endpoint | Description |
|---|---|---|
| GET | `/health` | Checks backend health |
| GET/POST | `/params` | Retrieves or updates retrieval and generation parameters |
| POST | `/query` | Accepts a question and generates a grounded response |
| POST | `/upload` | Uploads and processes PDF documents |
| GET | `/documents` | Lists indexed documents |

Interactive API documentation is available through FastAPI's `/docs` endpoint when the backend is running.

## Installation and Local Setup

### Prerequisites

- Python 3.12
- Git
- uv package manager
- NVIDIA or Google Gemini API key

### 1. Clone the repository

```bash
git clone https://github.com/shannoncoelho-mscai/RAG-Course-Note-Assistant.git

cd RAG-Course-Note-Assistant
```

### 2. Install dependencies

```bash
pip install uv
uv sync --frozen
```

### 3. Configure environment variables

Create a `.env` file using `.env.example` as a reference.

Configure the required API keys and application settings.

Never commit your actual `.env` file or API keys to GitHub.

### 4. Run the backend

```bash
uv run uvicorn api.main:app --host 0.0.0.0 --port 8080
```

### 5. Run the frontend

Open another terminal and execute:

```bash
uv run streamlit run ui/app.py
```

The Streamlit application will be available at:

```text
http://localhost:8501
```

The backend API documentation will be available at:

```text
http://localhost:8080/docs
```

**Note:** Full functionality requires the appropriate environment configuration, API credentials and an initialized document index. Cloud Storage configuration is required for the deployed persistence workflow.

## Docker Deployment

The project includes separate Dockerfiles for the backend and frontend.

Build the backend image:

```bash
docker build -t rag-api -f Dockerfile .
```

Build the frontend image:

```bash
docker build -t rag-ui -f Dockerfile.ui .
```

The containers can then be deployed to Google Cloud Run with the required environment variables, service configuration and permissions.

## CI/CD Pipeline

The project uses GitHub Actions for continuous integration and Google Cloud Build for continuous deployment.

**Continuous Integration:**
- Reproducible dependency installation using `uv sync --frozen`.
- Ruff code linting.
- Automated offline pytest suite.
- Docker image build.
- Trivy vulnerability scanning.

**Continuous Deployment:**

A Cloud Build trigger monitors pushes to the `main` branch. When changes are pushed:

1. The Docker image is built.
2. The image is pushed to Artifact Registry.
3. The corresponding Cloud Run service is updated.
4. A new immutable deployment revision is created.

Terraform manages the infrastructure configuration, while Cloud Build manages application releases.

## Security Considerations

- API keys are stored in Google Secret Manager.
- IAM permissions follow the principle of least privilege.
- PDF uploads are validated using file signatures and size restrictions.
- Duplicate document uploads are rejected.
- Dependencies are locked through `uv.lock`.
- Docker images undergo vulnerability scanning.
- Private course PDFs and local environment files are excluded through `.gitignore`.

The current deployment allows public access for demonstration purposes. Production deployment would require identity-aware access control.

## Limitations and Future Improvements

Potential improvements include:

- Implementing identity-aware authentication.
- Reducing cold-start latency.
- Expanding the evaluation dataset.
- Adding generation-quality evaluation using citation precision, recall and groundedness.
- Comparing alternative index persistence strategies.
- Improving multi-instance state synchronization.
- Adding more document formats and advanced retrieval strategies.

## Contributors

**Goa University — M.Sc. Artificial Intelligence**

| Contributor | Roll Number |
|---|---|
| Shubham Malik | 25P0620020 |
| Shannon Coelho | 25P0620019 |
| Vaibhav Naik | 25P0620022 |

## License

This repository is intended for academic and educational purposes. Please ensure that any course materials or copyrighted documents used with the application are handled in accordance with their applicable usage permissions.

---

**RAG Course-Note Assistant** — Bringing reliable, source-grounded AI assistance to course learning.
