# Profound Contract-Analysis RAG

A retrieval-augmented generation (RAG) system for procurement contract analysis. Ingests PDFs into a Supabase pgvector store, answers natural-language questions with hybrid retrieval + cross-encoder reranking, and routes specialized queries (dates, cross-corpus scans, risk audits) through deterministic non-RAG paths where they belong.

Built on FastAPI, deployed to AWS Lambda via a container image and a GitHub Actions OIDC pipeline.

---

## What it does

Ask questions about your contract portfolio in plain English:

- **Clause lookup** — *"What's the termination-for-convenience clause in MSA_Logistics?"*
- **Cross-corpus comparison** — *"Which contracts have liability caps above $5M?"*
- **Date queries** — *"Contracts expiring in October 2026"* — answered via SQL over structured metadata, not vector search.
- **Risk audit** — single-click standing scan of the whole corpus for auto-renewal traps, uncapped liability, asymmetric indemnification, IP assignment imbalances, etc.
- **Follow-ups** — multi-turn chat with query rewriting. *"What about the renewal clause?"* resolves against the previous turn.

---

## Architecture

```
┌──────────────────────────────────────────────────────────────┐
│  Browser UI (static/index.html) — SSE streaming, multi-turn  │
└──────────────────────────────────────────────────────────────┘
                │
                ▼
┌──────────────────────────────────────────────────────────────┐
│  FastAPI (server.py)  —  Intent router                       │
│                                                              │
│    small-talk → date-query → broad-query → RAG               │
│                     │            │          │                │
│                     ▼            ▼          ▼                │
│          SQL over          /scan/topic    Hybrid retrieval   │
│         contract_          (full-corpus   + cross-encoder    │
│         metadata          keyword scan)    rerank + LLM      │
└──────────────────────────────────────────────────────────────┘
                │
                ▼
┌──────────────────────────────────────────────────────────────┐
│   Supabase (Postgres + pgvector)                             │
│   ├─ contract_chunks    — embeddings + full-text index       │
│   └─ contract_metadata  — effective/expiration/term/parties  │
└──────────────────────────────────────────────────────────────┘
```

### Retrieval pipeline ([query_rag.py](query_rag.py))

1. **Query expansion** — GPT rewrites short questions into semantically richer queries.
2. **Dual-embedding vector search** — original + expanded, via a `match_contracts` Postgres RPC.
3. **Full-text search** — parallel FTS via `keyword_search_contracts` RPC (`to_tsvector` + `plainto_tsquery`).
4. **Preamble anchoring** — chunk_index = 0 of every matched document is guaranteed in context.
5. **Target-doc detection** — fuzzy match between the question and filenames; if a specific contract is named, all its chunks are loaded and non-target chunks filtered out.
6. **Cross-encoder reranking** — `ms-marco-MiniLM-L-12-v2` scores candidates against the original query.
7. **Neighbor expansion** — adjacent chunks (index ± 1) to top hits bypass the reranker and go straight to the LLM, catching clauses that sit next to high-scoring ones.
8. **Generation** — `gpt-4o-mini` with a strict legal-assistant system prompt.

### Intent routing ([server.py](server.py))

Regex-based detectors with an LLM used narrowly for parsing — not a trained classifier.

- **Small-talk short-circuit** — greetings/thanks/identity get canned replies, zero retrieval cost.
- **Conversational memory** — frontend tracks last 5 turns; backend rewrites follow-ups into standalone queries (no chain framework, built from scratch).
- **Date-query routing** — keyword detection + LLM-based JSON parsing of natural-language windows ("contracts expiring in May 2026") into SQL filters over `contract_metadata`.
- **Quantitative-filter stripping** — dollar thresholds ("above $5M") are removed from the retrieval topic to preserve recall, then reapplied by the LLM at synthesis time.
- **Broad-query routing** — patterns like "which contracts have X" route to the full-corpus topic scanner instead of top-k RAG.

### Full-corpus analytics

- **`/scan`** — standing risk audit. Scores every chunk by density of risk keywords, picks top-N per document, synthesizes a severity-ranked report (LOW/MEDIUM/HIGH/CRITICAL). Calibrated against market baselines so standard clauses (12-month liability caps, NDA injunctive relief, 30-day termination notice) don't get flagged as anomalies.
- **`/scan/topic`** — parameterized topic scan. GPT expands a topic into 15–25 legal synonyms, scores every chunk corpus-wide, synthesizes a grouped-by-document answer. Used directly or called by the broad-query router.

---

## Evaluation

Measured against the [CUAD benchmark](https://www.atticusprojectai.org/cuad) (SEC-filed commercial contracts with expert-labeled clause spans).

- **Test set**: 10 CUAD contracts × ~7 legal categories (Parties, Termination for Convenience, Cap on Liability, Notice to Terminate Renewal, Governing Law, Insurance, Expiration Date) ≈ 70 Q&A pairs.
- **Metrics**:
  - *Source retrieval*: was the correct document in the reranked top-k?
  - *Semantic accuracy*: `gpt-4o-mini` judges each answer CORRECT / PARTIAL / INCORRECT against the ground-truth span. Composite = `(correct + 0.5 · partial) / total`.
- **Result: 38% → 73% semantic accuracy** after adding reranking, hybrid search, preamble anchoring, and neighbor expansion.
- Known limits: small sample, LLM-as-judge bias, fuzzy span-recall as a secondary signal only.

Run: `python eval_cuad.py` (writes `cuad_eval_results.json`).

---

## Project structure

```
├── server.py               # FastAPI app, intent router, endpoints
├── query_rag.py            # Hybrid retrieval pipeline + reranker
├── ingest_to_db.py         # PDF → markdown → chunks → embeddings → Supabase
├── extract_metadata.py     # One-shot metadata extraction (dates, parties, etc.)
├── eval_cuad.py            # CUAD benchmark evaluation
├── generate_contracts.py   # Synthetic test-corpus generator (Gemini)
├── lambda_handler.py       # Mangum ASGI adapter for AWS Lambda
├── Dockerfile              # Multi-stage build; bakes cross-encoder into image
├── .github/workflows/
│   └── deploy.yml          # GitHub Actions → ECR → Lambda (OIDC)
└── static/
    └── index.html          # Single-file vanilla JS UI, SSE streaming
```

---

## Local setup

### Prerequisites

- Python 3.11
- A Supabase project with `pgvector` enabled
- An OpenAI API key

### Install

```bash
git clone https://github.com/<you>/Profound_Rag_Backend.git
cd Profound_Rag_Backend
python -m venv venv
source venv/bin/activate     # Windows: venv\Scripts\activate
pip install -r requirements.txt
```

### Environment

Create `.env`:

```bash
OPENAI_API_KEY=sk-...
SUPABASE_URL=https://<project>.supabase.co
SUPABASE_KEY=<service-role-or-anon-key>
```

### Database schema

Run in the Supabase SQL editor:

```sql
create extension if not exists vector;

create table if not exists contract_chunks (
    id            text primary key,
    document_name text not null,
    content       text not null,
    embedding     vector(1536),
    chunk_index   integer
);

create index if not exists contract_chunks_embedding_idx
    on contract_chunks using ivfflat (embedding vector_cosine_ops)
    with (lists = 10);

create table if not exists contract_metadata (
    document_name            text primary key,
    effective_date           date,
    expiration_date          date,
    term_months              integer,
    renewal_type             text,
    auto_renewal_notice_days integer,
    parties                  text[],
    governing_law            text,
    contract_type            text,
    extracted_at             timestamptz default now()
);

create index if not exists contract_metadata_expiration_idx on contract_metadata(expiration_date);
create index if not exists contract_metadata_renewal_idx    on contract_metadata(renewal_type);
```

Also create the two RPCs (`match_contracts`, `keyword_search_contracts`) — see the docstring at the top of [query_rag.py](query_rag.py).

### Ingest + extract

```bash
# Put PDFs in dummy_dataset/pdfs/ (or set PDF_DIR)
python ingest_to_db.py
python extract_metadata.py
```

### Run locally

```bash
uvicorn server:app --reload --port 8000
# open http://localhost:8000
```

---

## Deployment (AWS Lambda)

Deployment is triggered by `git push` to `main` via [.github/workflows/deploy.yml](.github/workflows/deploy.yml).

The pipeline:

1. Authenticates to AWS via GitHub OIDC (no static keys).
2. Builds the Docker image with `docker buildx` (`--provenance=false --platform linux/amd64` — Lambda rejects OCI manifests).
3. Pushes to ECR tagged with the commit SHA.
4. Calls `aws lambda update-function-code` with the new image.
5. Rotates ECR images (keeps last 5) to cap storage costs.

### GitHub settings required

Variables:
- `AWS_REGION`
- `ECR_REPOSITORY`
- `LAMBDA_FUNCTION_NAME`

Secrets:
- `AWS_DEPLOY_ROLE_ARN` (IAM role configured for GitHub OIDC federation with `ecr:*` on the repo and `lambda:UpdateFunctionCode`, `lambda:GetFunctionConfiguration` on the function).

### Lambda container notes

- Cross-encoder model is **baked into the image** at build time so cold starts skip the HuggingFace download.
- `HF_HOME=/tmp/huggingface` — Lambda's `/var/task` is read-only.
- Lambda timeout should be 3 minutes (some `/scan` runs can take ~60s on a 50-contract corpus).

---

## Tech stack

| Layer | Choice |
|---|---|
| Language | Python 3.11 |
| Web framework | FastAPI + Mangum (Lambda adapter) |
| Vector store | Supabase Postgres + pgvector |
| Embeddings | OpenAI `text-embedding-3-small` (1536-dim) |
| Generation | OpenAI `gpt-4o-mini` |
| Reranker | `cross-encoder/ms-marco-MiniLM-L-12-v2` |
| PDF parsing | `pymupdf4llm` (Markdown extraction) |
| Chunking | LangChain `MarkdownHeaderTextSplitter` + `RecursiveCharacterTextSplitter` |
| CI/CD | GitHub Actions → ECR → AWS Lambda |
| Auth (CI) | GitHub OIDC federation (no stored AWS keys) |
| Frontend | Single-file vanilla JS + custom markdown renderer |

---

## API

| Method | Path | Purpose |
|---|---|---|
| `GET`  | `/`               | Serves the web UI |
| `GET`  | `/health`         | Liveness check |
| `GET`  | `/documents`      | List all ingested contract filenames |
| `POST` | `/query`          | RAG query — returns JSON with answer + sources |
| `POST` | `/query/stream`   | Same, streamed as SSE |
| `POST` | `/scan`           | Full-corpus risk audit |
| `POST` | `/scan/topic`     | Parameterized full-corpus topic scan |

Request bodies and response schemas are declared with Pydantic in [server.py](server.py).

---

## License

MIT