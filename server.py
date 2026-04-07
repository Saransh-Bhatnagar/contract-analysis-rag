"""
server.py
---------
FastAPI wrapper around the RAG query engine.

Endpoints:
    GET  /health         — liveness check
    POST /query          — full RAG pipeline, returns JSON answer + sources
    POST /query/stream   — same pipeline, streams answer tokens via SSE

Run with:
    uvicorn server:app --reload --port 8000
"""

import time
from pathlib import Path
from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse, FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field
from openai import OpenAI

from query_rag import (
    expand_query,
    embed_query,
    search_contracts,
    keyword_search,
    fetch_preambles,
    fetch_neighbor_chunks,
    fetch_all_doc_chunks,
    merge_chunks,
    build_context_block,
    get_cross_encoder,
    detect_target_document,
    openai_client,
    supabase_client,
    GENERATION_MODEL,
    SYSTEM_PROMPT,
    TOP_K_CHUNKS,
    DOC_BOOST,
    enrich_reranker_query,
)


# ---------------------------------------------------------------------------
# Lifespan — warm up the cross-encoder once at startup
# ---------------------------------------------------------------------------
@asynccontextmanager
async def lifespan(app: FastAPI):
    get_cross_encoder()
    print("Server ready.")
    yield


app = FastAPI(
    title="Profound RAG — Contract Analysis API",
    version="1.0.0",
    lifespan=lifespan,
)

# CORS — allow any origin for dev/demo, tighten for production
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# Serve the UI at root
STATIC_DIR = Path(__file__).parent / "static"


@app.get("/", include_in_schema=False)
async def serve_ui():
    return FileResponse(STATIC_DIR / "index.html")


app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")


# ---------------------------------------------------------------------------
# Request / Response schemas
# ---------------------------------------------------------------------------
class QueryRequest(BaseModel):
    question: str = Field(..., min_length=1, max_length=1000)


class QueryResponse(BaseModel):
    answer: str
    sources: list[str]
    expanded_query: str
    chunks_retrieved: int
    time_seconds: float


# ---------------------------------------------------------------------------
# Shared retrieval logic (used by both /query and /query/stream)
# ---------------------------------------------------------------------------
def run_retrieval_pipeline(question: str) -> tuple[str, list[dict], list[dict]]:
    """
    Run the full retrieval pipeline: expand → embed → search → merge → rerank.
    Returns (expanded_query, top_chunks, all_chunks).
    """
    expanded = expand_query(question)

    query_vector = embed_query(question)
    if query_vector is None:
        raise HTTPException(status_code=502, detail="Failed to generate query embedding.")

    # Vector search (original + expanded)
    vector_orig = search_contracts(query_vector)
    vector_exp = []
    if expanded != question:
        exp_vector = embed_query(expanded)
        if exp_vector is not None:
            vector_exp = search_contracts(exp_vector)

    # FTS search (original + expanded)
    fts_orig = keyword_search(question)
    fts_exp = keyword_search(expanded) if expanded != question else []

    # Merge + preambles
    initial = merge_chunks(vector_orig, vector_exp, fts_orig, fts_exp)
    unique_docs = list({c.get("document_name") for c in initial if c.get("document_name")})
    preambles = fetch_preambles(unique_docs)
    all_chunks = merge_chunks(initial, preambles)

    # Propagate chunk_index for preamble boost
    preamble_ids = {c["id"] for c in preambles}
    for chunk in all_chunks:
        if chunk.get("id") in preamble_ids:
            chunk["chunk_index"] = 0

    if not all_chunks:
        return expanded, [], all_chunks

    # Detect if user targets a specific document
    target_doc = detect_target_document(question)

    # If targeting a specific doc, fetch ALL its chunks so the LLM sees the full contract
    if target_doc:
        doc_chunks = fetch_all_doc_chunks(target_doc)
        if doc_chunks:
            all_chunks = merge_chunks(all_chunks, doc_chunks)

    # Cross-encoder reranking
    pairs = [[question, c.get("content", "")] for c in all_chunks]
    scores = get_cross_encoder().predict(pairs)
    for chunk, score in zip(all_chunks, scores):
        boost = DOC_BOOST if target_doc and chunk.get("document_name") == target_doc else 0.0
        chunk["rerank_score"] = float(score) + boost

    all_chunks.sort(key=lambda x: x["rerank_score"], reverse=True)

    # When a target doc is detected, use a higher K to fit more of that document
    effective_top_k = max(TOP_K_CHUNKS, 48) if target_doc else TOP_K_CHUNKS

    # Filter out non-target-doc chunks to prevent cross-document contamination
    if target_doc:
        all_chunks = [c for c in all_chunks if c.get("document_name") == target_doc]

    # Select top-K with preamble cap
    MAX_PREAMBLES = 2
    top_chunks = []
    preamble_count = 0
    for chunk in all_chunks:
        if len(top_chunks) >= effective_top_k:
            break
        if chunk.get("chunk_index") == 0:
            if preamble_count >= MAX_PREAMBLES:
                continue
            preamble_count += 1
        top_chunks.append(chunk)

    # Post-reranker neighbor expansion (bypass reranker for adjacent chunks)
    neighbor_expansion = fetch_neighbor_chunks(top_chunks)
    if neighbor_expansion:
        top_chunks = merge_chunks(top_chunks, neighbor_expansion)

    return expanded, top_chunks, all_chunks


# ---------------------------------------------------------------------------
# GET /health
# ---------------------------------------------------------------------------
@app.get("/health")
async def health():
    return {"status": "ok"}


# ---------------------------------------------------------------------------
# GET /documents — list all ingested contracts (metadata query, not RAG)
# ---------------------------------------------------------------------------
@app.get("/documents")
async def list_documents():
    """Return every unique document name and total count."""
    response = supabase_client.table("contract_chunks") \
        .select("document_name") \
        .eq("chunk_index", 0) \
        .execute()
    docs = sorted({row["document_name"] for row in (response.data or [])})
    return {"count": len(docs), "documents": docs}


# ---------------------------------------------------------------------------
# POST /scan — full-corpus anomaly scan (not RAG — reads every document)
# ---------------------------------------------------------------------------
SCAN_KEYWORDS = [
    "liability", "indemnif", "damages", "penalty", "penalti",
    "liquidated", "termination", "cap", "uncapped", "unlimited",
    "waiv", "forfeit", "clawback",
    "auto-renew", "automatic renewal", "automatically renew",
    "non-renewal", "notice of non", "renewal term",
    "sole discretion", "without limitation", "minimum liability",
    "intellectual property", "ip assignment", "work product",
    "assign", "transfer", "ownership",
    "non-compete", "exclusiv",
]

SCAN_SYSTEM_PROMPT = """You are a senior contract auditor performing a full-corpus risk scan.

You will receive clauses extracted from multiple contracts, each section prefixed with === document_name ===.
Your job is to identify and rank ALL genuine anomalies — terms that materially deviate from the baselines below.

=== CALIBRATION: WHAT IS NORMAL (do NOT flag these) ===
- Liability cap = 12 months of fees paid: STANDARD. Only flag if cap is a specific dollar amount above $5M,
  uncapped, or the cap applies asymmetrically (one party only).
- Mutual indemnification for own breaches/negligence: STANDARD.
- One-sided indemnification (vendor indemnifies client only): COMMON in vendor agreements, flag only if
  there is zero reciprocal protection.
- Injunctive relief without bond in NDAs: STANDARD boilerplate in every NDA. Do NOT flag this.
- Confidentiality surviving 3-7 years: STANDARD. Flag only if indefinite or >10 years.
- Non-compete of 1-2 years in employment/advisor agreements: STANDARD. Flag only if >2 years or
  overbroad geography/scope.
- Service credits up to 25-30% of monthly fees for SLA breaches: STANDARD. Flag only if credits are the
  SOLE remedy with no termination right after repeated failures.
- Fixed-fee SOW amounts (even large ones like $95K): STANDARD unless fee is contingency-based or
  has automatic escalation clauses.
- Force majeure termination after 60-90 days: STANDARD.
- 30-day written notice for termination for convenience: STANDARD.

=== WHAT TO FLAG ===
- CRITICAL: Liability minimum or floor set at a specific large dollar amount (e.g. "shall not be less than
  $X,000,000"), uncapped liability imposed on one party, liability that reverses direction (client liable
  to vendor for all losses).
- CRITICAL: Auto-renewal clause where notice window to cancel is unusually short (under 30 days) or
  buried in the agreement.
- HIGH: Specific fixed dollar liability cap above $5M that significantly exceeds fees in the contract.
- HIGH: IP assignment that transfers ALL work product and pre-existing IP to the other party with no
  carve-outs.
- HIGH: Exclusivity or non-compete lasting more than 2 years, or with overbroad scope covering entire
  industries.
- MEDIUM: One-sided indemnification with no reciprocal protection whatsoever.
- MEDIUM: Service credits as SOLE remedy (no termination right) after repeated SLA failures.
- LOW: Any other clause that is unusual but not immediately dangerous.

=== RULES ===
- Report EVERY genuine finding. Do not stop at 10.
- IMPORTANT: Use the EXACT document name from the === header. Do not guess, abbreviate, or rename it.
- For each finding: name the contract, quote the exact clause, explain deviation from the baseline above,
  rate severity (LOW / MEDIUM / HIGH / CRITICAL).
- Sort from most to least severe. Cite dollar amounts and section references."""


class ScanResponse(BaseModel):
    findings: str
    documents_scanned: int
    chunks_analyzed: int
    time_seconds: float


@app.post("/scan", response_model=ScanResponse)
async def scan_anomalies():
    """Scan ALL contracts for hidden risks — bypasses RAG retrieval entirely."""
    start = time.time()
    from collections import defaultdict

    # Step 1 — get every document name in the corpus
    all_docs_resp = supabase_client.table("contract_chunks") \
        .select("document_name") \
        .eq("chunk_index", 0) \
        .execute()
    all_doc_names = [r["document_name"] for r in (all_docs_resp.data or [])]

    # Step 2 — for each document fetch its financial/risk-related chunks
    # using FTS so we get relevant clauses, not every chunk
    tsquery = " | ".join(SCAN_KEYWORDS)
    by_doc: dict = defaultdict(list)

    for doc_name in all_doc_names:
        try:
            resp = supabase_client.table("contract_chunks") \
                .select("id, document_name, content, chunk_index") \
                .eq("document_name", doc_name) \
                .execute()
            doc_chunks = resp.data or []
        except Exception:
            continue

        # Score each chunk by how many risk keywords it contains
        keywords_lower = [k.lower() for k in SCAN_KEYWORDS]
        scored = []
        for c in doc_chunks:
            text = c.get("content", "").lower()
            kw_count = sum(1 for kw in keywords_lower if kw in text)
            if kw_count > 0:
                scored.append((kw_count, c))

        # Take the top 5 most keyword-dense chunks per document
        MAX_CHUNKS_PER_DOC = 5
        if scored:
            scored.sort(key=lambda x: x[0], reverse=True)
            risk_chunks = [c for _, c in scored[:MAX_CHUNKS_PER_DOC]]
        else:
            # No keyword match — include preamble + first content chunk
            risk_chunks = doc_chunks[:2]

        for c in risk_chunks:
            by_doc[doc_name].append(c.get("content", ""))

    all_chunks = [
        {"document_name": doc, "content": content}
        for doc, contents in by_doc.items()
        for content in contents
    ]

    if not all_chunks:
        return ScanResponse(
            findings="No financial or risk-related clauses found in the corpus.",
            documents_scanned=0,
            chunks_analyzed=0,
            time_seconds=round(time.time() - start, 2),
        )

    docs_scanned = len(by_doc)

    # Build context grouped by document (by_doc already built above)
    context_parts = []
    for doc, contents in sorted(by_doc.items()):
        context_parts.append(f"=== {doc} ===")
        for i, text in enumerate(contents, 1):
            context_parts.append(f"[Clause {i}] {text}")
        context_parts.append("")

    context_block = "\n".join(context_parts)

    # Send to LLM
    completion = openai_client.chat.completions.create(
        model=GENERATION_MODEL,
        messages=[
            {"role": "system", "content": SCAN_SYSTEM_PROMPT},
            {"role": "user", "content": (
                f"Below are {len(all_chunks)} clauses from {docs_scanned} contracts.\n"
                f"Identify all anomalies and rank by severity.\n\n"
                f"{context_block}"
            )},
        ],
        temperature=0.1,
    )

    findings = completion.choices[0].message.content.strip()

    return ScanResponse(
        findings=findings,
        documents_scanned=docs_scanned,
        chunks_analyzed=len(all_chunks),
        time_seconds=round(time.time() - start, 2),
    )


# ---------------------------------------------------------------------------
# POST /scan/topic — cross-corpus topic extraction (not RAG, reads every doc)
# ---------------------------------------------------------------------------
TOPIC_SCAN_SYSTEM_PROMPT = """You are an expert legal assistant performing a full-corpus analysis.

You will receive clauses extracted from multiple contracts, each section prefixed with === document_name ===.
Your job is to find and summarise EVERY instance of the requested topic across ALL contracts.

=== RULES ===
- Cover EVERY contract that contains relevant clauses. Do not stop early.
- Use the EXACT document name from the === header. Do not guess, abbreviate, or rename.
- For each contract: name the document, quote the key clause(s), and summarise in plain language.
- Group your answer by contract document for easy comparison.
- IMPORTANT: ONLY list contracts that have relevant clauses. Do NOT include contracts
  where you found nothing — simply omit them. No "no relevant clauses found" entries.
- Include specific numbers: dollar amounts, percentages, day counts, dates.
- At the end, provide a brief comparative summary highlighting notable differences or outliers.
- Apply legal expertise: the same concept may use different wording across contracts.
  For example, "termination" may appear as "dissolution", "default and remedies",
  "cancellation", "exit rights", or "wind-down". Cover ALL equivalent terms."""


class TopicScanRequest(BaseModel):
    topic: str = Field(..., min_length=1, max_length=500)


class TopicScanResponse(BaseModel):
    answer: str
    documents_scanned: int
    documents_with_matches: int
    chunks_analyzed: int
    time_seconds: float


@app.post("/scan/topic", response_model=TopicScanResponse)
async def scan_topic(req: TopicScanRequest):
    """Scan ALL contracts for a user-specified topic — exhaustive cross-corpus search."""
    start = time.time()
    from collections import defaultdict

    # Step 1 — Use GPT to generate relevant keywords for the topic
    try:
        kw_completion = openai_client.chat.completions.create(
            model=GENERATION_MODEL,
            messages=[
                {
                    "role": "system",
                    "content": (
                        "You are a legal keyword expert. Given a topic, output 15-25 keywords "
                        "and short phrases (lowercase, comma-separated) that would appear in "
                        "contract clauses about that topic. You MUST include:\n"
                        "- Direct terms and common variations\n"
                        "- Legal synonyms and equivalent concepts (e.g. termination → dissolution, "
                        "default, cancellation, wind-down, exit, expiration)\n"
                        "- Related procedural terms (e.g. cure, notice, remedy)\n"
                        "- Partial stems that match variations (e.g. 'terminat' matches termination/terminate/terminated)\n"
                        "Output ONLY the comma-separated list, nothing else."
                    ),
                },
                {"role": "user", "content": req.topic},
            ],
            temperature=0.0,
            max_tokens=200,
        )
        kw_text = kw_completion.choices[0].message.content.strip()
        topic_keywords = [kw.strip().lower() for kw in kw_text.split(",") if kw.strip()]
    except Exception:
        # Fallback: split the topic itself into keywords
        topic_keywords = [w.lower() for w in req.topic.split() if len(w) > 2]

    # Step 2 — Get every document name
    all_docs_resp = supabase_client.table("contract_chunks") \
        .select("document_name") \
        .eq("chunk_index", 0) \
        .execute()
    all_doc_names = [r["document_name"] for r in (all_docs_resp.data or [])]

    # Step 3 — For each document, score chunks by topic keyword density
    MAX_CHUNKS_PER_DOC = 6
    by_doc: dict = defaultdict(list)

    for doc_name in all_doc_names:
        try:
            resp = supabase_client.table("contract_chunks") \
                .select("id, document_name, content, chunk_index") \
                .eq("document_name", doc_name) \
                .execute()
            doc_chunks = resp.data or []
        except Exception:
            continue

        scored = []
        for c in doc_chunks:
            text = c.get("content", "").lower()
            kw_count = sum(1 for kw in topic_keywords if kw in text)
            if kw_count > 0:
                scored.append((kw_count, c))

        if scored:
            scored.sort(key=lambda x: x[0], reverse=True)
            for _, c in scored[:MAX_CHUNKS_PER_DOC]:
                by_doc[doc_name].append(c.get("content", ""))

    total_chunks = sum(len(v) for v in by_doc.values())

    if not by_doc:
        return TopicScanResponse(
            answer=f"No clauses related to '{req.topic}' found across the corpus.",
            documents_scanned=len(all_doc_names),
            documents_with_matches=0,
            chunks_analyzed=0,
            time_seconds=round(time.time() - start, 2),
        )

    # Step 4 — Build context grouped by document
    context_parts = []
    for doc, contents in sorted(by_doc.items()):
        context_parts.append(f"=== {doc} ===")
        for i, text in enumerate(contents, 1):
            context_parts.append(f"[Clause {i}] {text}")
        context_parts.append("")

    context_block = "\n".join(context_parts)

    # Step 5 — Send to LLM
    completion = openai_client.chat.completions.create(
        model=GENERATION_MODEL,
        messages=[
            {"role": "system", "content": TOPIC_SCAN_SYSTEM_PROMPT},
            {"role": "user", "content": (
                f"Topic: {req.topic}\n\n"
                f"Below are {total_chunks} clauses from {len(by_doc)} contracts "
                f"that may contain information about this topic.\n"
                f"Analyse every contract and provide a comprehensive summary.\n\n"
                f"{context_block}"
            )},
        ],
        temperature=0.1,
    )

    answer = completion.choices[0].message.content.strip()

    return TopicScanResponse(
        answer=answer,
        documents_scanned=len(all_doc_names),
        documents_with_matches=len(by_doc),
        chunks_analyzed=total_chunks,
        time_seconds=round(time.time() - start, 2),
    )


# ---------------------------------------------------------------------------
# Broad query detection — auto-route to topic scan when user asks about
# a subject across ALL contracts from the chat box
# ---------------------------------------------------------------------------
BROAD_SIGNALS = [
    "all contracts", "every contract", "across all", "each contract",
    "all agreements", "every agreement", "across every", "all documents",
    "every document", "entire corpus", "all the contracts", "whole portfolio",
    "compare all", "list all", "summarize all", "summarise all",
]


def is_broad_query(question: str) -> bool:
    """Return True if the question asks about a topic across all contracts."""
    q = question.lower()
    return any(signal in q for signal in BROAD_SIGNALS)


def extract_topic_from_query(question: str) -> str:
    """Strip the 'across all contracts' part to get the core topic."""
    q = question
    # Remove broad signals to isolate the topic
    q_lower = q.lower()
    for signal in sorted(BROAD_SIGNALS, key=len, reverse=True):
        idx = q_lower.find(signal)
        if idx != -1:
            q = q[:idx] + q[idx + len(signal):]
            q_lower = q.lower()
    # Clean up leftover words
    for filler in ["what are the", "what is the", "list the", "show me",
                    "compare the", "summarize the", "summarise the",
                    "find the", "get the", "in", "from", "?"]:
        q_lower_stripped = q.lower().strip()
        if q_lower_stripped.startswith(filler):
            q = q[len(filler):].strip()
        if q_lower_stripped.endswith(filler):
            q = q[:-len(filler)].strip()
    return q.strip() or question


# ---------------------------------------------------------------------------
# POST /query — full answer as JSON (auto-routes broad queries to topic scan)
# ---------------------------------------------------------------------------
@app.post("/query", response_model=QueryResponse)
async def query(req: QueryRequest):
    start = time.time()

    # Auto-route: if the user asks about a topic across all contracts,
    # use the exhaustive topic scan instead of RAG
    if is_broad_query(req.question):
        topic = extract_topic_from_query(req.question)
        topic_req = TopicScanRequest(topic=topic)
        result = await scan_topic(topic_req)
        return QueryResponse(
            answer=result.answer,
            sources=[],
            expanded_query=f"[TOPIC SCAN] {topic}",
            chunks_retrieved=result.chunks_analyzed,
            time_seconds=result.time_seconds,
        )

    expanded, top_chunks, all_chunks = run_retrieval_pipeline(req.question)

    if not top_chunks:
        return QueryResponse(
            answer="No relevant contract excerpts found. Try rephrasing your question.",
            sources=[],
            expanded_query=expanded,
            chunks_retrieved=0,
            time_seconds=round(time.time() - start, 2),
        )

    # Generate answer
    context_block = build_context_block(top_chunks)
    user_message = (
        f"Contract excerpts for context:\n\n"
        f"{context_block}\n\n"
        f"{'=' * 60}\n\n"
        f"Question: {req.question}"
    )

    completion = openai_client.chat.completions.create(
        model=GENERATION_MODEL,
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": user_message},
        ],
        temperature=0.2,
    )

    answer = completion.choices[0].message.content.strip()
    sources = list({c.get("document_name", "") for c in top_chunks})

    return QueryResponse(
        answer=answer,
        sources=sorted(sources),
        expanded_query=expanded,
        chunks_retrieved=len(all_chunks),
        time_seconds=round(time.time() - start, 2),
    )


# ---------------------------------------------------------------------------
# POST /query/stream — SSE streaming answer
# ---------------------------------------------------------------------------
@app.post("/query/stream")
async def query_stream(req: QueryRequest):
    import json as _json

    # Auto-route broad queries to topic scan (streamed)
    if is_broad_query(req.question):
        topic = extract_topic_from_query(req.question)
        topic_req = TopicScanRequest(topic=topic)
        result = await scan_topic(topic_req)

        def broad_stream():
            meta = _json.dumps({
                "sources": [],
                "expanded_query": f"[TOPIC SCAN] {topic}",
                "chunks": result.chunks_analyzed,
                "scanMeta": f"{result.documents_with_matches}/{result.documents_scanned} documents matched",
            })
            yield f"data: {meta}\n\n"
            # Send full answer in chunks for smooth rendering
            text = result.answer
            chunk_size = 80
            for i in range(0, len(text), chunk_size):
                fragment = text[i:i + chunk_size].replace("\n", "\\n")
                yield f"data: {fragment}\n\n"
            yield "data: [DONE]\n\n"

        return StreamingResponse(broad_stream(), media_type="text/event-stream")

    expanded, top_chunks, all_chunks = run_retrieval_pipeline(req.question)

    if not top_chunks:
        async def empty_stream():
            yield "data: No relevant contract excerpts found.\n\n"
            yield "data: [DONE]\n\n"
        return StreamingResponse(empty_stream(), media_type="text/event-stream")

    context_block = build_context_block(top_chunks)
    user_message = (
        f"Contract excerpts for context:\n\n"
        f"{context_block}\n\n"
        f"{'=' * 60}\n\n"
        f"Question: {req.question}"
    )

    # OpenAI streaming
    stream = openai_client.chat.completions.create(
        model=GENERATION_MODEL,
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": user_message},
        ],
        temperature=0.2,
        stream=True,
    )

    def event_generator():
        # Send sources metadata first
        sources = sorted({c.get("document_name", "") for c in top_chunks})
        meta = _json.dumps({"sources": sources, "expanded_query": expanded})
        yield f"data: {meta}\n\n"

        for chunk in stream:
            delta = chunk.choices[0].delta
            if delta.content:
                # Escape newlines for SSE
                text = delta.content.replace("\n", "\\n")
                yield f"data: {text}\n\n"

        yield "data: [DONE]\n\n"

    return StreamingResponse(event_generator(), media_type="text/event-stream")