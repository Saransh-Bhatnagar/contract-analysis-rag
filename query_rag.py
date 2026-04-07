"""
query_rag.py
------------
Retrieval-Augmented Generation (RAG) query engine for the Legal Contracts pipeline.

Flow:
  1. Expand the user's question into a richer semantic form with GPT (query expansion).
  2. Embed the expanded query with text-embedding-3-small.
  3. Run THREE retrievals against Supabase and merge results:
       a. Vector search   — `match_contracts` RPC (cosine similarity)
       b. FTS search      — `keyword_search_contracts` RPC (Postgres full-text, stemmed)
       c. Preamble anchor — `contract_chunks` table lookup using Supabase Python client 
                           (fetches chunk_index = 0 for unique documents matched in 3a and 3b)
  4. Merge & deduplicate results from both original and expanded queries.
  5. Rerank top results using `cross-encoder/ms-marco-MiniLM-L-6-v2`.
  6. Pass top 5-7 chunks to GPT-4o-mini with the original question.
  7. Print the answer (with source citations) to the terminal.

Required Supabase SQL (run ONCE in the SQL editor before first use):
  create or replace function keyword_search_contracts(
      search_query text,
      match_count   int default 10
  )
  returns table (
      id            text,
      document_name text,
      content       text,
      similarity    float
  )
  language sql stable
  as $$
      select
          id,
          document_name,
          content,
          ts_rank(
              to_tsvector('english', content),
              plainto_tsquery('english', search_query)
          )::float as similarity
      from contract_chunks
      where to_tsvector('english', content) @@ plainto_tsquery('english', search_query)
      order by similarity desc
      limit match_count;
  $$;

  -- Preamble retrieval now replaced by standard Supabase client lookup
  -- `DROP FUNCTION IF EXISTS get_preamble_chunks();`


Supabase RPC — match_contracts (run once in the SQL editor if not present):
  create or replace function match_contracts(
      query_embedding   vector(1536),
      match_threshold   float,
      match_count       int
  )
  returns table (
      id            text,
      document_name text,
      content       text,
      similarity    float
  )
  language sql stable
  as $$
      select
          id,
          document_name,
          content,
          1 - (embedding <=> query_embedding) as similarity
      from contract_chunks
      where 1 - (embedding <=> query_embedding) > match_threshold
      order by similarity desc
      limit match_count;
  $$;
"""

import os
import sys
import logging

from dotenv import load_dotenv
from openai import OpenAI
from sentence_transformers import CrossEncoder
from supabase import create_client, Client

# Suppress noisy BERT / HuggingFace warnings from cross-encoder load
logging.getLogger("sentence_transformers").setLevel(logging.ERROR)
logging.getLogger("huggingface_hub").setLevel(logging.ERROR)
os.environ["HF_HUB_DISABLE_PROGRESS_BARS"] = "1"

# ---------------------------------------------------------------------------
# 1. Environment setup
# ---------------------------------------------------------------------------

load_dotenv()

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
SUPABASE_URL   = os.getenv("SUPABASE_URL") or os.getenv("NEXT_PUBLIC_SUPABASE_URL")
SUPABASE_KEY   = os.getenv("SUPABASE_KEY") or os.getenv("NEXT_PUBLIC_SUPABASE_ANON_KEY")

if not OPENAI_API_KEY:
    sys.exit("[ERROR] OPENAI_API_KEY is not set in the .env file.")
if not SUPABASE_URL:
    sys.exit("[ERROR] SUPABASE_URL (or NEXT_PUBLIC_SUPABASE_URL) is not set in the .env file.")
if not SUPABASE_KEY:
    sys.exit("[ERROR] SUPABASE_KEY (or NEXT_PUBLIC_SUPABASE_ANON_KEY) is not set in the .env file.")

# ---------------------------------------------------------------------------
# 2. Client initialisation
# ---------------------------------------------------------------------------

openai_client: OpenAI = OpenAI(api_key=OPENAI_API_KEY)
supabase_client: Client = create_client(SUPABASE_URL, SUPABASE_KEY)

# ---------------------------------------------------------------------------
# 3. Configuration
# ---------------------------------------------------------------------------

EMBEDDING_MODEL  = "text-embedding-3-small"
GENERATION_MODEL = "gpt-4o-mini"
MATCH_THRESHOLD  = 0.10  # cosine-similarity floor (lowened for max recall)
MATCH_COUNT      = 30     # max similarity chunks to retrieve
FTS_COUNT        = 30     # max FTS chunks to retrieve per call
RERANKER_MODEL   = "cross-encoder/ms-marco-MiniLM-L-12-v2"
TOP_K_CHUNKS     = 16     # Context window — higher = better cross-doc recall, more tokens

_cross_encoder = None


def get_cross_encoder() -> CrossEncoder:
    """Lazy-load the cross-encoder so imports don't pay the startup cost."""
    global _cross_encoder
    if _cross_encoder is None:
        import logging
        print(f"Loading reranker model ({RERANKER_MODEL})...")
        # Temporarily suppress all warnings during model load
        prev_level = logging.root.level
        logging.disable(logging.WARNING)
        _cross_encoder = CrossEncoder(RERANKER_MODEL)
        logging.disable(logging.NOTSET)
        logging.root.setLevel(prev_level)
    return _cross_encoder

SYSTEM_PROMPT = """You are an expert legal assistant specialising in contract analysis.

Core rules:
1. Answer using ONLY the contract excerpts supplied as context.
2. If the answer is genuinely absent from the context, say:
   "I'm sorry, that information is not present in the provided contract excerpts."
   However, if a relevant clause EXISTS in the excerpts but specific values are
   redacted (shown as [***], [REDACTED], or similar placeholders), DO NOT say the
   information is absent. Instead, describe the clause structure and note that
   the specific values have been redacted in the filing.
3. End every response with a "Sources:" bullet list of the `document_name` values
   of every excerpt you drew upon.
4. Never fabricate facts, dates, amounts, or legal interpretations.
5. Be concise but complete; prefer plain language over legal jargon.
   Read ALL excerpts carefully before concluding information is absent —
   the relevant clause may appear in any excerpt, not just the first few.
6. CRITICAL — Document attribution: Every fact, date, amount, or clause you cite
   MUST come from the specific excerpt that contains it. Always check which
   document_name the excerpt belongs to before stating a fact. NEVER attribute
   information from one document to another. If the user asks about a specific
   document, ONLY use excerpts from that document to answer — ignore excerpts
   from other documents even if they appear in the context.
7. Analytical and comparative questions: When the user asks a comparative or
   judgment question (e.g. "which is more important?", "which has the highest
   liability?", "rank these by risk", "which is better?", "compare X and Y"),
   you SHOULD reason from the excerpts and give a substantive answer. Use legal
   expertise to weigh factors like dollar amounts, scope, exclusivity, term length,
   liability exposure, and strategic importance. Always cite the specific clauses
   and documents that support your conclusion. Do NOT refuse with "not present
   in excerpts" — that response is only appropriate for purely factual lookups
   where the fact itself is missing. For analytical questions, the excerpts ARE
   the input to your reasoning, not the answer themselves.

Legal interpretation (apply automatically — no explicit prompting needed):
• You are a legal expert. When a user asks in plain or informal language, map their
  intent to the correct legal concepts using your domain knowledge — do NOT fail just
  because the user's words differ from the contract's words.
• Examples of mappings you should apply without being told:
    "deadlines" / "important dates" / "timelines"
        → notice periods, payment due dates, term start/end dates, renewal windows,
          cure periods, performance milestones, confidentiality durations,
          any clause with a specific number of days / months / years
    "get out of the contract" / "exit" / "cancel"
        → termination clauses, termination for convenience, breach cure rights,
          mutual termination, force majeure exits
    "who are the parties" / "who signed" / "company names"
        → recitals, definitions of Client / Customer / Vendor / Service Provider,
          signatory blocks
    "what am I liable for" / "liability"
        → indemnification, limitation of liability, caps, exclusions
• When asked for a summary across multiple contracts, compare them side-by-side and
  group your answer by contract document.

Anomaly and risk detection (apply automatically):
• When discussing liability, indemnification, payment, or financial terms across
  contracts, ALWAYS flag any amounts, caps, or obligations that are significantly
  different from the norm across the other contracts. Look for:
    - Liability caps that are dramatically higher or lower than others
    - One-sided indemnification where only one party bears risk
    - Uncapped liability clauses or clauses that override general caps
    - Unusual penalty structures or hidden escalation clauses
    - Auto-renewal traps with short opt-out windows
• Even if the user does not ask about outliers, mention anomalies when they appear
  in the excerpts you are analysing — a good legal assistant proactively flags risk."""

# ---------------------------------------------------------------------------
# 4. Core functions
# ---------------------------------------------------------------------------


# Legal keyword hints: when the user's question matches a topic, these phrases
# are appended to the reranker input to boost chunks containing exact legal language.
LEGAL_KEYWORD_HINTS = {
    "governing law": "governed by construed in accordance with laws of the state jurisdiction venue",
    "liability": "limitation of liability aggregate liability shall not exceed in no event cap on liability damages",
    "insurance": "insurance coverage policy insured underwriter commercially reasonable amounts product liability general liability",
    "expiration": "initial term term of this agreement shall expire effective date anniversary renewal term",
    "termination": "termination for convenience terminate without cause at will upon written notice right to terminate",
    "notice period": "days prior written notice non-renewal advance notice calendar days business days",
    "indemnif": "indemnification indemnify hold harmless defend third party claims losses damages",
}


def enrich_reranker_query(query: str) -> str:
    """Append legal keyword hints to the query for better cross-encoder reranking."""
    q_lower = query.lower()
    hints = []
    for trigger, phrase in LEGAL_KEYWORD_HINTS.items():
        if trigger in q_lower:
            hints.append(phrase)
    if hints:
        return query + " " + " ".join(hints)
    return query


def expand_query(original_query: str) -> str:
    """
    Use GPT-4o-mini to rewrite a short / vague user question into a richer
    semantic search query that is more likely to match relevant legal text.

    Falls back to the original query if the LLM call fails.
    """
    try:
        completion = openai_client.chat.completions.create(
            model=GENERATION_MODEL,
            messages=[
                {
                    "role": "system",
                    "content": (
                        "You are a legal search specialist. "
                        "Rewrite the user's question into a detailed, semantically rich "
                        "search query optimised for finding relevant legal contract clauses. "
                        "Expand abbreviations, add synonyms, and include related legal terms. "
                        "Output ONLY the rewritten query — no explanations, no quotes."
                    ),
                },
                {"role": "user", "content": original_query},
            ],
            temperature=0.0,
            max_tokens=120,
        )
        expanded = completion.choices[0].message.content.strip()
        return expanded if expanded else original_query
    except Exception:
        return original_query  # graceful degradation


def detect_target_document(question: str) -> str | None:
    """
    Detect if the user's question targets a specific document by name.
    Returns the matching document_name (e.g. "MSA_Cloud_Services.pdf") or None.
    """
    q_lower = question.lower().replace(".pdf", "").replace("-", "_").replace(" ", "_")
    try:
        resp = supabase_client.table("contract_chunks") \
            .select("document_name") \
            .eq("chunk_index", 0) \
            .execute()
        doc_names = sorted({r["document_name"] for r in (resp.data or [])})
    except Exception:
        return None

    # Exact match first (user typed "MSA_Cloud_Services" or similar)
    for doc in doc_names:
        doc_stem = doc.lower().replace(".pdf", "")
        if doc_stem in q_lower:
            return doc

    # Fuzzy match: convert underscores to spaces and check
    # e.g. "msa cloud services" matches "MSA_Cloud_Services.pdf"
    q_spaced = q_lower.replace("_", " ")
    for doc in doc_names:
        doc_spaced = doc.lower().replace(".pdf", "").replace("_", " ")
        if doc_spaced in q_spaced:
            return doc

    # Reverse substring: find the doc whose name has the longest overlap with the query.
    # Picks the most specific match to avoid false positives when multiple docs share
    # a common suffix like "STRATEGIC ALLIANCE AGREEMENT".
    best_doc = None
    best_len = 0
    for doc in doc_names:
        doc_spaced = doc.lower().replace(".pdf", "").replace("_", " ").replace("-", " ")
        doc_words = doc_spaced.split()
        for start in range(len(doc_words)):
            for end in range(start + 3, len(doc_words) + 1):
                phrase = " ".join(doc_words[start:end])
                if len(phrase) >= 20 and phrase in q_spaced and len(phrase) > best_len:
                    best_len = len(phrase)
                    best_doc = doc

    return best_doc


# Rerank score boost for chunks from the target document.
# This is large enough to guarantee target-doc chunks rank at the top,
# but still lets the reranker order them relative to each other.
DOC_BOOST = 10.0


def fetch_all_doc_chunks(document_name: str) -> list[dict]:
    """
    Fetch every chunk for a specific document, ordered by chunk_index.
    Used when the user targets a specific document so the LLM sees the
    full contract, not just whatever matched the search query.
    """
    try:
        response = supabase_client.table("contract_chunks") \
            .select("id, document_name, content, chunk_index") \
            .eq("document_name", document_name) \
            .order("chunk_index") \
            .execute()
        return response.data or []
    except Exception:
        return []


def keyword_search(query: str) -> list[dict]:
    """
    Full-text search via the `keyword_search_contracts` Supabase RPC.

    Uses Postgres `to_tsvector` / `plainto_tsquery` which handles stemming,
    stop-words, and ranking — far more robust than ilike pattern matching.
    Returns up to FTS_COUNT deduplicated chunks.
    """
    try:
        response = supabase_client.rpc(
            "keyword_search_contracts",
            {
                "search_query": query,
                "match_count":  FTS_COUNT,
            },
        ).execute()
        return response.data or []
    except Exception:
        return []  # degrade gracefully; vector results will still be used


def fetch_preambles(document_names: list[str]) -> list[dict]:
    """
    Fetch chunk_index = 0 for the specifically matched documents.
    This guarantees preamble context without pulling the entire DB.
    """
    if not document_names:
        return []
    
    try:
        response = supabase_client.table("contract_chunks") \
            .select("id, document_name, content") \
            .in_("document_name", document_names) \
            .eq("chunk_index", 0) \
            .execute()
        
        # Add a placeholder similarity score for merging logic
        data = response.data or []
        for chunk in data:
            chunk['similarity'] = 1.0  # Perfect score to ensure it survives deduplication if ranked
            chunk['chunk_index'] = 0   # Set explicitly so reranker can boost it
        return data
    except Exception:
        return []


def fetch_neighbor_chunks(chunks: list[dict]) -> list[dict]:
    """
    For each chunk, fetch the immediately adjacent chunks (index +/- 1) from
    the same document.  This captures context that spans chunk boundaries —
    critical for clauses like outlier liability terms that sit right after a
    standard limitation-of-liability section.
    """
    if not chunks:
        return []

    neighbor_keys = set()
    for c in chunks:
        doc = c.get("document_name", "")
        idx = c.get("chunk_index")
        if doc and idx is not None:
            if idx > 0:
                neighbor_keys.add((doc, idx - 1))
            neighbor_keys.add((doc, idx + 1))

    if not neighbor_keys:
        return []

    from collections import defaultdict
    doc_indices = defaultdict(list)
    for doc, idx in neighbor_keys:
        doc_indices[doc].append(idx)

    neighbors = []
    try:
        for doc, indices in doc_indices.items():
            response = supabase_client.table("contract_chunks") \
                .select("id, document_name, content, chunk_index") \
                .eq("document_name", doc) \
                .in_("chunk_index", indices) \
                .execute()
            for item in (response.data or []):
                item["similarity"] = 0.0
                neighbors.append(item)
    except Exception:
        pass

    return neighbors


def merge_chunks(*sources: list[dict]) -> list[dict]:
    """
    Merge any number of chunk lists, deduplicating by chunk id.
    Sources listed first take precedence (their similarity scores are kept).
    """
    merged: list[dict] = []
    seen_ids: set[str] = set()

    for source in sources:
        for chunk in source:
            cid = chunk.get("id", "")
            if cid not in seen_ids:
                seen_ids.add(cid)
                merged.append(chunk)

    return merged


def embed_query(query: str) -> list[float] | None:
    """Convert a text query into a 1536-dimension embedding vector."""
    try:
        response = openai_client.embeddings.create(
            model=EMBEDDING_MODEL,
            input=query,
        )
        return response.data[0].embedding
    except Exception:
        return None


def search_contracts(query_embedding: list[float]) -> list[dict]:
    """
    Call the Supabase `match_contracts` RPC and return the matching chunks.

    Each returned dict has at minimum:
        - document_name (str)
        - content       (str)
        - similarity    (float)
    """
    try:
        response = supabase_client.rpc(
            "match_contracts",
            {
                "query_embedding": query_embedding,
                "match_threshold": MATCH_THRESHOLD,
                "match_count":     MATCH_COUNT,
            },
        ).execute()
        return response.data or []
    except Exception:
        return []


def build_context_block(chunks: list[dict]) -> str:
    """
    Format retrieved chunks into a numbered context block for the LLM prompt.
    """
    if not chunks:
        return "(No relevant contract excerpts were found.)"

    parts = []
    for i, chunk in enumerate(chunks, start=1):
        doc   = chunk.get("document_name", "Unknown document")
        text  = chunk.get("content", "").strip()
        # Prefer rerank_score (set after cross-encoder) over raw cosine similarity
        score = chunk.get("rerank_score", chunk.get("similarity", 0.0))
        parts.append(
            f"[Excerpt {i}] Source: {doc} (relevance: {score:.3f})\n"
            f"{'-' * 60}\n"
            f"{text}\n"
        )
    return "\n".join(parts)


CHUNK_FILTER_PROMPT = """You are a legal document analyst. You will be given a question and a list of numbered contract excerpts.

Your task: identify which excerpts contain information relevant to answering the question.

Rules:
- Return ONLY a comma-separated list of excerpt numbers (e.g. "1, 4, 7, 12")
- Include any excerpt that might contain relevant clauses, even partially
- When in doubt, INCLUDE the excerpt — it is better to include too many than too few
- If none are relevant, return "NONE"
- Do NOT explain your reasoning — just output the numbers"""


def _filter_chunks_pass1(query: str, chunks: list[dict]) -> list[dict]:
    """
    Pass 1: Ask GPT to identify which chunks are relevant to the question.
    Returns only the relevant chunks for pass 2.
    Falls back to all chunks if the filter call fails.
    """
    if len(chunks) <= 16:
        # Small enough context — no need for two passes
        return chunks

    context_block = build_context_block(chunks)
    user_message = (
        f"Contract excerpts:\n\n"
        f"{context_block}\n\n"
        f"{'=' * 60}\n\n"
        f"Question: {query}\n\n"
        f"Which excerpt numbers contain information relevant to this question?"
    )

    try:
        completion = openai_client.chat.completions.create(
            model=GENERATION_MODEL,
            messages=[
                {"role": "system", "content": CHUNK_FILTER_PROMPT},
                {"role": "user",   "content": user_message},
            ],
            temperature=0.0,
            max_tokens=100,
        )
        response = completion.choices[0].message.content.strip()

        if response.upper() == "NONE":
            return chunks  # fall back to all

        # Parse comma-separated numbers
        import re
        numbers = [int(n) for n in re.findall(r'\d+', response)]
        if not numbers:
            return chunks

        # Convert 1-based excerpt numbers to 0-based indices
        filtered = [chunks[n - 1] for n in numbers if 1 <= n <= len(chunks)]
        return filtered if filtered else chunks

    except Exception:
        return chunks  # graceful fallback


def generate_answer(query: str, chunks: list[dict]) -> str:
    """
    Send the retrieved context + user query to GPT-4o-mini and return the answer.
    """
    context_block = build_context_block(chunks)

    user_message = (
        f"Contract excerpts for context:\n\n"
        f"{context_block}\n\n"
        f"{'=' * 60}\n\n"
        f"Question: {query}"
    )

    completion = openai_client.chat.completions.create(
        model=GENERATION_MODEL,
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user",   "content": user_message},
        ],
        temperature=0.2,   # low temperature → more factual, less hallucination
    )

    return completion.choices[0].message.content.strip()


# ---------------------------------------------------------------------------
# 5. CLI entry point
# ---------------------------------------------------------------------------


def main() -> None:
    print("=" * 60)
    print("  ⚖️  Legal Contracts RAG — Query Engine")
    print(f"  Embedding : {EMBEDDING_MODEL}")
    print(f"  Generator : {GENERATION_MODEL}")
    print(f"  Threshold : {MATCH_THRESHOLD}  |  Top-K : {TOP_K_CHUNKS}")
    print("=" * 60)
    print("  Type your question below. Type 'exit' to quit.\n")

    while True:
        try:
            query = input("🔍  Your question: ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\n\n👋  Goodbye!")
            break

        if not query:
            print("    ⚠️  Please enter a non-empty question.\n")
            continue

        if query.lower() == "exit":
            print("\n👋  Goodbye!")
            break

        print("\n⏳  Expanding query …")

        try:
            # Step 1 — expand the query for richer semantic matching
            expanded = expand_query(query)
            if expanded != query:
                print(f"    ✏️  Expanded: \"{expanded}\"")

            print("⏳  Retrieving (vector + FTS + preamble) …\n")

            # Step 2 & 3 — MULTI-QUERY RETRIEVAL (Original + Expanded)
            # -------------------------------------------------------------------
            
            # 3a — Vector Search
            query_vector = embed_query(query)
            if query_vector is None:
                print("❌  Failed to embed query. Check your OpenAI API key.\n")
                continue

            vector_chunks_orig = search_contracts(query_vector)

            vector_chunks_exp = []
            if expanded != query:
                expanded_vector = embed_query(expanded)
                if expanded_vector is not None:
                    vector_chunks_exp = search_contracts(expanded_vector)

            # 3b — Postgres FTS Search
            fts_chunks_orig = keyword_search(query)
            fts_chunks_exp = []
            if expanded != query:
                fts_chunks_exp = keyword_search(expanded)

            # 3c — Merge initial results to find unique document names
            initial_chunks = merge_chunks(vector_chunks_orig, vector_chunks_exp, fts_chunks_orig, fts_chunks_exp)
            unique_docs = list(set(chunk.get("document_name") for chunk in initial_chunks if chunk.get("document_name")))

            # 3d — Fetch preambles strictly for matched documents
            preamble_chunks = fetch_preambles(unique_docs)

            # Step 4 — merge all & deduplicate
            # -------------------------------------------------------------------
            all_chunks = merge_chunks(initial_chunks, preamble_chunks)

            # Propagate chunk_index from preamble results into merged chunks.
            # Without this, preambles found by vector/FTS lose their chunk_index
            # during dedup (initial_chunks win but lack chunk_index), so the
            # reranker boost silently fails to apply.
            preamble_ids = {c["id"] for c in preamble_chunks}
            for chunk in all_chunks:
                if chunk.get("id") in preamble_ids:
                    chunk["chunk_index"] = 0

            if not all_chunks:
                print(
                    "ℹ️  No contract excerpts found. Try rephrasing your question.\n"
                )
                continue

            print(f"📋  Found {len(all_chunks)} unique excerpt(s) before reranking.")
            print("⏳  Reranking context …\n")

            # Step 4b — Detect if user targets a specific document
            target_doc = detect_target_document(query)
            if target_doc:
                print(f"    📄  Document focus: {target_doc}")
                # Inject ALL chunks from the target document so the LLM
                # sees the full contract, not just search-matched fragments
                doc_chunks = fetch_all_doc_chunks(target_doc)
                if doc_chunks:
                    all_chunks = merge_chunks(all_chunks, doc_chunks)
                    print(f"    📄  Loaded {len(doc_chunks)} chunks from {target_doc}")

            # Step 5 — Cross-Encoder Reranking
            # -------------------------------------------------------------------
            # Score each chunk against the ORIGINAL user query for highest precision
            pairs = [[query, chunk.get('content', '')] for chunk in all_chunks]
            scores = get_cross_encoder().predict(pairs)

            # Attach scores and sort; boost target-doc chunks to the top
            for chunk, score in zip(all_chunks, scores):
                boost = DOC_BOOST if target_doc and chunk.get("document_name") == target_doc else 0.0
                chunk['rerank_score'] = float(score) + boost

            all_chunks.sort(key=lambda x: x['rerank_score'], reverse=True)

            # When a target doc is detected, use a higher K to fit more of that document
            effective_top_k = max(TOP_K_CHUNKS, 48) if target_doc else TOP_K_CHUNKS

            # Filter out non-target-doc chunks to prevent cross-document contamination
            if target_doc:
                all_chunks = [c for c in all_chunks if c.get("document_name") == target_doc]

            # Select top chunks with a preamble cap: include up to MAX_PREAMBLES
            # preamble chunks (chunk_index=0) so they don't crowd out actual content.
            MAX_PREAMBLES = 2
            top_chunks = []
            preamble_count = 0
            for chunk in all_chunks:
                if len(top_chunks) >= effective_top_k:
                    break
                is_preamble = chunk.get('chunk_index') == 0
                if is_preamble:
                    if preamble_count >= MAX_PREAMBLES:
                        continue
                    preamble_count += 1
                top_chunks.append(chunk)

            # Step 5b — Post-reranker neighbor expansion
            # ---------------------------------------------------------------
            # Fetch chunks adjacent (±1 index) to the top-ranked chunks.
            # These go straight to the LLM without reranking, so camouflaged
            # clauses next to high-scoring chunks (e.g. outlier liability
            # buried after a standard liability cap) get surfaced.
            neighbor_expansion = fetch_neighbor_chunks(top_chunks)
            if neighbor_expansion:
                top_chunks = merge_chunks(top_chunks, neighbor_expansion)
                print(
                    f"🔗  Expanded with {len(neighbor_expansion)} neighbor chunk(s) "
                    f"→ {len(top_chunks)} total chunks for generation.\n"
                )
            else:
                print(
                    f"🎯  Selected top {len(top_chunks)} chunks for generation.\n"
                )

            # Step 6 — generate
            answer = generate_answer(query, top_chunks)

        except Exception as exc:
            print(f"❌  An error occurred: {exc}\n")
            continue

        print("-" * 60)
        print(answer)
        print("-" * 60)
        print()


if __name__ == "__main__":
    main()
