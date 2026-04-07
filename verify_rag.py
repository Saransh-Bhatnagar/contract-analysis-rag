"""
verify_rag.py
-------------
Standalone verification script for the Legal Contracts RAG pipeline.
Calls the three retrieval functions directly (no CLI loop) and prints
PASS / FAIL for each assertion.

Run with:
    python verify_rag.py
"""

import sys
from query_rag import (
    keyword_search,
    fetch_preambles,
    search_contracts,
    embed_query,
    merge_chunks,
)

PASS = "✅  PASS"
FAIL = "❌  FAIL"


def check(label: str, condition: bool, detail: str = "") -> bool:
    status = PASS if condition else FAIL
    print(f"  {status}  {label}")
    if detail:
        print(f"         {detail}")
    return condition


def section(title: str) -> None:
    print(f"\n{'─' * 55}")
    print(f"  {title}")
    print(f"{'─' * 55}")


def run_verification() -> None:
    total, passed = 0, 0

    # ------------------------------------------------------------------
    # 1. Preamble pinning — at least one chunk per contract document
    # ------------------------------------------------------------------
    section("1. Preamble Pinning  (fetch_preambles)")
    # To test fetch_preambles we need to provide a known document name
    # We will test using "MSA_Complex.pdf" and "NDA_Mutual.pdf" 
    test_docs = ["MSA_Complex.pdf", "NDA_Mutual.pdf"]
    preamble = fetch_preambles(test_docs)
    docs_in_preamble = {r.get("document_name") for r in preamble}
    print(f"     Documents returned: {sorted(docs_in_preamble)}\n")

    t = check(
        "One preamble chunk returned per requested contract",
        len(preamble) == len(test_docs),
        detail=f"Got {len(preamble)} chunk(s) expected {len(test_docs)}"
    )
    total += 1; passed += int(t)

    has_index = all(r.get("chunk_index") == 0 for r in preamble)
    t = check(
        "Returned chunks are all chunk_index = 0",
        has_index,
        detail=f"chunk_index values: {[r.get('chunk_index') for r in preamble]}"
    )
    total += 1; passed += int(t)


    # ------------------------------------------------------------------
    # 2. FTS keyword search — matches across phrasing variants
    # ------------------------------------------------------------------
    section("2. FTS Keyword Search  (keyword_search)")

    fts_dates = keyword_search("effective date entered into parties")
    t = check(
        "FTS: 'effective date entered into' returns results",
        len(fts_dates) >= 1,
        detail=f"{len(fts_dates)} chunk(s) returned"
    )
    total += 1; passed += int(t)

    fts_termination = keyword_search("termination notice period")
    t = check(
        "FTS: 'termination notice period' returns results",
        len(fts_termination) >= 1,
        detail=f"{len(fts_termination)} chunk(s) returned"
    )
    total += 1; passed += int(t)

    fts_stemming = keyword_search("terminate")     # should match "termination" via stemming
    has_termination = any("terminat" in r.get("content", "").lower() for r in fts_stemming)
    t = check(
        "FTS stemming: 'terminate' matches 'termination' in content",
        has_termination,
        detail=f"{len(fts_stemming)} chunk(s) returned"
    )
    total += 1; passed += int(t)

    # ------------------------------------------------------------------
    # 3. Vector search — sanity check similarity threshold
    # ------------------------------------------------------------------
    section("3. Vector Search  (search_contracts)")

    query_vec = embed_query("liability limitation cap damages")
    vec = search_contracts(query_vec) if query_vec is not None else []
    t = check(
        "Vector: 'liability limitation' returns results",
        len(vec) >= 1,
        detail=(f"{len(vec)} chunk(s) returned; top similarity: "
                f"{vec[0].get('similarity', 0):.3f}") if vec else "no results (embedding may have failed)"
    )
    total += 1; passed += int(t)

    # ------------------------------------------------------------------
    # 4. Merge & deduplication
    # ------------------------------------------------------------------
    section("4. Merge & Deduplication  (merge_chunks)")

    # Craft artificial duplicates to test dedup logic
    dummy_a = [{"id": "aaa", "document_name": "A.pdf", "content": "foo", "similarity": 0.9}]
    dummy_b = [{"id": "aaa", "document_name": "A.pdf", "content": "foo", "similarity": 0.5},
               {"id": "bbb", "document_name": "B.pdf", "content": "bar", "similarity": 0.4}]

    merged = merge_chunks(dummy_a, dummy_b)
    t = check(
        "Duplicate id 'aaa' appears only once in merged result",
        len([c for c in merged if c["id"] == "aaa"]) == 1,
    )
    total += 1; passed += int(t)

    t = check(
        "Merged result contains both unique ids",
        {c["id"] for c in merged} == {"aaa", "bbb"},
    )
    total += 1; passed += int(t)

    t = check(
        "First-source similarity preserved for duplicate (0.9, not 0.5)",
        next(c["similarity"] for c in merged if c["id"] == "aaa") == 0.9,
    )
    total += 1; passed += int(t)

    # ------------------------------------------------------------------
    # Summary
    # ------------------------------------------------------------------
    print(f"\n{'═' * 55}")
    print(f"  RESULT: {passed} / {total} checks passed")
    print(f"{'═' * 55}\n")

    if passed < total:
        sys.exit(1)


if __name__ == "__main__":
    run_verification()
